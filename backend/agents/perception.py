# backend/agents/perception.py
# 感知层 (Perception Layer)：系统的"眼睛和耳朵"。
# 负责：
#   1. 情感分析 — 调用本地 BERT 模型做二分类（复用 EcomSentiment）
#   2. 情感细分 — 用 LLM 将 positive/negative 二次映射为细粒度标签
#   3. 意图识别 + NER — 用 LLM 结构化输出识别用户意图和关键实体
#
# 输出：PerceptionResult（统一的感知层输出对象）

import json
import os
import re
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import SentimentAnalysisError, IntentClassificationError
from backend.core.retry import with_retry_async
from backend.core.llm_factory import get_structured_llm, get_llm
from backend.models.schemas import (
    Sentiment, SentimentLabel, IntentCategory,
    PerceptionResult,
)
from backend.rag.prompts import EcomRAGPrompts

logger = get_logger(__name__)

# ── BERT依赖：try/except 包裹，pyarrow崩溃时降级为纯LLM模式 ──
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    _BERT_AVAILABLE = True
except Exception as e:
    torch = None  # type: ignore
    BertTokenizer = None  # type: ignore
    BertForSequenceClassification = None  # type: ignore
    _BERT_AVAILABLE = False
    logger.warning("perception.bert_unavailable", error=str(e)[:100], hint="将降级为纯LLM感知")

# ── 🆕 v4.0 意图映射：BERT 10分类 → 4分发Agent ──────────────

_LEGACY_10_TO_4: dict[str, str] = {
    "product_inquiry": "knowledge_qa",
    "price_inquiry": "knowledge_qa",
    "recommend_request": "knowledge_qa",
    "chitchat": "knowledge_qa",
    "other": "knowledge_qa",
    "order_tracking": "business",
    "modify_order": "business",
    "after_sales": "business",
    "knowledge_mgmt": "knowledge_mgmt",
    "complaint": "escalate",
}


def _map_legacy_intent(legacy_value: str) -> IntentCategory:
    """将 BERT 10分类标签映射到 v4.0 4分类。"""
    mapped = _LEGACY_10_TO_4.get(legacy_value, "knowledge_qa")
    return IntentCategory(mapped)


class PerceptionAgent:
    """感知层智能体：情感分析 + 意图识别 + NER 实体抽取。

    v1 策略：
    - 情感极性：优先加载本地 BERT 模型（复用 EcomSentiment 训练好的模型）
    - 情感细分：调用 LLM 将极性映射为 7 种细粒度标签
    - 意图+NER：调用 LLM structured output 一次性完成
    """

    def __init__(self):
        self._sentiment_model = None
        self._sentiment_tokenizer = None
        self._device = "cpu"  # 强制CPU（无GPU环境）
        self._sentiment_label_map = {"0": "negative", "1": "positive"}

        # 意图模型
        self._intent_model = None
        self._intent_tokenizer = None
        self._intent_label_map: dict[str, str] = {}

        # 加载本地模型（BERT可用时加载，否则走LLM降级）
        if _BERT_AVAILABLE:
            self._load_sentiment_model()
            self._load_intent_model()
        else:
            logger.info("perception.bert_skipped", hint="纯LLM模式，跳过BERT加载")

    # ── 情感模型加载 ─────────────────────────────────────────────

    def _load_sentiment_model(self):
        """加载情感分类模型（按优先级依次尝试）。"""
        # 优先级1：自己训练的模型 (backend/training/saved_models/sentiment_classifier)
        trained_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "training", "saved_models", "sentiment_classifier"
        )
        if self._try_load_from_dir(trained_path, is_sentiment=True):
            return

        # 优先级2：EcomSentiment 原来的模型
        settings = get_settings()
        if os.path.exists(settings.sentiment_bert_path):
            try:
                self._sentiment_tokenizer = BertTokenizer.from_pretrained(settings.sentiment_bert_path)
                self._sentiment_model = BertForSequenceClassification.from_pretrained(settings.sentiment_bert_path)
                self._sentiment_model.to(self._device)
                self._sentiment_model.eval()
                if os.path.exists(settings.sentiment_label_map):
                    with open(settings.sentiment_label_map, "r", encoding="utf-8") as f:
                        self._sentiment_label_map = json.load(f)
                logger.info("perception.sentiment_loaded_from_ecom", path=settings.sentiment_bert_path)
                return
            except Exception:
                pass

        # 优先级3：LLM 降级
        logger.info("perception.no_local_sentiment_model", fallback="llm_zero_shot")

    # ── 意图模型加载 ─────────────────────────────────────────────

    def _load_intent_model(self):
        """加载意图分类模型。"""
        trained_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "training", "saved_models", "intent_classifier"
        )
        if self._try_load_from_dir(trained_path, is_sentiment=False):
            return

        logger.info("perception.no_local_intent_model", fallback="llm_structured_output")

    def _try_load_from_dir(self, dir_path: str, is_sentiment: bool) -> bool:
        """尝试从指定目录加载 BERT 模型。成功返回 True。"""
        if not os.path.isdir(dir_path):
            return False
        try:
            tokenizer = BertTokenizer.from_pretrained(dir_path)
            model = BertForSequenceClassification.from_pretrained(dir_path)
            model.to(self._device)
            model.eval()

            label_file = os.path.join(dir_path, "label_map.json")
            label_map = {}
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    label_map = json.load(f)

            if is_sentiment:
                self._sentiment_tokenizer = tokenizer
                self._sentiment_model = model
                if label_map:
                    self._sentiment_label_map = label_map
                logger.info("perception.sentiment_loaded_trained", path=dir_path, labels=label_map)
            else:
                self._intent_tokenizer = tokenizer
                self._intent_model = model
                self._intent_label_map = label_map
                logger.info("perception.intent_loaded_trained", path=dir_path, labels=label_map)

            return True
        except Exception as e:
            logger.warning("perception.load_failed", path=dir_path, error=str(e)[:80])
            return False

    # ── 情感分析 ─────────────────────────────────────────────────

    async def analyze_sentiment(self, query: str) -> tuple[Sentiment, float]:
        """分析用户消息的情感极性。

        :return: (情感极性, 置信度)
        """
        # 方案 A：本地 BERT 模型
        if self._sentiment_model and self._sentiment_tokenizer:
            return self._local_sentiment_predict(query)

        # 方案 B：LLM 零样本情感分类（降级方案）
        logger.info("perception.using_llm_sentiment")
        return await self._llm_sentiment_predict(query)

    def _local_sentiment_predict(self, query: str) -> tuple[Sentiment, float]:
        """本地 BERT 模型推理。"""
        try:
            encoding = self._sentiment_tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            encoding = {k: v.to(self._device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self._sentiment_model(**encoding)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(outputs.logits, dim=1).item()
                confidence = probs[0][pred].item()

            # 标签可能是中文（"正面"/"负面"）或英文（"positive"/"negative"）
            label_str = self._sentiment_label_map.get(str(pred), "neutral")
            # 中文→英文映射
            cn_to_en = {"正面": "positive", "负面": "negative", "中性": "neutral"}
            en_label = cn_to_en.get(label_str, label_str)
            try:
                sentiment = Sentiment(en_label)
            except ValueError:
                sentiment = Sentiment.NEUTRAL
            logger.info(
                "perception.sentiment_local",
                query=query[:30],
                label=label_str,
                confidence=f"{confidence:.3f}",
            )
            return sentiment, confidence

        except Exception as e:
            logger.error("perception.local_predict_failed", error=str(e))
            raise SentimentAnalysisError(f"本地情感分析失败: {e}")

    def _local_intent_predict(self, query: str) -> dict:
        """本地 BERT 模型做意图识别（8分类）。

        返回格式与 LLM 一致: {"intent": "...", "entities": [], "query_summary": "..."}
        """
        encoding = self._intent_tokenizer(
            query, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._intent_model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(outputs.logits, dim=1).item()
            confidence = probs[0][pred].item()

        intent_str = self._intent_label_map.get(str(pred), "other")
        logger.info(
            "perception.intent_local",
            query=query[:30], intent=intent_str, confidence=f"{confidence:.3f}",
        )
        return {"intent": intent_str, "entities": [], "query_summary": query[:50], "confidence": confidence}

    async def _llm_sentiment_predict(self, query: str) -> tuple[Sentiment, float]:
        """LLM 零样本情感分类（降级方案）。"""
        prompt = f"""分析以下用户消息的情感极性，只输出一个英文单词（positive, negative, neutral）：
用户消息：{query}
情感："""

        try:
            llm = get_llm("sentiment_llm", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip().lower() if hasattr(response, 'text') else str(response).strip().lower()

            for label in ["positive", "negative", "neutral"]:
                if label in text:
                    sentiment = Sentiment(label)
                    logger.info("perception.sentiment_llm", query=query[:30], label=label)
                    return sentiment, 0.85  # LLM 结果给一个保守的置信度

            return Sentiment.NEUTRAL, 0.5

        except Exception as e:
            logger.error("perception.llm_sentiment_failed", error=str(e))
            return Sentiment.NEUTRAL, 0.5

    # ── 情感细分 ─────────────────────────────────────────────────

    async def refine_sentiment(self, query: str, polarity: Sentiment, confidence: float = 0.0) -> SentimentLabel:
        """将 BERT 的二分类结果细化为 7 种细粒度标签。

        优化：置信度 > 0.9 时跳过 LLM，直接映射，节省 ~1s。
        """
        # 快速路径：高置信度直接映射，不调 LLM
        if confidence > 0.9:
            fast_map = {
                Sentiment.POSITIVE: SentimentLabel.HAPPY,
                Sentiment.NEGATIVE: SentimentLabel.ANXIOUS,
                Sentiment.NEUTRAL: SentimentLabel.NEUTRAL,
            }
            label = fast_map.get(polarity, SentimentLabel.NEUTRAL)
            logger.info("perception.sentiment_fastpath", polarity=polarity.value, to=label.value, confidence=confidence)
            return label

        prompt = EcomRAGPrompts.sentiment_refine_prompt().format(
            polarity=polarity.value, query=query
        )

        try:
            llm = get_llm("intent", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip().lower() if hasattr(response, 'text') else str(response).strip().lower()

            # 从回复中提取标签
            for label in SentimentLabel:
                if label.value in text:
                    logger.info("perception.sentiment_refined", from_polarity=polarity.value, to=label.value)
                    return label

            # 默认映射
            fallback_map = {
                Sentiment.POSITIVE: SentimentLabel.HAPPY,
                Sentiment.NEGATIVE: SentimentLabel.ANXIOUS,
                Sentiment.NEUTRAL: SentimentLabel.NEUTRAL,
            }
            return fallback_map.get(polarity, SentimentLabel.NEUTRAL)

        except Exception as e:
            logger.error("perception.refine_sentiment_failed", error=str(e))
            return SentimentLabel.NEUTRAL

    # ── 意图识别 + NER ────────────────────────────────────────────

    async def extract_intent_and_entities(self, query: str) -> dict:
        """从用户消息中提取意图和实体。

        优先使用本地 BERT 模型（速度快、零成本），
        本地模型不可用时降级到 LLM 结构化输出。

        :return: {"intent": str, "entities": list[dict], "query_summary": str}
        """
        # 方案 A：本地 BERT 模型（~50ms，离线）
        if self._intent_model and self._intent_tokenizer:
            return self._local_intent_predict(query)

        # 方案 B：LLM 结构化输出（~1-2s，需联网）
        logger.info("perception.using_llm_intent")
        prompt = EcomRAGPrompts.intent_ner_prompt().format(query=query)

        try:
            llm = get_llm("intent", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(
                    "perception.intent_extracted",
                    intent=result.get("intent", "unknown"),
                    entity_count=len(result.get("entities", [])),
                )
                return result

            raise IntentClassificationError(f"无法从 LLM 回复中提取 JSON: {text[:100]}")

        except Exception as e:
            logger.error("perception.intent_extraction_failed", error=str(e))
            return {"intent": "other", "entities": [], "query_summary": query[:50]}

    # ── 感知层统一入口 ──────────────────────────────────────────

    async def perceive(self, query: str) -> PerceptionResult:
        """感知层统一入口：并行执行情感分析和意图识别。

        :param query: 用户原始输入
        :return: PerceptionResult（完整的感知分析结果）
        """
        logger.info("perception.started", query=query[:50])

        # 并行执行：情感分析 + 意图识别（互不依赖，可以并行）
        import asyncio

        sentiment_task = asyncio.create_task(self.analyze_sentiment(query))
        intent_task = asyncio.create_task(self.extract_intent_and_entities(query))

        # 等待两个任务完成
        (sentiment, confidence) = await sentiment_task
        intent_data = await intent_task

        # 情感细分（依赖 polarity 结果，需串行）
        sentiment_label = await self.refine_sentiment(query, sentiment, confidence)

        # 🆕 v4.0 归一化意图（中文标签 → IntentCategory 枚举）
        intent_str = intent_data.get("intent", "知识问答")
        _INTENT_CN2EN = {
            "知识问答": "knowledge_qa",
            "业务处理": "business",
            "知识管理": "knowledge_mgmt",
            "工单处理": "escalate",
        }
        en_intent = _INTENT_CN2EN.get(intent_str, intent_str)
        try:
            intent = IntentCategory(en_intent)
        except ValueError:
            # 🆕 v4.0: 旧版10分类标签 → 新版4分类映射
            intent = _map_legacy_intent(en_intent)

        result = PerceptionResult(
            original_query=query,
            sentiment=sentiment,
            sentiment_label=sentiment_label,
            sentiment_confidence=confidence,
            intent=intent,
            intent_confidence=intent_data.get("confidence", 0.85),
            entities=intent_data.get("entities", []),
            query_summary=intent_data.get("query_summary", query[:50]),
        )

        logger.info(
            "perception.completed",
            query=query[:30],
            sentiment=result.sentiment.value,
            sentiment_label=result.sentiment_label.value,
            intent=result.intent.value,
            entity_count=len(result.entities),
        )
        return result


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        agent = PerceptionAgent()

        tests = [
            "这件衣服是什么材质的？会不会起球？",
            "我买的鞋子开胶了！太垃圾了！我要退货！",
            "快递到哪了？怎么三天了还没动静？",
            "你好呀，今天有什么优惠活动吗？",
            "这个手机和那个比哪个更好？",
        ]

        for query in tests:
            print(f"\n{'='*60}")
            print(f"输入: {query}")
            result = await agent.perceive(query)
            print(f"情感: {result.sentiment.value} → {result.sentiment_label.value} ({result.sentiment_confidence:.2f})")
            print(f"意图: {result.intent.value} ({result.intent_confidence:.2f})")
            print(f"实体: {result.entities}")
            print(f"摘要: {result.query_summary}")

    asyncio.run(test())
