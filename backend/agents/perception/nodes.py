# backend/agents/perception/nodes.py
# Perception Agent — LangGraph 节点函数
#
# 每个节点函数接收 PerceptionState，返回部分 state dict。
# 保留 PerceptionAgent 类作为模块级单例，避免反复加载 BERT 模型。

import json
import os
import re
import asyncio
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

from backend.agents.perception.state import PerceptionState
from backend.agents.perception.prompts import (
    SENTIMENT_LLM_PROMPT,
    LEGACY_10_TO_4,
    FINE_TO_POLARITY,
    INTENT_CN2EN,
)

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


def _map_legacy_intent(legacy_value: str) -> IntentCategory:
    """将 BERT 10分类标签映射到 v4.0 4分类。"""
    mapped = LEGACY_10_TO_4.get(legacy_value, "knowledge_qa")
    return IntentCategory(mapped)


# ═══════════════════════════════════════════════════════════════
# PerceptionAgent 类（保留完整业务逻辑，模块级单例）
# ═══════════════════════════════════════════════════════════════

class PerceptionAgent:
    """感知层智能体：情感分析 + 意图识别 + NER 实体抽取。

    策略：
    - 情感极性：优先加载本地 BERT 模型
    - 情感细分：调用 LLM 将极性映射为 7 种细粒度标签
    - 意图+NER：调用 LLM structured output 一次性完成
    """

    def __init__(self):
        self._sentiment_model = None
        self._sentiment_tokenizer = None
        self._device = "cpu"
        self._sentiment_label_map = {"0": "negative", "1": "positive"}

        self._intent_model = None
        self._intent_tokenizer = None
        self._intent_label_map: dict[str, str] = {}

        if _BERT_AVAILABLE:
            self._load_sentiment_model()
            self._load_intent_model()
        else:
            logger.info("perception.bert_skipped", hint="纯LLM模式，跳过BERT加载")

    def _load_sentiment_model(self):
        """加载情感分类模型（按优先级依次尝试）。"""
        settings = get_settings()

        # 优先级1：自己训练的7分类模型
        # nodes.py 在 perception/ 子目录下，需往上3级到 backend/
        _backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        trained_path = os.path.join(
            _backend_dir, "training", "saved_models", "sentiment_classifier"
        )
        if self._try_load_from_dir(trained_path, is_sentiment=True):
            logger.info("perception.sentiment_loaded_trained_7class",
                       path=trained_path, labels=self._sentiment_label_map)
            return

        # 优先级2：ModelScope StructBERT 7分类情绪模型
        emotion_7class_path = settings.emotion_7class_model_dir
        if os.path.isdir(emotion_7class_path) and os.path.exists(
            os.path.join(emotion_7class_path, "pytorch_model.bin")
        ):
            if self._try_load_from_dir(emotion_7class_path, is_sentiment=True):
                logger.info("perception.sentiment_loaded_modelscope_7class",
                           path=emotion_7class_path, labels=self._sentiment_label_map)
                return

        # 优先级3：本地 BERT 情感模型
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

        # 优先级4：LLM 降级
        logger.info("perception.no_local_sentiment_model", fallback="llm_zero_shot")

    def _load_intent_model(self):
        """加载意图分类模型。"""
        _backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        trained_path = os.path.join(
            _backend_dir, "training", "saved_models", "intent_classifier"
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

    async def analyze_sentiment(self, query: str) -> tuple[Sentiment, float, str]:
        """分析用户消息的情感极性。"""
        if self._sentiment_model and self._sentiment_tokenizer:
            return self._local_sentiment_predict(query)

        logger.info("perception.using_llm_sentiment")
        sentiment, confidence = await self._llm_sentiment_predict(query)
        return sentiment, confidence, ""

    def _local_sentiment_predict(self, query: str) -> tuple[Sentiment, float, str]:
        """本地 BERT 模型推理（支持 2分类 和 7分类模型）。"""
        try:
            encoding = self._sentiment_tokenizer(
                query, truncation=True, padding=True, max_length=128, return_tensors="pt",
            )
            encoding = {k: v.to(self._device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self._sentiment_model(**encoding)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(outputs.logits, dim=1).item()
                confidence = probs[0][pred].item()

            num_labels = outputs.logits.shape[1]

            if num_labels >= 7:
                fine_label = self._sentiment_label_map.get(str(pred), "neutral")
                if confidence < 0.35:
                    fine_label = "neutral"
                    confidence = 0.5
                polarity_str = FINE_TO_POLARITY.get(fine_label, "neutral")
                try:
                    sentiment = Sentiment(polarity_str)
                except ValueError:
                    sentiment = Sentiment.NEUTRAL
            else:
                label_str = self._sentiment_label_map.get(str(pred), "neutral")
                cn_to_en = {"正面": "positive", "负面": "negative", "中性": "neutral"}
                polarity_str = cn_to_en.get(label_str, label_str)
                try:
                    sentiment = Sentiment(polarity_str)
                except ValueError:
                    sentiment = Sentiment.NEUTRAL
                fine_label = polarity_str

            logger.info(
                "perception.sentiment_local",
                query=query[:30], fine_label=fine_label,
                num_labels=num_labels, confidence=f"{confidence:.3f}",
            )
            return sentiment, confidence, fine_label

        except Exception as e:
            logger.error("perception.local_predict_failed", error=str(e))
            raise SentimentAnalysisError(f"本地情感分析失败: {e}")

    def _local_intent_predict(self, query: str) -> dict:
        """本地 BERT 模型做意图识别（8分类）。"""
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
        prompt = SENTIMENT_LLM_PROMPT.format(query=query)

        try:
            llm = get_llm("sentiment_llm", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip().lower() if hasattr(response, 'text') else str(response).strip().lower()

            for label in ["positive", "negative", "neutral"]:
                if label in text:
                    sentiment = Sentiment(label)
                    logger.info("perception.sentiment_llm", query=query[:30], label=label)
                    return sentiment, 0.85

            return Sentiment.NEUTRAL, 0.5

        except Exception as e:
            logger.error("perception.llm_sentiment_failed", error=str(e))
            return Sentiment.NEUTRAL, 0.5

    async def refine_sentiment(self, query: str, polarity: Sentiment, confidence: float = 0.0) -> SentimentLabel:
        """将 BERT 的二分类结果细化为 7 种细粒度标签。"""
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

            for label in SentimentLabel:
                if label.value in text:
                    logger.info("perception.sentiment_refined", from_polarity=polarity.value, to=label.value)
                    return label

            fallback_map = {
                Sentiment.POSITIVE: SentimentLabel.HAPPY,
                Sentiment.NEGATIVE: SentimentLabel.ANXIOUS,
                Sentiment.NEUTRAL: SentimentLabel.NEUTRAL,
            }
            return fallback_map.get(polarity, SentimentLabel.NEUTRAL)

        except Exception as e:
            logger.error("perception.refine_sentiment_failed", error=str(e))
            return SentimentLabel.NEUTRAL

    async def extract_intent_and_entities(self, query: str) -> dict:
        """从用户消息中提取意图和实体。"""
        if self._intent_model and self._intent_tokenizer:
            return self._local_intent_predict(query)

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

    async def perceive(self, query: str) -> PerceptionResult:
        """感知层统一入口：并行执行情感分析和意图识别。"""
        logger.info("perception.started", query=query[:50])

        sentiment_task = asyncio.create_task(self.analyze_sentiment(query))
        intent_task = asyncio.create_task(self.extract_intent_and_entities(query))

        (sentiment, confidence, local_fine_label) = await sentiment_task
        intent_data = await intent_task

        if local_fine_label:
            try:
                sentiment_label = SentimentLabel(local_fine_label)
            except ValueError:
                sentiment_label = await self.refine_sentiment(query, sentiment, confidence)
        else:
            sentiment_label = await self.refine_sentiment(query, sentiment, confidence)

        intent_str = intent_data.get("intent", "知识问答")
        en_intent = INTENT_CN2EN.get(intent_str, intent_str)
        try:
            intent = IntentCategory(en_intent)
        except ValueError:
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
            fine_intent=intent_str,
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


# ═══════════════════════════════════════════════════════════════
# 模块级单例（整个进程复用，避免反复加载 BERT 模型）
# ═══════════════════════════════════════════════════════════════

_perception_agent: Optional[PerceptionAgent] = None


def _get_agent() -> PerceptionAgent:
    """获取 PerceptionAgent 模块级单例。"""
    global _perception_agent
    if _perception_agent is None:
        _perception_agent = PerceptionAgent()
    return _perception_agent


# ═══════════════════════════════════════════════════════════════
# LangGraph 节点函数
# ═══════════════════════════════════════════════════════════════

async def perceive_node(state: PerceptionState) -> dict:
    """节点：感知层 —— 情感分析 + 意图识别 + NER。

    从 state["query"] 读取用户输入，返回 {"perception_result": PerceptionResult}。
    """
    query = state["query"]
    logger.info("perception.node_started", query=query[:30])

    try:
        agent = _get_agent()
        perception = await agent.perceive(query)
        return {"perception_result": perception}
    except Exception as e:
        logger.error("perception.node_failed", error=str(e))
        fallback = PerceptionResult(
            original_query=query,
            sentiment=Sentiment.NEUTRAL,
            sentiment_label=SentimentLabel.NEUTRAL,
            sentiment_confidence=0.5,
            intent=IntentCategory.KNOWLEDGE_QA,
        )
        return {"perception_result": fallback, "error": f"感知层异常: {e}"}


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        agent = _get_agent()

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
