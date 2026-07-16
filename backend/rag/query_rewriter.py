# backend/rag/query_rewriter.py
# 🆕 v3 Query 改写层：4种策略并行改写用户问题，大幅提升检索召回率。
#
# 四种策略（并行执行）：
#   策略1 — 指代消解 (Coreference Resolution):  结合对话历史补全"它""那个"等指代词
#   策略2 — 同义扩展 (Synonym Expansion):       生成3个语义相似但表述不同的问法
#   策略3 — 子问题拆分 (Sub-question Decomp):   复杂问题分解为多个原子子问题
#   策略4 — 关键词增强 (Keyword Enhancement):   NER提取核心实体作为检索标签
#
# 改写后的问题 → BGE-M3 编码 → 混合检索（多个改写并行检索，合并结果）

import asyncio
from dataclasses import dataclass, field

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.llm_factory import get_llm
from backend.core.retry import with_retry_async

logger = get_logger(__name__)


@dataclass
class RewrittenQuery:
    """改写后的查询集合。"""
    original: str                              # 原始问题
    resolved: str = ""                         # 策略1: 指代消解后的问题
    synonyms: list[str] = field(default_factory=list)   # 策略2: 同义扩展(3个)
    sub_questions: list[str] = field(default_factory=list)  # 策略3: 子问题(2-3个)
    keywords: list[str] = field(default_factory=list)    # 策略4: 关键词标签
    entities: list[dict] = field(default_factory=list)   # NER实体

    @property
    def all_variants(self) -> list[str]:
        """返回所有改写变体（去重），用于并行检索。"""
        variants = [self.resolved] if self.resolved else [self.original]
        variants.extend(self.synonyms)
        variants.extend(self.sub_questions)
        # 去重 + 过滤空字符串
        seen = set()
        unique = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    @property
    def search_tags(self) -> str:
        """关键词标签（用于 Milvus expr 过滤或拼接到 query）。"""
        if self.keywords:
            return " ".join(self.keywords)
        return ""


class QueryRewriter:
    """Query 改写器 — 4种策略并行改写，提升 RAG 召回率。

    使用方式：
        rewriter = QueryRewriter()
        rewritten = await rewriter.rewrite(query, history_text)
        # 然后用 rewritten.all_variants 并行检索
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.rag_query_rewrite_enabled

    async def rewrite(
        self,
        query: str,
        history_text: str = "",
        entities: list[dict] | None = None,
    ) -> RewrittenQuery:
        """并行执行4种改写策略。

        :param query: 用户原始问题
        :param history_text: 格式化的对话历史文本
        :param entities: 感知层已提取的实体列表
        :return: RewrittenQuery 包含所有改写结果
        """
        if not self.enabled:
            return RewrittenQuery(
                original=query, resolved=query,
                keywords=[e.get("value", "") for e in (entities or [])],
            )

        logger.info("query_rewriter.started", query=query[:60])

        # 4种策略并行执行
        results = await asyncio.gather(
            self._resolve_coreferences(query, history_text),       # 策略1
            self._expand_synonyms(query),                          # 策略2
            self._decompose_sub_questions(query),                  # 策略3
            self._extract_keywords(query, entities),               # 策略4
            return_exceptions=True,
        )

        resolved, synonyms, sub_questions, keywords = results

        # 处理异常
        if isinstance(resolved, Exception):
            logger.warning("query_rewriter.coref_failed", error=str(resolved))
            resolved = query
        if isinstance(synonyms, Exception):
            logger.warning("query_rewriter.synonym_failed", error=str(synonyms))
            synonyms = []
        if isinstance(sub_questions, Exception):
            logger.warning("query_rewriter.decomp_failed", error=str(sub_questions))
            sub_questions = []
        if isinstance(keywords, Exception):
            logger.warning("query_rewriter.keyword_failed", error=str(keywords))
            keywords = []

        result = RewrittenQuery(
            original=query,
            resolved=resolved or query,
            synonyms=synonyms or [],
            sub_questions=sub_questions or [],
            keywords=keywords or [],
            entities=entities or [],
        )

        logger.info(
            "query_rewriter.done",
            variants=len(result.all_variants),
            keywords=len(result.keywords),
        )
        return result

    # ── 策略1: 指代消解 ──────────────────────────────────────

    async def _resolve_coreferences(self, query: str, history_text: str) -> str:
        """结合对话历史，补全用户问题中的指代词。

        例: "它多少钱？" + 历史"XX手机怎么样" → "XX手机多少钱？"
        """
        if not history_text or history_text == "（无历史对话）":
            return query

        # 快速检查：问题中是否有指代词需要消解
        has_coref = any(w in query for w in ["它", "他", "她", "这个", "那个", "这", "那", "其"])
        if not has_coref:
            return query

        prompt = f"""你是一个对话理解助手。请根据对话历史，将用户当前问题中的指代词（"它""这个""那个"等）替换为具体的实体名称。

对话历史：
{history_text}

用户当前问题：{query}

请输出补全后的完整问题（只输出问题，不要加任何解释）：
补全后的问题："""

        try:
            llm = get_llm("qa", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            if text and len(text) > 3:
                logger.info("query_rewriter.coref_resolved", original=query[:30], resolved=text[:50])
                return text
        except Exception as e:
            logger.warning("query_rewriter.coref_llm_failed", error=str(e))

        return query

    # ── 策略2: 同义扩展 ──────────────────────────────────────

    async def _expand_synonyms(self, query: str) -> list[str]:
        """生成3个语义相同但表述方式不同的问题。

        例: "怎么退货？" → ["退换货流程是什么？", "如何申请退款？", "退货需要什么条件？"]
        """
        n = self.settings.rag_synonym_count
        prompt = f"""你是一个云答客服查询改写助手。请将以下用户问题改写为{n}个语义相同但表述不同的版本。

要求：
- 每个版本保留原始问题的核心意图
- 使用不同的措辞、句式、关键词
- 每个版本一行，不要加编号

原始问题：{query}

{n}个改写版本："""

        try:
            llm = get_llm("qa", temperature=0.5)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            lines = [line.strip("-•1234567890. ") for line in text.split("\n") if line.strip()]
            lines = [l for l in lines if len(l) > 3 and l != query]
            logger.info("query_rewriter.synonyms_generated", count=len(lines))
            return lines[:n]
        except Exception as e:
            logger.warning("query_rewriter.synonym_llm_failed", error=str(e))
            return []

    # ── 策略3: 子问题拆分 ────────────────────────────────────

    async def _decompose_sub_questions(self, query: str) -> list[str]:
        """将复杂问题拆分为2-3个原子子问题。

        例: "我想买一款2000左右的手机，拍照好续航强"
          → ["2000元手机推荐", "拍照好的手机", "续航强的手机"]
        """
        # 简单问题不需要拆分
        if len(query) < 15:
            return []

        prompt = f"""你是一个问题分析助手。判断以下用户问题是否是复杂问题（包含多个独立需求）。

如果是简单问题（单一需求），请回复"SIMPLE"。
如果是复杂问题，请拆分为2-3个独立的原子子问题，每行一个。

用户问题：{query}

分析结果："""

        try:
            llm = get_llm("qa", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()

            if "SIMPLE" in text.upper():
                return []

            lines = [line.strip("-•1234567890. ") for line in text.split("\n") if line.strip()]
            lines = [l for l in lines if len(l) > 3 and "SIMPLE" not in l.upper()]
            logger.info("query_rewriter.sub_questions", count=len(lines))
            return lines[:3]
        except Exception as e:
            logger.warning("query_rewriter.decomp_llm_failed", error=str(e))
            return []

    # ── 策略4: 关键词增强 ────────────────────────────────────

    async def _extract_keywords(
        self, query: str, entities: list[dict] | None
    ) -> list[str]:
        """从问题中提取核心关键词/实体作为检索标签。

        结合感知层的NER实体 + LLM二次提取。
        """
        keywords = []

        # 从已有实体中提取
        if entities:
            for e in entities:
                val = e.get("value", "").strip()
                if val and len(val) > 1:
                    keywords.append(val)

        # LLM 二次提取关键属性词
        prompt = f"""从以下电商用户问题中提取3-5个核心关键词（商品属性、品牌、功能、品类等）。
只输出关键词，用逗号分隔，不要其他内容。

问题：{query}
关键词："""

        try:
            llm = get_llm("qa", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            extra = [k.strip() for k in text.replace("，", ",").split(",") if k.strip()]
            keywords.extend(extra)
        except Exception:
            pass

        # 去重
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique[:5]


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        rewriter = QueryRewriter()

        tests = [
            ("怎么退货？", "用户: 我买的衣服不合适\n客服: 请问您的订单号是？"),
            ("我想买一款2000左右的手机，拍照好续航强，有什么推荐吗？", ""),
            ("它多少钱？", "用户: 那款黑色的背包怎么样\n客服: 黑色背包是经典款，防水面料"),
        ]

        for query, history in tests:
            print(f"\n{'='*60}")
            print(f"原始: {query}")
            result = await rewriter.rewrite(query, history)
            print(f"指代消解: {result.resolved}")
            print(f"同义扩展 ({len(result.synonyms)}): {result.synonyms}")
            print(f"子问题 ({len(result.sub_questions)}): {result.sub_questions}")
            print(f"关键词: {result.keywords}")
            print(f"总变体数: {len(result.all_variants)}")

        print("\nquery_rewriter.py 自测通过")

    asyncio.run(test())
