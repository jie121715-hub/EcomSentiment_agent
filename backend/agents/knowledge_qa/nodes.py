# backend/agents/knowledge_qa/nodes.py
# KnowledgeQA Agent — LangGraph 节点函数 + KnowledgeQAAgent 类
#
# 这是系统的核心Agent，处理约 80% 的用户请求。
# 完整 RAG 管线：Redis缓存 → Query改写 → 混合检索 → Reranker → LLM → 后处理 → 缓存

import time
import asyncio
from typing import Optional, AsyncGenerator

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.retry import with_retry_async
from backend.core.llm_factory import get_llm
from backend.models.schemas import (
    PerceptionResult, RouteDecision, RetrievalStrategy,
    AgentResponse, AgentMessage, ConversationHistory,
    IntentCategory, SentimentLabel,
)
from backend.rag.retriever import EcomRetriever
from backend.rag.prompts import EcomRAGPrompts

from backend.agents.knowledge_qa.state import KnowledgeQAState
from backend.agents.knowledge_qa.prompts import (
    QUICK_REPLIES, CHITCHAT_PROMPT, CHITCHAT_TONE_MAP,
)

logger = get_logger(__name__)


class KnowledgeQAAgent:
    """知识应答Agent — 完整 RAG 管线 + 缓存 + 后处理。"""

    def __init__(self):
        self.settings = get_settings()
        self.retriever = EcomRetriever()
        self._cache = None
        self._post_processor = None
        self._bm25 = None

    @property
    def cache(self):
        if self._cache is None:
            from backend.rag.cache import RAGCache
            self._cache = RAGCache()
        return self._cache

    @property
    def post_processor(self):
        if self._post_processor is None:
            from backend.rag.post_processor import AnswerPostProcessor
            self._post_processor = AnswerPostProcessor()
        return self._post_processor

    @property
    def bm25(self):
        if self._bm25 is None:
            from backend.rag.bm25_search import BM25FAQSearch
            self._bm25 = BM25FAQSearch()
        return self._bm25

    # ── 三层快速路径 ──────────────────────────────────────

    async def _try_fast_path(self, query: str, source_filter: str = "") -> str | None:
        cached = await self.cache.get(query, source_filter)
        if cached:
            logger.info("knowledge_qa.fast_path_redis", query=query[:30])
            return cached

        if self._is_product_inquiry(query):
            logger.info("knowledge_qa.skip_faq_product", query=query[:30])
            return None

        try:
            if not self.bm25._initialized:
                await self.bm25.initialize()
            faq_answer = await self.bm25.search(query)
            if faq_answer:
                logger.info("knowledge_qa.fast_path_bm25", query=query[:30])
                return faq_answer
        except Exception as e:
            logger.warning("knowledge_qa.bm25_fallback", error=str(e)[:60])

        return None

    @staticmethod
    def _is_product_inquiry(query: str) -> bool:
        biz_keywords = ["退货", "退款", "退换", "换货", "取消", "改地址",
                       "物流", "快递", "发货", "配送", "到哪了", "单号",
                       "订单", "下单", "运费", "包邮", "优惠券怎么", "申请", "流程", "步骤", "怎么办"]
        is_biz = any(w in query for w in biz_keywords)
        product_keywords = ["参数", "配置", "性能", "功能", "规格", "型号",
                           "多少钱", "价格", "好不好", "怎么样", "测评",
                           "对比", "区别", "哪个好", "推荐", "值得买",
                           "材质", "尺码", "适合", "适用", "兼容",
                           "续航", "电池", "屏幕", "拍照", "处理器", "芯片",
                           "产品", "商品", "牌子", "品牌", "是什么", "干什么",
                           "干嘛", "有什么用", "作用", "功效", "成分",
                           "介绍", "说明", "了解", "看看", "看下", "知道"]
        is_product = any(w in query for w in product_keywords)
        category_words = ["手机", "电脑", "耳机", "手表", "平板", "笔记本",
                         "衣服", "鞋子", "化妆品", "护肤品", "防晒", "口红",
                         "包包", "家电", "食品", "饮料", "零食"]
        is_category = any(w in query for w in category_words)
        return (is_product or is_category) and not is_biz

    # ── RAG 检索 ─────────────────────────────────────────

    async def retrieve(self, query: str, decision: RouteDecision,
                       history: list[ConversationHistory] | None = None,
                       shop_id: str = "") -> tuple[list, str, dict]:
        if decision.skip_rag:
            logger.info("knowledge_qa.rag_skipped")
            return [], "跳过检索", {}

        source_filter = decision.source_filter or None
        strategy = decision.strategy
        history_text = self._format_history(history or [])

        logger.info("knowledge_qa.retrieving", strategy=strategy.value, shop_id=shop_id)

        # 多租户：优先用 MultiTenantRetriever 检索 eco_rag（有 shop_id 则过滤，无则跨店检索）
        try:
            from backend.rag.multi_tenant_retriever import MultiTenantRetriever
            mt_retriever = MultiTenantRetriever()
            mt_docs = mt_retriever.search(query, shop_id=shop_id, top_k=self.settings.rag_candidate_m)
            if mt_docs:
                final_docs = [type('Doc', (), {
                    'page_content': d['content'],
                    'metadata': {'source': f"{d.get('collection', '')}/{d.get('category', '')}",
                                 'chunk_type': d.get('chunk_type', '')},
                })() for d in mt_docs]
                meta = {
                    "quality_passed": True,
                    "total_candidates": len(final_docs),
                    "source": "eco_rag",
                    "shop_id": shop_id or "all",
                }
                logger.info("knowledge_qa.eco_rag_done", docs=len(final_docs), shop_id=shop_id or "all")
                return final_docs, strategy.value, meta
        except Exception as e:
            logger.warning("knowledge_qa.eco_rag_failed", error=str(e)[:100])

        # 旧版检索（无 shop_id 或 eco_rag 失败）
        pipeline_result = await self.retriever.retrieve_with_pipeline(
            query=query, source_filter=source_filter, history_text=history_text,
        )

        docs = pipeline_result.get("docs", [])
        quality_passed = pipeline_result.get("quality_passed", True)
        quality_reason = pipeline_result.get("quality_reason", "")

        if not quality_passed:
            logger.warning("knowledge_qa.quality_failed", reason=quality_reason, doc_count=len(docs))

        final_docs = docs[:self.settings.rag_candidate_m]
        meta = {
            "quality_passed": quality_passed, "quality_reason": quality_reason,
            "total_candidates": pipeline_result.get("total_candidates", 0),
            "rewritten": pipeline_result.get("rewritten_query"),
        }

        logger.info("knowledge_qa.retrieval_done", total=pipeline_result.get("total_candidates", 0),
                   final=len(final_docs), quality=quality_passed)
        return final_docs, strategy.value, meta

    async def _legacy_retrieve(self, query: str, decision: RouteDecision) -> list:
        source_filter = decision.source_filter or None
        strategy = decision.strategy
        if strategy == RetrievalStrategy.DIRECT:
            return self.retriever.search(query, source_filter=source_filter)
        elif strategy == RetrievalStrategy.HYDE:
            return await self._retrieve_with_hyde(query, source_filter)
        elif strategy == RetrievalStrategy.SUBQUERY:
            return await self._retrieve_with_subqueries(query, source_filter)
        elif strategy == RetrievalStrategy.BACKTRACK:
            return await self._retrieve_with_backtrack(query, source_filter)
        return []

    async def _collect_llm_response(self, prompt: str) -> str:
        try:
            llm = get_llm("qa", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            logger.error("knowledge_qa.llm_collect_failed", error=str(e))
            return ""

    async def _retrieve_with_hyde(self, query: str, source_filter: str | None) -> list:
        hyde_prompt = EcomRAGPrompts.hyde_prompt().format(query=query)
        hypo_answer = await self._collect_llm_response(hyde_prompt)
        if hypo_answer:
            return self.retriever.search(hypo_answer, source_filter=source_filter)
        return self.retriever.search(query, source_filter=source_filter)

    async def _retrieve_with_subqueries(self, query: str, source_filter: str | None) -> list:
        subquery_prompt = EcomRAGPrompts.subquery_prompt().format(query=query)
        subqueries_text = await self._collect_llm_response(subquery_prompt)
        subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
        if not subqueries:
            return self.retriever.search(query, source_filter=source_filter)
        all_docs = []
        for sub_q in subqueries[:3]:
            all_docs.extend(self.retriever.search(sub_q, source_filter=source_filter))
        seen = set()
        unique = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
        return unique

    async def _retrieve_with_backtrack(self, query: str, source_filter: str | None) -> list:
        backtrack_prompt = EcomRAGPrompts.backtracking_prompt().format(query=query)
        simplified = await self._collect_llm_response(backtrack_prompt)
        if simplified:
            return self.retriever.search(simplified, source_filter=source_filter)
        return self.retriever.search(query, source_filter=source_filter)

    def _format_history(self, history: list[ConversationHistory]) -> str:
        if not history:
            return "（无历史对话）"
        lines = ["## 对话历史"]
        for i, turn in enumerate(history[-self.settings.max_history_turns:], 1):
            lines.append(f"用户第{i}轮: {turn.question}")
            lines.append(f"客服第{i}轮: {turn.answer}")
        lines.append("## 当前对话")
        return "\n".join(lines)

    # ── 回答生成 ─────────────────────────────────────────

    async def answer(
        self, query: str, perception: PerceptionResult, decision: RouteDecision,
        context_docs: list, history: list[ConversationHistory] | None = None,
        retrieval_meta: dict | None = None,
    ) -> AgentResponse:
        start_time = time.time()
        history_list = history or []
        source_filter = decision.source_filter or ""

        # 步骤1: Redis 缓存检查
        cached = await self.cache.get(query, source_filter)
        if cached:
            processing_time = (time.time() - start_time) * 1000
            return AgentResponse(success=True,
                message=AgentMessage(role="assistant", content=cached,
                    sentiment_detected=perception.sentiment_label.value, intent_detected=perception.intent.value),
                processing_time_ms=processing_time)

        # 闲聊快速通道
        quick = self._quick_reply_check(query)
        if quick:
            processing_time = (time.time() - start_time) * 1000
            return AgentResponse(success=True,
                message=AgentMessage(role="assistant", content=quick,
                    sentiment_detected=perception.sentiment_label.value, intent_detected=perception.intent.value),
                processing_time_ms=processing_time)

        # 第2层快速路径: BM25 + MySQL FAQ
        faq_answer = await self._try_fast_path(query, source_filter)
        if faq_answer:
            processing_time = (time.time() - start_time) * 1000
            return AgentResponse(success=True,
                message=AgentMessage(role="assistant", content=faq_answer,
                    sentiment_detected=perception.sentiment_label.value, intent_detected=perception.intent.value),
                processing_time_ms=processing_time)

        # 步骤2: 质量检查 + 高危门控
        meta = retrieval_meta or {}
        quality_ok = meta.get("quality_passed", True)
        no_docs = (not quality_ok and not context_docs)

        from backend.rag.post_processor import AnswerPostProcessor
        is_high_risk = AnswerPostProcessor.is_high_risk(query, perception.intent.value)
        if is_high_risk:
            policy_covered = any(
                any(kw in (doc.page_content if hasattr(doc,'page_content') else str(doc))
                    for kw in AnswerPostProcessor.HIGH_RISK_KEYWORDS[:6])
                for doc in context_docs
            ) if context_docs else False

            if not policy_covered:
                processing_time = (time.time() - start_time) * 1000
                return AgentResponse(success=True,
                    message=AgentMessage(role="assistant",
                        content=f"您咨询的问题涉及赔付/政策等重要信息，为确保100%准确，已为您转接人工客服。\n\n📞 客服电话：{self.settings.customer_service_phone}\n🕐 服务时间：7×24 小时\n\n💡 您也可以在「我的订单」页面查看本店退换货政策及售后规则。",
                        sentiment_detected=perception.sentiment_label.value, intent_detected=perception.intent.value),
                    processing_time_ms=processing_time)

        if no_docs:
            logger.info("knowledge_qa.llm_fallback", query=query[:30])

        # 步骤3: 构建上下文
        context = self.retriever.format_context(context_docs)
        history_text = self._format_history(history_list)

        taobao_context = ""
        if not context_docs and perception.intent == IntentCategory.KNOWLEDGE_QA:
            try:
                from backend.utils.taobao_importer import search_products
                items = await search_products(query)
                if items:
                    taobao_context = "## 淘宝店铺实时搜索结果\n"
                    for i, item in enumerate(items, 1):
                        taobao_context += f"[淘宝{i}] {item['title']}，售价{item['price']}元"
                        if item.get("props"):
                            taobao_context += f"，规格：{item['props'].replace(';', '，')}"
                        taobao_context += "\n"
                    logger.info("knowledge_qa.taobao_fallback", found=len(items))
            except Exception:
                pass

        full_context = context
        if taobao_context:
            full_context = (context + "\n\n" + taobao_context) if context else taobao_context
        if not full_context:
            full_context = "（本店暂无该商品数据，请用你的通识知识自然回答，把免责融进句子里而不是单独成段，参考示范：'不过库存变化快，具体的还是以商品页面为准哦～'）"

        # 步骤4: Prompt 组装
        is_recommend = any(w in perception.original_query for w in ["推荐", "建议", "哪个好", "选哪个", "买什么"])
        if is_recommend:
            prompt_template = EcomRAGPrompts.recommend_prompt()
            full_prompt = prompt_template.format(
                tone_instruction=decision.tone_instruction, history=history_text,
                question=query, context=full_context,
            )
        else:
            prompt_template = EcomRAGPrompts.rag_prompt()
            full_prompt = prompt_template.format(
                tone_instruction=decision.tone_instruction,
                extra_instruction=decision.dynamic_prompt_extra,
                context=full_context or "（无参考知识，请基于电商常识回答）",
                history=history_text, question=query,
                phone=self.settings.customer_service_phone,
            )

        # 步骤5: LLM 生成
        try:
            llm = get_llm("qa", temperature=0.1)
            response = await with_retry_async(llm.ainvoke, full_prompt)
            content = response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            logger.error("knowledge_qa.llm_call_failed", error=str(e))
            content = self._get_fallback_message()

        # 步骤6: 答案后处理
        post_result = await self.post_processor.process(answer=content, context_docs=context_docs, query=query)
        final_content = post_result["final_answer"]

        # 步骤7: 异步写入缓存
        if not post_result.get("hallucination_flag"):
            await self.cache.set(query, source_filter, final_content)

        processing_time = (time.time() - start_time) * 1000
        history_updated = history_list[-self.settings.max_history_turns:]
        history_updated.append(ConversationHistory(
            question=query, answer=final_content,
            sentiment=perception.sentiment_label.value, intent=perception.intent.value,
        ))

        logger.info("knowledge_qa.answer_done", time_ms=f"{processing_time:.0f}",
                   hallucination=post_result.get("hallucination_flag"))
        return AgentResponse(
            success=True,
            message=AgentMessage(role="assistant", content=final_content,
                sentiment_detected=f"{perception.sentiment.value} -> {perception.sentiment_label.value}",
                intent_detected=perception.intent.value),
            history_updated=history_updated,
            processing_time_ms=processing_time,
        )

    # ── 流式生成 ─────────────────────────────────────────

    async def answer_stream(
        self, query: str, perception: PerceptionResult, decision: RouteDecision,
        context_docs: list, history: list[ConversationHistory] | None = None,
    ) -> AsyncGenerator[str, None]:
        logger.info("knowledge_qa.answer_stream_started", query=query[:30])

        # 有 RAG 上下文时跳过 BM25，用 LLM 基于检索结果生成
        fast = None
        if not context_docs:
            fast = await self._try_fast_path(query, decision.source_filter or "")
        if fast:
            yield fast
            return

        context = self._format_context(context_docs)
        history_text = self._format_history(history or [])

        logger.info("knowledge_qa.context_debug", context=context[:300])

        rag_prompt = EcomRAGPrompts.rag_prompt()
        full_prompt = rag_prompt.format(
            tone_instruction=decision.tone_instruction,
            extra_instruction=decision.dynamic_prompt_extra,
            context=context or "（无参考知识，请基于电商常识回答）",
            history=history_text, question=query,
            phone=self.settings.customer_service_phone,
        )

        try:
            full_response = []
            llm = get_llm("qa", temperature=0.1, streaming=True)
            async for chunk in llm.astream(full_prompt):
                token = ""
                if hasattr(chunk, 'text') and chunk.text:
                    token = chunk.text
                elif hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                if token:
                    full_response.append(token)
                    yield token

            full_text = "".join(full_response)
            if full_text:
                await self.cache.set(query, decision.source_filter or "", full_text)
        except Exception as e:
            logger.error("knowledge_qa.stream_failed", error=str(e))
            yield self._get_fallback_message()

    # ── 闲聊处理 ─────────────────────────────────────────

    @staticmethod
    def _format_context(context_docs: list) -> str:
        """格式化上下文文档为 LLM prompt 可用的字符串。"""
        if not context_docs:
            return ""
        # 判断是 eco_rag 文档还是旧版 LangChain Document
        sample = context_docs[0]
        is_dict = isinstance(sample, dict)
        parts = []
        for i, doc in enumerate(context_docs, 1):
            if is_dict:
                content = doc.get('content', str(doc))
                cat = doc.get('category', '未知')
                col = doc.get('collection', '')
                parts.append(f"[{col}/{cat}]\n{content}")
            else:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                src = doc.metadata.get('source', '未知') if hasattr(doc, 'metadata') else '未知'
                parts.append(f"[来源{i}: {src}]\n{content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _quick_reply_check(query: str) -> str | None:
        q = query.strip().lower()
        if q in ["你好", "您好", "嗨", "hi", "hello", "在吗", "在不在"]:
            return QUICK_REPLIES["greeting"]
        if any(w in q for w in ["谢谢", "感谢", "多谢", "谢谢你", "thanks", "thank"]):
            return QUICK_REPLIES["thanks"]
        if any(w in q for w in ["再见", "拜拜", "晚安", "bye", "拜"]):
            return QUICK_REPLIES["bye"]
        if any(w in q for w in ["你真棒", "太厉害了", "好聪明", "不错不错"]):
            return QUICK_REPLIES["praise"]
        if any(w in q for w in ["你好吗", "你怎么样", "你会什么"]):
            return QUICK_REPLIES["how_are_you"]
        return None

    async def _handle_chitchat(self, query: str, sentiment_label: SentimentLabel) -> str:
        tone = CHITCHAT_TONE_MAP.get(sentiment_label.value, "保持友好热情")
        prompt = CHITCHAT_PROMPT.format(query=query, tone=tone)

        try:
            llm = get_llm("qa", temperature=0.5)
            response = await with_retry_async(llm.ainvoke, prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            logger.error("knowledge_qa.chitchat_llm_failed", error=str(e))
            return QUICK_REPLIES["greeting"]

    def _get_fallback_message(self) -> str:
        return (
            f"非常抱歉，当前系统繁忙，暂时无法为您提供准确的回复。"
            f"建议您稍后再试，或拨打客服电话 {self.settings.customer_service_phone} "
            f"联系人工客服获取帮助。"
        )


# ═══════════════════════════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════════════════════════

_knowledge_qa_agent: Optional[KnowledgeQAAgent] = None


def _get_agent() -> KnowledgeQAAgent:
    global _knowledge_qa_agent
    if _knowledge_qa_agent is None:
        _knowledge_qa_agent = KnowledgeQAAgent()
    return _knowledge_qa_agent


# ═══════════════════════════════════════════════════════════════
# LangGraph 节点函数
# ═══════════════════════════════════════════════════════════════

async def retrieve_node(state: KnowledgeQAState) -> dict:
    """节点：RAG 检索 —— 混合检索 + 精排 + 质量检查。"""
    query = state["query"]
    decision = state.get("decision")
    history = state.get("history", [])

    if decision is None or decision.skip_rag:
        return {"context_docs": [], "retrieval_meta": {}}

    logger.info("knowledge_qa.retrieve_node", strategy=decision.strategy.value if decision else "none")
    try:
        agent = _get_agent()
        shop_id = state.get("shop_id", "")
        docs, _, meta = await agent.retrieve(query, decision, history, shop_id)
        return {"context_docs": docs, "retrieval_meta": meta}
    except Exception as e:
        logger.error("knowledge_qa.retrieve_failed", error=str(e))
        return {"context_docs": [], "retrieval_meta": {}}


async def answer_node(state: KnowledgeQAState) -> dict:
    """节点：LLM 生成回复 —— 缓存 → 快速路径 → 质量门控 → Prompt → LLM → 后处理。"""
    query = state["query"]
    perception = state.get("perception")
    decision = state.get("decision")
    context_docs = state.get("context_docs", [])
    history = state.get("history", [])
    retrieval_meta = state.get("retrieval_meta")

    logger.info("knowledge_qa.answer_node", query=query[:30])
    try:
        agent = _get_agent()
        response = await agent.answer(
            query=query, perception=perception, decision=decision,
            context_docs=context_docs, history=history, retrieval_meta=retrieval_meta,
        )
        return {"agent_response": response}
    except Exception as e:
        logger.error("knowledge_qa.answer_failed", error=str(e))
        settings = get_settings()
        fallback = AgentResponse(
            success=False,
            message=AgentMessage(role="assistant",
                content=f"抱歉，处理您的请求时遇到问题。请拨打客服电话 {settings.customer_service_phone} 联系人工客服。"),
            processing_time_ms=0,
        )
        return {"agent_response": fallback}


async def fast_path_node(state: KnowledgeQAState) -> dict:
    """节点：快速路径检查 —— Redis + BM25 FAQ。"""
    query = state["query"]
    decision = state.get("decision")
    source_filter = decision.source_filter if decision else ""

    try:
        agent = _get_agent()
        result = await agent._try_fast_path(query, source_filter)
        if result:
            return {"agent_response": AgentResponse(
                success=True,
                message=AgentMessage(role="assistant", content=result,
                    sentiment_detected=state.get("perception", None) and state["perception"].sentiment_label.value,
                    intent_detected=state.get("perception", None) and state["perception"].intent.value),
            ), "fast_path_hit": True}
        return {"fast_path_hit": False}
    except Exception as e:
        logger.warning("knowledge_qa.fast_path_error", error=str(e))
        return {"fast_path_hit": False}


def check_fast_path(state: KnowledgeQAState) -> str:
    """快速路径路由：命中 → END，未命中 → retrieve。"""
    if state.get("fast_path_hit"):
        return "end"
    return "retrieve"
