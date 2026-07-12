# backend/agents/knowledge_qa.py
# 🆕 v3 — 知识应答Agent：接入完整 RAG 管线。
#
# 流程（对应 RAG 流程图）：
#   1. Redis 缓存检查 → 命中直接返回
#   2. Query 改写（4策略并行）
#   3. 混合检索（4路并行 + RRF 融合）
#   4. Reranker 精排（Cross-Encoder 深度打分）
#   5. 质量检查（首条 > 0.6？）→ 不通过则返回兜底话术
#   6. Prompt 组装（Top-5文档 + 改写Query + 历史）
#   7. LLM 生成（temperature 0.1 确保稳定性）
#   8. 答案后处理（幻觉检测/敏感词过滤/格式化）
#   9. 异步写入 Redis 缓存（TTL = 1小时）
#  10. 返回用户

import time
import asyncio
from typing import AsyncGenerator

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

logger = get_logger(__name__)


class KnowledgeQAAgent:
    """🆕 v3 知识应答Agent — 完整 RAG 管线 + 缓存 + 后处理。

    这是系统的核心Agent，处理约 80% 的用户请求。
    v3 新增：Query改写、混合检索、精排、质量检查、答案后处理、Redis缓存。
    """

    # ── 快速回复模板（不调LLM，毫秒级响应，合并自 chitchat.py）──
    QUICK_REPLIES = {
        "greeting": "您好呀！👋 我是您的专属智能客服小助手，有什么可以帮您的吗？",
        "thanks": "不客气！😊 能帮到您我也很开心～如果有其他问题随时找我哦！",
        "bye": "再见！祝您购物愉快～🛒✨ 有需要随时回来找我！",
        "praise": "谢谢您的认可！💪 我会继续努力的～",
        "how_are_you": "我很好呀，感谢关心！😄 24小时在线待命，随时准备帮您解决购物问题～",
    }

    def __init__(self):
        self.settings = get_settings()
        self.retriever = EcomRetriever()
        self._cache = None
        self._post_processor = None
        self._bm25 = None  # 🆕 BM25 FAQ 检索（懒加载）

    # ── 懒加载 v3 组件 ─────────────────────────────────────

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
        """🆕 BM25 FAQ 检索器懒加载。"""
        if self._bm25 is None:
            from backend.rag.bm25_search import BM25FAQSearch
            self._bm25 = BM25FAQSearch()
        return self._bm25

    # ═══════════════════════════════════════════════════════════
    # 🆕 三层快速路径：Redis → BM25+MySQL → RAG
    # ═══════════════════════════════════════════════════════════

    async def _try_fast_path(
        self, query: str, source_filter: str = ""
    ) -> str | None:
        """尝试快速路径返回答案（不走完整 RAG 管线）。

        第1层：Redis 精确缓存（RAGCache）— <5ms
        第2层：BM25 + MySQL FAQ — ~15ms
               ⚠️ 商品咨询类问题跳过FAQ (FAQ只匹配操作类问题)

        :return: 命中返回答案字符串，未命中返回 None
        """
        # ── 第1层：Redis 精确缓存 ──────────────────────────
        cached = await self.cache.get(query, source_filter)
        if cached:
            logger.info("knowledge_qa.fast_path_redis", query=query[:30])
            return cached

        # ── 🚫 商品咨询门控：这类问题FAQ答不了，直接走RAG ──
        if self._is_product_inquiry(query):
            logger.info("knowledge_qa.skip_faq_product", query=query[:30])
            return None

        # ── 第2层：BM25 + MySQL FAQ ───────────────────────
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
        """判断是否为商品咨询类问题（应跳过FAQ，走LLM通识兜底）。

        FAQ只擅长操作类问题（怎么退货/物流查询），不擅长知识类问题（XX手机参数/好不好用）。
        策略：有商品关键词 + 无业务操作词 → 跳过FAQ。
        """
        # 业务操作词（FAQ或BusinessAgent能处理的，不应跳过）
        biz_keywords = ["退货", "退款", "退换", "换货", "取消", "改地址",
                       "物流", "快递", "发货", "配送", "到哪了", "单号",
                       "订单", "下单", "运费", "包邮", "优惠券怎么",
                       "申请", "流程", "步骤", "怎么办"]
        is_biz = any(w in query for w in biz_keywords)

        # 商品咨询词（FAQ答不了，需要RAG+LLM通识）
        product_keywords = ["参数", "配置", "性能", "功能", "规格", "型号",
                           "多少钱", "价格", "好不好", "怎么样", "测评",
                           "对比", "区别", "哪个好", "推荐", "值得买",
                           "材质", "尺码", "适合", "适用", "兼容",
                           "续航", "电池", "屏幕", "拍照", "处理器", "芯片",
                           "产品", "商品", "牌子", "品牌", "是什么", "干什么",
                           "干嘛", "有什么用", "作用", "功效", "成分",
                           "介绍", "说明", "了解", "看看", "看下", "知道"]
        is_product = any(w in query for w in product_keywords)

        # 品类词（手机/电脑/化妆品等具体商品）
        category_words = ["手机", "电脑", "耳机", "手表", "平板", "笔记本",
                         "衣服", "鞋子", "化妆品", "护肤品", "防晒", "口红",
                         "包包", "家电", "食品", "饮料", "零食"]
        is_category = any(w in query for w in category_words)

        has_product_signal = is_product or is_category

        # 有商品信号 + 无业务操作词 → 跳过FAQ
        return has_product_signal and not is_biz

    # ═══════════════════════════════════════════════════════════
    # RAG 检索（v3 升级版）
    # ═══════════════════════════════════════════════════════════

    async def retrieve(
        self,
        query: str,
        decision: RouteDecision,
        history: list[ConversationHistory] | None = None,
    ) -> tuple[list, str, dict]:
        """🆕 v3 检索：支持完整管线（改写→混合检索→精排→质检）。

        :param query: 用户查询
        :param decision: 路由决策
        :param history: 对话历史（用于指代消解）
        :return: (文档列表, 策略名称, pipeline_meta)
        """
        if decision.skip_rag:
            logger.info("knowledge_qa.rag_skipped")
            return [], "跳过检索", {}

        source_filter = decision.source_filter or None
        strategy = decision.strategy
        history_text = self._format_history(history or [])

        logger.info("knowledge_qa.retrieving", strategy=strategy.value)

        # 🆕 v3 完整管线
        pipeline_result = await self.retriever.retrieve_with_pipeline(
            query=query,
            source_filter=source_filter,
            history_text=history_text,
        )

        docs = pipeline_result.get("docs", [])
        quality_passed = pipeline_result.get("quality_passed", True)
        quality_reason = pipeline_result.get("quality_reason", "")

        # 质量不通过时记录日志
        if not quality_passed:
            logger.warning("knowledge_qa.quality_failed",
                          reason=quality_reason, doc_count=len(docs))

        final_docs = docs[:self.settings.rag_candidate_m]
        meta = {
            "quality_passed": quality_passed,
            "quality_reason": quality_reason,
            "total_candidates": pipeline_result.get("total_candidates", 0),
            "rewritten": pipeline_result.get("rewritten_query"),
        }

        logger.info("knowledge_qa.retrieval_done",
                   total=pipeline_result.get("total_candidates", 0),
                   final=len(final_docs),
                   quality=quality_passed)

        return final_docs, strategy.value, meta

    # ── v2 兼容检索策略（当不使用完整管线时的回退）────────────

    async def _legacy_retrieve(
        self, query: str, decision: RouteDecision
    ) -> list:
        """v2 兼容检索：单策略检索。"""
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

    # ── 对话历史格式化 ────────────────────────────────────

    def _format_history(self, history: list[ConversationHistory]) -> str:
        if not history:
            return "（无历史对话）"
        lines = ["## 对话历史"]
        for i, turn in enumerate(history[-self.settings.max_history_turns:], 1):
            lines.append(f"用户第{i}轮: {turn.question}")
            lines.append(f"客服第{i}轮: {turn.answer}")
        lines.append("## 当前对话")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 🆕 v3 LLM 生成（完整管线）
    # ═══════════════════════════════════════════════════════════

    async def answer(
        self,
        query: str,
        perception: PerceptionResult,
        decision: RouteDecision,
        context_docs: list,
        history: list[ConversationHistory] | None = None,
        retrieval_meta: dict | None = None,
    ) -> AgentResponse:
        """🆕 v3 生成最终回复（缓存→质检→生成→后处理→写缓存）。

        :param query: 用户原始问题
        :param perception: 感知层结果
        :param decision: 路由决策
        :param context_docs: RAG检索到的文档
        :param history: 对话历史
        :param retrieval_meta: v3 检索管线元数据
        :return: AgentResponse
        """
        start_time = time.time()
        history_list = history or []
        source_filter = decision.source_filter or ""

        # ── 步骤1: Redis 缓存检查 ──────────────────────
        cached = await self.cache.get(query, source_filter)
        if cached:
            logger.info("knowledge_qa.cache_hit", query=query[:30])
            processing_time = (time.time() - start_time) * 1000
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant", content=cached,
                    sentiment_detected=perception.sentiment_label.value,
                    intent_detected=perception.intent.value,
                ),
                processing_time_ms=processing_time,
            )

        # ── 闲聊快速通道（所有意图都先过一遍模板匹配）───────
        quick = self._quick_reply_check(query)
        if quick:
            processing_time = (time.time() - start_time) * 1000
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant", content=quick,
                    sentiment_detected=perception.sentiment_label.value,
                    intent_detected=perception.intent.value,
                ),
                processing_time_ms=processing_time,
            )

        # ── 🆕 第2层快速路径：BM25 + MySQL FAQ ──────────
        # 在走完整 RAG 管线之前，先尝试 BM25 关键词匹配
        # FAQ 命中率 ~50%，延迟 ~15ms，大幅降低 RAG 调用量
        faq_answer = await self._try_fast_path(query, source_filter)
        if faq_answer:
            processing_time = (time.time() - start_time) * 1000
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant", content=faq_answer,
                    sentiment_detected=perception.sentiment_label.value,
                    intent_detected=perception.intent.value,
                ),
                processing_time_ms=processing_time,
            )

        # ── 步骤2: 质量检查 + 🆕 高危门控 ────────────
        meta = retrieval_meta or {}
        quality_ok = meta.get("quality_passed", True)
        no_docs = (not quality_ok and not context_docs)

        # 🆕 高危信息（政策/法律/赔付金额等涉及数字的承诺）：
        # 只要知识库里没搜到精确匹配的政策文档 → 强制转人工，绝不让LLM编造数字
        from backend.rag.post_processor import AnswerPostProcessor
        is_high_risk = AnswerPostProcessor.is_high_risk(query, perception.intent.value)
        if is_high_risk:
            # 检查检索到的文档是否真的包含相关政策内容（不是靠模糊匹配混过去的）
            policy_covered = any(
                any(kw in (doc.page_content if hasattr(doc,'page_content') else str(doc))
                    for kw in AnswerPostProcessor.HIGH_RISK_KEYWORDS[:6])  # 退款/退货/赔付等核心词
                for doc in context_docs
            ) if context_docs else False

            if not policy_covered:
                logger.warning("knowledge_qa.high_risk_blocked",
                              query=query[:40], intent=perception.intent.value,
                              doc_count=len(context_docs), covered=policy_covered)
                processing_time = (time.time() - start_time) * 1000
                return AgentResponse(
                    success=True,
                    message=AgentMessage(
                        role="assistant",
                        content=(
                            f"您咨询的问题涉及赔付/政策等重要信息，为确保100%准确，已为您转接人工客服。\n\n"
                            f"📞 客服电话：{self.settings.customer_service_phone}\n"
                            f"🕐 服务时间：7×24 小时\n\n"
                            f"💡 您也可以在「我的订单」页面查看本店退换货政策及售后规则。"
                        ),
                        sentiment_detected=perception.sentiment_label.value,
                        intent_detected=perception.intent.value,
                    ),
                    processing_time_ms=processing_time,
                )

        # 🔧 RAG无结果 + 非高危 → LLM通识知识兜底
        if no_docs:
            logger.info("knowledge_qa.llm_fallback", query=query[:30])

        # ── 步骤3: 构建上下文 ─────────────────────────
        context = self.retriever.format_context(context_docs)
        history_text = self._format_history(history_list)

        # 淘宝兜底
        taobao_context = ""
        # 知识问答意图 + RAG 无结果 → 尝试淘宝兜底
        if not context_docs and perception.intent == IntentCategory.KNOWLEDGE_QA:
            try:
                from backend.taobao_importer import search_products
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

        # ── 步骤4: Prompt 组装 ────────────────────────
        # 根据 query 内容判断是否需要推荐类 prompt
        is_recommend = any(w in perception.original_query for w in ["推荐", "建议", "哪个好", "选哪个", "买什么"])
        if is_recommend:
            prompt_template = EcomRAGPrompts.recommend_prompt()
            full_prompt = prompt_template.format(
                tone_instruction=decision.tone_instruction,
                history=history_text,
                question=query,
                context=full_context,
            )
        else:
            prompt_template = EcomRAGPrompts.rag_prompt()
            full_prompt = prompt_template.format(
                tone_instruction=decision.tone_instruction,
                extra_instruction=decision.dynamic_prompt_extra,
                context=full_context or "（无参考知识，请基于电商常识回答）",
                history=history_text,
                question=query,
                phone=self.settings.customer_service_phone,
            )

        # ── 步骤5: LLM 生成（温度0.1，确保稳定性）───
        try:
            llm = get_llm("qa", temperature=0.1)
            response = await with_retry_async(llm.ainvoke, full_prompt)
            content = response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            logger.error("knowledge_qa.llm_call_failed", error=str(e))
            content = self._get_fallback_message()

        # ── 步骤6: 答案后处理 ────────────────────────
        post_result = await self.post_processor.process(
            answer=content,
            context_docs=context_docs,
            query=query,
        )
        final_content = post_result["final_answer"]

        # ── 步骤7: 异步写入缓存 ──────────────────────
        if not post_result.get("hallucination_flag"):
            await self.cache.set(query, source_filter, final_content)

        processing_time = (time.time() - start_time) * 1000

        # 构建回复消息
        message = AgentMessage(
            role="assistant",
            content=final_content,
            sentiment_detected=f"{perception.sentiment.value} -> {perception.sentiment_label.value}",
            intent_detected=perception.intent.value,
        )

        # 构建对话历史更新
        history_updated = history_list[-self.settings.max_history_turns:]
        history_updated.append(ConversationHistory(
            question=query, answer=final_content,
            sentiment=perception.sentiment_label.value,
            intent=perception.intent.value,
        ))

        logger.info("knowledge_qa.answer_done",
                   time_ms=f"{processing_time:.0f}",
                   hallucination=post_result.get("hallucination_flag"))

        return AgentResponse(
            success=True,
            message=message,
            history_updated=history_updated,
            processing_time_ms=processing_time,
        )

    # ═══════════════════════════════════════════════════════════
    # 流式生成
    # ═══════════════════════════════════════════════════════════

    async def answer_stream(
        self,
        query: str,
        perception: PerceptionResult,
        decision: RouteDecision,
        context_docs: list,
        history: list[ConversationHistory] | None = None,
    ) -> AsyncGenerator[str, None]:
        """流式生成（v3简化版 — 流式跳过后处理，降低延迟）。"""
        logger.info("knowledge_qa.answer_stream_started", query=query[:30])

        # 🆕 三层快速路径：Redis → BM25 → RAG
        fast = await self._try_fast_path(query, decision.source_filter or "")
        if fast:
            yield fast
            return

        context = self.retriever.format_context(context_docs)
        history_text = self._format_history(history or [])

        rag_prompt = EcomRAGPrompts.rag_prompt()
        full_prompt = rag_prompt.format(
            tone_instruction=decision.tone_instruction,
            extra_instruction=decision.dynamic_prompt_extra,
            context=context or "（无参考知识，请基于电商常识回答）",
            history=history_text,
            question=query,
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

            # 流式结束后异步写缓存
            full_text = "".join(full_response)
            if full_text:
                await self.cache.set(query, decision.source_filter or "", full_text)

        except Exception as e:
            logger.error("knowledge_qa.stream_failed", error=str(e))
            yield self._get_fallback_message()

    # ── 🆕 闲聊处理（合并自 chitchat.py）──────────────────

    @staticmethod
    def _quick_reply_check(query: str) -> str | None:
        """匹配快速回复模板（不调LLM，毫秒级响应）。"""
        q = query.strip().lower()
        if q in ["你好", "您好", "嗨", "hi", "hello", "在吗", "在不在"]:
            return KnowledgeQAAgent.QUICK_REPLIES["greeting"]
        if any(w in q for w in ["谢谢", "感谢", "多谢", "谢谢你", "thanks", "thank"]):
            return KnowledgeQAAgent.QUICK_REPLIES["thanks"]
        if any(w in q for w in ["再见", "拜拜", "晚安", "bye", "拜"]):
            return KnowledgeQAAgent.QUICK_REPLIES["bye"]
        if any(w in q for w in ["你真棒", "太厉害了", "好聪明", "不错不错"]):
            return KnowledgeQAAgent.QUICK_REPLIES["praise"]
        if any(w in q for w in ["你好吗", "你怎么样", "你会什么"]):
            return KnowledgeQAAgent.QUICK_REPLIES["how_are_you"]
        return None

    async def _handle_chitchat(
        self,
        query: str,
        sentiment_label: SentimentLabel,
    ) -> str:
        """情感驱动的个性化闲聊（LLM轻量prompt）。"""
        tone_map = {
            SentimentLabel.HAPPY: "用户很开心，请用轻松幽默的语气互动，可以适度开玩笑",
            SentimentLabel.GRATEFUL: "用户很感激，请用温暖谦逊的语气回应",
            SentimentLabel.ANXIOUS: "用户有点焦虑，请用安抚性的语气，转移话题到轻松的方面",
            SentimentLabel.ANGRY: "用户有点生气，请先表达理解，用温柔的语气缓和气氛",
            SentimentLabel.CONFUSED: "用户困惑，请耐心引导，看看能不能帮上忙",
            SentimentLabel.DISAPPOINTED: "用户失望，请表达理解，提供积极建议",
            SentimentLabel.NEUTRAL: "用户情绪中性，保持友好热情即可",
        }
        tone = tone_map.get(sentiment_label, "保持友好热情")

        prompt = f"""你是一个友好的电商导购小助手，正在和用户闲聊。
用户刚刚说了："{query}"

语气要求：{tone}

请回复用户（简短，2-4句话）：
1. 对用户的话做出自然回应
2. 如果合适，可以自然地引导到购物话题（不要生硬推销）
回复："""

        try:
            llm = get_llm("qa", temperature=0.5)
            response = await with_retry_async(llm.ainvoke, prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            logger.error("knowledge_qa.chitchat_llm_failed", error=str(e))
            return self.QUICK_REPLIES["greeting"]

    # ── 兜底 ────────────────────────────────────────────

    def _get_fallback_message(self) -> str:
        return (
            f"非常抱歉，当前系统繁忙，暂时无法为您提供准确的回复。"
            f"建议您稍后再试，或拨打客服电话 {self.settings.customer_service_phone} "
            f"联系人工客服获取帮助。"
        )


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        agent = KnowledgeQAAgent()

        decision = RouteDecision(
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction="保持专业高效",
            source_filter="after_sales",
            skip_rag=False,
        )

        # v3 完整管线
        docs, strategy, meta = await agent.retrieve(
            "如何申请退货退款", decision,
        )
        print(f"策略: {strategy}, 文档数: {len(docs)}, 质量: {meta.get('quality_passed')}")
        if docs:
            print(f"Top-1: {docs[0].page_content[:100]}...")

        print("\nknowledge_qa.py v3 自测通过")

    asyncio.run(test())
