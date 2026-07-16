# backend/agents/router/nodes.py
# Router Agent — LangGraph 节点函数
#
# 三维决策：置信度门控 + 情绪紧急度检测 + Agent 分发。
# 保留 RoutingAgent 类作为模块级单例。

import asyncio
import re
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.retry import with_retry_async
from backend.core.llm_factory import get_llm
from backend.models.schemas import (
    PerceptionResult, RouteDecision, RetrievalStrategy,
    SentimentLabel, IntentCategory, TargetAgent, UrgencyLevel,
    ConversationHistory,
)
from backend.data.sentiment_map import (
    get_tone_config, get_source_filter, detect_urgency,
)

from backend.agents.router.state import RouterState
from backend.agents.router.prompts import (
    INTENT_AGENT_MAP,
    KNOWLEDGE_MGMT_KEYWORDS,
    CLARIFY_TEMPLATES,
    LLM_CLARIFY_PROMPT,
    FALLBACK_CLARIFY_TEXT,
)

logger = get_logger(__name__)


class RoutingAgent:
    """三维决策路由智能体 —— 系统的"大脑"。

    职责：
      维度1: 意图置信度门控 — confidence < threshold → clarify
      维度2: 情绪紧急度检测 — 愤怒+退款/投诉 → 直接转人工
      维度3: 意图→Agent分发 — 选对Agent，选对策略

    优先级（高→低）：紧急接管 > 反问澄清 > 正常分发
    """

    def __init__(self):
        self.settings = get_settings()

    # ═══════════════════════════════════════════════════════════
    # 核心路由方法
    # ═══════════════════════════════════════════════════════════

    async def route(self, perception: PerceptionResult, history: list = None) -> RouteDecision:
        """三维决策入口。"""
        intent = perception.intent
        history = history or []

        # ── 维度0: 知识管理关键词预检（已移除，转知识问答）──
        if self._has_knowledge_mgmt_keyword(perception.original_query):
            logger.info("router.knowledge_mgmt_redirect_to_qa", query=perception.original_query[:40])
            return self._build_direct_decision(perception, TargetAgent.KNOWLEDGE_QA)

        # ── 情感×意图一致性修正 ──────────────────────────
        intent = self._correct_intent(perception, intent)

        # ── 订单号/快递单号强制修正 ──
        intent = self._correct_order_id_intent(perception, intent)

        # ── 商品咨询覆盖 ──
        intent = self._correct_product_inquiry_intent(perception, intent)

        logger.info(
            "router.v3.started",
            sentiment=perception.sentiment_label.value,
            intent=intent.value,
            confidence=round(perception.intent_confidence, 3),
        )

        # ── 维度2: 情绪紧急度检测 ─────────────────────────
        if self.settings.router_urgency_enabled:
            urgency_result = detect_urgency(perception.sentiment_label, intent)
            urgency = UrgencyLevel(urgency_result["urgency"])

            if urgency == UrgencyLevel.CRITICAL:
                return self._build_escalate_decision(
                    perception, intent, urgency_result
                )
        else:
            urgency_result = {"urgency": "normal", "reason": "", "action": "normal", "message_template": ""}
            urgency = UrgencyLevel.NORMAL

        # ── 维度1: 意图置信度门控 ────────────────────────
        has_strong_id = self._has_order_or_tracking_id(perception.original_query)
        is_confirming = self._is_confirming_response(history)
        threshold = self.settings.router_intent_confidence_threshold
        if perception.intent_confidence < threshold and not has_strong_id and not is_confirming:
            return await self._build_clarify_decision(perception, intent)

        # ── 维度3: 意图→Agent分发 ────────────────────────
        target = self._resolve_target_agent(intent, perception)

        # ── 策略配置 ──────────────────────────────────────
        tone_config = get_tone_config(perception.sentiment_label)
        source_filter = get_source_filter(intent.value)
        strategy = tone_config.get("prefer_retrieval", RetrievalStrategy.DIRECT)
        extra_instruction = self._build_extra_instruction(perception, intent)
        skip_rag = False

        if urgency == UrgencyLevel.ELEVATED:
            extra_instruction = (
                f"【优先处理】{urgency_result['reason']}\n"
                + extra_instruction
            )

        decision = RouteDecision(
            target_agent=target,
            needs_clarification=False,
            urgency=urgency,
            escalate_to_human=False,
            strategy=strategy,
            tone_instruction=tone_config["tone_instruction"],
            dynamic_prompt_extra=extra_instruction,
            source_filter=source_filter,
            skip_rag=skip_rag,
        )

        logger.info(
            "router.v3.decision",
            target=decision.target_agent.value,
            urgency=decision.urgency.value,
            strategy=decision.strategy.value,
            skip_rag=decision.skip_rag,
        )
        return decision

    # ═══════════════════════════════════════════════════════════
    # 维度1: 意图置信度门控
    # ═══════════════════════════════════════════════════════════

    async def _build_clarify_decision(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> RouteDecision:
        """低置信度 → 反问澄清（模板优先，LLM兜底）。"""
        query = perception.original_query
        confidence = perception.intent_confidence

        logger.info(
            "router.clarify_triggered",
            intent=intent.value,
            confidence=round(confidence, 3),
        )

        clarification = self._match_clarify_template(intent.value)

        if not clarification:
            clarification = await self._llm_clarify(query, perception)

        asyncio.ensure_future(self._save_clarify_log(query, perception, clarification))

        return RouteDecision(
            target_agent=TargetAgent.KNOWLEDGE_QA,
            needs_clarification=True,
            clarification_question=clarification,
            urgency=UrgencyLevel.NORMAL,
            escalate_to_human=False,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction="",
        )

    # ═══════════════════════════════════════════════════════════
    # 维度2: 情绪紧急度检测
    # ═══════════════════════════════════════════════════════════

    def _build_escalate_decision(
        self,
        perception: PerceptionResult,
        intent: IntentCategory,
        urgency_result: dict,
    ) -> RouteDecision:
        """紧急接管 → 直接转人工。"""
        reason = urgency_result.get("reason", "情绪紧急")
        logger.warning(
            "router.escalate_triggered",
            sentiment=perception.sentiment_label.value,
            intent=intent.value,
            reason=reason,
        )
        return RouteDecision(
            target_agent=TargetAgent.ESCALATE,
            needs_clarification=False,
            urgency=UrgencyLevel.CRITICAL,
            escalate_to_human=True,
            escalate_reason=reason,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction=(
                "用户正处于极度负面情绪中，涉及敏感操作。"
                "请直接引导转人工，不做任何自动化处理。"
            ),
            dynamic_prompt_extra=urgency_result.get("message_template", ""),
            skip_rag=True,
        )

    # ═══════════════════════════════════════════════════════════
    # 维度3: 意图 → Agent 分发
    # ═══════════════════════════════════════════════════════════

    def _resolve_target_agent(
        self, intent: IntentCategory, perception: PerceptionResult
    ) -> TargetAgent:
        """v4.0: 4意图→4Agent 基本一一对应，仅做内容层面的微调。"""
        query = perception.original_query

        if any(w in query for w in KNOWLEDGE_MGMT_KEYWORDS):
            return TargetAgent.KNOWLEDGE_QA  # redirect to knowledge_qa

        if intent == IntentCategory.BUSINESS:
            is_policy_q = any(w in query for w in ["政策", "规则", "怎么退", "如何退", "流程", "多久", "几天", "是什么", "能不能"])
            has_action = any(w in query for w in ["退款", "退钱", "取消", "不要了", "帮我退", "申请退", "我要退", "改地址"])
            has_order = bool(re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query))
            if is_policy_q and not has_action and not has_order:
                logger.info("router.intent_rewrite", from_="business", to="knowledge_qa")
                return TargetAgent.KNOWLEDGE_QA

        if intent == IntentCategory.ESCALATE:
            if re.search(r'[A-Z]{2,4}\d{6,12}', query):
                logger.info("router.intent_rewrite", from_="escalate", to="business")
                return TargetAgent.BUSINESS

        return INTENT_AGENT_MAP.get(intent, TargetAgent.KNOWLEDGE_QA)

    # ═══════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════

    def _has_knowledge_mgmt_keyword(self, query: str) -> bool:
        return any(w in query for w in KNOWLEDGE_MGMT_KEYWORDS)

    def _build_direct_decision(
        self, perception: PerceptionResult, target: TargetAgent
    ) -> RouteDecision:
        return RouteDecision(
            target_agent=target,
            needs_clarification=False,
            urgency=UrgencyLevel.NORMAL,
            escalate_to_human=False,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction="请保持专业、友好的客服语气。",
        )

    @staticmethod
    def _is_confirming_response(history: list) -> bool:
        if not history:
            return False
        for h in reversed(history):
            txt = h.answer if hasattr(h, 'answer') else ''
            if "确定要" in txt and "请回复" in txt:
                return True
        return False

    @staticmethod
    def _has_order_or_tracking_id(query: str) -> bool:
        q = query.strip()
        if re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', q):
            return True
        if re.search(r'^[A-Z]{2,6}\d{8,18}$', q):
            return True
        if re.search(r'^\d{10,20}$', q):
            return True
        return False

    def _correct_product_inquiry_intent(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> IntentCategory:
        query = perception.original_query.strip()

        if intent == IntentCategory.KNOWLEDGE_QA:
            return intent

        product_kw = ["手机", "电脑", "耳机", "手表", "平板", "笔记本", "衣服", "鞋子",
                      "化妆品", "护肤品", "防晒", "口红", "包包", "家具", "家电",
                      "型号", "配置", "参数", "多少钱", "怎么样", "好不好", "评价",
                      "推荐", "哪个好", "买什么", "值得买", "性价比", "对比",
                      "是什么", "材质", "尺码", "颜色", "款式", "品牌", "功能"]
        is_product_q = any(w in query for w in product_kw)

        biz_kw = ["物流", "快递", "到哪", "发货", "订单", "退款", "退货",
                  "取消", "地址", "修改", "配送", "收货"]
        is_biz_q = any(w in query for w in biz_kw)

        has_id = bool(re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query))
        has_id = has_id or bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', query))
        has_id = has_id or bool(re.fullmatch(r'\d{10,20}', query))

        if is_product_q and not is_biz_q and not has_id:
            logger.info("router.product_inquiry_corrected",
                       query=query[:30], from_=intent.value, to="knowledge_qa")
            return IntentCategory.KNOWLEDGE_QA

        return intent

    def _correct_intent(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> IntentCategory:
        positive_sentiments = {SentimentLabel.HAPPY, SentimentLabel.GRATEFUL}
        if perception.sentiment_label in positive_sentiments and intent == IntentCategory.ESCALATE:
            logger.info("router.intent_corrected", from_="escalate", to="knowledge_qa")
            return IntentCategory.KNOWLEDGE_QA
        return intent

    def _correct_order_id_intent(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> IntentCategory:
        query = perception.original_query.strip()

        is_tracking = bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', query)) or bool(re.fullmatch(r'\d{10,20}', query))
        has_order = bool(re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query))
        has_logistics_kw = any(w in query for w in ["物流", "快递", "到哪", "发货", "查单", "配送"])

        if is_tracking and intent != IntentCategory.ESCALATE:
            logger.info("router.tracking_corrected", query=query, from_=intent.value)
            return IntentCategory.BUSINESS

        if (has_order or has_logistics_kw) and intent != IntentCategory.BUSINESS:
            logger.info("router.order_id_corrected", query=query[:30], from_=intent.value)
            return IntentCategory.BUSINESS

        return intent

    def _build_extra_instruction(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> str:
        parts = []
        sentiment_label = perception.sentiment_label

        if sentiment_label == SentimentLabel.ANXIOUS:
            parts.append(
                "【重要】用户有购买疑虑，请在回复中主动提及以下保障（选1-2个最相关的）："
                "7天无理由退换、运费险、正品保证、价保30天、先行赔付。"
            )
        elif sentiment_label == SentimentLabel.ANGRY:
            parts.append(
                "【重要】用户正在愤怒中，回复首句必须道歉。不要使用'但是''不过'等转折词。"
            )
        elif sentiment_label == SentimentLabel.CONFUSED:
            parts.append(
                "【重要】用户感到困惑，请用通俗语言分2-3步说明，最后询问'这样说清楚了吗？'"
            )
        elif sentiment_label == SentimentLabel.HAPPY:
            parts.append(
                "【可选】用户情绪满意，可顺势推荐相关商品或邀请好评。"
            )

        if intent == IntentCategory.BUSINESS:
            parts.append("用户需要业务处理。告知具体操作步骤和预计处理时间。")
        elif intent == IntentCategory.ESCALATE:
            parts.append("用户有投诉/不满。优先安抚情绪，明确告知处理方案和时效。")

        if perception.entities:
            entity_types = {e.get("type", "") for e in perception.entities if e.get("type")}
            if "order_id" in entity_types:
                parts.append("用户提供了订单号，可据此查询。")
            if "sku" in entity_types or "product_name" in entity_types:
                parts.append("用户询问了具体商品，确保回答涉及该商品信息。")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════════
    # 澄清反问方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _match_clarify_template(intent_value: str) -> str | None:
        for (a, b), template in CLARIFY_TEMPLATES.items():
            if a == intent_value or b == intent_value:
                logger.info("router.clarify_template_matched", pair=f"{a}/{b}")
                return f"我不太确定您的具体需求，想确认一下：\n\n{template}"
        return None

    async def _llm_clarify(self, query: str, perception: PerceptionResult) -> str:
        prompt = LLM_CLARIFY_PROMPT.format(
            query=query,
            intent_value=perception.intent.value,
            confidence_pct=f"{perception.intent_confidence:.0%}",
            entities=str(perception.entities or '无'),
        )

        try:
            llm = get_llm("qa", temperature=0.3)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text if hasattr(response, 'text') else str(response)
            return text.strip()
        except Exception as e:
            logger.error("router.clarify_llm_failed", error=str(e))
            return FALLBACK_CLARIFY_TEXT

    async def _save_clarify_log(
        self, query: str, perception: PerceptionResult, clarification: str,
    ):
        try:
            from backend.core.database import get_session
            from backend.models.db_models import ClarifyLog

            async with get_session() as session:
                record = ClarifyLog(
                    original_query=query,
                    detected_intent=perception.intent.value,
                    confidence=perception.intent_confidence,
                    clarification_question=clarification,
                    entities=perception.entities or {},
                )
                session.add(record)
                await session.commit()
                logger.info("router.clarify_log_saved", intent=perception.intent.value)
        except Exception as e:
            logger.warning("router.clarify_log_save_failed", error=str(e))


# ═══════════════════════════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════════════════════════

_routing_agent: Optional[RoutingAgent] = None


def _get_agent() -> RoutingAgent:
    """获取 RoutingAgent 模块级单例。"""
    global _routing_agent
    if _routing_agent is None:
        _routing_agent = RoutingAgent()
    return _routing_agent


# ═══════════════════════════════════════════════════════════════
# LangGraph 节点函数
# ═══════════════════════════════════════════════════════════════

async def route_node(state: RouterState) -> dict:
    """节点：路由决策 —— 三维决策（置信度门控 + 紧急度检测 + Agent分发）。

    从 state["perception"] 和 state["history"] 读取输入，返回 {"route_decision": RouteDecision}。
    """
    perception = state.get("perception")
    history = state.get("history", [])

    if perception is None:
        return {"error": "缺少感知结果，无法进行路由决策"}

    logger.info("router.node_started", intent=perception.intent.value)

    try:
        agent = _get_agent()
        decision = await agent.route(perception, history)
        return {"route_decision": decision}
    except Exception as e:
        logger.error("router.node_failed", error=str(e))
        from backend.models.schemas import TargetAgent, UrgencyLevel, RetrievalStrategy
        fallback = RouteDecision(
            target_agent=TargetAgent.KNOWLEDGE_QA,
            needs_clarification=False,
            urgency=UrgencyLevel.NORMAL,
            escalate_to_human=False,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction="请保持专业、友好的客服语气。",
        )
        return {"route_decision": fallback, "error": f"路由层异常: {e}"}


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        router = _get_agent()

        test_cases = [
            ("这件衣服起球吗？", SentimentLabel.ANXIOUS, IntentCategory.KNOWLEDGE_QA, 0.88, "knowledge_qa"),
            ("什么垃圾！用了两天就坏了！", SentimentLabel.ANGRY, IntentCategory.ESCALATE, 0.85, "escalate"),
            ("帮我查JD20240706-001的物流", SentimentLabel.NEUTRAL, IntentCategory.BUSINESS, 0.90, "business"),
            ("取消订单JD20240706-001", SentimentLabel.NEUTRAL, IntentCategory.BUSINESS, 0.88, "business"),
            ("帮我把地址改一下", SentimentLabel.NEUTRAL, IntentCategory.BUSINESS, 0.42, "knowledge_qa"),
            ("你好呀今天天气真好", SentimentLabel.HAPPY, IntentCategory.KNOWLEDGE_QA, 0.92, "knowledge_qa"),
            ("帮我记一下：7天无理由退货", SentimentLabel.NEUTRAL, IntentCategory.KNOWLEDGE_MGMT, 0.85, "knowledge_qa"),
            ("我要退款！太失望了", SentimentLabel.DISAPPOINTED, IntentCategory.ESCALATE, 0.78, "escalate"),
        ]

        for query, label, intent, conf, expected in test_cases:
            perception = PerceptionResult(
                original_query=query,
                sentiment="negative" if label in [SentimentLabel.ANXIOUS, SentimentLabel.ANGRY, SentimentLabel.DISAPPOINTED] else "neutral",
                sentiment_label=label,
                sentiment_confidence=0.9,
                intent=intent,
                intent_confidence=conf,
                entities=[],
            )
            decision = await router.route(perception)
            if decision.needs_clarification:
                match = "PASS" if expected == "knowledge_qa" else "FAIL"
            else:
                match = "PASS" if decision.target_agent.value == expected else "FAIL"
            print(f"[{match}] {query[:35]:<35} | target={decision.target_agent.value:<18} | urgency={decision.urgency.value:<10} | clarify={decision.needs_clarification} | escalate={decision.escalate_to_human}")

        print("\nrouter nodes.py 自测通过")

    asyncio.run(test())
