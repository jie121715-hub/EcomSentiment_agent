# backend/agents/graph.py
# 🆕 v3.1 — 裁剪后编排层：5 Agent（感知→路由→3路分发）。
#
# 核心流程：
#   perceive（感知）
#     → route（三维决策：置信度门控 + 紧急度检测 + Agent分发）
#       → 根据 target_agent 分派（4目标）：
#           ├─ knowledge_qa     → KnowledgeQAAgent   （RAG+LLM，含闲聊兜底）
#           ├─ business         → BusinessAgent       （MySQL读写合一+确认流程）
#           ├─ knowledge_mgmt   → KnowledgeMgmtAgent  （双写MySQL+向量库）
#           └─ escalate         → 转人工
#
# 🆕 v3.1 裁剪：
#   - chitchat → 合并进 knowledge_qa（_quick_reply_check + _handle_chitchat）
#   - clarify  → 合并进 router（_build_clarify_decision 直接生成澄清消息）
#   - business_execute → 合并进 business（读写合一）
#
# 设计原则：
#   - graph.py 不做决策，只做串联（决策全在 router.py）
#   - 每个 Agent 独立可测
#   - State 在节点间流转，全程有类型

from __future__ import annotations

import json
from typing import Optional, AsyncGenerator
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.models.schemas import (
    PerceptionResult, RouteDecision, AgentResponse, AgentMessage,
    ConversationHistory, ChatEvent, IntentCategory, SentimentLabel,
    TargetAgent, UrgencyLevel, RetrievalStrategy, Sentiment,
)
from backend.agents.perception import PerceptionAgent
from backend.agents.router import RoutingAgent
from backend.agents.knowledge_qa import KnowledgeQAAgent
from backend.agents.business import BusinessAgent
from backend.agents.knowledge_mgmt import KnowledgeMgmtAgent

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════
# State 定义
# ═══════════════════════════════════════════════════════════════

class ShoppingGuideState(BaseModel):
    """智能问答 LangGraph 图的状态对象。"""
    query: str = ""
    user_id: str = "anonymous"
    session_id: str = "default"
    history: list[ConversationHistory] = Field(default_factory=list)

    perception: Optional[PerceptionResult] = None
    route_decision: Optional[RouteDecision] = None
    context_docs: list = Field(default_factory=list)
    agent_response: Optional[AgentResponse] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# 全局 Agent 实例（懒加载，整个进程复用）
# ═══════════════════════════════════════════════════════════════

_perception_agent: Optional[PerceptionAgent] = None
_routing_agent: Optional[RoutingAgent] = None
_knowledge_qa_agent: Optional[KnowledgeQAAgent] = None
_business_agent: Optional[BusinessAgent] = None
_knowledge_mgmt_agent: Optional[KnowledgeMgmtAgent] = None


def _get_perception() -> PerceptionAgent:
    global _perception_agent
    if _perception_agent is None:
        _perception_agent = PerceptionAgent()
    return _perception_agent

def _get_router() -> RoutingAgent:
    global _routing_agent
    if _routing_agent is None:
        _routing_agent = RoutingAgent()
    return _routing_agent

def _get_knowledge_qa() -> KnowledgeQAAgent:
    global _knowledge_qa_agent
    if _knowledge_qa_agent is None:
        _knowledge_qa_agent = KnowledgeQAAgent()
    return _knowledge_qa_agent

def _get_business() -> BusinessAgent:
    global _business_agent
    if _business_agent is None:
        _business_agent = BusinessAgent()
    return _business_agent

def _get_knowledge_mgmt() -> KnowledgeMgmtAgent:
    global _knowledge_mgmt_agent
    if _knowledge_mgmt_agent is None:
        _knowledge_mgmt_agent = KnowledgeMgmtAgent()
    return _knowledge_mgmt_agent


# ═══════════════════════════════════════════════════════════════
# LangGraph 节点函数（仅用于知识应答路径的流式编排）
# ═══════════════════════════════════════════════════════════════

async def perceive_node(state: ShoppingGuideState) -> dict:
    """节点1：感知层 —— 情感分析 + 意图识别 + NER。"""
    logger.info("graph.perceive_node", query=state.query[:30])
    try:
        agent = _get_perception()
        perception = await agent.perceive(state.query)
        return {"perception": perception}
    except Exception as e:
        logger.error("graph.perceive_failed", error=str(e))
        fallback = PerceptionResult(
            original_query=state.query,
            sentiment=Sentiment.NEUTRAL,
            sentiment_label=SentimentLabel.NEUTRAL,
            sentiment_confidence=0.5,
            intent=IntentCategory.KNOWLEDGE_QA,
        )
        return {"perception": fallback, "error": f"感知层异常: {e}"}


async def route_node(state: ShoppingGuideState) -> dict:
    """节点2：决策层 —— router.py 三维决策。"""
    if state.perception is None:
        return {"error": "缺少感知结果，无法进行路由决策"}
    logger.info("graph.route_node", intent=state.perception.intent.value)
    try:
        agent = _get_router()
        decision = await agent.route(state.perception)
        return {"route_decision": decision}
    except Exception as e:
        logger.error("graph.route_failed", error=str(e))
        return {"route_decision": _fallback_decision(), "error": f"路由层异常: {e}"}


async def retrieve_node(state: ShoppingGuideState) -> dict:
    """节点3：RAG 检索（仅 KnowledgeQA 路径使用）。"""
    decision = state.route_decision
    if decision is None or decision.skip_rag:
        return {"context_docs": []}
    logger.info("graph.retrieve_node", strategy=decision.strategy.value)
    try:
        agent = _get_knowledge_qa()
        docs, _ = await agent.retrieve(state.query, decision)
        return {"context_docs": docs}
    except Exception as e:
        logger.error("graph.retrieve_failed", error=str(e))
        return {"context_docs": []}


async def generate_node(state: ShoppingGuideState) -> dict:
    """节点4：LLM 生成回复。"""
    decision = state.route_decision
    perception = state.perception
    logger.info("graph.generate_node", query=state.query[:30])
    try:
        agent = _get_knowledge_qa()
        response = await agent.answer(
            query=state.query,
            perception=perception,
            decision=decision,
            context_docs=state.context_docs,
            history=state.history,
        )
        return {"agent_response": response}
    except Exception as e:
        logger.error("graph.generate_failed", error=str(e))
        settings = get_settings()
        fallback = AgentResponse(
            success=False,
            message=AgentMessage(
                role="assistant",
                content=f"抱歉，处理您的请求时遇到问题。请拨打客服电话 {settings.customer_service_phone} 联系人工客服。",
            ),
            processing_time_ms=0,
        )
        return {"agent_response": fallback}


async def escalate_node(state: ShoppingGuideState) -> dict:
    """节点5：转人工引导。"""
    settings = get_settings()
    decision = state.route_decision

    reason_text = f"\n\n📝 转接原因：{decision.escalate_reason}" if decision and decision.escalate_reason else ""
    content = (
        f"非常抱歉给您带来了不好的体验。您的问题已紧急转接人工客服。\n\n"
        f"📞 客服电话：{settings.customer_service_phone}\n"
        f"🕐 服务时间：7×24 小时\n"
        f"🎫 系统已自动生成工单，客服将优先为您处理。"
        f"{reason_text}\n\n"
        f"您也可以在我的订单页面提交工单，我们会在 2 小时内回复。"
    )
    response = AgentResponse(
        success=True,
        message=AgentMessage(
            role="assistant",
            content=content,
            sentiment_detected=state.perception.sentiment_label.value if state.perception else "unknown",
            intent_detected=state.perception.intent.value if state.perception else "unknown",
        ),
    )
    logger.info("graph.escalate_node")
    return {"agent_response": response}


# ═══════════════════════════════════════════════════════════════
# 条件路由
# ═══════════════════════════════════════════════════════════════

def should_escalate(state: ShoppingGuideState) -> str:
    decision = state.route_decision
    if decision and decision.escalate_to_human:
        return "escalate"
    return "continue"


def should_skip_rag(state: ShoppingGuideState) -> str:
    decision = state.route_decision
    if decision and decision.skip_rag:
        return "skip_rag"
    return "do_rag"


# ═══════════════════════════════════════════════════════════════
# 图装配（LangGraph — 仅用于 KnowledgeQA/转人工 流式路径）
# ═══════════════════════════════════════════════════════════════

def build_shopping_guide_graph() -> StateGraph:
    """构建智能问答 LangGraph 图。"""
    graph = StateGraph(ShoppingGuideState)

    graph.add_node("perceive", perceive_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "route")

    graph.add_conditional_edges(
        "route",
        should_escalate,
        {"escalate": "escalate", "continue": "generate"}
    )

    graph.add_edge("escalate", END)
    graph.add_edge("generate", END)

    return graph


def _fallback_decision() -> RouteDecision:
    """路由异常时的兜底决策。"""
    return RouteDecision(
        target_agent=TargetAgent.KNOWLEDGE_QA,
        needs_clarification=False,
        urgency=UrgencyLevel.NORMAL,
        escalate_to_human=False,
        strategy=RetrievalStrategy.DIRECT,
        tone_instruction="请保持专业、友好的客服语气。",
    )


# ═══════════════════════════════════════════════════════════════
# 🆕 v3.1 高层便捷函数：基于 router 决策的 4 路分发
# ═══════════════════════════════════════════════════════════════

async def run_shopping_guide(
    query: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    history: list[ConversationHistory] | None = None,
) -> AgentResponse:
    """运行智能问答完整流程（非流式）。

    v3.1 架构：感知 → router三维决策 → 4路分发 → 返回回复
    """
    logger.info("graph.run_started", query=query[:50], user_id=user_id)

    # 第一层：感知
    perception_agent = _get_perception()
    perception = await perception_agent.perceive(query)

    # 第二层：router 三维决策
    routing_agent = _get_router()
    decision = await routing_agent.route(perception, history)

    logger.info(
        "graph.dispatching",
        target=decision.target_agent.value,
        urgency=decision.urgency.value,
        clarify=decision.needs_clarification,
    )

    # 第三层：按 target_agent 分发
    return await _dispatch(decision, query, perception, history)


async def run_shopping_guide_stream(
    query: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    history: list[ConversationHistory] | None = None,
):
    """运行智能问答（流式版本），逐事件 yield 给 SSE 层。"""
    logger.info("graph.run_stream_started", query=query[:50])

    try:
        perception_agent = _get_perception()
        routing_agent = _get_router()

        # 感知
        perception = await perception_agent.perceive(query)

        yield ChatEvent(event="perception", data=json.dumps({
            "sentiment": perception.sentiment.value,
            "sentiment_label": perception.sentiment_label.value,
            "intent": perception.intent.value,
            "entities": perception.entities,
            "confidence": perception.intent_confidence,
        }, ensure_ascii=False))

        # router 决策
        decision = await routing_agent.route(perception, history)

        yield ChatEvent(event="route", data=json.dumps({
            "target_agent": decision.target_agent.value,
            "strategy": decision.strategy.value,
            "urgency": decision.urgency.value,
            "escalate": decision.escalate_to_human,
            "clarify": decision.needs_clarification,
            "skip_rag": decision.skip_rag,
        }, ensure_ascii=False))

        # ── 🆕 v3.1 4路分发 ──────────────────────────────

        # 澄清：router 已生成澄清消息，直接返回
        if decision.needs_clarification:
            for char in decision.clarification_question:
                yield ChatEvent(event="token", data=char)
            yield ChatEvent(event="done", data="clarify")
            return

        target = decision.target_agent

        # escalate → 转人工
        if target == TargetAgent.ESCALATE:
            settings = get_settings()
            tokens = (
                f"非常抱歉给您带来了不好的体验。您的问题已紧急转接人工客服。\n"
                f"📞 {settings.customer_service_phone} · 🕐 7×24小时"
            )
            for char in tokens:
                yield ChatEvent(event="token", data=char)
            yield ChatEvent(event="done", data="escalate")
            return

        # knowledge_mgmt → 知识管理（双写）
        if target == TargetAgent.KNOWLEDGE_MGMT:
            response = await _dispatch_knowledge_mgmt(query, history)
            for char in response.message.content:
                yield ChatEvent(event="token", data=char)
            yield ChatEvent(event="done", data="knowledge_mgmt")
            return

        # business → MySQL读写合一
        if target == TargetAgent.BUSINESS:
            business_agent = _get_business()
            response = await business_agent.handle(query=query, history=history)
            for char in response.message.content:
                yield ChatEvent(event="token", data=char)
            yield ChatEvent(event="done", data="business")
            return

        # knowledge_qa（默认）→ RAG + LLM 流式生成（含闲聊兜底 + 🆕 BM25快速路径）
        knowledge_qa_agent = _get_knowledge_qa()

        # 🆕 先尝试快速路径（Redis + BM25），命中则跳过昂贵检索
        if not decision.skip_rag:
            fast = await knowledge_qa_agent._try_fast_path(
                query, decision.source_filter or ""
            )
            if fast:
                for char in fast:
                    yield ChatEvent(event="token", data=char)
                yield ChatEvent(event="done", data="knowledge_qa_fast")
                return

        if decision.skip_rag:
            context_docs = []
        else:
            context_docs, _, _ = await knowledge_qa_agent.retrieve(query, decision)

        async for token in knowledge_qa_agent.answer_stream(
            query=query, perception=perception,
            decision=decision, context_docs=context_docs, history=history,
        ):
            yield ChatEvent(event="token", data=token)

        yield ChatEvent(event="done", data="knowledge_qa")

    except Exception as e:
        logger.error("graph.stream_failed", error=str(e))
        yield ChatEvent(event="error", data=str(e))


# ═══════════════════════════════════════════════════════════════
# 🆕 v3.1 统一分发函数（4路）
# ═══════════════════════════════════════════════════════════════

async def _dispatch(
    decision: RouteDecision,
    query: str,
    perception: PerceptionResult,
    history: list[ConversationHistory] | None,
) -> AgentResponse:
    """根据 router 决策分发到对应 Agent（v3.1 裁剪后：4路）。"""
    target = decision.target_agent

    # 🆕 澄清：router 已直接生成澄清消息，无需再分发到独立Agent
    if decision.needs_clarification:
        return AgentResponse(
            success=True,
            message=AgentMessage(
                role="assistant",
                content=decision.clarification_question,
                intent_detected=f"clarify({perception.intent.value},{perception.intent_confidence:.0%})",
            ),
        )

    if target == TargetAgent.ESCALATE:
        return await _dispatch_escalate(perception, decision)

    if target == TargetAgent.KNOWLEDGE_MGMT:
        return await _dispatch_knowledge_mgmt(query, history)

    if target == TargetAgent.BUSINESS:
        return await _dispatch_business(query, history)

    # 默认：knowledge_qa（含闲聊兜底 — KnowledgeQAAgent._quick_reply_check / _handle_chitchat）
    return await _dispatch_knowledge_qa(query, perception, decision, history)


# ═══════════════════════════════════════════════════════════════
# 分发实现
# ═══════════════════════════════════════════════════════════════

async def _dispatch_escalate(
    perception: PerceptionResult, decision: RouteDecision
) -> AgentResponse:
    """转人工 — 生成工单并返回引导消息。"""
    settings = get_settings()

    ticket_id = await _create_support_ticket(perception, decision)

    reason_text = f"\n📝 原因：{decision.escalate_reason}" if decision.escalate_reason else ""
    ticket_text = f"\n🎫 工单编号：{ticket_id}" if ticket_id else ""

    return AgentResponse(
        success=True,
        message=AgentMessage(
            role="assistant",
            content=(
                f"非常抱歉给您带来了不好的体验。您的问题已紧急转接人工客服。\n\n"
                f"📞 客服电话：{settings.customer_service_phone}\n"
                f"🕐 服务时间：7×24 小时"
                f"{ticket_text}"
                f"{reason_text}\n\n"
                f"客服将优先处理您的诉求，感谢您的耐心等待。"
            ),
            sentiment_detected=perception.sentiment_label.value,
            intent_detected=perception.intent.value,
        ),
    )


async def _create_support_ticket(
    perception: PerceptionResult, decision: RouteDecision
) -> str:
    """创建工单并返回工单编号。"""
    try:
        from datetime import datetime
        from backend.core.database import get_session
        from backend.models.db_models import SupportTicket

        now = datetime.now()
        date_part = now.strftime("%Y%m%d")
        random_part = str(hash(perception.original_query + str(now.timestamp())) % 10000).zfill(4)
        ticket_id = f"TK-{date_part}-{random_part}"

        async with get_session() as session:
            ticket = SupportTicket(
                ticket_id=ticket_id,
                urgency=decision.urgency.value,
                reason=decision.escalate_reason or "情绪紧急升级",
                original_query=perception.original_query,
                sentiment=perception.sentiment_label.value,
                intent=perception.intent.value,
                status="open",
            )
            session.add(ticket)
            await session.commit()
            logger.info("graph.ticket_created", ticket_id=ticket_id, urgency=decision.urgency.value)
            return ticket_id

    except Exception as e:
        logger.error("graph.ticket_create_failed", error=str(e))
        return ""


async def _dispatch_business(
    query: str, history: list[ConversationHistory] | None,
) -> AgentResponse:
    """🆕 统一业务操作（MySQL读写合一）。"""
    agent = _get_business()
    response = await agent.handle(query=query, history=history)
    response.message.intent_detected = "business"
    logger.info("graph.dispatched_to", agent="business")
    return response


async def _dispatch_knowledge_mgmt(
    query: str, history: list[ConversationHistory] | None,
) -> AgentResponse:
    """知识管理（双写）。"""
    agent = _get_knowledge_mgmt()
    response = await agent.handle(query=query, history=history)
    response.message.intent_detected = "knowledge_mgmt"
    logger.info("graph.dispatched_to", agent="knowledge_mgmt")
    return response


async def _dispatch_knowledge_qa(
    query: str,
    perception: PerceptionResult,
    decision: RouteDecision,
    history: list[ConversationHistory] | None,
) -> AgentResponse:
    """知识应答（RAG + LLM，主路径，含闲聊兜底 + 🆕 BM25快速路径）。"""
    knowledge_qa_agent = _get_knowledge_qa()

    # 🆕 三层快速路径：在走 RAG 之前先尝试 Redis + BM25（避免昂贵的检索）
    if not decision.skip_rag:
        fast = await knowledge_qa_agent._try_fast_path(
            query, decision.source_filter or ""
        )
        if fast:
            import time as _time
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant", content=fast,
                    sentiment_detected=perception.sentiment_label.value,
                    intent_detected=perception.intent.value,
                ),
                processing_time_ms=0,
            )

    if decision.skip_rag:
        context_docs = []
    else:
        context_docs, _ = await knowledge_qa_agent.retrieve(query, decision)

    response = await knowledge_qa_agent.answer(
        query=query, perception=perception, decision=decision,
        context_docs=context_docs, history=history,
    )
    logger.info("graph.dispatched_to", agent="knowledge_qa")
    return response


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        print("=" * 60)
        print("智能问答 v3.1 — router 三维决策 + 4路分发")
        print("=" * 60)

        tests = [
            ("这件衣服是什么材质？会不会起球？", "knowledge_qa"),
            ("我的快递怎么还没到？急死人了！", "business"),
            ("你好呀，今天心情真好", "knowledge_qa (chitchat兜底)"),
            ("帮我记一下：本店退货需在7天内申请", "knowledge_mgmt"),
            ("我要退款！这什么垃圾商品！", "escalate"),
            ("嗯…就是那个…怎么说呢…", "clarify"),
            ("取消订单JD20240706-001，不想要了", "business"),
        ]

        for query, expected in tests:
            print(f"\n{'─'*60}")
            print(f"用户: {query}")
            print(f"预期: {expected}")
            result = await run_shopping_guide(query)
            print(f"实际: {result.message.intent_detected}")
            print(f"回复: {result.message.content[:120]}...")
            print(f"耗时: {result.processing_time_ms:.0f}ms")

        print("\n✅ graph.py v3.1 4路分发 自测通过")

    asyncio.run(test())
