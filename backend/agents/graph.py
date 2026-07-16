# backend/agents/graph.py
# 🆕 v4.1 — 生产级 LangGraph 编排层：4 Agent（感知→路由→3路分发）
#
# 核心流程（LangGraph 编译图）：
#   perceive → route → resolve_context → dispatch（条件路由）
#     ├─ clarify      → END
#     ├─ escalate     → END
#     ├─ business     → handle → END
#     └─ knowledge_qa → retrieve → answer → END
#
# 设计原则：
#   - graph.py 不做决策，只做串联（决策全在 router/）
#   - 每个 Agent 独立可测，节点函数来自各 Agent 包
#   - State 在节点间流转，全程有类型

from __future__ import annotations

import json
import time as _time_module
from typing import Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.models.schemas import (
    PerceptionResult, RouteDecision, AgentResponse, AgentMessage,
    ConversationHistory, ChatEvent, IntentCategory, SentimentLabel,
    TargetAgent, UrgencyLevel, RetrievalStrategy, Sentiment,
)

# ── 🆕 v4.1：从各 Agent 包导入节点函数 ──────────────────────
from backend.agents.perception.nodes import perceive_node
from backend.agents.router.nodes import route_node
from backend.agents.knowledge_qa.nodes import (
    retrieve_node as kqa_retrieve_node,
    answer_node as kqa_answer_node,
    _get_agent as _get_kqa_agent,
)
from backend.agents.business.nodes import (
    handle_node as business_handle_node,
    _get_agent as _get_business_agent,
)
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
    shop_id: str = ""                                # 上下文解析后的企业编号
    context_docs: list = Field(default_factory=list)
    agent_response: Optional[AgentResponse] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# 条件路由函数
# ═══════════════════════════════════════════════════════════════

def dispatch_router(state: ShoppingGuideState) -> str:
    """根据 route_decision 分发到对应 Agent 节点。

    4 路分发：clarify / escalate / knowledge_qa / business
    """
    decision = state.route_decision
    if decision is None:
        return "knowledge_qa"  # 兜底

    if decision.needs_clarification:
        return "clarify"

    if decision.escalate_to_human:
        return "escalate"

    target = decision.target_agent
    if target == TargetAgent.ESCALATE:
        return "escalate"
    elif target == TargetAgent.BUSINESS:
        return "business"
    else:
        return "knowledge_qa"


def should_retrieve(state: ShoppingGuideState) -> str:
    """判断是否需要 RAG 检索。"""
    decision = state.route_decision
    if decision and decision.skip_rag:
        return "skip_retrieve"
    return "do_retrieve"


# ═══════════════════════════════════════════════════════════════
# 图节点：适配 ShoppingGuideState ↔ 各 Agent State
# ═══════════════════════════════════════════════════════════════

async def _perceive_node(state: ShoppingGuideState) -> dict:
    """感知节点 —— 适配到 PerceptionState。"""
    result = await perceive_node({"query": state.query, "perception_result": None, "error": None})
    if result.get("error"):
        return {"perception": _fallback_perception(state.query), "error": result["error"]}
    return {"perception": result["perception_result"]}


async def _route_node(state: ShoppingGuideState) -> dict:
    """路由节点 —— 适配到 RouterState。"""
    if state.perception is None:
        return {"error": "缺少感知结果", "route_decision": _fallback_decision()}
    result = await route_node({
        "perception": state.perception,
        "history": state.history,
        "route_decision": None,
        "error": None,
    })
    return {"route_decision": result.get("route_decision", _fallback_decision()),
            "error": result.get("error")}


async def _resolve_context_node(state: ShoppingGuideState) -> dict:
    """上下文解析节点：解析 shop_id（订单号/用户归属/JWT）。"""
    from backend.agents.context_resolver import get_context_resolver

    resolver = get_context_resolver()
    entities = []
    if state.perception and state.perception.entities:
        entities = [{"type": e.get("type", ""), "value": e.get("value", "")}
                    for e in state.perception.entities]

    result = await resolver.resolve(
        user_id=state.user_id,
        entities=entities,
        current_shop_id=getattr(state, "shop_id", ""),
        session_id=state.session_id,
        query=state.query,
        intent=state.perception.fine_intent if state.perception else "",
    )
    logger.info("graph.context_resolved",
               shop_id=result["shop_id"], source=result["source"],
               confidence=result["confidence"])
    return {"shop_id": result["shop_id"]}


async def _retrieve_node(state: ShoppingGuideState) -> dict:
    """RAG 检索节点 —— 适配到 KnowledgeQAState。"""
    result = await kqa_retrieve_node({
        "query": state.query, "perception": state.perception,
        "decision": state.route_decision, "history": state.history,
        "context_docs": [], "retrieval_meta": None,
        "agent_response": None, "fast_path_hit": False, "error": None,
        "shop_id": state.shop_id,
    })
    return {"context_docs": result.get("context_docs", [])}


async def _answer_node(state: ShoppingGuideState) -> dict:
    """LLM 生成节点 —— 适配到 KnowledgeQAState。"""
    result = await kqa_answer_node({
        "query": state.query, "perception": state.perception,
        "decision": state.route_decision, "history": state.history,
        "context_docs": state.context_docs, "retrieval_meta": None,
        "agent_response": None, "fast_path_hit": False, "error": None,
    })
    return {"agent_response": result.get("agent_response")}


async def _clarify_node(state: ShoppingGuideState) -> dict:
    """澄清节点 —— 直接返回 route_decision 中的澄清消息。"""
    decision = state.route_decision
    clarification = decision.clarification_question if decision else "抱歉，我没太理解您的意思，能再说详细一点吗？"
    return {"agent_response": AgentResponse(
        success=True,
        message=AgentMessage(role="assistant", content=clarification,
            intent_detected=f"clarify({state.perception.intent.value if state.perception else 'unknown'})"),
    )}


async def _escalate_node(state: ShoppingGuideState) -> dict:
    """转人工节点。"""
    settings = get_settings()
    decision = state.route_decision
    reason_text = f"\n\n📝 转接原因：{decision.escalate_reason}" if decision and decision.escalate_reason else ""

    # 创建工单
    ticket_id = await _create_support_ticket(state.perception, decision)
    ticket_text = f"\n🎫 工单编号：{ticket_id}" if ticket_id else ""

    content = (
        f"非常抱歉给您带来了不好的体验。您的问题已紧急转接人工客服。\n\n"
        f"📞 客服电话：{settings.customer_service_phone}\n"
        f"🕐 服务时间：7×24 小时"
        f"{ticket_text}{reason_text}\n\n"
        f"客服将优先处理您的诉求，感谢您的耐心等待。"
    )
    return {"agent_response": AgentResponse(
        success=True,
        message=AgentMessage(role="assistant", content=content,
            sentiment_detected=state.perception.sentiment_label.value if state.perception else "unknown",
            intent_detected=state.perception.intent.value if state.perception else "unknown"),
    )}


async def _business_node(state: ShoppingGuideState) -> dict:
    """业务处理节点 —— 适配到 BusinessState。"""
    result = await business_handle_node({
        "query": state.query, "history": state.history,
        "action": None, "params": None, "needs_confirm": False,
        "agent_response": None, "error": None,
        "user_id": state.user_id, "shop_id": state.shop_id,
    })
    return {"agent_response": result.get("agent_response")}


# ═══════════════════════════════════════════════════════════════
# 图装配（生产级 LangGraph，4 路分发）
# ═══════════════════════════════════════════════════════════════

def build_shopping_guide_graph() -> StateGraph:
    """构建生产级智能问答 LangGraph 图。

    拓扑：
        perceive → route → resolve_context → dispatch（4路条件路由）
          ├─ clarify → END
          ├─ escalate → END
          ├─ business → END
          └─ knowledge_qa → retrieve → answer → END
    """
    graph = StateGraph(ShoppingGuideState)

    # 注册所有节点
    graph.add_node("perceive", _perceive_node)
    graph.add_node("route", _route_node)
    graph.add_node("resolve_context", _resolve_context_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("answer", _answer_node)
    graph.add_node("clarify", _clarify_node)
    graph.add_node("escalate", _escalate_node)
    graph.add_node("business", _business_node)

    # 入口
    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "route")
    graph.add_edge("route", "resolve_context")

    # 4 路条件分发（从 resolve 出发）
    graph.add_conditional_edges(
        "resolve_context", dispatch_router,
        {
            "clarify": "clarify",
            "escalate": "escalate",
            "business": "business",
            "knowledge_qa": "retrieve",
        },
    )

    # knowledge_qa 内部：retrieve(条件) → answer
    graph.add_conditional_edges(
        "retrieve", should_retrieve,
        {"do_retrieve": "answer", "skip_retrieve": "answer"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    # 其他路径 → END
    graph.add_edge("clarify", END)
    graph.add_edge("escalate", END)
    graph.add_edge("business", END)

    return graph


# ═══════════════════════════════════════════════════════════════
# 生产入口：基于 LangGraph 编译图
# ═══════════════════════════════════════════════════════════════

async def run_shopping_guide(
    query: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    history: list[ConversationHistory] | None = None,
    shop_id: str = "",
) -> AgentResponse:
    """运行智能问答完整流程（非流式）— 使用编译后的 LangGraph。

    v4.1：通过 graph.ainvoke() 调用编译图，实现真正的 LangGraph 编排。
    """
    logger.info("graph.run_started", query=query[:50], user_id=user_id)

    graph = build_shopping_guide_graph()
    compiled = graph.compile()

    initial_state = ShoppingGuideState(
        query=query, user_id=user_id, session_id=session_id,
        history=history or [], shop_id=shop_id,
    )

    try:
        result_state = await compiled.ainvoke(initial_state)
        response = result_state.get("agent_response")
        if response is None:
            return AgentResponse(
                success=False,
                message=AgentMessage(role="assistant", content="系统处理异常，请稍后重试。"),
            )
        return response
    except Exception as e:
        logger.error("graph.run_failed", error=str(e), exc_info=True)
        return AgentResponse(
            success=False,
            message=AgentMessage(role="assistant",
                content=f"抱歉，处理您的请求时遇到问题。请稍后重试或拨打客服电话 {get_settings().customer_service_phone}。"),
        )


async def run_shopping_guide_stream(
    query: str,
    user_id: str = "anonymous",
    session_id: str = "default",
    history: list[ConversationHistory] | None = None,
    shop_id: str = "",
):
    """运行智能问答（流式版本）— 保持手动编排以确保 token 级流式输出。

    注：流式路径保留手动串联，因为 LangGraph astream 不直接支持
    LLM token 级别的逐字输出。感知和路由事件仍逐事件 yield 给 SSE 层。
    """
    logger.info("graph.run_stream_started", query=query[:50])

    try:
        # ── 感知层 ──────────────────────────────────────
        from backend.agents.perception.nodes import _get_agent as _get_p
        perception_agent = _get_p()
        perception = await perception_agent.perceive(query)

        yield ChatEvent(event="perception", data=json.dumps({
            "sentiment": perception.sentiment.value,
            "sentiment_label": perception.sentiment_label.value,
            "intent": perception.intent.value,
            "entities": perception.entities,
            "confidence": perception.intent_confidence,
        }, ensure_ascii=False))

        # ── 路由层 ──────────────────────────────────────
        from backend.agents.router.nodes import _get_agent as _get_r
        routing_agent = _get_r()
        decision = await routing_agent.route(perception, history)

        yield ChatEvent(event="route", data=json.dumps({
            "target_agent": decision.target_agent.value,
            "strategy": decision.strategy.value,
            "urgency": decision.urgency.value,
            "escalate": decision.escalate_to_human,
            "clarify": decision.needs_clarification,
            "skip_rag": decision.skip_rag,
        }, ensure_ascii=False))

        # ── 上下文解析（shop_id）──
        from backend.agents.context_resolver import get_context_resolver
        resolver = get_context_resolver()
        entities = [{"type": e.get("type", ""), "value": e.get("value", "")}
                    for e in (perception.entities or [])]
        ctx = await resolver.resolve(
            user_id=user_id, entities=entities,
            current_shop_id=shop_id, session_id=session_id, query=query,
            intent=perception.fine_intent if perception else "",
        )
        shop_id = ctx["shop_id"]
        logger.info("graph.stream_context_resolved",
                   shop_id=shop_id, source=ctx["source"])

        # ── 4路分发 ─────────────────────────────────────

        # 澄清
        if decision.needs_clarification:
            for char in decision.clarification_question:
                yield ChatEvent(event="token", data=char)
            yield ChatEvent(event="done", data="clarify")
            return

        target = decision.target_agent

        # 转人工
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

        # 业务处理
        if target == TargetAgent.BUSINESS:
            biz_agent = _get_business_agent()
            response = await biz_agent.handle(query=query, history=history,
                                              user_id=user_id, shop_id=shop_id)
            for char in response.message.content:
                yield ChatEvent(event="token", data=char)
            yield ChatEvent(event="done", data="business")
            return

        # 知识问答（默认）— 含快速路径 + RAG + 流式生成
        kqa_agent = _get_kqa_agent()

        if not decision.skip_rag:
            # 有 shop_id 时跳过 BM25 快速路径，直接走 Milvus 租户检索
            if not shop_id:
                fast = await kqa_agent._try_fast_path(query, decision.source_filter or "")
            else:
                fast = None
            if fast:
                for char in fast:
                    yield ChatEvent(event="token", data=char)
                yield ChatEvent(event="done", data="knowledge_qa_fast")
                return

        if decision.skip_rag:
            context_docs = []
        else:
            context_docs, _, _ = await kqa_agent.retrieve(query, decision, history, shop_id)

        async for token in kqa_agent.answer_stream(
            query=query, perception=perception,
            decision=decision, context_docs=context_docs, history=history,
        ):
            yield ChatEvent(event="token", data=token)

        yield ChatEvent(event="done", data="knowledge_qa")

    except Exception as e:
        logger.error("graph.stream_failed", error=str(e))
        yield ChatEvent(event="error", data=str(e))


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _fallback_perception(query: str) -> PerceptionResult:
    return PerceptionResult(
        original_query=query,
        sentiment=Sentiment.NEUTRAL,
        sentiment_label=SentimentLabel.NEUTRAL,
        sentiment_confidence=0.5,
        intent=IntentCategory.KNOWLEDGE_QA,
    )


def _fallback_decision() -> RouteDecision:
    return RouteDecision(
        target_agent=TargetAgent.KNOWLEDGE_QA,
        needs_clarification=False,
        urgency=UrgencyLevel.NORMAL,
        escalate_to_human=False,
        strategy=RetrievalStrategy.DIRECT,
        tone_instruction="请保持专业、友好的客服语气。",
    )


async def _create_support_ticket(
    perception: Optional[PerceptionResult], decision: Optional[RouteDecision]
) -> str:
    """创建工单并返回工单编号（表已移除，直接返回空）。"""
    return ""


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        print("=" * 60)
        print("智能问答 v4.1 — LangGraph 编译图 + 4路分发")
        print("=" * 60)

        tests = [
            ("这件衣服是什么材质？会不会起球？", "knowledge_qa"),
            ("我的快递怎么还没到？急死人了！", "business"),
            ("你好呀，今天心情真好", "knowledge_qa (chitchat)"),
            ("帮我记一下：本店退货需在7天内申请", "knowledge_qa"),
            ("我要退款！这什么垃圾商品！", "escalate"),
            ("嗯…就是那个…怎么说呢…", "clarify"),
            ("取消订单JD20240706-001，不想要了", "business"),
        ]

        for query, expected in tests:
            print(f"\n{'─'*60}")
            print(f"用户: {query}")
            print(f"预期: {expected}")
            result = await run_shopping_guide(query)
            target = result.message.intent_detected or "unknown"
            print(f"实际: {target}")
            print(f"回复: {result.message.content[:120]}...")

        print("\n✅ graph.py v4.1 编译图 自测通过")

    asyncio.run(test())
