# backend/agents/router/state.py
# Router Agent — LangGraph 状态定义

from typing import Optional
from typing_extensions import TypedDict

from backend.models.schemas import (
    PerceptionResult, RouteDecision, ConversationHistory,
)


class RouterState(TypedDict):
    """路由决策 Agent 的状态定义。"""
    # ── 输入 ──
    perception: PerceptionResult                # 感知层输出
    history: list[ConversationHistory]          # 对话历史

    # ── 输出 ──
    route_decision: Optional[RouteDecision]     # 路由决策结果
    error: Optional[str]                        # 节点错误信息
