# backend/agents/business/state.py
# Business Agent — LangGraph 状态定义

from typing import Optional
from typing_extensions import TypedDict

from backend.models.schemas import AgentResponse, ConversationHistory


class BusinessState(TypedDict):
    """业务处理 Agent 的状态定义。"""
    # ── 输入 ──
    query: str                                  # 用户输入
    history: list[ConversationHistory]           # 对话历史
    user_id: str                                # 用户ID
    shop_id: str                                # 企业编号

    # ── 中间状态 ──
    action: Optional[str]                       # 操作类型 (logistics|order|stock|...)
    params: Optional[dict]                      # 操作参数
    needs_confirm: bool                         # 是否需要二次确认

    # ── 输出 ──
    agent_response: Optional[AgentResponse]      # 业务处理结果
    error: Optional[str]                         # 节点错误信息
