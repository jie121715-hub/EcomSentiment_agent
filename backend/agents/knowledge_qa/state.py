# backend/agents/knowledge_qa/state.py
# KnowledgeQA Agent — LangGraph 状态定义

from typing import Optional
from typing_extensions import TypedDict

from backend.models.schemas import (
    PerceptionResult, RouteDecision, AgentResponse, ConversationHistory,
)


class KnowledgeQAState(TypedDict):
    """知识应答 Agent 的状态定义。"""
    # ── 输入 ──
    query: str                                  # 用户原始问题
    perception: Optional[PerceptionResult]       # 感知层结果
    decision: Optional[RouteDecision]            # 路由决策
    history: list[ConversationHistory]           # 对话历史

    # ── 中间状态 ──
    context_docs: list                          # RAG 检索到的文档
    retrieval_meta: Optional[dict]               # 检索管线元数据
    fast_path_hit: bool                         # 快速路径是否命中

    # ── 输出 ──
    agent_response: Optional[AgentResponse]      # 最终回复
    error: Optional[str]                         # 节点错误信息
