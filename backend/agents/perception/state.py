# backend/agents/perception/state.py
# Perception Agent — LangGraph 状态定义

from typing import Optional
from typing_extensions import TypedDict

from backend.models.schemas import PerceptionResult


class PerceptionState(TypedDict):
    """感知层 Agent 的状态定义。

    节点通过读写此 State 进行数据传递。
    """
    # ── 输入 ──
    query: str                        # 用户原始输入

    # ── 输出 ──
    perception_result: Optional[PerceptionResult]  # 感知分析完整结果
    error: Optional[str]              # 节点错误信息
