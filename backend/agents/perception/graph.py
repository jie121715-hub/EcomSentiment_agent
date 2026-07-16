# backend/agents/perception/graph.py
# Perception Agent — LangGraph 图定义

from langgraph.graph import StateGraph, END

from backend.agents.perception.state import PerceptionState
from backend.agents.perception.nodes import perceive_node


def build_perception_graph():
    """构建感知层 Agent 的 LangGraph 状态图。

    单节点图：入口 → perceive_node → 结束。
    可用于独立测试感知层，也可作为父图的子图嵌入。

    Returns:
        编译后的 CompiledGraph。
    """
    builder = StateGraph(PerceptionState)

    builder.add_node("perceive", perceive_node)
    builder.set_entry_point("perceive")
    builder.add_edge("perceive", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_perception_graph()
    print("Perception 图编译成功")
    print("节点列表：", list(graph.nodes.keys()))
