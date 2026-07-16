# backend/agents/business/graph.py
# Business Agent — LangGraph 图定义

from langgraph.graph import StateGraph, END

from backend.agents.business.state import BusinessState
from backend.agents.business.nodes import handle_node


def build_business_graph():
    """构建业务处理 Agent 的 LangGraph 状态图。"""
    builder = StateGraph(BusinessState)
    builder.add_node("handle", handle_node)
    builder.set_entry_point("handle")
    builder.add_edge("handle", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_business_graph()
    print("Business 图编译成功")
    print("节点列表：", list(graph.nodes.keys()))
