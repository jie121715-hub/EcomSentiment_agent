# backend/agents/router/graph.py
# Router Agent — LangGraph 图定义

from langgraph.graph import StateGraph, END

from backend.agents.router.state import RouterState
from backend.agents.router.nodes import route_node


def build_router_graph():
    """构建路由决策 Agent 的 LangGraph 状态图。

    单节点图：入口 → route_node → 结束。
    可用于独立测试路由决策，也可作为父图的子图嵌入。

    Returns:
        编译后的 CompiledGraph。
    """
    builder = StateGraph(RouterState)

    builder.add_node("route", route_node)
    builder.set_entry_point("route")
    builder.add_edge("route", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_router_graph()
    print("Router 图编译成功")
    print("节点列表：", list(graph.nodes.keys()))
