# backend/agents/knowledge_qa/graph.py
# KnowledgeQA Agent — LangGraph 图定义

from langgraph.graph import StateGraph, END

from backend.agents.knowledge_qa.state import KnowledgeQAState
from backend.agents.knowledge_qa.nodes import (
    fast_path_node, retrieve_node, answer_node, check_fast_path,
)


def build_knowledge_qa_graph():
    """构建知识应答 Agent 的 LangGraph 状态图。

    流程：fast_path → [命中→END | 未命中→retrieve→answer→END]
    """
    builder = StateGraph(KnowledgeQAState)

    builder.add_node("fast_path", fast_path_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)

    builder.set_entry_point("fast_path")

    builder.add_conditional_edges(
        "fast_path", check_fast_path,
        {"end": END, "retrieve": "retrieve"},
    )

    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_knowledge_qa_graph()
    print("KnowledgeQA 图编译成功")
    print("节点列表：", list(graph.nodes.keys()))
