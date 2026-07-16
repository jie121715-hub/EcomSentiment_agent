# backend/agents/__init__.py
# 云答智能客服系统 - 多Agent体系
#
# v4.1 Agent 清单（4 Agent）：
#   perception       - 感知层：情感+意图+NER（BERT/transformers）
#   router           - 决策大脑：三维决策 + 澄清反问
#   knowledge_qa     - 知识应答 + 闲聊兜底：RAG+LLM
#   business         - 业务读写合一：MySQL + 确认流程
#
# 设计：使用惰性导入，独立脚本可按需引入单个Agent，不强制安装全部依赖。

__all__ = [
    "PerceptionAgent", "RoutingAgent", "KnowledgeQAAgent",
    "BusinessAgent",
    "build_shopping_guide_graph", "ShoppingGuideState",
    "run_shopping_guide", "run_shopping_guide_stream",
]


def __getattr__(name):
    """惰性导入：只在实际访问时才加载对应模块，避免 seed/sync 脚本需要全部依赖。"""
    _IMPORTS = {
        "PerceptionAgent":          ("backend.agents.perception", "PerceptionAgent"),
        "RoutingAgent":             ("backend.agents.router", "RoutingAgent"),
        "KnowledgeQAAgent":         ("backend.agents.knowledge_qa", "KnowledgeQAAgent"),
        "BusinessAgent":            ("backend.agents.business", "BusinessAgent"),
        "build_shopping_guide_graph": ("backend.agents.graph", "build_shopping_guide_graph"),
        "ShoppingGuideState":       ("backend.agents.graph", "ShoppingGuideState"),
        "run_shopping_guide":       ("backend.agents.graph", "run_shopping_guide"),
        "run_shopping_guide_stream": ("backend.agents.graph", "run_shopping_guide_stream"),
    }

    if name in _IMPORTS:
        mod_path, attr = _IMPORTS[name]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
