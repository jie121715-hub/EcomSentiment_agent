# backend/rag/__init__.py
# 惰性导入：避免 seed/sync 脚本需要 langchain 等重依赖。
#
# 🆕 v3 模块清单：
#   ParentChildChunker — 父子块切分器（子块检索→父块返回）
#   EcomRAGPrompts    — Prompt 模板集
#   EcomRetriever     — 检索器（v3 完整管线：改写→混合检索→精排→质检）
#   QueryRewriter     — Query 改写器（4策略并行）
#   HybridSearcher    — 混合检索器（4路并行 + RRF 融合）
#   EcomReranker      — BGE-Reranker 精排器（Cross-Encoder）
#   AnswerPostProcessor — 答案后处理器（幻觉检测/敏感词/格式化）
#   RAGCache          — Redis 缓存层

def __getattr__(name):
    _IMPORTS = {
        "ParentChildChunker":   ("backend.rag.chunker", "ParentChildChunker"),
        "EcomRAGPrompts":       ("backend.rag.prompts", "EcomRAGPrompts"),
        "EcomRetriever":        ("backend.rag.retriever", "EcomRetriever"),
        "QueryRewriter":        ("backend.rag.query_rewriter", "QueryRewriter"),
        "HybridSearcher":       ("backend.rag.hybrid_searcher", "HybridSearcher"),
        "EcomReranker":         ("backend.rag.reranker", "EcomReranker"),
        "AnswerPostProcessor":  ("backend.rag.post_processor", "AnswerPostProcessor"),
        "RAGCache":             ("backend.rag.cache", "RAGCache"),
    }
    if name in _IMPORTS:
        mod_path, attr = _IMPORTS[name]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
