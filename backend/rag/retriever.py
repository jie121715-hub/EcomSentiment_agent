# backend/rag/retriever.py
# 🆕 v3 — 升级为完整 RAG 检索管线：
#   Query改写(4策略) → 混合检索(4路并行) → RRF融合 → Reranker精排 → 质量检查
#
# 向量库双后端：优先 Milvus（生产级），自动降级 Chroma（本地开发/离线）。
# 嵌入模型：BAAI/bge-small-zh-v1.5（可升级为 BGE-M3 以支持稀疏+多向量）

import os
import asyncio
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import RAGRetrievalError

logger = get_logger(__name__)


class EcomRetriever:
    """🆕 v3 电商知识检索器 — 完整 RAG 管线。

    管线流程（对应流程图）：
      query → Query改写(4策略并行) → 混合检索(4路并行) → RRF融合
           → Reranker精排 → 质量检查(首条>0.6?) → Top-5 文档

    向量库双后端（自动检测降级）：
    - 优先：Milvus（生产级，需 Milvus 服务运行在 localhost:19530）
    - 降级：Chroma（本地 SQLite 模式，零依赖开箱即用）
    """

    def __init__(self):
        settings = get_settings()
        self.k = settings.rag_retrieval_k
        self.relevance_threshold = settings.rag_relevance_threshold
        self._backend = None  # "milvus" | "chroma"
        self.parent_child_enabled = settings.rag_parent_child_enabled

        # 🆕 父子块：parent_id → parent_content 映射表（检索后快速取父块）
        self._parent_map: dict[str, str] = {}

        # 初始化嵌入模型（🆕 BGE-M3 本地路径: 1024维稠密+稀疏+多向量）
        import os as _os
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")

        model_path = settings.embedding_model_name
        is_local = os.path.isdir(model_path) if model_path else False

        logger.info("retriever.loading_embedding_model",
                   path=model_path, local=is_local)

        encode_kw = {"normalize_embeddings": True}
        model_kw = {"device": "cpu"}

        if is_local:
            # 本地路径：直接用文件夹路径加载
            model_kw["local_files_only"] = True
        else:
            model_kw["local_files_only"] = True
            model_kw["trust_remote_code"] = True

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kw,
            encode_kwargs=encode_kw,
        )
        logger.info("retriever.embedding_loaded", path=model_path)

        # 初始化向量库（Milvus 优先 → Chroma 降级）
        self.vector_store = None
        self._init_vector_store()

        # 🆕 v3 组件（懒加载）
        self._query_rewriter = None
        self._hybrid_searcher = None
        self._reranker = None

    # ═══════════════════════════════════════════════════════════
    # 初始化（与 v2 兼容）
    # ═══════════════════════════════════════════════════════════

    def _init_vector_store(self):
        """初始化向量库：Milvus → Chroma 自动降级。"""
        settings = get_settings()

        if self._try_init_milvus():
            self._backend = "milvus"
            logger.info("retriever.backend", backend="milvus")
            if self._milvus_collection_is_empty():
                logger.warning("retriever.empty_collection",
                    hint="请运行 python -m backend.seed_all 导入数据")
            return

        logger.info("retriever.milvus_unavailable", fallback="chroma")
        self._init_chroma()
        self._backend = "chroma"
        logger.info("retriever.backend", backend="chroma")

        if self.vector_store and self.vector_store._collection.count() == 0:
            logger.warning("retriever.empty_chroma",
                hint="请运行 python -m backend.seed_all 导入数据")

    def _try_init_milvus(self) -> bool:
        settings = get_settings()
        try:
            from pymilvus import connections, utility
            connections.connect(
                alias="ecom_agent",
                host=settings.milvus_host, port=settings.milvus_port,
                timeout=5,
            )
            collections = utility.list_collections(using="ecom_agent")
            logger.info("retriever.milvus_connected",
                host=settings.milvus_host, port=settings.milvus_port,
                existing_collections=collections,
            )
            from langchain_community.vectorstores import Milvus
            # 🆕 统一使用 ecom_knowledge_v1（不再分散到多个 collection）
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name="ecom_knowledge_v1",
                connection_args={"host": settings.milvus_host, "port": settings.milvus_port},
                auto_id=True,
            )
            return True
        except Exception as e:
            logger.warning("retriever.milvus_connect_failed", error=str(e)[:80])
            try:
                from pymilvus import connections
                connections.disconnect("ecom_agent")
            except Exception:
                pass
        return False

    def _milvus_collection_is_empty(self) -> bool:
        try:
            from pymilvus import Collection
            col = Collection("ecom_knowledge_v1")
            return col.num_entities == 0
        except Exception:
            return True

    def _init_chroma(self):
        from langchain_community.vectorstores import Chroma
        settings = get_settings()
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        try:
            self.vector_store = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="ecom_knowledge_v1",
            )
            doc_count = self.vector_store._collection.count()
            logger.info("retriever.chroma_loaded", doc_count=doc_count)
        except Exception as e:
            logger.error("retriever.chroma_init_failed", error=str(e))
            self.vector_store = None

    # ═══════════════════════════════════════════════════════════
    # 🆕 v3 组件懒加载
    # ═══════════════════════════════════════════════════════════

    @property
    def query_rewriter(self):
        if self._query_rewriter is None:
            from backend.rag.query_rewriter import QueryRewriter
            self._query_rewriter = QueryRewriter()
        return self._query_rewriter

    @property
    def hybrid_searcher(self):
        if self._hybrid_searcher is None:
            from backend.rag.hybrid_searcher import HybridSearcher
            self._hybrid_searcher = HybridSearcher(self.vector_store, self.embeddings)
            self._hybrid_searcher.detect_backend(self._backend or "chroma")
        return self._hybrid_searcher

    @property
    def reranker(self):
        if self._reranker is None:
            from backend.rag.reranker import EcomReranker
            self._reranker = EcomReranker()
        return self._reranker

    # ═══════════════════════════════════════════════════════════
    # 🆕 v3 完整检索管线
    # ═══════════════════════════════════════════════════════════

    async def retrieve_with_pipeline(
        self,
        query: str,
        source_filter: str | None = None,
        history_text: str = "",
        entities: list[dict] | None = None,
        top_n: int | None = None,
    ) -> dict:
        """🆕 v3 完整 RAG 检索管线。

        流程：
          1. Query 改写（4策略并行）
          2. 混合检索（每个改写变体并行检索 → RRF 融合）
          3. Reranker 精排
          4. 质量检查

        :return: {
            "docs": list,              # 最终文档列表
            "quality_passed": bool,    # 质量是否达标
            "quality_reason": str,     # 未达标原因
            "rewritten_query": object, # 改写结果（调试用）
            "total_candidates": int,   # 候选总数
        }
        """
        k = top_n or self.k

        if self.vector_store is None:
            raise RAGRetrievalError("向量库未初始化")

        # ── 步骤1: Query 改写 ──────────────────────────
        rewritten = await self.query_rewriter.rewrite(
            query, history_text, entities
        )

        # ── 步骤2: 混合检索（每个变体并行检索）───────
        all_candidates = []
        variants = rewritten.all_variants
        if not variants:
            variants = [query]

        # 并行检索所有变体
        tasks = [
            self.hybrid_searcher.search(v, top_n=k * 3, source_filter=source_filter)
            for v in variants[:5]  # 最多5个变体，避免过度请求
        ]
        hybrid_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in hybrid_results:
            if isinstance(result, Exception):
                logger.warning("retriever.variant_search_failed", error=str(result))
                continue
            if hasattr(result, 'docs'):
                all_candidates.extend(result.docs)

        # 去重（子块 → 提取 parent_id → 返回父块完整内容）
        seen_content = set()
        child_docs = []
        for doc in all_candidates:
            if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                continue
            key = hash(doc.page_content[:200])
            if key not in seen_content:
                seen_content.add(key)
                child_docs.append(doc)

        parent_mapped = False

        # ── 🆕 父子块映射：从子块提取 parent_id → 取父块 ──
        if self.parent_child_enabled:
            # 分离子块和父块
            children = [d for d in child_docs if d.metadata.get("chunk_type") == "child"]
            parents_in_results = [d for d in child_docs if d.metadata.get("chunk_type") == "parent"]

            if children:
                # 从子块提取 parent_id，去重
                parent_ids = list(set(
                    d.metadata.get("parent_id", "") for d in children if d.metadata.get("parent_id")
                ))

                if parent_ids and self._parent_map:
                    # 优先用内存映射（快）
                    from backend.rag.chunker import ParentChildChunker
                    chunker = ParentChildChunker()
                    parent_docs = chunker.map_children_to_parent_docs(children, self._parent_map)
                    if parent_docs:
                        child_docs = parent_docs
                        parent_mapped = True
                        logger.info("retriever.parent_mapped_via_map",
                                   children=len(children), parents=len(parent_docs))
                elif parent_ids and self._backend == "milvus":
                    # 从 Milvus 查父块
                    parent_docs = self._fetch_parents_by_ids(parent_ids)
                    if parent_docs:
                        child_docs = parent_docs
                        parent_mapped = True
                        logger.info("retriever.parent_mapped_via_milvus",
                                   children=len(children), parents=len(parent_docs))
                elif parents_in_results:
                    # 搜索结果里已经有父块，直接用
                    child_docs = parents_in_results
                    parent_mapped = True
                    logger.info("retriever.parents_in_results", parents=len(parents_in_results))
                else:
                    # 无法映射父块（旧数据无 chunk_type），直接用搜索结果
                    logger.info("retriever.no_parent_map_fallback", docs=len(child_docs))
            elif parents_in_results:
                # 只有父块没有子块 → 直接用父块
                child_docs = parents_in_results
                logger.info("retriever.parents_direct", parents=len(parents_in_results))
            else:
                # 无 chunk_type 元数据的旧数据，直接返回
                logger.info("retriever.legacy_no_chunk_type", docs=len(child_docs))

        total_candidates = len(child_docs)
        logger.info("retriever.pipeline_candidates",
                   total=total_candidates, variants=len(variants),
                   parent_mapped=parent_mapped)

        # ── 步骤3: Reranker 精排 ──────────────────────
        if child_docs:
            ranked_docs = await self.reranker.rerank(query, child_docs, top_n=k)
        else:
            ranked_docs = []

        # ── 步骤4: 质量检查 ──────────────────────────
        quality = self.reranker.check_quality(ranked_docs)

        return {
            "docs": ranked_docs,
            "quality_passed": quality["passed"],
            "quality_reason": quality.get("reason", ""),
            "rewritten_query": rewritten,
            "total_candidates": total_candidates,
            "parent_mapped": parent_mapped,
        }

    # ═══════════════════════════════════════════════════════════
    # v2 兼容接口（保持向后兼容）
    # ═══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> list:
        """基础语义检索（v2 兼容接口）。

        对于 v3 完整管线，请使用 retrieve_with_pipeline()。
        """
        if self.vector_store is None:
            raise RAGRetrievalError("向量库未初始化")

        _k = k or self.k

        try:
            if source_filter and self._backend == "milvus":
                results = self.vector_store.similarity_search(
                    query, k=_k,
                    expr=f"source like '%{source_filter}%'",
                )
            elif source_filter:
                results = self.vector_store.similarity_search(
                    query, k=_k,
                    filter={"source": {"$contains": source_filter}},
                )
            else:
                results = self.vector_store.similarity_search(query, k=_k)

            logger.info("retriever.search_done", backend=self._backend,
                       query=query[:30], results=len(results), filter=source_filter or "none")
            return results[:self.k]
        except Exception as e:
            logger.error("retriever.search_failed", query=query[:30], error=str(e))
            raise RAGRetrievalError(f"检索失败: {e}")

    def _fetch_parents_by_ids(self, parent_ids: list[str]) -> list:
        """从 Milvus/Chroma 按 parent_id 批量取父块 Document。"""
        if not parent_ids or self.vector_store is None:
            return []

        try:
            if self._backend == "milvus":
                # Milvus: 用 expr 过滤
                ids_str = ", ".join([f'"{pid}"' for pid in parent_ids])
                expr = f'parent_id in [{ids_str}]'
                from langchain_community.vectorstores import Milvus
                docs = self.vector_store.similarity_search(
                    " ", k=len(parent_ids), expr=expr,
                )
                return docs
            else:
                # Chroma: 用 metadata filter
                docs = self.vector_store.similarity_search(
                    " ", k=len(parent_ids) * 2,
                    filter={"parent_id": {"$in": parent_ids}},
                )
                seen = set()
                unique = []
                for d in docs:
                    pid = d.metadata.get("parent_id", "")
                    if pid not in seen:
                        seen.add(pid)
                        unique.append(d)
                return unique[:len(parent_ids)]
        except Exception as e:
            logger.error("retriever.fetch_parents_failed", error=str(e))
            return []

    def format_context(self, docs: list) -> str:
        """将文档列表格式化为 LLM 可用的上下文字符串。"""
        if not docs:
            return ""

        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source', '未知来源') if hasattr(doc, 'metadata') else '未知来源'
            formatted.append(f"[来源{i}: {source}]\n{content}")

        return "\n\n---\n\n".join(formatted)

    def get_fallback_message(self) -> str:
        """检索质量不足时的兜底话术。"""
        return self.reranker.get_fallback_message()


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        print("=" * 60)
        print("EcomRetriever v3 自测（Milvus优先 → Chroma降级）")
        print("=" * 60)

        retriever = EcomRetriever()
        print(f"后端: {retriever._backend}")
        print(f"知识库可用: {retriever.vector_store is not None}")

        if retriever.vector_store:
            try:
                if retriever._backend == "milvus":
                    from pymilvus import Collection
                    col = Collection("ecom_knowledge_v1")
                    print(f"文档数: {col.num_entities}")
                else:
                    print(f"文档数: {retriever.vector_store._collection.count()}")
            except Exception:
                pass

            # 🆕 v3 管线测试
            print("\n--- v3 完整管线检索 ---")
            result = await retriever.retrieve_with_pipeline(
                "如何申请退货退款", source_filter="after_sales",
            )
            print(f"候选数: {result['total_candidates']}")
            print(f"最终文档: {len(result['docs'])}")
            print(f"质量通过: {result['quality_passed']}")
            if result['docs']:
                print(f"Top-1: {result['docs'][0].page_content[:80]}...")

            # v2 兼容接口
            print("\n--- v2 兼容接口 ---")
            results = retriever.search("如何申请退货退款", k=3)
            print(f"检索到 {len(results)} 个文档")
        else:
            print("向量库未初始化，请将知识文件放入 data 目录")

        print("\nretriever.py v3 自测完成")

    asyncio.run(test())
