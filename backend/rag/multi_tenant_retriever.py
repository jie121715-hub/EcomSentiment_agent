# backend/rag/multi_tenant_retriever.py
# 多租户 RAG 检索器 — 对接 eco_rag.policies + eco_rag.products
# 支持：稠密+稀疏混合检索、租户隔离(shop_id)、父子块映射

import os
import time
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class MultiTenantRetriever:
    """多租户检索器：根据 shop_id 限定检索范围，企业间天然隔离。

    检索流程：
      query → BGE-M3编码(稠密+稀疏)
           → 混合检索(policies + products 双Collection)
           → RRF融合
           → 子块 → 父块映射
           → 返回完整上下文

    使用方式：
        retriever = MultiTenantRetriever()
        docs = retriever.search("退货政策", shop_id="a001")
    """

    def __init__(self):
        settings = get_settings()
        self.db_name = "eco_rag"
        self.collections = ["policies", "products"]
        self.dense_field = "dense_vector"
        self.sparse_field = "sparse_vector"
        self.dense_top_k = settings.rag_dense_top_n       # 50
        self.sparse_top_k = settings.rag_sparse_top_n     # 30
        self.hybrid_top_k = settings.rag_hybrid_top_n     # 40
        self.final_k = settings.rag_retrieval_k           # 5
        self.rrf_k = settings.rag_rrf_k                   # 60

        # 懒加载
        self._model = None
        self._client = None

    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            settings = get_settings()
            self._model = SentenceTransformer(
                settings.embedding_model_name, device="cpu",
            )
            logger.info("mt_retriever.model_loaded")
        return self._model

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            self._client = MilvusClient(uri=uri, db_name=self.db_name, timeout=15)
            logger.info("mt_retriever.milvus_connected", uri=uri, db=self.db_name)
        return self._client

    # ═══════════════════════════════════════════════════════════
    # 主检索接口
    # ═══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        shop_id: str,
        collections: list[str] | None = None,
        top_k: int | None = None,
        return_parents: bool = True,
    ) -> list[dict]:
        """多租户混合检索。

        :param query: 用户查询
        :param shop_id: 企业编号，用于租户隔离
        :param collections: 检索哪些Collection，默认全部
        :param top_k: 返回文档数
        :param return_parents: 是否将子块映射为父块
        :return: [{"content": str, "source": str, "score": float, ...}, ...]
        """
        k = top_k or self.final_k
        cols = collections or self.collections
        shop_filter = f'shop_id == "{shop_id}"' if shop_id else ""

        # 1. 编码
        dense, sparse_dict = self._encode(query)

        # 2. 对每个Collection做稠密+稀疏双路检索
        all_candidates = []
        for col_name in cols:
            candidates = self._hybrid_search(
                col_name, dense, sparse_dict, shop_filter,
            )
            all_candidates.extend(candidates)

        if not all_candidates:
            return []

        # 3. RRF 融合去重排序
        merged = self._rrf_fusion(all_candidates, k=self.hybrid_top_k)

        # 4. 确保每个Collection至少有1条（避免全是policies没products）
        per_col = {}
        for d in merged:
            col = d.get("collection", "")
            if col not in per_col:
                per_col[col] = d
        balanced = list(per_col.values())
        for d in merged:
            if d not in balanced:
                balanced.append(d)

        # 5. 子块 → 父块映射
        if return_parents:
            balanced = self._map_to_parents(cols, balanced)

        # 6. 返回 Top-K
        return balanced[:k]

    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════

    def _encode(self, query: str) -> tuple:
        """BGE-M3 编码，返回 (dense_list, sparse_dict)。"""
        vec = self.model.encode([query], normalize_embeddings=True)
        dense = vec[0].tolist()
        sparse = {}  # SentenceTransformer 不含稀疏，后续可升级
        return dense, sparse

    def _hybrid_search(
        self, col_name: str, dense: list, sparse: dict, expr: str,
    ) -> list[dict]:
        """单 Collection 稠密+稀疏双路检索。"""
        candidates = []

        # 稠密检索
        try:
            dense_results = self.client.search(
                collection_name=col_name,
                data=[dense],
                limit=self.dense_top_k,
                anns_field=self.dense_field,
                search_params={"metric_type": "IP", "params": {"nprobe": 16}},
                filter=expr,
                output_fields=["id", "doc_id", "content", "chunk_type", "parent_id", "category"],
            )
            if dense_results and dense_results[0]:
                for hit in dense_results[0]:
                    candidates.append({
                        "id": hit["entity"].get("id", ""),
                        "doc_id": hit["entity"].get("doc_id", ""),
                        "content": hit["entity"].get("content", ""),
                        "chunk_type": hit["entity"].get("chunk_type", ""),
                        "parent_id": hit["entity"].get("parent_id", ""),
                        "category": hit["entity"].get("category", ""),
                        "collection": col_name,
                        "score": hit.get("distance", 0),
                        "source": "dense",
                    })
        except Exception as e:
            logger.warning("mt_retriever.dense_search_failed",
                          col=col_name, error=str(e)[:100])

        # 稀疏检索
        if sparse:
            try:
                sparse_results = self.client.search(
                    collection_name=col_name,
                    data=[sparse],
                    limit=self.sparse_top_k,
                    anns_field=self.sparse_field,
                    search_params={"metric_type": "IP"},
                    filter=expr,
                    output_fields=["id", "doc_id", "content", "chunk_type", "parent_id", "category"],
                )
                if sparse_results and sparse_results[0]:
                    for hit in sparse_results[0]:
                        candidates.append({
                            "id": hit["entity"].get("id", ""),
                            "doc_id": hit["entity"].get("doc_id", ""),
                            "content": hit["entity"].get("content", ""),
                            "chunk_type": hit["entity"].get("chunk_type", ""),
                            "parent_id": hit["entity"].get("parent_id", ""),
                            "category": hit["entity"].get("category", ""),
                            "collection": col_name,
                            "score": hit.get("distance", 0),
                            "source": "sparse",
                        })
            except Exception as e:
                logger.warning("mt_retriever.sparse_search_failed",
                              col=col_name, error=str(e)[:100])

        return candidates

    def _rrf_fusion(self, candidates: list[dict], k: int = 60) -> list[dict]:
        """RRF (Reciprocal Rank Fusion) 融合排序。"""
        # 按 (dense_rank, sparse_rank) 加权
        scored = {}
        for rank, cand in enumerate(candidates):
            cid = cand["id"]
            rrf_score = 1.0 / (k + rank + 1)
            if cid not in scored:
                scored[cid] = {"candidate": cand, "score": rrf_score}
            else:
                scored[cid]["score"] += rrf_score

        merged = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        return [m["candidate"] for m in merged]

    def _map_to_parents(self, collections: list[str], docs: list[dict]) -> list[dict]:
        """子块文档 → 查父块完整内容。"""
        # 分离子块和父块
        children = [d for d in docs if d.get("chunk_type") == "child"]
        parents = [d for d in docs if d.get("chunk_type") == "parent"]

        if not children:
            return parents or docs

        # 收集 parent_id
        parent_ids = list(set(d["parent_id"] for d in children if d.get("parent_id")))
        if not parent_ids:
            return parents or docs

        # 从 Milvus 查父块
        parent_docs = {}
        for col_name in collections:
            try:
                expr = f"id in {parent_ids}"
                results = self.client.query(
                    collection_name=col_name,
                    filter=expr,
                    output_fields=["id", "content", "doc_id", "category"],
                    limit=len(parent_ids),
                )
                for r in results:
                    parent_docs[r["id"]] = r
            except Exception as e:
                logger.warning("mt_retriever.parent_lookup_failed",
                              col=col_name, error=str(e)[:80])

        # 映射子块到父块
        mapped = []
        seen_ids = set()
        for child in children:
            pid = child.get("parent_id", "")
            if pid in parent_docs and pid not in seen_ids:
                p = parent_docs[pid]
                mapped.append({
                    "id": p["id"],
                    "doc_id": p.get("doc_id", ""),
                    "content": p["content"],
                    "chunk_type": "parent",
                    "category": p.get("category", ""),
                    "collection": child.get("collection", ""),
                    "score": child.get("score", 0),
                })
                seen_ids.add(pid)

        # 原有的父块也保留
        for p in parents:
            if p["id"] not in seen_ids:
                mapped.append(p)
                seen_ids.add(p["id"])

        return mapped if mapped else docs

    def format_context(self, docs: list[dict]) -> str:
        """文档列表 → LLM 可用的上下文字符串。"""
        if not docs:
            return ""
        parts = []
        for i, d in enumerate(docs, 1):
            cat = d.get("category", "未知")
            src = d.get("collection", "unknown")
            parts.append(f"[{src}/{cat}]\n{d['content']}")
        return "\n\n---\n\n".join(parts)


# ── 测试 ──
if __name__ == "__main__":
    import os as _os
    _os.environ["HF_HUB_OFFLINE"] = "1"
    _os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("MultiTenantRetriever 测试")
    r = MultiTenantRetriever()

    # 测试 a001
    print("\n--- a001 退货政策 ---")
    docs = r.search("退货政策是什么", shop_id="a001", top_k=3)
    for d in docs:
        print(f"  [{d['collection']}/{d['chunk_type']}] {d['content'][:50]}...")

    # 测试隔离
    print("\n--- b002 (不存在) ---")
    docs2 = r.search("退货政策", shop_id="b002", top_k=3)
    print(f"  结果数: {len(docs2)} (应为0)")

    print("\n测试完成")
