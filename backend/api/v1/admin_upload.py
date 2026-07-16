# backend/api/v1/admin_upload.py
# 文件上传知识库 API：PDF / DOCX 上传 → 文本提取 → 安全扫描 → 父子块切分 → 双写

import os as _os
import tempfile
import time as _time

from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.content_moderation import scan_content
from backend.dependencies import verify_api_key, get_current_user

logger = get_logger(__name__)
router = APIRouter()


@router.post("/knowledge/upload")
async def upload_knowledge_file(
    file: UploadFile = File(..., description="PDF 或 DOCX 文件"),
    category: str = Form(default="product", description="知识分类: product / policy / faq"),
    current_user: dict | None = Depends(get_current_user),
):
    """上传 PDF/DOCX 文件，提取文本并写入 MySQL + eco_rag。

    🔐 JWT 鉴权（自动带企业编号隔离）
    支持格式：.pdf / .docx / .md / .txt（最大 50MB）
    """
    # ── 0. 鉴权校验 ──
    if not current_user or current_user.get("role") not in ("merchant", "admin"):
        raise HTTPException(
            status_code=403,
            detail="仅商户或管理员可上传知识，请先登录企业账号",
        )

    merchant_id = current_user.get("merchant_id", "")
    if not merchant_id:
        raise HTTPException(status_code=400, detail="未绑定企业，请使用企业账号登录")

    # ── 1. 校验文件类型 ──
    filename = file.filename or "unknown"
    ext = _os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx", ".md", ".markdown", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 {ext}，仅支持 PDF / DOCX / Markdown / TXT",
        )

    # ── 2. 保存临时文件 ──
    try:
        content = await file.read()
        file_size = len(content)
        if file_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小不能超过 50MB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("admin.upload_received",
                   filename=filename, size=file_size, role=current_user.get("role", ""))

        # ── 3. 文档加载 — 提取文本 ──
        from backend.rag.doc_loader import DocLoader
        loader = DocLoader()
        docs = loader.load(tmp_path)

        # 清理临时文件
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

        if not docs:
            raise HTTPException(status_code=400, detail="文件内容为空或无法提取文本")

        full_text = docs[0].page_content
        text_length = len(full_text)

        # ── 4. 🛡️ 恶意内容扫描 ──
        scan_result = scan_content(full_text)

        if scan_result["verdict"] == "reject":
            risk_details = "; ".join(r["reason"] for r in scan_result["risks"])
            logger.warning("admin.upload_rejected",
                          filename=filename, risks=risk_details)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "文件内容包含违规信息，已拒绝上传",
                    "risks": [
                        {"pattern": r["pattern"], "reason": r["reason"]}
                        for r in scan_result["risks"]
                    ],
                },
            )

        needs_review = (scan_result["verdict"] == "review")

        # ── 5. 🧬 父子块切分 ──
        from backend.rag.chunker import ParentChildChunker
        from langchain_core.documents import Document as LCDocument

        chunker = ParentChildChunker()
        tagged_doc = LCDocument(
            page_content=full_text,
            metadata={
                "source": f"upload:{merchant_id}:{filename}",
                "category": category,
                "merchant_id": merchant_id,
                "uploaded_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        chunk_result = chunker.split_documents([tagged_doc])

        # ── 6. 写入 MySQL ──
        from backend.core.database import get_session
        from backend.models.db_models import CustomKnowledge

        status = "pending_review" if needs_review else "active"
        mysql_ids = []

        async with get_session() as session:
            for parent in chunk_result.parent_chunks:
                record = CustomKnowledge(
                    content=parent.page_content,
                    source=f"upload:{filename}",
                    category=category,
                    merchant_id=merchant_id,
                )
                session.add(record)
                await session.flush()
                mysql_ids.append(record.id)
            await session.commit()

        # ── 7. 🧬 写入 Milvus ──
        from backend.rag.retriever import EcomRetriever
        retriever = EcomRetriever()
        vector_ok = False
        parent_count = 0
        child_count = 0

        if retriever.vector_store is not None:
            try:
                retriever.vector_store.add_documents(chunk_result.child_chunks)
                child_count = len(chunk_result.child_chunks)

                retriever.vector_store.add_documents(chunk_result.parent_chunks)
                parent_count = len(chunk_result.parent_chunks)

                for pid, pcontent in chunk_result.parent_map.items():
                    retriever._parent_map[pid] = pcontent

                vector_ok = True
                logger.info("admin.upload_milvus_done",
                           parents=parent_count, children=child_count,
                           backend=retriever._backend)
            except Exception as e:
                logger.error("admin.upload_milvus_failed", error=str(e))
        else:
            logger.warning("admin.upload_no_vector_store")

        # ── 7.5 写入 eco_rag（新 Milvus，多租户隔离）──
        eco_rag_ok = False
        try:
            model = _get_bge_model()
            chunker = _get_chunker()
            milvus_client = _get_milvus_client()

            from langchain_core.documents import Document
            raw = Document(page_content=full_text, metadata={})
            cr = chunker.split_documents([raw])

            col_name = "policies" if category == "policy" else "products"
            doc_id = f"{merchant_id}_{category}_{int(_time.time())}"
            ts = int(_time.time())
            pid = f"{doc_id}_parent"
            rows = []

            for p in cr.parent_chunks:
                vec = model.encode([p.page_content], normalize_embeddings=True)
                rows.append(_make_row(pid, merchant_id, doc_id, "parent", pid, p.page_content, vec[0], category, ts))
            for j, c in enumerate(cr.child_chunks):
                vec = model.encode([c.page_content], normalize_embeddings=True)
                rows.append(_make_row(f"{doc_id}_child_{j}", merchant_id, doc_id, "child", pid, c.page_content, vec[0], category, ts))

            milvus_client.insert(collection_name=col_name, data=rows)
            eco_rag_ok = True
            logger.info("admin.upload_eco_rag_done", merchant=merchant_id, col=col_name, chunks=len(rows))
        except Exception as e:
            logger.warning("admin.upload_eco_rag_failed", error=str(e)[:120])

        # ── 8. 返回结果 ──
        logger.info("admin.upload_done",
                   filename=filename, text_len=text_length,
                   parents=parent_count, children=child_count,
                   vector_ok=vector_ok, eco_rag_ok=eco_rag_ok, needs_review=needs_review)

        result = {
            "success": True,
            "filename": filename,
            "file_size": file_size,
            "text_length": text_length,
            "mysql_ids": mysql_ids,
            "chunks": {
                "parents": parent_count,
                "children": child_count,
                "ratio": f"1:{child_count // max(parent_count, 1)}",
            },
            "vector_written": vector_ok,
            "backend": retriever._backend or "unknown",
            "status": status,
            "review_note": (
                "内容含可疑表述，已保存但待人工审核后生效"
                if needs_review else None
            ),
        }

        if scan_result["risks"]:
            result["scan_notes"] = [
                {"pattern": r["pattern"], "reason": r["reason"]}
                for r in scan_result["risks"]
            ]

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin.upload_failed", filename=filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# 文本上传（JWT 鉴权 → 写入 eco_rag）
# ═══════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field

# 全局懒加载（避免每次请求都加载模型）
_bge_model = None
_chunker = None
_milvus_client = None

def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        import os as _os
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        settings = get_settings()
        model_path = settings.embedding_model_name
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer(model_path, device="cpu")
        logger.info("admin_upload.bge_model_loaded")
    return _bge_model

def _get_chunker():
    global _chunker
    if _chunker is None:
        from backend.rag.chunker import ParentChildChunker
        settings = get_settings()
        _chunker = ParentChildChunker(
            parent_size=settings.rag_parent_chunk_size,
            parent_overlap=settings.rag_parent_chunk_overlap,
            child_size=settings.rag_child_chunk_size,
            child_overlap=settings.rag_child_chunk_overlap,
        )
    return _chunker

def _get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient
        _milvus_client = MilvusClient(uri="http://localhost:19530", db_name="eco_rag", timeout=15)
    return _milvus_client


class TextUploadRequest(BaseModel):
    content: str = Field(..., min_length=2, max_length=10000, description="知识内容")
    category: str = Field(default="policy", description="policy 或 product")


@router.post("/knowledge/upload-text")
async def upload_text(
    req: TextUploadRequest,
    current_user: dict = Depends(get_current_user),
):
    """上传文本知识到 eco_rag（JWT鉴权，自动带 shop_id 隔离）。"""
    role = current_user.get("role", "")
    shop_id = current_user.get("merchant_id", "")

    if role not in ("merchant", "admin"):
        raise HTTPException(status_code=403, detail="仅商户或管理员可上传")
    if not shop_id:
        raise HTTPException(status_code=400, detail="未绑定企业，请使用企业账号登录")

    try:
        from langchain_core.documents import Document

        model = _get_bge_model()
        chunker = _get_chunker()
        client = _get_milvus_client()

        raw = Document(page_content=req.content, metadata={})
        result = chunker.split_documents([raw])

        col_name = "policies" if req.category == "policy" else "products"
        doc_id = f"{shop_id}_{req.category}_{int(_time.time())}"
        ts = int(_time.time())
        pid = f"{doc_id}_parent"
        rows = []

        for p in result.parent_chunks:
            vec = model.encode([p.page_content], normalize_embeddings=True)
            rows.append(_make_row(pid, shop_id, doc_id, "parent", pid, p.page_content, vec[0], req.category, ts))

        for j, c in enumerate(result.child_chunks):
            vec = model.encode([c.page_content], normalize_embeddings=True)
            rows.append(_make_row(f"{doc_id}_child_{j}", shop_id, doc_id, "child", pid, c.page_content, vec[0], req.category, ts))

        client.insert(collection_name=col_name, data=rows)
        logger.info("admin.upload_text_done", shop_id=shop_id, collection=col_name, chunks=len(rows))

        return {
            "success": True,
            "shop_id": shop_id,
            "collection": col_name,
            "doc_id": doc_id,
            "chunks": {"parents": len(result.parent_chunks), "children": len(result.child_chunks)},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin.upload_text_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


def _make_row(row_id, shop_id, doc_id, chunk_type, parent_id, content, dense_vec, category, ts):
    return {
        "id": row_id,
        "shop_id": shop_id,
        "doc_id": doc_id,
        "chunk_type": chunk_type,
        "parent_id": parent_id,
        "content": content,
        "dense_vector": dense_vec.tolist(),
        "sparse_vector": {0: 0.001},  # SentenceTransformer 不含稀疏，用占位符
        "category": category,
        "created_at": ts,
    }


def _sparse_dict(sparse) -> dict:
    """BGE-M3 稀疏输出 → Milvus 格式 {int: float}。"""
    import collections as _col
    if not sparse:
        return {0: 0.001}
    if isinstance(sparse, (dict, _col.defaultdict)):
        return {int(k): float(v) for k, v in sparse.items()}
    if hasattr(sparse, "indices"):
        return {int(i): float(v) for i, v in zip(sparse.indices, sparse.data)}
    return {0: 0.001}


# ═══════════════════════════════════════════════════════════════
# 查询 eco_rag 知识列表
# ═══════════════════════════════════════════════════════════════

@router.get("/knowledge/rag-list")
async def list_rag_knowledge(
    collection: str = "policies",
    current_user: dict = Depends(get_current_user),
):
    """查询 eco_rag 中当前企业的知识列表（父块去重）。"""
    role = current_user.get("role", "")
    shop_id = current_user.get("merchant_id", "")

    if role not in ("merchant", "admin"):
        raise HTTPException(status_code=403, detail="仅商户或管理员可查看")
    if not shop_id:
        raise HTTPException(status_code=400, detail="未绑定企业")

    try:
        client = _get_milvus_client()
        col = collection if collection in ("policies", "products") else "policies"

        results = client.query(
            collection_name=col,
            filter=f'shop_id == "{shop_id}" and chunk_type == "parent"',
            output_fields=["id", "doc_id", "content", "category", "created_at"],
            limit=200,
        )
        items = []
        for r in results:
            items.append({
                "id": r.get("id", ""),
                "doc_id": r.get("doc_id", ""),
                "content": (r.get("content", "") or "")[:200],
                "category": r.get("category", ""),
                "collection": col,
                "created_at": r.get("created_at", 0),
            })
        return {"success": True, "shop_id": shop_id, "collection": col, "items": items, "total": len(items)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
