# backend/api/v1/admin_knowledge.py
# 知识库管理 API：CRUD + 同步

from fastapi import APIRouter, HTTPException, Depends

from backend.core.logger import get_logger
from backend.models.schemas import KnowledgeItem, KnowledgeSyncResponse
from backend.dependencies import get_current_user

logger = get_logger(__name__)
router = APIRouter()


@router.get("/knowledge")
async def list_knowledge(
    category: str = "",
    merchant_id: str = "",
    offset: int = 0,
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
):
    """查看知识库内容（分页），自动按企业隔离（admin 可看全部）。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge
    from sqlalchemy import select, func

    user_role = current_user.get("role", "")
    user_mid = current_user.get("merchant_id", "")

    async with get_session() as session:
        base = select(CustomKnowledge)
        if category:
            base = base.where(CustomKnowledge.category == category)

        # 企业隔离：非 admin 只看自己的
        if user_role != "admin":
            base = base.where(CustomKnowledge.merchant_id == user_mid)
        elif merchant_id:
            base = base.where(CustomKnowledge.merchant_id == merchant_id)

        # 总数
        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await session.execute(count_stmt)).scalar() or 0

        # 分页数据
        stmt = base.order_by(CustomKnowledge.id.asc()).offset(offset).limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [{
            "id": r.id, "content": r.content, "category": r.category,
            "merchant_id": r.merchant_id, "source": r.source,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        } for r in rows],
    }


@router.post("/knowledge")
async def add_knowledge(item: KnowledgeItem):
    """添加知识到 MySQL 并同步向量库。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge

    async with get_session() as session:
        record = CustomKnowledge(
            content=item.content,
            category=item.category,
            merchant_id=item.merchant_id,
            source=f"merchant:{item.merchant_id}",
        )
        session.add(record)
        await session.commit()
        kb_id = record.id

    # 不再自动同步旧 Milvus（维度不兼容），改为前端调 upload-text 写入 eco_rag
    logger.info("admin.knowledge_added", id=kb_id, category=item.category)
    return {
        "success": True, "id": kb_id,
        "content": item.content[:100] + "...",
    }


@router.post("/knowledge/batch")
async def batch_add_knowledge(items: list[KnowledgeItem]):
    """批量添加知识（一次最多100条）。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge

    if len(items) > 100:
        raise HTTPException(status_code=400, detail="单次最多100条")

    async with get_session() as session:
        for item in items:
            session.add(CustomKnowledge(
                content=item.content,
                category=item.category,
                merchant_id=item.merchant_id,
                source=f"merchant:{item.merchant_id}",
            ))
        await session.commit()

    logger.info("admin.knowledge_batch", count=len(items))
    return {"success": True, "count": len(items)}


@router.delete("/knowledge/{kb_id}")
async def delete_knowledge(kb_id: int):
    """删除指定知识（MySQL）。向量库残留需调 sync 清理。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge
    from sqlalchemy import select

    async with get_session() as session:
        result = await session.execute(select(CustomKnowledge).where(CustomKnowledge.id == kb_id))
        record = result.scalar_one_or_none()
        if not record:
            raise HTTPException(status_code=404, detail=f"知识 {kb_id} 不存在")
        await session.delete(record)
        await session.commit()

    return {"success": True, "deleted_id": kb_id}


@router.post("/knowledge/sync", response_model=KnowledgeSyncResponse)
async def sync_knowledge():
    """同步接口已废弃——请使用 upload-text 写入 eco_rag。"""
    return KnowledgeSyncResponse(success=True, synced=0, chunks=0, backend="deprecated", error="请使用 upload-text 接口")
