# backend/api/v1/admin_import.py
# 淘宝商品导入 API

from fastapi import APIRouter

from backend.core.logger import get_logger
from backend.models.schemas import TaobaoImportRequest

logger = get_logger(__name__)
router = APIRouter()


@router.post("/knowledge/import-taobao")
async def import_taobao(req: TaobaoImportRequest):
    """从淘宝 API 导入在售商品到知识库。"""
    from backend.utils.taobao_importer import import_products, TbConfig

    config = TbConfig(
        app_key=req.app_key,
        app_secret=req.app_secret,
        session_key=req.session_key,
        merchant_id=req.merchant_id,
        page_size=req.page_size,
    )
    result = await import_products(config)
    logger.info("admin.taobao_import", count=result.get("imported", 0))
    return result
