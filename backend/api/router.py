# backend/api/router.py
# API 路由聚合器：挂载所有 v1 子路由

from fastapi import APIRouter

from backend.api.v1 import auth, chat, admin_knowledge, admin_upload, admin_import

api_router = APIRouter()

# ── 认证接口 ──
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])

# ── 对话接口 ──
api_router.include_router(chat.router, prefix="", tags=["对话"])

# ── 知识库管理 ──
api_router.include_router(admin_knowledge.router, prefix="/admin", tags=["知识库管理"])

# ── 文件上传 ──
api_router.include_router(admin_upload.router, prefix="/admin", tags=["文件上传"])

# ── 淘宝导入 ──
api_router.include_router(admin_import.router, prefix="/admin", tags=["淘宝导入"])
