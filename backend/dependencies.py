# backend/dependencies.py
# FastAPI 依赖注入：① get_db 数据库会话  ② get_current_user JWT鉴权  ③ verify_api_key API Key鉴权

from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from backend.config import get_settings
from backend.core.database import get_session
from backend.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── 数据库会话依赖 ──────────────────────────────────────────

async def get_db() -> AsyncGenerator:
    """FastAPI 依赖：获取异步数据库会话，自动提交 / 回滚 / 关闭。"""
    session = get_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ── JWT 鉴权 ───────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """FastAPI 依赖：验证 JWT Bearer Token，返回当前用户信息。

    返回: {"user_id": str, "role": str, "merchant_id": str}
    Token 无效或缺失则抛 401。
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")
        role: str = payload.get("role", "customer")
        merchant_id: str = payload.get("merchant_id", "default")
        username: str = payload.get("username", "")

        if not user_id:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    return {"user_id": user_id, "role": role, "merchant_id": merchant_id, "username": username}


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict | None:
    """可选鉴权：有 Token 就解析，没有返回 None（不抛 401）。"""
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# ── API Key 鉴权（向后兼容 / 服务间调用）────────────────────

async def verify_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> dict:
    """验证 API Key 并返回用户身份。

    鉴权规则：
      - 无 Key / 无效 Key → role="customer"（只能聊天，不能上传）
      - merchant Key     → role="merchant"（可上传知识）
      - admin Key        → role="admin"（可上传 + 管理）
    """
    if not x_api_key:
        return {"role": "customer", "merchant_id": "default", "authenticated": False}

    # 管理员 Key
    if x_api_key == settings.admin_api_key:
        return {"role": "admin", "merchant_id": "all", "authenticated": True}

    # 商户 Key
    if x_api_key == settings.merchant_api_key:
        return {"role": "merchant", "merchant_id": "default", "authenticated": True}

    # 无效 Key → 降级为 customer
    return {"role": "customer", "merchant_id": "default", "authenticated": False}


async def verify_auth(
    # 优先 JWT，降级 API Key
    jwt_user: dict | None = Depends(get_optional_user),
    api_key_user: dict = Depends(verify_api_key),
) -> dict:
    """统一鉴权：优先 JWT Token，无 Token 时降级到 API Key。

    聊天接口可用此依赖，兼容 Web 前端（JWT）和微信桥接（API Key）。
    """
    if jwt_user is not None:
        return jwt_user
    return api_key_user
