# backend/api/v1/auth.py
# 登录认证接口：POST /auth/login（签发 JWT） + GET /auth/me（验证鉴权）

import asyncio
import types as _types
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt_mod
from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.dependencies import get_db, get_current_user
from backend.models.schemas import LoginRequest, TokenResponse

# ── 兼容性补丁（passlib 与 bcrypt>=4.0）─────────────────────
if not hasattr(_bcrypt_mod, "__about__"):
    _about = _types.SimpleNamespace(__version__=getattr(_bcrypt_mod, "__version__", "4.x"))
    _bcrypt_mod.__about__ = _about

logger = get_logger(__name__)
router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── 辅助函数 ────────────────────────────────────────────────

def _create_access_token(data: dict, expires_minutes: int) -> str:
    """把身份信息 + 过期时间打包，用密钥签名成 JWT 字符串。"""
    settings = get_settings()
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload["exp"] = expire
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def hash_password(password: str) -> str:
    """对明文密码进行 bcrypt 哈希。"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码与哈希是否匹配。"""
    return pwd_context.verify(plain_password, hashed_password)


# ── 登录接口 ────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """用户登录，返回 JWT Access Token（支持用户名、邮箱或手机号登录）。"""
    settings = get_settings()

    # 查数据库（用户名、邮箱或手机号）
    result = await db.execute(
        text(
            "SELECT id, username, password_hash, role, merchant_id, is_active "
            "FROM users WHERE username = :val OR email = :val OR phone = :val LIMIT 1"
        ),
        {"val": req.username},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    if not row.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用，请联系管理员",
        )

    # 密码校验（线程池执行 bcrypt 避免阻塞事件循环）
    loop = asyncio.get_running_loop()
    password_ok = await loop.run_in_executor(
        None,
        pwd_context.verify,
        req.password,
        row.password_hash,
    )

    if not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 签发 Token
    token = _create_access_token(
        data={
            "sub": str(row.id),
            "role": row.role,
            "merchant_id": row.merchant_id,
            "username": row.username,
        },
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )

    logger.info("auth.login_success", user_id=str(row.id), role=row.role)

    return TokenResponse(
        access_token=token,
        user_id=row.id,
        username=row.username,
        role=row.role,
    )


# ── 获取当前用户信息 ────────────────────────────────────────

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """获取当前登录用户信息（用于验证 Token 是否有效）。"""
    return current_user


# ── 注册接口（公开）──────────────────────────────────────────

@router.post("/register")
async def register(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """用户注册（公开接口，默认角色为 customer）。"""
    # 检查用户名是否已存在
    result = await db.execute(
        text("SELECT id FROM users WHERE username = :val OR email = :val LIMIT 1"),
        {"val": req.username},
    )
    if result.fetchone():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名或邮箱已被注册",
        )

    # 检查手机号是否已被注册
    if req.phone:
        result = await db.execute(
            text("SELECT id FROM users WHERE phone = :phone LIMIT 1"),
            {"phone": req.phone},
        )
        if result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="该手机号已被注册",
            )

    # 哈希密码
    loop = asyncio.get_running_loop()
    hashed = await loop.run_in_executor(None, pwd_context.hash, req.password)

    # 写入数据库（支持企业注册传 role + merchant_id）
    user_role = req.role or "customer"
    user_merchant_id = req.merchant_id or ""
    await db.execute(
        text(
            "INSERT INTO users (username, email, phone, password_hash, role, merchant_id, is_active) "
            "VALUES (:username, :email, :phone, :password_hash, :role, :merchant_id, TRUE)"
        ),
        {
            "username": req.username,
            "email": req.username if "@" not in req.username else req.username,
            "phone": req.phone or "",
            "password_hash": hashed,
            "role": user_role,
            "merchant_id": user_merchant_id,
        },
    )
    await db.commit()

    logger.info("auth.register_success", username=req.username, phone=req.phone or "", role=user_role)

    return {"success": True, "message": "注册成功，请登录"}


# ── 管理员创建用户接口 ───────────────────────────────────────

@router.post("/users")
async def admin_create_user(
    req: LoginRequest,
    role: str = "customer",
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """管理员/商户创建新用户（可指定角色）。"""
    user_role = current_user.get("role", "")
    if user_role not in ("admin", "merchant"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限：仅管理员或商户可创建用户",
        )

    if role not in ("customer", "merchant"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="角色仅支持 customer 或 merchant",
        )

    result = await db.execute(
        text("SELECT id FROM users WHERE username = :val OR email = :val LIMIT 1"),
        {"val": req.username},
    )
    if result.fetchone():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名或邮箱已存在",
        )

    loop = asyncio.get_running_loop()
    hashed = await loop.run_in_executor(None, pwd_context.hash, req.password)

    await db.execute(
        text(
            "INSERT INTO users (username, email, password_hash, role, is_active) "
            "VALUES (:username, :email, :password_hash, :role, TRUE)"
        ),
        {
            "username": req.username,
            "email": req.username if "@" not in req.username else req.username,
            "password_hash": hashed,
            "role": role,
        },
    )
    await db.commit()

    logger.info("auth.admin_create_user", created_by=current_user.get("user_id"), username=req.username, role=role)

    return {"success": True, "message": f"用户 {req.username}（{role}）创建成功"}
