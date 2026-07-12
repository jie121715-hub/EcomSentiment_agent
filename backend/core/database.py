# backend/core/database.py
# MySQL 异步连接层：SQLAlchemy async engine + session factory。
# 用法：async with get_session() as session: ...

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── Engine & Session ─────────────────────────────────────────

_engine = None
_session_factory = None


def _build_url() -> str:
    s = get_settings()
    return (
        f"mysql+asyncmy://{s.db_user}:{s.db_password}"
        f"@{s.db_host}:{s.db_port}/{s.db_name}"
        f"?charset=utf8mb4"
    )


def get_engine():
    global _engine
    if _engine is None:
        url = _build_url()
        _engine = create_async_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,     # 每小时回收连接，防止 MySQL wait_timeout
            echo=False,
        )
        logger.info("database.engine_created", host=get_settings().db_host, db=get_settings().db_name)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


def get_session() -> AsyncSession:
    """获取一个新的异步Session（记得用 async with）。"""
    return get_session_factory()()


# ── ORM 基类 ─────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── 初始化：建表 ─────────────────────────────────────────────

async def init_db():
    """应用启动时调用：创建所有表（不存在则建）。"""
    # 确保所有 ORM 模型已注册到 Base.metadata
    import backend.models.db_models  # noqa: F401  触发所有表的注册
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database.tables_created")
