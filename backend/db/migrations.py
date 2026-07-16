# backend/db/migrations.py
# 运行时增量迁移：应用启动时检查并补全缺失的表/列
#
# 设计原则：
#   - 每条迁移只做增量操作，兼容 MySQL 5.7+
#   - 迁移失败不阻塞启动（只打 warning）
#   - 新装部署走 create_all 全量建表

from backend.core.database import get_session
from backend.core.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)

# 迁移列表: (描述, SQL)
_MIGRATIONS: list[tuple[str, str]] = [
    # v1.1: 用户认证 — 兼容 MySQL 5.7（不用 IF NOT EXISTS）
    (
        "users.is_active",
        "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE",
    ),
    (
        "users.merchant_id",
        "ALTER TABLE users ADD COLUMN merchant_id VARCHAR(64) DEFAULT ''",
    ),
    # 后续迁移在此追加...
]


async def run_migrations() -> int:
    """执行所有未应用的增量迁移。返回成功执行的迁移数。

    兼容 MySQL 5.7：不使用 ADD COLUMN IF NOT EXISTS（MySQL 8.0 才支持），
    改为 try-except 捕获 Duplicate column 错误后静默跳过。
    """
    applied = 0
    async with get_session() as session:
        for desc, sql in _MIGRATIONS:
            try:
                await session.execute(text(sql))
                await session.commit()
                applied += 1
                logger.info("db.migration_applied", column=desc)
            except Exception as e:
                await session.rollback()
                err_str = str(e).lower()
                if "duplicate column" in err_str or "already exists" in err_str:
                    # 列已存在，正常跳过
                    logger.info("db.migration_skipped", column=desc, reason="already_exists")
                else:
                    logger.warning("db.migration_failed", column=desc, error=str(e)[:120])
    return applied
