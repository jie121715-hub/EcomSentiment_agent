#!/usr/bin/env python
# scripts/seed_users.py
# 初始化管理员和商户用户（bcrypt 密码哈希）
#
# 用法：
#   cd EcomSentiment_agent
#   python scripts/seed_users.py

import asyncio
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SEED_USERS = [
    {
        "username": "admin",
        "email": "admin@ecom-agent.com",
        "password": "admin123",
        "role": "admin",
        "merchant_id": "all",
    },
    {
        "username": "merchant",
        "email": "merchant@ecom-agent.com",
        "password": "merchant123",
        "role": "merchant",
        "merchant_id": "shop_001",
    },
    {
        "username": "customer",
        "email": "customer@example.com",
        "password": "customer123",
        "role": "customer",
        "merchant_id": "",
    },
]


async def seed():
    from backend.core.database import init_db, get_session
    from sqlalchemy import text

    # 确保表存在
    await init_db()

    async with get_session() as session:
        for user in SEED_USERS:
            # 检查是否已存在
            result = await session.execute(
                text("SELECT id FROM users WHERE username = :uname"),
                {"uname": user["username"]},
            )
            if result.fetchone():
                print(f"[跳过] 用户 {user['username']} 已存在")
                continue

            hashed = pwd_context.hash(user["password"])
            await session.execute(
                text(
                    "INSERT INTO users (username, email, password_hash, role, merchant_id, is_active) "
                    "VALUES (:uname, :email, :pwd, :role, :mid, TRUE)"
                ),
                {
                    "uname": user["username"],
                    "email": user["email"],
                    "pwd": hashed,
                    "role": user["role"],
                    "mid": user["merchant_id"],
                },
            )
            print(f"[创建] {user['username']} ({user['role']}) — 密码: {user['password']}")

    await session.commit()
    print("\n✅ 种子用户初始化完成")


if __name__ == "__main__":
    asyncio.run(seed())
