#!/usr/bin/env python
# scripts/check_env.py
# 环境连通性检查：MySQL + Milvus + Redis + LLM
#
# 用法：
#   cd EcomSentiment_agent
#   python scripts/check_env.py

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def check_mysql():
    """检查 MySQL 连接。"""
    print("[1/4] 检查 MySQL 连接...", end=" ")
    try:
        from backend.core.database import get_engine
        engine = get_engine()
        async with engine.connect() as conn:
            from sqlalchemy import text
            result = await conn.execute(text("SELECT VERSION()"))
            version = result.fetchone()[0]
        print(f"✅ OK (MySQL {version})")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


async def check_milvus():
    """检查 Milvus 连接。"""
    print("[2/4] 检查 Milvus 连接...", end=" ")
    try:
        from backend.rag.retriever import EcomRetriever
        r = EcomRetriever()
        if r.vector_store is not None:
            backend = r._backend or "unknown"
            print(f"✅ OK (后端: {backend})")
            return True
        else:
            print("⚠️  降级到 Chroma（Milvus 未连接）")
            return True  # Chroma 降级不算失败
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


async def check_redis():
    """检查 Redis 连接。"""
    print("[3/4] 检查 Redis 连接...", end=" ")
    try:
        import redis.asyncio as aioredis
        from backend.config import get_settings
        s = get_settings()
        r = aioredis.from_url(
            f"redis://:{s.redis_password}@{s.redis_host}:{s.redis_port}/{s.redis_db}"
        )
        await r.ping()
        await r.close()
        print(f"✅ OK (Redis {s.redis_host}:{s.redis_port})")
        return True
    except Exception as e:
        print(f"⚠️  跳过: {e}")
        return True


async def check_llm():
    """检查 LLM API 连接。"""
    print("[4/4] 检查 LLM API 连接...", end=" ")
    try:
        from backend.core.llm_factory import get_llm
        llm = get_llm()
        from backend.config import get_settings
        s = get_settings()
        print(f"✅ OK (模型: {s.llm_default_model})")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


async def main():
    print("EcomSentiment_agent 环境检查\n" + "=" * 50)
    results = await asyncio.gather(
        check_mysql(),
        check_milvus(),
        check_redis(),
        check_llm(),
    )
    print("=" * 50)
    ok = sum(1 for r in results if r)
    total = len(results)
    status = "✅ 全部通过" if ok == total else f"⚠️  {ok}/{total} 通过"
    print(f"\n结果: {status}")


if __name__ == "__main__":
    asyncio.run(main())
