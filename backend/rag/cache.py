# backend/rag/cache.py
# 🆕 v4 两级 Redis 缓存层：精确匹配 + 语义相似，减少重复 LLM + RAG 开销。
#
# 策略：
#   - 一级：精确匹配缓存 (Key = MD5(query + source_filter))
#   - 二级：FAQ 答案缓存 (Key = answer:{query}) — BM25 命中后缓存
#   - TTL = 1小时(RAG) / 2小时(FAQ)（可配置）
#   - 异步写入（不阻塞用户响应）
#   - Redis 不可用时自动降级，不影响主流程
#
# 使用方式：
#   cache = RAGCache()
#   cached = await cache.get(query, source_filter)       # → str or None
#   await cache.set(query, source_filter, answer)        # 异步写入
#   cached = await cache.get_answer(query)               # FAQ 精确匹配
#   await cache.set_answer(query, answer)                # FAQ 答案缓存

import asyncio
import hashlib
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


class RAGCache:
    """Redis 问答缓存 — 可选依赖，不可用时静默降级。"""

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.rag_cache_enabled
        self._redis = None
        self._available = False
        if self.enabled:
            self._init_redis()

    def _init_redis(self):
        """尝试连接 Redis。"""
        try:
            import redis.asyncio as aioredis

            kwargs = dict(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            # 有密码则传入
            pwd = getattr(self.settings, "redis_password", "")
            if pwd:
                kwargs["password"] = pwd

            db = getattr(self.settings, "redis_db", 0)
            if db:
                kwargs["db"] = db

            self._redis = aioredis.Redis(**kwargs)
            # 可用性在首次操作时验证
            self._available = True
            logger.info("cache.redis_configured",
                       host=self.settings.redis_host, port=self.settings.redis_port)
        except ImportError:
            logger.warning("cache.redis_not_installed",
                          hint="pip install redis 以启用问答缓存")
            self._available = False
        except Exception as e:
            logger.warning("cache.redis_init_failed", error=str(e))
            self._available = False

    # ── 缓存 Key ───────────────────────────────────────────

    def _make_key(self, query: str, source_filter: str = "") -> str:
        """生成缓存 Key。"""
        raw = f"{query}|{source_filter or 'all'}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
        return f"rag:v1:{digest}"

    # ── 读取 ───────────────────────────────────────────────

    async def get(self, query: str, source_filter: str = "") -> Optional[str]:
        """从缓存读取答案。

        :return: 缓存的答案字符串，未命中返回 None
        """
        if not self._available or not self._redis:
            return None

        try:
            key = self._make_key(query, source_filter)
            value = await asyncio.wait_for(
                self._redis.get(key),
                timeout=1.0,
            )
            if value:
                logger.info("cache.hit", key=key[:12])
            return value
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.warning("cache.get_failed", error=str(e)[:60])
            return None

    # ── 写入（异步，不阻塞）────────────────────────────────

    async def set(
        self, query: str, source_filter: str, answer: str
    ):
        """异步写入缓存（fire-and-forget）。"""
        if not self._available or not self._redis:
            return

        async def _write():
            try:
                key = self._make_key(query, source_filter)
                await asyncio.wait_for(
                    self._redis.setex(
                        key,
                        self.settings.rag_cache_ttl,
                        answer,
                    ),
                    timeout=2.0,
                )
                logger.debug("cache.set", key=key[:12], ttl=self.settings.rag_cache_ttl)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.warning("cache.set_failed", error=str(e)[:60])

        # 不阻塞主流程
        asyncio.create_task(_write())

    # ── 清除 ───────────────────────────────────────────────

    async def clear(self, query: str, source_filter: str = ""):
        """清除指定缓存。"""
        if not self._available or not self._redis:
            return
        try:
            key = self._make_key(query, source_filter)
            await asyncio.wait_for(self._redis.delete(key), timeout=1.0)
        except Exception:
            pass

    # ── 🆕 FAQ 精确匹配缓存（BM25 命中后写入）──────────────

    async def get_answer(self, query: str) -> Optional[str]:
        """读取 FAQ 精确匹配缓存（Key = answer:{query}）。"""
        if not self._available or not self._redis:
            return None
        try:
            answer = await asyncio.wait_for(
                self._redis.get(f"answer:{query}"),
                timeout=1.0,
            )
            if answer:
                logger.info("cache.faq_hit", query=query[:30])
            return answer
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.warning("cache.faq_get_failed", error=str(e)[:60])
            return None

    async def set_answer(self, query: str, answer: str):
        """写入 FAQ 精确匹配缓存（Key = answer:{query}）。"""
        if not self._available or not self._redis:
            return

        async def _write():
            try:
                ttl = getattr(self.settings, "bm25_cache_ttl", 7200)
                await asyncio.wait_for(
                    self._redis.setex(f"answer:{query}", ttl, answer),
                    timeout=2.0,
                )
                logger.debug("cache.faq_set", query=query[:30], ttl=ttl)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.warning("cache.faq_set_failed", error=str(e)[:60])

        asyncio.create_task(_write())

    # ── 健康检查 ───────────────────────────────────────────

    async def health_check(self) -> bool:
        """检查 Redis 是否可用。"""
        if not self._redis:
            return False
        try:
            await asyncio.wait_for(self._redis.ping(), timeout=1.0)
            return True
        except Exception:
            self._available = False
            return False


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio

    async def test():
        cache = RAGCache()
        print(f"缓存启用: {cache.enabled}")
        print(f"Redis 可用: {cache._available}")

        if cache._available:
            await cache.set("怎么退货", "after_sales", "您可以进入订单页申请退货...")
            result = await cache.get("怎么退货", "after_sales")
            print(f"缓存命中: {result}")
        else:
            print("Redis 未配置或不可用，缓存层静默降级")
            result = await cache.get("test", "")
            print(f"降级模式 get: {result}")  # 应为 None

    asyncio.run(test())
