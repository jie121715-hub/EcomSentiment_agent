# backend/agents/context_resolver.py
# 上下文解析器：从用户请求中解析 shop_id（多租户隔离关键）
#
# 解析策略（按优先级）：
#   1. JWT 中的 merchant_id（用户登录时绑定）
#   2. 订单号提取 → 查 orders 表 → 得 shop_id
#   3. 对话历史缓存（上一轮已解析的 shop_id）
#   4. 解析失败 → 不设限（admin 看全部 / 反问澄清）
#
# 集成方式：插入到 Graph 的 perceive → route → resolve → dispatch 中

from typing import Optional

from backend.core.logger import get_logger

logger = get_logger(__name__)


class ContextResolver:
    """多租户上下文解析器。

    使用方式：
        resolver = ContextResolver()
        shop_id = await resolver.resolve(
            user_id="7",
            entities=[{"type": "order_id", "value": "TK-20240715-0001"}],
            current_shop_id="a001",  # 从 JWT 来的
        )
    """

    # 缓存：session_id → shop_id（避免同一会话重复查库）
    _cache: dict[str, str] = {}

    async def resolve(
        self,
        user_id: str = "",
        entities: list[dict] | None = None,
        current_shop_id: str = "",
        session_id: str = "",
        query: str = "",
        intent: str = "",
    ) -> dict:
        """解析 shop_id，返回 {"shop_id": str, "source": str, "confidence": float}。

        source: "jwt" | "order" | "product" | "cache" | "none"
        """
        # ── 1. 对话缓存命中 ──
        if session_id and session_id in self._cache:
            logger.info("context_resolver.cache_hit", session_id=session_id,
                       shop_id=self._cache[session_id])
            return {"shop_id": self._cache[session_id], "source": "cache", "confidence": 0.9}

        entities = entities or []

        # 意图分类（BERT 4分类）：知识问答→跨店浏览；业务处理/工单→锁店铺
        BROWSE_CN = {"知识问答", "knowledge_qa"}
        LOCK_CN = {"业务处理", "business", "工单处理", "escalate"}

        is_browse = intent in BROWSE_CN
        is_lock = intent in LOCK_CN or (not is_browse and intent)  # 未知意图默认锁
        order_ids = [e["value"] for e in entities if e.get("type") in ("order_id", "order")]
        if order_ids:
            shop_id = await self._resolve_by_order(order_ids[0])
            if shop_id:
                self._cache_session(session_id, shop_id)
                return {"shop_id": shop_id, "source": "order", "confidence": 1.0}

        # ── 3. 订单号 → 锁店铺（最高优先级）──
        if order_ids:
            shop_id = await self._resolve_by_order(order_ids[0])
            if shop_id:
                self._cache_session(session_id, shop_id)
                return {"shop_id": shop_id, "source": "order", "confidence": 1.0}

        # ── 4. 商品浏览类 → 跨店检索 ──
        if is_browse:
            return {"shop_id": "", "source": "browse_all", "confidence": 0.8}

        # ── 5. 订单/售后类 → 锁 JWT 店铺 ──
        if is_lock and current_shop_id and current_shop_id != "default":
            self._cache_session(session_id, current_shop_id)
            return {"shop_id": current_shop_id, "source": "jwt", "confidence": 0.95}

        # ── 6. 用户最近订单归属（锁店铺场景）──
        if is_lock and user_id and user_id != "anonymous":
            shop_id = await self._resolve_by_user_recent_order(user_id)
            if shop_id:
                self._cache_session(session_id, shop_id)
                return {"shop_id": shop_id, "source": "user_history", "confidence": 0.7}

        # ── 7. 默认 → 跨店检索 ──
        return {"shop_id": "", "source": "none", "confidence": 0.0}

    async def _resolve_by_order(self, order_id: str) -> str:
        """根据订单号查询 shop_id。"""
        try:
            from backend.core.database import get_session
            from sqlalchemy import text

            async with get_session() as session:
                result = await session.execute(
                    text("SELECT shop_id FROM orders WHERE order_id = :oid LIMIT 1"),
                    {"oid": order_id},
                )
                row = result.fetchone()
                if row and row[0]:
                    logger.info("context_resolver.order_found",
                               order_id=order_id, shop_id=row[0])
                    return str(row[0])
        except Exception as e:
            logger.warning("context_resolver.order_lookup_failed", error=str(e)[:100])
        return ""

    async def _resolve_by_user_recent_order(self, user_id: str) -> str:
        """根据用户最近订单确定 shop_id。"""
        try:
            from backend.core.database import get_session
            from sqlalchemy import text

            async with get_session() as session:
                result = await session.execute(
                    text("SELECT shop_id, COUNT(*) as cnt FROM orders "
                         "WHERE user_id = :uid GROUP BY shop_id "
                         "ORDER BY cnt DESC LIMIT 1"),
                    {"uid": user_id},
                )
                row = result.fetchone()
                if row and row[0]:
                    logger.info("context_resolver.user_shop_found",
                               user_id=user_id, shop_id=row[0])
                    return str(row[0])
        except Exception as e:
            logger.warning("context_resolver.user_lookup_failed", error=str(e)[:100])
        return ""

    def _cache_session(self, session_id: str, shop_id: str):
        if session_id and shop_id:
            self._cache[session_id] = shop_id
            # 限制缓存大小
            if len(self._cache) > 10000:
                # 删除最早的 1000 条
                keys = list(self._cache.keys())[:1000]
                for k in keys:
                    del self._cache[k]

    @classmethod
    def clear_cache(cls, session_id: str = ""):
        """清除缓存（用户切换企业时调用）。"""
        if session_id:
            cls._cache.pop(session_id, None)
        else:
            cls._cache.clear()


# 全局单例
_resolver: Optional[ContextResolver] = None


def get_context_resolver() -> ContextResolver:
    global _resolver
    if _resolver is None:
        _resolver = ContextResolver()
    return _resolver
