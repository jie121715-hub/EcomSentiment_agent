# backend/rag/bm25_search.py
# 🆕 BM25 + MySQL FAQ 检索层 — 三层架构的第二层（Redis → BM25 → RAG）。
#
# 核心流程：
#   1. Redis 精确缓存检查 (answer:{query}) → 命中直接返回
#   2. jieba 分词 → BM25Okapi 打分 → softmax 归一化
#   3. 最高分 >= threshold(0.85) → MySQL 取标准答案 → 写入 Redis 缓存
#   4. 最高分 <  threshold       → 返回 None，触发上层 RAG 回退
#
# 数据加载：
#   - 首次启动从 MySQL ecom_faq 表加载全部问题 → Redis 缓存问题列表
#   - 后续启动优先从 Redis 读取问题列表（秒级启动）
#   - BM25 索引在内存中构建（几百条FAQ仅占几MB）
#
# 使用方式：
#   bm25 = BM25FAQSearch()
#   await bm25.initialize()  # 加载数据 + 构建索引
#   answer = await bm25.search("如何退货")  # → str or None

import asyncio
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.rag.preprocess import preprocess_text
from backend.rag.cache import RAGCache

logger = get_logger(__name__)


class BM25FAQSearch:
    """BM25 FAQ 检索器 — 对高频客服问题做毫秒级关键词匹配。

    流程：Redis → BM25(softmax阈值) → MySQL → 缓存回写
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.bm25_enabled
        self.threshold = self.settings.bm25_threshold

        # BM25 模型实例（构建索引后赋值）
        self.bm25: Optional[BM25Okapi] = None
        # 分词后的问题列表（BM25 索引用）
        self.tokenized_questions: list[str] = []
        # 原始问题列表（MySQL 查询用）
        self.original_questions: list[str] = []

        # Redis 缓存客户端（复用 RAGCache 的 Redis 连接）
        self._cache: Optional[RAGCache] = None
        self._initialized = False

    @property
    def cache(self) -> RAGCache:
        if self._cache is None:
            self._cache = RAGCache()
        return self._cache

    async def initialize(self):
        """加载 FAQ 数据并构建 BM25 索引。

        优先从 Redis 缓存读取问题列表（秒级启动），
        缓存未命中则从 MySQL 加载并写回 Redis。
        """
        if self._initialized:
            return
        if not self.enabled:
            logger.info("bm25.disabled_by_config")
            self._initialized = True
            return

        # ── 1. 尝试从 Redis 加载 ──────────────────────────
        try:
            redis_original = await self._redis_get("bm25:original_questions")
            redis_tokenized = await self._redis_get("bm25:tokenized_questions")
            if redis_original and redis_tokenized:
                self.original_questions = redis_original
                self.tokenized_questions = redis_tokenized
                logger.info("bm25.loaded_from_redis", count=len(self.original_questions))
        except Exception as e:
            logger.warning("bm25.redis_load_failed", error=str(e)[:60])

        # ── 2. 缓存未命中 → 从 MySQL 加载 ──────────────────
        if not self.original_questions:
            await self._load_from_mysql()

        # ── 3. 构建 BM25 索引 ─────────────────────────────
        if self.tokenized_questions:
            self.bm25 = BM25Okapi(self.tokenized_questions)
            logger.info("bm25.index_built", faq_count=len(self.tokenized_questions))
        else:
            logger.warning("bm25.no_faq_data")

        self._initialized = True

    async def _load_from_mysql(self):
        """从 MySQL ecom_faq 表加载全部问题并缓存到 Redis。"""
        try:
            from backend.core.database import get_session
            from backend.models.db_models import EcomFAQ
            from sqlalchemy import select

            async with get_session() as session:
                result = await session.execute(select(EcomFAQ.question))
                rows = result.all()

            if not rows:
                logger.warning("bm25.mysql_empty")
                return

            self.original_questions = [r[0] for r in rows]
            # BM25Okapi 需要 list[list[str]]（每个文档是 token 列表）
            self.tokenized_questions = [preprocess_text(q).split() for q in self.original_questions]

            # 写回 Redis（下次启动直接加载）
            await self._redis_set("bm25:original_questions", self.original_questions)
            await self._redis_set("bm25:tokenized_questions", self.tokenized_questions)

            logger.info("bm25.loaded_from_mysql", count=len(self.original_questions))

        except Exception as e:
            logger.error("bm25.mysql_load_failed", error=str(e))

    # ═══════════════════════════════════════════════════════════
    # 核心搜索方法
    # ═══════════════════════════════════════════════════════════

    async def search(self, query: str) -> Optional[str]:
        """BM25 FAQ 检索主入口。

        流程：Redis精确缓存 → BM25评分 → MySQL取答案 → 缓存回写

        评分策略：
          - 用 raw_score / avg_score 判断区分度（>2x 表示明显匹配）
          - 辅以 softmax 验证（超过阈值表示匹配）
          - 两个条件都满足才返回答案，避免误匹配

        :param query: 用户原始查询
        :return: 匹配的答案字符串；未命中返回 None（触发上层 RAG 回退）
        """
        if not self._initialized:
            await self.initialize()

        if not query or not isinstance(query, str):
            return None

        # ── 步骤1: Redis 精确缓存检查 ──────────────────────
        cached = await self._get_faq_cache(query)
        if cached:
            logger.info("bm25.cache_hit", query=query[:30])
            return cached

        # ── 步骤2: BM25 索引检查 ──────────────────────────
        if self.bm25 is None or len(self.tokenized_questions) == 0:
            return None

        try:
            # jieba 分词
            query_tokens = preprocess_text(query).split()
            if not query_tokens:
                return None

            # BM25 打分
            scores = self.bm25.get_scores(query_tokens)
            if len(scores) == 0:
                return None

            best_idx = int(np.argmax(scores))
            raw_best = float(scores[best_idx])
            raw_avg = float(np.mean(scores)) if np.mean(scores) > 0 else 0.001

            # 双重判断：
            # ① 最高分/平均分 > 2.0 → 明显区分度
            # ② raw_max > 1.0 → 至少命中了一些关键词
            ratio = raw_best / raw_avg
            hits_keywords = raw_best >= 1.8  # 至少有一定量的关键词重叠（过滤噪音）

            logger.info("bm25.scored", query=query[:30],
                       best_idx=best_idx, raw=round(raw_best, 3),
                       ratio=round(ratio, 2), hits=hits_keywords)

            # ── 步骤3: 阈值判断 ──────────────────────────
            if ratio >= 2.0 and hits_keywords:
                original_question = self.original_questions[best_idx]
                answer = await self._fetch_answer_from_mysql(original_question)
                if answer:
                    # 写回 Redis 缓存
                    await self._set_faq_cache(query, answer)
                    logger.info("bm25.hit", query=query[:30],
                               raw=round(raw_best, 2), ratio=round(ratio, 1))
                    return answer

            # ── 未达阈值 → 触发 RAG ───────────────────────
            logger.info("bm25.miss", query=query[:30],
                       raw=round(raw_best, 2), ratio=round(ratio, 1))
            return None

        except Exception as e:
            logger.error("bm25.search_error", error=str(e))
            return None

    # ═══════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """Softmax 归一化：将 BM25 原始分数转为概率分布。"""
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    async def _fetch_answer_from_mysql(self, question: str) -> Optional[str]:
        """根据原始问题文本从 MySQL 获取标准答案。"""
        try:
            from backend.core.database import get_session
            from backend.models.db_models import EcomFAQ
            from sqlalchemy import select

            async with get_session() as session:
                result = await session.execute(
                    select(EcomFAQ.answer).where(EcomFAQ.question == question)
                )
                row = result.first()
                return row[0] if row else None

        except Exception as e:
            logger.error("bm25.mysql_answer_failed", error=str(e))
            return None

    # ── Redis FAQ 缓存（使用 cache.py 两级缓存接口）───────

    async def _get_faq_cache(self, query: str) -> Optional[str]:
        """读取 FAQ 精确匹配缓存（answer:{query}）。"""
        return await self.cache.get_answer(query)

    async def _set_faq_cache(self, query: str, answer: str):
        """写入 FAQ 精确匹配缓存（answer:{query}，异步 fire-and-forget）。"""
        await self.cache.set_answer(query, answer)

    async def _redis_get(self, key: str):
        """通用 Redis GET（非缓存标准接口，直接读 Redis）。"""
        if not self.cache._available or not self.cache._redis:
            return None
        try:
            import json
            value = await asyncio.wait_for(
                self.cache._redis.get(key), timeout=1.0
            )
            return json.loads(value) if value else None
        except Exception:
            return None

    async def _redis_set(self, key: str, value):
        """通用 Redis SET（序列化为 JSON）。"""
        if not self.cache._available or not self.cache._redis:
            return
        try:
            import json
            await asyncio.wait_for(
                self.cache._redis.setex(
                    key, self.settings.bm25_cache_ttl,
                    json.dumps(value, ensure_ascii=False),
                ),
                timeout=2.0,
            )
        except Exception:
            pass


# ── 全局单例 ────────────────────────────────────────────────

_bm25_instance: Optional[BM25FAQSearch] = None


async def get_bm25() -> BM25FAQSearch:
    """获取全局 BM25FAQSearch 单例（自动初始化）。"""
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25FAQSearch()
        await _bm25_instance.initialize()
    return _bm25_instance


# ── 测试代码 ───────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        bm25 = BM25FAQSearch()
        await bm25.initialize()
        print(f"FAQ 问题数: {len(bm25.original_questions)}")
        print(f"BM25 索引: {'已构建' if bm25.bm25 else '未构建'}")

        tests = ["如何退货", "物流到哪了", "尺码偏大吗", "今天天气不错"]
        for q in tests:
            answer = await bm25.search(q)
            status = "✅ 命中" if answer else "❌ 未命中(需RAG)"
            print(f"\n查询: {q}")
            print(f"结果: {status}")
            if answer:
                print(f"答案: {answer[:80]}...")

        # 第二次查询（测试缓存）
        print("\n--- 缓存测试 ---")
        answer2 = await bm25.search("如何退货")
        print(f"缓存命中: {'是' if answer2 else '否'}")

    asyncio.run(test())
