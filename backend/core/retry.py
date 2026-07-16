# backend/core/retry.py
# 三层兜底与重试机制 —— 全项目统一容错入口。
# 三层结构：① 自动重试（指数退避）→ ② Agent 降级 → ③ 系统兜底
# 用法：@with_retry(max_attempts=3) 或 await with_retry_async(fn, *args)

import asyncio
import functools
import random
import time
from typing import Callable, TypeVar, Any

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import is_retryable, YunDaBaseError

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── 系统兜底文案 ──
_SYSTEM_FALLBACK_MSG = (
    "非常抱歉，系统当前遇到了一些问题，暂时无法为您提供准确的回复。"
    "建议您稍后再试，或拨打客服电话 {phone} 联系人工客服获取帮助。"
)


def _build_fallback_msg() -> str:
    settings = get_settings()
    return _SYSTEM_FALLBACK_MSG.format(phone=settings.customer_service_phone)


def _calc_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """计算指数退避延迟（加随机抖动，避免惊群效应）。
    attempt: 当前重试次数（从 1 开始）
    """
    delay = base_delay * (2 ** (attempt - 1))
    delay = min(delay, max_delay)
    jitter = delay * 0.1 * random.random()
    return delay + jitter


# ── 同步版：with_retry 装饰器 ──

def with_retry(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    fallback_msg: str | None = None,
):
    """三层兜底装饰器（同步版）。

    第一层：自动重试（仅重试 RetryableError 类异常）
    第二层：Agent 降级（重试耗尽 → 返回降级提示）
    第三层：系统兜底（所有异常 → 返回友好文案，绝不抛异常出去）
    """
    settings = get_settings()
    _max_attempts = max_attempts or settings.retry_max_attempts
    _base_delay = base_delay or settings.retry_base_delay
    _max_delay = max_delay or settings.retry_max_delay
    _fallback = fallback_msg or _build_fallback_msg()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error: Exception | None = None

            # 第一层：带指数退避的自动重试
            for attempt in range(1, _max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not is_retryable(e):
                        # 不可重试异常，直接跳到降级
                        logger.warning(
                            "retry.non_retryable_skip",
                            func=func.__name__,
                            error_type=type(e).__name__,
                            error=str(e)[:200],
                        )
                        break

                    if attempt < _max_attempts:
                        delay = _calc_delay(attempt, _base_delay, _max_delay)
                        logger.warning(
                            "retry.attempt_failed",
                            func=func.__name__,
                            attempt=f"{attempt}/{_max_attempts}",
                            delay=f"{delay:.2f}s",
                            error=str(e)[:100],
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "retry.all_attempts_exhausted",
                            func=func.__name__,
                            attempts=_max_attempts,
                            error=str(e)[:200],
                        )

            # 第二层：Agent 降级 —— 返回友好的降级信息
            logger.error(
                "retry.agent_degraded",
                func=func.__name__,
                last_error_type=type(last_error).__name__ if last_error else "unknown",
            )

            # 第三层：系统兜底 —— 返回系统容错文案
            return _fallback

        return wrapper  # type: ignore[return-value]

    return decorator


# ── 异步版：with_retry_async 函数 ──

async def with_retry_async(
    fn: Callable[..., Any],
    *args,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    fallback_msg: str | None = None,
    **kwargs,
) -> Any:
    """三层兜底（异步版）：包裹任意异步函数，提供三层容错保护。

    用法：
        result = await with_retry_async(llm.ainvoke, messages, max_attempts=3)
    """
    settings = get_settings()
    _max_attempts = max_attempts or settings.retry_max_attempts
    _base_delay = base_delay or settings.retry_base_delay
    _max_delay = max_delay or settings.retry_max_delay
    _fallback = fallback_msg or _build_fallback_msg()

    last_error: Exception | None = None

    # 第一层：带指数退避的自动重试
    for attempt in range(1, _max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if not is_retryable(e):
                logger.warning(
                    "retry_async.non_retryable_skip",
                    func=getattr(fn, "__name__", str(fn)),
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )
                break

            if attempt < _max_attempts:
                delay = _calc_delay(attempt, _base_delay, _max_delay)
                logger.warning(
                    "retry_async.attempt_failed",
                    func=getattr(fn, "__name__", str(fn)),
                    attempt=f"{attempt}/{_max_attempts}",
                    delay=f"{delay:.2f}s",
                    error=str(e)[:100],
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "retry_async.all_attempts_exhausted",
                    func=getattr(fn, "__name__", str(fn)),
                    attempts=_max_attempts,
                    error=str(e)[:200],
                )

    # 第二层 + 第三层：降级 → 系统兜底
    logger.error(
        "retry_async.agent_degraded",
        func=getattr(fn, "__name__", str(fn)),
        last_error_type=type(last_error).__name__ if last_error else "unknown",
    )
    return _fallback


# ── 测试代码 ──
if __name__ == "__main__":
    from backend.core.logger import configure_logging
    configure_logging()

    # 测试1：同步装饰器 —— 模拟可重试异常
    print("=" * 60)
    print("测试1: 同步版 with_retry（模拟 3 次重试后降级）")
    print("=" * 60)

    call_count = {"count": 0}

    @with_retry(max_attempts=3, base_delay=0.1)
    def test_sync_flaky():
        call_count["count"] += 1
        from backend.core.exceptions import LLMAPIError
        raise LLMAPIError(f"模拟 API 故障 #{call_count['count']}")

    result = test_sync_flaky()
    print(f"调用次数: {call_count['count']}")
    print(f"兜底结果: {result[:60]}...")

    # 测试2：异步版 —— 成功场景
    print("\n" + "=" * 60)
    print("测试2: 异步版 with_retry_async（模拟成功）")
    print("=" * 60)

    async def test_async():
        async def success_fn():
            return "✅ 请求成功"
        r = await with_retry_async(success_fn, max_attempts=3)
        print(f"结果: {r}")

    asyncio.run(test_async())

    print("\n✅ retry.py 自测通过")
