# backend/core/logger.py
# 全项目的日志工具：支持「事件名 + 键值对」的结构化日志写法。
# 用法：logger.info("user.login", user_id="u1", role="student")

import logging
import sys
from backend.config import get_settings


class _Logger:
    """日志包装类。
    作用：让我们能用 logger.info("事件名", key=value) 这种结构化写法，
    而不必每次手动拼字符串。内部委托给标准库 logging 输出。
    """

    def __init__(self, name: str):
        """name 通常传 __name__（当前模块名），日志里能看出是哪个模块打的。"""
        self._log = logging.getLogger(name)

    def _fmt(self, event: str, **kw) -> str:
        """内部方法：把「事件名 + 关键字参数」格式化成一行可读字符串。
        例：_fmt("user.login", user_id="u1") → "user.login | user_id='u1'"
        """
        if kw:
            return event + " | " + " ".join(f"{k}={v!r}" for k, v in kw.items())
        return event

    def debug(self, event: str, **kw):
        self._log.debug(self._fmt(event, **kw))

    def info(self, event: str, *args, **kw):
        if args:
            self._log.info(event, *args)
        else:
            self._log.info(self._fmt(event, **kw))

    def warning(self, event: str, *args, **kw):
        if args:
            self._log.warning(event, *args)
        else:
            self._log.warning(self._fmt(event, **kw))

    def error(self, event: str, **kw):
        exc_info = kw.pop("exc_info", False)
        self._log.error(self._fmt(event, **kw), exc_info=exc_info)

    def critical(self, event: str, **kw):
        self._log.critical(self._fmt(event, **kw))


def configure_logging() -> None:
    """全局日志配置：整个应用只在启动时（main.py）调用一次。"""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    # 压低第三方库的噪音日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> _Logger:
    """对外的工厂函数：每个模块用 get_logger(__name__) 拿到自己的日志器。"""
    return _Logger(name)
