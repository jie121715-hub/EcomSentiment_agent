# backend/core/exceptions.py
# 统一异常类层次，为三层兜底（retry.py）打基础。
# 异常分两类：可重试（网络抖动等）和不可重试（参数错误等），
# retry.py 据此决定要不要重试。


class EcomAgentBaseError(Exception):
    """所有自定义异常的基类。"""
    def __init__(self, message: str = "", details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# ── 可重试异常（网络/服务临时故障，重试可能恢复）──

class RetryableError(EcomAgentBaseError):
    """可重试异常的标记类。子类异常会被 with_retry 自动重试。"""
    pass


class LLMAPIError(RetryableError):
    """LLM API 调用失败（超时、限流、服务不可用等）。"""
    pass


class LLMTimeoutError(RetryableError):
    """LLM API 超时。"""
    pass


class LLMRateLimitError(RetryableError):
    """LLM API 限流。"""
    pass


class RAGRetrievalError(RetryableError):
    """RAG 向量检索失败（向量库连接失败等）。"""
    pass


# ── 不可重试异常（参数/认证/配置问题，重试也没用）──

class NonRetryableError(EcomAgentBaseError):
    """不可重试异常的标记类。子类异常不会重试，直接走降级。"""
    pass


class InvalidInputError(NonRetryableError):
    """用户输入不合法（空消息、超长文本等）。"""
    pass


class AuthenticationError(NonRetryableError):
    """API Key 无效或认证失败。"""
    pass


class ConfigurationError(NonRetryableError):
    """配置缺失或错误（如模型路径不存在、必填配置未设置）。"""
    pass


class ModelLoadError(NonRetryableError):
    """本地模型加载失败（文件不存在、格式错误等）。"""
    pass


class SentimentAnalysisError(EcomAgentBaseError):
    """情感分析出错。"""
    pass


class IntentClassificationError(EcomAgentBaseError):
    """意图分类出错。"""
    pass


# ── 工具函数 ──

def is_retryable(exception: Exception) -> bool:
    """判断异常是否应该重试。"""
    return isinstance(exception, RetryableError)
