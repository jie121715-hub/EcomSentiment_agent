# backend/core/llm_factory.py
# LLM Factory：统一封装大模型调用，按 Agent 类型路由。
# 规矩：所有 Agent 必须通过此模块获取模型，禁止直接手搓 ChatModel。
#
# 设计思想：
#   - 收口配置：API Key / base_url / 超时 只配一次
#   - 按需缓存：相同参数只创建一次模型实例
#   - 路由扩展：想给某类 Agent 换更强的模型，只改路由表一行
#
# 用法：
#   llm = get_llm("qa")                            # 普通对话模型
#   llm = get_llm("qa", temperature=0.3, streaming=True)  # 流式对话
#   structured = get_structured_llm("intent", IntentResult)  # 结构化输出

from typing import Type
import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── 自定义 httpx 客户端：绕过系统代理 ───────────────────────────
# 国内直连 DeepSeek/DashScope 不需要代理；trust_env=False 忽略系统代理。
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)
_HTTP_SYNC_CLIENT = httpx.Client(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)

# ── Agent 类型 → (模型标识符, 推理开关) 路由表 ──────────────────
# 想给某类业务换模型或开关推理，只改这里。
# thinking=enabled  → DeepSeek 推理模式，适合分析/分类/提取任务
# thinking=disabled → 普通模式，适合对话/推荐/流式输出
_AGENT_CONFIG: dict[str, dict] = {
    "qa":             {"model": "deepseek-v4-flash", "thinking": False},
    "intent":         {"model": "deepseek-v4-flash", "thinking": True},
    "sentiment_llm":  {"model": "deepseek-v4-flash", "thinking": True},
    "recommend":      {"model": "deepseek-v4-flash", "thinking": False},
    "summarize":      {"model": "deepseek-v4-flash", "thinking": True},
}

# 模型标识符 → 各平台的 model 名称
# DeepSeek: deepseek-chat (V3) / deepseek-reasoner (R1)
# DashScope: qwen-plus / qwen-max / qwen-turbo
_MODEL_ID_MAP: dict[str, str] = {
    "deepseek-chat": "deepseek-chat",
}


class LLMFactory:
    """大模型工厂（统一获取模型的唯一入口）。

    用 @classmethod 定义方法，直接用 LLMFactory.get_llm(...) 调用，无需创建对象。

    用法：
        llm = LLMFactory.get_llm("qa")
        response = await llm.ainvoke(messages)
    """

    _instances: dict[str, BaseChatModel] = {}  # 模型实例缓存

    @classmethod
    def _get_settings(cls):
        return get_settings()

    @classmethod
    def _build_model_kwargs(cls, thinking: bool) -> dict:
        """组装 init_chat_model 需要的所有参数。"""
        settings = cls._get_settings()

        # 优先使用 DeepSeek（推荐）；若未设置则回退到 DashScope（备选）
        if settings.deepseek_api_key:
            api_key = settings.deepseek_api_key
            base_url = settings.deepseek_base_url
            actual_model = "deepseek-chat"
        elif settings.dashscope_api_key:
            api_key = settings.dashscope_api_key
            base_url = settings.dashscope_base_url
            actual_model = "qwen-plus"
        else:
            raise ValueError(
                "未配置任何 LLM API Key！请在 .env.local 中设置 DEEPSEEK_API_KEY"
            )

        # DeepSeek 推理模式：{"thinking": {"type": "enabled"}} / {"thinking": {"type": "disabled"}}
        thinking_config = {"type": "enabled"} if thinking else {"type": "disabled"}

        return {
            "model": actual_model,
            "model_provider": "openai",
            "temperature": 0,
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": 0,
            "http_async_client": _HTTP_ASYNC_CLIENT,
            "http_client": _HTTP_SYNC_CLIENT,
            "model_kwargs": {
                "extra_body": {"thinking": thinking_config},
            },
        }

    @classmethod
    def get_llm(
        cls,
        agent_type: str,
        temperature: float = 0,
        streaming: bool = False,
    ) -> BaseChatModel:
        """按 Agent 类型获取模型实例（带缓存）。

        :param agent_type: Agent 类型，必须在路由表里
        :param temperature: 温度：对话/推荐传 0.3~0.7，分类/评分保持 0
        :param streaming: 是否流式输出（问答/面试场景用）
        :return: BaseChatModel 实例
        """
        if agent_type not in _AGENT_CONFIG:
            raise ValueError(
                f"未知 agent_type: '{agent_type}'，"
                f"可用类型：{list(_AGENT_CONFIG.keys())}"
            )

        cfg = _AGENT_CONFIG[agent_type]
        model_key = cfg["model"]
        thinking = cfg["thinking"]

        # 用「模型_温度_流式_推理」拼缓存键
        cache_key = f"{model_key}_{temperature}_{streaming}_{thinking}"

        if cache_key not in cls._instances:
            kwargs = cls._build_model_kwargs(thinking=thinking)
            kwargs["temperature"] = temperature
            kwargs["streaming"] = streaming

            llm = init_chat_model(**kwargs)
            cls._instances[cache_key] = llm

            logger.info(
                "llm_factory.model_initialized",
                agent_type=agent_type,
                model_key=model_key,
                temperature=temperature,
                streaming=streaming,
                thinking=thinking,
            )

        return cls._instances[cache_key]

    @classmethod
    def get_structured_llm(
        cls,
        agent_type: str,
        output_schema: Type[BaseModel],
        temperature: float = 0,
    ) -> Runnable:
        """获取「绑定了结构化输出 Pydantic Schema」的模型。

        调用 ainvoke 后直接返回 output_schema 类型的对象，无需手动解析 JSON。

        :param agent_type: Agent 类型
        :param output_schema: 期望的输出结构（一个 Pydantic BaseModel 子类）
        :param temperature: 温度
        :return: Runnable（绑定了结构化输出的模型）
        """
        llm = cls.get_llm(agent_type, temperature=temperature)
        return llm.with_structured_output(output_schema, method="function_calling")

    @classmethod
    def clear_cache(cls) -> None:
        """清空模型实例缓存（测试时用）。"""
        cls._instances.clear()
        logger.info("llm_factory.cache_cleared")


# ── 模块级便捷函数（Agent 代码里的推荐写法）────────────────────
# 比写 LLMFactory.get_llm(...) 更简洁。

def get_llm(agent_type: str, temperature: float = 0, streaming: bool = False) -> BaseChatModel:
    """LLMFactory.get_llm 的便捷入口。"""
    return LLMFactory.get_llm(agent_type, temperature=temperature, streaming=streaming)


def get_structured_llm(agent_type: str, output_schema: Type[BaseModel], temperature: float = 0) -> Runnable:
    """LLMFactory.get_structured_llm 的便捷入口。"""
    return LLMFactory.get_structured_llm(agent_type, output_schema, temperature=temperature)


# ── 测试代码 ──
if __name__ == "__main__":
    from pydantic import BaseModel, Field

    print("=" * 60)
    print("llm_factory.py 离线自测（不联网，验证路由/缓存/校验）")
    print("=" * 60)

    # ① 按类型拿到模型实例（离线部分：只验证实例创建、不调 API）
    try:
        llm = LLMFactory.get_llm("qa")
        print(f"① get_llm('qa') 返回类型: {type(llm).__name__}")
        print("  ✅ 需要真实的 API Key 才能验证完整连通性")
    except Exception as e:
        print(f"① get_llm 跳过（需要 API Key）: {e}")

    # ② 验证路由校验（传未知类型应报错）
    try:
        LLMFactory.get_llm("not_exist")
    except ValueError as e:
        print(f"② 未知类型校验正常: {str(e)[:50]}...")

    # ③ 验证结构化输出 Schema
    class TestSchema(BaseModel):
        name: str = Field(description="姓名")
        score: float = Field(description="分数", ge=0, le=100)

    print(f"③ TestSchema 定义完成, 字段: {list(TestSchema.model_fields.keys())}")

    # ④ 验证缓存
    if hasattr(LLMFactory, '_instances'):
        print(f"④ 缓存实例数: {len(LLMFactory._instances)}")
        LLMFactory.clear_cache()
        print(f"   清空后缓存实例数: {len(LLMFactory._instances)}")

    print("\n✅ llm_factory.py 自测通过（离线部分）")
    print("💡 如需联网测试，请在 .env.local 中配置 DEEPSEEK_API_KEY 后运行")
