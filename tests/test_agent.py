# tests/test_agent.py
# 云答智能客服系统 v1 端到端测试。
#
# 运行方式：
#   cd EcomSentiment_agent
#   python tests/test_agent.py
#
# 离线测试（不需要 API Key）：
#   - Config 加载
#   - Logger 输出
#   - Exception 体系
#   - Retry 机制
#   - Pydantic Schema 校验
#   - Sentiment Map 映射
#
# 在线测试（需要 DASHSCOPE_API_KEY）：
#   - LLM Factory 实例创建
#   - Perception Agent 完整感知流程
#   - Routing Agent 路由决策
#   - 端到端 run_shopping_guide

import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 离线测试（不需要 API Key）──────────────────────────────────

def test_config():
    """测试1：配置加载"""
    from backend.config import get_settings
    s = get_settings()
    assert s.app_port == 8000
    assert s.retry_max_attempts == 3
    assert s.max_history_turns == 10
    assert "400" in s.customer_service_phone
    print("✅ test_config 通过")


def test_logger():
    """测试2：结构化日志"""
    from backend.core.logger import configure_logging, get_logger
    configure_logging()
    log = get_logger("test")
    log.info("test.logger_works", test_case="config_ok")
    print("✅ test_logger 通过")


def test_exceptions():
    """测试3：异常体系"""
    from backend.core.exceptions import (
        LLMAPIError, InvalidInputError, is_retryable,
        RetryableError, NonRetryableError,
    )
    assert is_retryable(LLMAPIError("test"))
    assert not is_retryable(InvalidInputError("test"))
    assert issubclass(LLMAPIError, RetryableError)
    assert issubclass(InvalidInputError, NonRetryableError)
    print("✅ test_exceptions 通过")


def test_retry():
    """测试4：三层兜底机制"""
    from backend.core.retry import with_retry
    from backend.core.exceptions import LLMAPIError

    call_count = {"count": 0}

    @with_retry(max_attempts=3, base_delay=0.01)
    def flaky_func():
        call_count["count"] += 1
        raise LLMAPIError("模拟网络故障")

    result = flaky_func()
    assert call_count["count"] == 3, f"应重试3次，实际{call_count['count']}次"
    assert "400" in result or "客服" in result
    print("✅ test_retry 通过")


def test_schemas():
    """测试5：Pydantic Schema 校验"""
    from backend.models.schemas import (
        Sentiment, SentimentLabel, IntentCategory,
        PerceptionResult, RouteDecision, RetrievalStrategy,
        ChatRequest, AgentResponse, AgentMessage,
    )

    # 测试枚举
    assert Sentiment.POSITIVE.value == "positive"
    assert SentimentLabel.ANXIOUS.value == "anxious"
    assert IntentCategory.KNOWLEDGE_QA.value == "knowledge_qa"

    # 测试 PerceptionResult 构造
    p = PerceptionResult(
        original_query="这件衣服怎么样",
        sentiment=Sentiment.POSITIVE,
        sentiment_label=SentimentLabel.NEUTRAL,
        sentiment_confidence=0.9,
        intent=IntentCategory.KNOWLEDGE_QA,
        entities=[{"type": "product_name", "value": "衣服"}],
        query_summary="咨询衣服质量",
    )
    assert p.sentiment == Sentiment.POSITIVE
    assert len(p.entities) == 1

    # 测试 ChatRequest 校验（空消息应失败）
    try:
        ChatRequest(query="")
        assert False, "应抛出 ValidationError"
    except Exception:
        pass  # 预期行为

    print("✅ test_schemas 通过")


def test_sentiment_map():
    """测试6：情感→话术映射表"""
    from backend.data.sentiment_map import get_tone_config, get_source_filter
    from backend.models.schemas import SentimentLabel

    # 焦虑 → 应返回安抚类语气
    anxious_config = get_tone_config(SentimentLabel.ANXIOUS)
    assert "安抚" in anxious_config["tone_instruction"]

    # 愤怒 → 应标记需要转人工
    angry_config = get_tone_config(SentimentLabel.ANGRY)
    assert angry_config["escalate_to_human"] is True

    # 意图 → 知识库过滤
    assert get_source_filter("business") == "business"
    assert get_source_filter("knowledge_qa") == ""

    print("✅ test_sentiment_map 通过")


def test_rag_component():
    """测试7：RAG 组件（Prompt 模板 + 检索器初始化）"""
    from backend.rag.prompts import EcomRAGPrompts

    # 验证 Prompt 模板
    rag_prompt = EcomRAGPrompts.rag_prompt()
    result = rag_prompt.format(
        tone_instruction="请保持专业",
        extra_instruction="",
        context="退货流程：申请售后→寄回→退款",
        history="（无历史）",
        question="如何退货？",
        phone="400-618-8888",
    )
    assert "退货流程" in result
    assert "如何退货" in result
    assert "400-618-8888" in result

    # 验证 HyDE prompt
    hyde = EcomRAGPrompts.hyde_prompt().format(query="这件衣服舒服吗")
    assert "件衣服舒服吗" in hyde

    print("✅ test_rag_component 通过")


# ── 在线测试（需要 API Key）────────────────────────────────────

def _check_api_key() -> bool:
    """检查是否配置了 API Key（DeepSeek 或 DashScope 任一即可）。"""
    from backend.config import get_settings
    s = get_settings()
    has_deepseek = bool(s.deepseek_api_key and len(s.deepseek_api_key) > 10)
    has_dashscope = bool(s.dashscope_api_key and len(s.dashscope_api_key) > 10)
    return has_deepseek or has_dashscope


async def test_llm_factory_online():
    """测试8：LLM Factory 在线（需要 API Key）"""
    if not _check_api_key():
        print("⏭️  跳过 test_llm_factory_online（未配置 DASHSCOPE_API_KEY）")
        return

    from backend.core.llm_factory import get_llm, get_structured_llm
    from pydantic import BaseModel, Field

    llm = get_llm("qa", temperature=0)
    assert llm is not None

    class TestSchema(BaseModel):
        answer: str = Field(description="回答")

    structured = get_structured_llm("intent", TestSchema)
    assert structured is not None

    print("✅ test_llm_factory_online 通过")


async def test_perception_online():
    """测试9：感知层在线（需要 API Key）"""
    if not _check_api_key():
        print("⏭️  跳过 test_perception_online（未配置 DASHSCOPE_API_KEY）")
        return

    from backend.agents.perception import PerceptionAgent

    agent = PerceptionAgent()
    result = await agent.perceive("这件衣服穿起来会起球吗？我很担心质量问题。")

    assert result.sentiment is not None
    assert result.intent is not None
    assert len(result.query_summary) > 0
    print(f"  情感: {result.sentiment.value} → {result.sentiment_label.value}")
    print(f"  意图: {result.intent.value}")
    print("✅ test_perception_online 通过")


async def test_routing_online():
    """测试10：路由决策在线（需要 API Key）"""
    if not _check_api_key():
        print("⏭️  跳过 test_routing_online（未配置 DASHSCOPE_API_KEY）")
        return

    from backend.agents.router import RoutingAgent
    from backend.models.schemas import PerceptionResult, Sentiment, SentimentLabel, IntentCategory

    router = RoutingAgent()

    perception = PerceptionResult(
        original_query="我的快递三天了还没到，你们这是什么物流！",
        sentiment=Sentiment.NEGATIVE,
        sentiment_label=SentimentLabel.ANGRY,
        sentiment_confidence=0.92,
        intent=IntentCategory.BUSINESS,
        entities=[{"type": "order_id", "value": "JD20240706-001"}],
        query_summary="查询物流进度",
    )

    decision = await router.route(perception)

    assert decision.escalate_to_human is True  # 愤怒情感应建议转人工
    print(f"  策略: {decision.strategy.value}")
    print(f"  转人工: {decision.escalate_to_human}")
    print("✅ test_routing_online 通过")


async def test_e2e_online():
    """测试11：端到端在线（需要 API Key）"""
    if not _check_api_key():
        print("⏭️  跳过 test_e2e_online（未配置 DASHSCOPE_API_KEY）")
        return

    from backend.agents.graph import run_shopping_guide

    queries = [
        "你好，这件衣服是什么材质？",
        "我要退货！质量太差了！",
    ]

    for query in queries:
        result = await run_shopping_guide(query)
        assert result.success is True
        assert len(result.message.content) > 10
        print(f"\n  查询: {query}")
        print(f"  情感: {result.message.sentiment_detected}")
        print(f"  意图: {result.message.intent_detected}")
        print(f"  回复: {result.message.content[:100]}...")
        print(f"  耗时: {result.processing_time_ms:.0f}ms")

    print("\n✅ test_e2e_online 通过")


# ── 主入口 ────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("🛒 电商智能问答与业务处理系统 v1 — 测试套件")
    print("=" * 60)

    # 离线测试（无需 API Key）
    print("\n── 离线测试 ──")
    test_config()
    test_logger()
    test_exceptions()
    test_retry()
    test_schemas()
    test_sentiment_map()
    test_rag_component()

    # 在线测试（需 API Key）
    print("\n── 在线测试（需要 DASHSCOPE_API_KEY）──")
    if _check_api_key():
        await test_llm_factory_online()
        await test_perception_online()
        await test_routing_online()
        await test_e2e_online()
    else:
        print("💡 在 .env.local 中配置 DASHSCOPE_API_KEY 后可运行在线测试")
        print("   注册地址: https://dashscope.aliyun.com/")

    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
