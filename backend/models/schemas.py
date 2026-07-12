# backend/models/schemas.py
# 全项目所有 Pydantic 数据模型（Schema）的统一定义文件。
# 设计原则：
#   - 所有 Agent 之间传递的数据都是强类型 Pydantic 对象
#   - 用 Enum 约束枚举值，杜绝拼写错误
#   - 每个字段都有 description，帮助 LLM 理解结构化输出要求

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# 枚举定义：杜绝拼写错误，所有标签都用 Enum 约束
# ═══════════════════════════════════════════════════════════════

class Sentiment(str, Enum):
    """情感极性（从你的 BERT 二分类模型输出映射而来）。"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentLabel(str, Enum):
    """细粒度情感标签（LLM 二次细分）。"""
    HAPPY = "happy"                    # 开心满意
    GRATEFUL = "grateful"             # 感谢
    NEUTRAL = "neutral"               # 中性/信息查询
    CONFUSED = "confused"             # 困惑
    ANXIOUS = "anxious"               # 焦虑担忧
    ANGRY = "angry"                   # 愤怒不满
    DISAPPOINTED = "disappointed"     # 失望


class IntentCategory(str, Enum):
    """用户意图分类 — 4类，与4个分发Agent一一对应。

    BERT 10分类输出经映射层聚合为4类：
      product_inquiry/price_inquiry/recommend_request/chitchat/other → knowledge_qa
      order_tracking/modify_order/after_sales(操作类) → business
      knowledge_mgmt → knowledge_mgmt
      complaint(无订单号) → escalate
    """
    KNOWLEDGE_QA = "knowledge_qa"         # 知识问答（商品咨询/价格/推荐/闲聊/政策）
    BUSINESS = "business"                 # 业务处理（查物流/改订单/退款/售后操作）
    KNOWLEDGE_MGMT = "knowledge_mgmt"     # 知识管理（商户录入/查看/删除知识）
    ESCALATE = "escalate"                # 工单处理（投诉/愤怒情绪/高危问题）


class RetrievalStrategy(str, Enum):
    """RAG 检索策略。"""
    DIRECT = "直接检索"
    HYDE = "假设文档检索"
    SUBQUERY = "子查询检索"
    BACKTRACK = "回溯问题检索"


# ═══════════════════════════════════════════════════════════════
# 感知层 (Perception Layer) 输出
# ═══════════════════════════════════════════════════════════════

class PerceptionResult(BaseModel):
    """感知层统一输出：情感 + 意图 + 实体 + 摘要。

    这个对象是「感知层」交给「决策层」的通行证。
    """
    original_query: str = Field(description="用户原始输入（保留原文）")
    sentiment: Sentiment = Field(description="情感极性：positive/negative/neutral")
    sentiment_label: SentimentLabel = Field(description="细粒度情感标签")
    sentiment_confidence: float = Field(ge=0, le=1, description="情感分类置信度")
    intent: IntentCategory = Field(description="用户核心意图")
    intent_confidence: float = Field(default=0.85, ge=0, le=1, description="意图分类置信度")
    entities: list[dict[str, str]] = Field(
        default_factory=list,
        description="提取的实体列表，如 [{'type': 'SKU', 'value': 'ABC123'}, {'type': 'order_id', 'value': '20240706-001'}]"
    )
    query_summary: str = Field(default="", description="用户问题的简短摘要（用于检索优化）")


# ═══════════════════════════════════════════════════════════════
# 决策层 (Decision & Routing Layer) 输出
# ═══════════════════════════════════════════════════════════════

class TargetAgent(str, Enum):
    """router 分发的目标 Agent（v3 裁剪后：9→4 目标）。"""
    KNOWLEDGE_QA = "knowledge_qa"        # 知识应答 → KnowledgeQAAgent (RAG+LLM，含闲聊兜底)
    BUSINESS = "business"                # 业务操作 → BusinessAgent (MySQL读写合一，含确认流程)
    KNOWLEDGE_MGMT = "knowledge_mgmt"    # 知识管理 → KnowledgeMgmtAgent (双写)
    ESCALATE = "escalate"                # 转人工


class UrgencyLevel(str, Enum):
    """情绪紧急度。"""
    NORMAL = "normal"        # 正常处理
    ELEVATED = "elevated"    # 优先处理（焦虑/困惑）
    CRITICAL = "critical"    # 紧急接管（愤怒+投诉/退款 → 直接转人工）


class RouteDecision(BaseModel):
    """决策层输出：router.py 三维决策（意图+状态+情绪）的统一结果。

    这是整个系统的"大脑"输出 —— 决定了：
    1. 分发给哪个 Agent
    2. 是否需要先反问澄清
    3. 是否触发情绪紧急接管
    4. 用什么策略、什么语气来回复
    """
    # ── 分发决策 ──
    target_agent: TargetAgent = Field(
        default=TargetAgent.KNOWLEDGE_QA,
        description="router 决定分发给哪个 Agent 处理"
    )
    needs_clarification: bool = Field(
        default=False,
        description="意图置信度不足，需要反问澄清后再处理"
    )
    clarification_question: str = Field(
        default="",
        description="反问澄清的具体问题（由 router._build_clarify_decision 直接生成）"
    )

    # ── 情绪紧急度 ──
    urgency: UrgencyLevel = Field(
        default=UrgencyLevel.NORMAL,
        description="情绪紧急度：normal/elevated/critical"
    )
    escalate_to_human: bool = Field(
        default=False,
        description="是否转人工（critical 级别或投诉类）"
    )
    escalate_reason: str = Field(
        default="",
        description="转人工原因"
    )

    # ── 检索 & 语气（KnowledgeQA 路径使用）──
    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.DIRECT,
        description="RAG 检索策略"
    )
    tone_instruction: str = Field(
        default="请保持专业、友好的客服语气。",
        description="注入 LLM 的语气指令（由情感映射生成）"
    )
    dynamic_prompt_extra: str = Field(
        default="",
        description="根据情感+意图动态拼接的额外指令"
    )
    source_filter: str = Field(
        default="",
        description="知识库过滤类别：products/orders/after_sales/logistics/promotion"
    )
    skip_rag: bool = Field(
        default=False,
        description="是否跳过 RAG 检索"
    )


class ClarifyResult(BaseModel):
    """反问澄清结果。"""
    original_query: str = Field(description="用户原始输入")
    detected_intent: str = Field(description="检测到的可能意图（低置信度）")
    confidence: float = Field(description="当前置信度")
    questions: list[str] = Field(description="2-3个反问选项")
    clarification_log: str = Field(default="", description="澄清日志（用于模型优化）")


# ═══════════════════════════════════════════════════════════════
# 执行层 (Execution & Generation Layer) 输入/输出
# ═══════════════════════════════════════════════════════════════

class FunctionCall(BaseModel):
    """Function Calling 定义。"""
    name: str = Field(description="要调用的函数名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="函数参数")


class FunctionCallResult(BaseModel):
    """Function Calling 返回结果。"""
    call: FunctionCall = Field(description="原始调用")
    success: bool = Field(default=False, description="是否执行成功")
    result: Any = Field(default=None, description="执行结果")
    error: str = Field(default="", description="错误信息")


class ConversationHistory(BaseModel):
    """一轮对话记录。"""
    question: str = Field(description="用户问题")
    answer: str = Field(description="系统回答")
    sentiment: str = Field(default="", description="该轮情感")
    intent: str = Field(default="", description="该轮意图")


# ═══════════════════════════════════════════════════════════════
# API 层：请求 / 响应
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """POST /chat 请求体。"""
    query: str = Field(min_length=1, max_length=2000, description="用户输入")
    user_id: str = Field(default="anonymous", description="用户标识（用于记忆隔离）")
    session_id: str = Field(default="default", description="会话标识")
    history: list[ConversationHistory] = Field(
        default_factory=list,
        description="对话历史（最多 10 轮）"
    )


class AgentMessage(BaseModel):
    """Agent 返回给用户的单条消息。"""
    role: str = Field(default="assistant", description="角色")
    content: str = Field(description="回复内容")
    sentiment_detected: str = Field(default="", description="检测到的情感（透明化展示）")
    intent_detected: str = Field(default="", description="检测到的意图")
    function_calls: list[FunctionCallResult] = Field(
        default_factory=list,
        description="触发的 Function Call 结果"
    )


class AgentResponse(BaseModel):
    """Agent 完整响应（非流式）。"""
    success: bool = Field(default=True)
    message: AgentMessage = Field(description="回复消息")
    history_updated: list[ConversationHistory] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0, description="处理耗时（毫秒）")


class ChatEvent(BaseModel):
    """SSE 流式输出事件。"""
    event: str = Field(description="事件类型：token|perception|route|function_call|done|error")
    data: str = Field(description="事件携带的数据")
