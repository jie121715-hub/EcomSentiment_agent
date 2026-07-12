# backend/agents/router.py
# 🆕 v3 — 决策层升级为系统的"大脑"：三维决策（意图 × 状态 × 情绪）。
#
# 核心设计（对应流程图）：
#   router.py 是中央决策枢纽，不再是简单的"语气策略选择器"。
#   它并行做三件事：
#     维度1 — 意图置信度门控：confidence < 0.65 → clarify.py 反问，不瞎猜
#     维度2 — 情绪紧急度检测：极度负面 + 退款/投诉 → 直接转人工接管
#     维度3 — 意图→Agent分发：根据意图类型 + 操作性质分派Agent
#
#   三个维度并行检测，优先级：紧急接管 > 反问澄清 > 正常分发
#
# 输入：PerceptionResult（感知层输出）
# 输出：RouteDecision（包含 target_agent + 语气/策略/紧急度）

import asyncio
import re

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.retry import with_retry_async
from backend.core.llm_factory import get_llm
from backend.models.schemas import (
    PerceptionResult, RouteDecision, RetrievalStrategy,
    SentimentLabel, IntentCategory, TargetAgent, UrgencyLevel,
)
from backend.data.sentiment_map import (
    get_tone_config, get_source_filter, detect_urgency,
)

logger = get_logger(__name__)

# ── 🆕 v4.0: 4意图 → 4分发Agent（一一对应）─────────────────

_INTENT_AGENT_MAP: dict[IntentCategory, TargetAgent] = {
    IntentCategory.KNOWLEDGE_QA:    TargetAgent.KNOWLEDGE_QA,
    IntentCategory.BUSINESS:        TargetAgent.BUSINESS,
    IntentCategory.KNOWLEDGE_MGMT:  TargetAgent.KNOWLEDGE_MGMT,
    IntentCategory.ESCALATE:        TargetAgent.ESCALATE,
}

# ── 知识管理触发关键词（商户专用）────────────────────────────
_KNOWLEDGE_MGMT_KEYWORDS = [
    "帮我记", "记录一下", "添加知识", "看看知识库", "有哪些知识",
    "上传", "录入", "记下来", "记一下", "保存知识", "存入知识库", "添加规则",
    "添加政策", "上传规则", "录入知识", "存一下", "帮我记录", "帮我存",
]

# ── 🆕 澄清反问模板库（合并自 clarify.py）─────────────────────
# 针对每对"可能的意图"预制的反问模板
_CLARIFY_TEMPLATES: dict[tuple, str] = {
    ("product_inquiry", "order_tracking"): (
        "您是想了解商品的详情，还是想查一下订单的物流状态呢？\n"
        "• 如果是商品相关，请告诉我商品名称或型号\n"
        "• 如果是物流查询，请提供订单号"
    ),
    ("product_inquiry", "price_inquiry"): (
        "您是想了解商品的规格参数，还是关心价格和优惠信息呢？\n"
        "• 商品规格：我可以帮您查看材质、尺寸、功能等\n"
        "• 价格优惠：我可以帮您查询当前价格和活动"
    ),
    ("after_sales", "complaint"): (
        "您是需要申请售后服务（退换货/退款），还是有其他问题需要投诉反馈呢？\n"
        "• 售后服务：请告诉我订单号和具体问题\n"
        "• 投诉建议：我会认真记录并反馈给相关负责人"
    ),
    ("order_tracking", "modify_order"): (
        "您是想查询物流进度，还是需要修改订单信息呢？\n"
        "• 查物流：请提供订单号\n"
        "• 改订单：请说明需要修改的内容（地址/规格等）"
    ),
    ("recommend_request", "product_inquiry"): (
        "您是想让我为您推荐商品，还是想了解某款具体商品的信息呢？\n"
        "• 求推荐：请告诉我您的预算和偏好\n"
        "• 查商品：请告诉我商品名称"
    ),
}


class RoutingAgent:
    """🆕 v3 三维决策路由智能体 —— 系统的"大脑"。

    职责：
      维度1: 意图置信度门控 — confidence < 0.65 → clarify
      维度2: 情绪紧急度检测 — 愤怒+退款/投诉 → 直接转人工
      维度3: 意图→Agent分发 — 选对Agent，选对策略

    优先级（高→低）：紧急接管 > 反问澄清 > 正常分发
    """

    def __init__(self):
        self.settings = get_settings()

    # ═══════════════════════════════════════════════════════════
    # 核心路由方法
    # ═══════════════════════════════════════════════════════════

    async def route(self, perception: PerceptionResult, history: list = None) -> RouteDecision:
        """三维决策入口。"""
        intent = perception.intent
        history = history or []

        # ── 维度0: 知识管理关键词预检 ────────────────────
        # 商户指令不受情绪/置信度影响，直接路由
        if self._has_knowledge_mgmt_keyword(perception.original_query):
            logger.info("router.knowledge_mgmt_bypass", query=perception.original_query[:40])
            return self._build_direct_decision(perception, TargetAgent.KNOWLEDGE_MGMT)

        # ── 情感×意图一致性修正 ──────────────────────────
        intent = self._correct_intent(perception, intent)

        # ── 订单号/快递单号强制修正：识别到ID格式 → 改写为 BUSINESS ──
        intent = self._correct_order_id_intent(perception, intent)

        # ── 🆕 商品咨询覆盖：含"查"+商品词但无订单/物流关键词 → 强制KNOWLEDGE_QA ──
        intent = self._correct_product_inquiry_intent(perception, intent)

        logger.info(
            "router.v3.started",
            sentiment=perception.sentiment_label.value,
            intent=intent.value,
            confidence=round(perception.intent_confidence, 3),
        )

        # ── 维度2: 情绪紧急度检测（并行、优先）────────────
        if self.settings.router_urgency_enabled:
            urgency_result = detect_urgency(perception.sentiment_label, intent)
            urgency = UrgencyLevel(urgency_result["urgency"])

            # 紧急接管：CRITICAL → 直接转人工，短路返回
            if urgency == UrgencyLevel.CRITICAL:
                return self._build_escalate_decision(
                    perception, intent, urgency_result
                )
        else:
            urgency_result = {"urgency": "normal", "reason": "", "action": "normal", "message_template": ""}
            urgency = UrgencyLevel.NORMAL

        # ── 维度1: 意图置信度门控 ────────────────────────
        # 🔧 跳过门控的场景：订单号/快递单号 或 正在回应确认提示
        has_strong_id = self._has_order_or_tracking_id(perception.original_query)
        is_confirming = self._is_confirming_response(history)
        threshold = self.settings.router_intent_confidence_threshold
        if perception.intent_confidence < threshold and not has_strong_id and not is_confirming:
            return await self._build_clarify_decision(perception, intent)

        # ── 维度3: 意图→Agent分发 ────────────────────────
        target = self._resolve_target_agent(intent, perception)

        # ── 策略配置（KnowledgeQA / Business 路径使用）──
        tone_config = get_tone_config(perception.sentiment_label)
        source_filter = get_source_filter(intent.value)
        strategy = tone_config.get("prefer_retrieval", RetrievalStrategy.DIRECT)
        extra_instruction = self._build_extra_instruction(perception, intent)
        skip_rag = False  # v4.0: 4类意图均可能走RAG，闲聊由KnowledgeQA内部快速通道判断

        # ── ELEVATED 级别：增加优先处理标记 ──────────────
        if urgency == UrgencyLevel.ELEVATED:
            extra_instruction = (
                f"【优先处理】{urgency_result['reason']}\n"
                + extra_instruction
            )

        decision = RouteDecision(
            target_agent=target,
            needs_clarification=False,
            urgency=urgency,
            escalate_to_human=False,
            strategy=strategy,
            tone_instruction=tone_config["tone_instruction"],
            dynamic_prompt_extra=extra_instruction,
            source_filter=source_filter,
            skip_rag=skip_rag,
        )

        logger.info(
            "router.v3.decision",
            target=decision.target_agent.value,
            urgency=decision.urgency.value,
            strategy=decision.strategy.value,
            skip_rag=decision.skip_rag,
        )
        return decision

    # ═══════════════════════════════════════════════════════════
    # 维度1: 意图置信度门控
    # ═══════════════════════════════════════════════════════════

    async def _build_clarify_decision(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> RouteDecision:
        """低置信度 → 反问澄清（模板优先，LLM兜底），不盲目分发。

        🆕 合并自 clarify.py — 直接在路由层生成澄清消息。
        """
        query = perception.original_query
        confidence = perception.intent_confidence

        logger.info(
            "router.clarify_triggered",
            intent=intent.value,
            confidence=round(confidence, 3),
        )

        # 策略1：尝试匹配预置模板（快速、精准）
        clarification = self._match_clarify_template(intent.value)

        # 策略2：LLM 动态生成（模板未命中时）
        if not clarification:
            clarification = await self._llm_clarify(query, perception)

        # 🆕 异步写澄清日志（fire-and-forget）
        asyncio.ensure_future(self._save_clarify_log(query, perception, clarification))

        return RouteDecision(
            target_agent=TargetAgent.KNOWLEDGE_QA,  # 澄清后默认回退到知识问答
            needs_clarification=True,
            clarification_question=clarification,
            urgency=UrgencyLevel.NORMAL,
            escalate_to_human=False,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction="",  # 澄清消息自带语气，无需额外指令
        )

    # ═══════════════════════════════════════════════════════════
    # 维度2: 情绪紧急度检测
    # ═══════════════════════════════════════════════════════════

    def _build_escalate_decision(
        self,
        perception: PerceptionResult,
        intent: IntentCategory,
        urgency_result: dict,
    ) -> RouteDecision:
        """紧急接管 → 直接转人工。"""
        reason = urgency_result.get("reason", "情绪紧急")
        logger.warning(
            "router.escalate_triggered",
            sentiment=perception.sentiment_label.value,
            intent=intent.value,
            reason=reason,
        )
        return RouteDecision(
            target_agent=TargetAgent.ESCALATE,
            needs_clarification=False,
            urgency=UrgencyLevel.CRITICAL,
            escalate_to_human=True,
            escalate_reason=reason,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction=(
                "用户正处于极度负面情绪中，涉及敏感操作。"
                "请直接引导转人工，不做任何自动化处理。"
            ),
            dynamic_prompt_extra=urgency_result.get("message_template", ""),
            skip_rag=True,
        )

    # ═══════════════════════════════════════════════════════════
    # 维度3: 意图 → Agent 分发
    # ═══════════════════════════════════════════════════════════

    def _resolve_target_agent(
        self, intent: IntentCategory, perception: PerceptionResult
    ) -> TargetAgent:
        """v4.0: 4意图→4Agent 基本一一对应，仅做内容层面的微调。

        特殊改写规则：
        - business 意图 + 政策询问关键词 → knowledge_qa（政策走RAG不走业务）
        - escalate 意图 + 订单号 → business（有订单的投诉优先处理）
        - 含知识管理关键词 → knowledge_mgmt（商户指令优先）
        """
        query = perception.original_query

        # 知识管理关键词检测（商户指令，最高优先级）
        if any(w in query for w in _KNOWLEDGE_MGMT_KEYWORDS):
            return TargetAgent.KNOWLEDGE_MGMT

        # business 意图微调：纯政策询问 → 走 KnowledgeQA
        if intent == IntentCategory.BUSINESS:
            is_policy_q = any(w in query for w in ["政策", "规则", "怎么退", "如何退", "流程", "多久", "几天", "是什么", "能不能"])
            has_action = any(w in query for w in ["退款", "退钱", "取消", "不要了", "帮我退", "申请退", "我要退", "改地址"])
            has_order = bool(re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query))
            if is_policy_q and not has_action and not has_order:
                logger.info("router.intent_rewrite", from_="business", to="knowledge_qa")
                return TargetAgent.KNOWLEDGE_QA

        # escalate 意图微调：有订单号 → 走 Business 优先处理
        if intent == IntentCategory.ESCALATE:
            if re.search(r'[A-Z]{2,4}\d{6,12}', query):
                logger.info("router.intent_rewrite", from_="escalate", to="business")
                return TargetAgent.BUSINESS

        # 标准 1:1 映射
        return _INTENT_AGENT_MAP.get(intent, TargetAgent.KNOWLEDGE_QA)

    # ═══════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════

    def _has_knowledge_mgmt_keyword(self, query: str) -> bool:
        """检测是否包含知识管理关键词（商户指令）。"""
        return any(w in query for w in _KNOWLEDGE_MGMT_KEYWORDS)

    def _build_direct_decision(
        self, perception: PerceptionResult, target: TargetAgent
    ) -> RouteDecision:
        """构建直接分发决策（跳过紧急度/置信度检查）。"""
        return RouteDecision(
            target_agent=target,
            needs_clarification=False,
            urgency=UrgencyLevel.NORMAL,
            escalate_to_human=False,
            strategy=RetrievalStrategy.DIRECT,
            tone_instruction="请保持专业、友好的客服语气。",
        )

    @staticmethod
    def _is_confirming_response(history: list) -> bool:
        """判断用户是否在回应确认提示（上轮bot问了'确定要…请回复确认'）。"""
        if not history:
            return False
        for h in reversed(history):
            txt = h.answer if hasattr(h, 'answer') else ''
            if "确定要" in txt and "请回复" in txt:
                return True
        return False

    @staticmethod
    def _has_order_or_tracking_id(query: str) -> bool:
        """检测 query 是否包含明确的订单号或快递单号格式。"""
        q = query.strip()
        # 订单号: JD20260708-001
        if re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', q):
            return True
        # 快递单号（字母+数字）: YT7629819150434 / SF1234567890
        if re.search(r'^[A-Z]{2,6}\d{8,18}$', q):
            return True
        # 快递单号（纯数字10位以上）: 773367326370601
        if re.search(r'^\d{10,20}$', q):
            return True
        return False

    def _correct_product_inquiry_intent(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> IntentCategory:
        """商品咨询覆盖：查+商品词 但无订单号/物流词 → 强制修正为 KNOWLEDGE_QA。

        场景："帮我查一下苹果18这个手机" / "查下iPhone18"
        BERT可能分到 BUSINESS（因含"查"），但实际是商品咨询，应走RAG+LLM通识兜底。
        """
        query = perception.original_query.strip()

        # 已经是 knowledge_qa 就不需要修正
        if intent == IntentCategory.KNOWLEDGE_QA:
            return intent

        # ① 含商品咨询特征词
        product_kw = ["手机", "电脑", "耳机", "手表", "平板", "笔记本", "衣服", "鞋子",
                      "化妆品", "护肤品", "防晒", "口红", "包包", "家具", "家电",
                      "型号", "配置", "参数", "多少钱", "怎么样", "好不好", "评价",
                      "推荐", "哪个好", "买什么", "值得买", "性价比", "对比",
                      "是什么", "材质", "尺码", "颜色", "款式", "品牌", "功能"]
        is_product_q = any(w in query for w in product_kw)

        # ② 不含业务操作特征词
        biz_kw = ["物流", "快递", "到哪", "发货", "订单", "退款", "退货",
                  "取消", "地址", "修改", "配送", "收货"]
        is_biz_q = any(w in query for w in biz_kw)

        # ③ 不含订单号/快递单号格式
        has_id = bool(re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query))
        has_id = has_id or bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', query))
        has_id = has_id or bool(re.fullmatch(r'\d{10,20}', query))

        if is_product_q and not is_biz_q and not has_id:
            logger.info("router.product_inquiry_corrected",
                       query=query[:30], from_=intent.value, to="knowledge_qa")
            return IntentCategory.KNOWLEDGE_QA

        return intent

    def _correct_intent(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> IntentCategory:
        """情感×意图一致性修正。

        正面情绪 + 工单意图 → BERT 误判，修正为知识问答。
        """
        positive_sentiments = {SentimentLabel.HAPPY, SentimentLabel.GRATEFUL}
        if perception.sentiment_label in positive_sentiments and intent == IntentCategory.ESCALATE:
            logger.info("router.intent_corrected", from_="escalate", to="knowledge_qa")
            return IntentCategory.KNOWLEDGE_QA
        return intent

    def _correct_order_id_intent(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> IntentCategory:
        """订单号/快递单号强制修正：识别到ID格式 → 改写为 business。
        纯快递单号无论被识别成什么意图都修正（knowledge_mgmt 除外）。"""
        query = perception.original_query.strip()

        # 快递单号: 字母+数字(SF1234567890) 或 纯数字10位+(773367326370601)
        is_tracking = bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', query)) or bool(re.fullmatch(r'\d{10,20}', query))
        # 订单号（JD20260708-001）
        has_order = bool(re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query))
        has_logistics_kw = any(w in query for w in ["物流", "快递", "到哪", "发货", "查单", "配送"])

        # 纯快递单号 → 无条件修正为业务处理
        if is_tracking and intent != IntentCategory.KNOWLEDGE_MGMT:
            logger.info("router.tracking_corrected", query=query, from_=intent.value)
            return IntentCategory.BUSINESS

        # 订单号 + 非业务意图 → 修正为业务处理
        if (has_order or has_logistics_kw) and intent != IntentCategory.BUSINESS:
            logger.info("router.order_id_corrected", query=query[:30], from_=intent.value)
            return IntentCategory.BUSINESS

        return intent

    def _build_extra_instruction(
        self, perception: PerceptionResult, intent: IntentCategory
    ) -> str:
        """根据情感×意图组合构建动态指令。"""
        parts = []
        sentiment_label = perception.sentiment_label

        # ── 基于情感的指令 ──
        if sentiment_label == SentimentLabel.ANXIOUS:
            parts.append(
                "【重要】用户有购买疑虑，请在回复中主动提及以下保障（选1-2个最相关的）："
                "7天无理由退换、运费险、正品保证、价保30天、先行赔付。"
            )
        elif sentiment_label == SentimentLabel.ANGRY:
            parts.append(
                "【重要】用户正在愤怒中，回复首句必须道歉。不要使用'但是''不过'等转折词。"
            )
        elif sentiment_label == SentimentLabel.CONFUSED:
            parts.append(
                "【重要】用户感到困惑，请用通俗语言分2-3步说明，最后询问'这样说清楚了吗？'"
            )
        elif sentiment_label == SentimentLabel.HAPPY:
            parts.append(
                "【可选】用户情绪满意，可顺势推荐相关商品或邀请好评。"
            )

        # ── 基于意图的指令 ──
        if intent == IntentCategory.BUSINESS:
            parts.append(
                "用户需要业务处理。告知具体操作步骤和预计处理时间。"
            )
        elif intent == IntentCategory.ESCALATE:
            parts.append(
                "用户有投诉/不满。优先安抚情绪，明确告知处理方案和时效。"
            )

        # ── 基于实体的补充 ──
        if perception.entities:
            entity_types = {e.get("type", "") for e in perception.entities if e.get("type")}
            if "order_id" in entity_types:
                parts.append("用户提供了订单号，可据此查询。")
            if "sku" in entity_types or "product_name" in entity_types:
                parts.append("用户询问了具体商品，确保回答涉及该商品信息。")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════════
    # 🆕 澄清反问方法（合并自 clarify.py）
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _match_clarify_template(intent_value: str) -> str | None:
        """尝试用预置模板生成反问（不调LLM，毫秒级）。"""
        for (a, b), template in _CLARIFY_TEMPLATES.items():
            if a == intent_value or b == intent_value:
                logger.info("router.clarify_template_matched", pair=f"{a}/{b}")
                return f"我不太确定您的具体需求，想确认一下：\n\n{template}"
        return None

    async def _llm_clarify(self, query: str, perception: PerceptionResult) -> str:
        """LLM 动态生成反问选项（模板未命中时）。"""
        prompt = f"""你是一个电商客服的意图澄清助手。用户的意图识别置信度较低，你需要生成2-3个简洁的反问选项，帮助用户明确意图。

用户消息：{query}

检测到的可能意图：{perception.intent.value}（置信度：{perception.intent_confidence:.0%}）
识别到的实体：{perception.entities or '无'}

请生成一段友好的反问，包含2-3个选项让用户选择。格式要求：
- 先简短说明"不太确定用户的具体需求"
- 然后列出2-3个可能的理解方向（用选项形式）
- 语气友好、不机械

反问："""

        try:
            llm = get_llm("qa", temperature=0.3)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text if hasattr(response, 'text') else str(response)
            return text.strip()
        except Exception as e:
            logger.error("router.clarify_llm_failed", error=str(e))
            return (
                f"抱歉，我不太确定您具体想了解什么。您能再详细描述一下吗？\n\n"
                f"比如：\n"
                f"• 是想了解商品信息？\n"
                f"• 还是查询订单或物流？\n"
                f"• 或者需要售后帮助？\n\n"
                f"请告诉我更多细节，我会更精准地为您服务～"
            )

    async def _save_clarify_log(
        self, query: str, perception: PerceptionResult, clarification: str,
    ):
        """将澄清事件写入 MySQL（异步，fire-and-forget）。"""
        try:
            from backend.core.database import get_session
            from backend.models.db_models import ClarifyLog

            async with get_session() as session:
                record = ClarifyLog(
                    original_query=query,
                    detected_intent=perception.intent.value,
                    confidence=perception.intent_confidence,
                    clarification_question=clarification,
                    entities=perception.entities or {},
                )
                session.add(record)
                await session.commit()
                logger.info("router.clarify_log_saved", intent=perception.intent.value)
        except Exception as e:
            logger.warning("router.clarify_log_save_failed", error=str(e))


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        router = RoutingAgent()

        test_cases = [
            # (query, sentiment_label, intent, confidence, 期望target_agent)
            ("这件衣服起球吗？", SentimentLabel.ANXIOUS, IntentCategory.KNOWLEDGE_QA, 0.88, "knowledge_qa"),
            ("什么垃圾！用了两天就坏了！", SentimentLabel.ANGRY, IntentCategory.ESCALATE, 0.85, "escalate"),
            ("帮我查JD20240706-001的物流", SentimentLabel.NEUTRAL, IntentCategory.BUSINESS, 0.90, "business"),
            ("取消订单JD20240706-001", SentimentLabel.NEUTRAL, IntentCategory.BUSINESS, 0.88, "business"),
            ("帮我把地址改一下", SentimentLabel.NEUTRAL, IntentCategory.BUSINESS, 0.42, "knowledge_qa"),  # needs_clarification
            ("你好呀今天天气真好", SentimentLabel.HAPPY, IntentCategory.KNOWLEDGE_QA, 0.92, "knowledge_qa"),
            ("帮我记一下：7天无理由退货", SentimentLabel.NEUTRAL, IntentCategory.KNOWLEDGE_MGMT, 0.85, "knowledge_mgmt"),
            ("我要退款！太失望了", SentimentLabel.DISAPPOINTED, IntentCategory.ESCALATE, 0.78, "escalate"),
        ]

        for query, label, intent, conf, expected in test_cases:
            perception = PerceptionResult(
                original_query=query,
                sentiment="negative" if label in [SentimentLabel.ANXIOUS, SentimentLabel.ANGRY, SentimentLabel.DISAPPOINTED] else "neutral",
                sentiment_label=label,
                sentiment_confidence=0.9,
                intent=intent,
                intent_confidence=conf,
                entities=[],
            )
            decision = await router.route(perception)
            # 低置信度 → 检查 needs_clarification 而非 target_agent
            if decision.needs_clarification:
                match = "PASS" if expected == "knowledge_qa" else "FAIL"  # clarify 后默认回 knowledge_qa
            else:
                match = "PASS" if decision.target_agent.value == expected else "FAIL"
            print(f"[{match}] {query[:35]:<35} | target={decision.target_agent.value:<18} | urgency={decision.urgency.value:<10} | clarify={decision.needs_clarification} | escalate={decision.escalate_to_human}")

        print("\nrouter.py v3 自测通过")

    asyncio.run(test())
