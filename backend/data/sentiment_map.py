# backend/data/sentiment_map.py
# 情感→话术映射表：这是整个「动态 Prompt 构建器」的核心数据。
# 根据用户的细粒度情感标签，自动注入对应的语气指令和业务策略。
#
# 设计原则：
#   - 每个情感标签对应一套完整的「安抚策略 + 业务动作」
#   - 统一从这里的映射表读取，Agent 代码不做硬编码判断
#   - 业务方可随时调整话术，不影响代码逻辑
#
# 🆕 v3 新增：情绪紧急度矩阵（Emotion Urgency Matrix）
#   - 极度负面情绪 + 敏感意图（退款/投诉）→ 直接转人工，不走正常流程
#   - 这是 router.py "三维决策" 中的情绪维度

from backend.models.schemas import SentimentLabel, IntentCategory, RetrievalStrategy

# ── 情感→语气指令 + 业务策略 映射表 ────────────────────────────

SENTIMENT_TONE_MAP: dict[SentimentLabel, dict] = {
    SentimentLabel.HAPPY: {
        "label_cn": "😊 满意/开心",
        "tone_instruction": (
            "用户目前情绪积极、满意度高。请使用轻松、亲切的语气回复。"
            "可以顺势进行交叉推荐、邀请好评、引导复购。"
            "语气可带适度幽默感，但不要过度热情。"
        ),
        "strategy_hint": "积极维护：肯定用户 + 顺势推荐 + 邀请好评",
        "prefer_retrieval": RetrievalStrategy.DIRECT,
        "escalate_to_human": False,
    },
    SentimentLabel.GRATEFUL: {
        "label_cn": "🙏 感谢",
        "tone_instruction": (
            "用户表达感谢。请以温暖、谦逊的语气回应，表达对用户信任的珍视。"
            "可适度提及品牌服务承诺，强化用户的正面印象。"
        ),
        "strategy_hint": "感恩回应：表达感谢 + 服务承诺 + 保持连接",
        "prefer_retrieval": RetrievalStrategy.DIRECT,
        "escalate_to_human": False,
    },
    SentimentLabel.NEUTRAL: {
        "label_cn": "😐 中性/信息型",
        "tone_instruction": (
            "用户情绪中性，正在进行信息查询。请保持专业、高效、简洁的语气。"
            "精准回答问题，避免过多废话。适度推荐相关商品或服务。"
        ),
        "strategy_hint": "高效服务：精准回答 + 适度推荐",
        "prefer_retrieval": RetrievalStrategy.DIRECT,
        "escalate_to_human": False,
    },
    SentimentLabel.CONFUSED: {
        "label_cn": "🤔 困惑",
        "tone_instruction": (
            "用户对某些信息感到困惑或不理解。请使用耐心、清晰的语气，"
            "用通俗易懂的语言解释复杂概念。分步骤说明，配合举例。"
            "主动询问是否需要进一步解释。"
        ),
        "strategy_hint": "耐心解答：化繁为简 + 分步说明 + 主动确认理解",
        "prefer_retrieval": RetrievalStrategy.SUBQUERY,
        "escalate_to_human": False,
    },
    SentimentLabel.ANXIOUS: {
        "label_cn": "😰 焦虑/担忧",
        "tone_instruction": (
            "用户目前对购买/售后有疑虑和担忧。请使用安抚性、确定的语气。"
            "必须强调以下保障（按需选用）：7天无理由退换、运费险、正品保证、"
            "价保服务、先行赔付。给用户确定感，避免使用'可能''或许'等模糊词。"
            "主动提供人工客服通道作为备选。"
        ),
        "strategy_hint": "解除焦虑：强调保障 + 确定性承诺 + 主动安抚 + 备用人工通道",
        "prefer_retrieval": RetrievalStrategy.HYDE,
        "escalate_to_human": False,
    },
    SentimentLabel.ANGRY: {
        "label_cn": "😡 愤怒/不满",
        "tone_instruction": (
            "用户情绪激动、不满或愤怒。请第一时间真诚道歉（无论谁的责任），"
            "表达同理心-'我完全理解您的感受'。不要解释、不要推卸责任、不要使用'但是'。"
            "承诺快速处理，主动提供升级通道（主管/人工客服）。"
            "如为严重投诉，请直接建议转人工处理以确保问题最快解决。"
        ),
        "strategy_hint": "怒退四步法：道歉→共情→快速承诺→升级通道",
        "prefer_retrieval": RetrievalStrategy.BACKTRACK,
        "escalate_to_human": True,
    },
    SentimentLabel.DISAPPOINTED: {
        "label_cn": "😞 失望",
        "tone_instruction": (
            "用户表达了失望情绪。请表达理解和歉意，避免空泛的'我们很抱歉'。"
            "诚恳承认不足，给出具体的补救措施或补偿方案。"
            "强调改进决心，让用户感受到被重视。"
        ),
        "strategy_hint": "挽回信任：诚恳致歉 + 具体补救 + 强调重视",
        "prefer_retrieval": RetrievalStrategy.HYDE,
        "escalate_to_human": False,
    },
}


# ── 意图→知识库过滤映射 ──────────────────────────────────────

INTENT_SOURCE_FILTER_MAP: dict[str, str] = {
    "product_inquiry":    "products",
    "price_inquiry":      "promotion",
    "recommend_request":  "products",
    "order_tracking":     "logistics",
    "after_sales":        "after_sales",
    "complaint":          "after_sales",
    "modify_order":       "orders",
    "chitchat":           "",          # 闲聊不需要知识库
    "other":              "",
}


def get_tone_config(sentiment_label: SentimentLabel) -> dict:
    """根据细粒度情感标签获取语气配置。
    如果未找到对应配置，返回默认中性配置。
    """
    return SENTIMENT_TONE_MAP.get(
        sentiment_label,
        SENTIMENT_TONE_MAP[SentimentLabel.NEUTRAL]
    )


def get_source_filter(intent: str) -> str:
    """根据意图获取知识库过滤类别。"""
    return INTENT_SOURCE_FILTER_MAP.get(intent, "")


# ═══════════════════════════════════════════════════════════════
# 🆕 情绪紧急度矩阵 (Emotion Urgency Matrix)
# ═══════════════════════════════════════════════════════════════
# router.py 在分发前会并行检测此矩阵：
#   极度负面情绪 + 敏感意图（退款/投诉）→ UrgencyLevel.CRITICAL → 直接转人工
#   这是"三维决策"中的情绪维度——不等正常流程走完，立即拦截

# 触发紧急接管的意图（涉及金钱/权益，用户情绪容易升级）
_CRITICAL_INTENTS = {
    IntentCategory.ESCALATE,       # 工单处理(投诉)
    IntentCategory.BUSINESS,       # 业务处理(售后/退款/改单)
}

# 紧急情绪：这些情绪 + 敏感意图 = 直接转人工
_URGENCY_SENTIMENTS: dict[SentimentLabel, dict] = {
    SentimentLabel.ANGRY: {
        "urgency": "critical",
        "reason": "用户极度愤怒，无法通过自动化流程安抚，需人工立即介入",
        "action": "escalate_now",
        "message_template": (
            "非常抱歉给您带来了这么不愉快的体验。您的问题我已为您紧急转接人工客服，"
            "专属客服将在{wait_time}分钟内与您联系。\n\n"
            "📞 您也可以直接拨打客服热线：{phone}\n"
            "🎫 工单编号：{ticket_id}"
        ),
    },
    SentimentLabel.DISAPPOINTED: {
        "urgency": "elevated",
        "reason": "用户失望不满，需要优先处理和具体补救方案",
        "action": "priority_handle",
        "message_template": "",
    },
    SentimentLabel.ANXIOUS: {
        "urgency": "elevated",
        "reason": "用户焦虑担忧，需要确定性承诺和保障说明",
        "action": "emphasize_guarantees",
        "message_template": "",
    },
}


def detect_urgency(
    sentiment_label: SentimentLabel,
    intent: IntentCategory,
) -> dict:
    """情绪紧急度检测 —— router.py 的并行检测逻辑。

    返回：
    {
        "urgency": "normal" | "elevated" | "critical",
        "reason": str,
        "action": "escalate_now" | "priority_handle" | "emphasize_guarantees" | "normal",
        "message_template": str,  # critical 级别的转人工消息模板
    }
    """
    # 关键判断：极度负面情绪 + 敏感意图（退款/投诉） = 紧急接管
    sentiment_config = _URGENCY_SENTIMENTS.get(sentiment_label, {})

    if not sentiment_config:
        return {
            "urgency": "normal",
            "reason": "",
            "action": "normal",
            "message_template": "",
        }

    urgency = sentiment_config.get("urgency", "normal")

    # ANGRY + (ESCALATE | BUSINESS) → CRITICAL 转人工
    if urgency == "critical" and intent in _CRITICAL_INTENTS:
        return {
            "urgency": "critical",
            "reason": sentiment_config.get("reason", ""),
            "action": "escalate_now",
            "message_template": sentiment_config.get("message_template", ""),
        }

    # DISAPPOINTED + 敏感意图 → ELEVATED（优先处理，不转人工）
    # 注：只有 ANGRY 才触发转人工，失望用户仍可走自动化流程
    if urgency == "elevated" and intent in _CRITICAL_INTENTS:
        return {
            "urgency": "elevated",
            "reason": f"用户失望 + 敏感意图({intent.value})，优先处理",
            "action": "priority_handle",
            "message_template": "",
        }

    # ANXIOUS → ELEVATED（不到 critical，但需要特殊处理）
    if urgency == "elevated":
        return {
            "urgency": "elevated",
            "reason": sentiment_config.get("reason", ""),
            "action": sentiment_config.get("action", "priority_handle"),
            "message_template": "",
        }

    return {
        "urgency": "normal",
        "reason": "",
        "action": "normal",
        "message_template": "",
    }
