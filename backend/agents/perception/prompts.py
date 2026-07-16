# backend/agents/perception/prompts.py
# Perception Agent — LLM Prompt 模板

# ═══════════════════════════════════════════════════════════════
# LLM 零样本情感分类（降级方案，来自 PerceptionAgent._llm_sentiment_predict）
# ═══════════════════════════════════════════════════════════════

SENTIMENT_LLM_PROMPT = """分析以下用户消息的情感极性，只输出一个英文单词（positive, negative, neutral）：
用户消息：{query}
情感："""


# ═══════════════════════════════════════════════════════════════
# BERT 10分类 → 4分发Agent 映射表
# ═══════════════════════════════════════════════════════════════

LEGACY_10_TO_4: dict[str, str] = {
    "product_inquiry": "knowledge_qa",
    "price_inquiry": "knowledge_qa",
    "recommend_request": "knowledge_qa",
    "chitchat": "knowledge_qa",
    "other": "knowledge_qa",
    "order_tracking": "business",
    "modify_order": "business",
    "after_sales": "business",
    "knowledge_mgmt": "knowledge_qa",  # 已合并到知识问答
    "complaint": "escalate",
}


# ═══════════════════════════════════════════════════════════════
# 7分类 → 3分类极性映射
# ═══════════════════════════════════════════════════════════════

FINE_TO_POLARITY: dict[str, str] = {
    "happy": "positive", "grateful": "positive",
    "neutral": "neutral",
    "confused": "negative", "anxious": "negative",
    "angry": "negative", "disappointed": "negative",
}


# ═══════════════════════════════════════════════════════════════
# 中文意图 → IntentCategory 枚举映射
# ═══════════════════════════════════════════════════════════════

INTENT_CN2EN: dict[str, str] = {
    "知识问答": "knowledge_qa",
    "业务处理": "business",
    "知识管理": "knowledge_qa",  # 已合并到知识问答
    "工单处理": "escalate",
}
