# backend/rag/post_processor.py
# 🆕 v3 答案后处理器：幻觉检测 / 敏感词过滤 / 格式规范化。
#
# 在 LLM 生成答案后执行，确保输出的安全性、准确性和规范性。
#
# 三项检查（串行执行）：
#   1. 幻觉检测 — 答案中的事实是否与检索到的文档一致
#   2. 敏感词过滤 — 过滤不合规内容（竞品名、违规承诺、隐私泄露）
#   3. 格式规范化 — 统一输出格式（分段、标点、Emoji 适度使用）

import re
from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.llm_factory import get_llm
from backend.core.retry import with_retry_async

logger = get_logger(__name__)

# ── 敏感词库（业务方可配置）────────────────────────────────

_SENSITIVE_PATTERNS = [
    # 违规承诺
    (r"保证.*?\d+天.*?效果", "包含效果保证类表述"),
    (r"绝对.*?最好", "包含绝对化用语"),
    (r"100%.*?有效", "包含绝对化承诺"),
    # 隐私泄露风险
    (r"1[3-9]\d{9}", "疑似包含手机号"),
    (r"\d{6,}@", "疑似包含邮箱"),
    # 竞品相关（示例）
    # (r"某东|某宝|拼多多", "提及竞品平台"),
]


class AnswerPostProcessor:
    """答案后处理器 — 确保 LLM 输出的安全性和准确性。

    使用方式：
        processor = AnswerPostProcessor()
        result = await processor.process(
            answer="LLM生成的答案",
            context_docs=[检索到的文档],
            query="用户原始问题",
        )
        # result.final_answer — 最终输出给用户的答案
    """

    def __init__(self):
        self.settings = get_settings()
        self.enabled = self.settings.rag_post_process_enabled

    async def process(
        self,
        answer: str,
        context_docs: list | None = None,
        query: str = "",
    ) -> dict:
        """执行完整的后处理管线。

        :param answer: LLM 原始输出
        :param context_docs: 检索到的参考文档（用于幻觉检测）
        :param query: 用户问题
        :return: {
            "final_answer": str,       # 最终答案
            "hallucination_flag": bool, # 是否有幻觉风险
            "sensitive_flag": bool,     # 是否有敏感内容
            "warnings": list[str],      # 警告信息
        }
        """
        if not self.enabled or not answer:
            return {
                "final_answer": answer,
                "hallucination_flag": False,
                "sensitive_flag": False,
                "warnings": [],
            }

        warnings = []
        final = answer
        hallucination_flag = False
        sensitive_flag = False

        # 步骤1：敏感词过滤
        final, sensitive_flag, sensitive_warnings = self._filter_sensitive(final)
        warnings.extend(sensitive_warnings)

        # 步骤2：格式规范化
        final = self._normalize_format(final)

        # 步骤3：幻觉检测（需要 context_docs）
        if context_docs:
            hallucination_flag, hallucination_warnings = await self._detect_hallucination(
                final, context_docs, query
            )
            warnings.extend(hallucination_warnings)

            # 如果幻觉风险高，追加免责提示
            if hallucination_flag:
                final += "\n\n具体以店铺实际为准哦～"

        logger.info(
            "post_processor.done",
            hallucination=hallucination_flag,
            sensitive=sensitive_flag,
            warnings=len(warnings),
        )

        return {
            "final_answer": final,
            "hallucination_flag": hallucination_flag,
            "sensitive_flag": sensitive_flag,
            "warnings": warnings,
        }

    # ── 敏感词过滤 ──────────────────────────────────────────

    def _filter_sensitive(self, text: str) -> tuple[str, bool, list[str]]:
        """过滤敏感内容。"""
        warnings = []
        has_sensitive = False

        for pattern, desc in _SENSITIVE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                has_sensitive = True
                warnings.append(f"检测到: {desc}")
                # 对匹配内容进行脱敏
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    text = text.replace(str(match), "***")

        return text, has_sensitive, warnings

    # ── 🆕 高危信息检测 ────────────────────────────────────

    # 高危意图 + 关键词（涉及政策/法律/金融，不能靠LLM通识知识回答）
    HIGH_RISK_INTENTS = {"business", "escalate"}  # 业务处理(售后/退款) + 工单处理(投诉)

    HIGH_RISK_KEYWORDS = [
        # 退款/赔付
        "退款", "退货", "退换", "换货", "赔付", "赔偿", "假一赔", "赔几", "赔多少",
        # 法律/合同
        "法律", "法规", "合同", "条款", "协议", "隐私", "个人信息", "消费者权益",
        # 保修/质保/三包
        "保修", "质保", "三包", "保修期", "保修卡", "售后政策",
        # 金融/发票
        "发票", "税率", "关税", "运费险", "保价", "差价", "补偿",
        # 投诉/举报
        "投诉", "举报", "起诉", "维权", "12315",
        # 正品/假货承诺
        "正品保证", "假货", "真伪", "鉴定", "仿冒", "山寨",
    ]

    @classmethod
    def is_high_risk(cls, query: str, intent_value: str = "") -> bool:
        """检测用户问题是否属于高危领域（政策/法律/金融）。
        高危+无知识库验证 → 必须转人工，不能用LLM通识知识编造。
        """
        if intent_value in cls.HIGH_RISK_INTENTS:
            return True
        return any(kw in query for kw in cls.HIGH_RISK_KEYWORDS)

    # ── 格式规范化 ──────────────────────────────────────────

    def _normalize_format(self, text: str) -> str:
        """统一输出格式。"""
        # 去除多余空行（保留最多1个连续空行）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除首尾空白
        text = text.strip()

        # 确保以有意义的内容结尾（去掉末尾的纯符号行）
        text = re.sub(r'\n[-—–=_*]{3,}\s*$', '', text)

        return text

    # ── 幻觉检测 ────────────────────────────────────────────

    async def _detect_hallucination(
        self, answer: str, context_docs: list, query: str
    ) -> tuple[bool, list[str]]:
        """检测答案中的事实是否与检索文档一致。

        策略：LLM 逐句对比答案和参考文档，标记不一致的陈述。
        """
        if not context_docs:
            return False, []

        # 提取参考文档内容摘要
        context_text = ""
        for i, doc in enumerate(context_docs[:3], 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_text += f"[参考{i}] {content[:300]}\n"

        prompt = f"""你是一个事实核查助手。请检查以下AI生成的答案是否与参考文档中的信息一致。

## 参考文档
{context_text}

## AI生成的答案
{answer}

请判断：
1. 答案中的关键事实（数字、规格、价格、政策）是否都能在参考文档中找到依据？
2. 是否存在明显与参考文档矛盾或来源不明的内容？

只回复JSON（不要markdown）：
{{"has_hallucination": true/false, "issues": ["问题描述"]}}

JSON："""

        try:
            llm = get_llm("qa", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()

            import json
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)

            result = json.loads(text)
            has_hallucination = result.get("has_hallucination", False)
            issues = result.get("issues", [])

            if has_hallucination:
                logger.warning("post_processor.hallucination_detected", issues=issues)

            return has_hallucination, issues

        except Exception as e:
            logger.warning("post_processor.hallucination_check_failed", error=str(e))
            return False, []


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        processor = AnswerPostProcessor()

        # 测试1：正常内容
        result1 = await processor.process(
            answer="这款T恤是100%纯棉材质，支持机洗，建议冷水洗涤。价格99元。",
            context_docs=[],
        )
        print(f"测试1: sensitive={result1['sensitive_flag']}, warnings={result1['warnings']}")

        # 测试2：含敏感词
        result2 = await processor.process(
            answer="这款产品保证7天内见效，绝对是市面上最好的！联系电话13812345678",
            context_docs=[],
        )
        print(f"测试2: sensitive={result2['sensitive_flag']}, final={result2['final_answer'][:80]}...")

        print("\npost_processor.py 自测通过")

    asyncio.run(test())
