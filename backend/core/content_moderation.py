# backend/core/content_moderation.py
# 知识库上传内容安全扫描 — 检测恶意/风险模式，三级分流（pass / review / reject）

# ── 破坏性模式：承诺超额赔付、假冒官方、虚假法律声明等 ──
MALICIOUS_PATTERNS = [
    # 超额赔付承诺（风险极高）
    ("假一赔十", "承诺超额赔付「假一赔十」，与法律规定的假一赔三不符，涉嫌虚假承诺"),
    ("假一赔百", "承诺超额赔付，涉嫌欺诈"),
    ("十倍赔偿", "承诺超额赔偿，存在法律风险"),
    ("百倍赔偿", "承诺超额赔偿，涉嫌虚假宣传"),
    ("无条件退款", "承诺无条件退款，不符合平台售后规则"),
    ("永久保修", "承诺永久保修，超出法定三包期限"),
    ("终身质保", "承诺终身质保，可能无法兑现"),
    ("全额退款不退货", "承诺仅退款不退货，违反平台交易规则"),
    # 假冒官方
    ("官方授权", "声称官方授权，需提供授权证明文件"),
    ("品牌直营", "声称品牌直营，需提供品牌授权链路证明"),
    ("国家级", "使用「国家级」等权威背书字眼，需提供认证证书"),
    ("100%有效", "使用绝对化承诺用语，涉嫌虚假宣传"),
    ("治愈率", "涉及医疗功效承诺，违反广告法"),
    ("包治", "涉及医疗功效承诺，违反广告法"),
    # 平台违规
    ("加微信", "引导站外交易（微信），违反平台规定"),
    ("扫码下单", "可能引导站外交易，需人工审核"),
    ("私聊下单", "引导私下交易，存在交易风险"),
    ("货到付款", "涉及非平台担保交易模式，需确认合规性"),
]

# 安全关键词（正常政策描述，不应被误杀）
SAFE_PATTERNS = [
    "7天无理由", "退换货", "退款", "退货", "运费险",
    "包邮", "优惠券", "满减", "折扣", "质保", "保修",
]


def scan_content(content: str) -> dict:
    """扫描上传内容，检测恶意/风险模式。

    :return: {"safe": bool, "risks": list[dict], "verdict": "pass"|"review"|"reject"}
    """
    risks = []
    for pattern, reason in MALICIOUS_PATTERNS:
        if pattern in content:
            risks.append({"pattern": pattern, "reason": reason})

    if not risks:
        return {"safe": True, "risks": [], "verdict": "pass"}

    # 判断风险等级
    critical_keywords = ["假一赔", "十倍", "百倍", "治愈", "包治"]
    has_critical = any(kw in content for kw in critical_keywords)

    return {
        "safe": False,
        "risks": risks,
        "verdict": "reject" if has_critical else "review",
    }
