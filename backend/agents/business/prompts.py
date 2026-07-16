# backend/agents/business/prompts.py
# Business Agent — Prompt 模板 + 常量

# ═══════════════════════════════════════════════════════════════
# 查询类型关键词映射
# ═══════════════════════════════════════════════════════════════

LOGISTICS_KEYWORDS = ["物流", "快递", "到哪", "发货", "配送", "运输", "单号"]
ORDER_KEYWORDS = ["订单", "下单", "买了", "购买记录", "订单详情"]
STOCK_KEYWORDS = ["库存", "有货", "有没有货", "缺货", "补货", "到货"]

# ═══════════════════════════════════════════════════════════════
# 写操作类型
# ═══════════════════════════════════════════════════════════════

EXECUTE_ACTIONS = {
    "modify_address": "修改收货地址",
    "cancel_order": "取消订单",
    "apply_refund": "申请退款",
    "modify_quantity": "修改商品数量/规格",
}

# ═══════════════════════════════════════════════════════════════
# 写操作关键词
# ═══════════════════════════════════════════════════════════════

WRITE_KEYWORDS = [
    "取消", "不要了", "退单", "退款", "退钱", "退",
    "改地址", "换地址", "修改地址", "改收货",
    "改规格", "换颜色", "换尺码", "修改数量",
]

# ═══════════════════════════════════════════════════════════════
# LLM 查询解析 Prompt
# ═══════════════════════════════════════════════════════════════

PARSE_QUERY_PROMPT = """分析以下用户请求，判断需要执行的业务操作。

可选操作：
{actions_desc}

返回纯JSON（不要markdown）：
{{
    "action": "logistics|order|stock|modify_address|cancel_order|apply_refund|modify_quantity|none",
    "params": {{
        "order_id": "订单号（没有则为空字符串）",
        "sku": "商品SKU（没有则为空字符串）",
        "product_name": "商品名称（没有则为空字符串）",
        "new_value": "修改后的值（写操作时填写）",
        "reason": "原因（写操作时填写）"
    }},
    "needs_confirm": true/false
}}

needs_confirm 规则：
- logistics/order/stock（读操作）→ false
- cancel_order/apply_refund/modify_quantity → true（不可逆/涉及金钱）
- modify_address → false（可再改）

用户请求：{query}
JSON："""
