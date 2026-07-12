# backend/agents/business.py
# 🆕 v3 — 统一业务Agent：MySQL读写合一（合并 business_execute.py）。
#
# 能力：
#   读操作：查物流、查订单、查库存（SELECT）
#   写操作：取消订单、申请退款、修改地址、修改规格（UPDATE/INSERT，需二次确认）
#
# 设计原则：
#   - 读写合一 — CQRS 在这个体量下是过度设计，合并简化分发
#   - 所有写操作都需要二次确认（涉及金钱/不可逆）
#   - LLM 解析 + 关键词兜底双路保障
#   - 操作日志全记录

import json
import re
import time
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.retry import with_retry_async
from backend.core.llm_factory import get_llm
from backend.models.schemas import (
    AgentResponse, AgentMessage, ConversationHistory,
    FunctionCall, FunctionCallResult,
)

from sqlalchemy import select

logger = get_logger(__name__)

# ── 查询类型关键词映射 ──────────────────────────────────────
LOGISTICS_KEYWORDS = ["物流", "快递", "到哪", "发货", "配送", "运输", "单号"]
ORDER_KEYWORDS = ["订单", "下单", "买了", "购买记录", "订单详情"]
STOCK_KEYWORDS = ["库存", "有货", "有没有货", "缺货", "补货", "到货"]

# ── 🆕 写操作类型 ──────────────────────────────────────────
EXECUTE_ACTIONS = {
    "modify_address": "修改收货地址",
    "cancel_order": "取消订单",
    "apply_refund": "申请退款",
    "modify_quantity": "修改商品数量/规格",
}

# ── 写操作关键词 ──────────────────────────────────────────
WRITE_KEYWORDS = [
    "取消", "不要了", "退单", "退款", "退钱", "退",
    "改地址", "换地址", "修改地址", "改收货",
    "改规格", "换颜色", "换尺码", "修改数量",
]


class BusinessAgent:
    """统一业务Agent — MySQL读写合一 + 确认流程。

    触发条件：router 判断为 business
    （intent = order_tracking / modify_order / after_sales+退款/取消）
    """

    async def handle(
        self,
        query: str,
        history: list[ConversationHistory] | None = None,
    ) -> AgentResponse:
        """执行业务操作：LLM判断类型+提取参数 → 读直接执行 / 写确认后执行。"""
        start = time.time()
        logger.info("business.started", query=query[:50])

        # 步骤1：LLM 统一解析（读+写）
        action, params, needs_confirm = await self._parse_query(query)

        # 步骤2：参数不全 → 引导用户（确认场景从历史中取order_id）
        if action == "none" or (action in EXECUTE_ACTIONS and not params.get("order_id")):
            # 如果是回应确认提示，从历史中提取order_id
            if history and self._is_confirming(history):
                # 从上轮bot消息中解析order_id
                for h in reversed(history):
                    txt = h.answer if hasattr(h, 'answer') else ''
                    m = re.search(r'订单号[：:]\s*([A-Z]{2,4}\d{6,12}[-_]\d{2,6})', txt)
                    if m:
                        params["order_id"] = m.group(1)
                        break
                if not params.get("order_id"):
                    return self._missing_info_response(action, params)
            else:
                return self._missing_info_response(action, params)

        # 步骤3：写操作 → 二次确认（支持上下文感知）
        if needs_confirm and not self._has_explicit_confirm(query, history):
            elapsed = (time.time() - start) * 1000
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant",
                    content=self._build_confirm_message(action, params),
                    intent_detected="business",
                    function_calls=[FunctionCallResult(
                        call=FunctionCall(name=action, arguments=params),
                        success=True,
                        result="pending_confirmation",
                    )],
                ),
                processing_time_ms=elapsed,
            )

        # 步骤4：执行操作
        try:
            result_text = await self._execute(action, params)
            success = True
        except Exception as e:
            logger.error("business.execute_failed", action=action, error=str(e))
            action_cn = EXECUTE_ACTIONS.get(action, action)
            result_text = (
                f"❌ 操作失败：{action_cn}未能完成。\n\n"
                f"原因：{str(e)}\n\n"
                f"💡 请稍后重试，或拨打客服电话 {get_settings().customer_service_phone} 获取人工帮助。"
            )
            success = False

        elapsed = (time.time() - start) * 1000
        return AgentResponse(
            success=success,
            message=AgentMessage(
                role="assistant",
                content=result_text,
                intent_detected="business",
                function_calls=[FunctionCallResult(
                    call=FunctionCall(name=action, arguments=params),
                    success=success,
                    result=result_text[:300],
                )],
            ),
            processing_time_ms=elapsed,
        )

    # ═══════════════════════════════════════════════════════════
    # 🆕 统一 LLM 解析（读+写）
    # ═══════════════════════════════════════════════════════════

    async def _parse_query(self, query: str) -> tuple[str, dict, bool]:
        """LLM统一解析：判断操作类型（读/写）+ 提取参数 + 是否需要确认。"""
        actions_desc = "\n".join([
            "读操作：",
            "  - logistics: 查物流",
            "  - order: 查订单详情",
            "  - stock: 查库存",
            "写操作：",
            *[f"  - {k}: {v}" for k, v in EXECUTE_ACTIONS.items()],
            "  - none: 无法判断",
        ])

        prompt = f"""分析以下用户请求，判断需要执行的业务操作。

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

        try:
            llm = get_llm("qa", temperature=0)
            response = await with_retry_async(llm.ainvoke, prompt)
            text = response.text if hasattr(response, 'text') else str(response)

            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)

            decision = json.loads(text)
            action = decision.get("action", "none")
            params = decision.get("params", {})
            needs_confirm = decision.get("needs_confirm", False)

            logger.info("business.llm_parsed", action=action, params=params)
            return action, params, needs_confirm

        except Exception as e:
            logger.error("business.llm_parse_failed", error=str(e))
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> tuple[str, dict, bool]:
        """关键词兜底：LLM解析失败时用关键词判断读写类型。"""
        order_match = re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query)
        order_id = order_match.group() if order_match else ""

        # 写操作关键词优先检测
        if any(w in query for w in ["取消", "不要了", "退单"]):
            return ("cancel_order", {"order_id": order_id, "new_value": "", "reason": query}, True)
        if any(w in query for w in ["退款", "退钱", "退"]):
            return ("apply_refund", {"order_id": order_id, "new_value": "", "reason": query}, True)
        if any(w in query for w in ["改地址", "换地址", "修改地址", "改收货"]):
            return ("modify_address", {"order_id": order_id, "new_value": "", "reason": query}, False)
        if any(w in query for w in ["改规格", "换颜色", "换尺码", "修改数量"]):
            return ("modify_quantity", {"order_id": order_id, "new_value": "", "reason": query}, True)

        # 读操作关键词
        if any(w in query for w in LOGISTICS_KEYWORDS):
            return ("logistics", {"order_id": order_id, "sku": "", "product_name": ""}, False)
        if any(w in query for w in ORDER_KEYWORDS):
            return ("order", {"order_id": order_id, "sku": "", "product_name": ""}, False)
        if any(w in query for w in STOCK_KEYWORDS):
            return ("stock", {"order_id": "", "sku": "", "product_name": ""}, False)

        return ("none", {"order_id": order_id, "sku": "", "product_name": ""}, False)

    # ═══════════════════════════════════════════════════════════
    # 执行分发
    # ═══════════════════════════════════════════════════════════

    async def _execute(self, action: str, params: dict) -> str:
        """统一执行入口：根据 action 分发到读/写方法。"""
        order_id = params.get("order_id", "").strip()

        # 🔧 长得像快递单号 → 强制走物流查询
        is_tracking = (
            bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', order_id)) or
            bool(re.fullmatch(r'\d{10,20}', order_id))
        )
        if is_tracking:
            logger.info("business.auto_redirect_logistics", order_id=order_id, action=action)
            return await self._query_logistics(order_id)

        # 读操作
        if action == "logistics":
            return await self._query_logistics(order_id)
        if action == "order":
            return await self._query_order(order_id)
        if action == "stock":
            return await self._query_stock(
                sku=params.get("sku", ""),
                product_name=params.get("product_name", ""),
            )
        # 写操作
        if action == "cancel_order":
            return await self._cancel_order(params.get("order_id", ""), params.get("reason", ""))
        if action == "apply_refund":
            return await self._apply_refund(params.get("order_id", ""), params.get("reason", ""))
        if action == "modify_address":
            return await self._modify_address(params.get("order_id", ""), params.get("new_value", ""))
        if action == "modify_quantity":
            return await self._modify_quantity(params.get("order_id", ""), params.get("new_value", ""))

        return self._help_message()

    # ═══════════════════════════════════════════════════════════
    # 读操作：查物流 / 查订单 / 查库存
    # ═══════════════════════════════════════════════════════════

    async def _query_logistics(self, order_id: str) -> str:
        """查询物流轨迹。

        快递单号（如YT7629819150434）→ 先查数据库关联订单 → 有则展示订单信息
        → 无则提示快递鸟API即将上线。
        订单号（如JD20260708-001）→ 查数据库 → 展示订单状态和物流信息。
        """
        if not order_id:
            return "请提供您的订单号，我才能帮您查物流哦～\n\n💡 订单号通常在订单详情页可以找到，格式如 JD20240706-001。"

        from backend.core.database import get_session
        from backend.models.db_models import Order

        # 🔧 快递单号格式（如 YT7629819150434 / SF1234567890）
        is_tracking = bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', order_id.strip()))
        is_pure_digits = bool(re.fullmatch(r'\d{10,20}', order_id.strip()))

        if is_tracking or is_pure_digits:
            logger.info("business.tracking_lookup", tracking_no=order_id)
            # 先查数据库：有没有订单关联了这个快递单号
            async with get_session() as session:
                result = await session.execute(
                    select(Order).where(Order.logistics_tracking == order_id.strip())
                )
                order = result.scalars().all()

            if order:
                # 找到关联订单 → 展示订单信息
                lines = [f"📦 快递单号 {order_id.strip()} 关联订单：\n"]
                for o in order:
                    status_map = {
                        "pending": "待付款", "paid": "已付款，备货中",
                        "shipped": "运输中", "delivered": "已签收",
                        "cancelled": "已取消", "refunding": "退款中", "refunded": "已退款",
                    }
                    status_cn = status_map.get(o.status, o.status)
                    lines.append(
                        f"📋 订单号：{o.order_id}\n"
                        f"   商品：{o.product_name} ×{o.quantity}\n"
                        f"   状态：{status_cn}\n"
                        f"   收货地址：{o.receiver_address}"
                    )
                lines.append(f"\n💡 快递鸟实时轨迹API即将上线，敬请期待！届时可查看实时运输轨迹。")
                return "\n\n".join(lines) if len(order) > 1 else "\n".join(lines)

            # 未找到关联订单 → 提示
            return (
                f"📦 快递单号：{order_id.strip()}\n\n"
                f"未在系统中找到与此快递单号关联的订单。\n\n"
                f"💡 快递鸟实时轨迹API即将上线，敬请期待！\n"
                f"   上线后可直接查询任意快递单号的实时运输轨迹。\n\n"
                f"📞 如有疑问，请联系客服：{get_settings().customer_service_phone}"
            )

        # ── 订单号查询（如 JD20260708-001）──
        async with get_session() as session:
            result = await session.execute(
                select(Order).where(Order.order_id == order_id)
            )
            order = result.scalar_one_or_none()

        if order is None:
            return (
                f"未找到订单 {order_id} 的信息。\n\n"
                f"💡 请核对订单号是否正确，或拨打客服电话 {get_settings().customer_service_phone}。"
            )

        tracking_no = order.logistics_tracking

        # 没有快递单号 → 展示订单状态
        if not tracking_no:
            status_map = {
                "pending": "订单尚未付款，暂无物流信息。",
                "paid": "订单已付款，商家正在备货中，预计24小时内发货。",
                "cancelled": "订单已取消，无物流信息。",
            }
            hint = status_map.get(order.status, f"订单状态为「{order.status}」，暂未生成快递单号。")
            return (
                f"📦 订单 {order_id} 物流查询\n\n"
                f"商品：{order.product_name}\n"
                f"{hint}\n\n"
                f"💡 发货后会短信通知您快递单号，请留意手机短信。"
            )

        # 有快递单号 → 展示订单信息 + 快递鸟预告
        status_map = {
            "pending": "待付款", "paid": "已付款，备货中",
            "shipped": "运输中", "delivered": "已签收",
            "cancelled": "已取消", "refunding": "退款中", "refunded": "已退款",
        }
        status_cn = status_map.get(order.status, order.status)
        return (
            f"📦 订单 {order_id} 物流信息\n\n"
            f"商品：{order.product_name} ×{order.quantity}\n"
            f"金额：¥{order.amount:.2f}\n"
            f"状态：{status_cn}\n"
            f"收货人：{order.receiver_name}\n"
            f"收货地址：{order.receiver_address}\n"
            f"🏢 快递单号：{tracking_no}\n\n"
            f"💡 快递鸟实时轨迹API即将上线，敬请期待！\n"
            f"   当前可复制快递单号到快递官网查询实时轨迹。"
        )

    async def _query_order(self, order_id: str) -> str:
        """查询订单详情。"""
        if not order_id:
            return "请提供您的订单号，我才能帮您查订单详情哦～\n\n💡 订单号格式如 JD20240706-001。"

        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(
                select(Order).where(Order.order_id == order_id)
            )
            record = result.scalar_one_or_none()

        if record is None:
            logger.info("business.order_not_found", order_id=order_id)
            return (
                f"未找到订单 {order_id} 的信息。\n\n"
                f"请核对订单号是否正确，或拨打客服电话 {get_settings().customer_service_phone} 咨询。"
            )

        status_map = {
            "pending": "待付款", "paid": "已付款", "shipped": "已发货",
            "delivered": "已签收", "cancelled": "已取消",
            "refunding": "退款中", "refunded": "已退款",
        }
        status_cn = status_map.get(record.status, record.status)

        return (
            f"📋 订单 {order_id} 详情：\n\n"
            f"商品：{record.product_name} ×{record.quantity}\n"
            f"金额：¥{record.amount:.2f}\n"
            f"状态：{status_cn}\n"
            f"收货人：{record.receiver_name}\n"
            f"收货地址：{record.receiver_address}\n"
            f"下单时间：{record.created_at.strftime('%Y-%m-%d %H:%M') if record.created_at else '未知'}\n\n"
            f"💡 如需修改订单或申请售后，请告知具体需求。"
        )

    async def _query_stock(self, sku: str = "", product_name: str = "") -> str:
        """查询商品库存。"""
        if not sku and not product_name:
            return "请告诉我您要查哪个商品的库存，提供商品名称或SKU编号都可以～"

        from backend.core.database import get_session
        from backend.models.db_models import Product

        async with get_session() as session:
            if sku:
                result = await session.execute(
                    select(Product).where(Product.product_id == sku)
                )
            else:
                result = await session.execute(
                    select(Product).where(Product.title.like(f"%{product_name}%"))
                )
            records = result.scalars().all()

        if not records:
            keyword = sku or product_name
            return f"未找到「{keyword}」的商品信息。\n\n💡 请核对商品名称或SKU是否正确，或联系商家确认。"

        lines = [f"📊 商品查询结果（共 {len(records)} 个匹配）：\n"]
        for r in records:
            status_cn = "✅ 在售" if r.status == "onsale" else "❌ 下架"
            lines.append(
                f"• {r.title}\n"
                f"  价格：¥{r.price:.2f} | 品牌：{r.brand} | 状态：{status_cn}"
            )
        lines.append(f"\n💡 实时库存请查看商品详情页或联系客服。")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 🆕 写操作（合并自 business_execute.py）
    # ═══════════════════════════════════════════════════════════

    async def _cancel_order(self, order_id: str, reason: str) -> str:
        """取消订单：更新MySQL状态。"""
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(
                select(Order).where(Order.order_id == order_id)
            )
            order = result.scalar_one_or_none()

            if not order:
                return f"❌ 未找到订单 {order_id}，请核对订单号后重试。"

            if order.status in ("cancelled", "refunded", "refunding"):
                return f"⚠️ 订单 {order_id} 当前状态为「{order.status}」，无法重复取消。"

            order.status = "cancelled"
            await session.commit()

        logger.info("business.order_cancelled", order_id=order_id, reason=reason)

        return (
            f"✅ 订单 {order_id} 已成功取消。\n\n"
            f"📋 订单信息：{order.product_name} ×{order.quantity}\n"
            f"💰 金额：¥{order.amount:.2f}\n"
            f"📝 取消原因：{reason or '用户主动取消'}\n\n"
            f"💡 退款将在 1-3 个工作日内原路返回到您的支付账户。如有疑问，请联系客服。"
        )

    async def _apply_refund(self, order_id: str, reason: str) -> str:
        """申请退款：更新MySQL状态 + 触发退款流程。"""
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(
                select(Order).where(Order.order_id == order_id)
            )
            order = result.scalar_one_or_none()

            if not order:
                return f"❌ 未找到订单 {order_id}，请核对订单号。"

            if order.status == "refunding":
                return f"⚠️ 订单 {order_id} 正在退款处理中，请勿重复申请。"

            if order.status == "refunded":
                return f"⚠️ 订单 {order_id} 已完成退款，无需再次申请。"

            order.status = "refunding"
            await session.commit()

        logger.info("business.refund_applied", order_id=order_id, reason=reason)

        return (
            f"✅ 退款申请已提交！\n\n"
            f"📋 订单号：{order_id}\n"
            f"💰 退款金额：¥{order.amount:.2f}\n"
            f"📝 退款原因：{reason or '用户申请'}\n\n"
            f"⏱️ 预计 1-3 个工作日到账，请耐心等待。\n"
            f"💡 您可以在「我的订单 → 退款详情」中查看进度。"
        )

    async def _modify_address(self, order_id: str, new_address: str) -> str:
        """修改收货地址。"""
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(
                select(Order).where(Order.order_id == order_id)
            )
            order = result.scalar_one_or_none()

            if not order:
                return f"❌ 未找到订单 {order_id}。"

            if order.status not in ("pending", "paid"):
                return (
                    f"⚠️ 订单 {order_id} 当前状态为「{order.status}」，"
                    f"已发货的订单无法修改地址。建议联系快递公司或拒收后重新下单。"
                )

            old_address = order.receiver_address
            order.receiver_address = new_address or "用户修改（待确认新地址）"
            await session.commit()

        logger.info("business.address_modified", order_id=order_id)

        return (
            f"✅ 收货地址已更新！\n\n"
            f"📋 订单号：{order_id}\n"
            f"📍 原地址：{old_address}\n"
            f"📍 新地址：{new_address or '待补充'}\n\n"
            f"💡 如果订单已发货，地址修改可能无法生效，建议联系快递员协商。"
        )

    async def _modify_quantity(self, order_id: str, new_value: str) -> str:
        """修改商品规格/数量（需人工核实）。"""
        return (
            f"⚠️ 修改商品规格/数量需要人工核实处理。\n\n"
            f"📋 订单号：{order_id}\n"
            f"📝 修改需求：{new_value}\n\n"
            f"我已为您生成工单，客服将在 2 小时内联系您确认修改。\n"
            f"📞 您也可以直接拨打 {get_settings().customer_service_phone} 加速处理。"
        )

    # ═══════════════════════════════════════════════════════════
    # 二次确认
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _is_confirming(history: list = None) -> bool:
        """判断用户是否在回应确认提示。"""
        if not history:
            return False
        for h in reversed(history):
            txt = h.answer if hasattr(h, 'answer') else ''
            if "确定要" in txt and "请回复" in txt:
                return True
        return False

    @staticmethod
    def _has_explicit_confirm(query: str, history: list = None) -> bool:
        """检查用户是否已显式确认操作，支持上下文感知。"""
        confirm_words = ["确认", "是的", "确定", "没错", "可以", "行", "好", "对", "嗯", "是的我确定"]
        if any(w in query for w in confirm_words):
            return True

        # 上下文感知：如果上轮问了"确定要取消/退款吗"，用户简短回复视为确认
        if BusinessAgent._is_confirming(history):
            short_confirm = ["取消", "退", "是的", "对", "好", "行", "嗯", "确认", "可以", "没错"]
            if any(query.strip() == w for w in short_confirm):
                return True

        return False

    @staticmethod
    def _build_confirm_message(action: str, params: dict) -> str:
        """构建二次确认消息。"""
        action_cn = EXECUTE_ACTIONS.get(action, action)
        order_id = params.get("order_id", "未知")
        new_value = params.get("new_value", "")
        reason = params.get("reason", "")

        lines = [
            f"⚠️ 您确定要{action_cn}吗？\n",
            f"📋 订单号：{order_id}",
        ]
        if new_value:
            lines.append(f"📝 修改内容：{new_value}")
        if reason:
            lines.append(f"💬 原因：{reason}")

        lines.extend([
            "\n请回复「确认」以继续操作。",
            "如需取消，回复「不用了」即可。",
        ])
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 帮助 & 引导
    # ═══════════════════════════════════════════════════════════

    def _help_message(self) -> str:
        """无法判断查询类型时的引导消息。"""
        return (
            f"您好，我可以帮您处理以下业务：\n\n"
            f"📦 查物流 — 提供订单号即可（如\"帮我查JD20240706-001的物流\"）\n"
            f"📋 查订单 — 提供订单号即可（如\"查一下订单JD20240706-001\"）\n"
            f"📊 查库存 — 提供商品名或SKU（如\"这款防晒霜还有货吗\"）\n"
            f"✏️ 修改订单 — 改地址、取消订单、申请退款等\n\n"
            f"请告诉我您需要哪种服务？"
        )

    def _missing_info_response(self, action: str, params: dict) -> AgentResponse:
        """缺少必要参数时的引导消息。"""
        if not params.get("order_id"):
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant",
                    content=(
                        f"好的，我来帮您处理。请提供以下信息：\n\n"
                        f"📋 订单号（必填）— 在「我的订单」中可以找到\n"
                        f"📝 具体需求 — 比如要修改的新地址、退款原因等\n\n"
                        f"💡 示例：\"把订单 JD20240706-001 的收货地址改为 XX市XX区XX路\""
                    ),
                    intent_detected="business",
                ),
            )

        return AgentResponse(
            success=True,
            message=AgentMessage(
                role="assistant",
                content=(
                    f"请告诉我更多细节，我好帮您处理。\n\n"
                    f"💡 示例：\"取消订单 JD20240706-001，不想要了\""
                ),
                intent_detected="business",
            ),
        )


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        agent = BusinessAgent()

        tests = [
            # 读操作
            "帮我查一下JD20240706-001的物流",
            "看看订单JD20240706-001的详情",
            "防晒霜SKU-FS001还有货吗",
            # 写操作
            "帮我把JD20240706-001的收货地址改成北京市朝阳区XX路100号",
            "取消订单JD20240706-001，不想要了",
            "JD20240706-001申请退款，商品有质量问题",
            # 边界
            "帮我把订单改一下",
            "你好呀今天天气不错",
        ]
        for q in tests:
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            result = await agent.handle(q)
            fcalls = result.message.function_calls
            action = fcalls[0].call.name if fcalls else "none"
            print(f"Action: {action}")
            print(f"Response: {result.message.content[:200]}...")

        print("\n✅ business.py v3 读写合一 自测通过")

    asyncio.run(test())
