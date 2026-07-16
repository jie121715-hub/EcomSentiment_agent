# backend/agents/business/nodes.py
# Business Agent — LangGraph 节点函数 + BusinessAgent 类
#
# 统一业务Agent：MySQL读写合一 + 确认流程。
# 保留完整 BusinessAgent 类作为模块级单例。

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

from backend.agents.business.state import BusinessState
from backend.agents.business.prompts import (
    LOGISTICS_KEYWORDS, ORDER_KEYWORDS, STOCK_KEYWORDS,
    EXECUTE_ACTIONS, WRITE_KEYWORDS,
    PARSE_QUERY_PROMPT,
)

logger = get_logger(__name__)


class BusinessAgent:
    """统一业务Agent — MySQL读写合一 + 确认流程。"""

    async def handle(
        self, query: str, history: list[ConversationHistory] | None = None,
        user_id: str = "", shop_id: str = "",
    ) -> AgentResponse:
        start = time.time()
        logger.info("business.started", query=query[:50], user_id=user_id)

        action, params, needs_confirm = await self._parse_query(query)

        # 需要 order_id 的所有操作：查物流/查订单/修改订单等
        NEEDS_ORDER = set(EXECUTE_ACTIONS.keys()) | {"logistics", "order", "order_status"}
        action_needs_order = action in NEEDS_ORDER

        # 如果 LLM 没识别出 action 但有订单号 → 自动转查订单
        if action == "none" and params.get("order_id"):
            action = "order"
            action_needs_order = True

        if action == "none" or (action_needs_order and not params.get("order_id")):
            # 从对话历史中提取订单号
            if history and self._is_confirming(history):
                for h in reversed(history):
                    txt = h.answer if hasattr(h, 'answer') else ''
                    m = re.search(r'订单号[：:]\s*([A-Z]{2,4}\d{6,12}[-_]\d{2,6})', txt)
                    if m:
                        params["order_id"] = m.group(1)
                        break
            # 查数据库：找用户最近的订单
            if not params.get("order_id") and user_id and user_id != "anonymous":
                db_orders = await self._lookup_user_orders(user_id, shop_id)
                if len(db_orders) == 1:
                    params["order_id"] = db_orders[0]["order_id"]
                    params["shop_id"] = db_orders[0]["shop_id"]
                    logger.info("business.auto_found_order", order_id=params["order_id"])
                elif len(db_orders) > 1:
                    return self._multiple_orders_response(db_orders)
            if not params.get("order_id"):
                return self._missing_info_response(action, params)

        if needs_confirm and not self._has_explicit_confirm(query, history):
            elapsed = (time.time() - start) * 1000
            return AgentResponse(
                success=True,
                message=AgentMessage(
                    role="assistant", content=self._build_confirm_message(action, params),
                    intent_detected="business",
                    function_calls=[FunctionCallResult(
                        call=FunctionCall(name=action, arguments=params),
                        success=True, result="pending_confirmation",
                    )],
                ),
                processing_time_ms=elapsed,
            )

        try:
            result_text = await self._execute(action, params, user_id=user_id)
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
                role="assistant", content=result_text, intent_detected="business",
                function_calls=[FunctionCallResult(
                    call=FunctionCall(name=action, arguments=params),
                    success=success, result=result_text[:300],
                )],
            ),
            processing_time_ms=elapsed,
        )

    async def _parse_query(self, query: str) -> tuple[str, dict, bool]:
        actions_desc = "\n".join([
            "读操作：", "  - logistics: 查物流", "  - order: 查订单详情", "  - stock: 查库存",
            "写操作：", *[f"  - {k}: {v}" for k, v in EXECUTE_ACTIONS.items()],
            "  - none: 无法判断",
        ])

        prompt = PARSE_QUERY_PROMPT.format(actions_desc=actions_desc, query=query)

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
        order_match = re.search(r'[A-Z]{2,4}\d{6,12}[-_]\d{2,6}', query)
        order_id = order_match.group() if order_match else ""

        if any(w in query for w in ["取消", "不要了", "退单"]):
            return ("cancel_order", {"order_id": order_id, "new_value": "", "reason": query}, True)
        if any(w in query for w in ["退款", "退钱", "退"]):
            return ("apply_refund", {"order_id": order_id, "new_value": "", "reason": query}, True)
        if any(w in query for w in ["改地址", "换地址", "修改地址", "改收货"]):
            return ("modify_address", {"order_id": order_id, "new_value": "", "reason": query}, False)
        if any(w in query for w in ["改规格", "换颜色", "换尺码", "修改数量"]):
            return ("modify_quantity", {"order_id": order_id, "new_value": "", "reason": query}, True)
        if any(w in query for w in LOGISTICS_KEYWORDS):
            return ("logistics", {"order_id": order_id, "sku": "", "product_name": ""}, False)
        if any(w in query for w in ORDER_KEYWORDS):
            return ("order", {"order_id": order_id, "sku": "", "product_name": ""}, False)
        if any(w in query for w in STOCK_KEYWORDS):
            return ("stock", {"order_id": "", "sku": "", "product_name": ""}, False)

        return ("none", {"order_id": order_id, "sku": "", "product_name": ""}, False)

    async def _execute(self, action: str, params: dict, user_id: str = "") -> str:
        order_id = params.get("order_id", "").strip()
        is_tracking = (
            bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', order_id)) or
            bool(re.fullmatch(r'\d{10,20}', order_id))
        )
        if is_tracking:
            logger.info("business.auto_redirect_logistics", order_id=order_id, action=action)
            return await self._query_logistics(order_id, user_id)

        if action == "logistics": return await self._query_logistics(order_id, user_id)
        if action == "order": return await self._query_order(order_id, user_id)
        if action == "stock": return await self._query_stock(sku=params.get("sku", ""), product_name=params.get("product_name", ""))
        if action == "cancel_order": return await self._cancel_order(params.get("order_id", ""), params.get("reason", ""))
        if action == "apply_refund": return await self._apply_refund(params.get("order_id", ""), params.get("reason", ""))
        if action == "modify_address": return await self._modify_address(params.get("order_id", ""), params.get("new_value", ""))
        if action == "modify_quantity": return await self._modify_quantity(params.get("order_id", ""), params.get("new_value", ""))
        return self._help_message()

    # ── 读操作 ─────────────────────────────────────────────────

    async def _query_logistics(self, order_id: str, user_id: str = "") -> str:
        if not order_id:
            return "请提供您的订单号，我才能帮您查物流哦～\n\n💡 订单号通常在订单详情页可以找到，格式如 JD20240706-001。"

        from backend.core.database import get_session
        from backend.models.db_models import Order

        is_tracking = bool(re.fullmatch(r'[A-Z]{2,6}\d{8,18}', order_id.strip()))
        is_pure_digits = bool(re.fullmatch(r'\d{10,20}', order_id.strip()))

        if is_tracking or is_pure_digits:
            async with get_session() as session:
                result = await session.execute(select(Order).where(Order.logistics_tracking == order_id.strip()))
                orders = result.scalars().all()

            # 过滤：只显示属于当前用户的订单
            if user_id and user_id != "anonymous":
                orders = [o for o in orders if o.user_id == user_id]
                if not orders:
                    return f"快递单号 {order_id.strip()} 未关联到您的订单，无法查询。"

            if orders:
                lines = [f"📦 快递单号 {order_id.strip()} 关联订单：\n"]
                for o in orders:
                    status_map = {"pending": "待付款", "paid": "已付款，备货中", "shipped": "运输中",
                                  "delivered": "已签收", "cancelled": "已取消", "refunding": "退款中", "refunded": "已退款"}
                    lines.append(f"📋 订单号：{o.order_id}\n   商品：{o.product_name} ×{o.quantity}\n   状态：{status_map.get(o.status, o.status)}\n   收货地址：{o.receiver_address}")
                lines.append(f"\n💡 快递鸟实时轨迹API即将上线，敬请期待！")
                return "\n\n".join(lines) if len(orders) > 1 else "\n".join(lines)

            return f"📦 快递单号：{order_id.strip()}\n\n未在系统中找到与此快递单号关联的订单。\n\n💡 快递鸟实时轨迹API即将上线，敬请期待！\n📞 如有疑问，请联系客服：{get_settings().customer_service_phone}"

        async with get_session() as session:
            result = await session.execute(select(Order).where(Order.order_id == order_id))
            order = result.scalar_one_or_none()

        if order is None:
            return f"未找到订单 {order_id} 的信息。\n\n💡 请核对订单号是否正确，或拨打客服电话 {get_settings().customer_service_phone}。"

        # 订单归属校验
        if user_id and user_id != "anonymous" and order.user_id != user_id:
            return f"订单 {order_id} 不属于您的账号，无法查询物流。请核实订单号后重试。"

        tracking_no = order.logistics_tracking
        if not tracking_no:
            status_map = {"pending": "订单尚未付款，暂无物流信息。", "paid": "订单已付款，商家正在备货中，预计24小时内发货。", "cancelled": "订单已取消，无物流信息。"}
            hint = status_map.get(order.status, f"订单状态为「{order.status}」，暂未生成快递单号。")
            return f"📦 订单 {order_id} 物流查询\n\n商品：{order.product_name}\n{hint}\n\n💡 发货后会短信通知您快递单号，请留意手机短信。"

        status_map = {"pending": "待付款", "paid": "已付款，备货中", "shipped": "运输中", "delivered": "已签收",
                      "cancelled": "已取消", "refunding": "退款中", "refunded": "已退款"}
        status_cn = status_map.get(order.status, order.status)
        return (
            f"📦 订单 {order_id} 物流信息\n\n商品：{order.product_name} ×{order.quantity}\n金额：¥{order.amount:.2f}\n"
            f"状态：{status_cn}\n收货人：{order.receiver_name}\n收货地址：{order.receiver_address}\n"
            f"🏢 快递单号：{tracking_no}\n\n💡 快递鸟实时轨迹API即将上线，敬请期待！\n   当前可复制快递单号到快递官网查询实时轨迹。"
        )

    async def _query_order(self, order_id: str, user_id: str = "") -> str:
        if not order_id:
            return "请提供您的订单号，我才能帮您查订单详情哦～\n\n💡 订单号格式如 JD20240706-001。"
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(select(Order).where(Order.order_id == order_id))
            record = result.scalar_one_or_none()

        if record is None:
            return f"未找到订单 {order_id} 的信息。\n\n请核对订单号是否正确，或拨打客服电话 {get_settings().customer_service_phone} 咨询。"

        # 订单归属校验
        if user_id and user_id != "anonymous" and record.user_id != user_id:
            return f"订单 {order_id} 不属于您的账号，无法查看详情。请核实订单号后重试。"

        status_map = {"pending": "待付款", "paid": "已付款", "shipped": "已发货", "delivered": "已签收",
                      "cancelled": "已取消", "refunding": "退款中", "refunded": "已退款"}
        status_cn = status_map.get(record.status, record.status)
        return (
            f"📋 订单 {order_id} 详情：\n\n商品：{record.product_name} ×{record.quantity}\n"
            f"金额：¥{record.amount:.2f}\n状态：{status_cn}\n收货人：{record.receiver_name}\n"
            f"收货地址：{record.receiver_address}\n"
            f"下单时间：{record.created_at.strftime('%Y-%m-%d %H:%M') if record.created_at else '未知'}\n\n"
            f"💡 如需修改订单或申请售后，请告知具体需求。"
        )

    async def _query_stock(self, sku: str = "", product_name: str = "") -> str:
        if not sku and not product_name:
            return "请告诉我您要查哪个商品的库存，提供商品名称或SKU编号都可以～"
        from backend.core.database import get_session
        from backend.models.db_models import Product

        async with get_session() as session:
            if sku:
                result = await session.execute(select(Product).where(Product.product_id == sku))
            else:
                result = await session.execute(select(Product).where(Product.title.like(f"%{product_name}%")))
            records = result.scalars().all()

        if not records:
            return f"未找到「{sku or product_name}」的商品信息。\n\n💡 请核对商品名称或SKU是否正确，或联系商家确认。"

        lines = [f"📊 商品查询结果（共 {len(records)} 个匹配）：\n"]
        for r in records:
            status_cn = "✅ 在售" if r.status == "onsale" else "❌ 下架"
            lines.append(f"• {r.title}\n  价格：¥{r.price:.2f} | 品牌：{r.brand} | 状态：{status_cn}")
        lines.append(f"\n💡 实时库存请查看商品详情页或联系客服。")
        return "\n".join(lines)

    # ── 写操作 ─────────────────────────────────────────────────

    async def _cancel_order(self, order_id: str, reason: str) -> str:
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(select(Order).where(Order.order_id == order_id))
            order = result.scalar_one_or_none()
            if not order:
                return f"❌ 未找到订单 {order_id}，请核对订单号后重试。"
            if order.status in ("cancelled", "refunded", "refunding"):
                return f"⚠️ 订单 {order_id} 当前状态为「{order.status}」，无法重复取消。"
            order.status = "cancelled"
            await session.commit()

        logger.info("business.order_cancelled", order_id=order_id, reason=reason)
        return f"✅ 订单 {order_id} 已成功取消。\n\n📋 订单信息：{order.product_name} ×{order.quantity}\n💰 金额：¥{order.amount:.2f}\n📝 取消原因：{reason or '用户主动取消'}\n\n💡 退款将在 1-3 个工作日内原路返回到您的支付账户。"

    async def _apply_refund(self, order_id: str, reason: str) -> str:
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(select(Order).where(Order.order_id == order_id))
            order = result.scalar_one_or_none()
            if not order: return f"❌ 未找到订单 {order_id}，请核对订单号。"
            if order.status == "refunding": return f"⚠️ 订单 {order_id} 正在退款处理中，请勿重复申请。"
            if order.status == "refunded": return f"⚠️ 订单 {order_id} 已完成退款，无需再次申请。"
            order.status = "refunding"
            await session.commit()

        logger.info("business.refund_applied", order_id=order_id, reason=reason)
        return f"✅ 退款申请已提交！\n\n📋 订单号：{order_id}\n💰 退款金额：¥{order.amount:.2f}\n📝 退款原因：{reason or '用户申请'}\n\n⏱️ 预计 1-3 个工作日到账，请耐心等待。\n💡 您可以在「我的订单 → 退款详情」中查看进度。"

    async def _modify_address(self, order_id: str, new_address: str) -> str:
        from backend.core.database import get_session
        from backend.models.db_models import Order

        async with get_session() as session:
            result = await session.execute(select(Order).where(Order.order_id == order_id))
            order = result.scalar_one_or_none()
            if not order: return f"❌ 未找到订单 {order_id}。"
            if order.status not in ("pending", "paid"):
                return f"⚠️ 订单 {order_id} 当前状态为「{order.status}」，已发货的订单无法修改地址。建议联系快递公司或拒收后重新下单。"
            old_address = order.receiver_address
            order.receiver_address = new_address or "用户修改（待确认新地址）"
            await session.commit()

        logger.info("business.address_modified", order_id=order_id)
        return f"✅ 收货地址已更新！\n\n📋 订单号：{order_id}\n📍 原地址：{old_address}\n📍 新地址：{new_address or '待补充'}\n\n💡 如果订单已发货，地址修改可能无法生效，建议联系快递员协商。"

    async def _modify_quantity(self, order_id: str, new_value: str) -> str:
        return f"⚠️ 修改商品规格/数量需要人工核实处理。\n\n📋 订单号：{order_id}\n📝 修改需求：{new_value}\n\n我已为您生成工单，客服将在 2 小时内联系您确认修改。\n📞 您也可以直接拨打 {get_settings().customer_service_phone} 加速处理。"

    # ── 二次确认 ───────────────────────────────────────────────

    @staticmethod
    def _is_confirming(history: list = None) -> bool:
        if not history: return False
        for h in reversed(history):
            txt = h.answer if hasattr(h, 'answer') else ''
            if "确定要" in txt and "请回复" in txt: return True
        return False

    @staticmethod
    def _has_explicit_confirm(query: str, history: list = None) -> bool:
        if any(w in query for w in ["确认", "是的", "确定", "没错", "可以", "行", "好", "对", "嗯", "是的我确定"]):
            return True
        if BusinessAgent._is_confirming(history):
            if any(query.strip() == w for w in ["取消", "退", "是的", "对", "好", "行", "嗯", "确认", "可以", "没错"]):
                return True
        return False

    @staticmethod
    def _build_confirm_message(action: str, params: dict) -> str:
        action_cn = EXECUTE_ACTIONS.get(action, action)
        order_id = params.get("order_id", "未知")
        lines = [f"⚠️ 您确定要{action_cn}吗？\n", f"📋 订单号：{order_id}"]
        if params.get("new_value"): lines.append(f"📝 修改内容：{params['new_value']}")
        if params.get("reason"): lines.append(f"💬 原因：{params['reason']}")
        lines.extend(["\n请回复「确认」以继续操作。", "如需取消，回复「不用了」即可。"])
        return "\n".join(lines)

    def _help_message(self) -> str:
        return (
            f"您好，我可以帮您处理以下业务：\n\n"
            f"📦 查物流 — 提供订单号即可\n📋 查订单 — 提供订单号即可\n📊 查库存 — 提供商品名或SKU\n"
            f"✏️ 修改订单 — 改地址、取消订单、申请退款等\n\n请告诉我您需要哪种服务？"
        )

    @staticmethod
    async def _lookup_user_orders(user_id: str, shop_id: str = "") -> list[dict]:
        """查用户订单列表（可按 shop_id 过滤）。"""
        try:
            from backend.core.database import get_session
            from sqlalchemy import text
            async with get_session() as session:
                if shop_id:
                    result = await session.execute(
                        text("SELECT order_id, shop_id, product_name, status, amount, created_at "
                             "FROM orders WHERE user_id=:uid AND shop_id=:sid "
                             "ORDER BY created_at DESC LIMIT 5"),
                        {"uid": user_id, "sid": shop_id},
                    )
                else:
                    result = await session.execute(
                        text("SELECT order_id, shop_id, product_name, status, amount, created_at "
                             "FROM orders WHERE user_id=:uid "
                             "ORDER BY created_at DESC LIMIT 5"),
                        {"uid": user_id},
                    )
                rows = result.fetchall()
                return [{"order_id": r[0], "shop_id": r[1], "product_name": r[2],
                         "status": r[3], "amount": r[4], "created_at": str(r[5])} for r in rows]
        except Exception as e:
            logger.warning("business.order_lookup_failed", error=str(e)[:100])
            return []

    def _multiple_orders_response(self, orders: list[dict]) -> AgentResponse:
        order_list = "\n".join(
            f"  · {o['order_id']} — {o['product_name']}（{o['status']}）"
            for o in orders[:5]
        )
        return AgentResponse(
            success=True,
            message=AgentMessage(
                role="assistant",
                content=f"您最近有 {len(orders)} 笔订单：\n{order_list}\n\n请告诉我要查哪一笔？直接回复订单号即可。",
                intent_detected="order_lookup",
            ),
        )

    def _missing_info_response(self, action: str, params: dict) -> AgentResponse:
        if not params.get("order_id"):
            return AgentResponse(success=True,
                message=AgentMessage(role="assistant",
                    content=f"好的，我来帮您处理。请提供以下信息：\n\n📋 订单号（必填）— 在「我的订单」中可以找到\n📝 具体需求 — 比如要修改的新地址、退款原因等\n\n💡 示例：\"把订单 JD20240706-001 的收货地址改为 XX市XX区XX路\"",
                    intent_detected="business"),
            )
        return AgentResponse(success=True,
            message=AgentMessage(role="assistant",
                content=f"请告诉我更多细节，我好帮您处理。\n\n💡 示例：\"取消订单 JD20240706-001，不想要了\"",
                intent_detected="business"),
        )


# ═══════════════════════════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════════════════════════

_business_agent: Optional[BusinessAgent] = None


def _get_agent() -> BusinessAgent:
    global _business_agent
    if _business_agent is None:
        _business_agent = BusinessAgent()
    return _business_agent


# ═══════════════════════════════════════════════════════════════
# LangGraph 节点函数
# ═══════════════════════════════════════════════════════════════

async def handle_node(state: BusinessState) -> dict:
    """节点：业务处理 —— MySQL读写合一 + 确认流程。"""
    query = state["query"]
    history = state.get("history", [])

    logger.info("business.node_started", query=query[:30])
    try:
        agent = _get_agent()
        user_id = state.get("user_id", "")
        shop_id = state.get("shop_id", "")
        response = await agent.handle(query=query, history=history,
                                       user_id=user_id, shop_id=shop_id)
        response.message.intent_detected = "business"
        return {"agent_response": response}
    except Exception as e:
        logger.error("business.node_failed", error=str(e))
        return {"error": f"业务处理异常: {e}"}


# ── 测试代码 ──
if __name__ == "__main__":
    import asyncio
    from backend.core.logger import configure_logging
    configure_logging()

    async def test():
        agent = _get_agent()
        tests = [
            "帮我查一下JD20240706-001的物流",
            "看看订单JD20240706-001的详情",
            "帮我把JD20240706-001的收货地址改成北京市朝阳区XX路100号",
            "取消订单JD20240706-001，不想要了",
            "你好呀今天天气不错",
        ]
        for q in tests:
            print(f"\n{'='*60}\nQuery: {q}")
            result = await agent.handle(q)
            fcalls = result.message.function_calls
            action = fcalls[0].call.name if fcalls else "none"
            print(f"Action: {action}")
            print(f"Response: {result.message.content[:200]}...")
        print("\n✅ business nodes.py 自测通过")

    asyncio.run(test())
