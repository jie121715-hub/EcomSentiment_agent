# backend/utils/logistics.py
# 物流轨迹查询：快递鸟 API（支持600+快递公司）。
#
# 配置：在 .env.local 中设置 KDN_EBUSINESS_ID 和 KDN_API_KEY
# 注册：https://www.kdniao.com/ → 用户中心获取凭证
#
# 常用快递编码：
#   SF: 顺丰, YTO: 圆通, ZTO: 中通, STO: 申通, YD: 韵达
#   HTKY: 百世汇通, EMS: EMS, JD: 京东物流, DBL: 德邦

import hashlib
import base64
import json
import httpx
from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

KDN_API_URL = "https://api.kdniao.com/Ebusiness/EbusinessOrderHandle.aspx"

# ── 常用快递公司编码（单号特征识别 + 用户指定）──
EXPRESS_COMPANIES = {
    "SF": "顺丰速运", "YTO": "圆通速递", "ZTO": "中通快递",
    "STO": "申通快递", "YD": "韵达快递", "HTKY": "百世汇通",
    "EMS": "EMS", "JD": "京东物流", "DBL": "德邦物流",
    "YZPY": "邮政包裹", "UC": "优速快递", "QFKD": "全峰快递",
    "HHTT": "天天快递", "JTSD": "极兔速递", "DHL": "DHL",
}


def _kdniao_sign(request_data: str, api_key: str) -> str:
    """快递鸟签名：Base64(MD5_Hex(request_data + api_key))。"""
    raw = request_data + api_key
    md5_hex = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return base64.b64encode(md5_hex.encode("utf-8")).decode("utf-8")


def _guess_express_code(tracking_no: str) -> str:
    """根据快递单号特征猜测快递公司（兜底用 SF）。"""
    tn = tracking_no.strip()
    # 顺丰：12位数字 或 SF开头
    if len(tn) == 12 and tn.isdigit():
        return "SF"
    if tn.upper().startswith("SF"):
        return "SF"
    # 京东：JD开头
    if tn.upper().startswith("JD"):
        return "JD"
    # EMS：13位，字母开头
    if len(tn) == 13 and tn[0].isalpha() and tn[1:].isdigit():
        return "EMS"
    # 圆通：YT开头 或 10位数字
    if tn.upper().startswith("YT"):
        return "YTO"
    # 中通：7开头15位 或 ZTO开头
    if (len(tn) == 15 and tn.startswith("7")) or tn.upper().startswith("ZTO"):
        return "ZTO"
    # 申通：STO开头 或 368/468开头12位
    if tn.upper().startswith("STO"):
        return "STO"
    # 韵达：YD开头 或 10开头13位
    if tn.upper().startswith("YD"):
        return "YD"
    # 极兔：JT开头
    if tn.upper().startswith("JT"):
        return "JTSD"
    return "SF"  # 默认尝试顺丰


async def query_logistics(tracking_no: str, express_code: str = "") -> dict:
    """查询物流轨迹（快递鸟API）。

    参数:
        tracking_no: 快递单号
        express_code: 快递公司编码（空则自动识别）

    返回:
        {
            "success": True/False,
            "company": "圆通速递",
            "tracking_no": "YT1234567890",
            "state": "已签收",          # 0=无轨迹 1=已揽收 2=在途中 3=签收 4=问题件
            "traces": [
                {"time": "2024-07-08 14:30:00", "desc": "已签收，签收人：本人"},
                ...
            ],
            "error": ""
        }
    """
    settings = get_settings()

    if not (settings.kdn_ebusiness_id and settings.kdn_api_key):
        return {"success": False, "error": "快递鸟未配置（KDN_EBUSINESS_ID / KDN_API_KEY 为空）"}

    # 自动识别快递公司
    if not express_code:
        express_code = _guess_express_code(tracking_no)

    request_data = json.dumps({
        "LogisticCode": tracking_no,
    })

    data_sign = _kdniao_sign(request_data, settings.kdn_api_key)

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
            resp = await client.post(KDN_API_URL, data={
                "RequestData": request_data,
                "EBusinessID": settings.kdn_ebusiness_id,
                "RequestType": "8002",  # 即时查询接口
                "DataSign": data_sign,
                "DataType": 2,            # JSON 返回
            })
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.error("kdniao_timeout", tracking_no=tracking_no)
        return {"success": False, "error": "物流查询超时，请稍后重试"}
    except Exception as e:
        logger.error("kdniao_request_failed", error=str(e))
        return {"success": False, "error": f"物流查询失败: {e}"}

    # 解析返回
    if data.get("Success"):
        shipper = EXPRESS_COMPANIES.get(data.get("ShipperCode", ""), data.get("ShipperCode", "未知"))
        state_map = {"0": "暂无轨迹", "1": "已揽收", "2": "运输中", "3": "已签收", "4": "问题件"}
        traces_raw = data.get("Traces", [])
        traces = [
            {"time": t.get("AcceptTime", ""), "desc": t.get("AcceptStation", "")}
            for t in traces_raw
        ]
        logger.info("kdniao_success", tracking_no=tracking_no, company=shipper, traces=len(traces))
        return {
            "success": True,
            "company": shipper,
            "tracking_no": tracking_no,
            "state": state_map.get(str(data.get("State", "0")), "未知"),
            "traces": traces,
        }
    else:
        reason = data.get("Reason", "未知错误")
        logger.error("kdniao_biz_error", tracking_no=tracking_no, reason=reason)
        # 如果自动识别的快递公司不对，尝试用顺丰再查一次
        if express_code != "SF":
            logger.info("kdniao_retry_sf", tracking_no=tracking_no)
            return await query_logistics(tracking_no, "SF")
        return {"success": False, "error": reason}


def format_trace(tracking_no: str, result: dict) -> str:
    """将物流查询结果格式化为用户可读的文本。"""
    if not result.get("success"):
        error = result.get("error", "未知错误")
        return (
            f"📦 物流单号：{tracking_no}\n\n"
            f"⚠️ 暂时无法获取物流信息：{error}\n\n"
            f"💡 建议：\n"
            f"• 核对单号是否正确\n"
            f"• 在「我的订单 → 物流详情」中查看\n"
            f"• 复制单号到快递官网查询\n"
            f"• 联系客服：{get_settings().customer_service_phone}"
        )

    traces = result.get("traces", [])
    company = result.get("company", "未知")
    state = result.get("state", "未知")

    lines = [
        f"📦 物流查询结果",
        f"",
        f"🏢 快递公司：{company}",
        f"🔢 快递单号：{tracking_no}",
        f"📌 当前状态：{state}",
        f"",
        f"📍 物流轨迹：",
    ]

    if not traces:
        lines.append("  ⏳ 暂无轨迹信息，请稍后再查")
    else:
        for i, t in enumerate(traces[:15]):  # 最近15条
            icon = "🟢" if i == 0 else "🔵"
            lines.append(f"  {icon} {t['time']}  {t['desc']}")

    lines.extend([
        "",
        f"💡 如有疑问请联系客服：{get_settings().customer_service_phone}",
    ])
    return "\n".join(lines)
