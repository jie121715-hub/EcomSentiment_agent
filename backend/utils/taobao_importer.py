# backend/utils/taobao_importer.py
# 淘宝开放平台商品导入器：taobao.items.onsale.get → MySQL → 向量库。
#
# 用法一（API）：POST /admin/knowledge/import-taobao
#   { "app_key": "...", "app_secret": "...", "session_key": "...", "merchant_id": "my_shop" }
#
# 用法二（命令行）：
#   python backend/taobao_importer.py --app-key=xxx --app-secret=xxx --session=xxx
#
# 前置条件：需要在 open.taobao.com 注册应用并获取授权。

import hashlib, time, httpx, json, sys, os
from dataclasses import dataclass
from backend.core.logger import get_logger

logger = get_logger(__name__)

TB_API_URL = "https://eco.taobao.com/router/rest"


@dataclass
class TbConfig:
    app_key: str
    app_secret: str
    session_key: str          # 卖家授权token
    merchant_id: str = "taobao_shop"
    page_size: int = 50


def _sign(params: dict, secret: str) -> str:
    """淘宝 API 签名算法（MD5）。"""
    sorted_params = sorted(params.items())
    sign_str = secret + "".join(f"{k}{v}" for k, v in sorted_params) + secret
    return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()


def _build_params(config: TbConfig, method: str, extra: dict = None) -> dict:
    """构建淘宝 API 请求参数。"""
    params = {
        "method": method,
        "app_key": config.app_key,
        "session": config.session_key,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "json",
        "v": "2.0",
        "sign_method": "md5",
        "fields": "num_iid,title,nick,pic_url,price,item_url,props_name,quantity,sell_point,desc",
        "page_size": str(config.page_size),
        "page_no": "1",
    }
    if extra:
        params.update(extra)
    params["sign"] = _sign(params, config.app_secret)
    return params


def _taobao_item_to_knowledge(item: dict, merchant_id: str) -> str:
    """将淘宝商品转为知识库文本。"""
    title = item.get("title", "未知商品")
    price = item.get("price", "?")
    props = item.get("props_name", "").replace(":", ": ").replace(";", "，")
    sell_point = item.get("sell_point", "") or ""
    desc = (item.get("desc", "") or "")[:80]

    parts = [f"{title}，售价{price}元。"]
    if props:
        parts.append(f"规格：{props}。")
    if sell_point:
        parts.append(f"卖点：{sell_point}。")
    if desc:
        parts.append(f"描述：{desc}。")

    return "".join(parts)


async def import_products(config: TbConfig) -> dict:
    """从淘宝 API 导入在售商品到 MySQL + 向量库。

    流程：
    1. 调 taobao.items.onsale.get 获取在售商品列表
    2. 每个商品转为知识库文本
    3. 批量写入 MySQL
    4. 全量同步向量库
    """
    logger.info("taobao_importer.started", merchant=config.merchant_id)

    # Step 1: 调淘宝 API 获取商品列表
    params = _build_params(config, "taobao.items.onsale.get")
    imported = 0

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            resp = await client.post(TB_API_URL, data=params)
            data = resp.json()
    except Exception as e:
        logger.error("taobao_importer.api_error", error=str(e))
        return {"success": False, "error": f"淘宝 API 调用失败: {e}"}

    # 解析返回
    items_key = "taobao_items_onsale_get_response"
    if items_key not in data:
        err = data.get("error_response", {}).get("msg", str(data)[:200])
        logger.error("taobao_importer.api_biz_error", error=err)
        return {"success": False, "error": f"淘宝 API 业务错误: {err}"}

    items = data[items_key].get("items", {}).get("item", [])
    if not items:
        logger.info("taobao_importer.no_items")
        return {"success": True, "imported": 0, "message": "店铺无在售商品"}

    # 确保是列表
    if isinstance(items, dict):
        items = [items]

    logger.info("taobao_importer.fetched", count=len(items))

    # Step 2: 转为知识库文本，批量写入 MySQL
    from backend.core.database import get_session, init_db
    from backend.models.db_models import CustomKnowledge

    await init_db()

    async with get_session() as session:
        for item in items:
            content = _taobao_item_to_knowledge(item, config.merchant_id)
            session.add(CustomKnowledge(
                content=content,
                source=f"taobao:{config.merchant_id}",
                category="product",
                merchant_id=config.merchant_id,
            ))
            imported += 1
        await session.commit()

    logger.info("taobao_importer.saved_to_mysql", count=imported)

    logger.info("taobao_importer.done", imported=imported)
    return {
        "success": True,
        "imported": imported,
        "sync": sync_result,
    }


async def search_products(keyword: str, config: TbConfig = None) -> list[dict]:
    """淘宝商品搜索（RAG兜底）：根据关键词搜索在售商品。

    返回: [{"title": "...", "price": "...", "pic_url": "...", "item_url": "..."}, ...]
    失败返回空列表。
    """
    if config is None:
        from backend.config import get_settings
        s = get_settings()
        if not (s.taobao_app_key and s.taobao_app_secret and s.taobao_session_key):
            return []
        config = TbConfig(
            app_key=s.taobao_app_key,
            app_secret=s.taobao_app_secret,
            session_key=s.taobao_session_key,
        )

    # 用 items.onsale.get + 关键词过滤（淘宝API不支持直接全文搜索，用标题模糊匹配做客户端过滤）
    params = _build_params(config, "taobao.items.onsale.get", extra={"page_size": "50"})

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
            resp = await client.post(TB_API_URL, data=params)
            data = resp.json()
    except Exception as e:
        logger.error("taobao_search.api_error", error=str(e))
        return []

    items_key = "taobao_items_onsale_get_response"
    if items_key not in data:
        return []

    items = data[items_key].get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not items:
        return []

    # 客户端关键词过滤
    kw_lower = keyword.lower()
    matched = []
    for item in items:
        title = (item.get("title") or "").lower()
        if kw_lower in title or any(w in title for w in kw_lower.split()):
            matched.append({
                "title": item.get("title", ""),
                "price": item.get("price", ""),
                "pic_url": item.get("pic_url", ""),
                "item_url": item.get("item_url", ""),
                "props": item.get("props_name", ""),
            })
        if len(matched) >= 5:  # 最多5条
            break

    logger.info("taobao_search.done", keyword=keyword, found=len(matched))
    return matched


# ── 命令行入口 ──
if __name__ == "__main__":
    import asyncio, argparse

    p = argparse.ArgumentParser(description="从淘宝 API 导入商品到知识库")
    p.add_argument("--app-key", required=True, help="淘宝 App Key")
    p.add_argument("--app-secret", required=True, help="淘宝 App Secret")
    p.add_argument("--session", required=True, help="卖家授权 Session Key")
    p.add_argument("--merchant-id", default="taobao_shop", help="商户标识")
    p.add_argument("--page-size", type=int, default=50)

    args = p.parse_args()
    config = TbConfig(
        app_key=args.app_key,
        app_secret=args.app_secret,
        session_key=args.session,
        merchant_id=args.merchant_id,
        page_size=args.page_size,
    )
    result = asyncio.run(import_products(config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
