"""
本地商品种子数据 — 模拟淘宝店铺商品，写入 MySQL + Milvus

用法:
  python -m backend.seed_products                    # 导入默认数据
  python -m backend.seed_products --shop shop_001    # 指定店铺
"""

import asyncio, argparse
from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── 模拟商品数据（模拟淘宝店铺在售商品）─────────────────

MOCK_PRODUCTS = [
    # (title, price, original_price, specs, category, brand, description)
    ("夏季纯棉圆领T恤 男士短袖 透气百搭", 89.0, 199.0,
     {"颜色": ["白色","黑色","灰色","藏青"], "尺码": ["S","M","L","XL","XXL"]},
     "男装", "棉衣坊",
     "100%新疆长绒棉，柔软亲肤不起球。经典圆领设计，简约百搭。螺纹领口不易变形，双车线工艺耐穿不松垮。机洗建议冷水，不可漂白，中温熨烫。"),

    ("冰丝阔腿裤女夏季薄款 高腰垂感直筒裤", 129.0, 299.0,
     {"颜色": ["黑色","卡其色","深灰","雾霾蓝"], "尺码": ["S","M","L","XL"]},
     "女装", "丝语坊",
     "冰丝面料，清凉透气，三伏天不粘腿。高腰设计收腹显瘦，阔腿版型遮肉显高。垂感极佳不易皱，松紧腰头舒适不勒。建议手洗阴干，不可暴晒。"),

    ("运动跑鞋男款 轻便透气网面鞋 软底减震", 199.0, 459.0,
     {"颜色": ["黑白","灰蓝","全黑","白红"], "尺码": ["38","39","40","41","42","43","44"]},
     "鞋靴", "步凡运动",
     "飞织网面360度透气，单只仅重220g。EVA缓震中底，回弹率65%。橡胶防滑大底，湿滑路面不打滑。适合跑步/健身/日常通勤。"),

    ("大容量双肩包男女 防泼水旅行背包 电脑包", 149.0, 329.0,
     {"颜色": ["黑色","深蓝","灰色"], "尺码": ["标准"]},
     "箱包", "行者背包",
     "40L大容量，独立电脑仓可放17寸笔记本。防泼水面料，小雨无忧。S型肩带+透气背板，长时间背负不累。多隔层收纳，出差旅行一包搞定。"),

    ("蓝牙降噪耳机 头戴式无线 超长续航80小时", 259.0, 599.0,
     {"颜色": ["黑色","白色","蓝色"], "版本": ["标准版","Pro版(主动降噪)"]},
     "数码", "声悦科技",
     "ANC主动降噪，深度-35dB。40mm大动圈单元，Hi-Res金标认证。蓝牙5.3稳定连接，Type-C快充10分钟用5小时。折叠收纳，送收纳盒+音频线。"),

    ("不锈钢保温杯 大容量1000ml 24小时保冷保温", 69.0, 139.0,
     {"颜色": ["黑色","白色","银色","渐变蓝"], "容量": ["500ml","750ml","1000ml"]},
     "家居", "暖生活",
     "316不锈钢内胆，食品级安全。真空断热层，保冷24h/保温12h。大口径易清洗，硅胶密封圈滴水不漏。车载杯架适配，出行必备。"),

    ("防晒霜SPF50+ 清爽不油腻 面部全身可用", 79.0, 169.0,
     {"规格": ["50ml便携装","100ml家庭装"]},
     "美妆", "肌研堂",
     "物化结合防晒，SPF50+ PA++++。水感质地一抹化水，不假白不搓泥。含烟酰胺+维E，防晒同时养肤。洗面奶可卸，无需卸妆水。"),

    ("电动牙刷成人 声波震动 IPX7防水 5档模式", 159.0, 359.0,
     {"颜色": ["黑色","白色","粉色","绿色"], "版本": ["标准版(3刷头)","套装(6刷头+旅行盒)"]},
     "个护", "净齿科技",
     "31000次/分钟声波震动，5档模式(清洁/美白/敏感/按摩/抛光)。IPX7全身防水，感应充电续航60天。2分钟定时+30秒换区提醒。"),

    ("纯棉四件套 床上用品 双人床笠款 亲肤透气", 229.0, 499.0,
     {"颜色": ["灰白格","蓝条纹","纯灰","米白色","豆沙粉"], "尺寸": ["1.5m床(被套200x230)","1.8m床(被套220x240)"]},
     "家居", "梦之家",
     "100%新疆长绒棉，60支高支高密。活性印染不褪色不起球，A类母婴级安全标准。床笠360度包裹，防滑不移位。机洗不变形，越洗越柔软。"),

    ("充电宝20000mAh 大容量便携 22.5W快充", 99.0, 199.0,
     {"颜色": ["黑色","白色"], "版本": ["标准版","自带线版"]},
     "数码", "电量侠",
     "20000mAh大容量，可充手机4-6次。22.5W超级快充，兼容PD/QC协议。自带双线(Lightning+Type-C)，免带线。LED数显电量，小机身可上飞机。"),

    # 退款/售后相关的政策场景
    ("商品退换货服务说明", 0, 0,
     {},
     "服务", "平台通用",
     "本店支持7天无理由退换货。退货条件：吊牌完整、包装完好、不影响二次销售。退货运费由买家承担（质量问题除外）。退款时效：确认收货后退款1-3个工作日到账。换货流程：订单页申请→寄回商品→仓库验货→发出新商品→短信通知。"),

    ("店铺发货及物流说明", 0, 0,
     {},
     "服务", "平台通用",
     "下单后48小时内发货，默认中通/圆通快递。全国包邮（新疆/西藏/内蒙古除外，需补运费差价）。物流查询方式：我的订单→查看物流。如超过72小时未更新物流信息，请联系客服查询。"),
]

# ── 导入逻辑 ──────────────────────────────────────────


async def seed_products(shop_id: str = "shop_001", clear_first: bool = False):
    """将模拟商品写入 MySQL products 表 + Milvus ecom_products_v1。

    流程：
      MySQL: 写入商品结构化信息
      Milvus: 商品描述 BGE 编码 → 父子块切分 → 子块入库
    """
    s = get_settings()

    # ── 1. MySQL: 写入 products 表 ──────────────────
    from backend.core.database import get_session
    from backend.models.db_models import Product

    async with get_session() as session:
        from sqlalchemy import delete

        if clear_first:
            await session.execute(delete(Product).where(Product.shop_id == shop_id))
            await session.commit()

        saved = 0
        for (title, price, orig_price, specs, cat, brand, desc) in MOCK_PRODUCTS:
            pid = f"mock_{shop_id}_{saved+1:04d}"
            product = Product(
                product_id=pid,
                shop_id=shop_id,
                title=title,
                price=price,
                original_price=orig_price,
                description=desc,
                specs=specs,
                category=cat,
                brand=brand,
                status="onsale",
            )
            session.add(product)
            saved += 1
        await session.commit()

    print(f"[MySQL] {saved} products written to 'products' table")

    # ── 2. Milvus: 编码 + 父子块 + 入库 ────────────
    print("[Milvus] Encoding and indexing...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from backend.rag.chunker import ParentChildChunker
    from pymilvus import MilvusClient

    # 构建文档（商品标题+描述）
    documents = []
    for (title, price, orig_price, specs, cat, brand, desc) in MOCK_PRODUCTS:
        if not desc:
            continue
        text = f"{title}。品牌：{brand}。类目：{cat}。售价{price}元（原价{orig_price}元）。{desc}"
        documents.append(Document(
            page_content=text,
            metadata={"source": f"mock:{shop_id}", "knowledge_type": "product", "category": cat},
        ))

    # 父子块切分
    chunker = ParentChildChunker()
    result = chunker.split_documents(documents)
    print(f"  Parents: {result.total_parents}, Children: {result.total_children}")

    # BGE 编码
    model_path = s.embedding_model_name
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    children_texts = [c.page_content for c in result.child_chunks]
    print(f"  Encoding {len(children_texts)} chunks...")
    vectors = []
    batch = 50
    for i in range(0, len(children_texts), batch):
        batch_texts = children_texts[i:i+batch]
        vectors.extend(embeddings.embed_documents(batch_texts))

    # 写入 ecom_products_v1
    client = MilvusClient(uri=f"http://{s.milvus_host}:{s.milvus_port}")

    insert_data = []
    for i, child in enumerate(result.child_chunks):
        insert_data.append({
            "vector": vectors[i],
            "shop_id": shop_id,
            "product_id": f"mock_{shop_id}_{i:04d}",
            "content": child.page_content[:4000],
            "knowledge_type": "product",
            "source": child.metadata.get("source", f"mock:{shop_id}"),
            "category": child.metadata.get("category", "general"),
        })

    for i in range(0, len(insert_data), 200):
        client.insert(collection_name=s.milvus_product_collection, data=insert_data[i:i+200])

    client.close()
    print(f"[Milvus] {len(insert_data)} child chunks written to '{s.milvus_product_collection}'")

    # ── 3. MySQL: shops 表 ──────────────────────────
    from backend.models.db_models import Shop
    async with get_session() as session:
        from sqlalchemy import select
        existing = (await session.execute(select(Shop).where(Shop.shop_id == shop_id))).scalar_one_or_none()
        if not existing:
            session.add(Shop(
                shop_id=shop_id,
                shop_name="Mock旗舰店",
                taobao_seller_nick="mock_seller",
                status="active",
            ))
            await session.commit()
            print(f"[MySQL] Shop '{shop_id}' created")

    print("\nDone! Seed data ready.")


if __name__ == "__main__":
    from backend.core.logger import configure_logging
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--shop", default="shop_001")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    asyncio.run(seed_products(shop_id=args.shop, clear_first=args.clear))
