# backend/seed_all.py
# 一键导入全部模拟数据 → MySQL → 向量库。
#
# 用法：
#   python -m backend.seed_all              # 导入全部
#   python -m backend.seed_all --clear      # 清空

import sys, os, re, asyncio, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.database import get_session, init_db, get_engine
from backend.core.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _parse_sql_values(filepath: str) -> list[list]:
    """解析 INSERT INTO ... VALUES (...) 语句，提取每行数据的原始值列表。
    返回 list[list[str]]，每个子列表是一行的原始字符串值（未做类型转换）。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 去掉注释行
    lines = [l for l in content.split("\n") if not l.strip().startswith("--")]
    content = "\n".join(lines)

    # 找到 VALUES 之后的内容
    idx = content.upper().find("VALUES")
    if idx == -1:
        return []
    values_block = content[idx + 6:]  # 跳过 "VALUES"

    rows = []
    # 用状态机解析：找到每个顶层 (...) 组
    depth = 0
    buf = ""
    in_string = False
    for ch in values_block:
        if ch == "'" and not (buf.endswith("\\")):
            in_string = not in_string
        if not in_string:
            if ch == '(':
                depth += 1
                if depth == 1:
                    buf = ""
                    continue
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    rows.append(buf)
                    buf = ""
                    continue
        if depth >= 1:
            buf += ch

    # 解析每行的值
    result = []
    for row_str in rows:
        values = _split_row(row_str)
        if values:
            result.append(values)
    return result


def _split_row(row_str: str) -> list:
    """将一行 VALUES 原始字符串拆分为值列表，正确处理引号内逗号。"""
    values = []
    buf = ""
    in_string = False
    for ch in row_str:
        if ch == "'" and not buf.endswith("\\"):
            in_string = not in_string
            buf += ch
        elif ch == ',' and not in_string:
            values.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        values.append(buf.strip())
    return values


def _convert(val: str):
    """将SQL字面值转为Python类型。"""
    val = val.strip()
    if val.upper() == "NULL":
        return None
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]  # 去掉引号
    if val.startswith("{") and val.endswith("}"):
        import json
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    # 数字
    try:
        if '.' in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


# ═══════════════════════════════════════════════════════════════
# 订单种子数据 (Python 直接定义，避免 SQL 截断问题)
# ═══════════════════════════════════════════════════════════════

ORDERS = [
    ("JD20260708-001", "shop_001", "user_005", "P1001", "vivo X200 Pro 12+256GB 蔡司影像", "SKU-VIVO-X200P-256", 1, 5999.00, "shipped", "张伟", "13900000001", "北京市朝阳区建国路88号SOHO现代城A座1201", "SF1234567890"),
    ("JD20260708-002", "shop_001", "user_012", "P1002", "华为Mate70 Pro 12+512GB 昆仑玻璃", "SKU-HW-MATE70P-512", 1, 6999.00, "shipped", "李娜", "13900000002", "上海市浦东新区陆家嘴环路1000号恒生银行大厦18F", "YTO9876543210"),
    ("JD20260708-003", "shop_001", "user_008", "P1003", "小米15 Ultra 16+1TB 徕卡光学", "SKU-XM-15U-1TB", 1, 7499.00, "shipped", "王强", "13900000003", "广州市天河区天河路385号太古汇2座2503", "STO4567891234"),
    ("JD20260708-004", "shop_001", "user_015", "P1004", "OPPO Find X8 Pro 12+256GB 哈苏影像", "SKU-OPPO-FX8P-256", 1, 5299.00, "shipped", "刘洋", "13900000004", "深圳市南山区科技南路18号深圳湾科技生态园9栋B座15F", "ZTO7891234567"),
    ("JD20260708-005", "shop_001", "user_003", "P2001", "Sony WH-1000XM5 蓝牙降噪耳机", "SKU-SONY-WH1000XM5", 1, 1999.00, "shipped", "陈静", "13900000005", "杭州市西湖区文三路100号高新大厦B座801", "YTO3216549870"),
    ("JD20260708-006", "shop_001", "user_018", "P2002", "Bose QC45 无线降噪耳机", "SKU-BOSE-QC45", 1, 1599.00, "shipped", "赵敏", "13900000006", "成都市高新区天府大道中段688号天府软件园E区3栋", "SF6547893210"),
    ("JD20260708-007", "shop_001", "user_001", "P1001", "vivo X200 Pro 12+256GB 蔡司影像", "SKU-VIVO-X200P-256", 1, 5999.00, "delivered", "黄鑫", "13900000013", "北京市海淀区中关村大街27号中关村大厦808", "SF1234567891"),
    ("JD20260708-008", "shop_001", "user_009", "P2001", "Sony WH-1000XM5 蓝牙降噪耳机", "SKU-SONY-WH1000XM5", 1, 1999.00, "delivered", "高敏", "13900000014", "上海市静安区南京西路1601号越洋广场25F", "YTO9876543211"),
    ("JD20260708-009", "shop_001", "user_016", "P3001", "安踏C202 5代 运动跑鞋 男款", "SKU-ANTA-C202-42", 1, 899.00, "delivered", "唐磊", "13900000015", "广州市海珠区新港东路1000号保利世贸中心E座1206", "STO4567891235"),
    ("JD20260708-010", "shop_001", "user_004", "P4001", "夏季纯棉圆领T恤 男士短袖", "SKU-COTTON-TEE-L-BLUE", 3, 297.00, "delivered", "罗婷", "13900000016", "深圳市福田区深南大道7006号万科富春东方大厦1503", "ZTO7891234568"),
    ("JD20260708-011", "shop_001", "user_013", "P1002", "华为Mate70 Pro 12+512GB 昆仑玻璃", "SKU-HW-MATE70P-512", 1, 6999.00, "delivered", "韩冰", "13900000017", "杭州市滨江区江南大道3588号恒生大厦B座12F", "YTO3216549871"),
    ("JD20260708-012", "shop_001", "user_020", "P2002", "Bose QC45 无线降噪耳机", "SKU-BOSE-QC45", 1, 1599.00, "delivered", "沈月", "13900000018", "成都市锦江区红星路三段1号IFS 3号楼28F", "SF6547893211"),
    ("JD20260708-013", "shop_001", "user_007", "P2002", "Bose QC45 无线降噪耳机", "SKU-BOSE-QC45", 1, 1599.00, "pending", "孙健", "13900000007", "武汉市江汉区建设大道568号新世界国贸大厦2205", None),
    ("JD20260708-014", "shop_001", "user_014", "P4001", "夏季纯棉圆领T恤 男士短袖", "SKU-COTTON-TEE-L-BLUE", 2, 198.00, "pending", "周婷", "13900000008", "南京市鼓楼区中山路55号新华大厦19F", None),
    ("JD20260708-015", "shop_001", "user_002", "P1001", "vivo X200 Pro 12+256GB 蔡司影像", "SKU-VIVO-X200P-256", 1, 5999.00, "pending", "吴刚", "13900000009", "西安市雁塔区高新路88号尚品国际A座1208", None),
    ("JD20260708-016", "shop_001", "user_019", "P3002", "李宁飞电4 Ultra 碳板跑鞋", "SKU-LN-FD4U-41", 1, 1299.00, "pending", "郑丽", "13900000010", "重庆市渝北区金开大道68号协信中心C座16F", None),
    ("JD20260708-017", "shop_001", "user_005", "P4003", "商务休闲衬衫 长袖 免烫抗皱", "SKU-SHIRT-WHITE-L", 2, 598.00, "paid", "张伟", "13900000001", "北京市朝阳区建国路88号SOHO现代城A座1201", None),
    ("JD20260708-018", "shop_001", "user_012", "P1001", "vivo X200 Pro 12+256GB 蔡司影像", "SKU-VIVO-X200P-256", 1, 5999.00, "paid", "李娜", "13900000002", "上海市浦东新区陆家嘴环路1000号恒生银行大厦18F", None),
    ("JD20260708-019", "shop_001", "user_008", "P2001", "Sony WH-1000XM5 蓝牙降噪耳机", "SKU-SONY-WH1000XM5", 1, 1999.00, "paid", "王强", "13900000003", "广州市天河区天河路385号太古汇2座2503", None),
    ("JD20260708-020", "shop_001", "user_015", "P3001", "安踏C202 5代 运动跑鞋 男款", "SKU-ANTA-C202-42", 1, 899.00, "paid", "刘洋", "13900000004", "深圳市南山区科技南路18号深圳湾科技生态园9栋B座15F", None),
    ("JD20260708-021", "shop_001", "user_016", "P4001", "夏季纯棉圆领T恤 男士短袖", "SKU-COTTON-TEE-L-BLUE", 2, 198.00, "refunding", "唐磊", "13900000015", "广州市海珠区新港东路1000号保利世贸中心E座1206", "STO4567891236"),
    ("JD20260708-022", "shop_001", "user_004", "P2001", "Sony WH-1000XM5 蓝牙降噪耳机", "SKU-SONY-WH1000XM5", 1, 1999.00, "refunding", "罗婷", "13900000016", "深圳市福田区深南大道7006号万科富春东方大厦1503", "ZTO7891234569"),
    ("JD20260708-023", "shop_001", "user_013", "P3001", "安踏C202 5代 运动跑鞋 男款", "SKU-ANTA-C202-42", 1, 899.00, "refunding", "韩冰", "13900000017", "杭州市滨江区江南大道3588号恒生大厦B座12F", "YTO3216549873"),
    ("JD20260708-024", "shop_001", "user_020", "P1002", "华为Mate70 Pro 12+512GB 昆仑玻璃", "SKU-HW-MATE70P-512", 1, 6999.00, "refunding", "沈月", "13900000018", "成都市锦江区红星路三段1号IFS 3号楼28F", "SF6547893213"),
    ("JD20260708-025", "shop_001", "user_007", "P1005", "荣耀Magic7 Pro 12+512GB AI智慧", "SKU-HONOR-M7P-512", 1, 5699.00, "refunded", "孙健", "13900000007", "武汉市江汉区建设大道568号新世界国贸大厦2205", "ZTO1472583693"),
    ("JD20260708-026", "shop_001", "user_014", "P2002", "Bose QC45 无线降噪耳机", "SKU-BOSE-QC45", 1, 1599.00, "refunded", "周婷", "13900000008", "南京市鼓楼区中山路55号新华大厦19F", "STO2583691473"),
    ("JD20260708-027", "shop_001", "user_011", "P1003", "小米15 Ultra 16+1TB 徕卡光学", "SKU-XM-15U-1TB", 1, 7499.00, "cancelled", "许强", "13900000019", "武汉市武昌区中北路86号汉街总部国际F座18F", None),
    ("JD20260708-028", "shop_001", "user_017", "P4002", "防晒霜SPF50+ 清爽不油腻 50ml", "SKU-SUNSCREEN-50ML", 1, 189.00, "cancelled", "何洁", "13900000020", "南京市秦淮区中山南路49号商茂世纪广场23F", None),
    ("JD20260708-029", "shop_001", "user_005", "P3001", "安踏C202 5代 运动跑鞋 男款", "SKU-ANTA-C202-42", 1, 899.00, "cancelled", "张伟", "13900000001", "北京市朝阳区建国路88号SOHO现代城A座1201", None),
    ("JD20260708-030", "shop_001", "user_012", "P1005", "荣耀Magic7 Pro 12+512GB AI智慧", "SKU-HONOR-M7P-512", 1, 5699.00, "cancelled", "李娜", "13900000002", "上海市浦东新区陆家嘴环路1000号恒生银行大厦18F", None),
]


async def seed_orders() -> int:
    from backend.models.db_models import Order
    async with get_session() as session:
        for row in ORDERS:
            session.add(Order(
                order_id=row[0], shop_id=row[1], user_id=row[2],
                product_id=row[3], product_name=row[4], sku=row[5],
                quantity=row[6], amount=row[7], status=row[8],
                receiver_name=row[9], receiver_phone=row[10],
                receiver_address=row[11], logistics_tracking=row[12],
            ))
        await session.commit()
    print(f"  ✅ 订单: {len(ORDERS)} 条 → orders")
    return len(ORDERS)


async def seed_products_from_sql() -> int:
    """从 seed_products.sql 解析并写入 products 表。"""
    from backend.models.db_models import Product
    filepath = os.path.join(DATA_DIR, "seed_products.sql")
    rows_raw = _parse_sql_values(filepath)

    async with get_session() as session:
        count = 0
        for vals in rows_raw:
            if len(vals) < 12:
                continue
            converted = [_convert(v) for v in vals]
            session.add(Product(
                product_id=converted[0], shop_id=converted[1],
                title=converted[2], price=converted[3],
                original_price=converted[4], description=converted[5],
                specs=converted[6], image_url=converted[7],
                category=converted[8], brand=converted[9],
                sales_count=converted[10], status=converted[11],
            ))
            count += 1
        await session.commit()
    print(f"  ✅ 商品: {count} 条 → products")
    return count


async def seed_knowledge_from_sql() -> int:
    """从 seed_knowledge.sql 解析并写入 custom_knowledge 表。"""
    from backend.models.db_models import CustomKnowledge
    filepath = os.path.join(DATA_DIR, "seed_knowledge.sql")
    rows_raw = _parse_sql_values(filepath)

    async with get_session() as session:
        count = 0
        for vals in rows_raw:
            if len(vals) < 4:
                continue
            converted = [_convert(v) for v in vals]
            session.add(CustomKnowledge(
                content=converted[0], source=converted[1],
                category=converted[2], merchant_id=converted[3],
            ))
            count += 1
        await session.commit()
    print(f"  ✅ 知识库: {count} 条 → custom_knowledge")
    return count


async def sync_knowledge_to_vector():
    print("\n📡 同步知识库到向量库...")
    try:
        from backend.agents.knowledge_mgmt import KnowledgeMgmtAgent
        agent = KnowledgeMgmtAgent()
        result = await agent.sync_all_to_vector()
        if result.get("success"):
            print(f"  ✅ 向量库: {result.get('synced', 0)} 条 → {result.get('chunks', '?')} 块 ({result.get('backend', '')})")
        else:
            print(f"  ⚠️ 向量库失败: {result.get('error', '未知')}")
        return result
    except Exception as e:
        print(f"  ⚠️ 向量库异常: {e}")
        return {"success": False, "error": str(e)}


async def seed_all():
    print("=" * 60)
    print("🛒 EcomSentiment_agent 模拟数据一键导入")
    print("=" * 60)

    await init_db()

    # 清空
    eng = get_engine()
    async with eng.begin() as conn:
        for t in ["orders", "products", "custom_knowledge"]:
            await conn.execute(text(f"DELETE FROM `{t}`"))
    print("  🗑️  已清空旧数据\n")

    n1 = await seed_products_from_sql()
    n2 = await seed_orders()
    n3 = await seed_knowledge_from_sql()

    await sync_knowledge_to_vector()

    print(f"\n{'='*60}")
    print(f"🎉 完成！MySQL: 商品{n1} + 订单{n2} + 知识库{n3} = {n1+n2+n3} 条")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    args = p.parse_args()

    if args.clear:
        async def _c():
            await init_db()
            eng = get_engine()
            async with eng.begin() as conn:
                for t in ["orders", "products", "custom_knowledge"]:
                    await conn.execute(text(f"DELETE FROM `{t}`"))
            print("✅ 已清空")
        asyncio.run(_c())
    else:
        asyncio.run(seed_all())
