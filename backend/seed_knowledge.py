# backend/seed_knowledge.py
# 知识库管理工具：种子数据 + MySQL↔向量库同步。
#
# 用法（一条命令搞定）：
#   python backend/seed_knowledge.py              # 导入内置种子数据 + 同步向量库
#   python backend/seed_knowledge.py --sync-only   # 仅将 MySQL 已有数据同步到向量库
#
# 日常管理：通过 API 增删改查，不用跑脚本。
#   POST   /admin/knowledge          → 添加知识
#   GET    /admin/knowledge          → 查看所有知识
#   DELETE /admin/knowledge/{id}     → 删除指定知识
#   POST   /admin/knowledge/sync     → 同步 MySQL → 向量库

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio, argparse
from backend.core.database import get_session, init_db
from backend.models.db_models import CustomKnowledge
from backend.core.logger import get_logger
from sqlalchemy import select, func, delete

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════
# 种子数据（少量示例，实际数据通过 API 或淘宝 API 灌入）
# ═══════════════════════════════════════════════════════════════

SEED_DATA = [
    # ── 手机商品 ──
    {
        "content": "vivo Y100 5G手机 骁龙695 8+128GB 售价1799元。拍照：后置6400万OIS主摄+超级夜景模式。屏幕：6.58寸LCD 120Hz。功能：NFC、5000mAh+44W快充。适用：预算2000内、拍照好、长续航。",
        "category": "product", "merchant_id": "vivo_official",
    },
    {
        "content": "vivo iQOO Z8x 5G手机 天玑7200 8+256GB 售价1599元。拍照：5000万AI主摄+夜景算法。屏幕：6.64寸LCD 120Hz。功能：NFC、6000mAh+44W快充。适用：极致性价比、游戏+拍照、学生党。",
        "category": "product", "merchant_id": "vivo_official",
    },
    {
        "content": "OPPO K11x 5G手机 骁龙695 8+256GB 售价1899元。拍照：1亿像素主摄+9合1像素融合。屏幕：6.72寸LCD 120Hz。功能：NFC、5000mAh+67W快充。适用：超高像素拍照、快充体验。",
        "category": "product", "merchant_id": "oppo_official",
    },
    {
        "content": "荣耀X50 5G手机 骁龙6Gen1 8+128GB 售价1699元。拍照：1.08亿主摄+RAW域算法。屏幕：6.78寸OLED曲面屏 120Hz 1920Hz PWM调光。功能：NFC、5800mAh+35W快充、屏下指纹。适用：OLED屏幕、长续航。",
        "category": "product", "merchant_id": "honor_official",
    },
    {
        "content": "红米Note13 Pro 5G手机 骁龙7sGen2 8+256GB 售价1699元。拍照：2亿OIS主摄+800万超广角+200万微距。屏幕：6.67寸OLED直屏 120Hz。功能：NFC、红外、5100mAh+67W快充、IP54防水。适用：极致拍照性价比、红外NFC。",
        "category": "product", "merchant_id": "xiaomi_official",
    },
    {
        "content": "华为畅享70 Pro 5G手机 骁龙680 8+128GB 售价1799元。拍照：1.08亿主摄+华为AI算法。屏幕：6.7寸LCD护眼屏 90Hz。功能：NFC、5000mAh+40W快充、鸿蒙OS。适用：华为生态、系统流畅、拍照实用。",
        "category": "product", "merchant_id": "huawei_official",
    },
    # ── 3000-5000元 中高端手机 ──
    {
        "content": "vivo X200 Pro 5G旗舰手机 天玑9400 12+256GB 售价4999元。拍照：蔡司2亿像素APO超级长焦+5000万主摄+5000万超广角，自研V3影像芯片，全焦段人像大师。屏幕：6.78寸AMOLED曲面屏 120Hz LTPO。功能：NFC、6000mAh蓝海电池+90W有线+30W无线、IP69防水、超声波指纹。适用：专业摄影、演唱会神器、商务旗舰。",
        "category": "product", "merchant_id": "vivo_official",
    },
    {
        "content": "vivo X300 Pro 5G旗舰手机 天玑9500 12+256GB 售价4999元。拍照：蔡司2亿像素APO长焦+1英寸主摄+5000万超广角，自研V4影像芯片，支持8K视频录制、100倍数字变焦。屏幕：6.82寸2K AMOLED微曲屏 1-144Hz自适应刷新。功能：NFC、6500mAh+120W有线+50W无线、IP68防水、超声波指纹。适用：顶级影像旗舰、专业创作、游戏性能怪兽。",
        "category": "product", "merchant_id": "vivo_official",
    },
    {
        "content": "小米14 5G旗舰手机 骁龙8Gen3 12+256GB 售价4299元。拍照：徕卡光学Summilux镜头 5000万主摄+5000万浮动长焦+5000万超广角。屏幕：6.36寸1.5K OLED直屏 120Hz LTPO。功能：NFC、4610mAh+90W有线+50W无线、IP68、数字车钥匙。适用：小屏旗舰、徕卡摄影、全能水桶机。",
        "category": "product", "merchant_id": "xiaomi_official",
    },
    {
        "content": "小米14 Pro 5G旗舰手机 骁龙8Gen3 12+256GB 售价4999元。拍照：徕卡光学 5000万可变光圈主摄+5000万长焦+5000万超广角。屏幕：6.73寸2K OLED微曲屏 120Hz LTPO 龙晶玻璃。功能：NFC、4880mAh+120W有线+50W无线、IP68、数字车钥匙。适用：大屏旗舰、专业影像、极客玩家。",
        "category": "product", "merchant_id": "xiaomi_official",
    },
    {
        "content": "OPPO Find X8 Pro 5G旗舰手机 天玑9400 12+256GB 售价4999元。拍照：哈苏影像 双潜望长焦(3倍+6倍)+5000万主摄+5000万超广角，无影抓拍。屏幕：6.78寸1.5K OLED微曲屏 120Hz LTPO。功能：NFC、5910mAh+80W有线+50W无线、IP69防水。适用：哈苏色彩、演唱会长焦、商务全能。",
        "category": "product", "merchant_id": "oppo_official",
    },
    {
        "content": "华为Pura 70 Pro 5G旗舰手机 麒麟9010 12+256GB 售价4999元。拍照：超聚光主摄+4800万长焦微距+1250万超广角，风驰闪拍。屏幕：6.8寸1.5K OLED等深四曲屏 120Hz LTPO。功能：NFC、5050mAh+100W有线+80W无线、IP68、北斗卫星消息、鸿蒙OS。适用：华为生态、卫星通信、抓拍神器。",
        "category": "product", "merchant_id": "huawei_official",
    },
    # ── 电商政策 ──
    {
        "content": "退换货政策：签收7天内无理由退换，需吊牌包装完好。流程：订单页→申请售后→填原因→审核(24h内)→寄回(48h内)→退款(3工作日)。不支持：定制商品、食品生鲜、已拆封个护。运费：质量问题商家承担，个人原因买家承担。",
        "category": "policy", "merchant_id": "platform",
    },
    {
        "content": "物流规则：24h内发货（预售除外）。普通快递2-5天，顺丰1-2天。偏远地区加收5元延长2-3天。满99包邮，不满99收8元运费。大件(>20kg)按体积重量计费。查物流：我的订单→物流详情。",
        "category": "policy", "merchant_id": "platform",
    },
    {
        "content": "售后保障：正品保证支持专柜验货。价保15天（降价退差价）。先行赔付（质量问题平台垫付退款）。手机电脑可买延保+1或2年。客服电话400-618-8888，7×24小时服务。",
        "category": "policy", "merchant_id": "platform",
    },
]


async def seed():
    """写入种子数据到 MySQL（已存在则跳过）。"""
    async with get_session() as session:
        result = await session.execute(select(func.count(CustomKnowledge.id)))
        if result.scalar() > 0:
            print(f"知识库已有数据，跳过。如需重新导入请先清空。")
            return 0

        for item in SEED_DATA:
            session.add(CustomKnowledge(
                content=item["content"],
                source=f"merchant:{item['merchant_id']}",
                category=item["category"],
                merchant_id=item["merchant_id"],
            ))
        await session.commit()
        total = len(SEED_DATA)
        print(f"种子数据写入完成：{total} 条（商品 {sum(1 for i in SEED_DATA if i['category']=='product')} + 政策 {sum(1 for i in SEED_DATA if i['category']=='policy')}）")
        return total


async def sync_to_vector():
    """MySQL → 向量库全量同步。"""
    print("正在同步 MySQL → 向量库...")
    # 直接导入，绕开 __init__.py 的 transformers 依赖
    import importlib
    mod = importlib.import_module("backend.agents.knowledge_mgmt")
    agent = mod.KnowledgeMgmtAgent()
    r = await agent.sync_all_to_vector()
    if r["success"]:
        print(f"同步完成：{r['synced']} 条 → {r.get('chunks', '?')} 分块 ({r['backend']})")
    else:
        print(f"同步失败：{r.get('error', '')}")
    return r


async def clear_all():
    """清空 MySQL 知识库。"""
    async with get_session() as session:
        r = await session.execute(select(func.count(CustomKnowledge.id)))
        total = r.scalar()
        await session.execute(delete(CustomKnowledge))
        await session.commit()
        print(f"已从 MySQL 删除 {total} 条。")


async def main():
    p = argparse.ArgumentParser(description="知识库管理")
    p.add_argument("--sync-only", action="store_true", help="仅同步 MySQL→向量库")
    p.add_argument("--clear", action="store_true", help="清空知识库")
    args = p.parse_args()

    await init_db()

    if args.clear:
        await clear_all()
        return
    if args.sync_only:
        await sync_to_vector()
        return

    count = await seed()
    if count:
        await sync_to_vector()
    print("\n知识库就绪！或通过 API: POST /admin/knowledge 添加数据。")


if __name__ == "__main__":
    asyncio.run(main())
