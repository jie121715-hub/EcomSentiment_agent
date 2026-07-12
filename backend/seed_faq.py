# backend/seed_faq.py
# 🆕 FAQ 种子数据导入：从 CSV 或内置数据写入 MySQL ecom_faq 表。
#
# 使用方式：
#   python -m backend.seed_faq
#
# 数据来源：
#   - 优先级1: 从 EcomSentiment_RAG/mysql_qa/data/电商FAQ问答.csv 读取
#   - 优先级2: 使用内置 FAQ 数据（基于客服场景常见问题）

import asyncio
import csv
import os
import sys

# 确保 project root 在 path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from backend.core.logger import configure_logging, get_logger
from backend.core.database import get_session, init_db
from backend.models.db_models import EcomFAQ
from sqlalchemy import select, delete

logger = get_logger(__name__)

# ── 内置 FAQ 数据（兜底）────────────────────────────────
_BUILTIN_FAQ = [
    # 产品咨询
    ("产品咨询", "这款商品是什么材质的", "您好，该商品的材质信息可以在商品详情页查看，通常包含面料成分、材质比例等详细说明。如有疑问可联系在线客服。"),
    ("产品咨询", "这个衣服尺码偏大还是偏小", "您好，不同品牌的尺码标准可能不同，建议您参考商品详情页的尺码表，并根据自身实际尺寸选择。如不确定可咨询客服获取建议。"),
    ("产品咨询", "这款手机支持5G网络吗", "您好，请查看商品规格参数中「网络制式」一栏，标注支持5G即为5G手机。目前大部分新款智能手机均支持5G。"),
    ("产品咨询", "这个商品有优惠吗", "您好，商品当前的价格和优惠信息可以在商品详情页查看。建议关注店铺活动，大促期间通常会有额外折扣哦～"),
    ("产品咨询", "有适合送父母的手机推荐吗", "您好！送父母的话，推荐考虑屏幕大、电池耐用、操作简单的机型。比如荣耀Magic7 Pro有4320Hz超高频调光护眼屏，或者红米K80 Pro性价比高电池大。具体可以告诉我预算，我帮您精准推荐～"),
    # 订单问题
    ("订单问题", "如何查询我的订单状态", "您好，您可以在「我的订单」中查看所有订单及当前状态（待付款/待发货/已发货/已完成）。点击对应订单可查看物流详情。"),
    ("订单问题", "下单后多久能发货", "您好，一般情况下付款成功后24-48小时内安排发货，大促期间可能延迟至72小时。预售商品以页面标注的发货时间为准。"),
    ("订单问题", "如何修改订单地址", "您好，未发货的订单可以在订单详情页直接修改收货地址；已发货的订单需联系客服协助修改。请注意地址修改可能影响配送时效。"),
    ("订单问题", "如何取消订单", "您好，未发货的订单可以在订单详情页点击「取消订单」。已发货的订单无法直接取消，可以在收到货后申请退货退款。"),
    ("订单问题", "我的订单为什么被取消了", "您好，订单被取消可能有以下原因：1）系统检测到异常订单自动取消；2）商家因缺货取消；3）超时未付款自动取消。建议查看订单详情或联系客服了解具体原因。"),
    # 售后服务
    ("售后服务", "如何申请退货退款", "您好，您可在订单详情页点击「申请售后」，选择「退货退款」并填写原因及上传凭证。审核通过后，请在7天内将商品寄回指定地址。"),
    ("售后服务", "退货的运费谁承担", "您好，因商品质量问题导致的退货，运费由商家承担；因个人原因（如不喜欢、买错）导致的退货，运费由买家自行承担。"),
    ("售后服务", "退款多久能到账", "您好，商品退回仓库验收通过后，退款将在1-3个工作日内原路返回。如超期未到账请联系客服查询。"),
    ("售后服务", "收到商品有质量问题怎么办", "您好，非常抱歉给您带来不便！请在订单详情页申请售后，选择「质量问题」并上传商品问题部位的清晰照片。我们会在24小时内审核处理，确认为质量问题后将为您安排退换货，运费由商家承担。"),
    ("售后服务", "换货流程是怎样的", "您好，在订单详情页申请售后选择「换货」，填写换货原因和想要的规格。审核通过后将商品寄回，仓库收到后发出新商品。换货周期一般为5-7个工作日。"),
    # 促销活动
    ("促销活动", "如何使用优惠券", "您好，在提交订单页面点击「优惠券」选择可用优惠券即可。请注意优惠券有使用门槛和有效期，具体以券面说明为准。"),
    ("促销活动", "现在的活动折扣是多少", "您好，当前促销活动信息可在首页活动专区或商品详情页查看。不同商品参与的活动不同，建议以实际下单时的价格为准。"),
    ("促销活动", "什么时候有活动", "您好，店铺常规活动包括每月会员日、618年中大促、双11、双12、年货节等。建议关注店铺首页或加入会员获取活动提醒哦～"),
    # 物流配送
    ("物流配送", "如何查看物流信息", "您好，已发货的订单可在订单详情页点击「查看物流」获取实时物流轨迹。也可复制快递单号到快递官网查询。"),
    ("物流配送", "可以指定快递公司吗", "您好，目前系统会根据您的收货地址自动匹配最优快递公司，暂不支持手动指定。如有特殊情况可备注留言。"),
    ("物流配送", "偏远地区包邮吗", "您好，不同商品的包邮政策不同，具体以商品页面标注为准。部分商品对新疆、西藏等偏远地区可能收取额外运费。"),
    ("物流配送", "快递显示签收但我没收到", "您好，请先确认是否有家人/同事/物业代收。如确认未收到，请在订单详情页联系客服，我们会联系快递公司核实并为您处理。"),
    ("物流配送", "物流好几天没更新了", "您好，物流信息延迟更新可能是快递中转或系统同步问题。建议您耐心等待1-2天，如仍未更新请联系客服帮您查询。"),
]


async def seed_faq(csv_path: str = ""):
    """导入 FAQ 数据到 MySQL ecom_faq 表。

    优先从 CSV 文件读取，文件不存在则使用内置数据。
    """
    await init_db()

    rows_to_insert = []

    # ── 尝试从 CSV 读取 ──────────────────────────────────
    if csv_path and os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        rows_to_insert.append((row[0].strip(), row[1].strip(), row[2].strip()))
            logger.info("seed_faq.csv_loaded", path=csv_path, count=len(rows_to_insert))
        except Exception as e:
            logger.error("seed_faq.csv_failed", error=str(e))

    # ── CSV 为空则用内置数据 ─────────────────────────────
    if not rows_to_insert:
        rows_to_insert = _BUILTIN_FAQ
        logger.info("seed_faq.using_builtin", count=len(rows_to_insert))

    # ── 写入 MySQL ───────────────────────────────────────
    async with get_session() as session:
        # 先清空旧数据
        await session.execute(delete(EcomFAQ))
        await session.commit()

        for category, question, answer in rows_to_insert:
            session.add(EcomFAQ(category=category, question=question, answer=answer))

        await session.commit()

    logger.info("seed_faq.done", total=len(rows_to_insert))
    print(f"[OK] FAQ 种子数据导入完成，共 {len(rows_to_insert)} 条")


async def main():
    configure_logging()

    # 查找旧项目的 FAQ CSV
    csv_paths = [
        os.path.join(_project_root, "..", "EcomSentiment_RAG", "mysql_qa", "data", "电商FAQ问答.csv"),
        os.path.join(_project_root, "data", "faq.csv"),
    ]
    csv_file = ""
    for p in csv_paths:
        if os.path.isfile(p):
            csv_file = p
            print(f"[FOUND] FAQ CSV: {p}")
            break

    if not csv_file:
        print("[INFO] 未找到 CSV 文件，使用内置 FAQ 数据（24条）")

    await seed_faq(csv_file)

    # 验证导入
    async with get_session() as session:
        result = await session.execute(select(EcomFAQ))
        rows = result.scalars().all()
        print(f"\n[STATS] MySQL ecom_faq 表当前数据：{len(rows)} 条")
        print("-" * 60)
        for r in rows[:5]:
            print(f"  [{r.category}] {r.question} -> {r.answer[:40]}...")
        if len(rows) > 5:
            print(f"  ... 还有 {len(rows) - 5} 条")


if __name__ == "__main__":
    asyncio.run(main())
