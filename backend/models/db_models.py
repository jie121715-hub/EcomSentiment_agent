# backend/models/db_models.py
# 🆕 v3 数据层：MySQL 只存结构化本地数据。
#
# 数据来源分工：
#   MySQL   — 淘宝API同步(订单+商品缓存) + 本地生成(画像/对话/日志/工单)
#   Milvus  — 商品描述向量 + 店铺政策向量 (Collection级多租户, shop_id隔离)
#   淘宝API  — 实时库存、物流轨迹、商品搜索 (RAG兜底)
#   Redis   — 高频缓存、会话历史
#
# 表清单 (7张):
#   shops / products / orders / user_profile / conversation_history / clarify_logs / support_tickets

from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, Float, JSON, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


# ═══════════════════════════════════════════════════════════════
# 店铺
# ═══════════════════════════════════════════════════════════════

class Shop(Base):
    """店铺表 — 接入系统的店铺基础信息。"""
    __tablename__ = "shops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shop_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="店铺唯一标识")
    shop_name: Mapped[str] = mapped_column(String(128), comment="店铺名称")
    taobao_seller_nick: Mapped[str] = mapped_column(String(64), default="", comment="淘宝卖家昵称")
    access_token: Mapped[str] = mapped_column(String(255), default="", comment="淘宝API访问令牌")
    refresh_token: Mapped[str] = mapped_column(String(255), default="", comment="刷新令牌")
    token_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="令牌过期时间")
    status: Mapped[str] = mapped_column(String(16), default="active", comment="active / inactive / suspended")
    milvus_product_collection: Mapped[str] = mapped_column(String(64), default="ecom_products_v1", comment="对应的Milvus商品Collection")
    milvus_policy_collection: Mapped[str] = mapped_column(String(64), default="ecom_policies_v1", comment="对应的Milvus政策Collection")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Shop {self.shop_id}: {self.shop_name}>"


# ═══════════════════════════════════════════════════════════════
# 商品 & 订单（淘宝API同步缓存）
# ═══════════════════════════════════════════════════════════════

class Product(Base):
    """商品缓存表 — 淘宝API同步的结构化商品信息。"""
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="商品唯一ID (taobao_num_iid)")
    shop_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属店铺ID")
    title: Mapped[str] = mapped_column(String(255), comment="商品标题")
    price: Mapped[float] = mapped_column(Float, default=0.0, comment="售价")
    original_price: Mapped[float] = mapped_column(Float, default=0.0, comment="原价")
    description: Mapped[str] = mapped_column(Text, default="", comment="商品描述(纯文本)")
    specs: Mapped[dict] = mapped_column(JSON, default=dict, comment="规格参数 {颜色: [黑,白], 尺码: [S,M,L]}")
    image_url: Mapped[str] = mapped_column(String(512), default="", comment="主图URL")
    category: Mapped[str] = mapped_column(String(64), default="", comment="商品类目")
    brand: Mapped[str] = mapped_column(String(64), default="", comment="品牌")
    sales_count: Mapped[int] = mapped_column(Integer, default=0, comment="销量")
    status: Mapped[str] = mapped_column(String(16), default="onsale", comment="onsale / offsale / deleted")
    synced_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="最后同步时间")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Product {self.product_id}: {self.title[:30]}>"


class Order(Base):
    """订单缓存表 — 淘宝API同步 + 本地生成的订单记录。"""
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="订单号")
    shop_id: Mapped[str] = mapped_column(String(64), index=True, comment="所属店铺ID")
    user_id: Mapped[str] = mapped_column(String(64), index=True, comment="下单用户ID")
    product_id: Mapped[str] = mapped_column(String(64), default="", comment="商品ID")
    product_name: Mapped[str] = mapped_column(String(255), default="", comment="商品名称(快照)")
    sku: Mapped[str] = mapped_column(String(64), default="", comment="SKU")
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    amount: Mapped[float] = mapped_column(Float, default=0.0, comment="实付金额")
    status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True,
        comment="pending/paid/shipped/delivered/cancelled/refunding/refunded"
    )
    receiver_name: Mapped[str] = mapped_column(String(64), default="")
    receiver_phone: Mapped[str] = mapped_column(String(32), default="")
    receiver_address: Mapped[str] = mapped_column(String(255), default="")
    logistics_tracking: Mapped[str] = mapped_column(String(128), default="", comment="最新快递单号(API获取)")
    synced_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="最后同步时间")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Order {self.order_id}: {self.status}>"


# ═══════════════════════════════════════════════════════════════
# 用户 & 对话
# ═══════════════════════════════════════════════════════════════

class UserProfile(Base):
    """用户画像表 — 轻量版：记录偏好标签。"""
    __tablename__ = "user_profile"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    shop_id: Mapped[str] = mapped_column(String(64), default="", index=True, comment="关联店铺(多租户)")
    role: Mapped[str] = mapped_column(String(16), default="customer")
    tags: Mapped[dict] = mapped_column(JSON, default=dict, comment="偏好标签")
    conversation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_active: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User {self.user_id}>"


class ConversationRecord(Base):
    """对话历史表 — 持久化所有用户对话。"""
    __tablename__ = "conversation_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), default="anonymous", index=True)
    shop_id: Mapped[str] = mapped_column(String(64), default="", index=True, comment="关联店铺")
    session_id: Mapped[str] = mapped_column(String(64), default="default")
    role: Mapped[str] = mapped_column(String(16), default="user")
    content: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str] = mapped_column(String(32), default="")
    intent: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Conversation {self.id}: {self.role} [{self.intent}]>"


# ═══════════════════════════════════════════════════════════════
# 运营表
# ═══════════════════════════════════════════════════════════════

class ClarifyLog(Base):
    """澄清日志表 — 低置信度反问事件，用于模型迭代。"""
    __tablename__ = "clarify_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), default="anonymous", index=True)
    shop_id: Mapped[str] = mapped_column(String(64), default="", comment="关联店铺")
    session_id: Mapped[str] = mapped_column(String(64), default="default")
    original_query: Mapped[str] = mapped_column(Text, comment="用户原始问题")
    detected_intent: Mapped[str] = mapped_column(String(32), comment="检测到的意图（低置信度）")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    clarification_question: Mapped[str] = mapped_column(Text, comment="反问内容")
    user_response: Mapped[str] = mapped_column(Text, default="", comment="用户回复")
    entities: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<ClarifyLog {self.id}: intent={self.detected_intent}>"


class SupportTicket(Base):
    """工单表 — 转人工/紧急升级事件。"""
    __tablename__ = "support_tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(32), unique=True, index=True, comment="TK-YYYYMMDD-XXXX")
    user_id: Mapped[str] = mapped_column(String(64), default="anonymous", index=True)
    shop_id: Mapped[str] = mapped_column(String(64), default="", index=True, comment="关联店铺")
    session_id: Mapped[str] = mapped_column(String(64), default="default")
    urgency: Mapped[str] = mapped_column(String(16), default="normal", comment="normal/elevated/critical")
    reason: Mapped[str] = mapped_column(String(255))
    original_query: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str] = mapped_column(String(32), default="")
    intent: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(16), default="open", comment="open/assigned/resolved/closed")
    assigned_to: Mapped[str] = mapped_column(String(64), default="")
    resolution: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<SupportTicket {self.ticket_id}: [{self.status}]>"


class CustomKnowledge(Base):
    """知识库表 — 商户自定义知识（FAQ/政策/商品说明），MySQL持久化 + Milvus向量化。"""
    __tablename__ = "custom_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, comment="知识内容")
    source: Mapped[str] = mapped_column(String(128), default="", comment="来源标识 (merchant:xxx / taobao:xxx)")
    category: Mapped[str] = mapped_column(String(32), default="general", comment="分类: product/policy/faq/general")
    merchant_id: Mapped[str] = mapped_column(String(64), default="", index=True, comment="商户标识")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<CustomKnowledge {self.id}: [{self.category}] {self.content[:40]}>"


# ═══════════════════════════════════════════════════════════════
# 🆕 FAQ 问答表（BM25快速检索用）
# ═══════════════════════════════════════════════════════════════

class EcomFAQ(Base):
    """电商FAQ问答表 — 高频问题标准答案，配合BM25做毫秒级检索。

    数据来源：商户录入 + 历史高频问题沉淀。
    检索流程：Redis缓存 → BM25评分 → 本表取答案 → 缓存到Redis。
    """
    __tablename__ = "ecom_faq"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(
        String(100), default="general", index=True,
        comment="问题类别: 产品咨询/订单问题/售后服务/促销活动/物流配送"
    )
    question: Mapped[str] = mapped_column(Text, comment="常见问题")
    answer: Mapped[str] = mapped_column(Text, comment="标准答案")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<EcomFAQ {self.id}: [{self.category}] {self.question[:30]}>"
