-- scripts/init_db.sql
-- YunDa_agent 完整数据库 DDL（MySQL 8.0）
-- 用法：mysql -u root -p ecom_agent < scripts/init_db.sql

-- ═══════════════════════════════════════════════════════════
-- 1. 用户认证
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE COMMENT '用户名',
    email VARCHAR(128) NOT NULL UNIQUE COMMENT '邮箱',
    phone VARCHAR(20) DEFAULT '' COMMENT '手机号',
    password_hash VARCHAR(256) NOT NULL COMMENT 'bcrypt密码哈希',
    role VARCHAR(16) DEFAULT 'customer' COMMENT 'admin / merchant / customer',
    merchant_id VARCHAR(64) DEFAULT '' COMMENT '绑定的商户ID',
    is_active BOOLEAN DEFAULT TRUE COMMENT '账号激活状态',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_users_username (username),
    INDEX idx_users_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 2. 店铺
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS shops (
    id INT AUTO_INCREMENT PRIMARY KEY,
    shop_id VARCHAR(64) NOT NULL UNIQUE COMMENT '店铺唯一标识',
    shop_name VARCHAR(128) COMMENT '店铺名称',
    taobao_seller_nick VARCHAR(64) DEFAULT '' COMMENT '淘宝卖家昵称',
    access_token VARCHAR(255) DEFAULT '' COMMENT '淘宝API访问令牌',
    refresh_token VARCHAR(255) DEFAULT '' COMMENT '刷新令牌',
    token_expires_at DATETIME COMMENT '令牌过期时间',
    status VARCHAR(16) DEFAULT 'active' COMMENT 'active / inactive / suspended',
    milvus_product_collection VARCHAR(64) DEFAULT 'ecom_products_v1',
    milvus_policy_collection VARCHAR(64) DEFAULT 'ecom_policies_v1',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_shops_shop_id (shop_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 3. 商品缓存
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id VARCHAR(64) NOT NULL UNIQUE COMMENT '商品唯一ID',
    shop_id VARCHAR(64) DEFAULT '' COMMENT '所属店铺ID',
    title VARCHAR(255) COMMENT '商品标题',
    price FLOAT DEFAULT 0.0 COMMENT '售价',
    original_price FLOAT DEFAULT 0.0 COMMENT '原价',
    description TEXT COMMENT '商品描述',
    specs JSON COMMENT '规格参数',
    image_url VARCHAR(512) DEFAULT '' COMMENT '主图URL',
    category VARCHAR(64) DEFAULT '' COMMENT '商品类目',
    brand VARCHAR(64) DEFAULT '' COMMENT '品牌',
    sales_count INT DEFAULT 0 COMMENT '销量',
    status VARCHAR(16) DEFAULT 'onsale' COMMENT 'onsale / offsale / deleted',
    synced_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_products_product_id (product_id),
    INDEX idx_products_shop_id (shop_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 4. 订单
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(64) NOT NULL UNIQUE COMMENT '订单号',
    shop_id VARCHAR(64) DEFAULT '' COMMENT '所属店铺ID',
    user_id VARCHAR(64) DEFAULT '' COMMENT '下单用户ID',
    product_id VARCHAR(64) DEFAULT '' COMMENT '商品ID',
    product_name VARCHAR(255) DEFAULT '' COMMENT '商品名称',
    sku VARCHAR(64) DEFAULT '' COMMENT 'SKU',
    quantity INT DEFAULT 1,
    amount FLOAT DEFAULT 0.0 COMMENT '实付金额',
    status VARCHAR(32) DEFAULT 'pending' COMMENT 'pending/paid/shipped/delivered/cancelled/refunding/refunded',
    receiver_name VARCHAR(64) DEFAULT '',
    receiver_phone VARCHAR(32) DEFAULT '',
    receiver_address VARCHAR(255) DEFAULT '',
    logistics_tracking VARCHAR(128) DEFAULT '' COMMENT '快递单号',
    synced_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_orders_order_id (order_id),
    INDEX idx_orders_user_id (user_id),
    INDEX idx_orders_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 5. 用户画像
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS user_profile (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL UNIQUE,
    shop_id VARCHAR(64) DEFAULT '' COMMENT '关联店铺',
    role VARCHAR(16) DEFAULT 'customer',
    tags JSON COMMENT '偏好标签',
    conversation_count INT DEFAULT 0,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_profile_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 6. 对话历史
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS conversation_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) DEFAULT 'anonymous',
    shop_id VARCHAR(64) DEFAULT '' COMMENT '关联店铺',
    session_id VARCHAR(64) DEFAULT 'default',
    role VARCHAR(16) DEFAULT 'user',
    content TEXT,
    sentiment VARCHAR(32) DEFAULT '',
    intent VARCHAR(32) DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conv_user_id (user_id),
    INDEX idx_conv_session_id (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 7. 商户知识库
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS custom_knowledge (
    id INT AUTO_INCREMENT PRIMARY KEY,
    content TEXT COMMENT '知识内容',
    source VARCHAR(128) DEFAULT '' COMMENT '来源标识',
    category VARCHAR(32) DEFAULT 'general' COMMENT 'product/policy/faq/general',
    merchant_id VARCHAR(64) DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_ck_merchant (merchant_id),
    INDEX idx_ck_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ═══════════════════════════════════════════════════════════
-- 8. FAQ 问答表
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS ecom_faq (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(100) DEFAULT 'general' COMMENT '产品咨询/订单问题/售后服务/促销活动/物流配送',
    question TEXT COMMENT '常见问题',
    answer TEXT COMMENT '标准答案',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_faq_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
