# 云答智能客服系统

基于 **感知-路由-分发** Multi-Agent 协同架构的智能客服系统，支持情感分析、意图识别、RAG 语义检索、业务查询、多租户隔离等全链路客服场景。

## 🏗️ 系统架构

```
用户输入 → 🎯 感知Agent (BERT情感+意图+LLM NER)
       → 🧠 路由Agent (置信度门控/情绪紧急度/意图分发)
       → 上下文解析 (多租户shop_id)
       → 4路分发:
           ├─ 📚 知识应答Agent (Redis→BM25→RAG 三层检索引擎)
           ├─ 📦 业务Agent (查物流/订单/库存+改地址/退款)
           ├─ ❓ 澄清反问 (低置信度→引导用户明确意图)
           └─ 📞 转人工 (情绪critical / 法律高风险)
```

### 核心技术栈

| 层级 | 技术 |
|------|------|
| **Web框架** | FastAPI + SSE 流式响应 |
| **Agent编排** | LangGraph 状态图 (perceive → route → resolve_context → dispatch) |
| **LLM** | DeepSeek V4 (主) / 阿里百炼 DashScope (备) |
| **情感/意图** | BERT 本地模型 — 7类情感 + 3类意图 |
| **向量检索** | Milvus (主) + Chroma (降级) — BGE-M3 1024维嵌入 |
| **RAG管线** | 混合检索(稠密+稀疏) + RRF融合 + BGE-Reranker精排 + Query改写 |
| **FAQ检索** | BM25 (jieba分词+softmax门控) + Redis缓存 |
| **数据库** | MySQL 8.0 (SQLAlchemy async) |
| **缓存** | Redis (两级缓存: 精确匹配 + FAQ) |
| **前端** | Vue 3 + TypeScript + Element Plus + Pinia |
| **部署** | Docker Compose (MySQL + Milvus + Redis + Backend + Frontend) |

## 🚀 快速开始

### 方式一: Docker Compose（推荐）

```bash
cd EcomSentiment_agent
docker-compose up -d
```

访问：
- 前端界面: http://localhost:3000
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 方式二: 本地开发

**前置条件：** Python 3.11+、Node.js 20+、MySQL 8.0、Milvus 2.4、Redis 7

```bash
# 1. 创建 conda 环境
conda create -n ecom_agent python=3.11 -y
conda activate ecom_agent

# 2. 安装后端依赖
cd EcomSentiment_agent
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.local.example .env.local   # 编辑填入实际值

# 4. 初始化数据库
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS ecom_agent"
mysql -u root -p ecom_agent < scripts/init_db.sql

# 5. 初始化 Milvus Collection
python scripts/init_milvus_v2.py

# 6. 种子数据
python scripts/seed_users.py
python scripts/seed_tenant_data.py

# 7. 下载模型（如未手动拷贝）
python scripts/download_emotion_model.py

# 8. 启动后端 (端口 8000)
python main.py

# 9. 启动前端 (新终端，端口 3000)
cd frontend
npm install
npm run dev
```

### 模型文件

项目依赖以下本地模型，需放到 `models/` 目录：

```
models/
├── bert-base-chinese/             # BERT 基座（情感/意图分类）
├── bge-m3/                        # BGE-M3 嵌入模型（RAG 向量检索）
├── emotion_7class/                # StructBERT 情绪7分类（ModelScope）
├── gemma-3-1b-ecommerce-intent/   # Gemma 意图分类
├── gemma-3-1b-it/                 # Gemma 备用
└── qwen2.5-0.5b-instruct/        # Qwen 二分类意图训练
```

其中 `bert-base-chinese` 和 `bge-m3` 需手动拷贝，`emotion_7class` 可运行 `python scripts/download_emotion_model.py` 自动下载。

### 演示账号

| 角色 | 用户名 | 密码 |
|------|--------|------|
| 管理员 | admin | admin123 |
| 商户 | merchant | merchant123 |
| 用户 | customer | customer123 |

## 📡 API 接口

### 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/login` | 用户登录，返回 JWT |
| POST | `/api/v1/auth/register` | 用户注册 |
| GET | `/api/v1/auth/me` | 获取当前用户信息 |

### 对话

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 非流式对话 |
| POST | `/api/v1/chat/stream` | SSE 流式对话 |
| GET | `/health` | 健康检查 |

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/admin/knowledge` | 查看知识库列表（分页） |
| POST | `/api/v1/admin/knowledge` | 添加知识 |
| POST | `/api/v1/admin/knowledge/batch` | 批量添加 |
| DELETE | `/api/v1/admin/knowledge/{id}` | 删除知识 |
| POST | `/api/v1/admin/knowledge/sync` | 全量同步向量库 |
| POST | `/api/v1/admin/knowledge/upload` | 上传 PDF/DOCX |
| POST | `/api/v1/admin/knowledge/import-taobao` | 淘宝商品导入 |

## 🤖 Agent 体系

### 1. 感知Agent (`PerceptionAgent`)
- **BERT 情感分类**: 7 类细粒度情感（开心/感谢/中性/困惑/焦虑/愤怒/失望）
- **BERT 意图分类**: 3 类核心意图（知识问答/业务处理/工单处理）
- **LLM 实体抽取**: 订单号、SKU、物流单号、商品名等
- **模型自发现**: 优先使用 `saved_models/` 下的自训练模型 → ModelScope 预训练 → 本地 BERT → LLM 零样本

### 2. 路由Agent (`RoutingAgent`)
- **三维决策**: 置信度门控 + 情绪紧急度 + 意图分发
- **门控阈值**: 意图置信度 < 0.50 → 反问澄清
- **情绪接管**: 愤怒 + 投诉/退款 → 直接转人工 (critical)
- **策略注入**: 检索策略 + 语气指令 + 动态提示词
- **知识管理兼容**: 旧 knowledge_mgmt 意图自动重定向到 knowledge_qa

### 3. 知识应答Agent (`KnowledgeQAAgent`)
- **三层检索漏斗**:
  1. **Redis 缓存** (毫秒级, 命中率 ~30%) — 精确匹配 → 直接返回
  2. **BM25 FAQ** (~10ms, 命中率 ~50%) — jieba分词 + BM25Okapi + softmax门控
  3. **RAG 语义检索** (~2s, 覆盖率 ~10%) — 混合检索 + BGE精排 + LLM生成
- **Query 改写**: 4 策略并行（指代消解/同义扩展/子问题拆分/关键词增强）
- **父子块机制**: 子块检索 → parent_id去重 → 父块给LLM
- **闲聊兜底**: 内置快速回复模板 + LLM闲聊生成

### 4. 业务Agent (`BusinessAgent`)
- **读操作**: 查物流轨迹、查订单状态、查商品库存
- **写操作**: 改收货地址、取消订单、申请退款（含二次确认机制）
- **LLM Function Calling**: 自动抽取意图和参数，异常时关键词兜底
- **订单归属校验**: 自动校验订单是否属于当前用户

## 🛡️ 安全机制

- **JWT 认证**: Bearer Token 鉴权，支持 admin/merchant/customer 三种角色
- **API Key**: X-API-Key 向后兼容（微信桥接等服务间调用）
- **内容审核**: 上传知识时扫描 18 种恶意模式
- **三级分流**: pass（直接入库）/ review（待审核）/ reject（拒绝上传）
- **高危门控**: 退款/法律问题无知识库文档 → 强制转人工
- **多租户隔离**: shop_id 粒度数据隔离 + Milvus Collection 级隔离

## 📁 项目结构

```
EcomSentiment_agent/
  main.py                       # 本地开发启动器
  requirements.txt              # Python 依赖
  config.toml                   # cc-connect 微信机器人配置
  .env.local                    # 环境变量（API Key / 数据库 / Milvus 等）

  backend/
    main.py                     # FastAPI 应用入口（lifespan预加载）
    config.py                   # 全局配置中心 (60+项)
    dependencies.py             # FastAPI DI (JWT + API Key)

    api/
      router.py                 # 路由聚合器
      v1/
        auth.py                 # 认证接口 (login/register/me)
        chat.py                 # 对话接口 (非流式 + SSE流式)
        admin_knowledge.py      # 知识库 CRUD + 批量同步
        admin_upload.py         # 文件上传 (PDF/DOCX)
        admin_import.py         # 淘宝商品导入

    agents/                     # 4个 LangGraph Agent
      __init__.py               # 惰性导入入口
      graph.py                  # LangGraph 编排层 (7节点4路分发)
      context_resolver.py       # 多租户上下文解析
      perception/               # 感知Agent (情感+意图+NER)
      router/                   # 路由Agent (三维决策)
      knowledge_qa/             # 知识应答Agent (RAG+闲聊兜底)
      business/                 # 业务Agent (MySQL读写合一)

    core/                       # 基础设施
      database.py               # MySQL 异步引擎 (SQLAlchemy)
      llm_factory.py            # LLM 工厂 (DeepSeek/DashScope路由)
      logger.py                 # 结构化日志
      retry.py                  # 指数退避重试 (三层兜底)
      exceptions.py             # 异常体系 (可重试/不可重试)
      response.py               # 标准API响应格式
      content_moderation.py     # 内容安全扫描 (18种恶意模式)

    rag/                        # RAG 检索引擎
      retriever.py              # 检索器 (Milvus优先→Chroma降级)
      multi_tenant_retriever.py # 多租户检索 (Collection级隔离)
      chunker.py                # 父子块切分
      hybrid_searcher.py        # 混合检索 + RRF融合 (4路并行)
      bm25_search.py            # BM25 + FAQ 检索 (jieba分词)
      query_rewriter.py         # Query改写 (4策略并行)
      reranker.py               # BGE-Reranker 精排
      post_processor.py         # 答案后处理 (幻觉检测/敏感词/格式化)
      cache.py                  # Redis 两级缓存
      doc_loader.py             # PDF/DOCX 文档加载
      preprocess.py             # jieba 分词预处理
      prompts.py                # Few-shot Prompt 模板集

    models/                     # 数据模型
      schemas.py                # Pydantic Schema (API模型)
      db_models.py              # SQLAlchemy ORM (9表)

    data/                       # 运行时数据
      sentiment_map.py          # 情感→语气+策略映射表

    training/                   # BERT 模型训练
      config.py                 # 训练配置
      train_intent.py           # 意图分类器 (4分类)
      train_sentiment.py        # 情感分类器 (7分类)
      train_binary_intent.py    # 二分类意图 (Qwen+QLoRA)
      saved_models/             # 训练好的模型权重

    db/
      migrations.py             # 增量数据库迁移

    utils/
      wechat_bridge.py          # 微信桥接 (ACP协议)
      logistics.py              # 快递鸟物流查询 API
      taobao_importer.py        # 淘宝开放平台商品导入
      preload_dlls.py           # Windows DLL预加载 (pyarrow兼容)

  frontend/                     # Vue 3 SPA 前端
    vite.config.ts              # Vite 配置 (proxy /api → :8000)
    src/
      main.ts                   # Vue 入口
      App.vue                   # 根组件
      router/index.ts           # Vue Router + 路由守卫
      stores/auth.ts            # Pinia 认证状态
      api/                      # Axios API 客户端
        client.ts               # Axios 实例 (JWT拦截器)
        auth.ts                 # 认证 API
        chat.ts                 # 对话 API
        admin.ts                # 管理 API
      composables/
        useSSEChat.ts           # SSE 流式对话 Hook
      views/
        LoginView.vue           # 登录页
        ChatView.vue            # 对话页 (核心)
        DashboardView.vue       # 工作台
        AdminView.vue           # 管理后台
      components/
        layout/                 # 布局组件 (Header/Sidebar/AppLayout)
        chat/                   # 对话组件 (ChatBubble/ChatInput/PerceptionCard/RoutingCard/MarkdownRenderer)

  scripts/                      # 运维脚本
    init_db.sql                 # 完整 DDL (MySQL建表)
    init_milvus_v2.py           # Milvus Collection 初始化
    check_env.py                # 环境连通性检查
    check_connections.py        # 数据库/向量库连接测试
    seed_users.py               # 用户种子数据
    seed_tenant_data.py         # 租户种子数据
    download_emotion_model.py   # ModelScope 情绪模型下载
    preprocess_intent_4class.py # 意图4分类数据预处理
    preprocess_emotion_7class.py# 情感7分类数据预处理
    preprocess_binary_intent.py # 二分类意图数据预处理
    inspect_training_data.py    # 训练数据检查工具

  data/                         # 数据文件
    intent_4class_labels.json   # 意图标签映射
    sentiment_7class_labels.json# 情感标签配置

  models/                       # 本地模型文件
    bert-base-chinese/          # BERT 基座模型
    bge-m3/                     # BGE-M3 嵌入模型
    emotion_7class/             # StructBERT 情绪模型
    gemma-3-1b-ecommerce-intent/# Gemma 意图分类
    gemma-3-1b-it/              # Gemma 备用
    qwen2.5-0.5b-instruct/     # Qwen 二分类

  tests/
    test_agent.py               # 端到端 Agent 测试

  docs/
    architecture_flow.md        # 完整架构流程图
```

## 🔧 环境变量

关键配置项（完整清单见 `.env.local`）：

```bash
# LLM
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_DEFAULT_MODEL=deepseek-v4-flash

# 数据库
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=123456
DB_NAME=ecom_agent

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=1234

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080

# RAG
EMBEDDING_MODEL_NAME=./models/bge-m3
RAG_RETRIEVAL_K=5
MILVUS_EMBEDDING_DIM=1024

# 应用
APP_ENV=local
APP_DEBUG=False
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```


## 📚 开发指南

### 添加新的意图分类

1. 在 `data/` 目录准备训练数据
2. 运行 `python scripts/preprocess_intent_4class.py` 预处理
3. 运行 `python -m backend.training.train_intent` 训练
4. 模型自动保存到 `backend/training/saved_models/intent_classifier/`

### 添加新的检索策略

参考 `backend/rag/hybrid_searcher.py` 的多路并行模式，扩展现有检索器：
- 稠密向量检索 (BGE-M3)
- 稀疏向量检索
- 多向量检索
- BM25 关键词检索
- RRF 融合 (k=60)

### 添加新 Agent

1. 在 `backend/agents/` 下创建新包（`__init__.py` + `state.py` + `nodes.py` + `prompts.py` + `graph.py`）
2. 在 `backend/agents/__init__.py` 注册惰性导入
3. 在 `backend/agents/graph.py` 注册节点和条件边
4. 在 `backend/models/schemas.py` 添加对应的枚举/模型

### 运行测试

```bash
# 环境检查
python scripts/check_env.py

# 连接测试
python scripts/check_connections.py

# 端到端测试
python tests/test_agent.py
```

## 📄 License

MIT
