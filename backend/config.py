# backend/config.py
# 全项目唯一的「配置中心」：从 .env.local 读取所有配置项，供任何模块取用。
# 设计原则：任何地方要用配置都从这里取，绝不在代码里硬编码。

from pydantic_settings import BaseSettings   # Pydantic 的「配置基类」，会自动从 .env.local 读取 + 类型校验
from functools import lru_cache              # 缓存函数结果，保证只创建一次


class Settings(BaseSettings):
    """配置模型：每个类属性对应 .env.local 里的一项配置。
    继承 BaseSettings 后，Pydantic 会自动把同名（大小写不敏感）的配置读进来并转成对应类型。
    """

    # ── 大模型（DeepSeek 推荐）──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # ── 大模型（阿里云百炼 DashScope 备选）──
    dashscope_api_key: str = ""                          # 备选：DashScope API Key
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ── 淘宝开放平台（RAG搜不到时自动兜底搜索）──
    taobao_app_key: str = ""
    taobao_app_secret: str = ""
    taobao_session_key: str = ""

    # ── 快递鸟物流查询 ──
    kdn_ebusiness_id: str = ""
    kdn_api_key: str = ""

    # ── LLM 默认参数 ──
    llm_default_model: str = "deepseek-v4"               # 默认模型：DeepSeek V4
    llm_default_temperature: float = 0.0                  # 默认温度：0（稳定输出）
    llm_streaming_temperature: float = 0.3                # 流式输出温度

    # ── 情感分析模型路径 ──
    bert_base_path: str = "../TMFCode_随堂代码/04-bert/bert-base-chinese"
    sentiment_model_path: str = "../EcomSentiment/04-bert/save_models/bert_classifier_model.pt"
    sentiment_bert_path: str = "../EcomSentiment/04-bert/SturctBERT"
    sentiment_label_map: str = "../EcomSentiment/04-bert/SturctBERT/label_mapping.json"
    # 自训练模型（PerceptionAgent 自动发现，优先级最高）
    trained_intent_dir: str = "./backend/training/saved_models/intent_classifier"
    trained_sentiment_dir: str = "./backend/training/saved_models/sentiment_classifier"

    # ── RAG 检索配置（v3 升级：混合检索 + 精排 + 质检）──
    rag_data_dir: str = "../EcomSentiment_RAG/rag_qa/data/ecom_data"
    rag_retrieval_k: int = 5                              # 最终返回文档数（精排后 Top-K）
    rag_candidate_m: int = 3                              # 最终选取的上下文文档数
    rag_relevance_threshold: float = 0.6                  # 检索质量阈值（首条 < 此值 → 兜底）

    # RAG v3 新增：混合检索参数
    rag_hybrid_enabled: bool = True                       # 是否启用混合检索（稠密+稀疏）
    rag_dense_top_n: int = 50                             # 稠密向量检索候选数
    rag_sparse_top_n: int = 30                            # 稀疏向量检索候选数
    rag_hybrid_top_n: int = 40                            # 混合检索候选数
    rag_multi_vector_top_n: int = 30                      # 多向量检索候选数
    rag_rrf_k: int = 60                                   # RRF 融合参数

    # RAG v3 新增：父子块切分
    rag_parent_child_enabled: bool = True                 # 是否启用父子块策略
    rag_parent_chunk_size: int = 1000                     # 父块大小（给LLM的完整上下文）
    rag_parent_chunk_overlap: int = 200                   # 父块重叠
    rag_child_chunk_size: int = 300                       # 子块大小（用于向量检索）
    rag_child_chunk_overlap: int = 50                     # 子块重叠

    # RAG v3 新增：Query 改写
    rag_query_rewrite_enabled: bool = True                # 是否启用 Query 改写
    rag_synonym_count: int = 3                            # 同义扩展生成数量

    # RAG v3 新增：精排 & 后处理
    rag_reranker_enabled: bool = True                     # 是否启用 BGE-Reranker 精排
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"   # 精排模型
    rag_post_process_enabled: bool = True                 # 是否启用答案后处理

    # RAG v3 新增：缓存
    rag_cache_enabled: bool = True                       # Redis 问答缓存（高频问答加速）
    redis_host: str = "localhost"                         # Redis 地址
    redis_port: int = 6379                                # Redis 端口
    redis_password: str = "1234"                          # Redis 密码
    redis_db: int = 0                                     # Redis 数据库编号
    rag_cache_ttl: int = 3600                             # 缓存过期时间（秒）

    # 🆕 BM25 + MySQL FAQ 检索（三层架构第二层）
    bm25_enabled: bool = True                            # 是否启用 BM25 FAQ 检索
    bm25_threshold: float = 0.85                         # BM25 softmax 阈值（超过此值视为命中）
    bm25_cache_ttl: int = 7200                           # BM25 结果 Redis 缓存时间（秒）

    # ── 向量数据库（Milvus 优先 → Chroma 自动降级）──
    milvus_host: str = "localhost"                           # Milvus 服务地址
    milvus_port: int = 19530                                 # Milvus gRPC 端口
    # 🆕 v3 双 Collection + 多租户 (1024维 BGE-M3)
    milvus_product_collection: str = "ecom_products_v1"     # 商品描述向量
    milvus_policy_collection: str = "ecom_policies_v1"      # 店铺政策向量
    milvus_embedding_dim: int = 1024                        # BGE-M3 稠密向量维度
    default_shop_id: str = "shop_001"                        # 默认店铺ID(多租户)
    chroma_persist_dir: str = "./data/chroma_db"             # Chroma 降级路径
    embedding_model_name: str = "C:/Users/23387/Desktop/新建文件夹/EcomSentiment_RAG/rag_qa/models/bge-m3"  # 🆕 BGE-M3 本地路径
    hf_endpoint: str = "https://hf-mirror.com"               # HuggingFace 镜像（国内加速）

    # ── MySQL 数据库 ──
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "123456"
    db_name: str = "ecom_agent"

    # ── 应用基础配置 ──
    app_env: str = "local"                                 # 运行环境：local / dev / prod
    app_debug: bool = True                                 # 调试模式
    app_host: str = "0.0.0.0"                              # 监听地址
    app_port: int = 8000                                   # 监听端口
    log_level: str = "INFO"                                # 日志级别
    customer_service_phone: str = "400-618-8888"           # 人工客服电话

    # ── 重试配置 ──
    retry_max_attempts: int = 3                            # 最大重试次数
    retry_base_delay: float = 1.0                          # 重试基础延迟（秒）
    retry_max_delay: float = 30.0                          # 重试最大延迟（秒）

    # ── Router v3 决策配置 ──
    router_intent_confidence_threshold: float = 0.50       # 意图置信度阈值（低于此值 → clarify反问）
    router_urgency_enabled: bool = True                    # 是否启用情绪紧急度检测

    # ── 🆕 API 鉴权配置 ──
    admin_api_key: str = "admin"                       # 管理员 API Key
    merchant_api_key: str = "merchant"                 # 商户 API Key

    # ── 对话记忆配置 ──
    max_history_turns: int = 10                            # 最多保留的对话轮数

    @property
    def database_url(self) -> str:
        """拼出模型文件目录的相对路径（v1 暂不接 Postgres，预留）。"""
        return f"sqlite:///./data/agent_v1.db"

    class Config:
        """Pydantic 的元配置：告诉 BaseSettings 该怎么读取配置。"""
        env_file = ".env.local"                            # 从这个文件读取
        env_file_encoding = "utf-8"
        case_sensitive = False                             # 大小写不敏感
        extra = "ignore"                                   # 多出来的字段忽略，不报错


@lru_cache()                                              # 保证只创建一次 Settings 实例
def get_settings() -> Settings:
    """获取全局唯一的配置对象。任何模块要用配置，都调用这个函数。"""
    return Settings()
