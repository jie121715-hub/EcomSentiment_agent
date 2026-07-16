# backend/main.py
# 云答智能客服系统 — FastAPI 应用入口。
#
# 启动方式：
#   cd EcomSentiment_agent
#   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
#
# 访问：
#   - API 文档: http://localhost:8000/docs
#   - 聊天接口: POST http://localhost:8000/api/v1/chat
#   - 流式接口: POST http://localhost:8000/api/v1/chat/stream
#   - 健康检查: GET  http://localhost:8000/health

# ── Windows 事件循环兼容修复（Python 3.13 + asyncmy 必须）──
import sys as _sys
if _sys.platform == "win32":
    import asyncio as _asyncio
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

# ── HuggingFace 国内镜像（必须在所有 import 之前设置）──
import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── Windows DLL 预加载：避免 pyarrow 多线程加载崩溃 ──
from backend.utils import preload_dlls as _pd
_pd._preload()

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend.config import get_settings
from backend.core.logger import configure_logging, get_logger
from backend.api.router import api_router

logger = get_logger(__name__)


# ── 应用生命周期 ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理。"""
    settings = get_settings()
    configure_logging()
    logger.info(
        "app.starting",
        env=settings.app_env,
        host=settings.app_host,
        port=settings.app_port,
    )

    # 初始化 MySQL 表
    try:
        from backend.core.database import init_db
        await init_db()
        logger.info("app.mysql_ready")
    except Exception as e:
        logger.warning("app.mysql_init_failed", error=str(e))

    # 预热 Agent（提前加载模型，避免首次请求等待）
    try:
        logger.info("app.preloading_agents")
        from backend.agents.perception import PerceptionAgent
        from backend.agents.router import RoutingAgent
        from backend.agents.knowledge_qa import KnowledgeQAAgent
        from backend.agents.business import BusinessAgent
        PerceptionAgent()
        RoutingAgent()
        KnowledgeQAAgent()
        BusinessAgent()
        logger.info("app.agents_ready")
    except Exception as e:
        logger.warning("app.preload_warning", error=str(e))

    # 预热 BGE-M3 模型（加载 + dummy encode + chunker + retriever）
    try:
        logger.info("app.preloading_bge_m3")
        from backend.api.v1.admin_upload import _get_bge_model, _get_chunker, _get_milvus_client
        model = _get_bge_model()
        _get_chunker()
        _get_milvus_client()
        _ = model.encode(["预热"], normalize_embeddings=True)

        # 预热 MultiTenantRetriever（聊天检索用）
        from backend.rag.multi_tenant_retriever import MultiTenantRetriever
        mt = MultiTenantRetriever()
        _ = mt.model
        _ = mt.client

        logger.info("app.bge_m3_ready")
    except Exception as e:
        logger.warning("app.bge_m3_preload_warning", error=str(e))

    # 预热 BM25 FAQ 索引
    try:
        logger.info("app.preloading_bm25")
        from backend.rag.bm25_search import get_bm25
        await get_bm25()
        logger.info("app.bm25_ready")
    except Exception as e:
        logger.warning("app.bm25_preload_warning", error=str(e))

    yield  # 应用运行中...

    # 关闭时
    logger.info("app.shutting_down")


# ── 创建 FastAPI 应用 ────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="🛒 云答智能客服系统",
    description="""
## 智能客服系统 v2

基于 **感知-路由-分发** 多Agent协同架构的智能客服系统。

### Agent 体系
| Agent | 职责 | 数据源 |
|-------|------|--------|
| 🎯 **感知Agent** | 情感分析 + 意图识别 + 实体抽取 | BERT本地模型 |
| 🧠 **路由Agent** | 策略选择 + 语气注入 + 转人工判断 | 情感→话术映射表 |
| 📚 **知识应答Agent** | RAG语义检索 + LLM生成回复（含闲聊兜底） | Milvus向量库 |
| 📦 **业务查询Agent** | 查物流/订单/库存 + 修改订单 | MySQL |

### 请求流向
```
用户输入 → 感知(情感+意图+实体)
       → 路由(三维决策)
       → 分发:
           ├─ 澄清反问 → 返回选项供用户确认
           ├─ 转人工   → 工单接管
           ├─ 业务处理 → 查物流/订单/库存/修改（MySQL）
           └─ 知识问答 → RAG检索 + LLM生成（含闲聊兜底）
```
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局异常处理 ────────────────────────────────────────────

from backend.core.response import error_response


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """HTTP 异常 → 标准错误响应。"""
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(code=exc.status_code, message=str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """请求参数校验失败 → 422 标准错误响应。"""
    errors = exc.errors()
    detail = errors[0]["msg"] if errors else "参数校验失败"
    return JSONResponse(
        status_code=422,
        content=error_response(code=422, message="参数校验失败", error=detail),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """未捕获异常 → 500 标准错误响应。"""
    logger.error("unhandled_exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_response(code=500, message="服务器内部错误"),
    )

# ── 挂载 API 路由 ──
app.include_router(api_router, prefix="/api/v1")


# ── 基础路由（健康检查、首页）─────────────────────────────

@app.get("/health")
async def health_check():
    """健康检查端点。"""
    return {
        "status": "ok",
        "service": "云答智能客服系统",
        "version": "1.0.0",
        "env": settings.app_env,
    }


@app.get("/")
async def root():
    """根路径：API 服务信息。前端请使用 npm run dev 启动 Vue 开发服务器。"""
    return JSONResponse({
        "service": "云答智能客服系统",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "cd frontend && npm run dev  →  http://localhost:3000",
    })
