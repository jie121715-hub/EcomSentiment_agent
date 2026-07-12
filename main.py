# main.py
# 电商领域智能问答与业务处理Agent系统 v1 — FastAPI 应用入口。
#
# 启动方式：
#   cd EcomSentiment_agent
#   python main.py
#   # 或: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# 访问：
#   - API 文档: http://localhost:8000/docs
#   - 聊天接口: POST http://localhost:8000/chat
#   - 流式接口: POST http://localhost:8000/chat/stream
#   - 健康检查: GET  http://localhost:8000/health

# ── HuggingFace 国内镜像（必须在所有 import 之前设置）──
import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── 🆕 Windows DLL 预加载：避免 pyarrow 多线程加载崩溃 ──
import _preload_dlls as _pd  # noqa: E402
_pd._preload()

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from backend.config import get_settings
from backend.core.logger import configure_logging, get_logger
from backend.models.schemas import (
    ChatRequest, AgentResponse, ChatEvent,
)
from backend.agents.graph import run_shopping_guide, run_shopping_guide_stream

logger = get_logger(__name__)


# ── 应用生命周期 ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理。"""
    # 启动时
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
        from backend.agents.knowledge_mgmt import KnowledgeMgmtAgent
        PerceptionAgent()
        RoutingAgent()
        KnowledgeQAAgent()
        BusinessAgent()
        KnowledgeMgmtAgent()
        logger.info("app.agents_ready")
    except Exception as e:
        logger.warning("app.preload_warning", error=str(e))

    # 🆕 预热 BM25 FAQ 索引（加载 MySQL FAQ 数据 → 构建 BM25 索引）
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
    title="🛒 电商领域智能问答与业务处理Agent系统 v1",
    description=f"""
## 智能客服系统 v2

基于 **感知-路由-分发** 多Agent协同架构的智能客服系统。

### Agent 体系
| Agent | 职责 | 数据源 |
|-------|------|--------|
| 🎯 **感知Agent** | 情感分析 + 意图识别 + 实体抽取 | BERT本地模型 |
| 🧠 **路由Agent** | 策略选择 + 语气注入 + 转人工判断 | 情感→话术映射表 |
| 📚 **知识应答Agent** | RAG语义检索 + LLM生成回复 | Milvus/Chroma向量库 |
| 📦 **业务查询Agent** | 查物流/订单/库存 | MySQL |
| 💬 **闲聊Agent** | 社交对话兜底 | LLM（不走RAG） |
| 📝 **知识收纳Agent** | 商户录入政策/规则 | MySQL + 向量库双写 |

### 请求流向
```
用户输入 → 感知(情感+意图+实体)
       → 意图分发:
           ├─ 闲聊 → 闲聊Agent（纯聊天）
           ├─ 查物流/改订单 → 业务查询Agent（MySQL精确查询）
           ├─ 知识管理 → 知识收纳Agent（商户录入知识）
           └─ 默认 → 知识应答Agent（RAG检索 + LLM生成）
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

# 挂载静态文件（前端页面）
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── API 路由 ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """健康检查端点。"""
    return {
        "status": "ok",
        "service": "电商领域智能问答与业务处理Agent系统 v1",
        "version": "1.0.0",
        "env": settings.app_env,
    }


@app.post("/chat", response_model=AgentResponse)
async def chat(request: ChatRequest):
    """对话接口（非流式）：发送用户消息，获取完整回复。

    请求示例：
    ```json
    {
        "query": "这件衣服会起球吗？",
        "user_id": "user_001",
        "session_id": "sess_001",
        "history": []
    }
    ```
    """
    start_time = time.time()
    logger.info("api.chat_request", query=request.query[:50], user_id=request.user_id)

    try:
        response = await run_shopping_guide(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            history=request.history,
        )
        elapsed = (time.time() - start_time) * 1000
        logger.info("api.chat_done", time_ms=f"{elapsed:.0f}")
        return response

    except Exception as e:
        logger.error("api.chat_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """对话接口（SSE 流式）：发送用户消息，实时接收 token 级别回复。

    事件类型：
    - `perception`: 感知结果（情感+意图）
    - `route`: 路由决策（策略+语气）
    - `token`: LLM 生成 token
    - `done`: 生成完成
    - `error`: 出错
    """
    logger.info("api.chat_stream_request", query=request.query[:50])

    async def event_generator():
        try:
            async for event in run_shopping_guide_stream(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                history=request.history,
            ):
                # SSE 格式: event: <type>\ndata: <json>\n\n
                event_type = event.event if hasattr(event, 'event') else "token"
                event_data = event.data if hasattr(event, 'data') else ""
                yield f"event: {event_type}\ndata: {event_data}\n\n"
        except Exception as e:
            logger.error("api.stream_error", error=str(e))
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


# ── 知识库管理 API ───────────────────────────────────────────

from pydantic import BaseModel as PydanticBaseModel, Field

class KnowledgeItem(PydanticBaseModel):
    content: str = Field(..., description="知识内容（商品介绍/政策规则等）")
    category: str = Field(default="product", description="分类: product / policy / faq")
    merchant_id: str = Field(default="default", description="商户ID")

class KnowledgeSyncResponse(PydanticBaseModel):
    success: bool
    synced: int = 0
    chunks: int = 0
    backend: str = ""
    error: str = ""


@app.get("/admin/knowledge")
async def list_knowledge(category: str = "", merchant_id: str = "", limit: int = 50):
    """查看知识库内容。可选按分类/商户过滤。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge
    from sqlalchemy import select

    async with get_session() as session:
        stmt = select(CustomKnowledge)
        if category:
            stmt = stmt.where(CustomKnowledge.category == category)
        if merchant_id:
            stmt = stmt.where(CustomKnowledge.merchant_id == merchant_id)
        stmt = stmt.order_by(CustomKnowledge.id.desc()).limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return {
        "total": len(rows),
        "items": [{
            "id": r.id, "content": r.content, "category": r.category,
            "merchant_id": r.merchant_id, "source": r.source,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        } for r in rows],
    }


@app.post("/admin/knowledge")
async def add_knowledge(item: KnowledgeItem):
    """添加知识到 MySQL 并同步向量库。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge
    from backend.agents.knowledge_mgmt import KnowledgeMgmtAgent

    # 写入 MySQL
    async with get_session() as session:
        record = CustomKnowledge(
            content=item.content,
            category=item.category,
            merchant_id=item.merchant_id,
            source=f"merchant:{item.merchant_id}",
        )
        session.add(record)
        await session.commit()
        kb_id = record.id

    # 同步向量库
    agent = KnowledgeMgmtAgent()
    sync_result = await agent.sync_all_to_vector()

    logger.info("admin.knowledge_added", id=kb_id, category=item.category)
    return {
        "success": True, "id": kb_id,
        "content": item.content[:100] + "...",
        "sync": sync_result,
    }


@app.post("/admin/knowledge/batch")
async def batch_add_knowledge(items: list[KnowledgeItem]):
    """批量添加知识（一次最多100条）。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge
    from backend.agents.knowledge_mgmt import KnowledgeMgmtAgent

    if len(items) > 100:
        raise HTTPException(status_code=400, detail="单次最多100条")

    async with get_session() as session:
        for item in items:
            session.add(CustomKnowledge(
                content=item.content,
                category=item.category,
                merchant_id=item.merchant_id,
                source=f"merchant:{item.merchant_id}",
            ))
        await session.commit()

    agent = KnowledgeMgmtAgent()
    sync_result = await agent.sync_all_to_vector()

    logger.info("admin.knowledge_batch", count=len(items))
    return {"success": True, "count": len(items), "sync": sync_result}


@app.delete("/admin/knowledge/{kb_id}")
async def delete_knowledge(kb_id: int):
    """删除指定知识（MySQL）。向量库残留需调 sync 清理。"""
    from backend.core.database import get_session
    from backend.models.db_models import CustomKnowledge
    from sqlalchemy import select

    async with get_session() as session:
        result = await session.execute(select(CustomKnowledge).where(CustomKnowledge.id == kb_id))
        record = result.scalar_one_or_none()
        if not record:
            raise HTTPException(status_code=404, detail=f"知识 {kb_id} 不存在")
        await session.delete(record)
        await session.commit()

    return {"success": True, "deleted_id": kb_id}


@app.post("/admin/knowledge/sync", response_model=KnowledgeSyncResponse)
async def sync_knowledge():
    """全量同步：MySQL → 向量库（数据变更后调用）。"""
    from backend.agents.knowledge_mgmt import KnowledgeMgmtAgent
    agent = KnowledgeMgmtAgent()
    result = await agent.sync_all_to_vector()
    return KnowledgeSyncResponse(**result)


# ── 淘宝导入 API ───────────────────────────────────────────

class TaobaoImportRequest(PydanticBaseModel):
    app_key: str = Field(..., description="淘宝开放平台 App Key")
    app_secret: str = Field(..., description="淘宝开放平台 App Secret")
    session_key: str = Field(..., description="卖家授权 session（oauth token）")
    merchant_id: str = Field(default="taobao_shop", description="商户标识")
    page_size: int = Field(default=50, ge=1, le=200)


@app.post("/admin/knowledge/import-taobao")
async def import_taobao(req: TaobaoImportRequest):
    """从淘宝 API 导入在售商品到知识库。"""
    from backend.taobao_importer import import_products, TbConfig

    config = TbConfig(
        app_key=req.app_key,
        app_secret=req.app_secret,
        session_key=req.session_key,
        merchant_id=req.merchant_id,
        page_size=req.page_size,
    )
    result = await import_products(config)
    logger.info("admin.taobao_import", count=result.get("imported", 0))
    return result


# ── 🛡️ API Key 鉴权 ──────────────────────────────────────

from fastapi import Header

async def verify_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> dict:
    """验证 API Key 并返回用户身份。

    鉴权规则：
      - 无 Key / 无效 Key → role="customer"（只能聊天，不能上传）
      - merchant Key     → role="merchant"（可上传知识）
      - admin Key        → role="admin"（可上传 + 管理）

    使用方式（FastAPI Dependency）：
      @app.post("/admin/xxx")
      async def admin_endpoint(auth: dict = Depends(verify_api_key)):
          if auth["role"] not in ("merchant", "admin"):
              raise HTTPException(403)
    """
    if not x_api_key:
        return {"role": "customer", "merchant_id": "default", "authenticated": False}

    settings = get_settings()

    # 管理员 Key
    if x_api_key == settings.admin_api_key:
        return {"role": "admin", "merchant_id": "all", "authenticated": True}

    # 商户 Key
    if x_api_key == settings.merchant_api_key:
        return {"role": "merchant", "merchant_id": "default", "authenticated": True}

    # 无效 Key → 降级为 customer
    return {"role": "customer", "merchant_id": "default", "authenticated": False}


# ── 🆕 文件上传知识库 API ────────────────────────────────────

from fastapi import UploadFile, File, Form, Depends

# ── 🛡️ 恶意内容检测 ──────────────────────────────────────
# 破坏性模式：承诺超额赔付、假冒官方、虚假法律声明等
_MALICIOUS_PATTERNS = [
    # 超额赔付承诺（风险极高）
    ("假一赔十", "承诺超额赔付「假一赔十」，与法律规定的假一赔三不符，涉嫌虚假承诺"),
    ("假一赔百", "承诺超额赔付，涉嫌欺诈"),
    ("十倍赔偿", "承诺超额赔偿，存在法律风险"),
    ("百倍赔偿", "承诺超额赔偿，涉嫌虚假宣传"),
    ("无条件退款", "承诺无条件退款，不符合平台售后规则"),
    ("永久保修", "承诺永久保修，超出法定三包期限"),
    ("终身质保", "承诺终身质保，可能无法兑现"),
    ("全额退款不退货", "承诺仅退款不退货，违反平台交易规则"),
    # 假冒官方
    ("官方授权", "声称官方授权，需提供授权证明文件"),
    ("品牌直营", "声称品牌直营，需提供品牌授权链路证明"),
    ("国家级", "使用「国家级」等权威背书字眼，需提供认证证书"),
    ("100%有效", "使用绝对化承诺用语，涉嫌虚假宣传"),
    ("治愈率", "涉及医疗功效承诺，违反广告法"),
    ("包治", "涉及医疗功效承诺，违反广告法"),
    # 平台违规
    ("加微信", "引导站外交易（微信），违反平台规定"),
    ("扫码下单", "可能引导站外交易，需人工审核"),
    ("私聊下单", "引导私下交易，存在交易风险"),
    ("货到付款", "涉及非平台担保交易模式，需确认合规性"),
]

# 安全关键词（正常政策描述，不应被误杀）
_SAFE_PATTERNS = [
    "7天无理由", "退换货", "退款", "退货", "运费险",
    "包邮", "优惠券", "满减", "折扣", "质保", "保修",
]


def _scan_content(content: str) -> dict:
    """扫描上传内容，检测恶意/风险模式。

    :return: {"safe": bool, "risks": [str], "verdict": "pass|review|reject"}
    """
    risks = []
    for pattern, reason in _MALICIOUS_PATTERNS:
        if pattern in content:
            risks.append({"pattern": pattern, "reason": reason})

    if not risks:
        return {"safe": True, "risks": [], "verdict": "pass"}

    # 判断风险等级
    critical_keywords = ["假一赔", "十倍", "百倍", "治愈", "包治"]
    has_critical = any(kw in content for kw in critical_keywords)

    return {
        "safe": False,
        "risks": risks,
        "verdict": "reject" if has_critical else "review",
    }


@app.post("/admin/knowledge/upload")
async def upload_knowledge_file(
    file: UploadFile = File(..., description="PDF 或 DOCX 文件"),
    category: str = Form(default="product", description="知识分类: product / policy / faq"),
    auth: dict = Depends(verify_api_key),
):
    """上传 PDF/DOCX 文件，自动提取文本并写入知识库。

    🛡️ 安全机制：
      - API Key 鉴权：请求头 X-API-Key 校验身份
      - 无 Key → customer（拒绝上传）
      - merchant Key → 可上传，merchant_id 自动绑定
      - admin Key → 可上传 + 管理全部商户
      - 内容扫描：检测超额赔付、虚假承诺、站外导流等恶意模式

    支持格式：.pdf / .docx（最大 50MB）
    """
    import os as _os
    import tempfile
    import time as _time

    # ── 0. 鉴权校验 ──────────────────────────────────────
    if auth["role"] not in ("merchant", "admin"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "仅商户或管理员可上传知识",
                "hint": "请在请求头中添加 X-API-Key",
                "current_role": auth["role"],
            },
        )

    merchant_id = auth["merchant_id"]

    # ── 1. 校验文件类型 ──────────────────────────────────
    filename = file.filename or "unknown"
    ext = _os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 {ext}，仅支持 PDF / DOCX",
        )

    # ── 2. 保存临时文件 ──────────────────────────────────
    try:
        content = await file.read()
        file_size = len(content)
        if file_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小不能超过 50MB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("admin.upload_received",
                   filename=filename, size=file_size, role=auth["role"])

        # ── 3. 文档加载 — 提取文本 ──────────────────────
        from backend.rag.doc_loader import DocLoader
        loader = DocLoader()
        docs = loader.load(tmp_path)

        # 清理临时文件
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

        if not docs:
            raise HTTPException(status_code=400, detail="文件内容为空或无法提取文本")

        full_text = docs[0].page_content
        text_length = len(full_text)

        # ── 4. 🛡️ 恶意内容扫描 ──────────────────────────
        scan_result = _scan_content(full_text)

        if scan_result["verdict"] == "reject":
            risk_details = "; ".join(r["reason"] for r in scan_result["risks"])
            logger.warning("admin.upload_rejected",
                          filename=filename, risks=risk_details)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "文件内容包含违规信息，已拒绝上传",
                    "risks": [
                        {"pattern": r["pattern"], "reason": r["reason"]}
                        for r in scan_result["risks"]
                    ],
                },
            )

        needs_review = (scan_result["verdict"] == "review")

        # ── 5. 🧬 父子块切分（与RAG管线一致）─────────────
        from backend.rag.chunker import ParentChildChunker
        from langchain_core.documents import Document as LCDocument

        chunker = ParentChildChunker()
        # 给文档打上元数据标签
        tagged_doc = LCDocument(
            page_content=full_text,
            metadata={
                "source": f"upload:{merchant_id}:{filename}",
                "category": category,
                "merchant_id": merchant_id,
                "uploaded_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        chunk_result = chunker.split_documents([tagged_doc])

        # ── 6. 写入 MySQL（原文持久化）───────────────────
        from backend.core.database import get_session
        from backend.models.db_models import CustomKnowledge

        status = "pending_review" if needs_review else "active"
        mysql_ids = []

        async with get_session() as session:
            # 父块存 MySQL（完整上下文）
            for parent in chunk_result.parent_chunks:
                record = CustomKnowledge(
                    content=parent.page_content,
                    source=f"upload:{filename}",
                    category=category,
                    merchant_id=merchant_id,
                )
                session.add(record)
                await session.flush()  # 获取自增ID
                mysql_ids.append(record.id)
            await session.commit()

        # ── 7. 🧬 写入 Milvus（父子块双写，与RAG检索一致）─
        from backend.rag.retriever import EcomRetriever
        retriever = EcomRetriever()
        vector_ok = False
        parent_count = 0
        child_count = 0

        if retriever.vector_store is not None:
            try:
                # 子块写入（检索用，带 chunk_type="child" + parent_id）
                retriever.vector_store.add_documents(chunk_result.child_chunks)
                child_count = len(chunk_result.child_chunks)

                # 父块写入（LLM上下文用，带 chunk_type="parent"）
                retriever.vector_store.add_documents(chunk_result.parent_chunks)
                parent_count = len(chunk_result.parent_chunks)

                # 更新内存映射表
                for pid, pcontent in chunk_result.parent_map.items():
                    retriever._parent_map[pid] = pcontent

                vector_ok = True
                logger.info("admin.upload_milvus_done",
                           parents=parent_count, children=child_count,
                           backend=retriever._backend)
            except Exception as e:
                logger.error("admin.upload_milvus_failed", error=str(e))
        else:
            logger.warning("admin.upload_no_vector_store")

        # ── 8. 返回结果 ──────────────────────────────────
        elapsed = _time.time() - _time.time()  # approximate
        logger.info("admin.upload_done",
                   filename=filename, text_len=text_length,
                   parents=parent_count, children=child_count,
                   vector_ok=vector_ok, needs_review=needs_review)

        result = {
            "success": True,
            "filename": filename,
            "file_size": file_size,
            "text_length": text_length,
            "mysql_ids": mysql_ids,
            "chunks": {
                "parents": parent_count,
                "children": child_count,
                "ratio": f"1:{child_count // max(parent_count, 1)}",
            },
            "vector_written": vector_ok,
            "backend": retriever._backend or "unknown",
            "status": status,
            "review_note": (
                "内容含可疑表述，已保存但待人工审核后生效"
                if needs_review else None
            ),
        }

        if scan_result["risks"]:
            result["scan_notes"] = [
                {"pattern": r["pattern"], "reason": r["reason"]}
                for r in scan_result["risks"]
            ]

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin.upload_failed", filename=filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


# ── 🆕 全链路调试页面 ─────────────────────────────────────

@app.get("/debug")
async def debug_full():
    """全链路Agent调试面板 — 可视化Perception→Router→Dispatch完整链路。"""
    import os as _os
    debug_path = _os.path.join(_os.path.dirname(__file__), "static", "debug_full.html")
    if _os.path.isfile(debug_path):
        return FileResponse(debug_path)
    return JSONResponse({"error": "debug_full.html not found"})


# ── 调试页面 ─────────────────────────────────────────────────

@app.get("/")
async def root():
    """根路径：返回智能问答前端页面。"""
    import os as _os
    index_path = _os.path.join(_os.path.dirname(__file__), "static", "index.html")
    if _os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "service": "电商领域智能问答与业务处理Agent系统 v1",
        "docs": "/docs",
        "frontend": "http://localhost:8000",
    })


# ── 启动入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
    )
