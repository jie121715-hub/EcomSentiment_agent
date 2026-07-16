# backend/api/v1/chat.py
# 对话接口：POST /chat（非流式） + POST /chat/stream（SSE 流式）

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.models.schemas import ChatRequest, AgentResponse
from backend.agents.graph import run_shopping_guide, run_shopping_guide_stream

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=AgentResponse)
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
            shop_id=request.shop_id,
        )
        elapsed = (time.time() - start_time) * 1000
        logger.info("api.chat_done", time_ms=f"{elapsed:.0f}")
        return response

    except Exception as e:
        logger.error("api.chat_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@router.post("/chat/stream")
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
                shop_id=request.shop_id,
            ):
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
            "X-Accel-Buffering": "no",
        },
    )
