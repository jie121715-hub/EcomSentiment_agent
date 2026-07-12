#!/usr/bin/env python
# wechat_bridge.py — ACP-compatible bridge: WeChat messages → FastAPI /chat → WeChat response
#
# 配合 cc-connect 使用:
#   cc-connect weixin setup --project ecom
#   cc-connect --config config.toml
#
# ACP protocol (JSON-RPC over stdio):
#   Request:  {"jsonrpc":"2.0","method":"chat","params":{"prompt":"..."},"id":1}
#   Response: {"jsonrpc":"2.0","result":{"content":"..."},"id":1}

import sys
import json
import httpx
import asyncio

API_URL = "http://localhost:8000/chat"


async def handle_chat(prompt: str, session_id: str = "wechat") -> str:
    """Forward message to FastAPI /chat and return the answer."""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(API_URL, json={
                "query": prompt,
                "user_id": f"wechat_{session_id}",
                "session_id": session_id,
                "history": [],
            })
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "抱歉，系统暂无响应")
            return f"系统异常 (HTTP {resp.status_code})"
        except httpx.ConnectError:
            return "⚠️ 后端服务未启动，请先运行 python main.py"
        except Exception as e:
            return f"⚠️ 请求失败: {str(e)[:100]}"


async def main():
    """ACP stdio loop — read JSON-RPC, call FastAPI, write response."""
    loop = asyncio.get_event_loop()

    # Use stdin reader for non-blocking reads
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line = await reader.readline()
            if not line:
                break

            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue

            req_id = req.get("id")
            method = req.get("method", "")
            params = req.get("params", {})

            if method == "chat":
                prompt = params.get("prompt", "")
                session_id = params.get("session_id", "wechat")
                answer = await handle_chat(prompt, session_id)

                resp = {
                    "jsonrpc": "2.0",
                    "result": {"content": answer},
                    "id": req_id,
                }
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "result": "pong", "id": req_id}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    "id": req_id,
                }

            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()

        except Exception as e:
            err = {"jsonrpc": "2.0", "error": {"code": -1, "message": str(e)}, "id": None}
            sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
