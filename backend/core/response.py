# backend/core/response.py
# 标准 API 响应封装：所有端点统一返回 {success, code, message, data, error}

from typing import Any, Optional


def success_response(data: Any = None, message: str = "ok") -> dict:
    """成功响应。"""
    return {
        "success": True,
        "code": 200,
        "message": message,
        "data": data,
        "error": None,
    }


def error_response(code: int, message: str, error: Optional[str] = None) -> dict:
    """错误响应。"""
    return {
        "success": False,
        "code": code,
        "message": message,
        "data": None,
        "error": error or message,
    }
