# main.py
# 云答智能客服系统 v1 — 本地开发启动器
#
# 启动方式：
#   python main.py
#
# 生产部署请使用：
#   uvicorn backend.main:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn
    from backend.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
    )
