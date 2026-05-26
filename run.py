"""PyAgent 启动入口 — 在 PyCharm 中直接 Run/Debug 此文件即可"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
