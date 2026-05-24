from pathlib import Path
from fastapi import FastAPI
from sqlalchemy import text as sa_text

from src.db.engine import create_engine, init_db, async_session_factory
from src.tools.registry import ToolRegistry
from src.tools.builtin import (
    ReadFileTool, WriteFileTool, ListDirTool,
    SearchFileTool, GrepContentTool,
    ExecShellTool, HttpRequestTool, DbQueryTool,
)
from src.skills.registry import SkillRegistry
from src.models import load_model_config, get_routing_config, OpenAICompatProvider
from src.memory.sqlite_provider import SqliteMemoryProvider
from src.server.routers.session import router as session_router
from src.server.routers.chat import router as chat_router
from config.settings import DB_PATH, MODEL_CONFIG_PATH, SKILLS_CONFIG_PATH


def create_app(db_url: str | None = None):
    app = FastAPI(title="PyAgent", version="0.1.0")
    _db_url = db_url or f"sqlite+aiosqlite:///{DB_PATH}"

    @app.on_event("startup")
    async def startup():
        # Database
        engine = create_engine(_db_url)
        await init_db(engine)

        # Checkpoint tables for LangGraph
        async with engine.begin() as conn:
            await conn.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
            """))
            await conn.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
            """))

        # Checkpoint saver via custom ainvoke config (no SqliteSaver needed)
        app.state.checkpoint_saver = None

        # Tool Registry
        app.state.tool_registry = ToolRegistry()
        app.state.tool_registry.register(ReadFileTool())
        app.state.tool_registry.register(WriteFileTool())
        app.state.tool_registry.register(ListDirTool())
        app.state.tool_registry.register(SearchFileTool())
        app.state.tool_registry.register(GrepContentTool())
        app.state.tool_registry.register(ExecShellTool())
        app.state.tool_registry.register(HttpRequestTool())
        app.state.tool_registry.register(DbQueryTool())

        # Skill Registry
        if Path(SKILLS_CONFIG_PATH).exists():
            app.state.skill_registry = SkillRegistry.load_from_yaml(str(SKILLS_CONFIG_PATH))
        else:
            app.state.skill_registry = SkillRegistry()

        # Model Provider
        providers = load_model_config(MODEL_CONFIG_PATH)
        routing = get_routing_config(MODEL_CONFIG_PATH)
        main_cfg = routing.get("main_agent", {})
        provider_name = main_cfg.get("provider", "deepseek")
        if provider_name not in providers:
            raise RuntimeError(f"Provider '{provider_name}' not found in model config")
        app.state.model_provider = OpenAICompatProvider(providers[provider_name])

        # Memory Provider
        app.state.memory_provider = SqliteMemoryProvider(_db_url)
        await app.state.memory_provider.initialize()

    # Routes
    app.include_router(chat_router)
    app.include_router(session_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
