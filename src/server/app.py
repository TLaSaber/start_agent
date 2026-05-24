"""
PyAgent 主应用 —— FastAPI 应用的创建与生命周期管理。

本模块是 PyAgent 服务器的入口点，负责：
    1. 创建和配置 FastAPI 应用实例；
    2. 在启动时初始化所有组件（数据库、工具、模型、记忆）；
    3. 注册所有路由；
    4. 提供 /health 健康检查端点。

FastAPI 应用的生命周期：
    启动（startup）→ 运行（serving）→ 关闭（shutdown）
    当前只实现了 startup 事件，所有初始化逻辑在应用首次接收
    请求前完成。
"""

from pathlib import Path
from fastapi import FastAPI

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
from langgraph.checkpoint.sqlite import SqliteSaver
from config.settings import DB_PATH, MODEL_CONFIG_PATH, SKILLS_CONFIG_PATH


def create_app(db_url: str | None = None):
    """创建并配置 PyAgent FastAPI 应用。

    本函数是应用的工厂方法，每次调用都会创建一个全新的应用实例，
    并注册 startup 事件处理函数。

    FastAPI 的 app.state 机制：
        FastAPI 提供了 app.state 对象，用于存储应用级别的全局状态。
        在 startup 事件中初始化的各个组件都被挂载到 app.state 上，
        后续路由处理器通过 request.app.state 访问这些组件。

    参数：
        db_url: 可选的数据库连接 URL。如不指定，则使用配置文件中
                DB_PATH 构建的默认 SQLite URL。
                URL 格式：sqlite+aiosqlite:///路径
                其中 aiosqlite 是 SQLite 的异步 Python 驱动。

    返回：
        配置完成的 FastAPI 应用实例。
    """
    app = FastAPI(title="PyAgent", version="0.1.0")
    # 如果未指定数据库 URL，使用默认的 SQLite 文件路径
    _db_url = db_url or f"sqlite+aiosqlite:///{DB_PATH}"

    @app.on_event("startup")
    async def startup():
        """应用启动时的初始化事件。

        FastAPI 的 startup 事件在应用开始接收请求之前触发。
        本函数负责初始化全部后端组件，按照依赖顺序依次进行：

        1. 数据库引擎和表结构（其他组件可能依赖数据库）
        2. LangGraph Checkpoint Saver（会话状态持久化）
        3. 工具注册表（向 Agent 注册所有可用工具）
        4. 技能注册表（从 YAML 加载技能定义）
        5. 模型提供者（根据路由配置选择主智能体模型）
        6. 记忆提供者（初始化记忆存储表）
        """

        # === 1. 数据库初始化 ===
        # 创建 SQLAlchemy 异步引擎和会话工厂
        engine = create_engine(_db_url)
        # 创建所有 ORM 模型对应的表（sessions 表等）
        await init_db(engine)

        # === 2. LangGraph Checkpoint Saver ===
        # Checkpoint（检查点）是 LangGraph 的会话持久化机制：
        # Agent 的每一步执行状态都会被保存到数据库，使得：
        #   - 会话可以随时中断和恢复
        #   - 支持人类介入（human-in-the-loop）审批流程
        # SqliteSaver 会自动创建所需的 checkpoint 相关表
        app.state.checkpoint_saver = SqliteSaver.from_conn_string(_db_url)

        # === 3. 工具注册表 ===
        # 注册 Agent 可以调用的所有内置工具。
        # 工具是 Agent 与外部世界交互的接口——Agent 本身只是一个
        # 语言模型，必须通过工具才能执行实际操作（读文件、写文件、
        # 执行命令等）。
        app.state.tool_registry = ToolRegistry()
        app.state.tool_registry.register(ReadFileTool())
        app.state.tool_registry.register(WriteFileTool())
        app.state.tool_registry.register(ListDirTool())
        app.state.tool_registry.register(SearchFileTool())
        app.state.tool_registry.register(GrepContentTool())
        app.state.tool_registry.register(ExecShellTool())
        app.state.tool_registry.register(HttpRequestTool())
        app.state.tool_registry.register(DbQueryTool())

        # === 4. 技能注册表 ===
        # 从 YAML 文件加载技能定义（SkillDefinition）。
        # 如果技能配置文件不存在，则使用空的注册表（不使用技能系统）。
        if Path(SKILLS_CONFIG_PATH).exists():
            app.state.skill_registry = SkillRegistry.load_from_yaml(
                str(SKILLS_CONFIG_PATH)
            )
        else:
            app.state.skill_registry = SkillRegistry()

        # === 5. 模型提供者 ===
        # 先加载所有模型提供商的连接配置，
        # 然后根据 routing 配置决定主智能体使用哪个提供商。
        #
        # routing 配置示例：
        #   routing:
        #     main_agent:
        #       provider: deepseek      # 主智能体使用 DeepSeek
        #     expert_agent:
        #       provider: openai         # 专家智能体使用 OpenAI
        providers = load_model_config(MODEL_CONFIG_PATH)
        routing = get_routing_config(MODEL_CONFIG_PATH)
        main_cfg = routing.get("main_agent", {})
        provider_name = main_cfg.get("provider", "deepseek")
        if provider_name not in providers:
            raise RuntimeError(
                f"Provider '{provider_name}' not found in model config"
            )
        app.state.model_provider = OpenAICompatProvider(providers[provider_name])

        # === 6. 记忆提供者 ===
        # 初始化 SQLite 记忆存储，自动创建 memories 表
        app.state.memory_provider = SqliteMemoryProvider(_db_url)
        await app.state.memory_provider.initialize()

    # === 路由注册 ===
    # 注册各个 API 路由，FastAPI 会自动生成 OpenAPI 文档
    app.include_router(chat_router)    # POST /chat — 对话端点
    app.include_router(session_router) # /session/* — 会话 CRUD

    # === 健康检查端点 ===
    @app.get("/health")
    async def health():
        """健康检查接口。

        返回 {"status": "ok"} 表示应用正常运行。
        可用于负载均衡器的健康探测或 Docker 的 HEALTHCHECK 指令。
        """
        return {"status": "ok"}

    return app


# 模块级别的应用实例，供 uvicorn 等 ASGI 服务器直接引用启动
# 启动命令示例：uvicorn src.server.app:app --host 0.0.0.0 --port 8000
app = create_app()
