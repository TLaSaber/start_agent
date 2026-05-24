"""
数据库引擎模块 —— SQLAlchemy 异步引擎的创建与管理。

本模块负责创建和管理 SQLAlchemy 异步数据库引擎及会话工厂，
使用模块级全局变量实现单例模式（一个进程中只创建一个引擎实例）。

为什么使用异步引擎？
    PyAgent 基于 FastAPI（异步框架），所有请求处理都是异步的。
    使用异步数据库引擎（create_async_engine）可以避免数据库操作
    阻塞事件循环，确保在高并发场景下服务器仍能保持响应。

    对应的，需要使用 aiosqlite 作为 SQLite 的异步驱动程序，
    连接 URL 格式为：sqlite+aiosqlite:///path/to/db.sqlite
"""

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)

# 模块级全局变量，保存引擎和会话工厂的单例
_engine = None
_async_session_factory = None


def create_engine(db_url: str):
    """创建 SQLAlchemy 异步引擎和会话工厂（单例模式）。

    本函数使用全局变量保存引擎实例，确保在同一个进程中只创建
    一次数据库连接池。后续调用直接返回已存在的实例。

    SQLAlchemy 异步组件说明：
        - create_async_engine: 创建异步引擎，管理数据库连接池。
          连接池使得多个请求可以复用已建立的数据库连接，避免
          每次操作都创建新连接的开销。
          echo=False 表示不打印 SQL 语句（生产环境关闭调试日志）。

        - async_sessionmaker: 创建异步会话工厂。
          每次需要操作数据库时，从工厂获取一个新的 AsyncSession 实例。
          expire_on_commit=False 表示提交事务后，对象属性不会过期，
          避免了在事务外访问对象属性时的 DetachedInstanceError。

    参数：
        db_url: 数据库连接 URL。
                例如 "sqlite+aiosqlite:///./pyagent.db"。

    返回：
        创建的 AsyncEngine 实例。
    """
    global _engine, _async_session_factory

    # 如果引擎已存在则直接返回（单例模式）
    if _engine is not None:
        return _engine

    _engine = create_async_engine(db_url, echo=False)
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return _engine


def get_engine():
    """获取已创建的数据库引擎。

    在调用 create_engine() 之后使用本方法获取引擎实例。
    如果在未初始化的情况下调用，会抛出 RuntimeError。

    返回：
        已初始化的 AsyncEngine 实例。

    异常：
        RuntimeError: 引擎未初始化。
    """
    if _engine is None:
        raise RuntimeError(
            "Database engine not initialized. Call create_engine() first."
        )
    return _engine


def async_session_factory():
    """获取一个新的异步数据库会话。

    每次调用都会返回一个新的 AsyncSession 实例，应当使用
    async with 语句管理其生命周期，确保使用后正确关闭。

    使用示例：
        async with async_session_factory() as session:
            result = await session.execute(text("SELECT ..."))
            await session.commit()

    返回：
        一个新的 AsyncSession 实例。

    异常：
        RuntimeError: 会话工厂未初始化（create_engine() 未调用）。
    """
    if _async_session_factory is None:
        raise RuntimeError(
            "Session factory not initialized. Call create_engine() first."
        )
    return _async_session_factory()


async def init_db(engine=None):
    """初始化数据库表结构。

    调用 SQLAlchemy ORM 的 Base.metadata.create_all 方法，
    根据 ORM 模型定义自动创建所有尚未存在的数据库表。

    关于 create_all 的安全特性：
        create_all 内部使用 IF NOT EXISTS 语义，这意味着：
            - 如果表已存在，不会重复创建；
            - 如果模型定义有变更（新增字段），不会修改已有表；
            - 因此可以安全地在应用每次启动时调用。
        注意：这不会执行数据库迁移。如果需要修改已有表结构，
        应使用 Alembic 等迁移工具。

    参数：
        engine: 可选的数据库引擎。如果不提供，使用 get_engine()
               获取已初始化的引擎。
    """
    from src.db.models import Base

    eng = engine or get_engine()
    async with eng.begin() as conn:
        # run_sync 用于在异步连接中执行同步的 SQLAlchemy 操作
        await conn.run_sync(Base.metadata.create_all)
