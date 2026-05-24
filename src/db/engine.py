from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

_engine = None
_async_session_factory = None


def create_engine(db_url: str):
    global _engine, _async_session_factory
    _engine = create_async_engine(db_url, echo=False)
    _async_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine


def get_engine():
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call create_engine() first.")
    return _engine


def async_session_factory():
    if _async_session_factory is None:
        raise RuntimeError("Session factory not initialized. Call create_engine() first.")
    return _async_session_factory()


async def init_db(engine=None):
    from src.db.models import Base
    eng = engine or get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
