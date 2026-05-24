import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


@pytest.mark.asyncio
async def test_engine_creates_tables(raw_db_path):
    from src.db.engine import create_engine, async_session_factory

    engine = create_engine(f"sqlite+aiosqlite:///{raw_db_path}")
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_session_model_columns(raw_db_path):
    from src.db.engine import create_engine, init_db
    from src.db.models import Session

    engine = create_engine(f"sqlite+aiosqlite:///{raw_db_path}")
    await init_db(engine)

    async with engine.connect() as conn:
        result = await conn.execute(text("PRAGMA table_info(sessions)"))
        columns = {row[1] for row in result.fetchall()}
        assert columns >= {"id", "user_id", "title", "status", "message_count", "created_at", "updated_at"}
