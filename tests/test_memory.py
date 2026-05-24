import pytest
from datetime import datetime, timezone


def test_memory_entry_creation():
    from src.memory.models import MemoryEntry

    entry = MemoryEntry(
        id="mem-1",
        user_id="user-1",
        content="用户偏好使用 tab 缩进",
        category="preference",
        source="user_command",
    )
    assert entry.category == "preference"
    assert entry.source == "user_command"
    assert entry.ttl_days is None  # permanent


def test_memory_provider_is_abstract():
    from src.memory.provider import MemoryProvider
    with pytest.raises(TypeError):
        MemoryProvider()


@pytest.mark.asyncio
async def test_sqlite_provider_archive_and_recall(raw_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{raw_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    entry_id = await provider.archive(
        user_id="user-1",
        session_id="sess-1",
        content="用户偏好使用 tab 缩进",
        category="preference",
        source="user_command",
    )
    assert entry_id is not None

    results = await provider.recall("tab 缩进", top_k=5, user_id="user-1")
    assert len(results) >= 1
    assert results[0].content == "用户偏好使用 tab 缩进"


@pytest.mark.asyncio
async def test_sqlite_provider_recall_respects_user_id(raw_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{raw_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    await provider.archive(user_id="user-1", content="user-1 的记忆", category="fact", source="auto_archive")
    await provider.archive(user_id="user-2", content="user-2 的记忆", category="fact", source="auto_archive")

    results = await provider.recall("记忆", top_k=10, user_id="user-1")
    assert all(r.user_id == "user-1" for r in results)


@pytest.mark.asyncio
async def test_sqlite_provider_delete(raw_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{raw_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    entry_id = await provider.archive(user_id="user-1", content="临时记忆", category="fact", source="auto_archive")
    deleted = await provider.delete(entry_id)
    assert deleted is True

    results = await provider.recall("临时记忆", top_k=5, user_id="user-1")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_sqlite_provider_list_by_user(raw_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{raw_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    await provider.archive(user_id="user-1", content="记忆 A", category="fact", source="auto_archive")
    await provider.archive(user_id="user-1", content="记忆 B", category="knowledge", source="auto_archive")

    results = await provider.list_by_user("user-1", limit=50)
    assert len(results) == 2
