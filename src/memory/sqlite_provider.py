import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from src.memory.provider import MemoryProvider
from src.memory.models import MemoryEntry


class SqliteMemoryProvider(MemoryProvider):
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url, echo=False)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self):
        async with self._engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    source TEXT NOT NULL DEFAULT 'auto_archive',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_days INTEGER
                )
            """))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content)"
            ))

    async def archive(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        source: str = "auto_archive",
        session_id: str | None = None,
        ttl_days: int | None = None,
    ) -> str:
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO memories (id, user_id, session_id, content, category, source, created_at, ttl_days)
                    VALUES (:id, :user_id, :session_id, :content, :category, :source, :created_at, :ttl_days)
                """),
                {
                    "id": memory_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "content": content,
                    "category": category,
                    "source": source,
                    "created_at": now,
                    "ttl_days": ttl_days,
                },
            )
            await session.commit()
        return memory_id

    async def recall(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[MemoryEntry]:
        keywords = query.strip().split()
        conditions = []
        params = {"top_k": top_k}

        if user_id:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id

        like_clauses = []
        for i, kw in enumerate(keywords):
            like_clauses.append(f"content LIKE :kw{i}")
            params[f"kw{i}"] = f"%{kw}%"

        conditions.append(f"({' OR '.join(like_clauses)})")
        where = " AND ".join(conditions)

        sql = f"""
            SELECT id, user_id, session_id, content, category, source, created_at, ttl_days
            FROM memories
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT :top_k
        """

        async with self._session_factory() as session:
            result = await session.execute(text(sql), params)
            rows = result.fetchall()

        return [
            MemoryEntry(
                id=row[0], user_id=row[1], session_id=row[2],
                content=row[3], category=row[4], source=row[5],
                created_at=str(row[6]) if row[6] else None, ttl_days=row[7],
            )
            for row in rows
        ]

    async def delete(self, memory_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                text("DELETE FROM memories WHERE id = :id"),
                {"id": memory_id},
            )
            await session.commit()
            return result.rowcount > 0

    async def list_by_user(self, user_id: str, limit: int = 50) -> list[MemoryEntry]:
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT id, user_id, session_id, content, category, source, created_at, ttl_days
                    FROM memories
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"user_id": user_id, "limit": limit},
            )
            rows = result.fetchall()

        return [
            MemoryEntry(
                id=row[0], user_id=row[1], session_id=row[2],
                content=row[3], category=row[4], source=row[5],
                created_at=str(row[6]) if row[6] else None, ttl_days=row[7],
            )
            for row in rows
        ]
