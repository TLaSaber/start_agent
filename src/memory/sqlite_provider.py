"""
SQLite 记忆提供者实现 —— 基于 SQLite 数据库的持久化记忆存储。

本模块实现了 MemoryProvider 抽象接口，使用 SQLite 作为后端存储。
核心功能是通过 SQL 的 LIKE 运算符实现简单的关键词匹配召回。

为什么先使用 SQLite + LIKE 而不是向量数据库？
    1. 零依赖：Python 内置支持，无需额外安装服务；
    2. 足够简单：对于关键词级别的匹配场景，LIKE 查询完全够用；
    3. 渐进式设计：未来可以无缝替换为向量相似度搜索实现；
    4. 文件级存储：memories.db 就是一个文件，备份和迁移都很方便。

当前实现的限制：
    - LIKE 匹配是精确的关键词包含匹配，不理解语义；
    - 不支持同义词、近义词匹配；
    - 对于长文本，关键词拆分的策略过于简单（仅按空格切分）。
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from src.memory.provider import MemoryProvider
from src.memory.models import MemoryEntry


class SqliteMemoryProvider(MemoryProvider):
    """基于 SQLite 的持久化记忆提供者。

    使用 SQLAlchemy 异步引擎操作 SQLite 数据库，通过 LIKE 运算符
    实现关键词级别的记忆检索。

    实例变量：
        _engine: SQLAlchemy 异步引擎实例，管理数据库连接池。
        _session_factory: 异步会话工厂，用于创建数据库会话。
                          expire_on_commit=False 表示提交后对象不过期。
    """

    def __init__(self, db_url: str):
        """初始化 SQLite 记忆提供者。

        参数：
            db_url: 数据库连接 URL，例如 "sqlite+aiosqlite:///./memories.db"。
                    sqlite+aiosqlite 中的 aiosqlite 是异步 SQLite 驱动。
        """
        # echo=False 表示不打印 SQL 语句（生产环境关闭调试日志）
        self._engine = create_async_engine(db_url, echo=False)
        # expire_on_commit=False：提交后保留对象属性，避免 DetachedInstanceError
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """初始化数据库表结构和索引。

        在应用启动时调用，负责：
            1. 创建 memories 表（如果不存在）；
            2. 创建 user_id 上的索引，加速按用户查询；
            3. 创建 content 上的索引，加速 LIKE 全文匹配。

        注意：INDEX 在 SQLite 中对 LIKE 查询的加速效果有限，
        但对于前缀匹配仍有帮助。未来如果性能成为瓶颈，可以考虑
        使用 FTS5（全文搜索引擎）。
        """
        async with self._engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,          -- UUID 主键
                    user_id TEXT NOT NULL,        -- 用户 ID
                    session_id TEXT,              -- 可选的会话 ID
                    content TEXT NOT NULL,        -- 记忆内容
                    category TEXT NOT NULL DEFAULT 'fact',     -- 类别
                    source TEXT NOT NULL DEFAULT 'auto_archive', -- 来源
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 创建时间
                    ttl_days INTEGER              -- 过期天数（NULL=永不过期）
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
        """归档一条新记忆到 SQLite 数据库。

        实现步骤：
            1. 使用 uuid4() 生成全局唯一的记忆 ID；
            2. 获取当前 UTC 时间作为创建时间戳；
            3. 执行 INSERT 语句将记录写入数据库；
            4. 提交事务，返回记忆 ID。

        参数：
            同 MemoryProvider.archive。

        返回：
            新生成的 UUID 字符串，作为记忆的唯一标识。
        """
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

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """基于关键词匹配召回相关记忆。

        SQL 查询构建逻辑详解：

        1. 关键词提取：
           用户输入的 query 按空白字符切分为关键词列表。
           例如 "Python 异步编程" → ["Python", "异步编程"]

        2. LIKE 子句构建：
           对每个关键词 k，生成 "content LIKE :kwN" 子句，
           其中参数 :kwN 的值为 "%k%"（% 是 SQL 的通配符，
           表示匹配任意字符序列）。
           例如：content LIKE '%Python%' OR content LIKE '%异步编程%'

        3. WHERE 条件组合：
           - 如果指定了 user_id，先加上 user_id = :user_id 条件；
           - 所有 LIKE 子句之间用 OR 连接（匹配任一关键词即可）；
           - 最后用 AND 将 user_id 条件和 LIKE 条件组合。

        4. 排序和限制：
           按 created_at DESC 倒序排列（最新的在前），
           用 LIMIT :top_k 限制返回数量。

        示例：
            query="API 设计", user_id="u1" 生成的 SQL：
            ```sql
            SELECT ... FROM memories
            WHERE user_id = :user_id
              AND (content LIKE :kw0 OR content LIKE :kw1)
            ORDER BY created_at DESC
            LIMIT :top_k
            ```
            参数：user_id="u1", kw0="%API%", kw1="%设计%", top_k=5

        参数：
            同 MemoryProvider.recall。

        返回：
            匹配到的记忆条目列表。
        """
        # 按空白字符拆分查询文本为关键词
        keywords = query.strip().split()
        conditions = []
        params = {"top_k": top_k}

        # 如果指定了 user_id，添加用户过滤条件
        if user_id:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id

        # 为每个关键词生成 content LIKE '%keyword%' 子句
        like_clauses = []
        for i, kw in enumerate(keywords):
            like_clauses.append(f"content LIKE :kw{i}")
            params[f"kw{i}"] = f"%{kw}%"

        # 所有 LIKE 子句用 OR 连接：匹配任何一个关键词即可
        conditions.append(f"({' OR '.join(like_clauses)})")
        # 最终用 AND 组合所有条件
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

        # 将数据库行转换为 MemoryEntry 对象列表
        return [
            MemoryEntry(
                id=row[0], user_id=row[1], session_id=row[2],
                content=row[3], category=row[4], source=row[5],
                created_at=str(row[6]) if row[6] else None,
                ttl_days=row[7],
            )
            for row in rows
        ]

    async def delete(self, memory_id: str) -> bool:
        """从数据库中删除一条记忆。

        参数：
            memory_id: 要删除的记忆 ID。

        返回：
            True 表示成功删除了一条记录；
            False 表示没有找到该 ID 对应的记录（rowcount == 0）。
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text("DELETE FROM memories WHERE id = :id"),
                {"id": memory_id},
            )
            await session.commit()
            # rowcount 表示受影响的行数
            return result.rowcount > 0

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """列出指定用户的所有记忆，按时间倒序排列。

        参数：
            user_id: 用户标识。
            limit: 最大返回条数，默认 50。

        返回：
            MemoryEntry 列表。
        """
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
                created_at=str(row[6]) if row[6] else None,
                ttl_days=row[7],
            )
            for row in rows
        ]
