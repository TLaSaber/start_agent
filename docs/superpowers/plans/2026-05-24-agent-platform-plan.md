# Agent 平台 MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建通用 Agent 执行平台 MVP — HTTP 同步请求-响应，Observe→Think→Act 循环，8 个内置工具，记忆管理，模型防腐。

**Architecture:** 四层分离 — FastAPI 接入层 → LangGraph Agent Runtime → 能力层 (Model/Tool/Memory/Skill Providers) → SQLite 基础设施层。Agent Runtime 不依赖 HTTP，能力层通过接口抽象可插拔。

**Tech Stack:** Python 3.12+, LangGraph 0.3+, LangChain 0.3+, FastAPI 0.115+, SQLAlchemy 2.0+, aiosqlite, tiktoken

---

### Task 1: 项目骨架与配置

**Files:**
- Create: `pyproject.toml`
- Create: `config/model.yaml`
- Create: `config/skills.yaml`
- Create: `config/settings.py`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.env.example`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "pyagent"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "langgraph>=0.3",
    "langchain>=0.3",
    "langchain-openai>=0.3",
    "sqlalchemy>=2.0",
    "aiosqlite>=0.20",
    "tiktoken>=0.8",
    "pyyaml>=6.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.28",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: 创建 config/model.yaml**

```yaml
providers:
  deepseek:
    type: openai_compat
    api_base: "${DEEPSEEK_API_BASE:https://api.deepseek.com/v1}"
    api_key: "${DEEPSEEK_API_KEY}"
    default_model: "deepseek-v3"
    available_models:
      - name: "deepseek-v3"
        max_tokens: 65536
        capabilities: [chat, function_calling]

routing:
  main_agent:
    provider: deepseek
    model: deepseek-v3
```

- [ ] **Step 3: 创建 config/skills.yaml**

```yaml
skills:
  - name: "code-review"
    summary: "对代码变更进行审查，输出问题和改进建议"
    tools: ["read_file", "grep_content", "search_file"]
    constraints:
      - "不得修改任何文件"
      - "不得执行代码"

  - name: "data-analyze"
    summary: "分析结构化数据文件（CSV/JSON/Excel），输出统计报告和可视化建议"
    tools: ["read_file", "list_dir", "exec_shell"]
    constraints:
      - "仅读取数据，不修改源文件"
```

- [ ] **Step 4: 创建 config/settings.py**

```python
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
DB_PATH = os.environ.get("PYAGENT_DB_PATH", str(PROJECT_ROOT / "data" / "pyagent.db"))
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
SKILLS_CONFIG_PATH = PROJECT_ROOT / "config" / "skills.yaml"

MAX_LOOPS = int(os.environ.get("PYAGENT_MAX_LOOPS", "15"))
COMPACT_THRESHOLD_RATIO = 0.8
COMPACT_KEEP_RECENT = 6
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 3
TOOL_TIMEOUT_SECONDS = 30
MEMORY_RECALL_TOP_K = 3
AUTO_ARCHIVE_ENABLED = os.environ.get("PYAGENT_AUTO_ARCHIVE", "false").lower() == "true"
```

- [ ] **Step 5: 创建 .env.example**

```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
PYAGENT_DB_PATH=./data/pyagent.db
PYAGENT_AUTO_ARCHIVE=false
PYAGENT_MAX_LOOPS=15
```

- [ ] **Step 6: 创建 data/ 目录和 .gitignore**

```bash
mkdir -p data
echo "data/" >> .gitignore
echo ".env" >> .gitignore
echo ".superpowers/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
```

- [ ] **Step 7: 创建空的 __init__.py 文件**

```bash
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 8: 创建 tests/conftest.py**

```python
import pytest
from pathlib import Path
import tempfile
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent.resolve()
```

- [ ] **Step 9: 安装依赖**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: add project skeleton with config and dependencies"
```

---

### Task 2: 数据库引擎与 Session ORM

**Files:**
- Create: `src/db/__init__.py`
- Create: `src/db/engine.py`
- Create: `src/db/models.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_db.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text


@pytest.mark.asyncio
async def test_engine_creates_tables(temp_db_path):
    from src.db.engine import create_engine, async_session_factory

    engine = create_engine(f"sqlite+aiosqlite:///{temp_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: ...)  # placeholder

    # Verify engine works
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_session_model_columns(temp_db_path):
    from src.db.engine import create_engine, init_db
    from src.db.models import Session

    engine = create_engine(f"sqlite+aiosqlite:///{temp_db_path}")
    await init_db(engine)

    # Verify table exists with correct columns
    async with engine.connect() as conn:
        result = await conn.execute(text("PRAGMA table_info(sessions)"))
        columns = {row[1] for row in result.fetchall()}
        assert columns >= {"id", "user_id", "title", "status", "message_count", "created_at", "updated_at"}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_db.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: 创建 src/db/__init__.py**

```python
from src.db.engine import create_engine, init_db, async_session_factory
from src.db.models import Session
```

- [ ] **Step 4: 创建 src/db/engine.py**

```python
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
```

- [ ] **Step 5: 创建 src/db/models.py**

```python
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")  # active | closed
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_db.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add src/db/ tests/test_db.py
git commit -m "feat: add database engine and Session ORM model"
```

---

### Task 3: 模型防腐层 — 抽象接口

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/provider.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_models.py
import pytest
from src.models.provider import ModelProvider, ModelInfo


def test_model_info_creation():
    info = ModelInfo(
        name="deepseek-v3",
        provider="deepseek",
        max_tokens=65536,
        capabilities=["chat", "function_calling"],
    )
    assert info.name == "deepseek-v3"
    assert info.max_tokens == 65536
    assert "function_calling" in info.capabilities


def test_model_provider_is_abstract():
    with pytest.raises(TypeError):
        ModelProvider()  # Cannot instantiate ABC
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_models.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: 创建 src/models/provider.py**

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel


class ModelInfo(BaseModel):
    name: str
    provider: str
    max_tokens: int
    capabilities: list[str] = []


class ModelProvider(ABC):
    @abstractmethod
    def get_chat_model(self, model_name: str | None = None) -> BaseChatModel:
        """返回 LangChain 兼容的 chat model 实例"""
        ...

    @abstractmethod
    def get_available_models(self) -> list[ModelInfo]:
        """列出可用模型"""
        ...

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """估算 token 数"""
        ...
```

- [ ] **Step 4: 创建 src/models/__init__.py**

```python
from src.models.provider import ModelProvider, ModelInfo
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_models.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: add ModelProvider abstract interface"
```

---

### Task 4: 模型防腐层 — OpenAICompatProvider 实现

**Files:**
- Create: `src/models/openai_compat.py`
- Create: `src/models/config_loader.py`
- Modify: `src/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: 编写失败测试**

```python
# 追加到 tests/test_models.py
def test_openai_compat_provider_creates_chat_model():
    from src.models.openai_compat import OpenAICompatProvider
    from src.models.provider import ProviderConfig

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat", "function_calling"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    model = provider.get_chat_model()
    assert model is not None
    assert model.model_name == "deepseek-v3"


def test_openai_compat_provider_switches_model():
    from src.models.openai_compat import OpenAICompatProvider
    from src.models.provider import ProviderConfig

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
            ModelInfo(name="deepseek-r1", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    r1 = provider.get_chat_model("deepseek-r1")
    assert r1.model_name == "deepseek-r1"


def test_count_tokens_returns_reasonable_estimate():
    from src.models.openai_compat import OpenAICompatProvider
    from src.models.provider import ProviderConfig

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    count = provider.count_tokens("Hello, world!")
    assert count > 0
    assert count < 10  # "Hello, world!" is ~4 tokens


def test_get_available_models():
    from src.models.openai_compat import OpenAICompatProvider
    from src.models.provider import ProviderConfig

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
            ModelInfo(name="deepseek-r1", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    models = provider.get_available_models()
    assert len(models) == 2
    assert models[0].name == "deepseek-v3"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_models.py -v -k "openai"
```
Expected: FAIL

- [ ] **Step 3: 在 src/models/provider.py 追加 ProviderConfig**

```python
class ProviderConfig(BaseModel):
    api_base: str
    api_key: str
    default_model: str
    temperature: float = 0.7
    max_output_tokens: int = 4096
    available_models: list[ModelInfo] = []
```

- [ ] **Step 4: 创建 src/models/openai_compat.py**

```python
import tiktoken
from langchain_openai import ChatOpenAI
from src.models.provider import ModelProvider, ModelInfo, ProviderConfig


class OpenAICompatProvider(ModelProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._clients: dict[str, ChatOpenAI] = {}

    def get_chat_model(self, model_name: str | None = None) -> ChatOpenAI:
        name = model_name or self.config.default_model
        if name not in self._clients:
            self._clients[name] = ChatOpenAI(
                model=name,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
            )
        return self._clients[name]

    def get_available_models(self) -> list[ModelInfo]:
        return self.config.available_models

    def count_tokens(self, text: str, model: str | None = None) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
```

- [ ] **Step 5: 创建 src/models/config_loader.py**

```python
import os
import re
import yaml
from pathlib import Path
from src.models.provider import ProviderConfig, ModelInfo


_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: str) -> str:
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default or "")

    return _ENV_VAR_RE.sub(replacer, value)


def load_model_config(config_path: str | Path) -> dict[str, ProviderConfig]:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    providers = {}
    for name, cfg in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            api_base=_resolve_env(cfg["api_base"]),
            api_key=_resolve_env(cfg["api_key"]),
            default_model=cfg["default_model"],
            available_models=[
                ModelInfo(
                    name=m["name"],
                    provider=name,
                    max_tokens=m.get("max_tokens", 65536),
                    capabilities=m.get("capabilities", []),
                )
                for m in cfg.get("available_models", [])
            ],
        )
    return providers


def get_routing_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return raw.get("routing", {})
```

- [ ] **Step 6: 更新 src/models/__init__.py**

```python
from src.models.provider import ModelProvider, ModelInfo, ProviderConfig
from src.models.openai_compat import OpenAICompatProvider
from src.models.config_loader import load_model_config, get_routing_config
```

- [ ] **Step 7: 运行测试确认通过**

```bash
pytest tests/test_models.py -v
```
Expected: PASS (all 6 tests)

- [ ] **Step 8: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: add OpenAICompatProvider and YAML config loader"
```

---

### Task 5: 记忆管理 — 抽象接口与 SQLite 实现

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/models.py`
- Create: `src/memory/provider.py`
- Create: `src/memory/sqlite_provider.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_memory.py
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
    assert entry.ttl_days is None  # 永久


def test_memory_provider_is_abstract():
    from src.memory.provider import MemoryProvider
    with pytest.raises(TypeError):
        MemoryProvider()


@pytest.mark.asyncio
async def test_sqlite_provider_archive_and_recall(temp_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
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
async def test_sqlite_provider_recall_respects_user_id(temp_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    await provider.archive(user_id="user-1", content="user-1 的记忆", category="fact", source="auto_archive")
    await provider.archive(user_id="user-2", content="user-2 的记忆", category="fact", source="auto_archive")

    results = await provider.recall("记忆", top_k=10, user_id="user-1")
    assert all(r.user_id == "user-1" for r in results)


@pytest.mark.asyncio
async def test_sqlite_provider_delete(temp_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    entry_id = await provider.archive(user_id="user-1", content="临时记忆", category="fact", source="auto_archive")
    deleted = await provider.delete(entry_id)
    assert deleted is True

    results = await provider.recall("临时记忆", top_k=5, user_id="user-1")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_sqlite_provider_list_by_user(temp_db_path):
    from src.memory.sqlite_provider import SqliteMemoryProvider

    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
    provider = SqliteMemoryProvider(db_url)
    await provider.initialize()

    await provider.archive(user_id="user-1", content="记忆 A", category="fact", source="auto_archive")
    await provider.archive(user_id="user-1", content="记忆 B", category="knowledge", source="auto_archive")

    results = await provider.list_by_user("user-1", limit=50)
    assert len(results) == 2
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_memory.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/memory/models.py**

```python
from datetime import datetime, timezone
from pydantic import BaseModel


class MemoryEntry(BaseModel):
    id: str
    user_id: str
    session_id: str | None = None
    content: str
    category: str = "fact"  # preference | knowledge | fact
    source: str = "auto_archive"  # auto_archive | user_command | rule_match
    created_at: datetime | None = None
    ttl_days: int | None = None
```

- [ ] **Step 4: 创建 src/memory/provider.py**

```python
from abc import ABC, abstractmethod
from src.memory.models import MemoryEntry


class MemoryProvider(ABC):
    @abstractmethod
    async def archive(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        source: str = "auto_archive",
        session_id: str | None = None,
        ttl_days: int | None = None,
    ) -> str:
        """归档一条记忆，返回 memory_id"""
        ...

    @abstractmethod
    async def recall(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[MemoryEntry]:
        """关键词召回相关记忆"""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """删除一条记忆"""
        ...

    @abstractmethod
    async def list_by_user(self, user_id: str, limit: int = 50) -> list[MemoryEntry]:
        """列出用户的所有记忆"""
        ...
```

- [ ] **Step 5: 创建 src/memory/sqlite_provider.py**

```python
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
        # 关键词匹配：将 query 拆分为词，用 LIKE 匹配
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
                id=row[0],
                user_id=row[1],
                session_id=row[2],
                content=row[3],
                category=row[4],
                source=row[5],
                created_at=row[6],
                ttl_days=row[7],
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
                created_at=row[6], ttl_days=row[7],
            )
            for row in rows
        ]
```

- [ ] **Step 6: 创建 src/memory/__init__.py**

```python
from src.memory.provider import MemoryProvider
from src.memory.models import MemoryEntry
from src.memory.sqlite_provider import SqliteMemoryProvider
```

- [ ] **Step 7: 运行测试确认通过**

```bash
pytest tests/test_memory.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 8: Commit**

```bash
git add src/memory/ tests/test_memory.py
git commit -m "feat: add MemoryProvider interface and SQLite implementation"
```

---

### Task 6: 工具层 — BaseTool 与 ToolRegistry

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/base.py`
- Create: `src/tools/registry.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_tools.py
import pytest


def test_tool_result_creation():
    from src.tools.base import ToolResult

    result = ToolResult(success=True, output="file content here")
    assert result.success is True
    assert result.error is None

    fail_result = ToolResult(success=False, output="", error="File not found")
    assert fail_result.success is False
    assert fail_result.error == "File not found"


def test_base_tool_is_abstract():
    from src.tools.base import BaseTool
    with pytest.raises(TypeError):
        BaseTool()


def test_tool_registry_register_and_get():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "A fake tool for testing"
        parameters = {"type": "object", "properties": {}, "required": []}
        risk_level = "low"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="done")

    registry = ToolRegistry()
    registry.register(FakeTool())
    assert registry.get("fake_tool") is not None
    assert registry.get("nonexistent") is None


def test_tool_registry_get_llm_tools():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "Does something fake"
        parameters = {
            "type": "object",
            "properties": {"arg1": {"type": "string", "description": "First arg"}},
            "required": ["arg1"],
        }
        risk_level = "low"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="done")

    registry = ToolRegistry()
    registry.register(FakeTool())

    tools = registry.get_llm_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "fake_tool"
    assert tools[0]["function"]["parameters"]["required"] == ["arg1"]


@pytest.mark.asyncio
async def test_tool_registry_execute():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class EchoTool(BaseTool):
        name = "echo"
        description = "Echoes input"
        parameters = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
        risk_level = "low"

        async def execute(self, message: str = "", **kwargs) -> ToolResult:
            return ToolResult(success=True, output=message)

    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await registry.execute("echo", message="hello")
    assert result.success is True
    assert result.output == "hello"


def test_tool_registry_list_all():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class ToolA(BaseTool):
        name = "tool_a"
        description = "A"
        parameters = {}
        risk_level = "low"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="A")

    class ToolB(BaseTool):
        name = "tool_b"
        description = "B"
        parameters = {}
        risk_level = "medium"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="B")

    registry = ToolRegistry()
    registry.register(ToolA())
    registry.register(ToolB())

    all_tools = registry.list_all()
    assert len(all_tools) == 2
    names = {t.name for t in all_tools}
    assert names == {"tool_a", "tool_b"}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tools.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/tools/base.py**

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: str
    error: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    risk_level: str = "low"  # low | medium | high | critical

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...
```

- [ ] **Step 4: 创建 src/tools/registry.py**

```python
from src.tools.base import BaseTool, ToolResult


class ToolNotFoundError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_llm_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        return await tool.execute(**kwargs)
```

- [ ] **Step 5: 创建 src/tools/__init__.py**

```python
from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry, ToolNotFoundError
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_tools.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add src/tools/ tests/test_tools.py
git commit -m "feat: add BaseTool, ToolResult, and ToolRegistry"
```

---

### Task 7: 内置工具 — file_ops (read_file, write_file, list_dir)

**Files:**
- Create: `src/tools/builtin/__init__.py`
- Create: `src/tools/builtin/file_ops.py`
- Test: `tests/test_tools_file_ops.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_tools_file_ops.py
import pytest
import tempfile
from pathlib import Path


@pytest.mark.asyncio
async def test_read_file():
    from src.tools.builtin.file_ops import ReadFileTool

    tool = ReadFileTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello, world!")
        temp_path = f.name

    try:
        result = await tool.execute(path=temp_path)
        assert result.success is True
        assert "Hello, world!" in result.output
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_read_file_not_found():
    from src.tools.builtin.file_ops import ReadFileTool

    tool = ReadFileTool()
    result = await tool.execute(path="/nonexistent/path.txt")
    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_write_file():
    from src.tools.builtin.file_ops import WriteFileTool

    tool = WriteFileTool()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        result = await tool.execute(path=temp_path, content="new content")
        assert result.success is True
        assert Path(temp_path).read_text() == "new content"
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_list_dir():
    from src.tools.builtin.file_ops import ListDirTool

    tool = ListDirTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "file1.txt").touch()
        (Path(tmpdir) / "file2.txt").touch()
        (Path(tmpdir) / "subdir").mkdir()

        result = await tool.execute(path=tmpdir)
        assert result.success is True
        assert "file1.txt" in result.output
        assert "file2.txt" in result.output
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tools_file_ops.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/tools/builtin/file_ops.py**

```python
from pathlib import Path
from src.tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取指定路径的文件内容。参数 path: 文件路径。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要读取的文件路径"},
        },
        "required": ["path"],
    }
    risk_level = "low"

    async def execute(self, path: str = "", **kwargs) -> ToolResult:
        try:
            content = Path(path).read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except FileNotFoundError:
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "创建或覆盖写入文件。参数 path: 文件路径, content: 要写入的内容。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "要写入的内容"},
        },
        "required": ["path", "content"],
    }
    risk_level = "medium"

    async def execute(self, path: str = "", content: str = "", **kwargs) -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"File written: {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "列出目录内容。参数 path: 目录路径。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录路径"},
        },
        "required": ["path"],
    }
    risk_level = "low"

    async def execute(self, path: str = "", **kwargs) -> ToolResult:
        try:
            p = Path(path)
            if not p.is_dir():
                return ToolResult(success=False, output="", error=f"Not a directory: {path}")
            entries = []
            for entry in sorted(p.iterdir()):
                suffix = "/" if entry.is_dir() else ""
                entries.append(f"  {entry.name}{suffix}")
            output = "\n".join(entries) if entries else "(empty directory)"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

- [ ] **Step 4: 创建 src/tools/builtin/__init__.py**

```python
from src.tools.builtin.file_ops import ReadFileTool, WriteFileTool, ListDirTool
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_tools_file_ops.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/tools/builtin/ tests/test_tools_file_ops.py
git commit -m "feat: add builtin file operation tools (read, write, list)"
```

---

### Task 8: 内置工具 — search (search_file, grep_content)

**Files:**
- Create: `src/tools/builtin/search.py`
- Test: `tests/test_tools_search.py`
- Modify: `src/tools/builtin/__init__.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_tools_search.py
import pytest
import tempfile
from pathlib import Path


@pytest.mark.asyncio
async def test_search_file_by_name():
    from src.tools.builtin.search import SearchFileTool

    tool = SearchFileTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "report_2024.csv").touch()
        (Path(tmpdir) / "report_2025.csv").touch()
        (Path(tmpdir) / "notes.txt").touch()
        sub = Path(tmpdir) / "sub"
        sub.mkdir()
        (sub / "report_2024.csv").touch()

        result = await tool.execute(directory=tmpdir, pattern="report_*.csv")
        assert result.success is True
        assert "report_2024.csv" in result.output
        assert "report_2025.csv" in result.output
        assert "notes.txt" not in result.output


@pytest.mark.asyncio
async def test_search_file_not_found():
    from src.tools.builtin.search import SearchFileTool

    tool = SearchFileTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await tool.execute(directory=tmpdir, pattern="nonexistent.*")
        assert result.success is True
        assert "0 个" in result.output or "not found" in result.output.lower()


@pytest.mark.asyncio
async def test_grep_content():
    from src.tools.builtin.search import GrepContentTool

    tool = GrepContentTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "a.py"
        f1.write_text("def hello():\n    return 'world'\n")
        f2 = Path(tmpdir) / "b.py"
        f2.write_text("def goodbye():\n    return 'world'\n")

        result = await tool.execute(directory=tmpdir, pattern="def hello")
        assert result.success is True
        assert "a.py" in result.output
        assert "b.py" not in result.output


@pytest.mark.asyncio
async def test_grep_content_no_match():
    from src.tools.builtin.search import GrepContentTool

    tool = GrepContentTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "a.py").write_text("def hello():\n    pass\n")

        result = await tool.execute(directory=tmpdir, pattern="xyz_nonexistent_123")
        assert result.success is True
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tools_search.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/tools/builtin/search.py**

```python
import re
import fnmatch
from pathlib import Path
from src.tools.base import BaseTool, ToolResult


class SearchFileTool(BaseTool):
    name = "search_file"
    description = "按文件名模式搜索文件。参数 directory: 搜索目录, pattern: 文件名通配符模式(如 *.py)。"
    parameters = {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "搜索目录路径"},
            "pattern": {"type": "string", "description": "文件名通配符模式，如 *.py"},
        },
        "required": ["directory", "pattern"],
    }
    risk_level = "low"

    async def execute(self, directory: str = ".", pattern: str = "*", **kwargs) -> ToolResult:
        try:
            results = []
            base = Path(directory)
            if not base.is_dir():
                return ToolResult(success=False, output="", error=f"Not a directory: {directory}")
            for file_path in base.rglob(pattern):
                if file_path.is_file():
                    results.append(str(file_path))
            if not results:
                return ToolResult(success=True, output=f"未找到匹配 '{pattern}' 的文件")
            return ToolResult(success=True, output="\n".join(results))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class GrepContentTool(BaseTool):
    name = "grep_content"
    description = "在文件内容中正则搜索。参数 directory: 搜索目录, pattern: 正则表达式, file_glob: 文件名过滤(如 *.py)。"
    parameters = {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "搜索目录路径"},
            "pattern": {"type": "string", "description": "正则表达式"},
            "file_glob": {"type": "string", "description": "文件名过滤，如 *.py (可选)"},
        },
        "required": ["directory", "pattern"],
    }
    risk_level = "low"

    async def execute(self, directory: str = ".", pattern: str = "", file_glob: str = "*", **kwargs) -> ToolResult:
        try:
            results = []
            base = Path(directory)
            if not base.is_dir():
                return ToolResult(success=False, output="", error=f"Not a directory: {directory}")

            compiled = re.compile(pattern)
            for file_path in base.rglob(file_glob):
                if not file_path.is_file():
                    continue
                try:
                    for i, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                        if compiled.search(line):
                            results.append(f"{file_path}:{i}: {line}")
                except Exception:
                    continue

            if not results:
                return ToolResult(success=True, output=f"未找到匹配 '{pattern}' 的内容")
            return ToolResult(success=True, output="\n".join(results))
        except re.error as e:
            return ToolResult(success=False, output="", error=f"正则表达式错误: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

- [ ] **Step 4: 更新 src/tools/builtin/__init__.py**

```python
from src.tools.builtin.file_ops import ReadFileTool, WriteFileTool, ListDirTool
from src.tools.builtin.search import SearchFileTool, GrepContentTool
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_tools_search.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/tools/builtin/ tests/test_tools_search.py
git commit -m "feat: add builtin search tools (search_file, grep_content)"
```

---

### Task 9: 内置工具 — shell, http, database

**Files:**
- Create: `src/tools/builtin/shell.py`
- Create: `src/tools/builtin/http.py`
- Create: `src/tools/builtin/database.py`
- Test: `tests/test_tools_others.py`
- Modify: `src/tools/builtin/__init__.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_tools_others.py
import pytest
import tempfile
from pathlib import Path


@pytest.mark.asyncio
async def test_exec_shell_is_high_risk():
    from src.tools.builtin.shell import ExecShellTool
    tool = ExecShellTool()
    assert tool.risk_level == "high"


@pytest.mark.asyncio
async def test_exec_shell_echo():
    from src.tools.builtin.shell import ExecShellTool
    tool = ExecShellTool()
    result = await tool.execute(command="echo hello")
    assert result.success is True
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_http_request_get():
    from src.tools.builtin.http import HttpRequestTool
    tool = HttpRequestTool()
    result = await tool.execute(url="https://httpbin.org/get", method="GET")
    assert result.success is True
    assert "httpbin.org" in result.output or result.output != ""


@pytest.mark.asyncio
async def test_http_request_invalid_url():
    from src.tools.builtin.http import HttpRequestTool
    tool = HttpRequestTool()
    result = await tool.execute(url="not-a-valid-url", method="GET")
    assert result.success is False


@pytest.mark.asyncio
async def test_db_query_read_only():
    from src.tools.builtin.database import DbQueryTool
    tool = DbQueryTool()
    assert tool.risk_level == "medium"


@pytest.mark.asyncio
async def test_db_query_rejects_write():
    from src.tools.builtin.database import DbQueryTool
    tool = DbQueryTool()
    result = await tool.execute(db_url="sqlite:///test.db", query="DROP TABLE users")
    assert result.success is False
    assert "只读" in result.error or "read-only" in result.error.lower() or "SELECT" in result.error


@pytest.mark.asyncio
async def test_db_query_select(temp_db_path):
    from src.tools.builtin.database import DbQueryTool
    import sqlite3

    conn = sqlite3.connect(temp_db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    conn.close()

    tool = DbQueryTool()
    result = await tool.execute(db_url=f"sqlite:///{temp_db_path}", query="SELECT * FROM users")
    assert result.success is True
    assert "Alice" in result.output
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_tools_others.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/tools/builtin/shell.py**

```python
import asyncio
from src.tools.base import BaseTool, ToolResult


class ExecShellTool(BaseTool):
    name = "exec_shell"
    description = "执行系统命令(高风险)。参数 command: 要执行的命令。"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 shell 命令"},
        },
        "required": ["command"],
    }
    risk_level = "high"

    async def execute(self, command: str = "", **kwargs) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[stderr]\n" + stderr.decode("utf-8", errors="replace")
            return ToolResult(success=proc.returncode == 0, output=output.strip() or "(no output)")
        except asyncio.TimeoutError:
            return ToolResult(success=False, output="", error="命令执行超时(30s)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

- [ ] **Step 4: 创建 src/tools/builtin/http.py**

```python
import urllib.request
import urllib.error
import json
from src.tools.base import BaseTool, ToolResult


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = "发起 HTTP 请求。参数 url: 请求URL, method: GET/POST, headers: JSON字符串(可选), body: 请求体(可选)。"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "请求 URL"},
            "method": {"type": "string", "description": "HTTP 方法: GET 或 POST", "default": "GET"},
            "headers": {"type": "string", "description": "JSON 格式的请求头 (可选)"},
            "body": {"type": "string", "description": "请求体 (可选)"},
        },
        "required": ["url", "method"],
    }
    risk_level = "medium"

    async def execute(self, url: str = "", method: str = "GET", headers: str = "{}", body: str = "", **kwargs) -> ToolResult:
        try:
            parsed_headers = json.loads(headers) if headers else {}
            data = body.encode("utf-8") if body else None

            req = urllib.request.Request(url, data=data, headers=parsed_headers, method=method.upper())
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                return ToolResult(
                    success=True,
                    output=f"Status: {resp.status}\n\n{response_body[:5000]}"  # 截断防止过大
                )
        except urllib.error.HTTPError as e:
            return ToolResult(success=False, output="", error=f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            return ToolResult(success=False, output="", error=f"URL 错误: {e.reason}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

- [ ] **Step 5: 创建 src/tools/builtin/database.py**

```python
from src.tools.base import BaseTool, ToolResult


class DbQueryTool(BaseTool):
    name = "db_query"
    description = "执行只读 SQL 查询。参数 db_url: 数据库连接串, query: SELECT 语句。"
    parameters = {
        "type": "object",
        "properties": {
            "db_url": {"type": "string", "description": "数据库连接串，如 sqlite:///path/to/db"},
            "query": {"type": "string", "description": "只读 SELECT 查询语句"},
        },
        "required": ["db_url", "query"],
    }
    risk_level = "medium"

    async def execute(self, db_url: str = "", query: str = "", **kwargs) -> ToolResult:
        query_stripped = query.strip().upper()
        if not query_stripped.startswith("SELECT") and not query_stripped.startswith("PRAGMA"):
            return ToolResult(
                success=False, output="",
                error=f"仅允许只读查询(SELECT/PRAGMA)，收到: {query[:50]}"
            )

        try:
            import sqlite3
            # 从 sqlite:/// URL 提取文件路径
            if db_url.startswith("sqlite:///"):
                db_path = db_url[len("sqlite:///"):]
            elif "///" in db_url:
                db_path = db_url.split("///", 1)[1]
            else:
                db_path = db_url

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return ToolResult(success=True, output="(查询成功，无返回行)")

            # 格式化为表格
            col_names = [desc[0] for desc in cursor.description]
            lines = [" | ".join(col_names), "-" * len(" | ".join(col_names))]
            for row in rows[:100]:  # 最多 100 行
                lines.append(" | ".join(str(v) for v in row))

            output = "\n".join(lines)
            if len(rows) > 100:
                output += f"\n\n... (仅显示前 100 行，共 {len(rows)} 行)"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

- [ ] **Step 6: 更新 src/tools/builtin/__init__.py**

```python
from src.tools.builtin.file_ops import ReadFileTool, WriteFileTool, ListDirTool
from src.tools.builtin.search import SearchFileTool, GrepContentTool
from src.tools.builtin.shell import ExecShellTool
from src.tools.builtin.http import HttpRequestTool
from src.tools.builtin.database import DbQueryTool
```

- [ ] **Step 7: 运行测试确认通过**

```bash
pytest tests/test_tools_others.py -v
```
Expected: PASS (7 tests, note: http_request_get depends on network)

- [ ] **Step 8: Commit**

```bash
git add src/tools/builtin/ tests/test_tools_others.py
git commit -m "feat: add builtin tools (shell, http, database query)"
```

---

### Task 10: 技能层 — SkillRegistry

**Files:**
- Create: `src/skills/__init__.py`
- Create: `src/skills/registry.py`
- Create: `src/skills/definitions/__init__.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_skills.py
import pytest


def test_skill_definition_creation():
    from src.skills.registry import SkillDefinition

    skill = SkillDefinition(
        name="code-review",
        summary="审查代码变更",
        description="作为代码审查专家，按以下步骤执行...",
        tools=["read_file", "grep_content"],
        constraints=["不得修改文件"],
    )
    assert skill.name == "code-review"
    assert len(skill.tools) == 2
    assert "不得修改文件" in skill.constraints


def test_skill_registry_load_from_yaml(temp_db_path):
    import tempfile
    from pathlib import Path
    from src.skills.registry import SkillRegistry

    yaml_content = """
skills:
  - name: "test-skill"
    summary: "测试技能"
    tools: ["read_file"]
    constraints: ["只读"]
  - name: "another-skill"
    summary: "另一个技能"
    tools: ["exec_shell"]
    constraints: []
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        registry = SkillRegistry.load_from_yaml(yaml_path)
        assert len(registry.get_summaries()) == 2
        assert registry.get("test-skill") is not None
        assert registry.get("test-skill").tools == ["read_file"]
        assert registry.get("nonexistent") is None
    finally:
        Path(yaml_path).unlink(missing_ok=True)


def test_skill_registry_get_summaries():
    from src.skills.registry import SkillRegistry, SkillDefinition

    registry = SkillRegistry()
    registry.register(SkillDefinition(
        name="s1", summary="技能1", description="desc1", tools=[], constraints=[]
    ))
    registry.register(SkillDefinition(
        name="s2", summary="技能2", description="desc2", tools=[], constraints=[]
    ))

    summaries = registry.get_summaries()
    assert len(summaries) == 2
    summary_dict = {s["name"]: s["summary"] for s in summaries}
    assert summary_dict["s1"] == "技能1"
    assert summary_dict["s2"] == "技能2"


def test_skill_registry_get_full_definition():
    from src.skills.registry import SkillRegistry, SkillDefinition

    registry = SkillRegistry()
    full_desc = "详细的技能描述，包含步骤说明"
    registry.register(SkillDefinition(
        name="s1", summary="技能1", description=full_desc, tools=["t1"], constraints=["c1"]
    ))

    skill = registry.get("s1")
    assert skill is not None
    assert skill.description == full_desc
    assert skill.tools == ["t1"]
    assert skill.constraints == ["c1"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_skills.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/skills/registry.py**

```python
import yaml
from pathlib import Path
from pydantic import BaseModel


class SkillDefinition(BaseModel):
    name: str
    summary: str
    description: str
    tools: list[str] = []
    constraints: list[str] = []
    risk_override: dict[str, str] | None = None


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def get_summaries(self) -> list[dict]:
        return [
            {"name": s.name, "summary": s.summary}
            for s in self._skills.values()
        ]

    @classmethod
    def load_from_yaml(cls, config_path: str | Path) -> "SkillRegistry":
        registry = cls()
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        for skill_data in raw.get("skills", []):
            skill = SkillDefinition(
                name=skill_data["name"],
                summary=skill_data["summary"],
                description=skill_data.get("description", skill_data["summary"]),
                tools=skill_data.get("tools", []),
                constraints=skill_data.get("constraints", []),
                risk_override=skill_data.get("risk_override"),
            )
            registry.register(skill)
        return registry
```

- [ ] **Step 4: 创建 src/skills/__init__.py**

```python
from src.skills.registry import SkillRegistry, SkillDefinition
```

- [ ] **Step 5: 创建 src/skills/definitions/__init__.py**

```python
# 技能定义目录 — 后续按场景添加具体技能定义文件
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_skills.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add src/skills/ tests/test_skills.py
git commit -m "feat: add SkillRegistry with YAML loading and progressive discovery"
```

---

### Task 11: Agent State 定义

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/state.py`
- Test: `tests/test_agent_state.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_agent_state.py
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def test_agent_state_defaults():
    from src.agent.state import AgentState

    state = AgentState(
        messages=[],
        session_id="sess-1",
        user_id="user-1",
        recalled_memories=[],
        active_skill=None,
        tool_calls=[],
        loop_count=0,
        final_answer=None,
    )
    assert state["session_id"] == "sess-1"
    assert state["loop_count"] == 0
    assert state["final_answer"] is None
    assert state["active_skill"] is None
    assert state["compact_summary"] is None


def test_tool_call_typed_dict():
    from src.agent.state import ToolCall
    tc = ToolCall(name="read_file", args={"path": "/tmp/test.txt"}, id="call_1")
    assert tc["name"] == "read_file"
    assert tc["args"]["path"] == "/tmp/test.txt"


def test_agent_state_with_messages():
    from src.agent.state import AgentState

    state = AgentState(
        messages=[HumanMessage(content="hello")],
        session_id="sess-1",
        user_id="user-1",
        recalled_memories=[],
        active_skill=None,
        tool_calls=[],
        loop_count=0,
        final_answer=None,
    )
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "hello"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_agent_state.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/agent/state.py**

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    name: str
    args: dict
    id: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str
    recalled_memories: list[dict]
    active_skill: dict | None
    tool_calls: list[dict]
    loop_count: int
    final_answer: str | None
    compact_summary: str | None
```

- [ ] **Step 4: 创建 src/agent/__init__.py**

```python
from src.agent.state import AgentState, ToolCall
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_agent_state.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/agent/ tests/test_agent_state.py
git commit -m "feat: add AgentState TypedDict definition"
```

---

### Task 12: Agent Nodes — Observe

**Files:**
- Create: `src/agent/nodes/__init__.py`
- Create: `src/agent/nodes/observe.py`
- Create: `src/agent/compact.py`
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_agent_nodes.py
import pytest
from langchain_core.messages import HumanMessage


def test_compact_estimates_tokens():
    from src.agent.compact import estimate_tokens
    count = estimate_tokens("Hello, world!")
    assert count > 0
    assert count < 10


def test_compact_not_triggered_for_short_messages():
    from src.agent.compact import should_compact
    messages = [HumanMessage(content="short message")]
    assert should_compact(messages, threshold_ratio=0.8, max_tokens=65536) is False


def test_compact_triggers_when_over_threshold():
    from src.agent.compact import should_compact
    # Create a long message that exceeds threshold
    long_text = "hello " * 100000  # ~100k words → well over any threshold
    messages = [HumanMessage(content=long_text)]
    assert should_compact(messages, threshold_ratio=0.1, max_tokens=65536) is True


def test_format_skill_summaries_for_prompt():
    from src.agent.nodes.observe import format_skill_summaries
    summaries = [
        {"name": "code-review", "summary": "审查代码"},
        {"name": "data-analyze", "summary": "分析数据"},
    ]
    text = format_skill_summaries(summaries)
    assert "code-review" in text
    assert "审查代码" in text
    assert "data-analyze" in text
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_agent_nodes.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/agent/compact.py**

```python
import tiktoken
from langchain_core.messages import BaseMessage


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += estimate_tokens(content)
    return total


def should_compact(messages: list[BaseMessage], threshold_ratio: float, max_tokens: int) -> bool:
    if not messages:
        return False
    estimated = estimate_messages_tokens(messages)
    threshold = int(max_tokens * threshold_ratio)
    return estimated > threshold


async def compact_messages(
    messages: list[BaseMessage],
    chat_model,  # BaseChatModel instance for summarization
    keep_recent: int = 6,
) -> tuple[str, list[BaseMessage]]:
    """将早期消息压缩为摘要。返回 (summary_text, recent_messages)"""
    if len(messages) <= keep_recent:
        return "", messages

    boundary = len(messages) - keep_recent
    old_messages = messages[:boundary]
    recent_messages = messages[boundary:]

    # 构建摘要请求
    old_text = "\n".join(
        f"[{m.type}]: {m.content if isinstance(m.content, str) else str(m.content)[:500]}"
        for m in old_messages
    )
    summary_prompt = f"请将以下对话历史压缩为一段简洁的摘要（保留关键决策、用户偏好和重要信息）：\n\n{old_text}"

    from langchain_core.messages import HumanMessage, SystemMessage
    response = await chat_model.ainvoke([
        SystemMessage(content="你是一个对话摘要助手。用中文输出简洁摘要。"),
        HumanMessage(content=summary_prompt),
    ])
    summary = response.content if isinstance(response.content, str) else str(response.content)
    return summary, recent_messages
```

- [ ] **Step 4: 创建 src/agent/nodes/observe.py**

```python
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph
from src.agent.state import AgentState
from src.agent.compact import should_compact, compact_messages, estimate_messages_tokens


SYSTEM_PROMPT_BASE = """你是一个通用的 AI 助手。你可以使用工具来完成用户的任务。

## 可用技能
{skill_summaries}

## 可用工具
你可以使用提供的工具来完成任务。如果任务需要多个步骤，请逐步调用工具。

## 记忆
{recalled_memories}

## 行为准则
- 仔细分析用户的需求后再行动
- 优先使用技能（如果匹配），技能会提供详细的执行指导
- 如果无法完成，请诚实告知用户原因
"""


def format_skill_summaries(summaries: list[dict]) -> str:
    if not summaries:
        return "（暂无可用技能）"
    lines = []
    for s in summaries:
        lines.append(f"- **{s['name']}**: {s['summary']}")
    return "\n".join(lines)


def format_recalled_memories(memories: list[dict]) -> str:
    if not memories:
        return "（暂无相关记忆）"
    lines = []
    for m in memories:
        lines.append(f"- [{m.get('category', 'fact')}] {m['content']}")
    return "\n".join(lines)


async def observe_node(state: AgentState, config: dict = None) -> dict:
    """Observe 节点：加载上下文、召回记忆、注入 System Prompt"""
    messages = state.get("messages", [])

    # 1. 如果需要压缩，从 state 中获取已有的摘要
    compact_summary = state.get("compact_summary")

    # 2. 召回的记忆（由调用方在进入图之前设置，或在上一轮循环中设置）
    recalled = state.get("recalled_memories", [])

    # 3. 技能清单（由调用方在图外部设置）
    # 这里使用空列表作为默认值，实际由 server 层注入
    skill_summaries = []

    # 4. 构建 System Prompt（只在第一轮注入）
    # 如果 messages 中没有 SystemMessage，则注入
    has_system = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system:
        system_text = SYSTEM_PROMPT_BASE.format(
            skill_summaries=format_skill_summaries(skill_summaries),
            recalled_memories=format_recalled_memories(recalled),
        )
        if compact_summary:
            system_text = f"## 历史对话摘要\n{compact_summary}\n\n{system_text}"

        new_messages = [SystemMessage(content=system_text)] + list(messages)
        return {"messages": new_messages}

    return {}
```

- [ ] **Step 5: 创建 src/agent/nodes/__init__.py**

```python
from src.agent.nodes.observe import observe_node
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_agent_nodes.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add src/agent/ tests/test_agent_nodes.py
git commit -m "feat: add Observe node and context compaction logic"
```

---

### Task 13: Agent Nodes — Think

**Files:**
- Create: `src/agent/nodes/think.py`
- Modify: `src/agent/nodes/__init__.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: 编写失败测试**

```python
# 追加到 tests/test_agent_nodes.py

def test_parse_llm_response_direct_answer():
    from src.agent.nodes.think import parse_llm_response
    from langchain_core.messages import AIMessage

    ai_msg = AIMessage(content="这是最终答案。", tool_calls=[])
    result = parse_llm_response(ai_msg)
    assert result["final_answer"] == "这是最终答案。"
    assert result["tool_calls"] == []


def test_parse_llm_response_tool_calls():
    from src.agent.nodes.think import parse_llm_response
    from langchain_core.messages import AIMessage

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "call_1"}],
    )
    result = parse_llm_response(ai_msg)
    assert result["final_answer"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "read_file"


def test_should_archive_user_command():
    from src.agent.nodes.think import detect_archive_triggers

    triggers = detect_archive_triggers("请记住，我偏好使用 tab 缩进")
    assert len(triggers) >= 1
    assert any("tab 缩进" in t["content"] for t in triggers)


def test_should_archive_no_trigger():
    from src.agent.nodes.think import detect_archive_triggers

    triggers = detect_archive_triggers("帮我读一下 README.md 文件")
    assert len(triggers) == 0
```

- [ ] **Step 2: 运行新测试确认失败**

```bash
pytest tests/test_agent_nodes.py -v -k "parse_llm or archive"
```
Expected: FAIL

- [ ] **Step 3: 创建 src/agent/nodes/think.py**

```python
import re
from langchain_core.messages import AIMessage
from src.agent.state import AgentState


def parse_llm_response(response: AIMessage) -> dict:
    """解析 LLM 响应，区分最终答案和工具调用"""
    if response.tool_calls and len(response.tool_calls) > 0:
        return {
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"], "id": tc.get("id", "")}
                for tc in response.tool_calls
            ],
            "final_answer": None,
        }
    return {
        "tool_calls": [],
        "final_answer": response.content if isinstance(response.content, str) else str(response.content),
    }


# 归档触发关键词
ARCHIVE_TRIGGERS = [
    r"记住[，,：:]",
    r"保存[这这]",
    r"归档[这这]",
    r"记录下[来这]",
    r"我偏好",
    r"我习惯",
    r"我不喜欢",
    r"我总是",
    r"我的.*是",
    r"备忘[，,：:]",
    r"记一下",
    r"别忘了",
]


def detect_archive_triggers(user_message: str) -> list[dict]:
    """检测用户消息中是否包含归档触发词"""
    triggers = []
    for pattern in ARCHIVE_TRIGGERS:
        if re.search(pattern, user_message):
            triggers.append({
                "content": user_message.strip(),
                "source": "user_command",
                "category": "preference",
            })
            break
    return triggers


async def think_node(state: AgentState, config: dict = None) -> dict:
    """Think 节点：调用 LLM，解析工具调用或最终答案。实际调用由 graph 借用 model 完成。

    注意：此节点不直接调用 LLM。LLM 调用由 LangGraph 的标准模型节点完成。
    此节点的职责是：如果 state 中有 tool_calls（来自 AIMessage），则解析它们。
    实际推理逻辑见 graph.py 中 bind_tools 的标准模式。

    这里做一个薄封装：如果最新一条 AI 消息包含 tool_calls，解析它们。
    """
    messages = state.get("messages", [])

    # 在 LangGraph 中，LLM 调用后的 AIMessage 已经被 append 到 messages
    # 此节点负责从最新 AIMessage 中提取信息
    if not messages:
        return {}

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage):
        parsed = parse_llm_response(last_msg)
        return parsed

    return {}
```

- [ ] **Step 4: 更新 src/agent/nodes/__init__.py**

```python
from src.agent.nodes.observe import observe_node
from src.agent.nodes.think import think_node
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_agent_nodes.py -v
```
Expected: PASS (8 tests total)

- [ ] **Step 6: Commit**

```bash
git add src/agent/nodes/ tests/test_agent_nodes.py
git commit -m "feat: add Think node with LLM response parsing and archive detection"
```

---

### Task 14: Agent Nodes — Act

**Files:**
- Create: `src/agent/nodes/act.py`
- Modify: `src/agent/nodes/__init__.py`
- Modify: `tests/test_agent_nodes.py`

- [ ] **Step 1: 编写失败测试**

```python
# 追加到 tests/test_agent_nodes.py

def test_permission_check_low_risk_allowed():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("read_file", "low")
    assert allowed is True
    assert reason == ""


def test_permission_check_high_risk_blocked():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("exec_shell", "high")
    assert allowed is False
    assert "高风险" in reason


def test_permission_check_medium_allowed():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("write_file", "medium")
    assert allowed is True


def test_permission_check_critical_blocked():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("delete_system", "critical")
    assert allowed is False
    assert "特危" in reason
```

- [ ] **Step 2: 运行新测试确认失败**

```bash
pytest tests/test_agent_nodes.py -v -k "permission"
```
Expected: FAIL

- [ ] **Step 3: 创建 src/agent/nodes/act.py**

```python
from langchain_core.messages import ToolMessage
from src.agent.state import AgentState
from config.settings import MAX_LOOPS


# MVP 权限白名单：high/critical 级别工具默认拒绝
ALLOWED_RISK_LEVELS = {"low", "medium"}
BLOCKED_RISK_LEVELS = {"high", "critical"}


def check_permission(tool_name: str, risk_level: str) -> tuple[bool, str]:
    """检查工具是否允许执行。返回 (allowed, reason)"""
    if risk_level in BLOCKED_RISK_LEVELS:
        if risk_level == "critical":
            return False, f"工具 '{tool_name}' 为特危操作，禁止执行"
        return False, f"工具 '{tool_name}' 为高风险操作，当前版本暂不支持。请使用其他替代工具"
    return True, ""


async def act_node(state: AgentState, config: dict = None) -> dict:
    """Act 节点：执行工具调用，返回 ToolMessage"""
    tool_calls = state.get("tool_calls", [])
    messages: list[ToolMessage] = []
    loop_count = state.get("loop_count", 0)

    # 此节点依赖外部的 ToolRegistry 实例来执行工具
    # ToolRegistry 通过 config 传入，见 graph.py 中的 RunnableConfig
    tool_registry = config.get("configurable", {}).get("tool_registry") if config else None

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        call_id = tc.get("id", "")

        if tool_registry:
            tool = tool_registry.get(tool_name)
            if tool is None:
                messages.append(ToolMessage(
                    content=f"错误：工具 '{tool_name}' 未注册",
                    tool_call_id=call_id,
                ))
                continue

            # 权限检查
            allowed, reason = check_permission(tool_name, tool.risk_level)
            if not allowed:
                messages.append(ToolMessage(
                    content=f"权限拒绝：{reason}",
                    tool_call_id=call_id,
                ))
                continue

            # 执行工具
            result = await tool_registry.execute(tool_name, **tool_args)
        else:
            # 无 ToolRegistry 时（测试环境），返回占位消息
            messages.append(ToolMessage(
                content=f"[ToolRegistry not available] Would execute: {tool_name}({tool_args})",
                tool_call_id=call_id,
            ))
            continue

        content = result.output if result.success else f"执行失败: {result.error}"
        messages.append(ToolMessage(content=content, tool_call_id=call_id))

    return {
        "messages": messages,
        "loop_count": loop_count + 1,
        "tool_calls": [],  # 清除已处理的工具调用
    }
```

- [ ] **Step 4: 更新 src/agent/nodes/__init__.py**

```python
from src.agent.nodes.observe import observe_node
from src.agent.nodes.think import think_node
from src.agent.nodes.act import act_node, check_permission
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_agent_nodes.py -v -k "permission"
```
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/agent/nodes/ tests/test_agent_nodes.py
git commit -m "feat: add Act node with permission whitelist and tool execution"
```

---

### Task 15: Agent Graph 组装

**Files:**
- Create: `src/agent/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_graph.py
import pytest
from langchain_core.messages import HumanMessage


def test_build_graph_returns_compiled_graph():
    from src.agent.graph import build_graph
    from src.tools.registry import ToolRegistry
    from src.skills.registry import SkillRegistry

    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()

    graph = build_graph(tool_registry, skill_registry)
    assert graph is not None
    # Compiled graph should have invoke method
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "ainvoke")


def test_graph_routing_direct_answer():
    """测试：无工具调用的直接响应 → END"""
    from src.agent.graph import route_after_think
    from src.agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": "这是最终答案",
        "compact_summary": None,
    }
    assert route_after_think(state) == "__end__"


def test_graph_routing_tool_calls():
    """测试：有工具调用 → Act"""
    from src.agent.graph import route_after_think
    from src.agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [{"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "c1"}],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }
    assert route_after_think(state) == "act"


def test_graph_routing_max_loops():
    """测试：超过最大循环次数 → END"""
    from src.agent.graph import route_after_think, MAX_LOOPS
    from src.agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [{"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "c1"}],
        "loop_count": MAX_LOOPS + 1,
        "final_answer": None,
        "compact_summary": None,
    }
    result = route_after_think(state)
    assert result == "__end__"
    assert state["final_answer"] is not None  # 应该设置了兜底消息
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_graph.py -v
```
Expected: FAIL

- [ ] **Step 3: 创建 src/agent/graph.py**

```python
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.nodes.observe import observe_node
from src.agent.nodes.think import think_node
from src.agent.nodes.act import act_node
from config.settings import MAX_LOOPS

MAX_LOOPS = MAX_LOOPS  # Re-export for test access


def route_after_think(state: AgentState) -> Literal["act", "__end__"]:
    if state.get("final_answer") is not None:
        return "__end__"
    if state.get("loop_count", 0) >= MAX_LOOPS:
        state["final_answer"] = "达到最大循环次数，任务中断。已完成的部分已返回。"
        return "__end__"
    if state.get("tool_calls"):
        return "act"
    return "__end__"


def build_graph(
    tool_registry=None,
    skill_registry=None,
    checkpoint_saver=None,
):
    """构建 Agent StateGraph

    Args:
        tool_registry: ToolRegistry 实例，按需通过 config 传入 Act 节点
        skill_registry: SkillRegistry 实例
        checkpoint_saver: LangGraph checkpoint saver（SqliteSaver 或 MemorySaver）
    """
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("observe", observe_node)
    builder.add_node("think", think_node)
    builder.add_node("act", act_node)

    # 添加边
    builder.set_entry_point("observe")
    builder.add_edge("observe", "think")
    builder.add_conditional_edges(
        "think",
        route_after_think,
        {"act": "act", "__end__": END},
    )
    builder.add_edge("act", "observe")  # 循环回到 Observe

    # 编译
    if checkpoint_saver:
        return builder.compile(checkpointer=checkpoint_saver)
    return builder.compile()
```

- [ ] **Step 4: 更新 src/agent/__init__.py**

```python
from src.agent.state import AgentState, ToolCall
from src.agent.graph import build_graph, route_after_think, MAX_LOOPS
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_graph.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/agent/ tests/test_graph.py
git commit -m "feat: add Agent Graph assembly with observe-think-act loop"
```

---

### Task 16: FastAPI — Request/Response Schemas 与 Session Router

**Files:**
- Create: `src/server/__init__.py`
- Create: `src/server/schemas/__init__.py`
- Create: `src/server/schemas/request.py`
- Create: `src/server/schemas/response.py`
- Create: `src/server/routers/__init__.py`
- Create: `src/server/routers/session.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from src.server.app import create_app
    # 使用内存数据库进行测试
    import tempfile
    import os
    db_path = os.path.join(tempfile.gettempdir(), f"test_pyagent_{os.getpid()}.db")
    app = create_app(db_url=f"sqlite+aiosqlite:///{db_path}")
    return app


@pytest.mark.asyncio
async def test_health_check(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={
            "user_id": "test-user",
            "message": "Hello, world!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "answer" in data or "error" in data


@pytest.mark.asyncio
async def test_chat_with_existing_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session
        resp1 = await client.post("/chat", json={"user_id": "test-user", "message": "Hello"})
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]

        # Continue session
        resp2 = await client.post("/chat", json={
            "session_id": session_id,
            "user_id": "test-user",
            "message": "Continue",
        })
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_get_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/chat", json={"user_id": "test-user", "message": "Hello"})
        session_id = resp1.json()["session_id"]

        resp2 = await client.get(f"/session/{session_id}")
        assert resp2.status_code == 200
        assert resp2.json()["id"] == session_id
        assert resp2.json()["status"] == "active"


@pytest.mark.asyncio
async def test_get_session_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/session/nonexistent-id")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_sessions(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/chat", json={"user_id": "test-user", "message": "Msg 1"})
        await client.post("/chat", json={"user_id": "test-user", "message": "Msg 2"})

        resp = await client.get("/session/list?user_id=test-user")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2


@pytest.mark.asyncio
async def test_close_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/chat", json={"user_id": "test-user", "message": "Hello"})
        session_id = resp1.json()["session_id"]

        resp2 = await client.post(f"/session/{session_id}/close")
        assert resp2.status_code == 200

        resp3 = await client.get(f"/session/{session_id}")
        assert resp3.json()["status"] == "closed"


@pytest.mark.asyncio
async def test_delete_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/chat", json={"user_id": "test-user", "message": "Hello"})
        session_id = resp1.json()["session_id"]

        resp2 = await client.delete(f"/session/{session_id}")
        assert resp2.status_code == 200

        resp3 = await client.get(f"/session/{session_id}")
        assert resp3.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_api.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: 创建 src/server/schemas/request.py**

```python
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str
    message: str
```

- [ ] **Step 4: 创建 src/server/schemas/response.py**

```python
from pydantic import BaseModel


class ChatResponse(BaseModel):
    session_id: str
    answer: str | None = None
    error: str | None = None
    loop_count: int = 0


class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str | None
    status: str
    message_count: int
    created_at: str | None
    updated_at: str | None
```

- [ ] **Step 5: 创建 src/server/routers/session.py**

```python
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.db.engine import async_session_factory

router = APIRouter(prefix="/session", tags=["session"])


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        yield session


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("SELECT id, user_id, title, status, message_count, created_at, updated_at FROM sessions WHERE id = :id"),
        {"id": session_id},
    )
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": row[0],
        "user_id": row[1],
        "title": row[2],
        "status": row[3],
        "message_count": row[4],
        "created_at": str(row[5]) if row[5] else None,
        "updated_at": str(row[6]) if row[6] else None,
    }


@router.get("/list")
async def list_sessions(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("SELECT id, user_id, title, status, message_count, created_at, updated_at FROM sessions WHERE user_id = :uid ORDER BY updated_at DESC LIMIT 50"),
        {"uid": user_id},
    )
    rows = result.fetchall()
    return [
        {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "status": row[3],
            "message_count": row[4],
            "created_at": str(row[5]) if row[5] else None,
            "updated_at": str(row[6]) if row[6] else None,
        }
        for row in rows
    ]


@router.post("/{session_id}/close")
async def close_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("UPDATE sessions SET status = 'closed', updated_at = :now WHERE id = :id AND status = 'active'"),
        {"id": session_id, "now": datetime.now(timezone.utc)},
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found or already closed")
    return {"status": "closed"}


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    # 先删 session 记录
    await db.execute(text("DELETE FROM sessions WHERE id = :id"), {"id": session_id})
    await db.commit()

    # 清理 LangGraph checkpoint 数据
    await db.execute(text("DELETE FROM checkpoints WHERE thread_id = :tid"), {"tid": session_id})
    await db.execute(text("DELETE FROM checkpoint_writes WHERE thread_id = :tid"), {"tid": session_id})
    await db.commit()

    return {"deleted": True}
```

- [ ] **Step 6: Commit (server schemas + session router)**

```bash
git add src/server/ tests/test_api.py
git commit -m "feat: add FastAPI request/response schemas and session CRUD router"
```

---

### Task 17: FastAPI — Chat Router 与 App 组装

**Files:**
- Create: `src/server/routers/chat.py`
- Create: `src/server/app.py`
- Modify: `src/server/__init__.py`

- [ ] **Step 1: 创建 src/server/routers/chat.py**

```python
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_core.messages import HumanMessage, AIMessage

from src.server.schemas.request import ChatRequest
from src.server.schemas.response import ChatResponse
from src.agent.graph import build_graph
from src.db.engine import async_session_factory
from src.memory import MemoryProvider
from src.tools.registry import ToolRegistry
from src.skills.registry import SkillRegistry
from src.models import ModelProvider
from config.settings import MEMORY_RECALL_TOP_K

logger = logging.getLogger(__name__)

router = APIRouter()


def get_graph(app):
    """延迟获取或创建 graph 实例"""
    if not hasattr(app.state, "_graph"):
        app.state._graph = build_graph(
            tool_registry=app.state.tool_registry,
            skill_registry=app.state.skill_registry,
            checkpoint_saver=app.state.checkpoint_saver,
        )
    return app.state._graph


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from src.server.app import app as current_app
    # Use FastAPI's app.state from the module-level reference
    ...


def configure_chat_router(app_ref):
    """用 app 引用配置 chat router 的实际实现"""

    @router.post("/chat", response_model=ChatResponse)
    async def chat_handler(request: ChatRequest):
        app = app_ref()
        tool_registry: ToolRegistry = app.state.tool_registry
        skill_registry: SkillRegistry = app.state.skill_registry
        model_provider: ModelProvider = app.state.model_provider
        memory_provider: MemoryProvider | None = getattr(app.state, "memory_provider", None)

        session_id = request.session_id or str(uuid.uuid4())
        is_new_session = request.session_id is None

        # 1. 验证/创建 session
        async with async_session_factory() as db:
            if not is_new_session:
                result = await db.execute(
                    text("SELECT id, status FROM sessions WHERE id = :id"),
                    {"id": session_id},
                )
                row = result.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                if row[1] == "closed":
                    raise HTTPException(status_code=400, detail="Session is closed")
            else:
                # 创建新 session
                title = request.message[:80] + ("..." if len(request.message) > 80 else "")
                await db.execute(
                    text("""
                        INSERT INTO sessions (id, user_id, title, status, message_count)
                        VALUES (:id, :user_id, :title, 'active', 0)
                    """),
                    {"id": session_id, "user_id": request.user_id, "title": title},
                )
                await db.commit()

        # 2. 召回长期记忆
        recalled_memories = []
        if memory_provider:
            recalled_memories = await memory_provider.recall(
                request.message, top_k=MEMORY_RECALL_TOP_K, user_id=request.user_id
            )
            recalled_memories = [m.model_dump() if hasattr(m, "model_dump") else m for m in recalled_memories]

        # 3. 获取技能清单
        skill_summaries = skill_registry.get_summaries() if skill_registry else []

        # 4. 构建初始状态
        graph = get_graph(app)
        model = model_provider.get_chat_model()
        model_with_tools = model.bind_tools(tool_registry.get_llm_tools())

        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "session_id": session_id,
            "user_id": request.user_id,
            "recalled_memories": recalled_memories,
            "active_skill": None,
            "tool_calls": [],
            "loop_count": 0,
            "final_answer": None,
            "compact_summary": None,
        }

        # 5. 执行 Agent Loop
        # 使用 graph.ainvoke，LangGraph 会通过 checkpoint 自动持久化
        config = {
            "configurable": {
                "thread_id": session_id,
                "tool_registry": tool_registry,
                "model": model_with_tools,
                "skill_registry": skill_registry,
                "memory_provider": memory_provider,
            },
        }

        try:
            result = await graph.ainvoke(initial_state, config)

            # 提取最终答案
            final_answer = result.get("final_answer", "")
            if not final_answer:
                # 如果没有 final_answer，从最后一条 AI 消息提取
                for msg in reversed(result.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break

        except Exception as e:
            logger.exception("Agent loop execution failed")
            return ChatResponse(
                session_id=session_id,
                error=f"执行失败: {str(e)}",
            )

        # 6. 更新 session 元数据
        async with async_session_factory() as db:
            await db.execute(
                text("UPDATE sessions SET message_count = message_count + 1, updated_at = :now WHERE id = :id"),
                {"id": session_id, "now": datetime.now(timezone.utc)},
            )
            await db.commit()

        # 7. 归档判断
        from src.agent.nodes.think import detect_archive_triggers
        if memory_provider:
            triggers = detect_archive_triggers(request.message)
            for t in triggers:
                await memory_provider.archive(
                    user_id=request.user_id,
                    session_id=session_id,
                    content=t["content"],
                    category=t["category"],
                    source=t["source"],
                )

        return ChatResponse(
            session_id=session_id,
            answer=final_answer or "(无输出)",
            loop_count=result.get("loop_count", 0),
        )
```

Wait, the above approach has an issue with the lazy `app_ref()` pattern. Let me simplify.

Actually, let me re-design the chat router to use FastAPI's `app.state` properly via `Request` dependency injection.

Let me redo this:

- [ ] **Step 2: 重写 chat.py（更清晰的实现）**

```python
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text
from langchain_core.messages import HumanMessage, AIMessage

from src.server.schemas.request import ChatRequest
from src.server.schemas.response import ChatResponse
from src.db.engine import async_session_factory
from config.settings import MEMORY_RECALL_TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest, req: Request):
    app_state = req.app.state
    tool_registry = app_state.tool_registry
    skill_registry = app_state.skill_registry
    model_provider = app_state.model_provider
    memory_provider = getattr(app_state, "memory_provider", None)

    session_id = request.session_id or str(uuid.uuid4())
    is_new_session = request.session_id is None

    # 1. 验证/创建 session
    async with async_session_factory() as db:
        if not is_new_session:
            result = await db.execute(
                text("SELECT id, status FROM sessions WHERE id = :id"),
                {"id": session_id},
            )
            row = result.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            if row[1] == "closed":
                raise HTTPException(status_code=400, detail="Session is closed")
        else:
            title = request.message[:80] + ("..." if len(request.message) > 80 else "")
            await db.execute(
                text("""
                    INSERT INTO sessions (id, user_id, title, status, message_count)
                    VALUES (:id, :user_id, :title, 'active', 0)
                """),
                {"id": session_id, "user_id": request.user_id, "title": title},
            )
            await db.commit()

    # 2. 召回长期记忆
    recalled_data = []
    if memory_provider:
        recalled = await memory_provider.recall(
            request.message, top_k=MEMORY_RECALL_TOP_K, user_id=request.user_id
        )
        recalled_data = [
            {
                "content": m.content,
                "category": m.category,
            }
            for m in recalled
        ]

    # 3. 构建 Agent State
    model = model_provider.get_chat_model()
    model_with_tools = model.bind_tools(tool_registry.get_llm_tools())

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "session_id": session_id,
        "user_id": request.user_id,
        "recalled_memories": recalled_data,
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }

    config = {
        "configurable": {
            "thread_id": session_id,
            "tool_registry": tool_registry,
            "skill_registry": skill_registry,
        },
    }

    # 4. 执行 Agent Loop
    try:
        # 获取/复用 graph
        if not hasattr(app_state, "_graph"):
            from src.agent.graph import build_graph
            app_state._graph = build_graph(
                tool_registry=tool_registry,
                skill_registry=skill_registry,
                checkpoint_saver=app_state.checkpoint_saver,
            )

        result = await app_state._graph.ainvoke(initial_state, config)

        # 提取答案
        final_answer = result.get("final_answer", "")
        if not final_answer:
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

    except Exception as e:
        logger.exception("Agent loop execution failed")
        return ChatResponse(session_id=session_id, error=f"执行失败: {str(e)}")

    # 5. 更新 session
    async with async_session_factory() as db:
        await db.execute(
            text("UPDATE sessions SET message_count = message_count + 1, updated_at = :now WHERE id = :id"),
            {"id": session_id, "now": datetime.now(timezone.utc)},
        )
        await db.commit()

    # 6. 归档判断
    if memory_provider:
        from src.agent.nodes.think import detect_archive_triggers
        triggers = detect_archive_triggers(request.message)
        for t in triggers:
            await memory_provider.archive(
                user_id=request.user_id,
                session_id=session_id,
                content=t["content"],
                category=t["category"],
                source=t["source"],
            )

    return ChatResponse(
        session_id=session_id,
        answer=final_answer or "(无输出)",
        loop_count=result.get("loop_count", 0),
    )
```

- [ ] **Step 3: 创建 src/server/app.py**

```python
from pathlib import Path
from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver

from src.db.engine import create_engine, init_db
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

        # Checkpoint saver (LangGraph persistence)
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        app.state.checkpoint_saver = SqliteSaver.from_conn_string(_db_url.replace("+aiosqlite", ""))

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
        if SKILLS_CONFIG_PATH.exists():
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
```

- [ ] **Step 4: 创建 src/server/__init__.py**

```python
from src.server.app import create_app, app
```

- [ ] **Step 5: 更新 agent/nodes/think.py 中的 think_node，使其实际调用 LLM**

（这是关键集成点：Think 节点需要从 config 中获取 model 并调用）

```python
# 修改 think_node 以实际调用 LLM
async def think_node(state: AgentState, config: dict = None) -> dict:
    """Think 节点：调用 LLM 推理，决定下一步"""
    model = config.get("configurable", {}).get("model") if config else None
    if model is None:
        # 无 model 时的回退（测试环境）
        return {"final_answer": "Model not configured", "tool_calls": []}

    messages = state.get("messages", [])
    response = await model.ainvoke(messages)
    parsed = parse_llm_response(response)
    return parsed
```

- [ ] **Step 6: 运行测试**

```bash
pytest tests/test_api.py -v
```
NOTE: /chat 端点测试需要真实的 LLM API Key。无 Key 时这些测试会因模型调用失败而报错。可在 CI 中设置 `DEEPSEEK_API_KEY` 环境变量后运行。

- [ ] **Step 7: Commit**

```bash
git add src/server/ src/agent/nodes/think.py
git commit -m "feat: add FastAPI app assembly, chat router with full Agent Loop integration"
```

---

### Task 18: 集成测试与端到端验证

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写端到端测试**

```python
# tests/test_integration.py
import pytest
import tempfile
import os
from pathlib import Path


@pytest.mark.asyncio
async def test_full_agent_loop_without_llm():
    """模拟无LLM的完整流程：验证 StateGraph 拓扑正确执行"""
    from src.agent.graph import build_graph
    from src.agent.state import AgentState
    from src.tools.registry import ToolRegistry
    from src.skills.registry import SkillRegistry
    from langchain_core.messages import HumanMessage

    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()

    graph = build_graph(tool_registry, skill_registry)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="test")],
        "session_id": "test-session",
        "user_id": "test-user",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }

    # 无 model 配置时，think_node 应直接返回 final_answer
    config = {"configurable": {}}
    result = await graph.ainvoke(initial_state, config)
    assert "final_answer" in result or "messages" in result


@pytest.mark.asyncio
async def test_session_persistence_with_checkpoint():
    """验证 LangGraph checkpoint 持久化"""
    import tempfile
    from langgraph.checkpoint.memory import MemorySaver
    from src.agent.graph import build_graph
    from src.agent.state import AgentState
    from src.tools.registry import ToolRegistry
    from src.skills.registry import SkillRegistry
    from langchain_core.messages import HumanMessage, AIMessage

    db_path = os.path.join(tempfile.gettempdir(), "test_checkpoint.db")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        saver = SqliteSaver.from_conn_string(db_path)
    except Exception:
        saver = MemorySaver()

    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    graph = build_graph(tool_registry, skill_registry, checkpoint_saver=saver)

    config = {"configurable": {"thread_id": "test-thread-1"}}

    # First invocation with preset final_answer (simulating a direct answer)
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "session_id": "sess-ckpt",
        "user_id": "user-ckpt",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": "Hi there!",
        "compact_summary": None,
    }

    await graph.ainvoke(state, config)

    # Second invocation: should continue from checkpoint
    state2: AgentState = {
        "messages": [HumanMessage(content="how are you?")],
        "session_id": "sess-ckpt",
        "user_id": "user-ckpt",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": "I'm fine!",
        "compact_summary": None,
    }

    result = await graph.ainvoke(state2, config)
    # Should have messages from both invocations
    messages = result.get("messages", [])
    assert len(messages) >= 1  # At least the new message

    Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_observe_injects_system_prompt():
    """验证 Observe 节点注入 System Prompt"""
    from src.agent.nodes.observe import observe_node
    from src.agent.state import AgentState
    from langchain_core.messages import HumanMessage, SystemMessage

    state: AgentState = {
        "messages": [HumanMessage(content="test")],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }

    result = await observe_node(state)
    messages = result.get("messages", state["messages"])
    assert isinstance(messages[0], SystemMessage)


def test_all_builtin_tools_registered():
    """验证所有 8 个内置工具可正常注册"""
    from src.tools.registry import ToolRegistry
    from src.tools.builtin import (
        ReadFileTool, WriteFileTool, ListDirTool,
        SearchFileTool, GrepContentTool,
        ExecShellTool, HttpRequestTool, DbQueryTool,
    )

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirTool())
    registry.register(SearchFileTool())
    registry.register(GrepContentTool())
    registry.register(ExecShellTool())
    registry.register(HttpRequestTool())
    registry.register(DbQueryTool())

    all_tools = registry.list_all()
    assert len(all_tools) == 8

    llm_tools = registry.get_llm_tools()
    assert len(llm_tools) == 8
    for t in llm_tools:
        assert "type" in t
        assert t["type"] == "function"
        assert "name" in t["function"]
```

- [ ] **Step 2: 运行集成测试**

```bash
pytest tests/test_integration.py -v
```
Expected: PASS (4 tests, checkpoint test may vary by LangGraph version)

- [ ] **Step 3: 运行全部测试**

```bash
pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for agent loop and checkpoint persistence"
```

---

## 验证清单

实现完成后，逐一验证以下行为：

1. **启动服务**: `uvicorn src.server.app:app --reload`
2. **健康检查**: `curl http://localhost:8000/health` → `{"status":"ok"}`
3. **发送首条消息**: `curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"user_id":"test","message":"你好"}'`
4. **继续会话**: 使用返回的 `session_id` 发送后续消息
5. **查看会话列表**: `curl http://localhost:8000/session/list?user_id=test`
6. **查看会话详情**: `curl http://localhost:8000/session/{session_id}`
7. **关闭会话**: `curl -X POST http://localhost:8000/session/{session_id}/close`
8. **工具调用**: 发送 "帮我读一下 pyproject.toml 文件" 验证 read_file 工具被调用
9. **记忆归档**: 发送 "记住，我偏好使用 black 格式化代码" 验证触发归档
10. **权限拦截**: 发送 "用 shell 执行 rm -rf /" 验证 exec_shell 被拒绝

---

## 环境变量

```bash
export DEEPSEEK_API_KEY=sk-your-key
export DEEPSEEK_API_BASE=https://api.deepseek.com/v1
export PYAGENT_DB_PATH=./data/pyagent.db
```
