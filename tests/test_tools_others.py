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
async def test_db_query_select(raw_db_path):
    from src.tools.builtin.database import DbQueryTool
    import sqlite3

    conn = sqlite3.connect(raw_db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()
    conn.close()

    tool = DbQueryTool()
    result = await tool.execute(db_url=f"sqlite:///{raw_db_path}", query="SELECT * FROM users")
    assert result.success is True
    assert "Alice" in result.output
