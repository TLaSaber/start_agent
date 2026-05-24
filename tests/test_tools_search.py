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
        assert "未找到" in result.output


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
