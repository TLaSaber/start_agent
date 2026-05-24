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
