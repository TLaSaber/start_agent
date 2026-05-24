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
