import re
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
