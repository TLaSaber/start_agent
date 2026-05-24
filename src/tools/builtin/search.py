"""
文件搜索工具模块。

本模块提供了两种文件搜索能力：
1. SearchFileTool —— 按文件名搜索（glob 模式匹配）
2. GrepContentTool —— 在文件内容中搜索（正则表达式匹配）

glob 模式与正则表达式的区别（面向新手）：
----------------------------------------
这是两种不同的"模式匹配"语言，很多新手容易混淆。

【glob 模式】（用于 SearchFileTool）
  用途：匹配文件名（路径）
  语法简单、直观，常用的通配符：
    *     匹配任意多个字符（不包括路径分隔符）
    **    匹配任意多个字符（包括路径分隔符，即递归匹配子目录）
    ?     匹配单个任意字符
    [abc] 匹配方括号中的任意一个字符
  示例：
    *.py      匹配所有 .py 文件
    test_*.py 匹配所有 test_ 开头的 .py 文件
    **/*.txt  递归匹配所有 .txt 文件

【正则表达式】（用于 GrepContentTool）
  用途：匹配文本内容（文件内部的文字）
  语法更强大但也更复杂：
    .      匹配任意单个字符
    *      匹配前一个字符 0 次或多次
    +      匹配前一个字符 1 次或多次
    ^      匹配行首
    $      匹配行尾
    \d     匹配数字
    \w     匹配字母/数字/下划线
  示例：
    import\s+os     匹配 "import os" 之类的导入语句
    def\s+\w+\(     匹配函数定义
    \berror\b       匹配完整的单词 "error"

简单记忆：glob 是"找文件名"，正则表达式是"找文件内容"。
"""

import re
from pathlib import Path
from src.tools.base import BaseTool, ToolResult


class SearchFileTool(BaseTool):
    """
    按文件名搜索工具。

    【功能】
    在指定目录中递归搜索匹配 glob 模式的文件名。
    比如搜索 *.py 会找到目录下所有 Python 文件（包括子目录中的）。

    【使用场景】
    - 查找某个配置文件在哪里
    - 统计项目中有多少 Python 文件
    - 找所有包含 "test" 的文件

    【安全等级】
    risk_level = "low"：只读操作，安全。

    【parameters 说明】
    - directory: string, 必需。搜索的根目录。
    - pattern: string, 必需。文件名 glob 匹配模式（如 *.py、*.txt）。

    实现细节：
    - base.rglob(pattern): rglob 是 pathlib 的方法，表示"递归全局匹配"。
      它会遍历 directory 下的所有子目录，逐一检查文件名是否匹配 pattern。
    - file_path.is_file(): 只返回文件，不返回目录。如果也想搜索目录名，
      可以去掉这个过滤条件。
    """
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
        """
        执行文件名搜索。

        Args:
            directory: 搜索根目录（默认当前目录）
            pattern: glob 匹配模式（默认 * 匹配所有文件）

        Returns:
            ToolResult: 匹配的文件路径列表，每行一个。
                        如果没有匹配，返回友好提示信息。
        """
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
    """
    文件内容搜索工具（类似命令行中的 grep）。

    【功能】
    在指定目录的所有文件中搜索匹配正则表达式的内容行。
    输出格式为：文件路径:行号: 匹配的行内容

    【使用场景】
    - 查找某个函数在哪些文件中被调用
    - 搜索所有包含特定关键词的日志文件
    - 查找配置项在哪里被定义

    【安全等级】
    risk_level = "low"：只读操作，安全。

    【parameters 说明】
    - directory: string, 必需。搜索的根目录。
    - pattern: string, 必需。要搜索的正则表达式。
    - file_glob: string, 可选。只搜索匹配此 glob 模式的文件（如 *.py）。
                默认搜索所有文件。

    实现细节：
    - compiled = re.compile(pattern): 预编译正则表达式。
      预编译可以提高多次匹配的效率（虽然这里只用一次）。
    - enumerate(..., 1): 从 1 开始计数，符合人类阅读习惯（行号从 1 开始）。
    - compiled.search(line): 在每行中搜索匹配。如果只是想匹配行首，
      可以用 re.match() 代替。
    - 文件读取异常用 try/except 跳过：有些文件可能是二进制文件，
      读文本会出错，我们跳过它们继续搜索其他文件。
    """
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
        """
        执行文件内容搜索。

        Args:
            directory: 搜索根目录（默认当前目录）
            pattern: 要搜索的正则表达式
            file_glob: 文件名过滤模式（默认 * 搜索所有文件）

        Returns:
            ToolResult: 匹配结果，每行格式为 "文件路径:行号: 内容"。
                        如果正则表达式非法，返回明确的错误信息。
        """
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
            # re.error: 正则表达式语法错误时抛出
            return ToolResult(success=False, output="", error=f"正则表达式错误: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
