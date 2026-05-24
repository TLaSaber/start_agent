"""
文件操作工具模块。

本模块实现了三个最基本的文件操作工具：读取文件、写入文件、列出目录。
这些工具是 LLM 与文件系统交互的基础能力。

"parameters" 字段的 JSON Schema 格式说明（面向新手）：
--------------------------------------------------
每个工具的 parameters 字段使用 JSON Schema 标准来描述参数格式。LLM 会
"阅读"这个 Schema，然后生成符合要求的参数。

JSON Schema 的基本结构：
{
    "type": "object",                            # 固定值，表示参数是一个对象
    "properties": {                              # 定义对象的各个属性
        "参数名": {
            "type": "string",                    # 参数类型：string/number/boolean/array
            "description": "参数说明（给LLM看）"  # LLM 根据描述决定填什么值
        }
    },
    "required": ["必须的参数名"]                  # 哪些参数是必须的
}

为什么用 JSON Schema？
因为 LLM 原生理解 JSON 格式，JSON Schema 是业界标准的参数描述语言，
几乎所有 LLM API（OpenAI、Claude 等）都支持这种格式。
"""

from pathlib import Path
from src.tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    """
    读取文件内容工具。

    【功能】
    根据提供的文件路径，读取文件的全部内容并以文本形式返回。
    支持任何文本文件（.py、.txt、.json、.md 等），编码为 UTF-8。

    【使用场景】
    - LLM 需要查看某个文件的内容来理解代码
    - 用户要求读取配置文件、日志文件等

    【安全等级】
    risk_level = "low"：只读操作，不修改任何数据，相对安全。
    但需要注意：如果文件很大（比如上万的日志），读取的内容会消耗大量 token。

    【parameters 说明】
    - path: string, 必需。要读取的文件路径。
            可以是绝对路径（如 C:/project/file.txt）或相对路径。
    """
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
        """
        执行文件读取操作。

        实现细节：
        - 使用 pathlib.Path 来处理路径，这是 Python 3.4+ 推荐的路径操作方式
        - 使用 read_text(encoding="utf-8") 读取文本文件，明确指定 UTF-8 编码
        - try/except 捕获三种异常：文件不存在、编码错误、其他意外错误

        Args:
            path: 要读取的文件路径（由 LLM 根据 parameters 的 Schema 生成）

        Returns:
            ToolResult:
                - success=True, output=文件内容（正常情况）
                - success=False, output="", error=错误信息（出错了）

        错误处理策略：
        - FileNotFoundError: 文件不存在，给出清晰的提示
        - Exception: 捕获一切其他异常（权限不足、编码问题等），返回错误描述
          使用通用的 Exception 是为了确保任何意外情况都不会导致程序崩溃
        """
        try:
            content = Path(path).read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except FileNotFoundError:
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WriteFileTool(BaseTool):
    """
    写入/创建文件工具。

    【功能】
    将指定内容写入文件。如果文件已存在，会覆盖原有内容；
    如果文件所在的目录不存在，会自动创建目录。

    【使用场景】
    - LLM 需要创建新的代码文件或配置文件
    - 需要修改现有文件的内容

    【安全等级】
    risk_level = "medium"：可以修改文件系统，比只读操作风险高。
    可能的风险：
    - 覆盖重要的配置文件
    - 在敏感目录创建文件
    - 写入大量数据填满磁盘

    【parameters 说明】
    - path: string, 必需。要写入的文件路径。
    - content: string, 必需。要写入的文件内容。
    """
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
        """
        执行文件写入操作。

        实现细节：
        - p.parent.mkdir(parents=True, exist_ok=True)
          这行代码是关键！它在写入文件前确保父目录存在。
          - p.parent: 获取文件所在目录
          - parents=True: 如果父目录的父目录也不存在，递归创建
          - exist_ok=True: 如果目录已存在，不报错
          这避免了因为目录不存在而写入失败的问题。

        - p.write_text(content, encoding="utf-8")
          使用 write_text 写入字符串内容，指定 UTF-8 编码。
          注意：这会完全覆盖文件原有内容，不是追加！

        Args:
            path: 文件路径
            content: 要写入的内容

        Returns:
            ToolResult: 成功时返回包含路径的确认信息
        """
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"File written: {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ListDirTool(BaseTool):
    """
    列出目录内容工具。

    【功能】
    列出指定目录下的所有文件和子目录。类似命令行中的 ls 或 dir 命令。
    不会递归列出子目录的内容，只显示一级。

    【使用场景】
    - LLM 需要了解某个目录的结构
    - 用户在找某个文件但不记得完整路径
    - 浏览项目结构

    【安全等级】
    risk_level = "low"：只读操作，安全。

    【parameters 说明】
    - path: string, 必需。要列出内容的目录路径。
    """
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
        """
        执行目录列出操作。

        实现细节：
        - sorted(p.iterdir()): iterdir() 返回目录中所有条目的迭代器，
          sorted() 按名称排序，确保结果顺序一致（这对 LLM 很友好）
        - suffix = "/" if entry.is_dir() else "":
          给目录名加上 "/" 后缀，类似 ls -F 的效果，让结果更直观
        - "(empty directory)": 当目录为空时给出友好提示，而不是返回空字符串

        Args:
            path: 目录路径

        Returns:
            ToolResult: 成功时 output 是格式化的目录列表
                        每个条目占一行，目录以 "/" 结尾

        示例输出:
            file1.py
            file2.txt
            subdir/
        """
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
