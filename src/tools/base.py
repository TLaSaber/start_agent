"""
工具模块的基类定义。

本模块是 PyAgent 工具系统的基石，定义了所有工具必须遵守的接口规范。
工具（Tool）是 LLM 与外部世界交互的桥梁 —— LLM 通过"调用工具"来执行
代码、读写文件、搜索内容、访问网络等操作。

工具系统的核心思想：
1. 每个工具都是一个独立的类，继承自 BaseTool
2. 工具通过 JSON Schema 告诉 LLM"我接受什么参数"
3. 工具执行后返回统一的 ToolResult 格式
4. 风险等级让系统可以根据安全策略决定是否放行某个工具调用
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel


class ToolResult(BaseModel):
    """
    工具执行结果的统一数据格式。

    所有工具的 execute() 方法都必须返回 ToolResult 实例，这样上层调用者
    （如 ToolRegistry 或 Agent）就可以用统一的方式处理成功和失败情况。

    字段说明：
        success (bool): 执行是否成功。True 表示正常完成，False 表示出错。
        output (str):  执行成功时的输出内容（文本形式）。
                        失败时通常是空字符串。
        error (str | None): 执行失败时的错误信息。成功时为 None。
                            有了这个字段，调用者可以精确知道失败原因，
                            而不必从 output 中猜测。

    典型用法：
        # 成功时返回
        return ToolResult(success=True, output="文件内容...")
        # 失败时返回
        return ToolResult(success=False, output="", error="文件不存在")
    """
    success: bool          # 是否成功
    output: str            # 执行结果文本
    error: str | None = None  # 错误信息（成功时为 None）


class BaseTool(ABC):
    """
    所有工具的抽象基类。

    为什么要用 ABC（抽象基类）？
    --------------------------
    Python 的 ABC 机制强制要求子类必须实现所有标有 @abstractmethod 的方法。
    这里我们对 execute() 方法标记了 @abstractmethod，意味着：
      - 如果有人想创建一个新工具但忘了写 execute()，Python 会在实例化时报错
      - 这就像一份"契约"——所有工具都必须遵守相同的调用接口

    类属性的作用：
    -----------------
    下面定义的 name、description、parameters、risk_level 都是"类属性"，
    它们属于类本身而不是某个实例。这意味着所有该工具的实例共享这些值。
    这样设计的好处是：注册工具时可以直接访问这些信息，而不需要先创建实例。

    每个字段的含义（对新手特别重要）：
        name (str):       工具的唯一标识符。LLM 就是通过这个名字来"点名"
                          调用某个工具的，所以必须简短且语义明确。
                          例如："read_file"、"exec_shell"。

        description (str): 工具的功能描述，写给 LLM "看"的。
                           LLM 会根据这个描述来决定"当前任务该用哪个工具"。
                           描述写得越清晰，LLM 的选择就越准确。

        parameters (dict): JSON Schema 格式的参数定义。
                           这是一个特殊的字典，用 JSON Schema 标准来描述
                           工具接受哪些参数、每个参数的类型和含义。
                           LLM 会读取这个 Schema 来生成正确的参数。
                           示例：
                           {
                               "type": "object",
                               "properties": {
                                   "path": {"type": "string", "description": "文件路径"}
                               },
                               "required": ["path"]
                           }

        risk_level (str):  安全风险等级。可取值：
                           - "low"（低风险）：如读文件、搜索，通常自动放行
                           - "medium"（中风险）：如写文件、HTTP 请求，可能需要确认
                           - "high"（高风险）：如执行 shell 命令，必须人工确认
                           - "critical"（极高风险）：如删除操作，严格限制
    """
    name: str                          # 工具名称（唯一标识）
    description: str                   # 工具描述（给 LLM 看）
    parameters: dict                   # 参数定义（JSON Schema 格式）
    risk_level: str = "low"            # 风险等级（默认低风险）

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具的核心方法。

        所有子类必须实现此方法。因为这是异步方法（async def），
        所以即使工具内部有耗时的 I/O 操作（如读文件、发 HTTP 请求），
        也不会阻塞整个程序。

        Args:
            **kwargs: 关键字参数。具体接收哪些参数由子类的 parameters
                      字段定义，LLM 会根据 parameters 的 JSON Schema
                      自动填充这些参数的值。

        Returns:
            ToolResult: 统一格式的执行结果。
                        不管内部逻辑多复杂，最终都要包装成 ToolResult 返回。

        实现提示：
        - 建议用 try/except 包裹核心逻辑，确保任何异常都能被捕获
        - 成功时返回 ToolResult(success=True, output=...)
        - 失败时返回 ToolResult(success=False, output="", error=str(e))
        """
        ...
