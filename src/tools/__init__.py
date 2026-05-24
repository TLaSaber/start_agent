"""
tools 模块的包初始化文件。

这个文件定义了 tools 包的公开 API。当外部代码使用
`from src.tools import ToolRegistry` 时，实际上是从这里导入的。

__init__.py 的两个作用：
1. 告诉 Python 这个目录是一个"包"（package），可以从中导入模块
2. 作为"便捷入口"，将常用的类直接暴露在包级别，避免用户需要记住
   完整的模块路径（比如 from src.tools.registry import ToolRegistry）

这里我们把最核心的三个东西导出了：
- BaseTool:  工具基类，所有工具都继承它
- ToolResult: 工具执行结果
- ToolRegistry: 工具注册中心
- ToolNotFoundError: 工具未找到异常
"""
from src.tools.base import BaseTool, ToolResult
from src.tools.registry import ToolRegistry, ToolNotFoundError
