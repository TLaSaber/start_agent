"""
builtin（内置工具）包的初始化文件。

这个包包含了 PyAgent 内置的所有工具实现。所谓"内置"，就是框架自带的、
开箱即用的工具，不需要用户额外安装或配置。

目前内置了以下工具：
1. 文件操作类
   - ReadFileTool:  读取文件内容（低风险）
   - WriteFileTool: 写入文件内容（中风险）
   - ListDirTool:   列出目录内容（低风险）

2. 搜索类
   - SearchFileTool: 按文件名模式搜索文件（低风险）
   - GrepContentTool: 在文件内容中搜索文本（低风险）

3. Shell 执行
   - ExecShellTool: 执行系统命令（高风险，需谨慎）

4. HTTP 请求
   - HttpRequestTool: 发送 HTTP 请求（中风险）

5. 数据库查询
   - DbQueryTool: 执行只读 SQL 查询（中风险）

当需要添加新的内置工具时，应该：
1. 创建一个新的 .py 文件
2. 在其中定义继承 BaseTool 的工具类
3. 在这个 __init__.py 中导入新工具类
4. 在服务启动时注册到 ToolRegistry 中
"""
from src.tools.builtin.file_ops import ReadFileTool, WriteFileTool, ListDirTool
from src.tools.builtin.search import SearchFileTool, GrepContentTool
from src.tools.builtin.shell import ExecShellTool
from src.tools.builtin.http import HttpRequestTool
from src.tools.builtin.database import DbQueryTool
