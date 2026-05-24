"""
Shell 命令执行工具模块。

本模块提供了在系统 Shell 中执行命令的能力。这是整个工具系统中最强大的工具，
也是最危险的 —— 因此被标记为 "high" 风险等级。

为什么 ExecShellTool 是高风险（high risk）？
------------------------------------------
Shell 命令可以：
  1. 读取任意文件（包括密码、密钥等敏感信息）
  2. 修改或删除任意文件
  3. 执行任何程序（包括恶意软件）
  4. 访问网络
  5. 消耗系统资源（如 fork bomb）

因此，ExecShellTool 需要严格的安全控制：
  - 在 ToolRegistry 层面，通常需要用户手动确认才能执行
  - 在生产环境中，可能需要白名单机制（只允许特定命令）
  - 建议设置命令超时，防止命令一直运行不返回

asyncio.create_subprocess_shell 说明（面向新手）：
------------------------------------------------
这是 Python asyncio 库中用于创建子进程的函数。

传统做法：
    import os
    os.system("ls -l")  # 同步执行，会阻塞整个程序

异步做法（本工具使用）：
    proc = await asyncio.create_subprocess_shell(
        command,                    # 要执行的命令（字符串）
        stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
        stderr=asyncio.subprocess.PIPE,  # 捕获标准错误
    )
    stdout, stderr = await proc.communicate()  # 等待命令执行完成

这样做的好处：
  1. 不会阻塞整个程序 —— 在执行命令的同时，其他异步任务可以继续运行
  2. 可以捕获 stdout 和 stderr
  3. 可以设置超时控制
"""

import asyncio
from src.tools.base import BaseTool, ToolResult


class ExecShellTool(BaseTool):
    """
    执行系统命令工具。

    【功能】
    在操作系统的 Shell 中执行指定的命令，并返回命令的输出结果。
    支持任何操作系统命令（Windows 的 cmd 命令、Linux/Mac 的 shell 命令等）。

    【使用场景】
    - 运行 git 命令（如 git status、git log）
    - 编译代码（如 npm build、python setup.py）
    - 运行测试（如 pytest）
    - 执行系统管理任务
    注意：所有操作都在当前工作目录下执行。

    【安全等级】
    risk_level = "high"：具有完全的系统访问能力，需要严格管控。
    系统中通常应该配置安全策略，确保：
    - 只有授权的 Agent 才能调用此工具
    - 调用前需要用户确认
    - 命令有超时限制（当前为 30 秒）

    【parameters 说明】
    - command: string, 必需。要执行的 shell 命令。
              可以是任何合法的系统命令。
    """
    name = "exec_shell"
    description = "执行系统命令(高风险)。参数 command: 要执行的命令。"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 shell 命令"},
        },
        "required": ["command"],
    }
    risk_level = "high"

    async def execute(self, command: str = "", **kwargs) -> ToolResult:
        """
        执行 shell 命令。

        实现流程：
        1. 通过 asyncio.create_subprocess_shell 创建异步子进程
        2. 通信管道捕获命令的 stdout（标准输出）和 stderr（标准错误）
        3. 通过 asyncio.wait_for 设置 30 秒超时
        4. 解码输出（UTF-8），合并 stdout 和 stderr
        5. 根据返回码判断命令是否成功（0=成功，非0=失败）

        asyncio.wait_for 的作用：
        ------------------------
        wait_for(coro, timeout) 是异步超时控制的标准方法：
        - coro: 一个协程（async 函数调用）
        - timeout: 超时秒数
        如果协程在 timeout 秒内没有完成，抛出 asyncio.TimeoutError。
        这里用它来防止命令无限期运行（如启动了一个永不停机的服务）。

        proc.communicate() 的作用：
        -------------------------
        communicate() 会等待子进程结束，并一次性读取所有输出。
        它内部会处理管道缓冲区，防止死锁。
        返回 (stdout_bytes, stderr_bytes) 两个字节串。

        errors="replace" 的作用：
        -----------------------
        当解码字节时，如果遇到无法解码的字符（如二进制数据），
        用 �（U+FFFD 替换字符）代替，而不是抛出异常。
        这确保了即使命令输出包含非 UTF-8 字符，工具也不会崩溃。

        Args:
            command: 要执行的 shell 命令

        Returns:
            ToolResult:
                - success=True 当命令返回码为 0
                - success=False 当命令返回码非 0 或执行异常
                - output 包含 stdout，如果有 stderr 则追加在 [stderr] 段落后

        异常处理：
        - TimeoutError: 命令执行超过 30 秒
        - Exception: 其他所有意外错误（如命令不存在、权限不足等）
        """
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # 等待命令执行完成，最多等 30 秒
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[stderr]\n" + stderr.decode("utf-8", errors="replace")
            # proc.returncode: 子进程退出码，0 表示成功
            return ToolResult(success=proc.returncode == 0, output=output.strip() or "(no output)")
        except asyncio.TimeoutError:
            return ToolResult(success=False, output="", error="命令执行超时(30s)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
