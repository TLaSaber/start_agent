import asyncio
from src.tools.base import BaseTool, ToolResult


class ExecShellTool(BaseTool):
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
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[stderr]\n" + stderr.decode("utf-8", errors="replace")
            return ToolResult(success=proc.returncode == 0, output=output.strip() or "(no output)")
        except asyncio.TimeoutError:
            return ToolResult(success=False, output="", error="命令执行超时(30s)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
