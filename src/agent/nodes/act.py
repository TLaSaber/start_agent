from langchain_core.messages import ToolMessage
from src.agent.state import AgentState
from config.settings import MAX_LOOPS


ALLOWED_RISK_LEVELS = {"low", "medium"}
BLOCKED_RISK_LEVELS = {"high", "critical"}


def check_permission(tool_name: str, risk_level: str) -> tuple[bool, str]:
    """Check if a tool is allowed to execute. Returns (allowed, reason)."""
    if risk_level in BLOCKED_RISK_LEVELS:
        if risk_level == "critical":
            return False, f"工具 '{tool_name}' 为特危操作，禁止执行"
        return False, f"工具 '{tool_name}' 为高风险操作，当前版本暂不支持。请使用其他替代工具"
    return True, ""


async def act_node(state: AgentState, config: dict = None) -> dict:
    """Act node: execute tool calls, return ToolMessages"""
    tool_calls = state.get("tool_calls", [])
    messages: list[ToolMessage] = []
    loop_count = state.get("loop_count", 0)

    tool_registry = config.get("configurable", {}).get("tool_registry") if config else None
    skill_registry = config.get("configurable", {}).get("skill_registry") if config else None
    active_skill = state.get("active_skill")

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        call_id = tc.get("id", "")

        # Skill whitelist check
        if active_skill and skill_registry:
            skill = skill_registry.get(active_skill.get("name", ""))
            if skill and tool_name not in skill.tools:
                messages.append(ToolMessage(
                    content=f"技能 '{skill.name}' 不允许使用工具 '{tool_name}'。允许的工具: {', '.join(skill.tools)}",
                    tool_call_id=call_id,
                ))
                continue

        if tool_registry:
            tool = tool_registry.get(tool_name)
            if tool is None:
                messages.append(ToolMessage(
                    content=f"错误：工具 '{tool_name}' 未注册",
                    tool_call_id=call_id,
                ))
                continue

            allowed, reason = check_permission(tool_name, tool.risk_level)
            if not allowed:
                messages.append(ToolMessage(
                    content=f"权限拒绝：{reason}",
                    tool_call_id=call_id,
                ))
                continue

            result = await tool_registry.execute(tool_name, **tool_args)
            content = result.output if result.success else f"执行失败: {result.error}"
        else:
            content = f"[ToolRegistry not available] Would execute: {tool_name}({tool_args})"

        messages.append(ToolMessage(content=content, tool_call_id=call_id))

    return {
        "messages": messages,
        "loop_count": loop_count + 1,
        "tool_calls": [],
    }
