from langchain_core.messages import SystemMessage
from src.agent.state import AgentState


SYSTEM_PROMPT_BASE = """你是一个通用的 AI 助手。你可以使用工具来完成用户的任务。

## 可用技能
{skill_summaries}

## 记忆
{recalled_memories}

## 行为准则
- 仔细分析用户的需求后再行动
- 优先使用技能（如果匹配），技能会提供详细的执行指导
- 如果无法完成，请诚实告知用户原因
"""


def format_skill_summaries(summaries: list[dict]) -> str:
    if not summaries:
        return "（暂无可用技能）"
    lines = []
    for s in summaries:
        lines.append(f"- **{s['name']}**: {s['summary']}")
    return "\n".join(lines)


def format_recalled_memories(memories: list[dict]) -> str:
    if not memories:
        return "（暂无相关记忆）"
    lines = []
    for m in memories:
        cat = m.get("category", "fact")
        lines.append(f"- [{cat}] {m['content']}")
    return "\n".join(lines)


async def observe_node(state: AgentState, config: dict = None) -> dict:
    """Observe node: load context, recall memories, inject System Prompt"""
    messages = state.get("messages", [])
    compact_summary = state.get("compact_summary")
    recalled = state.get("recalled_memories", [])

    # Get skill summaries from config
    skill_summaries = []
    if config and config.get("configurable", {}).get("skill_registry"):
        skill_summaries = config["configurable"]["skill_registry"].get_summaries()

    # Check if SystemMessage already injected
    has_system = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system:
        system_text = SYSTEM_PROMPT_BASE.format(
            skill_summaries=format_skill_summaries(skill_summaries),
            recalled_memories=format_recalled_memories(recalled),
        )
        if compact_summary:
            system_text = f"## 历史对话摘要\n{compact_summary}\n\n{system_text}"

        new_messages = [SystemMessage(content=system_text)] + list(messages)
        return {"messages": new_messages}

    return {}
