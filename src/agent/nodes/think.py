import asyncio
import re
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.state import AgentState
from config.settings import MAX_LOOPS, LLM_MAX_RETRIES


def parse_llm_response(response: AIMessage) -> dict:
    """Parse LLM response: distinguish between final_answer and tool_calls"""
    if response.tool_calls and len(response.tool_calls) > 0:
        return {
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"], "id": tc.get("id", "")}
                for tc in response.tool_calls
            ],
            "final_answer": None,
        }
    return {
        "tool_calls": [],
        "final_answer": response.content if isinstance(response.content, str) else str(response.content),
    }


ARCHIVE_TRIGGERS = [
    r"记住[，,：:]",
    r"保存[这那]",
    r"归档[这那]",
    r"记录下[来这]",
    r"我偏好",
    r"我习惯",
    r"我不喜欢",
    r"我总是",
    r"我的.*是",
    r"备忘[，,：:]",
    r"记一下",
    r"别忘了",
]


def detect_archive_triggers(user_message: str) -> list[dict]:
    """Detect if user message triggers memory archival"""
    triggers = []
    for pattern in ARCHIVE_TRIGGERS:
        if re.search(pattern, user_message):
            triggers.append({
                "content": user_message.strip(),
                "source": "user_command",
                "category": "preference",
            })
            break
    return triggers


async def think_node(state: AgentState, config: dict = None) -> dict:
    """Think node: call LLM reasoning and decide next action.

    Gets the model from config.configurable.model, invokes it with
    the current messages (which may include tool results), and parses
    the response into final_answer or tool_calls.
    """
    model = config.get("configurable", {}).get("model") if config else None
    if model is None:
        return {"final_answer": "Model not configured", "tool_calls": []}

    # Fix 3: Check max_loops before invoking LLM
    if state.get("loop_count", 0) >= MAX_LOOPS:
        return {
            "final_answer": "达到最大循环次数，任务中断。已完成的部分已返回。",
            "tool_calls": [],
        }

    messages = state.get("messages", [])

    # Fix 5: LLM invocation with exponential backoff retry
    last_error = None
    response = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = await model.ainvoke(messages)
            break
        except Exception as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait)
            else:
                return {
                    "final_answer": f"LLM 调用失败（已重试 {LLM_MAX_RETRIES} 次）: {str(last_error)}",
                    "tool_calls": [],
                }

    parsed = parse_llm_response(response)
    new_state = dict(parsed)

    # Fix 4: Skill progressive discovery - detect skill activation from user messages
    skill_registry = config.get("configurable", {}).get("skill_registry") if config else None
    if skill_registry:
        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        if last_human:
            for skill_summary in skill_registry.get_summaries():
                if skill_summary["name"] in last_human.lower():
                    skill = skill_registry.get(skill_summary["name"])
                    if skill:
                        new_state["active_skill"] = {
                            "name": skill.name,
                            "description": skill.description,
                            "tools": skill.tools,
                        }
                        break

    return new_state
