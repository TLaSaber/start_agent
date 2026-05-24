import re
from langchain_core.messages import AIMessage
from src.agent.state import AgentState


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

    messages = state.get("messages", [])
    response = await model.ainvoke(messages)
    parsed = parse_llm_response(response)
    return parsed
