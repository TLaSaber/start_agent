import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += estimate_tokens(content)
    return total


def should_compact(messages: list[BaseMessage], threshold_ratio: float, max_tokens: int) -> bool:
    if not messages:
        return False
    estimated = estimate_messages_tokens(messages)
    threshold = int(max_tokens * threshold_ratio)
    return estimated > threshold


async def compact_messages(
    messages: list[BaseMessage],
    chat_model,
    keep_recent: int = 6,
) -> tuple[str, list[BaseMessage]]:
    """Compress early messages into a summary. Returns (summary_text, recent_messages)."""
    if len(messages) <= keep_recent:
        return "", messages

    boundary = len(messages) - keep_recent
    old_messages = messages[:boundary]
    recent_messages = messages[boundary:]

    old_text = "\n".join(
        f"[{m.type}]: {m.content if isinstance(m.content, str) else str(m.content)[:500]}"
        for m in old_messages
    )
    summary_prompt = (
        f"请将以下对话历史压缩为一段简洁的摘要"
        f"（保留关键决策、用户偏好和重要信息）：\n\n{old_text}"
    )

    response = await chat_model.ainvoke([
        SystemMessage(content="你是一个对话摘要助手。用中文输出简洁摘要。"),
        HumanMessage(content=summary_prompt),
    ])
    summary = response.content if isinstance(response.content, str) else str(response.content)
    return summary, recent_messages
