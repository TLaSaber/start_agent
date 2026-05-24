from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    name: str
    args: dict
    id: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str
    recalled_memories: list[dict]
    active_skill: dict | None
    tool_calls: list[dict]
    loop_count: int
    final_answer: str | None
    compact_summary: str | None
