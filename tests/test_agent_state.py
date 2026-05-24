from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def test_agent_state_defaults():
    from src.agent.state import AgentState

    state = AgentState(
        messages=[],
        session_id="sess-1",
        user_id="user-1",
        recalled_memories=[],
        active_skill=None,
        tool_calls=[],
        loop_count=0,
        final_answer=None,
        compact_summary=None,
    )
    assert state["session_id"] == "sess-1"
    assert state["loop_count"] == 0
    assert state["final_answer"] is None
    assert state["active_skill"] is None
    assert state["compact_summary"] is None


def test_tool_call_typed_dict():
    from src.agent.state import ToolCall
    tc = ToolCall(name="read_file", args={"path": "/tmp/test.txt"}, id="call_1")
    assert tc["name"] == "read_file"
    assert tc["args"]["path"] == "/tmp/test.txt"


def test_agent_state_with_messages():
    from src.agent.state import AgentState

    state = AgentState(
        messages=[HumanMessage(content="hello")],
        session_id="sess-1",
        user_id="user-1",
        recalled_memories=[],
        active_skill=None,
        tool_calls=[],
        loop_count=0,
        final_answer=None,
        compact_summary=None,
    )
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "hello"
