import pytest
from langchain_core.messages import HumanMessage


def test_build_graph_returns_compiled_graph():
    from src.agent.graph import build_graph
    from src.tools.registry import ToolRegistry
    from src.skills.registry import SkillRegistry

    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()

    graph = build_graph(tool_registry, skill_registry)
    assert graph is not None
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "ainvoke")


def test_graph_routing_direct_answer():
    from src.agent.graph import route_after_think
    from src.agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": "这是最终答案",
        "compact_summary": None,
    }
    assert route_after_think(state) == "__end__"


def test_graph_routing_tool_calls():
    from src.agent.graph import route_after_think
    from src.agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [{"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "c1"}],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }
    assert route_after_think(state) == "act"


def test_graph_routing_max_loops():
    """max_loops is now handled in think_node, not route_after_think.
    route_after_think routes on tool_calls/final_answer only."""
    from src.agent.graph import route_after_think
    from config.settings import MAX_LOOPS
    from src.agent.state import AgentState

    state: AgentState = {
        "messages": [],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [{"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "c1"}],
        "loop_count": MAX_LOOPS + 1,
        "final_answer": None,
        "compact_summary": None,
    }
    result = route_after_think(state)
    # Without max_loops check, routes to "act" because tool_calls exist
    assert result == "act"
    assert state["final_answer"] is None
