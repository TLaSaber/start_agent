import pytest
import os
import tempfile
from pathlib import Path


@pytest.mark.asyncio
async def test_full_agent_loop_without_llm():
    """Simulate full agent loop without LLM: verify StateGraph topology executes correctly."""
    from src.agent.graph import build_graph
    from src.agent.state import AgentState
    from src.tools.registry import ToolRegistry
    from src.skills.registry import SkillRegistry
    from langchain_core.messages import HumanMessage

    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()

    graph = build_graph(tool_registry, skill_registry)

    initial_state: AgentState = {
        "messages": [HumanMessage(content="test")],
        "session_id": "test-session",
        "user_id": "test-user",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }

    config = {"configurable": {}}
    result = await graph.ainvoke(initial_state, config)
    assert "final_answer" in result or "messages" in result


@pytest.mark.asyncio
async def test_observe_injects_system_prompt():
    """Verify Observe node injects System Prompt."""
    from src.agent.nodes.observe import observe_node
    from src.agent.state import AgentState
    from langchain_core.messages import HumanMessage, SystemMessage

    state: AgentState = {
        "messages": [HumanMessage(content="test")],
        "session_id": "sess-1",
        "user_id": "user-1",
        "recalled_memories": [],
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }

    result = await observe_node(state)
    messages = result.get("messages", state["messages"])
    assert isinstance(messages[0], SystemMessage)


def test_all_builtin_tools_registered():
    """Verify all 8 builtin tools can be registered."""
    from src.tools.registry import ToolRegistry
    from src.tools.builtin import (
        ReadFileTool, WriteFileTool, ListDirTool,
        SearchFileTool, GrepContentTool,
        ExecShellTool, HttpRequestTool, DbQueryTool,
    )

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirTool())
    registry.register(SearchFileTool())
    registry.register(GrepContentTool())
    registry.register(ExecShellTool())
    registry.register(HttpRequestTool())
    registry.register(DbQueryTool())

    all_tools = registry.list_all()
    assert len(all_tools) == 8

    llm_tools = registry.get_llm_tools()
    assert len(llm_tools) == 8
    for t in llm_tools:
        assert "type" in t
        assert t["type"] == "function"
        assert "name" in t["function"]


def test_model_config_loading():
    """Verify model YAML config can be loaded."""
    from src.models.config_loader import load_model_config, get_routing_config
    from config.settings import MODEL_CONFIG_PATH

    if not MODEL_CONFIG_PATH.exists():
        pytest.skip("Model config not found")

    providers = load_model_config(MODEL_CONFIG_PATH)
    assert "deepseek" in providers
    provider = providers["deepseek"]
    assert provider.default_model == "deepseek-v3"
    assert len(provider.available_models) >= 1

    routing = get_routing_config(MODEL_CONFIG_PATH)
    assert "main_agent" in routing


def test_skills_config_loading():
    """Verify skills YAML config can be loaded."""
    from src.skills.registry import SkillRegistry
    from config.settings import SKILLS_CONFIG_PATH

    if not SKILLS_CONFIG_PATH.exists():
        pytest.skip("Skills config not found")

    registry = SkillRegistry.load_from_yaml(str(SKILLS_CONFIG_PATH))
    summaries = registry.get_summaries()
    assert len(summaries) >= 2
    assert any(s["name"] == "code-review" for s in summaries)
