from typing import Literal
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.observe import observe_node
from src.agent.nodes.think import think_node
from src.agent.nodes.act import act_node


def route_after_think(state: AgentState) -> Literal["act", "__end__"]:
    if state.get("final_answer") is not None:
        return "__end__"
    if state.get("tool_calls"):
        return "act"
    return "__end__"


def build_graph(
    tool_registry=None,
    skill_registry=None,
    checkpoint_saver=None,
):
    """Build the Agent StateGraph.

    Args:
        tool_registry: ToolRegistry instance
        skill_registry: SkillRegistry instance
        checkpoint_saver: LangGraph checkpoint saver (SqliteSaver or MemorySaver)
    """
    builder = StateGraph(AgentState)

    builder.add_node("observe", observe_node)
    builder.add_node("think", think_node)
    builder.add_node("act", act_node)

    builder.set_entry_point("observe")
    builder.add_edge("observe", "think")
    builder.add_conditional_edges(
        "think",
        route_after_think,
        {"act": "act", "__end__": END},
    )
    builder.add_edge("act", "observe")

    if checkpoint_saver:
        return builder.compile(checkpointer=checkpoint_saver)
    return builder.compile()
