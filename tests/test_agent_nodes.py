import pytest
from langchain_core.messages import HumanMessage


def test_compact_estimates_tokens():
    from src.agent.compact import estimate_tokens
    count = estimate_tokens("Hello, world!")
    assert count > 0
    assert count < 10


def test_compact_not_triggered_for_short_messages():
    from src.agent.compact import should_compact
    messages = [HumanMessage(content="short message")]
    assert should_compact(messages, threshold_ratio=0.8, max_tokens=65536) is False


def test_compact_triggers_when_over_threshold():
    from src.agent.compact import should_compact
    long_text = "hello " * 100000
    messages = [HumanMessage(content=long_text)]
    assert should_compact(messages, threshold_ratio=0.1, max_tokens=65536) is True


def test_format_skill_summaries_for_prompt():
    from src.agent.nodes.observe import format_skill_summaries
    summaries = [
        {"name": "code-review", "summary": "审查代码"},
        {"name": "data-analyze", "summary": "分析数据"},
    ]
    text = format_skill_summaries(summaries)
    assert "code-review" in text
    assert "审查代码" in text
    assert "data-analyze" in text


def test_format_skill_summaries_empty():
    from src.agent.nodes.observe import format_skill_summaries
    text = format_skill_summaries([])
    assert "暂无" in text


def test_format_recalled_memories():
    from src.agent.nodes.observe import format_recalled_memories
    memories = [
        {"content": "用户偏好 tab 缩进", "category": "preference"},
    ]
    text = format_recalled_memories(memories)
    assert "tab 缩进" in text
    assert "preference" in text


def test_format_recalled_memories_empty():
    from src.agent.nodes.observe import format_recalled_memories
    text = format_recalled_memories([])
    assert "暂无" in text
