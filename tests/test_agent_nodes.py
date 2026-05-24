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


def test_parse_llm_response_direct_answer():
    from src.agent.nodes.think import parse_llm_response
    from langchain_core.messages import AIMessage

    ai_msg = AIMessage(content="这是最终答案。", tool_calls=[])
    result = parse_llm_response(ai_msg)
    assert result["final_answer"] == "这是最终答案。"
    assert result["tool_calls"] == []


def test_parse_llm_response_tool_calls():
    from src.agent.nodes.think import parse_llm_response
    from langchain_core.messages import AIMessage

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "call_1"}],
    )
    result = parse_llm_response(ai_msg)
    assert result["final_answer"] is None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "read_file"


def test_should_archive_user_command():
    from src.agent.nodes.think import detect_archive_triggers

    triggers = detect_archive_triggers("请记住，我偏好使用 tab 缩进")
    assert len(triggers) >= 1
    assert any("tab 缩进" in t["content"] for t in triggers)


def test_should_archive_no_trigger():
    from src.agent.nodes.think import detect_archive_triggers

    triggers = detect_archive_triggers("帮我读一下 README.md 文件")
    assert len(triggers) == 0


def test_permission_check_low_risk_allowed():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("read_file", "low")
    assert allowed is True
    assert reason == ""


def test_permission_check_high_risk_blocked():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("exec_shell", "high")
    assert allowed is False
    assert "高风险" in reason


def test_permission_check_medium_allowed():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("write_file", "medium")
    assert allowed is True


def test_permission_check_critical_blocked():
    from src.agent.nodes.act import check_permission

    allowed, reason = check_permission("delete_system", "critical")
    assert allowed is False
    assert "特危" in reason
