import pytest
import tempfile
from pathlib import Path


def test_skill_definition_creation():
    from src.skills.registry import SkillDefinition

    skill = SkillDefinition(
        name="code-review",
        summary="审查代码变更",
        description="作为代码审查专家，按以下步骤执行...",
        tools=["read_file", "grep_content"],
        constraints=["不得修改文件"],
    )
    assert skill.name == "code-review"
    assert len(skill.tools) == 2
    assert "不得修改文件" in skill.constraints


def test_skill_registry_load_from_yaml():
    from src.skills.registry import SkillRegistry

    yaml_content = """
skills:
  - name: "test-skill"
    summary: "测试技能"
    tools: ["read_file"]
    constraints: ["只读"]
  - name: "another-skill"
    summary: "另一个技能"
    tools: ["exec_shell"]
    constraints: []
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        registry = SkillRegistry.load_from_yaml(yaml_path)
        assert len(registry.get_summaries()) == 2
        assert registry.get("test-skill") is not None
        assert registry.get("test-skill").tools == ["read_file"]
        assert registry.get("nonexistent") is None
    finally:
        Path(yaml_path).unlink(missing_ok=True)


def test_skill_registry_get_summaries():
    from src.skills.registry import SkillRegistry, SkillDefinition

    registry = SkillRegistry()
    registry.register(SkillDefinition(
        name="s1", summary="技能1", description="desc1", tools=[], constraints=[]
    ))
    registry.register(SkillDefinition(
        name="s2", summary="技能2", description="desc2", tools=[], constraints=[]
    ))

    summaries = registry.get_summaries()
    assert len(summaries) == 2
    summary_dict = {s["name"]: s["summary"] for s in summaries}
    assert summary_dict["s1"] == "技能1"
    assert summary_dict["s2"] == "技能2"


def test_skill_registry_get_full_definition():
    from src.skills.registry import SkillRegistry, SkillDefinition

    registry = SkillRegistry()
    full_desc = "详细的技能描述，包含步骤说明"
    registry.register(SkillDefinition(
        name="s1", summary="技能1", description=full_desc, tools=["t1"], constraints=["c1"]
    ))

    skill = registry.get("s1")
    assert skill is not None
    assert skill.description == full_desc
    assert skill.tools == ["t1"]
    assert skill.constraints == ["c1"]
