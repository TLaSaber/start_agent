# tests/test_tools.py
import pytest


def test_tool_result_creation():
    from src.tools.base import ToolResult

    result = ToolResult(success=True, output="file content here")
    assert result.success is True
    assert result.error is None

    fail_result = ToolResult(success=False, output="", error="File not found")
    assert fail_result.success is False
    assert fail_result.error == "File not found"


def test_base_tool_is_abstract():
    from src.tools.base import BaseTool
    with pytest.raises(TypeError):
        BaseTool()


def test_tool_registry_register_and_get():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "A fake tool for testing"
        parameters = {"type": "object", "properties": {}, "required": []}
        risk_level = "low"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="done")

    registry = ToolRegistry()
    registry.register(FakeTool())
    assert registry.get("fake_tool") is not None
    assert registry.get("nonexistent") is None


def test_tool_registry_get_llm_tools():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class FakeTool(BaseTool):
        name = "fake_tool"
        description = "Does something fake"
        parameters = {
            "type": "object",
            "properties": {"arg1": {"type": "string", "description": "First arg"}},
            "required": ["arg1"],
        }
        risk_level = "low"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="done")

    registry = ToolRegistry()
    registry.register(FakeTool())

    tools = registry.get_llm_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "fake_tool"
    assert tools[0]["function"]["parameters"]["required"] == ["arg1"]


@pytest.mark.asyncio
async def test_tool_registry_execute():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class EchoTool(BaseTool):
        name = "echo"
        description = "Echoes input"
        parameters = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
        risk_level = "low"

        async def execute(self, message: str = "", **kwargs) -> ToolResult:
            return ToolResult(success=True, output=message)

    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await registry.execute("echo", message="hello")
    assert result.success is True
    assert result.output == "hello"


def test_tool_registry_list_all():
    from src.tools.base import BaseTool, ToolResult
    from src.tools.registry import ToolRegistry

    class ToolA(BaseTool):
        name = "tool_a"
        description = "A"
        parameters = {}
        risk_level = "low"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="A")

    class ToolB(BaseTool):
        name = "tool_b"
        description = "B"
        parameters = {}
        risk_level = "medium"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="B")

    registry = ToolRegistry()
    registry.register(ToolA())
    registry.register(ToolB())

    all_tools = registry.list_all()
    assert len(all_tools) == 2
    names = {t.name for t in all_tools}
    assert names == {"tool_a", "tool_b"}
