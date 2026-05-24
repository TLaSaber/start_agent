from src.tools.base import BaseTool, ToolResult


class ToolNotFoundError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_llm_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        return await tool.execute(**kwargs)
