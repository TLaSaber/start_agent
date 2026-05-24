from abc import ABC, abstractmethod
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: str
    error: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    risk_level: str = "low"  # low | medium | high | critical

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...
