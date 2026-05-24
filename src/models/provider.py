from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel


class ModelInfo(BaseModel):
    name: str
    provider: str
    max_tokens: int
    capabilities: list[str] = []


class ModelProvider(ABC):
    @abstractmethod
    def get_chat_model(self, model_name: str | None = None) -> BaseChatModel:
        """返回 LangChain 兼容的 chat model 实例"""
        ...

    @abstractmethod
    def get_available_models(self) -> list[ModelInfo]:
        """列出可用模型"""
        ...

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """估算 token 数"""
        ...
