import tiktoken
from langchain_openai import ChatOpenAI
from src.models.provider import ModelProvider, ModelInfo, ProviderConfig


class OpenAICompatProvider(ModelProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._clients: dict[str, ChatOpenAI] = {}

    def get_chat_model(self, model_name: str | None = None) -> ChatOpenAI:
        name = model_name or self.config.default_model
        if name not in self._clients:
            self._clients[name] = ChatOpenAI(
                model=name,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
            )
        return self._clients[name]

    def get_available_models(self) -> list[ModelInfo]:
        return self.config.available_models

    def count_tokens(self, text: str, model: str | None = None) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
