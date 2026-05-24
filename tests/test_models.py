import pytest
from src.models.provider import ModelProvider, ModelInfo, ProviderConfig


def test_model_info_creation():
    info = ModelInfo(
        name="deepseek-v3",
        provider="deepseek",
        max_tokens=65536,
        capabilities=["chat", "function_calling"],
    )
    assert info.name == "deepseek-v3"
    assert info.max_tokens == 65536
    assert "function_calling" in info.capabilities


def test_model_provider_is_abstract():
    with pytest.raises(TypeError):
        ModelProvider()  # Cannot instantiate ABC


def test_openai_compat_provider_creates_chat_model():
    from src.models.openai_compat import OpenAICompatProvider

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat", "function_calling"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    model = provider.get_chat_model()
    assert model is not None
    assert model.model_name == "deepseek-v3"


def test_openai_compat_provider_switches_model():
    from src.models.openai_compat import OpenAICompatProvider

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
            ModelInfo(name="deepseek-r1", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    r1 = provider.get_chat_model("deepseek-r1")
    assert r1.model_name == "deepseek-r1"


def test_count_tokens_returns_reasonable_estimate():
    from src.models.openai_compat import OpenAICompatProvider

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    count = provider.count_tokens("Hello, world!")
    assert count > 0
    assert count < 10


def test_get_available_models():
    from src.models.openai_compat import OpenAICompatProvider

    config = ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key="test-key",
        default_model="deepseek-v3",
        available_models=[
            ModelInfo(name="deepseek-v3", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
            ModelInfo(name="deepseek-r1", provider="deepseek", max_tokens=65536, capabilities=["chat"]),
        ],
    )
    provider = OpenAICompatProvider(config)
    models = provider.get_available_models()
    assert len(models) == 2
    assert models[0].name == "deepseek-v3"
