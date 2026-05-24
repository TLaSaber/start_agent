import pytest
from src.models.provider import ModelProvider, ModelInfo


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
