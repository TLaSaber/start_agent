"""
OpenAI 兼容协议模型提供者实现。

设计意图：
    目前市面上绝大多数大模型 API 都兼容 OpenAI 的协议格式
    （即 /v1/chat/completions 接口 + JSON 请求/响应格式），
    包括但不限于 DeepSeek、智谱 GLM、通义千问、Anthropic 等。
    本模块正是利用这一事实，通过 LangChain 的 ChatOpenAI 类
    去对接所有兼容 OpenAI 协议的服务，从而实现"一套代码对接所有模型"。

    这种设计带来了极大的灵活性和可扩展性：
        - 切换模型只需修改 YAML 配置文件中的 api_base 和 api_key；
        - 无需为每个模型提供商编写单独的对接代码；
        - 即使未来使用不完全兼容的模型，也只需继承 ModelProvider 新增实现。
"""

import tiktoken
from langchain_openai import ChatOpenAI
from src.models.provider import ModelProvider, ModelInfo, ProviderConfig


class OpenAICompatProvider(ModelProvider):
    """基于 OpenAI 兼容协议的大模型提供者实现。

    本类通过 LangChain 的 ChatOpenAI 包装器，连接任何兼容
    OpenAI API 格式的模型服务。核心思想是：协议统一，模型可换。

    使用场景：
        - DeepSeek：api_base = "https://api.deepseek.com/v1"
        - 本地部署的 vLLM / Ollama：api_base = "http://localhost:8000/v1"
        - 各类国产大模型中转服务

    实例变量：
        config: ProviderConfig 配置实例，包含连接参数和可用模型列表。
        _clients: 客户端缓存字典，以模型名称为键、ChatOpenAI 实例为值。
                  缓存的目的在于复用已创建的客户端，避免重复初始化开销。
    """

    def __init__(self, config: ProviderConfig):
        """初始化 OpenAI 兼容提供者。

        参数：
            config: ProviderConfig 实例，必须包含 api_base、api_key、
                    default_model 等连接信息。
        """
        self.config = config
        # _clients 用作缓存池：同一个模型只创建一次 ChatOpenAI 实例
        self._clients: dict[str, ChatOpenAI] = {}

    def get_chat_model(self, model_name: str | None = None) -> ChatOpenAI:
        """获取指定名称的聊天模型实例。

        model_name 切换机制说明：
            本方法通过 model_name 参数实现了模型的动态切换。
            如果调用方传入了一个具体的模型名称（比如专家智能体需要
            使用更强/更便宜的模型），就会使用该名称创建或获取对应的
            客户端实例；如果为 None，则回退到配置文件中的 default_model。

            实例被缓存到 _clients 字典中，后续相同名称的请求直接复用。
            这意味着：
                - 首次调用会创建新实例（包括网络连接初始化）；
                - 后续调用直接返回缓存实例，零额外开销；
                - 不同模型名称对应不同的实例，互不干扰。

        参数：
            model_name: 目标模型名称。为 None 时使用配置的默认模型。

        返回：
            已配置好 api_key、base_url、temperature、max_tokens 的
            ChatOpenAI 实例。
        """
        # 如果未指定模型名称，则使用配置文件中声明的默认模型
        name = model_name or self.config.default_model
        # 缓存未命中时才创建新实例
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
        """返回本提供商下所有可用的模型信息列表。

        实际上就是返回配置文件中 available_models 节点的内容，
        在应用启动时由 config_loader 解析并注入。

        返回：
            配置中声明的 ModelInfo 列表。
        """
        return self.config.available_models

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """使用 tiktoken 库估算文本的 token 数量。

        tiktoken 是 OpenAI 开源的快速分词（tokenization）库，
        用于将文本切分为模型能理解的 token 序列。

        实现细节：
            1. 优先使用 tiktoken.encoding_for_model() 根据模型名称
               自动选择对应的编码器（encoding）；
            2. 如果模型名称不在 tiktoken 的已知列表中（例如 DeepSeek
               等非 OpenAI 模型），会抛出 KeyError；
            3. 捕获 KeyError 后回退到 "cl100k_base" 编码器，这是
               GPT-4 和 GPT-3.5-turbo 使用的编码方案，对大多数现代
               模型具有良好的兼容性。

        为什么可能不精确：
            - tiktoken 主要支持 OpenAI 模型；
            - 非 OpenAI 模型可能使用完全不同的 tokenizer；
            - 这里的估算值是一个"近似值"，主要用于上下文长度判断，
              不需要绝对精确。

        参数：
            text: 需要估算 token 数量的文本。
            model: 模型名称，用于选择对应的编码器。

        返回：
            估算的 token 数量（整数）。
        """
        try:
            # 尝试根据模型名称获取对应的编码器
            encoding = tiktoken.encoding_for_model(model or "gpt-4")
        except KeyError:
            # 模型不在 tiktoken 的已知列表中时，回退到通用编码器
            encoding = tiktoken.get_encoding("cl100k_base")
        # 将文本编码为 token 列表，统计其长度
        return len(encoding.encode(text))
