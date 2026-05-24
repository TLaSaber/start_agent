"""
模型层 —— 模型提供者的抽象基类与数据模型。

本模块定义了 PyAgent 与各种大语言模型（LLM）对接时所需的
核心抽象接口和数据结构，是整个系统「模型防腐层」的基石。

设计意图：
    通过 ModelProvider 抽象基类，将上层业务逻辑（Agent 循环、
    工具调用、记忆管理等）与具体模型实现（OpenAI、DeepSeek 等）
    解耦。新增模型提供商时只需实现该接口，无需修改上层代码。
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel


class ModelInfo(BaseModel):
    """模型元信息的数据类。

    描述一个可供使用的大语言模型的基本信息，相当于模型的"身份证"。

    属性说明：
        name: 模型名称，例如 "gpt-4o"、"deepseek-chat"。
        provider: 模型提供商标识，例如 "openai"、"deepseek"，用于路由选择。
        max_tokens: 该模型支持的最大上下文 token 数（输入 + 输出）。
        capabilities: 模型能力标签列表，例如 ["chat", "function_calling"]，
                     用于在路由时判断模型是否能满足特定需求。
    """

    name: str
    provider: str
    max_tokens: int
    capabilities: list[str] = []


class ProviderConfig(BaseModel):
    """模型提供商的连接配置。

    每个 YAML 配置文件中定义的 provider 节点都会被解析为该类的实例，
    包含了连接远程模型服务所需的所有参数。

    属性说明：
        api_base: API 端点地址，例如 "https://api.openai.com/v1"。
        api_key: 认证密钥，通常通过环境变量注入（如 ${OPENAI_API_KEY}）。
        default_model: 默认使用的模型名称，当调用方没有指定模型时使用此值。
        temperature: 生成温度参数（0.0 ~ 2.0），值越高输出越随机。
                     默认 0.7 是一个在创造性和确定性之间折中的值。
        max_output_tokens: 每次生成的最大输出 token 数，默认 4096。
        available_models: 该提供商下可用的模型列表，每个元素是 ModelInfo 实例。
    """

    api_base: str
    api_key: str
    default_model: str
    temperature: float = 0.7
    max_output_tokens: int = 4096
    available_models: list[ModelInfo] = []


class ModelProvider(ABC):
    """模型提供者的抽象基类（Abstract Base Class）。

    这是整个项目的「模型防腐层」（Anti-Corruption Layer）的核心接口。
    防腐层的目的是在外部依赖（各家大模型 API）和内部核心逻辑之间
    建立一道屏障，使得外部 API 的变化不会污染内部代码。

    所有的具体模型提供商（如 OpenAI、DeepSeek 兼容接口）都必须
    实现本接口的三个方法。

    为什么需要防腐层？
        1. 各家模型 API 的参数、返回格式、认证方式各不相同；
        2. Agent 循环中需要统一调用模型，不应该感知具体实现；
        3. 切换模型提供商时只需新增一个实现类，无需修改 Agent 逻辑。
    """

    @abstractmethod
    def get_chat_model(self, model_name: str | None = None) -> BaseChatModel:
        """获取一个 LangChain 兼容的聊天模型实例。

        该方法将具体的模型 API 封装成 LangChain 的 BaseChatModel 接口，
        使得上层代码可以通过统一的 LangChain 接口调用不同的模型。

        参数：
            model_name: 可选的模型名称。如果为 None，则使用配置中的 default_model。
                        这种设计允许在运行时动态切换模型（例如主智能体和专家
                        智能体可能使用不同的模型）。

        返回：
            一个 BaseChatModel 实例（通常是 ChatOpenAI），已经配置好了
            api_base、api_key、temperature 等参数。
        """
        ...

    @abstractmethod
    def get_available_models(self) -> list[ModelInfo]:
        """列出该提供商下所有可用的模型及能力信息。

        返回的列表通常来自 YAML 配置文件中 available_models 的声明，
        用于系统在运行时做模型选择和路由决策。

        返回：
            ModelInfo 列表，描述了可用模型及其能力边界。
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """估算一段文本的 token 数量。

        Token 是大语言模型处理文本的最小单位（可以理解为"词元"）。
        估算 token 数用于：
            1. 判断上下文是否超过模型限制；
            2. 决定是否需要对历史消息做摘要压缩；
            3. 控制生成成本。

        参数：
            text: 需要估算的文本内容。
            model: 可选的目标模型名称，不同模型使用不同的 tokenizer
                   （分词器）可能导致计数差异。

        返回：
            估算的 token 数量。
        """
        ...
