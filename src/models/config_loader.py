"""
模型配置文件加载器 —— YAML 配置加载与环境变量替换。

本模块负责从 YAML 配置文件中读取模型提供商信息，并在此过程中
自动解析 `${VAR_NAME}` 或 `${VAR_NAME:default_value}` 格式的
环境变量占位符。

设计意图：
    敏感信息（如 API Key）不应硬编码在配置文件中，而应通过
    环境变量注入。本模块的 _resolve_env 函数在加载配置时
    自动完成环境变量的替换，既保证了安全性，又保持了配置文件的
    可读性。

    同时，本模块将主智能体和专家智能体的模型配置分离到
    "routing" 节点下，支持不同角色使用不同模型。
"""

import os
import re
import yaml
from pathlib import Path
from src.models.provider import ProviderConfig, ModelInfo

# 环境变量占位符的正则表达式
# 匹配格式：${VAR_NAME} 或 ${VAR_NAME:default_value}
# 例如 ${OPENAI_API_KEY} 或 ${OPENAI_API_KEY:sk-default}
_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: str) -> str:
    """解析字符串中的环境变量占位符。

    正则替换原理：
        正则表达式 `\$\{(\w+)(?::([^}]*))?\}` 包含三个部分：
            1. `\$\{`    —— 匹配字面量 "${"
            2. `(\w+)`    —— 捕获组1：环境变量名（字母、数字、下划线）
            3. `(?::([^}]*))?` —— 可选的非捕获组 + 捕获组2：
                - `:` 后面跟的内容是默认值
                - `([^}]*)` 匹配任意非 "}" 字符作为默认值
            4. `\}`       —— 匹配字面量 "}"

        replacer 回调函数的逻辑：
            - 如果环境变量存在（os.environ.get 返回非空），则替换为其值；
            - 如果环境变量不存在但提供了默认值，则替换为默认值；
            - 如果环境变量不存在也没有默认值，则替换为空字符串。

        示例：
            "api_key = ${MY_KEY}"       → "api_key = actual_key_value"
            "url = ${MY_URL:http://default}" → "url = http://default"
            "key = ${UNDEFINED}"        → "key = "

    参数：
        value: 可能包含 ${VAR} 或 ${VAR:default} 占位符的字符串。

    返回：
        所有占位符被替换后的字符串。
    """
    def replacer(match):
        # match.group(1) = 环境变量名
        var_name = match.group(1)
        # match.group(2) = 可选的默认值（冒号后面的部分），可能为 None
        default = match.group(2)
        # 获取环境变量值，不存在时使用默认值或空字符串
        return os.environ.get(var_name, default or "")
    # 对整个字符串执行全局替换
    return _ENV_VAR_RE.sub(replacer, value)


def load_model_config(config_path: str | Path) -> dict[str, ProviderConfig]:
    """加载 YAML 配置文件，返回以提供商名称为键的配置字典。

    加载流程：
        1. 使用 yaml.safe_load 读取 YAML 文件；
        2. 遍历 providers 节点下的每个提供商配置；
        3. 对 api_base 和 api_key 字段执行环境变量替换；
        4. 将原始字典组装为 ProviderConfig 实例。

    参数：
        config_path: YAML 配置文件的路径。

    返回：
        字典，键为提供商名称（如 "deepseek"），值为 ProviderConfig 实例。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    providers = {}
    for name, cfg in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            # api_base 和 api_key 支持环境变量替换
            api_base=_resolve_env(cfg["api_base"]),
            api_key=_resolve_env(cfg["api_key"]),
            default_model=cfg["default_model"],
            # 将原始字典列表解析为 ModelInfo 对象列表
            available_models=[
                ModelInfo(
                    name=m["name"],
                    provider=name,
                    max_tokens=m.get("max_tokens", 65536),
                    capabilities=m.get("capabilities", []),
                )
                for m in cfg.get("available_models", [])
            ],
        )
    return providers


def get_routing_config(config_path: str | Path) -> dict:
    """获取模型路由配置。

    作用说明：
        在 PyAgent 的多智能体架构中，主智能体（main_agent）和
        专家智能体（expert_agent）可以使用不同的模型。例如：
            - 主智能体使用便宜的模型（如 deepseek-chat）处理常规对话；
            - 专家智能体使用更强的模型（如 gpt-4o）处理复杂任务。

        YAML 配置文件的 routing 节点结构示例：
            routing:
              main_agent:
                provider: deepseek
              expert_agent:
                provider: openai

    参数：
        config_path: YAML 配置文件的路径。

    返回：
        routing 节点下的原始字典内容，通常包含 main_agent 和
        expert_agent 的子节点。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("routing", {})
