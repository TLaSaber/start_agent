"""
技能（Skills）注册表 —— 技能定义模型与注册表实现。

在 PyAgent 中，"技能"（Skill）是赋予 Agent 特定领域能力的配置单元。
每个技能定义了：
    1. 什么时候应该激活（提示词模板）；
    2. 激活后能使用哪些工具（工具白名单）；
    3. 有哪些行为限制（约束条件）。

类比理解：
    技能就像是 Agent 的"工作岗位说明书"——
    告诉 Agent 在特定场景下应该扮演什么角色、可以使用什么工具、
    遵守什么规则。
"""

import yaml
from pathlib import Path
from pydantic import BaseModel


class SkillDefinition(BaseModel):
    """技能定义数据模型。

    描述一个技能的核心要素，包含四个部分：

    1. 标识和描述（name + summary + description）：
       用于在技能注册表中识别和选择技能。

    2. 提示词模板（通过 description 字段承载）：
       当 Agent 激活某个技能时，会将该技能的描述作为系统提示词
       注入到对话上下文中，引导 Agent 的行为模式。
       例如：一个 "代码审查" 技能的 description 可能会写
       "你是一位资深的代码审查专家，请从安全性、性能、可维护性
        三个维度分析代码"。

    3. 工具白名单（tools）：
       技能激活时，Agent 只能使用白名单中列出的工具。
       这是安全控制的重要手段——限制技能的能力范围。
       例如："文件搜索" 技能可能只允许使用 SearchFileTool 和
       GrepContentTool，而不允许 ExecShellTool。

    4. 约束和行为控制（constraints + risk_override）：
       进一步限制 Agent 在技能下的行为，例如：
           - "不允许执行任何写操作"
           - "每次回复前必须引用来源"
       risk_override 可以覆盖工具默认的风险等级设置（高级功能）。

    属性说明：
        name: 技能的唯一名称，用于在注册表中查找和引用。
        summary: 简短摘要（一句话描述），用于在技能列表中快速浏览。
        description: 详细的技能描述，作为提示词模板使用。
                     如果 YAML 中没有提供，则回退使用 summary。
        tools: 该技能允许使用的工具名称白名单。空列表表示无限制。
        constraints: 约束条件列表。例如 ["no_write_operations"]。
        risk_override: 可选的工具风险等级覆盖配置。
                       例如 {"ExecShellTool": "high"}。
                       可以临时提升或降低特定工具的安全等级。
    """

    name: str
    summary: str
    description: str
    tools: list[str] = []
    constraints: list[str] = []
    risk_override: dict[str, str] | None = None


class SkillRegistry:
    """技能注册表 —— 管理所有可用技能。

    SkillRegistry 负责两件事：
        1. 从 YAML 配置文件中加载技能定义；
        2. 提供按名称查询和获取技能摘要列表的能力。

    渐进式发现机制说明：
        本注册表实现了"两级信息展示"模式：

        get_summaries() —— "清单"级别
            只返回技能的 name 和 summary，信息量小、响应快。
            适用于在 Agent 决策是否需要激活技能时快速浏览。
            例如 Agent 可以获取所有可用技能的名称和一句话简介，
            判断当前场景是否匹配某个技能。

        get() —— "完整定义"级别
            返回完整的 SkillDefinition，包含 description（提示词模板）、
            tools（工具白名单）、constraints（约束条件）等全部信息。
            适用于 Agent 决定激活某个技能后，获取完整的配置。

        这种设计避免了在 Agent 做"是否激活技能"决策时就传输大量
        不必要的提示词文本，节省了 token 和响应时间。
    """

    def __init__(self):
        """初始化空的技能注册表。"""
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """注册一个技能到注册表中。

        如果已存在同名技能，会覆盖旧定义（后注册的生效）。

        参数：
            skill: SkillDefinition 实例。
        """
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """获取指定技能的完整定义。

        这是"完整定义"级别的查询，返回包括提示词模板、工具白名单
        等全部信息在内 SkillDefinition。

        参数：
            name: 技能名称。

        返回：
            SkillDefinition 实例，如果未找到则返回 None。
        """
        return self._skills.get(name)

    def get_summaries(self) -> list[dict]:
        """获取所有技能的摘要列表（仅名称和一句话简介）。

        这是"清单"级别的查询，适用于快速浏览场景。
        每次 Agent 对话循环中，都会通过此方法获取可用技能列表，
        判断是否需要激活某个技能。

        返回：
            字典列表，每个字典包含 "name" 和 "summary" 两个字段。
            例如：[{"name": "code_review", "summary": "代码审查专家"}]
        """
        return [
            {"name": s.name, "summary": s.summary}
            for s in self._skills.values()
        ]

    @classmethod
    def load_from_yaml(cls, config_path: str | Path) -> "SkillRegistry":
        """从 YAML 配置文件加载技能定义，创建并返回填充好的注册表。

        YAML 文件格式示例：
            ```yaml
            skills:
              - name: code_review
                summary: 代码审查
                description: 你是一位资深的代码审查专家...
                tools:
                  - ReadFileTool
                  - GrepContentTool
                constraints:
                  - no_write_operations
              - name: debug_helper
                summary: 调试助手
                description: 你是一位调试专家...
            ```

        参数：
            config_path: YAML 配置文件的路径。

        返回：
            一个已加载所有技能定义的 SkillRegistry 实例。
        """
        registry = cls()
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        for skill_data in raw.get("skills", []):
            skill = SkillDefinition(
                name=skill_data["name"],
                summary=skill_data["summary"],
                # description 可选，不提供时回退使用 summary
                description=skill_data.get("description", skill_data["summary"]),
                tools=skill_data.get("tools", []),
                constraints=skill_data.get("constraints", []),
                risk_override=skill_data.get("risk_override"),
            )
            registry.register(skill)
        return registry
