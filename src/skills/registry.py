import yaml
from pathlib import Path
from pydantic import BaseModel


class SkillDefinition(BaseModel):
    name: str
    summary: str
    description: str
    tools: list[str] = []
    constraints: list[str] = []
    risk_override: dict[str, str] | None = None


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def get_summaries(self) -> list[dict]:
        return [
            {"name": s.name, "summary": s.summary}
            for s in self._skills.values()
        ]

    @classmethod
    def load_from_yaml(cls, config_path: str | Path) -> "SkillRegistry":
        registry = cls()
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        for skill_data in raw.get("skills", []):
            skill = SkillDefinition(
                name=skill_data["name"],
                summary=skill_data["summary"],
                description=skill_data.get("description", skill_data["summary"]),
                tools=skill_data.get("tools", []),
                constraints=skill_data.get("constraints", []),
                risk_override=skill_data.get("risk_override"),
            )
            registry.register(skill)
        return registry
