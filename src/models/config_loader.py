import os
import re
import yaml
from pathlib import Path
from src.models.provider import ProviderConfig, ModelInfo


_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: str) -> str:
    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default or "")
    return _ENV_VAR_RE.sub(replacer, value)


def load_model_config(config_path: str | Path) -> dict[str, ProviderConfig]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    providers = {}
    for name, cfg in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            api_base=_resolve_env(cfg["api_base"]),
            api_key=_resolve_env(cfg["api_key"]),
            default_model=cfg["default_model"],
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
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("routing", {})
