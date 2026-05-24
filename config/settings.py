import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
DB_PATH = os.environ.get("PYAGENT_DB_PATH", str(PROJECT_ROOT / "data" / "pyagent.db"))
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"
SKILLS_CONFIG_PATH = PROJECT_ROOT / "config" / "skills.yaml"

MAX_LOOPS = int(os.environ.get("PYAGENT_MAX_LOOPS", "15"))
COMPACT_THRESHOLD_RATIO = 0.8
COMPACT_KEEP_RECENT = 6
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 3
TOOL_TIMEOUT_SECONDS = 30
MEMORY_RECALL_TOP_K = 3
AUTO_ARCHIVE_ENABLED = os.environ.get("PYAGENT_AUTO_ARCHIVE", "false").lower() == "true"
