import pytest
from pathlib import Path
import tempfile
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")


@pytest.fixture
def raw_db_path():
    """Provide a raw temp db path without schema (tests create their own schema)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    try:
        Path(f.name).unlink(missing_ok=True)
    except PermissionError:
        # Windows holds SQLite file locks after engine cleanup; ignore on teardown
        pass


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent.resolve()
