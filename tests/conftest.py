import pytest
from pathlib import Path
import tempfile
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent.resolve()
