"""Keep tests independent of local case databases and import paths."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolated_investigation_store(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_DB_PATH", str(tmp_path / "investigations.sqlite3"))
