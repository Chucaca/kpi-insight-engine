import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kpi_insight.driver_tree import load_tree  # noqa: E402
from kpi_insight.synthetic import load  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Never call a real LLM from tests."""
    monkeypatch.setenv("KPI_LLM_PROVIDER", "none")


@pytest.fixture(scope="session")
def df():
    return load()


@pytest.fixture(scope="session")
def tree():
    return load_tree()
