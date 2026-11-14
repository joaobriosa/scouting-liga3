import pytest


@pytest.fixture(autouse=True)
def _online_by_default(monkeypatch):
    """Unit tests use mock transports: SCOUTING_OFFLINE from the shell must not
    leak into them (tests that need it set it themselves)."""
    monkeypatch.delenv("SCOUTING_OFFLINE", raising=False)
