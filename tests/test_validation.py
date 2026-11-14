"""Data validation on data/interim/*.parquet (run scripts/build_minutes.py first)."""

import pytest

from scouting import config, validation

pytestmark = pytest.mark.skipif(
    not (config.INTERIM_DIR / "appearances.parquet").exists(),
    reason="no data/ (collect and build your own data first)",
)


@pytest.fixture(scope="module")
def results():
    return validation.run_all()


@pytest.mark.parametrize("check", list(validation.CHECKS))
def test_check(results, check):
    problems = results[check]
    assert problems.empty, f"{check}: {len(problems)} problems\n{problems.to_string()}"
