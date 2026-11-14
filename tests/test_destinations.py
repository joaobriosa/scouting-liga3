import pandas as pd
import pytest

from scouting.destinations import (
    country_rule,
    foreign_candidates,
    future_career,
    group_for,
    load_leagues_config,
    next_season,
)

CFG = load_leagues_config()


@pytest.mark.parametrize(
    "country, tier, expected",
    [
        ("ES", 1, "A"), ("ES", 2, "A"), ("ES", 3, "C"),          # big 5
        ("GB-ENG", 2, "A"),
        ("NL", 1, "A"), ("NL", 2, "B"), ("NL", 3, "C"),          # top 15, not big 5
        ("BG", 1, "A"), ("BG", 2, "C"), ("BG", 3, "C"),          # rank 16-30
        ("SI", 1, "A"), ("CY", 2, "C"),
        ("MT", 1, "C"), ("MT", 2, "C"), ("LU", None, "C"),       # rank > 30: no tier needed
        ("SA", 1, "A"), ("MX", 2, "B"), ("BR", 3, "C"),          # non-UEFA group A list
        ("ID", None, "B"), ("QA", 1, "B"),                       # other non-UEFA
        ("ES", None, None),                                      # tier needed but unknown
    ],
)
def test_group_for(country, tier, expected):
    assert group_for(country, tier, CFG) == expected


def test_country_rule_requests_only_where_needed():
    assert country_rule("ES", CFG) == "tier"
    assert country_rule("JP", CFG) == "tier"
    assert country_rule("MT", CFG) == "C"
    assert country_rule("ID", CFG) == "B"


def test_next_season():
    assert next_season("2023/24", 1) == "2024/25"
    assert next_season("2021/22", 2) == "2023/24"


def test_foreign_candidates_sum_over_two_seasons():
    ps = pd.DataFrame([{"player_source_id": 1, "season": "2021/22"},
                       {"player_source_id": 2, "season": "2021/22"}])
    career = pd.DataFrame([
        # player 1: 3 + 2 matches abroad over s+1 and s+2 -> kept (sum 5)
        {"player_source_id": 1, "season": "2022/23", "country": "ES", "matches": 3},
        {"player_source_id": 1, "season": "2023/24", "country": "FR", "matches": 2},
        # player 2: 4 matches abroad -> dropped
        {"player_source_id": 2, "season": "2022/23", "country": "ES", "matches": 4},
        {"player_source_id": 2, "season": "2023/24", "country": "PT", "matches": 30},
    ])
    out = foreign_candidates(future_career(ps, career))
    assert set(out["player_source_id"]) == {1} and len(out) == 2
