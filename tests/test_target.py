import pandas as pd

from scouting import target as tg

OPTA = {
    "meta": {"threshold": 50.0, "sensitivity_thresholds": [49.0, 51.0]},  # fictional values
    "ratings": {"Strong": 60.0, "Mid": 50.5, "Near": 49.5, "Weak": 40.0},
    "map": {"ES|Strong": "Strong", "ES|Mid": "Mid", "BG|Near": "Near", "BR|Weak": "Weak", "KR|K1": "Strong"},
    "club_overrides": {"77": "KR|K1"},
}


def _ps(pid, season, team_id, b=False):
    return {"player_source_id": pid, "season": season, "team_id": team_id, "player_name": f"P{pid}",
            "changed_team_midseason": False, "minutes": 1000, "is_b_team": b}


def _row(pid, k, club, dest, mins, rating=None, season="2021/22"):
    fut = tg.next_season(season, k)
    return {"player_source_id": pid, "season": season, "future": fut, "k": k, "club_source_id": club,
            "dest": dest, "opta_rating": rating, "matches": 10, "season_source_id": 150 + k,
            "fake_minutes": mins}


def test_opta_rating_mapping_and_override():
    assert tg.opta_rating("ES", "Mid", 1, OPTA) == 50.5
    assert tg.opta_rating("ES", "Unknown league", 1, OPTA) is None
    assert tg.opta_rating("KR", None, 77, OPTA) == 60.0      # hand-set club override


def test_build_target_opta_rule(monkeypatch):
    ps = pd.DataFrame([_ps(i, "2021/22", 10 * i) for i in range(1, 8)])
    dest = pd.DataFrame([
        _row(1, 1, 99, "PT", 300), _row(1, 2, 98, "F", 200, 60.0),   # PT + strong foreign -> 1
        _row(2, 1, 97, "F", 500, 50.5),                             # 50.5: 1 at 50.0, 0 at 51.0
        _row(3, 1, 96, "F", 500, 49.5),                             # 49.5: 0 at 50.0, 1 at 49.0
        _row(4, 1, 40, "PT", 900),                                  # up with own team: individual NA
        _row(5, 1, 95, None, 3000),                                 # weak / unrated league: 0
        _row(7, 1, 70, "PT", 200), _row(7, 2, 92, "PT", 500),       # promoted, 500 elsewhere -> individual 1
    ])
    dest["games_path"] = "x"
    mins = {(r.player_source_id, r.k, r.club_source_id): r.fake_minutes for r in dest.itertuples()}
    monkeypatch.setattr(tg, "league_minutes", lambda r: mins.get((r.player_source_id, r.k, r.club_source_id)))
    monkeypatch.setattr(tg, "pt_league_teams", lambda: pd.DataFrame(
        [{"future": "2022/23", "club_source_id": 40, "pt_tier": 2},
         {"future": "2022/23", "club_source_id": 70, "pt_tier": 2}]))

    out = tg.build_target(ps, dest.drop(columns="fake_minutes"), OPTA).set_index("player_source_id")

    assert out.loc[1, "subiu"] == 1 and out.loc[1, "subiu_individual"] == 1
    assert (out.loc[2, "subiu"], out.loc[2, "subiu_t51_0"], out.loc[2, "subiu_t49_0"]) == (1, 0, 1)
    assert (out.loc[3, "subiu"], out.loc[3, "subiu_t49_0"]) == (0, 1)
    assert out.loc[4, "subiu"] == 1 and out.loc[4, "promoted_with_team"]
    assert pd.isna(out.loc[4, "subiu_individual"]) and not out.loc[4, "target_observed_individual"]
    assert out.loc[5, "subiu"] == 0
    # 6: no future rows at all -> 0, observed (flag only for sensitivity)
    assert (out.loc[6, "subiu"], out.loc[6, "target_observed"], out.loc[6, "has_future_data"]) == (0, True, False)
    assert (out.loc[7, "subiu_individual"], out.loc[7, "target_observed_individual"]) == (1, True)
    assert out["target_observed"].all()
