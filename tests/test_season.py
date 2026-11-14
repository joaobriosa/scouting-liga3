import pandas as pd

from scouting.season import build_player_season_minutes


def _app(match_id, pid, team_id, started, unused, minutes, captain=False, goals=0, yellow=0, red=False,
         position=None):
    return {
        "match_id": match_id, "player_source_id": pid, "team_id": team_id, "team": f"T{team_id}",
        "position": position,
        "started": started, "unused_sub": unused, "minutes_played": minutes,
        "minutes_unknown": False, "is_captain": captain, "goals": goals,
        "yellow_cards": yellow, "red_card": red,
    }


def test_player_season_aggregation():
    matches = pd.DataFrame([
        {"match_id": 1, "season": "2023/24", "home_team_id": 10, "away_team_id": 20},
        {"match_id": 2, "season": "2023/24", "home_team_id": 20, "away_team_id": 10},
    ])
    apps = pd.DataFrame([
        _app(1, 100, 10, True, False, 90, captain=True, goals=1, position="MF"),
        _app(2, 100, 10, True, False, 60, yellow=1, position="FW"),
        _app(1, 200, 10, False, True, 0),
        _app(2, 200, 10, False, False, 30, red=True),
    ])
    apps["minutes_played"] = apps["minutes_played"].astype("Int64")
    apps["is_captain"] = apps["is_captain"].astype("boolean")
    players = pd.DataFrame([
        {"player_source_id": 100, "player_name": "A", "birth_date": pd.Timestamp("2000-01-01"),
         "position_profile": "Guarda Redes"},
        {"player_source_id": 200, "player_name": "B", "birth_date": pd.Timestamp("2001-01-01"),
         "position_profile": "Extremo Esquerdo"},
    ])

    out = build_player_season_minutes(matches, apps, players).set_index("player_source_id")

    a = out.loc[100]
    assert (a.matches_played, a.starts, a.full_matches, a.captain_count, a.minutes, a.goals) == (2, 2, 1, 1, 150, 1)
    assert a.pct_team_minutes == round(100 * 150 / 180, 1)
    # 1 start as MF, 1 as FW: tie broken by minutes (90 as MF vs 60 as FW).
    # Formation position wins over the profile ("Guarda Redes").
    assert (a.position, a.position_share, a.position_starts, a.position_from_profile) == ("MF", 0.5, 2, False)
    assert not a.changed_team_midseason

    b = out.loc[200]
    assert (b.matches_played, b.starts, b.unused_sub_count, b.minutes, b.red_cards) == (1, 0, 1, 30, 1)
    # Never started -> profile fallback, flagged.
    assert (b.position, b.position_from_profile) == ("FW", True)


def test_changed_team_midseason():
    matches = pd.DataFrame([
        {"match_id": 1, "season": "2023/24", "home_team_id": 10, "away_team_id": 20},
        {"match_id": 2, "season": "2023/24", "home_team_id": 30, "away_team_id": 20},
    ])
    apps = pd.DataFrame([_app(1, 100, 10, True, False, 90), _app(2, 100, 30, True, False, 90)])
    apps["minutes_played"] = apps["minutes_played"].astype("Int64")
    apps["is_captain"] = apps["is_captain"].astype("boolean")
    players = pd.DataFrame([{"player_source_id": 100, "player_name": "A",
                             "birth_date": pd.Timestamp("2000-01-01"), "position_profile": None}])
    out = build_player_season_minutes(matches, apps, players)
    assert len(out) == 2 and out["changed_team_midseason"].all()


def test_team_context_final_phase_and_ppg():
    from scouting.season import team_context

    cal = pd.DataFrame([
        # phase 1: A beats B, B-C draw, C vs A decided off the pitch ("IJ")
        {"season": "s", "stage": "first", "series": "A", "match_id": 1, "home_team_id": 1, "away_team_id": 2, "result": "2-0"},
        {"season": "s", "stage": "first", "series": "A", "match_id": 2, "home_team_id": 2, "away_team_id": 3, "result": "1-1"},
        {"season": "s", "stage": "first", "series": "A", "match_id": 3, "home_team_id": 3, "away_team_id": 1, "result": "IJ"},
        # phase 2: A in promotion group and play-off, B promotion group, C relegation group
        {"season": "s", "stage": "second", "series": "Subida 1", "match_id": 4, "home_team_id": 1, "away_team_id": 2, "result": "0-0"},
        {"season": "s", "stage": "second", "series": "Playoff Subida", "match_id": 5, "home_team_id": 1, "away_team_id": 9, "result": "1-0"},
        {"season": "s", "stage": "second", "series": "Manutenção 1", "match_id": 6, "home_team_id": 3, "away_team_id": 8, "result": "1-0"},
    ])
    matches = pd.DataFrame([{"match_id": 3, "result_note": "C: Derrota (0-3) ..."}])
    out = team_context(cal, matches).set_index("team_id")
    assert out.loc[1, "team_final_phase"] == "play-off"
    assert out.loc[2, "team_final_phase"] == "subida"
    assert out.loc[3, "team_final_phase"] == "manutenção"
    assert out.loc[1, "team_points_per_game_phase1"] == 3.0  # win + official 0-3 win away
    assert out.loc[3, "team_points_per_game_phase1"] == 0.5  # draw + official loss


def test_red_minute_unknown_removed_from_share():
    matches = pd.DataFrame([
        {"match_id": 1, "season": "s", "home_team_id": 10, "away_team_id": 20, "match_length": 90},
        {"match_id": 2, "season": "s", "home_team_id": 10, "away_team_id": 20, "match_length": 120},
    ])
    a1 = _app(1, 100, 10, True, False, 90)
    a2 = _app(2, 100, 10, True, False, None, red=True)
    a2["minutes_unknown"] = True
    apps = pd.DataFrame([a1, a2])
    apps["red_minute_unknown"] = [False, True]
    apps["minutes_played"] = apps["minutes_played"].astype("Int64")
    apps["is_captain"] = apps["is_captain"].astype("boolean")
    players = pd.DataFrame([{"player_source_id": 100, "player_name": "A",
                             "birth_date": pd.Timestamp("2000-01-01"), "position_profile": None}])
    r = build_player_season_minutes(matches, apps, players).iloc[0]
    assert r.team_possible_minutes == 210            # 90 + 120 (extra time counts 120)
    assert r.player_possible_minutes == 90           # match 2 removed for this player
    assert (r.minutes, r.pct_team_minutes, r.red_minute_unknown_matches) == (90, 100.0, 1)


def test_sub_minute_unknown_removed_from_share():
    matches = pd.DataFrame([
        {"match_id": 1, "season": "s", "home_team_id": 10, "away_team_id": 20, "match_length": 90},
        {"match_id": 2, "season": "s", "home_team_id": 10, "away_team_id": 20, "match_length": 90},
    ])
    a1 = _app(1, 100, 10, True, False, 45)
    a2 = _app(2, 100, 10, False, False, None)   # came on at "n/d"
    a2["minutes_unknown"] = True
    apps = pd.DataFrame([a1, a2])
    apps["red_minute_unknown"] = False
    apps["minutes_played"] = apps["minutes_played"].astype("Int64")
    apps["is_captain"] = apps["is_captain"].astype("boolean")
    players = pd.DataFrame([{"player_source_id": 100, "player_name": "A",
                             "birth_date": pd.Timestamp("2000-01-01"), "position_profile": None}])
    r = build_player_season_minutes(matches, apps, players).iloc[0]
    assert (r.player_possible_minutes, r.minutes, r.pct_team_minutes) == (90, 45, 50.0)
