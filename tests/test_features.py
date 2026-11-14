import pandas as pd

from scouting import features as ft


def test_age_at_season_uses_1_january():
    age = ft.age_at_season(pd.Series(["2000-01-01"]), pd.Series(["2021/22"]))
    assert age.iloc[0] == 22.0


def test_on_pitch_goals_counts_only_while_on_pitch():
    matches = pd.DataFrame([{"match_id": 1, "season": "2021/22", "home_team_id": 10, "away_team_id": 20}])
    apps = pd.DataFrame([
        # starter, off at 60; sub on at 60
        {"match_id": 1, "team_id": 10, "player_source_id": 1, "minute_in": 0, "minute_out": 60,
         "minutes_played": 60, "unused_sub": False, "minutes_unknown": False},
        {"match_id": 1, "team_id": 10, "player_source_id": 2, "minute_in": 60, "minute_out": 90,
         "minutes_played": 30, "unused_sub": False, "minutes_unknown": False},
    ])
    goals = pd.DataFrame([{"match_id": 1, "minute": 10, "scoring_team_id": 10},
                          {"match_id": 1, "minute": 60, "scoring_team_id": 20},
                          {"match_id": 1, "minute": 80, "scoring_team_id": 10}])
    out = ft.on_pitch_goals(matches, apps, goals).set_index("player_source_id")
    assert (out.loc[1, "on_pitch_gf"], out.loc[1, "on_pitch_ga"]) == (1, 1)
    assert (out.loc[2, "on_pitch_gf"], out.loc[2, "on_pitch_ga"]) == (1, 0)
    assert out.loc[2, "on_pitch_minutes"] == 30


def test_previous_season_level():
    units = pd.DataFrame({"player_source_id": [1, 2, 3, 4], "season": ["2022/23"] * 4})
    career = pd.DataFrame([
        {"player_source_id": 1, "season": "2021/22", "club": "A", "club_source_id": 5, "country": "PT",
         "squad": None, "matches": 20},
        {"player_source_id": 2, "season": "2021/22", "club": "B", "club_source_id": 6, "country": "PT",
         "squad": "Jun.A S19", "matches": 25},
        {"player_source_id": 2, "season": "2021/22", "club": "C", "club_source_id": 7, "country": "PT",
         "squad": None, "matches": 2},   # senior row with matches wins over youth
        {"player_source_id": 3, "season": "2021/22", "club": "D", "club_source_id": 8, "country": "ES",
         "squad": None, "matches": 10},
    ])
    liga3 = pd.DataFrame({"season": ["2021/22"], "team_id": [5]})
    pt = pd.DataFrame({"future": ["2021/22"], "club_source_id": [7]})
    out = list(ft.previous_season_level(units, career, liga3, pt))
    assert out == ["Liga 3", "I Liga / Liga 2", "estrangeiro", "sem registo"]


def test_team_goal_difference_per90():
    matches = pd.DataFrame([{"match_id": 1, "season": "2021/22", "home_team_id": 10, "away_team_id": 20,
                             "match_length": 90},
                            {"match_id": 2, "season": "2021/22", "home_team_id": 20, "away_team_id": 10,
                             "match_length": 90}])
    goals = pd.DataFrame([{"match_id": 1, "minute": 10, "scoring_team_id": 10},
                          {"match_id": 1, "minute": 50, "scoring_team_id": 10},
                          {"match_id": 2, "minute": 5, "scoring_team_id": 20}])
    out = ft.team_goal_difference_per90(matches, goals).set_index("team_id")
    assert out.loc[10, "team_gd90"] == 0.5 and out.loc[20, "team_gd90"] == -0.5
