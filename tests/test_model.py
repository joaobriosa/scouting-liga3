import numpy as np
import pandas as pd

from scouting.model import FoldFeatures, precision_at_k, training_population


def _X(goals, pos, rel):
    n = len(goals)
    return pd.DataFrame({"age": [24.0] * n, "pct_team_minutes": [50.0] * n,
                         "team_points_per_game_phase1": [1.5] * n, "rel_gd90": rel,
                         "goals_per90": goals, "low_minutes": [g is None for g in goals],
                         "position_model": pos, "prev_level3": ["liga3_ou_pt_senior"] * n})


def test_fold_features_learn_only_from_fit_data():
    train = _X([float(i) / 10 for i in range(10)] + [None], ["FW"] * 10 + ["unknown"],
               [0.0] * 5 + [1.0] * 5 + [None])
    ff = FoldFeatures().fit(train)
    test = _X([0.45, None, 5.0], ["FW", "MF", "FW"], [None, 2.0, 0.0])
    out = ff.transform(test)
    assert out.loc[0, "goals_pct_in_position"] == 0.5          # 5 of 10 training FW at or below 0.45
    assert out.loc[2, "goals_pct_in_position"] == 1.0
    assert out.loc[1, "goals_pct_in_position"] == ff.medians_["goals_pct_in_position"]  # NA -> train median, never 0
    assert out.loc[0, "rel_gd90"] == 0.5                         # train median of rel_gd90
    assert out.loc[1, "low_minutes"] == 1.0
    assert list(out.filter(like="pos_").loc[1]) == [0, 0, 1, 0, 0]


def test_precision_at_k():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == 0.5


def test_training_population_excludes_test_players_and_b_teams():
    df = pd.DataFrame({"player_source_id": [1, 1, 2, 3, 4], "season": ["2021/22", "2023/24", "2022/23", "2022/23", "2021/22"],
                       "is_b_team": [False, False, True, False, False],
                       "target_observed_individual": [True, True, True, True, False],
                       "subiu_individual": pd.array([0, 1, 1, 0, pd.NA], dtype="Int64")})
    assert list(training_population(df)["player_source_id"]) == [3]
