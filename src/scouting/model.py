"""Model data, fold-aware feature transforms and cross-validation helpers.

Main model (owner, 2026-11-14): non-B Liga 3 player-seasons of the training
seasons, without players who appear in the test season, target
subiu_individual (observed only). Anything learned from data (medians,
percentiles, scaling, C) is fitted inside each training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

TRAIN_SEASONS = ["2021/22", "2022/23"]
TEST_SEASON = "2023/24"
# Seasons with a complete label (s+1 and s+2 over). Annual cycle: each June,
# add the season from two years before (2024/25 in June 2027). Collection of
# future-career pages (build_target.py --collect, collect_club_divisions.py)
# and the final model use only these seasons.
LABELLED_SEASONS = ["2021/22", "2022/23", "2023/24"]
TARGET = "subiu_individual"
AGE_CENTER = 24.0
POSITIONS = ["GK", "DF", "MF", "FW", "unknown"]
LEVELS = ["profissional_ou_estrangeiro", "liga3_ou_pt_senior", "formacao_ou_sem_registo"]


def with_target_variant(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Copy of df whose subiu_individual / target_observed_individual are the
    sensitivity variant with this suffix (e.g. "_t50_0" for a 50.0 threshold)."""
    out = df.copy()
    out[TARGET] = df[f"{TARGET}{suffix}"]
    out["target_observed_individual"] = df[f"target_observed_individual{suffix}"]
    return out


def training_population(df: pd.DataFrame, observed_col: str = "target_observed_individual") -> pd.DataFrame:
    test_ids = set(df.loc[df["season"] == TEST_SEASON, "player_source_id"])
    out = df[df["season"].isin(TRAIN_SEASONS) & ~df["player_source_id"].isin(test_ids)
             & ~df["is_b_team"].astype(bool) & df[observed_col].astype(bool)].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


class FoldFeatures(BaseEstimator, TransformerMixin):
    """Builds the final model matrix from the raw feature table.

    - age_c, age_c2: age - 24 and its square;
    - per-90 rates below 450 minutes (NA) imputed with the training median,
      plus low_minutes;
    - goals_pct_in_position: percentile of goals per 90 within the position
      (training distribution; 'unknown' uses all positions);
    - position and previous-season level as dummies.
    `columns` selects a subset of the output (None = all).
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        ok = X["goals_per90"].notna()
        self.dist_ = {p: np.sort(X.loc[ok & X["position_model"].eq(p), "goals_per90"].to_numpy(float))
                      for p in POSITIONS}
        self.dist_["all"] = np.sort(X.loc[ok, "goals_per90"].to_numpy(float))
        pct = self._goal_pct(X)
        self.medians_ = {
            "goals_pct_in_position": float(np.nanmedian(pct)),
            "rel_gd90": float(X["rel_gd90"].astype(float).median()),
            "team_points_per_game_phase1": float(X["team_points_per_game_phase1"].astype(float).median()),
        }
        return self

    def _goal_pct(self, X):
        out = np.full(len(X), np.nan)
        for i, (p, g) in enumerate(zip(X["position_model"], X["goals_per90"])):
            if pd.isna(g):
                continue
            d = self.dist_.get(p) if p != "unknown" else self.dist_["all"]
            if d is None or len(d) < 10:
                d = self.dist_["all"]
            out[i] = np.searchsorted(d, float(g), side="right") / len(d)
        return out

    def transform(self, X):
        age_c = X["age"].astype(float) - AGE_CENTER
        f = pd.DataFrame(index=X.index)
        f["age_c"] = age_c
        f["age_c2"] = age_c ** 2
        f["pct_team_minutes"] = X["pct_team_minutes"].astype(float)
        f["team_points_per_game_phase1"] = X["team_points_per_game_phase1"].astype(float).fillna(
            self.medians_["team_points_per_game_phase1"])
        f["rel_gd90"] = X["rel_gd90"].astype(float).fillna(self.medians_["rel_gd90"])
        f["goals_pct_in_position"] = pd.Series(self._goal_pct(X), index=X.index).fillna(
            self.medians_["goals_pct_in_position"])
        f["low_minutes"] = X["low_minutes"].astype(float)
        for p in POSITIONS:
            f[f"pos_{p}"] = X["position_model"].eq(p).astype(float)
        for lv in LEVELS:
            f[f"prev_{lv}"] = X["prev_level3"].eq(lv).astype(float)
        f = f.fillna({"age_c": 0.0, "age_c2": 0.0})
        return f if self.columns is None else f[self.columns]


def precision_at_k(y, score, k):
    order = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(np.asarray(y)[order].mean())


def group_folds(groups, n_splits=5, seed=42):
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def metrics(y, score, k=50):
    return {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            f"prec_top{k}": precision_at_k(y, score, k)}
