"""Build data/processed/features.parquet and reports/feature_signal.md:
subida rate per quartile (or category) of each candidate feature.

Population (main model): non-B Liga 3 player-seasons of the training seasons
(2021/22, 2022/23), without players who appear in 2023/24 (test), with an
observed subiu_individual. The test season is not looked at.

    python scripts/feature_signal.py
"""

from __future__ import annotations

import sys

import pandas as pd

from scouting import config
from scouting.features import build_features
from scouting.model import TEST_SEASON, TRAIN_SEASONS, training_population
from scouting.target import pt_league_teams

NUMERIC = ["age", "pct_team_minutes", "start_rate", "full_match_rate", "goals_per90",
           "on_pitch_gf90", "on_pitch_ga90", "on_pitch_gd90", "rel_gd90",
           "team_points_per_game_phase1", "captain_rate"]
CATEGORICAL = ["position_model", "prev_season_level", "prev_level3", "low_minutes",
               "changed_team_midseason"]


def load():
    i, p = config.INTERIM_DIR, config.PROCESSED_DIR
    return (pd.read_parquet(p / "player_season_minutes.parquet"), pd.read_parquet(i / "matches.parquet"),
            pd.read_parquet(i / "appearances.parquet"), pd.read_parquet(i / "goal_events.parquet"),
            pd.read_parquet(i / "player_career.parquet"), pd.read_parquet(p / "target.parquet"))


def rate_table(df: pd.DataFrame, col: str, y: str = "subiu_individual") -> pd.DataFrame:
    """Rate per quartile (numeric) or category, NA as its own group, in order."""
    if col == "captain_rate":  # mostly 0: three groups instead of quartiles
        key = pd.Series("sem titularidades", index=df.index)
        key[df[col].eq(0)] = "0"
        key[df[col].gt(0) & df[col].lt(0.5)] = "]0, 50%["
        key[df[col].ge(0.5)] = ">= 50%"
        key = pd.Categorical(key, ["sem titularidades", "0", "]0, 50%[", ">= 50%"])
    elif col in NUMERIC:
        bins = pd.qcut(df[col], 4, duplicates="drop", precision=2)
        key = bins.cat.add_categories("NA").fillna("NA")
    else:
        key = df[col].astype("string").fillna("NA")
    t = df.groupby(key, observed=True, sort=True)[y].agg(n="size", positivos="sum")
    t["taxa_%"] = (100 * t["positivos"] / t["n"]).round(1)
    t.index = t.index.astype(str)
    return t.rename_axis("grupo").reset_index()


def markdown(t: pd.DataFrame) -> str:
    rows = [list(map(str, t.columns)), ["---"] * len(t.columns)] + t.astype(str).values.tolist()
    return chr(10).join("| " + " | ".join(r) + " |" for r in rows)


def main() -> int:
    ps, matches, apps, goals, career, target = load()
    feats = build_features(ps, matches, apps, goals, career, pt_league_teams())
    tcols = [c for c in target.columns
             if c.startswith(("subiu", "target_observed")) or c == "has_future_data"]
    df = feats.merge(target[["player_source_id", "season", *tcols]],
                     on=["player_source_id", "season"], validate="1:1")
    df.to_parquet(config.PROCESSED_DIR / "features.parquet", index=False)

    pop = training_population(df)
    base = 100 * pop["subiu_individual"].mean()
    lines = ["# Sinal das features candidatas", "",
             f"População: equipas não-B, {', '.join(TRAIN_SEASONS)}, sem jogadores que aparecem em "
             f"{TEST_SEASON}, subiu_individual observado. n = {len(pop)}, positivos = "
             f"{int(pop['subiu_individual'].sum())}, taxa base = {base:.1f}%.", ""]
    for col in NUMERIC + CATEGORICAL:
        t = rate_table(pop, col)
        lines += [f"## {col}", "", markdown(t), ""]
    out = config.PROJECT_ROOT / "reports" / "feature_signal.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
