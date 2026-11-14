"""Model features per Liga 3 player-season, using only season s and earlier.

Unit: player-season (Liga 3 teams of the same season summed), the same unit
as the target. Nothing here may look at s+1 or later.
"""

from __future__ import annotations

import pandas as pd

from scouting.destinations import next_season

MIN_MINUTES_RATE = 450  # per-90 rates are NA below this (too noisy)

# Youth / reserve squads in the career table ("Jun.A S19", "S23", "B", ...).
YOUTH_SQUAD_RE = r"(?:Jun\.|S1\d|S2\d)"


def age_at_season(birth_date: pd.Series, season: pd.Series) -> pd.Series:
    """Age in years on 1 January of the season (middle of the season)."""
    ref = pd.to_datetime((season.str[:4].astype(int) + 1).astype(str) + "-01-01")
    return ((ref - pd.to_datetime(birth_date)).dt.days / 365.25).round(2)


def on_pitch_goals(matches: pd.DataFrame, appearances: pd.DataFrame,
                   goals: pd.DataFrame) -> pd.DataFrame:
    """Goals for and against the player's team while he was on the pitch,
    per player, season and team. A goal at minute m counts if minute_in < m <=
    minute_out (minute_in = 0 for starters). Appearances with unknown minutes
    are left out of both the goals and the minutes."""
    a = appearances[~appearances["unused_sub"] & ~appearances["minutes_unknown"].astype(bool)]
    a = a[["match_id", "team_id", "player_source_id", "minute_in", "minute_out", "minutes_played"]]
    a = a.merge(matches[["match_id", "season", "home_team_id", "away_team_id"]], on="match_id")
    g = goals[["match_id", "minute", "scoring_team_id"]].dropna(subset=["minute"])
    x = a.merge(g, on="match_id", how="left")
    on = (x["minute"] > x["minute_in"]) & (x["minute"] <= x["minute_out"])
    x["gf"] = (on & (x["scoring_team_id"] == x["team_id"])).astype(int)
    x["ga"] = (on & x["scoring_team_id"].notna() & (x["scoring_team_id"] != x["team_id"])).astype(int)
    per_app = x.groupby(["player_source_id", "season", "team_id", "match_id"], as_index=False).agg(
        gf=("gf", "sum"), ga=("ga", "sum"), mins=("minutes_played", "first"))
    return per_app.groupby(["player_source_id", "season", "team_id"], as_index=False).agg(
        on_pitch_gf=("gf", "sum"), on_pitch_ga=("ga", "sum"), on_pitch_minutes=("mins", "sum"))


def team_goal_difference_per90(matches: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """Team goal difference per 90 minutes over the collected matches of the
    season (goals with a known minute, as for the on-pitch counts)."""
    m = matches[["match_id", "season", "home_team_id", "away_team_id"]].copy()
    m["match_length"] = matches["match_length"] if "match_length" in matches else 90
    long = pd.concat([m.rename(columns={"home_team_id": "team_id"}).drop(columns="away_team_id"),
                      m.rename(columns={"away_team_id": "team_id"}).drop(columns="home_team_id")])
    g = goals.dropna(subset=["minute"]).merge(long[["match_id", "season", "team_id"]], on="match_id")
    g["gd"] = (g["scoring_team_id"] == g["team_id"]).astype(int) * 2 - 1
    gd = g.groupby(["season", "team_id"])["gd"].sum()
    mins = long.groupby(["season", "team_id"])["match_length"].sum()
    out = pd.DataFrame({"team_gd": gd, "team_minutes": mins}).fillna({"team_gd": 0})
    out["team_gd90"] = 90 * out["team_gd"] / out["team_minutes"]
    return out.rename_axis(["season", "team_id"]).reset_index()


LEVEL3 = {
    "I Liga / Liga 2": "profissional_ou_estrangeiro",
    "estrangeiro": "profissional_ou_estrangeiro",
    "Liga 3": "liga3_ou_pt_senior",
    "PT sénior outra": "liga3_ou_pt_senior",
    "formação": "formacao_ou_sem_registo",
    "sem registo": "formacao_ou_sem_registo",
}


def previous_season_level(units: pd.DataFrame, career: pd.DataFrame,
                          liga3_teams: pd.DataFrame, pt_teams: pd.DataFrame) -> pd.Series:
    """Level of the player in season s-1, from the (cached) career table.

    Categories: 'sem registo' (no row in s-1), 'formação' (only youth / U23
    squads), 'estrangeiro', 'I Liga / Liga 2', 'Liga 3', 'PT sénior outra'
    (senior Portuguese club below Liga 3, or division unknown). The row with
    the most matches decides; a youth row only wins if there is no senior row
    with matches."""
    prev_of = {s: _prev(s) for s in units["season"].unique()}
    u = units[["player_source_id", "season"]].copy()
    u["prev"] = u["season"].map(prev_of)
    c = career.rename(columns={"season": "prev"})
    x = u.merge(c, on=["player_source_id", "prev"], how="left")
    x["youth"] = x["squad"].fillna("").str.contains(YOUTH_SQUAD_RE)
    x["senior_played"] = ~x["youth"] & x["matches"].fillna(0).gt(0)
    l3 = set(zip(liga3_teams["season"], liga3_teams["team_id"]))
    pt = set(zip(pt_teams["future"], pt_teams["club_source_id"]))

    def level(r):
        if pd.isna(r.club):
            return "sem registo"
        if r.youth:
            return "formação"
        if r.country != "PT":
            return "estrangeiro"
        if (r.prev, r.club_source_id) in pt:
            return "I Liga / Liga 2"
        if (r.prev, r.club_source_id) in l3:
            return "Liga 3"
        return "PT sénior outra"

    x["level"] = [level(r) for r in x.itertuples()]
    # Prefer senior rows with matches, then most matches.
    x = x.sort_values(["senior_played", "matches"], ascending=False, na_position="last")
    x = x.drop_duplicates(["player_source_id", "season"])
    return u.merge(x[["player_source_id", "season", "level"]], on=["player_source_id", "season"])["level"]


def _prev(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}/{str(y + 1)[-2:]}"


def build_features(ps: pd.DataFrame, matches: pd.DataFrame, appearances: pd.DataFrame,
                   goals: pd.DataFrame, career: pd.DataFrame, pt_teams: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season with the candidate features."""
    assert next_season("2021/22", 1) == "2022/23"  # same season format as the target
    ps = ps.assign(
        # +1 so that a team still counts when the player had 0 minutes there.
        team_ppg_x_min=ps["team_points_per_game_phase1"].astype(float) * (ps["minutes"] + 1),
        w=ps["minutes"] + 1,
    )
    u = ps.groupby(["player_source_id", "season"], as_index=False).agg(
        birth_date=("birth_date", "first"),
        position=("position", "first"),
        position_from_profile=("position_from_profile", "first"),
        is_b_team=("is_b_team", "max"),
        changed_team_midseason=("changed_team_midseason", "max"),
        minutes=("minutes", "sum"),
        player_possible_minutes=("player_possible_minutes", "sum"),
        team_matches=("team_matches", "sum"),
        matches_played=("matches_played", "sum"),
        starts=("starts", "sum"),
        full_matches=("full_matches", "sum"),
        captain_count=("captain_count", "sum"),
        goals=("goals", "sum"),
        team_ppg_x_min=("team_ppg_x_min", "sum"),
        w=("w", "sum"),
    )
    u["age"] = age_at_season(u["birth_date"], u["season"])
    u["pct_team_minutes"] = (100 * u["minutes"] / u["player_possible_minutes"]).round(1)
    u["start_rate"] = (u["starts"] / u["team_matches"]).round(3)
    u["full_match_rate"] = (u["full_matches"] / u["team_matches"]).round(3)
    u["captain_rate"] = (u["captain_count"] / u["starts"].where(u["starts"] > 0)).round(3)
    enough = u["minutes"] >= MIN_MINUTES_RATE
    u["goals_per90"] = (90 * u["goals"] / u["minutes"]).where(enough).round(3)
    # Team strength, weighted by the player's minutes if he played for two teams.
    u["team_points_per_game_phase1"] = (u["team_ppg_x_min"] / u["w"]).round(3)
    # rel_gd90: on-pitch goal difference per 90 minus the team's goal
    # difference per 90 in the season (weighted by on-pitch minutes per team).
    og = on_pitch_goals(matches, appearances, goals).merge(
        team_goal_difference_per90(matches, goals)[["season", "team_id", "team_gd90"]],
        on=["season", "team_id"], how="left")
    og["team_gd90_x_min"] = og["team_gd90"] * og["on_pitch_minutes"]
    og = og.groupby(["player_source_id", "season"], as_index=False).agg(
        on_pitch_gf=("on_pitch_gf", "sum"), on_pitch_ga=("on_pitch_ga", "sum"),
        on_pitch_minutes=("on_pitch_minutes", "sum"), team_gd90_x_min=("team_gd90_x_min", "sum"))
    u = u.merge(og, on=["player_source_id", "season"], how="left")
    ok = u["on_pitch_minutes"].fillna(0) >= MIN_MINUTES_RATE
    u["on_pitch_gf90"] = (90 * u["on_pitch_gf"] / u["on_pitch_minutes"]).where(ok).round(3)
    u["on_pitch_ga90"] = (90 * u["on_pitch_ga"] / u["on_pitch_minutes"]).where(ok).round(3)
    u["on_pitch_gd90"] = (u["on_pitch_gf90"] - u["on_pitch_ga90"]).round(3)
    u["team_gd90_on_pitch_weighted"] = (u["team_gd90_x_min"] / u["on_pitch_minutes"]).where(ok)
    u["rel_gd90"] = (u["on_pitch_gd90"] - u["team_gd90_on_pitch_weighted"]).round(3)
    # Below 450 minutes the per-90 rates are NA; the model imputes them with
    # the training median and uses this indicator (never 0).
    u["low_minutes"] = u["minutes"] < MIN_MINUTES_RATE

    # Position from the season's starting XIs; no starts -> "unknown" (the
    # profile is the current one, so it is not used for non-starters).
    # Exception: 2021/22 match sheets have no tactical pitch at all, so its
    # starters take the profile line (agrees with the formation line in 90%
    # of the 2022/23-2023/24 starters) and are flagged position_from_profile_starter.
    starter = u["starts"].gt(0)
    from_pitch = ~u["position_from_profile"].astype(bool) & u["position"].notna()
    u["position_from_profile_starter"] = starter & ~from_pitch & u["position"].notna()
    u["position_model"] = u["position"].where(starter).fillna("unknown")

    liga3 = ps[["season", "team_id"]].drop_duplicates()
    u["prev_season_level"] = previous_season_level(u, career, liga3, pt_teams).to_numpy()
    u["prev_level3"] = u["prev_season_level"].map(LEVEL3)
    return u.drop(columns=["team_ppg_x_min", "w", "team_gd90_x_min"])
