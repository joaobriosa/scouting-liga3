"""Consistency checks on the interim tables.

Each check returns a DataFrame with one row per problem (empty = all good).
Known, explained source problems live in validation_exceptions.csv and are
removed from the results by `apply_exceptions`.

Calendar checks are scope-aware: they compare collected matches against the
calendar rounds (edition + round label) that have been collected so far, so
they work for a single round as well as for the whole season.
"""

from __future__ import annotations

import pandas as pd

from scouting import config

EXCEPTIONS_PATH = config.PROJECT_ROOT / "validation_exceptions.csv"
TABLES = ("matches", "appearances", "goal_events", "players", "calendar", "parse_failures",
          "unknown_events")


def load_tables() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(config.INTERIM_DIR / f"{name}.parquet") for name in TABLES}


# --------------------------------------------------------------------------
# Minutes and line-ups
# --------------------------------------------------------------------------
def check_team_minutes(apps: pd.DataFrame, match_length: pd.DataFrame) -> pd.DataFrame:
    """Team total must be 11 x match length (90, or 120 with extra time) minus
    minutes lost to on-pitch red cards."""
    # Teams with unknown minutes are reported by check_minutes_unknown instead.
    unknown = apps.loc[apps["minutes_unknown"], ["match_id", "team"]].drop_duplicates()
    a = apps.merge(unknown.assign(_skip=True), on=["match_id", "team"], how="left")
    a = a.merge(match_length, on="match_id", how="left")
    a = a[a["_skip"].isna()].copy()
    on_pitch_red = a["red_card"] & ~a["red_card_off_pitch"]
    a["lost"] = 0
    a.loc[on_pitch_red, "lost"] = (a.loc[on_pitch_red, "match_length"]
                                  - a.loc[on_pitch_red, "minute_out"].astype(int))
    g = a.groupby(["match_id", "team"], as_index=False).agg(
        minutes=("minutes_played", "sum"), lost=("lost", "sum"), length=("match_length", "first")
    )
    g["expected"] = 11 * g["length"] - g["lost"]
    g["diff"] = g["minutes"] - g["expected"]
    return g[g["diff"] != 0]


def check_player_minutes_range(apps: pd.DataFrame, match_length: pd.DataFrame) -> pd.DataFrame:
    apps = apps.merge(match_length, on="match_id", how="left")
    bad = (apps["minutes_played"] < 0) | (apps["minutes_played"] > apps["match_length"])
    return apps.loc[bad.fillna(False), ["match_id", "team", "player_source_id", "player_name", "minutes_played"]]


def check_minutes_unknown(apps: pd.DataFrame) -> pd.DataFrame:
    """Players whose sub/red-card minute is 'n/d' in the source."""
    return apps.loc[apps["minutes_unknown"], ["match_id", "team", "player_source_id", "player_name"]]


def check_starters(apps: pd.DataFrame) -> pd.DataFrame:
    g = apps.groupby(["match_id", "team"], as_index=False)["started"].sum()
    return g[g["started"] != 11]


# --------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------
def check_goals_vs_score(matches: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """goal_events per team (own goals credited to the benefiting team) == final score."""
    counted = goals.groupby(["match_id", "scoring_team_id"]).size()
    rows = []
    for m in matches.itertuples():
        for side in ("home", "away"):
            tid = getattr(m, f"{side}_team_id")
            expected = getattr(m, f"{side}_goals")
            got = int(counted.get((m.match_id, tid), 0))
            if pd.isna(expected) or got != expected:  # NA = result decided off the pitch
                rows.append({"match_id": m.match_id, "team": getattr(m, f"{side}_team"),
                             "score": expected, "goal_events": got})
    return pd.DataFrame(rows, columns=["match_id", "team", "score", "goal_events"])


# --------------------------------------------------------------------------
# Calendar coherence
# --------------------------------------------------------------------------
def collected_scope(matches: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Calendar rows belonging to the (edition, round) pairs collected so far."""
    rounds = calendar.loc[calendar["match_id"].isin(matches["match_id"]), ["edition_id", "phase"]]
    return calendar.merge(rounds.drop_duplicates(), on=["edition_id", "phase"])


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["match_id", "edition_id"]
    return pd.concat([
        df[cols + ["home_team_id", "home_team"]].rename(columns={"home_team_id": "team_id", "home_team": "team"}),
        df[cols + ["away_team_id", "away_team"]].rename(columns={"away_team_id": "team_id", "away_team": "team"}),
    ])


def check_calendar_format(calendar: pd.DataFrame) -> pd.DataFrame:
    """Each edition is a double round robin: every team plays 2*(n_teams-1) matches.

    Exception: a 2-team edition is a knockout tie and may be a single-match
    final (1 match) or a two-legged play-off (2 matches).
    """
    long = _team_long(calendar)
    counts = long.groupby(["edition_id", "team_id", "team"], as_index=False).size()
    n_teams = counts.groupby("edition_id")["team_id"].transform("nunique")
    counts["expected"] = 2 * (n_teams - 1)
    single_final = (n_teams == 2) & (counts["size"] == 1)
    return counts[(counts["size"] != counts["expected"]) & ~single_final]


def check_matches_per_team(matches: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Per team and edition, collected matches == calendar matches (in the collected scope)."""
    scope = collected_scope(matches, calendar)
    cal = _team_long(scope).groupby(["edition_id", "team_id", "team"], as_index=False).size()
    got = _team_long(matches).groupby(["edition_id", "team_id"]).size().rename("collected")
    out = cal.merge(got, on=["edition_id", "team_id"], how="outer")
    out["collected"] = out["collected"].fillna(0).astype(int)
    out = out.rename(columns={"size": "calendar"})
    return out[out["calendar"] != out["collected"]]


def check_missing_matches(matches: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Calendar matches in the collected scope that are not in `matches`."""
    scope = collected_scope(matches, calendar)
    missing = scope[~scope["match_id"].isin(matches["match_id"])]
    return missing[["match_id", "edition_name", "phase", "home_team", "away_team", "result"]]


def check_calendar_results(matches: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Calendar matches in scope whose result is not a plain score (e.g. 'IJ')."""
    scope = collected_scope(matches, calendar)
    return scope.loc[~scope["result_is_score"].astype(bool),
                     ["match_id", "edition_name", "phase", "home_team", "away_team", "result"]]


def check_duplicate_match_ids(matches: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """No match_id may appear twice, within or across seasons, in matches or calendar."""
    rows = []
    for name, df in (("matches", matches), ("calendar", calendar)):
        dup = df[df["match_id"].duplicated(keep=False)]
        for (mid,), g in dup.groupby(["match_id"]):
            rows.append({"table": name, "match_id": mid, "count": len(g),
                         "seasons": ", ".join(sorted(g["season"].astype(str).unique()))})
    return pd.DataFrame(rows, columns=["table", "match_id", "count", "seasons"])


def check_unknown_matches(matches: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Collected matches that are not in the season calendar."""
    return matches.loc[~matches["match_id"].isin(calendar["match_id"]),
                       ["match_id", "edition_name", "home_team", "away_team"]]


# --------------------------------------------------------------------------
# Parsing and players
# --------------------------------------------------------------------------
def check_parse_failures(failures: pd.DataFrame) -> pd.DataFrame:
    return failures


def check_players_missing(apps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Every player in appearances needs a row in players."""
    missing = ~apps["player_source_id"].isin(players["player_source_id"])
    return apps.loc[missing, ["player_source_id", "player_name"]].drop_duplicates()


def check_players_have_birth_date(players: pd.DataFrame) -> pd.DataFrame:
    return players.loc[players["birth_date"].isna(), ["player_source_id", "player_name", "profile_url"]]


CHECKS = {
    "team_minutes": lambda t: check_team_minutes(t["appearances"], t["matches"][["match_id", "match_length"]]),
    "player_minutes_range": lambda t: check_player_minutes_range(
        t["appearances"], t["matches"][["match_id", "match_length"]]),
    "minutes_unknown": lambda t: check_minutes_unknown(t["appearances"]),
    "starters": lambda t: check_starters(t["appearances"]),
    "goals_vs_score": lambda t: check_goals_vs_score(t["matches"], t["goal_events"]),
    "calendar_format": lambda t: check_calendar_format(t["calendar"]),
    "matches_per_team": lambda t: check_matches_per_team(t["matches"], t["calendar"]),
    "missing_matches": lambda t: check_missing_matches(t["matches"], t["calendar"]),
    "calendar_results": lambda t: check_calendar_results(t["matches"], t["calendar"]),
    "unknown_matches": lambda t: check_unknown_matches(t["matches"], t["calendar"]),
    "duplicate_match_ids": lambda t: check_duplicate_match_ids(t["matches"], t["calendar"]),
    "parse_failures": lambda t: check_parse_failures(t["parse_failures"]),
    "unknown_events": lambda t: t["unknown_events"],
    "players_missing": lambda t: check_players_missing(t["appearances"], t["players"]),
    "birth_date": lambda t: check_players_have_birth_date(t["players"]),
}


# --------------------------------------------------------------------------
# Explained exceptions
# --------------------------------------------------------------------------
def load_exceptions() -> pd.DataFrame:
    if not EXCEPTIONS_PATH.exists():
        return pd.DataFrame(columns=["check", "match_id", "team", "reason"])
    return pd.read_csv(EXCEPTIONS_PATH, dtype={"match_id": "Int64", "team": "string"})


def apply_exceptions(check: str, problems: pd.DataFrame, exceptions: pd.DataFrame) -> pd.DataFrame:
    """Drop problems listed in validation_exceptions.csv (blank team = whole match)."""
    exc = exceptions[exceptions["check"] == check]
    if exc.empty or problems.empty or "match_id" not in problems:
        return problems
    whole = set(exc.loc[exc["team"].isna(), "match_id"].astype(int))
    pairs = set(zip(exc.loc[exc["team"].notna(), "match_id"].astype(int),
                    exc.loc[exc["team"].notna(), "team"]))
    keep = [
        int(r.match_id) not in whole and (int(r.match_id), getattr(r, "team", None)) not in pairs
        for r in problems.itertuples()
    ]
    return problems[keep]


def run_all(tables: dict[str, pd.DataFrame] | None = None, use_exceptions: bool = True) -> dict[str, pd.DataFrame]:
    tables = tables or load_tables()
    exceptions = load_exceptions() if use_exceptions else load_exceptions().iloc[0:0]
    return {name: apply_exceptions(name, fn(tables), exceptions) for name, fn in CHECKS.items()}
