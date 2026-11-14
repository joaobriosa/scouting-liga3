"""Aggregate appearances into one row per player, team and season.

A player who changed club mid-season gets one row per team, because the
share of team minutes only makes sense within one team.
"""

from __future__ import annotations

import pandas as pd

from scouting.minutes import MATCH_LENGTH


OFFICIAL_SCORE_RE = r"\((\d+)-(\d+)\)"


def team_context(calendar: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per season and team: final phase reached and points per game in phase 1.

    Built from the calendar, so it covers the whole season even for matches
    not collected. team_final_phase: "play-off" if the team played the
    promotion play-off, else "subida" (promotion group / champion stage),
    else "manutenção". Points use 3/1/0; a result decided off the pitch
    (e.g. "IJ") uses the official score from the match sheet note, if known.
    """
    cal = calendar.copy()
    long = pd.concat([
        cal[["season", "series", "stage", "match_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        cal[["season", "series", "stage", "match_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ])

    second = long[long["stage"] == "second"].copy()
    second["phase"] = "manutenção"
    second.loc[second["series"].str.startswith(("Subida", "Apuramento")), "phase"] = "subida"
    second.loc[second["series"] == "Playoff Subida", "phase"] = "play-off"
    rank = {"manutenção": 0, "subida": 1, "play-off": 2}
    second["rank"] = second["phase"].map(rank)
    final = (second.sort_values("rank").drop_duplicates(["season", "team_id"], keep="last")
             [["season", "team_id", "phase"]].rename(columns={"phase": "team_final_phase"}))

    # Phase-1 scores: calendar result, or official score for off-pitch decisions.
    first = cal[cal["stage"] == "first"].copy()
    score = first["result"].str.extract(r"^(\d+)-(\d+)$").astype(float)
    official = (matches.set_index("match_id")["result_note"].dropna()
                .str.extract(OFFICIAL_SCORE_RE).astype(float))
    missing = score[0].isna() & first["match_id"].isin(official.index)
    score.loc[missing, [0, 1]] = official.loc[first.loc[missing, "match_id"]].to_numpy()
    first["hg"], first["ag"] = score[0], score[1]
    first = first.dropna(subset=["hg", "ag"])

    def pts(gf, ga):
        return (gf > ga) * 3 + (gf == ga) * 1

    rows = pd.concat([
        pd.DataFrame({"season": first["season"], "team_id": first["home_team_id"],
                      "points": pts(first["hg"], first["ag"])}),
        pd.DataFrame({"season": first["season"], "team_id": first["away_team_id"],
                      "points": pts(first["ag"], first["hg"])}),
    ])
    ppg = rows.groupby(["season", "team_id"], as_index=False).agg(
        team_points_per_game_phase1=("points", "mean"), team_matches_phase1=("points", "size"))
    ppg["team_points_per_game_phase1"] = ppg["team_points_per_game_phase1"].round(3)
    return ppg.merge(final, on=["season", "team_id"], how="outer")


def team_possible_minutes(matches: pd.DataFrame) -> pd.DataFrame:
    """Matches and possible minutes per team and season (match_length per
    match: 90, or 120 with extra time)."""
    m = matches.copy()
    if "match_length" not in m:
        m["match_length"] = MATCH_LENGTH
    long = pd.concat([
        m[["season", "match_id", "match_length", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        m[["season", "match_id", "match_length", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ])
    g = long.groupby(["season", "team_id"], as_index=False).agg(
        team_matches=("match_id", "nunique"), team_possible_minutes=("match_length", "sum"))
    return g


def profile_line(position: str | None) -> str | None:
    """Map a zerozero profile position ('Defesa Central', 'Extremo Esquerdo', ...)
    to GK/DF/MF/FW. Wingers go to FW, as in most formations on the match sheets."""
    if not isinstance(position, str):
        return None
    p = position.lower()
    if "guarda" in p:
        return "GK"
    if p.startswith("defesa") or p.startswith("lateral"):
        return "DF"
    if p.startswith("médio") or p.startswith("medio"):
        return "MF"
    if p.startswith(("extremo", "ponta", "avançado", "2.º avançado")) or "avançado" in p:
        return "FW"
    return None


def season_positions(matches: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    """Most frequent line (GK/DF/MF/FW) in the starting XIs of the season.

    Positions come from the tactical pitch of each match sheet, so only starts
    count. Ties are broken by minutes played in that line. Players who never
    started get no position (NA).
    """
    a = appearances.merge(matches[["match_id", "season"]], on="match_id")
    a = a[a["started"] & a["position"].notna()]
    g = a.groupby(["player_source_id", "season", "position"], as_index=False).agg(
        n=("match_id", "size"), mins=("minutes_played", "sum")
    )
    g["position_starts"] = g.groupby(["player_source_id", "season"])["n"].transform("sum")
    g = g.sort_values(["n", "mins"], ascending=False).drop_duplicates(["player_source_id", "season"])
    g["position_share"] = (g["n"] / g["position_starts"]).round(2)
    return g[["player_source_id", "season", "position", "position_share", "position_starts"]]


def build_player_season_minutes(
    matches: pd.DataFrame, appearances: pd.DataFrame, players: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cols_m = ["match_id", "season"] + (["match_length"] if "match_length" in matches else [])
    a = appearances.merge(matches[cols_m], on="match_id", how="left")
    if "match_length" not in a:
        a["match_length"] = MATCH_LENGTH
    a["played"] = ~a["unused_sub"]
    a["full_match"] = a["minutes_played"].fillna(-1).eq(a["match_length"])
    a["captain"] = a["is_captain"].fillna(False).astype(bool)
    if "red_minute_unknown" not in a:
        a["red_minute_unknown"] = False
    # Any match with unknown minutes for this player (sub or red card "n/d")
    # is removed from both sides of his share of team minutes.
    a["excluded_minutes"] = a["match_length"].where(a["minutes_unknown"].astype(bool), 0)

    g = a.groupby(["player_source_id", "team_id", "team", "season"], as_index=False).agg(
        squad_listings=("match_id", "nunique"),
        matches_played=("played", "sum"),
        starts=("started", "sum"),
        unused_sub_count=("unused_sub", "sum"),
        full_matches=("full_match", "sum"),
        captain_count=("captain", "sum"),
        minutes=("minutes_played", "sum"),
        minutes_unknown_matches=("minutes_unknown", "sum"),
        goals=("goals", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_card", "sum"),
        red_minute_unknown_matches=("red_minute_unknown", "sum"),
        excluded_minutes=("excluded_minutes", "sum"),
    )
    g = g.merge(team_possible_minutes(matches), on=["season", "team_id"], how="left")
    # Matches with unknown minutes for the player are removed from both sides
    # of the share (his minutes there are NA anyway).
    g["player_possible_minutes"] = g["team_possible_minutes"] - g["excluded_minutes"]
    g["pct_team_minutes"] = (100 * g["minutes"] / g["player_possible_minutes"]).round(1)

    g = g.merge(players[["player_source_id", "player_name", "birth_date", "position_profile"]],
                on="player_source_id", how="left")
    # Position is per player and season (across teams if he changed club).
    g = g.merge(season_positions(matches, appearances), on=["player_source_id", "season"], how="left")
    # Players who never started have no formation position: fall back to the
    # (current) profile position, mapped to the same four lines, and flag it.
    fallback = g["position"].isna()
    g["position_from_profile"] = fallback & g["position_profile"].map(profile_line).notna()
    g.loc[fallback, "position"] = g.loc[fallback, "position_profile"].map(profile_line)

    # Reserve teams of bigger clubs (SC Braga B, Sporting B, Vitória SC B).
    g["is_b_team"] = g["team"].str.match(r".*\sB$")
    g["changed_team_midseason"] = (
        g.groupby(["player_source_id", "season"])["team_id"].transform("nunique") > 1
    )
    if calendar is not None:
        g = g.merge(team_context(calendar, matches), on=["season", "team_id"], how="left")
    else:
        g["team_final_phase"] = pd.NA
        g["team_points_per_game_phase1"] = pd.NA
        g["team_matches_phase1"] = pd.NA
    cols = [
        "player_source_id", "player_name", "birth_date", "team", "team_id", "season",
        "is_b_team", "changed_team_midseason", "team_final_phase", "team_points_per_game_phase1",
        "team_matches_phase1",
        "position", "position_share", "position_starts", "position_from_profile",
        "matches_played", "starts", "unused_sub_count", "full_matches", "captain_count",
        "minutes", "goals", "yellow_cards", "red_cards", "pct_team_minutes",
        "squad_listings", "team_matches", "team_possible_minutes", "player_possible_minutes",
        "minutes_unknown_matches", "red_minute_unknown_matches",
    ]
    int_cols = ["matches_played", "starts", "unused_sub_count", "full_matches", "captain_count",
                "minutes", "goals", "yellow_cards", "red_cards", "squad_listings",
                "minutes_unknown_matches", "red_minute_unknown_matches", "player_possible_minutes"]
    g[int_cols] = g[int_cols].astype(int)
    return g[cols].sort_values(["team", "minutes"], ascending=[True, False]).reset_index(drop=True)
