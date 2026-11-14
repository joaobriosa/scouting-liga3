"""Turn parsed zerozero matches into the matches / appearances / goal_events /
players tables (pandas DataFrames)."""

from __future__ import annotations

import re

import pandas as pd

from scouting.minutes import compute_minutes
from scouting.parsing.zerozero import LineupEntry, edition_stage_series, season_from_edition
from scouting.sources import zerozero as zz

COMPETITION = "Liga 3"

# Header notes next to a goal minute: "(g.p.)" = penalty, "(p.b.)" = own goal.
# Other notes are kept in goal_events.note so new notations get noticed.
PENALTY_NOTE_RE = re.compile(r"\(g\.p\.\)", re.I)
OWN_GOAL_NOTE_RE = re.compile(r"\(p\.b\.\)", re.I)


def _first(events, kind):
    ev = next((e for e in events if e.kind == kind), None)
    return (ev.minute, ev.added) if ev else (None, None)


def build_match_row(m: dict) -> dict:
    return {
        "match_id": m["match_id"],
        "season": season_from_edition(m["edition_name"]),
        "competition": COMPETITION,
        "edition_id": m["edition_id"],
        "edition_name": m["edition_name"],
        "stage": m["stage"],
        "series": m["series"],
        "round": m["round"],
        "date": pd.to_datetime(m["date"]),
        "home_team": m["home_team"],
        "home_team_id": m["home_team_id"],
        "away_team": m["away_team"],
        "away_team_id": m["away_team_id"],
        "home_goals": m["home_goals"],
        "away_goals": m["away_goals"],
        "result_code": m["result_code"],
        "result_note": m["result_note"],
        "home_penalties": m["home_penalties"],
        "away_penalties": m["away_penalties"],
        "extra_time": m["extra_time"],
        "match_length": m["match_length"],
        "home_formation": m["home_formation"],
        "away_formation": m["away_formation"],
        "source_url": zz.match_url(m["match_id"]),
    }


def build_goal_rows(m: dict) -> list[dict]:
    team = {"home": (m["home_team"], m["home_team_id"]), "away": (m["away_team"], m["away_team_id"])}
    roster = {(e.team_side, e.player_source_id) for e in m["lineups"]}
    assists = [
        (e.team_side, ev.minute, ev.added, e.player_source_id)
        for e in m["lineups"] for ev in e.events if ev.kind == "assist"
    ]
    rows = []
    for g in m["header_goals"]:
        side = g["side"]
        # Own goal: "(p.b.)" note, or credited to a side the scorer does not play for.
        own_goal = bool(OWN_GOAL_NOTE_RE.search(g["note"])) or (
            (side, g["scorer_source_id"]) not in roster
        )
        assist = next(
            (pid for s, mi, ad, pid in assists if s == side and mi == g["minute"] and ad == g["added"]),
            None,
        )
        rows.append({
            "match_id": m["match_id"],
            "minute": g["minute"],
            "added_time_minute": g["added"],
            "scoring_team": team[side][0],
            "scoring_team_id": team[side][1],
            "scorer_source_id": g["scorer_source_id"],
            "scorer_name": g["scorer_name"],
            "is_penalty": bool(PENALTY_NOTE_RE.search(g["note"])),
            "is_own_goal": own_goal,
            "assist_source_id": assist if not own_goal else None,
            "note": g["note"] or None,
            "source": "header",
        })

    # The header is the primary source (the line-up shows a single goal icon for
    # braces), but it can miss scorers, e.g. in matches decided off the pitch.
    # Goals a player has in the line-up beyond what the header credits are added
    # from the line-up, so no recorded action of a player who played is lost.
    own = {(r["scorer_source_id"], r["minute"]) for r in rows if r["is_own_goal"]}
    for e in m["lineups"]:
        lineup_goals = [
            ev for ev in e.events
            if ev.kind == "goal" and (e.player_source_id, ev.minute) not in own
        ]
        credited = sum(
            1 for r in rows if r["scorer_source_id"] == e.player_source_id and not r["is_own_goal"]
        )
        for ev in lineup_goals[credited:]:
            side = e.team_side
            rows.append({
                "match_id": m["match_id"],
                "minute": ev.minute,
                "added_time_minute": ev.added,
                "scoring_team": team[side][0],
                "scoring_team_id": team[side][1],
                "scorer_source_id": e.player_source_id,
                "scorer_name": e.player_name,
                "is_penalty": False,  # line-up icons do not mark penalties
                "is_own_goal": False,
                "assist_source_id": next(
                    (pid for s, mi, ad, pid in assists if s == side and mi == ev.minute and ad == ev.added),
                    None,
                ),
                "note": "added from line-up (missing in header)",
                "source": "lineup",
            })
    return rows


def build_appearance_row(m: dict, e: LineupEntry, goal_rows: list[dict]) -> dict:
    team_name = m["home_team"] if e.team_side == "home" else m["away_team"]
    team_id = m["home_team_id"] if e.team_side == "home" else m["away_team_id"]
    in_min, in_add = _first(e.events, "sub_in")
    out_min, out_add = _first(e.events, "sub_out")
    red_min, red_add = _first(e.events, "red")

    kinds = {ev.kind for ev in e.events}
    on_pitch = e.started or "sub_in" in kinds
    has_red = "red" in kinds
    # A substitution or on-pitch red card without a minute ("n/d") makes the
    # player's minutes unknown; we flag it instead of guessing.
    minutes_unknown = on_pitch and any(
        ev.minute is None for ev in e.events if ev.kind in ("sub_in", "sub_out", "red")
    )
    if minutes_unknown:
        minute_in = minute_out = played = None
    else:
        minute_in, minute_out, played = compute_minutes(
            e.started, in_min, out_min, red_min if on_pitch else None, m["match_length"]
        )
    # A red card only removes a player from the pitch if he was still on it:
    # not for unused subs, nor for players already substituted (e.g. off at 46',
    # red at 85' on the bench).
    red_off_pitch = has_red and (
        not on_pitch
        or (red_min is not None and out_min is not None and red_min > out_min)
    )
    # Which exit ended the stint, to keep its raw added time.
    if minute_out is None:
        out_add_final = None
    elif red_min is not None and not red_off_pitch and (out_min is None or red_min <= out_min):
        out_add_final = red_add
    else:
        out_add_final = out_add

    goals = sum(
        1 for g in goal_rows
        if g["scorer_source_id"] == e.player_source_id and not g["is_own_goal"]
    )
    return {
        "match_id": m["match_id"],
        "team": team_name,
        "team_id": team_id,
        "player_name": e.player_name,
        "player_source_id": e.player_source_id,
        "shirt_number": e.shirt_number,
        "position": e.position,
        "is_captain": e.is_captain,
        "started": e.started,
        "unused_sub": not on_pitch,
        "minute_in": minute_in,
        "minute_in_added": in_add if not e.started else None,
        "minute_out": minute_out,
        "minute_out_added": out_add_final,
        "minutes_played": played,
        "minutes_unknown": minutes_unknown,
        # Sent off while on the pitch, minute "n/d": the match is left out of
        # this player's share of team minutes (numerator and denominator).
        "red_minute_unknown": has_red and on_pitch and red_min is None,
        "goals": goals,
        # A second-yellow red counts as two yellows plus the red card.
        "yellow_cards": sum(1 for ev in e.events if ev.kind == "yellow")
        + sum(1 for ev in e.events if ev.kind == "red" and ev.second_yellow),
        "red_card": has_red,
        "red_card_second_yellow": any(ev.kind == "red" and ev.second_yellow for ev in e.events),
        "red_card_minute": red_min,
        "red_card_off_pitch": red_off_pitch,
    }


def build_tables(parsed_matches: list[dict]) -> dict[str, pd.DataFrame]:
    matches, apps, goals = [], [], []
    for m in parsed_matches:
        matches.append(build_match_row(m))
        g = build_goal_rows(m)
        goals += g
        apps += [build_appearance_row(m, e, g) for e in m["lineups"]]

    int_cols = ["shirt_number", "minute_in", "minute_in_added", "minute_out",
                "minute_out_added", "minutes_played", "red_card_minute"]
    appearances = pd.DataFrame(apps)
    for c in int_cols:
        appearances[c] = appearances[c].astype("Int64")
    # Some match sheets mark no captain at all: that is "unknown", not False.
    appearances["is_captain"] = appearances["is_captain"].astype("boolean")
    no_captain = appearances.groupby(["match_id", "team_id"])["is_captain"].transform("sum") == 0
    appearances.loc[no_captain, "is_captain"] = pd.NA
    goal_events = pd.DataFrame(goals, columns=[
        "match_id", "minute", "added_time_minute", "scoring_team", "scoring_team_id",
        "scorer_source_id", "scorer_name", "is_penalty", "is_own_goal",
        "assist_source_id", "note", "source",
    ])
    for c in ["added_time_minute", "assist_source_id"]:
        goal_events[c] = goal_events[c].astype("Int64")
    matches_df = pd.DataFrame(matches)
    for c in ["home_goals", "away_goals", "home_penalties", "away_penalties"]:
        matches_df[c] = matches_df[c].astype("Int64")
    return {
        "matches": matches_df,
        "appearances": appearances,
        "goal_events": goal_events,
    }


def build_calendar(rows: list[dict]) -> pd.DataFrame:
    """Season calendar as listed by the source (all fixtures, collected or not)."""
    # Only drop rows repeated within the same edition (pagination overlap);
    # a match_id in two editions/seasons must stay visible to validation.
    df = pd.DataFrame(rows).drop_duplicates(["match_id", "edition_id"])
    df["season"] = df["edition_name"].map(season_from_edition)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    stage_series = df["edition_name"].map(edition_stage_series)
    df["stage"] = stage_series.str[0]
    df["series"] = stage_series.str[1]
    df["result_is_score"] = df["result"].str.fullmatch(r"\d+-\d+")
    return df


def build_players(parsed_players: list[dict]) -> pd.DataFrame:
    """`position_profile` is the player's *current* position on the profile
    page; the position actually played in a season is in player_season_minutes."""
    df = pd.DataFrame(parsed_players)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["profile_url"] = df["player_source_id"].map(zz.player_url)
    df = df.rename(columns={"position": "position_profile"})
    return df[["player_source_id", "player_name", "full_name", "birth_date",
               "position_profile", "nationality", "profile_url"]]


EPOCA_ID_RE = r"epoca_id=(\d+)"


def normalize_career_seasons(df: pd.DataFrame, calendar_countries: set[str]) -> pd.DataFrame:
    """Turn zerozero year labels ("2025") into seasons ("2025/26").

    zerozero shows a year instead of a season when the player played in a
    calendar-year league in that epoca; the row's epoca_id is then the one of
    the season that ends in that year (2025 -> epoca of 2024/25). Rows of
    calendar-year countries map Y -> Y/Y+1 (owner, 2026-11-14); rows of other
    countries under a year label keep the season of their epoca_id. The original
    label stays in season_label; season_source_id stays the zerozero epoca_id
    (it addresses the games and team pages)."""
    df = df.copy()
    df["season_label"] = df["season"]
    eid = df["games_path"].str.extract(EPOCA_ID_RE)[0].astype("Int64")
    df["season_source_id"] = df["season_source_id"].astype("Int64").fillna(eid)
    std = df["season"].str.fullmatch(r"\d{4}/\d{2}")
    epoca_season = (df[std].dropna(subset=["season_source_id"])
                    .drop_duplicates("season_source_id").set_index("season_source_id")["season"])
    year = df["season"].str.fullmatch(r"\d{4}")
    cal = year & df["country"].isin(calendar_countries)
    y = df.loc[cal, "season"].astype(int)
    df.loc[cal, "season"] = y.astype(str) + "/" + ((y + 1) % 100).astype(str).str.zfill(2)
    other = year & ~cal & df["season_source_id"].isin(epoca_season.index)
    df.loc[other, "season"] = df.loc[other, "season_source_id"].map(epoca_season)
    df["calendar_year_row"] = cal
    return df


def build_player_career(rows: list[dict], calendar_countries: set[str] | None = None) -> pd.DataFrame:
    """Season-by-season career from the profile pages.

    competition, tier and minutes are not on the profile (it aggregates all
    competitions per club), so they stay NA until they are filled from
    league/edition data. matches/goals/assists cover all competitions.
    """
    df = pd.DataFrame(rows)
    df["competition"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["tier"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["minutes"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for c in ["season_source_id", "club_source_id", "matches", "goals", "assists"]:
        df[c] = df[c].astype("Int64")
    if calendar_countries is not None:
        df = normalize_career_seasons(df, calendar_countries)
    else:
        df["season_label"], df["calendar_year_row"] = df["season"], False
    return df[["player_source_id", "season", "season_label", "calendar_year_row", "club", "competition",
               "country", "tier", "matches", "minutes", "goals", "assists", "club_source_id",
               "season_source_id", "squad", "is_loan", "loan_from", "games_path"]]
