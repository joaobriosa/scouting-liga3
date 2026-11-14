"""Target variable: did a Liga 3 player reach Liga 2 / I Liga, or a foreign
league at least as strong as Liga Portugal 2, in the two following seasons?

Unit: player-season (Liga 3 teams of the same season summed).

Rule (owner, 2026-11-14), league minutes summed over s+1 and s+2:
  subiu = 1 if >= 450 league minutes in I Liga, Liga 2 or a foreign league
  whose Opta Power Rankings average rating is >= the rating of Liga Portugal 2
  (config/opta_leagues.toml, single snapshot 2026-11-13); otherwise 0.
  Leagues without a rating or without a mapping count as below. There are no
  exclusions for subiu (the old group B is gone).
A player-season with no career row at all in s+1 and s+2 gets subiu = 0;
has_future_data = False flags it for the sensitivity analysis.

subiu_individual: same rule, counting only clubs different from the player's
Liga 3 team(s) in season s (a B team and its first team are different teams).
promoted_with_team: one of his Liga 3 teams in s plays Liga 2 / I Liga in s+1.
A player who went up with his own team and played >= 450 league minutes there
(Liga 2 / I Liga), without reaching 450 elsewhere, gets subiu_individual = NA
and target_observed_individual = False.

League minutes are counted per club from the per-match rows of the games page
(the page ignores the club filter); youth leagues, cups and continental club
competitions are not league minutes.
Sensitivity thresholds (meta.sensitivity_thresholds in the config) give
subiu_t<x>_<y>, subiu_individual_t<x>_<y>, ... with the same rules.
"""

from __future__ import annotations

import tomllib

import pandas as pd

from scouting import config
from scouting.destinations import future_career, next_season
from scouting.parsing.zerozero import (
    club_league_minutes,
    parse_edition_teams,
    parse_player_games_summary,
)
from scouting.sources import zerozero as zz

THRESHOLD_MINUTES = 450
MIN_MATCHES_FOR_LOOKUP = 5  # 4 matches can never reach 450 minutes
OPTA_CONFIG = config.PROJECT_ROOT / "config" / "opta_leagues.toml"


def pt_league_teams() -> pd.DataFrame:
    """Teams of I Liga (tier 1) and Liga 2 (tier 2) per season, from cached
    edition pages."""
    rows = []
    for tier, seasons in zz.PT_LEAGUE_EDITIONS.items():
        for season, edition_id in seasons.items():
            path = config.RAW_DIR / "zerozero" / "edition" / f"{edition_id}.html"
            for t in parse_edition_teams(path.read_text(encoding="utf-8")):
                rows.append({"future": season, "club_source_id": t["team_id"], "pt_tier": tier})
    return pd.DataFrame(rows)


def load_opta() -> dict:
    if not OPTA_CONFIG.exists():
        raise FileNotFoundError(
            "config/opta_leagues.toml not found: copy config/opta_leagues.example.toml and fill in "
            "the league ratings yourself (see the note at the top of the example).")
    with OPTA_CONFIG.open("rb") as f:
        return tomllib.load(f)


def opta_rating(country, league, club_id, opta: dict) -> float | None:
    """Opta average rating of a destination league (None = no rating)."""
    override = opta.get("club_overrides", {}).get(str(int(club_id))) if pd.notna(club_id) else None
    key = override or (f"{country}|{league}" if isinstance(league, str) else None)
    name = opta["map"].get(key) if key else None
    return opta["ratings"].get(name) if name else None


def _page_league(r) -> str | None:
    """Main league of the row's club on its cached games page (None if no page)."""
    path = config.RAW_DIR / "zerozero" / f"{games_cache_key(r)}.html"
    if not path.exists():
        return None
    per_club = club_league_minutes(path.read_text(encoding="utf-8"), int(r.club_source_id),
                                   getattr(r, "club", None))
    return max(per_club, key=per_club.get) if per_club else None


def destination_rows(player_seasons: pd.DataFrame, career: pd.DataFrame,
                     foreign: pd.DataFrame, opta: dict | None = None) -> pd.DataFrame:
    """Career rows in s+1/s+2 with the destination league and its Opta rating.

    dest: 'PT' (Liga 2 / I Liga), 'F' (foreign league with a rating >= the
    lowest sensitivity threshold, so its minutes are needed) or None.
    League of a foreign row: team page (when collected), else the club's main
    league on the cached games page, else a hand-set club override."""
    opta = opta or load_opta()
    fut = future_career(player_seasons, career)
    fut = fut.merge(pt_league_teams(), on=["future", "club_source_id"], how="left")
    lg = foreign[["player_source_id", "season", "future", "club_source_id", "league"]].drop_duplicates(
        ["player_source_id", "season", "future", "club_source_id"])
    fut = fut.merge(lg, on=["player_source_id", "season", "future", "club_source_id"], how="left")
    foreign_row = fut["pt_tier"].isna() & fut["country"].ne("PT") & fut["matches"].gt(0)
    fut["league_source"] = None
    fut.loc[foreign_row & fut["league"].notna(), "league_source"] = "team_page"
    need_page = foreign_row & fut["league"].isna()
    fut.loc[need_page, "league"] = [_page_league(r) for r in fut[need_page].itertuples()]
    fut.loc[need_page & fut["league"].notna(), "league_source"] = "games_page"
    fut["opta_rating"] = [
        opta_rating(c, lg_, cid, opta) if f else None
        for c, lg_, cid, f in zip(fut["country"], fut["league"], fut["club_source_id"], foreign_row)
    ]
    fut["opta_rating"] = fut["opta_rating"].astype(float)
    lowest = min([opta["meta"]["threshold"], *opta["meta"]["sensitivity_thresholds"]])
    fut["dest"] = None
    fut.loc[fut["pt_tier"].notna(), "dest"] = "PT"
    fut.loc[fut["dest"].isna() & fut["opta_rating"].ge(lowest), "dest"] = "F"
    return fut


def rows_needing_minutes(dest: pd.DataFrame) -> pd.DataFrame:
    """Rows whose league minutes are needed: PT and rated foreign rows of
    player-seasons with >= 5 such matches."""
    d = dest[dest["matches"].gt(0) & dest["dest"].isin(["PT", "F"])].copy()
    tot = d.groupby(["player_source_id", "season"])["matches"].transform("sum")
    return d[tot >= MIN_MATCHES_FOR_LOOKUP]


def games_cache_key(r) -> str:
    return f"player_games/{int(r.player_source_id)}_{int(r.season_source_id)}_{int(r.club_source_id)}"


def league_minutes(r) -> int | None:
    """League minutes at the row's club from its cached games page (None if
    the page is not cached).

    The games page ignores the club filter and lists every match of the
    player in that epoca, so minutes are taken from the per-match rows of
    this club only. Fallback to the per-competition summary (all clubs) only
    when the page has no per-match rows or they cannot be matched (rare)."""
    path = config.RAW_DIR / "zerozero" / f"{games_cache_key(r)}.html"
    if not path.exists():
        return None
    html = path.read_text(encoding="utf-8")
    per_club = club_league_minutes(html, int(r.club_source_id), getattr(r, "club", None))
    if per_club is not None:
        return sum(per_club.values())
    rows = parse_player_games_summary(html)
    return sum(x["minutes"] for x in rows if x["is_league"])


def threshold_suffix(t: float) -> str:
    return "t" + f"{t:.1f}".replace(".", "_")


def target_thresholds(opta: dict) -> list[tuple[float, str]]:
    """(threshold, column suffix): main threshold first, with no suffix."""
    main_t = opta["meta"]["threshold"]
    return [(main_t, "")] + [(t, "_" + threshold_suffix(t)) for t in opta["meta"]["sensitivity_thresholds"]]


def build_target(player_season_minutes: pd.DataFrame, dest: pd.DataFrame,
                 opta: dict | None = None) -> pd.DataFrame:
    """One row per Liga 3 player-season with subiu, subiu_individual, etc."""
    opta = opta or load_opta()
    ps = player_season_minutes
    unit = ps.groupby(["player_source_id", "season"], as_index=False).agg(
        player_name=("player_name", "first"),
        liga3_team_ids=("team_id", lambda s: sorted(set(s))),
        changed_team_midseason=("changed_team_midseason", "max"),
        is_b_team=("is_b_team", "max"),  # played for a B team in Liga 3 that season
        minutes_liga3=("minutes", "sum"),
    )

    need = rows_needing_minutes(dest)
    need = need.assign(league_minutes=[league_minutes(r) for r in need.itertuples()])
    missing_pages = int(need["league_minutes"].isna().sum())
    teams = unit.set_index(["player_source_id", "season"])["liga3_team_ids"]
    need["other_club"] = [
        r.club_source_id not in teams.get((r.player_source_id, r.season), [])
        for r in need.itertuples()
    ]
    m = need.assign(mins=need["league_minutes"].fillna(0))
    key = ["player_source_id", "season"]
    out = unit.set_index(key)

    def agg(mask):
        return m[mask].groupby(key)["mins"].sum()

    out["minutes_pt_own_club"] = agg(m["dest"].eq("PT") & ~m["other_club"])
    for t, sfx in target_thresholds(opta):
        up = m["dest"].eq("PT") | m["opta_rating"].ge(t)
        out[f"minutes_up{sfx}"] = agg(up)
        out[f"minutes_up_other_club{sfx}"] = agg(up & m["other_club"])
    mcols = [c for c in out.columns if c.startswith("minutes_up") or c == "minutes_pt_own_club"]
    out[mcols] = out[mcols].fillna(0).astype(int)
    out = out.reset_index()

    has_future = dest.groupby(key).size().rename("n_future_rows")
    out = out.merge(has_future, on=key, how="left")
    out["n_future_rows"] = out["n_future_rows"].fillna(0).astype(int)
    out["has_future_data"] = out["n_future_rows"].gt(0)

    # promoted_with_team: a Liga 3 team of season s is in Liga 2 / I Liga in s+1.
    pt = pt_league_teams()
    pt_set = set(zip(pt["future"], pt["club_source_id"]))
    out["promoted_with_team"] = [
        any((next_season(s, 1), t) in pt_set for t in tids)
        for s, tids in zip(out["season"], out["liga3_team_ids"])
    ]
    went_up_with_team = out["promoted_with_team"] & (out["minutes_pt_own_club"] >= THRESHOLD_MINUTES)

    for t, sfx in target_thresholds(opta):
        out[f"subiu{sfx}"] = (out[f"minutes_up{sfx}"] >= THRESHOLD_MINUTES).astype("Int64")
        ind = out[f"minutes_up_other_club{sfx}"] >= THRESHOLD_MINUTES
        excluded = ~ind & went_up_with_team
        out[f"subiu_individual{sfx}"] = ind.astype("Int64").mask(excluded)
        out[f"target_observed_individual{sfx}"] = ~excluded
    out["target_observed"] = True
    out["went_up_with_own_team"] = out["subiu_individual"].isna()
    out.attrs["missing_minutes_pages"] = missing_pages
    return out
