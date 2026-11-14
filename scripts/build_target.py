"""Build the target variable (subiu / subiu_individual) for every Liga 3
player-season. Needs build_minutes.py first.

    python scripts/build_target.py --dry-run   # count pages still to fetch
    python scripts/build_target.py --collect   # fetch team and games pages (polite, cached)
    python scripts/build_target.py --collect --teams-only
    python scripts/build_target.py             # build data/processed/target.parquet

Requests are only made with --collect: team pages of foreign clubs whose
group depends on the division, and per-player games pages (league minutes)
for career rows that can still reach 450 minutes.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from scouting import config
from scouting.destinations import (
    classify_foreign,
    country_rule,
    future_career,
    load_leagues_config,
)
from scouting.http import PageError
from scouting.model import LABELLED_SEASONS
from scouting.parsing.zerozero import parse_team_season
from scouting.sources import zerozero as zz
from scouting.target import (
    MIN_MATCHES_FOR_LOOKUP,
    THRESHOLD_MINUTES,
    league_minutes,
    build_target,
    load_opta,
    destination_rows,
    games_cache_key,
    pt_league_teams,
    rows_needing_minutes,
)

log = logging.getLogger("target")
RAW = config.RAW_DIR / "zerozero"


def load():
    ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
    career = pd.read_parquet(config.INTERIM_DIR / "player_career.parquet")
    known = career.dropna(subset=["season_source_id"]).drop_duplicates("season")
    season_ids = dict(zip(known["season"], known["season_source_id"]))
    career["season_source_id"] = career["season_source_id"].fillna(career["season"].map(season_ids))
    return ps, career


def foreign_rows(ps: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    """Foreign rows (matches > 0) of player-seasons whose foreign + Liga 2/I Liga
    matches in s+1 and s+2 sum to at least 5 (minutes are summed across them)."""
    fut = future_career(ps, career)
    fut = fut.merge(pt_league_teams(), on=["future", "club_source_id"], how="left")
    rel = fut[fut["matches"].gt(0) & (fut["country"].ne("PT") | fut["pt_tier"].notna())]
    tot = rel.groupby(["player_source_id", "season"])["matches"].transform("sum")
    rel = rel[tot >= MIN_MATCHES_FOR_LOOKUP]
    return rel[rel["country"].ne("PT")].copy()


def team_page_path(club_id, season_id):
    return RAW / "team" / f"{int(club_id)}_{int(season_id)}.html"


def classified_foreign(ps, career, cfg):
    fr = foreign_rows(ps, career)
    pages = {}
    for r in fr.drop_duplicates(["club_source_id", "season_source_id"]).itertuples():
        p = team_page_path(r.club_source_id, r.season_source_id)
        if country_rule(r.country, cfg) == "tier" and p.exists():
            pages[(r.club_source_id, r.season_source_id)] = parse_team_season(p.read_text(encoding="utf-8"))
    return fr, classify_foreign(fr, pages, cfg)


def missing_team_pages(fr, cfg):
    need = fr[fr["country"].map(lambda c: country_rule(c, cfg)) == "tier"]
    need = need.drop_duplicates(["club_source_id", "season_source_id"])
    return need[[not team_page_path(r.club_source_id, r.season_source_id).exists()
                 for r in need.itertuples()]]


def missing_games_pages(need):
    return need[[not (RAW / f"{games_cache_key(r)}.html").exists() for r in need.itertuples()]]


def _fetch(client, rows: pd.DataFrame, label: str) -> None:
    todo = missing_games_pages(rows)
    log.info("%s: %d pages (%d already cached)", label, len(todo), len(rows) - len(todo))
    for i, r in enumerate(todo.itertuples(), 1):
        if not isinstance(r.games_path, str):
            log.warning("No games link for player %s (%s %s)", r.player_source_id, r.club, r.future)
            continue
        try:
            client.get(zz.BASE_URL + r.games_path, games_cache_key(r))
        except PageError as exc:
            log.warning("Skipping games page %s: %s", r.games_path, exc)
        if i % 25 == 0:
            log.info("%s %d/%d", label, i, len(todo))


def _minutes_so_far(rows: pd.DataFrame, liga3_teams: pd.Series, strict: float) -> pd.DataFrame:
    """League minutes per player-season from cached pages, counting only rows
    that count at the strictest threshold: total and at clubs other than the
    player's Liga 3 team(s)."""
    r = rows[rows["dest"].eq("PT") | rows["opta_rating"].ge(strict)]
    r = r.assign(mins=[league_minutes(x) or 0 for x in r.itertuples()])
    r["other_club"] = [x.club_source_id not in liga3_teams.get((x.player_source_id, x.season), [])
                       for x in r.itertuples()]
    total = r.groupby(["player_source_id", "season"])["mins"].sum()
    other = r[r["other_club"]].groupby(["player_source_id", "season"])["mins"].sum()
    return pd.DataFrame({"total": total, "other": other}).fillna(0)


def staged_collect(client, need: pd.DataFrame, ps: pd.DataFrame, opta: dict) -> None:
    """Fetch games pages in stages, skipping pages that cannot change the target.

    1. Liga 2 / I Liga / rated foreign rows in s+1.
    2. Same in s+2, only for player-seasons still below 450 minutes at the
       strictest threshold (so every sensitivity threshold stays exact); when
       450 is already reached at the player's own Liga 3 club, only rows at
       other clubs (needed for subiu_individual).
    """
    liga3 = ps.groupby(["player_source_id", "season"])["team_id"].agg(lambda s: set(s))
    key = ["player_source_id", "season"]
    strict = max([opta["meta"]["threshold"], *opta["meta"]["sensitivity_thresholds"]])

    s1 = need[need["k"] == 1]
    _fetch(client, s1, "Stage 1 (s+1)")
    m1 = _minutes_so_far(s1, liga3, strict) if len(s1) else pd.DataFrame(columns=["total", "other"])

    s2 = need[need["k"] == 2].copy()
    s2 = s2.join(m1, on=key).fillna({"total": 0, "other": 0})
    s2["other_club"] = [x.club_source_id not in liga3.get((x.player_source_id, x.season), set())
                        for x in s2.itertuples()]
    s2 = s2[(s2["total"] < THRESHOLD_MINUTES) | ((s2["other"] < THRESHOLD_MINUTES) & s2["other_club"])]
    _fetch(client, s2, "Stage 2 (s+2)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--teams-only", action="store_true", help="with --collect: only team pages")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_leagues_config()
    ps, career = load()
    fr, classified = classified_foreign(ps, career, cfg)
    # Requests only for labelled seasons; other seasons use the cache as it is.
    teams_todo = missing_team_pages(fr[fr["season"].isin(LABELLED_SEASONS)], cfg)

    if args.dry_run or args.collect:
        log.info("Team pages to fetch: %d", len(teams_todo))
    if args.collect and len(teams_todo):
        with zz.make_client() as client:
            for r in teams_todo.itertuples():
                try:
                    zz.fetch_team_season(client, int(r.club_source_id), int(r.season_source_id))
                except PageError as exc:
                    log.warning("Skipping team %s: %s", r.club, exc)
        fr, classified = classified_foreign(ps, career, cfg)

    unknown = classified[classified["group"].isna()]
    if len(unknown):
        log.warning("%d foreign rows with unknown league tier (add to config/foreign_leagues.toml):\n%s",
                    len(unknown), unknown[["club", "country", "future", "league"]].drop_duplicates().to_string())

    opta = load_opta()
    dest = destination_rows(ps, career, classified, opta)
    need = rows_needing_minutes(dest)
    need = need[need["season"].isin(LABELLED_SEASONS)]  # only used for --dry-run / --collect
    games_todo = missing_games_pages(need)
    if args.dry_run or args.collect:
        log.info("Games pages to fetch (at most): %d (PT rows %d, rated foreign rows %d)", len(games_todo),
                 int(games_todo["dest"].eq("PT").sum()), int(games_todo["dest"].eq("F").sum()))
    if args.dry_run or (args.collect and args.teams_only):
        return 0
    if args.collect:
        with zz.make_client() as client:
            staged_collect(client, need, ps, opta)
            log.info("Network requests made in this run: %d", client.requests_made)
        return 0

    classified.to_parquet(config.INTERIM_DIR / "foreign_destinations.parquet", index=False)
    dest.to_parquet(config.INTERIM_DIR / "destination_rows.parquet", index=False)
    target = build_target(ps, dest, opta)
    if target.attrs.get("missing_minutes_pages"):
        log.info("%d games pages not cached (skipped by the staged rules, or unavailable)", target.attrs["missing_minutes_pages"])
    out = config.PROCESSED_DIR / "target.parquet"
    target.drop(columns=["liga3_team_ids"]).to_parquet(out, index=False)
    log.info("target: %d player-seasons -> %s", len(target), out.relative_to(config.PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
