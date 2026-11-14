"""Fetch the season team page of the foreign clubs that need a division
(see scouting.destinations). Resumable and polite like collect_season.py.

    python scripts/collect_club_divisions.py --dry-run   # count only
    python scripts/collect_club_divisions.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from scouting import config
from scouting.destinations import country_rule, foreign_candidates, future_career, load_leagues_config
from scouting.http import PageError
from scouting.model import LABELLED_SEASONS
from scouting.sources import zerozero as zz

log = logging.getLogger("divisions")


def pairs_to_fetch() -> pd.DataFrame:
    cfg = load_leagues_config()
    ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
    career = pd.read_parquet(config.INTERIM_DIR / "player_career.parquet")
    cand = foreign_candidates(future_career(ps, career))
    cand = cand[cand["season"].isin(LABELLED_SEASONS)]  # requests only for labelled seasons
    cand = cand[cand["country"].map(lambda c: country_rule(c, cfg)) == "tier"].copy()
    # A few career rows have no season id in the source: it follows from the
    # season string (zerozero ids are consecutive: 2023/24 = 153, 2024/25 = 154).
    known = career.dropna(subset=["season_source_id"]).drop_duplicates("season")
    season_ids = dict(zip(known["season"], known["season_source_id"]))
    cand["season_source_id"] = cand["season_source_id"].fillna(cand["future"].map(season_ids))
    missing = cand["club_source_id"].isna() | cand["season_source_id"].isna()
    if missing.any():
        log.warning("Skipping %d rows without club or season id", int(missing.sum()))
    cand = cand[~missing]
    return (cand[["club_source_id", "club", "country", "future", "season_source_id"]]
            .drop_duplicates(["club_source_id", "season_source_id"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    pairs = pairs_to_fetch()
    log.info("Club-seasons needing a division: %d (%d clubs)", len(pairs), pairs["club_source_id"].nunique())
    if args.dry_run:
        print(pairs.groupby("country").size().sort_values(ascending=False).to_string())
        return 0
    unavailable = []
    with zz.make_client() as client:
        todo = [r for r in pairs.itertuples()
                if not client.is_cached(f"team/{r.club_source_id}_{r.season_source_id}")]
        log.info("To fetch: %d", len(todo))
        for i, r in enumerate(todo, 1):
            try:
                zz.fetch_team_season(client, int(r.club_source_id), int(r.season_source_id))
            except PageError as exc:
                unavailable.append((r.club_source_id, r.season_source_id))
                log.warning("Skipping %s: %s", r.club, exc)
            if i % 25 == 0:
                log.info("teams %d/%d", i, len(todo))
    if unavailable:
        log.warning("%d team pages unavailable, re-run later: %s", len(unavailable), unavailable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
