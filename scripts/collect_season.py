"""Collect raw pages (calendar, matches, players) for one zerozero edition.

Resumable: every page is cached in data/raw/zerozero/, so re-running skips
what is already on disk. Requests are sequential with 3-5 s pauses.

Examples:
    python scripts/collect_season.py --limit 5
    python scripts/collect_season.py --round J1
    python scripts/collect_season.py            # whole edition (2023/24)
    python scripts/collect_season.py --season 2022/23 --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys

from scouting.http import PageError
from scouting.parsing import zerozero as zzp
from scouting.sources import zerozero as zz

log = logging.getLogger("collect")


def collect_calendar(client, edition_id: int) -> list[dict]:
    first = zz.fetch_calendar_page(client, edition_id, 1)
    rows = zzp.parse_calendar(first)
    for page in range(2, zzp.calendar_last_page(first) + 1):
        rows += zzp.parse_calendar(zz.fetch_calendar_page(client, edition_id, page))
    # Keep only played matches (the calendar can list postponed fixtures too).
    seen, out = set(), []
    for r in rows:
        if r["match_id"] not in seen:
            seen.add(r["match_id"])
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=sorted(zz.LIGA3_EDITIONS), default="2023/24",
                    help="Liga 3 season (default 2023/24)")
    ap.add_argument("--edition", type=int, help="explicit zerozero edition id (overrides --season)")
    ap.add_argument("--limit", type=int, help="only the first N matches of the calendar")
    ap.add_argument("--round", help="only matches with this phase label, e.g. J1")
    ap.add_argument("--skip-players", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with zz.make_client() as client:
        edition = args.edition or zz.LIGA3_EDITIONS[args.season]
        calendar = collect_calendar(client, edition)
        log.info("Calendar: %d matches", len(calendar))
        if args.round:
            calendar = [r for r in calendar if r["phase"] == args.round]
        if args.limit:
            calendar = calendar[: args.limit]

        player_ids: set[int] = set()
        failed: list[tuple[int, str]] = []
        unavailable: list[str] = []
        for i, row in enumerate(calendar, 1):
            mid = row["match_id"]
            try:
                html = zz.fetch_match(client, mid)
            except PageError as exc:
                unavailable.append(f"match/{mid}")
                log.warning("Skipping match %s: %s", mid, exc)
                continue
            try:
                parsed = zzp.parse_match(html, mid)
            except zzp.ParseError as exc:
                failed.append((mid, str(exc)))
                log.warning("Match %s not parsed: %s", mid, exc)
                continue
            player_ids.update(e.player_source_id for e in parsed["lineups"])
            log.info("[%d/%d] match %s ok", i, len(calendar), mid)

        if not args.skip_players:
            todo = sorted(p for p in player_ids if not client.is_cached(f"player/{p}"))
            log.info("Players: %d total, %d to fetch", len(player_ids), len(todo))
            for i, pid in enumerate(todo, 1):
                try:
                    zz.fetch_player(client, pid)
                except PageError as exc:
                    unavailable.append(f"player/{pid}")
                    log.warning("Skipping player %s: %s", pid, exc)
                if i % 25 == 0:
                    log.info("players %d/%d", i, len(todo))

    if failed:
        log.warning("%d matches failed to parse: %s", len(failed), failed)
    if unavailable:
        log.warning("%d pages unavailable (server error), re-run later to retry: %s",
                    len(unavailable), unavailable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
