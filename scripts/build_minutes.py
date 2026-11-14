"""Parse cached zerozero pages and write the interim Parquet tables.

Only reads data/raw/ (no network). Run collect_season.py first.

    python scripts/build_minutes.py

Writes data/interim/: matches, appearances, goal_events, players, calendar,
parse_failures; and data/processed/player_season_minutes.parquet.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from scouting import config
from scouting.parsing import zerozero as zzp
from scouting.destinations import load_leagues_config
from scouting.season import build_player_season_minutes
from scouting.tables import build_calendar, build_player_career, build_players, build_tables

log = logging.getLogger("build")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = config.RAW_DIR / "zerozero"

    calendar_rows = [
        r
        for p in sorted((raw / "calendar").glob("*.html"))
        for r in zzp.parse_calendar(p.read_text(encoding="utf-8"))
    ]

    parsed, failed = [], []
    for path in sorted((raw / "match").glob("*.html")):
        mid = int(path.stem)
        try:
            parsed.append(zzp.parse_match(path.read_text(encoding="utf-8"), mid))
        except zzp.ParseError as exc:
            failed.append({"match_id": mid, "error": str(exc)})
    if failed:
        log.warning("%d matches failed to parse: %s", len(failed), failed)

    tables = build_tables(parsed)
    tables["calendar"] = build_calendar(calendar_rows)
    tables["parse_failures"] = pd.DataFrame(failed, columns=["match_id", "error"])
    # Event icons the parser does not recognise: must stay empty (see validation).
    tables["unknown_events"] = pd.DataFrame(
        [
            {"match_id": m["match_id"], "player_source_id": e.player_source_id,
             "player_name": e.player_name, "kind": ev.kind, "minute": ev.minute}
            for m in parsed for e in m["lineups"] for ev in e.events
            if ev.kind.startswith("unknown")
        ],
        columns=["match_id", "player_source_id", "player_name", "kind", "minute"],
    )

    needed = set(tables["appearances"]["player_source_id"])
    players_parsed, career_rows = [], []
    for p in sorted((raw / "player").glob("*.html")):
        pid = int(p.stem)
        if pid not in needed:
            continue
        html = p.read_text(encoding="utf-8")
        players_parsed.append(zzp.parse_player(html, pid))
        career_rows += zzp.parse_player_career(html, pid)
    tables["players"] = build_players(players_parsed)
    calendar = set(load_leagues_config()["calendar_year"]["countries"])
    tables["player_career"] = build_player_career(career_rows, calendar)
    missing = needed - set(tables["players"]["player_source_id"])
    if missing:
        log.warning("%d players without a cached profile page", len(missing))

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out = config.INTERIM_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        log.info("%s: %d rows -> %s", name, len(df), out.relative_to(config.PROJECT_ROOT))

    season = build_player_season_minutes(
        tables["matches"], tables["appearances"], tables["players"], tables["calendar"]
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "player_season_minutes.parquet"
    season.to_parquet(out, index=False)
    log.info("player_season_minutes: %d rows -> %s", len(season), out.relative_to(config.PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
