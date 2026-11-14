"""Where Liga 3 players went in seasons s+1 and s+2, and which foreign leagues
count as promotion (groups A/B/C, see config/foreign_leagues.toml)."""

from __future__ import annotations

import tomllib

import pandas as pd

from scouting import config

LEAGUES_CONFIG = config.PROJECT_ROOT / "config" / "foreign_leagues.toml"
MIN_FOREIGN_MATCHES = 5  # summed over s+1 and s+2; below this 450 minutes is impossible


def load_leagues_config() -> dict:
    with LEAGUES_CONFIG.open("rb") as f:
        return tomllib.load(f)


def next_season(season: str, k: int) -> str:
    start = int(season[:4]) + k
    return f"{start}/{(start + 1) % 100:02d}"


def future_career(player_seasons: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    """Career rows of each Liga 3 player-season in s+1 and s+2 (column `future`)."""
    coh = player_seasons[["player_source_id", "season"]].drop_duplicates()
    fut = pd.concat([coh.assign(future=coh["season"].map(lambda s: next_season(s, k)), k=k)
                     for k in (1, 2)])
    out = fut.merge(career.rename(columns={"season": "future"}), on=["player_source_id", "future"])
    out["matches"] = out["matches"].fillna(0).astype(int)
    return out


def foreign_candidates(future: pd.DataFrame) -> pd.DataFrame:
    """Foreign rows (with >0 matches) of player-seasons that played at least
    MIN_FOREIGN_MATCHES abroad summed over s+1 and s+2."""
    fr = future[future["country"].ne("PT") & future["matches"].gt(0)]
    tot = fr.groupby(["player_source_id", "season"])["matches"].transform("sum")
    return fr[tot >= MIN_FOREIGN_MATCHES].copy()


def country_rule(country: str, cfg: dict) -> str:
    """'tier' if the group depends on the division (a request is needed),
    otherwise the fixed group 'A'/'B'/'C' for every division of that country."""
    order = cfg["ranking"]["order"]
    g = cfg["groups"]
    if country in order:
        rank = order.index(country) + 1
        if rank > g["low_rank_c"]:
            return "C"  # tier 1-2 -> C (decided), tier >= 3 -> C
        return "tier"
    if country in g["non_uefa_group_a"]:
        return "tier"
    return "B"  # other non-UEFA countries


def group_for(country: str, tier: int | None, cfg: dict) -> str | None:
    """Group A/B/C for a club of `country` playing in `tier`; None if the
    tier is needed but unknown."""
    fixed = country_rule(country, cfg)
    if fixed != "tier":
        return fixed
    if tier is None:
        return None
    if tier >= 3:
        return "C"
    g = cfg["groups"]
    if country in g["non_uefa_group_a"]:
        return "A" if tier == 1 else "B"
    rank = cfg["ranking"]["order"].index(country) + 1
    if tier == 1 and rank <= g["tier1_a_max_rank"]:
        return "A"
    if tier == 2 and country in g["big5_second_tier_a"]:
        return "A"
    if tier == 2 and rank > g["top_rank"]:
        return "C"  # ranks 16-30, tier 2
    return "B"      # tier 2 of ranks <= 15 outside the big 5


def league_tier(country: str, competition: str | None, cfg: dict) -> int | None:
    if competition is None:
        return None
    return cfg.get("tiers", {}).get(country, {}).get(competition)


def classify_foreign(cand: pd.DataFrame, team_pages: dict, cfg: dict) -> pd.DataFrame:
    """Add league, tier and group A/B/C to foreign candidate rows.

    team_pages: (club_source_id, season_source_id) -> parsed team season
    (scouting.parsing.zerozero.parse_team_season), for the countries whose
    group depends on the division.
    """
    out = cand.copy()
    leagues, tiers, assumed = [], [], []
    for r in out.itertuples():
        if country_rule(r.country, cfg) != "tier":
            leagues.append(None); tiers.append(None); assumed.append(False)
            continue
        page = team_pages.get((r.club_source_id, r.season_source_id))
        league = page["league"]["competition"] if page and page["league"] else None
        tier = league_tier(r.country, league, cfg)
        # Amateur club not covered by the source: no competition, or only cups.
        is_assumed = page is not None and page["league"] is None
        if is_assumed:
            tier = 3  # treated as tier >= 3 (the source covers tiers 1-2 everywhere)
        leagues.append(league); tiers.append(tier); assumed.append(is_assumed)
    out["league"] = leagues
    out["tier"] = pd.array(tiers, dtype="Int64")
    out["tier_assumed"] = assumed
    out["group"] = [
        group_for(c, None if pd.isna(t) else int(t), cfg) if not (country_rule(c, cfg) == "tier" and pd.isna(t))
        else None
        for c, t in zip(out["country"], out["tier"])
    ]
    out.loc[out["tier"].eq(9).fillna(False), "group"] = "C"  # youth/reserve league
    return out
