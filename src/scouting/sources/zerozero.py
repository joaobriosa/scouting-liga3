"""zerozero.pt source: URLs and page fetching only (no parsing here).

Keeping fetching isolated means this module can later be swapped for a
licensed API without touching parsing, minutes or validation code.
"""

from __future__ import annotations

import re

from scouting import config
from scouting.http import PoliteClient

SOURCE = "zerozero"
BASE_URL = "https://www.zerozero.pt"

# Liga 3 umbrella editions (each covers the series and the 2nd phase).
# From the season selector on /competicao/liga-3.
LIGA3_EDITIONS = {
    "2021/22": 144937,
    "2022/23": 165887,
    "2023/24": 176012,
    "2025/26": 201248,  # unlabelled season: used only for the 2025/26 ranking
}
LIGA3_2023_24_EDITION_ID = LIGA3_EDITIONS["2023/24"]

# Portuguese top two tiers, for seasons s+1 and s+2 of the target variable.
# From /competicao/liga-portuguesa and /competicao/segunda-liga-portuguesa.
PT_LEAGUE_EDITIONS = {
    1: {"2020/21": 147383, "2021/22": 156405, "2022/23": 165864, "2023/24": 175797, "2024/25": 187713, "2025/26": 201241},
    2: {"2020/21": 147384, "2021/22": 156409, "2022/23": 165890, "2023/24": 175798, "2024/25": 187796, "2025/26": 201246},
}

EDITION_URL_RE = re.compile(r"/edicao/([a-z0-9-]+)/(\d+)")


def make_client() -> PoliteClient:
    if not config.CONTACT_EMAIL:
        raise SystemExit("Set SCOUTING_CONTACT_EMAIL (your contact address for the User-Agent) before collecting.")
    return PoliteClient(SOURCE)


def edition_url(edition_id: int) -> str:
    return f"{BASE_URL}/edicao.php?id_edicao={edition_id}"


def match_url(match_id: int) -> str:
    # zerozero resolves the id regardless of the slug.
    return f"{BASE_URL}/jogo.php?id={match_id}"


def player_url(player_id: int) -> str:
    return f"{BASE_URL}/jogador.php?id={player_id}"


def fetch_edition(client: PoliteClient, edition_id: int) -> str:
    return client.get(edition_url(edition_id), f"edition/{edition_id}")


def edition_slug(client: PoliteClient, edition_id: int) -> str:
    """The calendar view only works on the canonical slug URL
    (/edicao/<slug>/<id>/calendario); the slug is read from the edition page."""
    html = fetch_edition(client, edition_id)
    for slug, eid in EDITION_URL_RE.findall(html):
        if int(eid) == edition_id:
            return slug
    raise ValueError(f"No slug found for edition {edition_id}")


def calendar_url(slug: str, edition_id: int, page: int = 1) -> str:
    return f"{BASE_URL}/edicao/{slug}/{edition_id}/calendario?page={page}"


def fetch_calendar_page(client: PoliteClient, edition_id: int, page: int = 1) -> str:
    key = f"calendar/{edition_id}_p{page}"
    if client.is_cached(key):
        return client.get("", key)
    html = client.get(calendar_url(edition_slug(client, edition_id), edition_id, page), key)
    if 'id="team_games"' not in html:
        # Wrong URL silently returns the edition home page: do not keep it.
        client.cache_path(key).unlink()
        raise ValueError(f"Calendar page {page} of edition {edition_id} has no fixtures table")
    return html


def fetch_match(client: PoliteClient, match_id: int) -> str:
    return client.get(match_url(match_id), f"match/{match_id}")


def fetch_player(client: PoliteClient, player_id: int) -> str:
    return client.get(player_url(player_id), f"player/{player_id}")


def team_season_url(team_id: int, season_id: int) -> str:
    return f"{BASE_URL}/equipa.php?id={team_id}&epoca_id={season_id}"


def fetch_team_season(client: PoliteClient, team_id: int, season_id: int) -> str:
    """Team page for one season (shows the competitions it played)."""
    return client.get(team_season_url(team_id, season_id), f"team/{team_id}_{season_id}")
