"""Parse zerozero.pt match and player pages into plain dicts.

Minutes are kept raw here (base minute + added time); the minutes rules live
in scouting.minutes so they can change without re-parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser, LexborNode

from scouting.minutes import EXTRA_TIME_LENGTH, MATCH_LENGTH

PT_MONTHS = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}

MINUTE_RE = re.compile(r"(\d+)\s*(?:\+\s*(\d+))?\s*'")
PLAYER_HREF_RE = re.compile(r"/jogador/[^/]+/(\d+)")
TEAM_HREF_RE = re.compile(r"/equipa/[^/]+/(\d+)")
EDITION_HREF_RE = re.compile(r"/edicao/[^/]+/(\d+)")


class ParseError(ValueError):
    pass


def parse_minute(text: str) -> tuple[int | None, int | None]:
    """'45+2'' -> (45, 2); '81'' -> (81, None); 'n/d' or blank -> (None, None)."""
    if text.strip().strip("'").strip().lower() in ("n/d", ""):
        return None, None
    m = MINUTE_RE.search(text)
    if not m:
        raise ParseError(f"Cannot parse minute from {text!r}")
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def _text(node: LexborNode | None) -> str:
    return re.sub(r"\s+", " ", node.text(deep=True)).strip() if node else ""


def _id_from(href: str | None, pattern: re.Pattern) -> int | None:
    m = pattern.search(href or "")
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Match page
# --------------------------------------------------------------------------
@dataclass
class PlayerEvent:
    kind: str  # "goal", "assist", "yellow", "red", "sub_in", "sub_out", other title
    minute: int | None  # None when the source shows "n/d" or leaves it blank
    added: int | None
    second_yellow: bool = False  # red card that came from a second yellow


@dataclass
class LineupEntry:
    team_side: str  # "home" / "away"
    player_source_id: int
    player_name: str
    shirt_number: int | None
    is_captain: bool
    started: bool
    position: str | None = None
    events: list[PlayerEvent] = field(default_factory=list)


EVENT_KINDS = {
    "Golos": "goal",
    "Assistência": "assist",
    "Amarelos": "yellow",
    "Vermelhos": "red",
    "Entrou": "sub_in",
    "Penaltis Falhados": "missed_penalty",
    "Penaltis Defendidos": "saved_penalty",
}


def _parse_events(events_node: LexborNode | None) -> list[PlayerEvent]:
    """Events are <span icon/> followed by <div>minute</div>, repeated."""
    if events_node is None:
        return []
    out: list[PlayerEvent] = []
    pending_kind: str | None = None
    second_yellow = False
    for child in events_node.iter():
        if child.tag == "span":
            title = child.attributes.get("title")
            cls = child.attributes.get("class") or ""
            if title:
                pending_kind = EVENT_KINDS.get(title, f"unknown:{title}")
                # Second yellow: a "Vermelhos" icon drawn in yellow, followed by
                # an untitled red icon, then the sending-off minute.
                second_yellow = title == "Vermelhos" and "yellow" in cls
            elif "red" in cls and pending_kind == "red":
                continue  # second half of the second-yellow icon pair
            elif "grey" in cls and child.text().strip() == "8":
                pending_kind = "sub_out"  # untitled grey "8" glyph = substituted off
            else:
                pending_kind = f"unknown:{cls}:{child.text().strip()}"
        elif child.tag == "div" and pending_kind is not None:
            minute, added = parse_minute(child.text())
            out.append(PlayerEvent(pending_kind, minute, added, second_yellow))
            pending_kind, second_yellow = None, False
    return out


def _parse_player(node: LexborNode, side: str, started: bool) -> LineupEntry:
    link = node.css_first("div.name a[href^='/jogador/']")
    if link is None:
        raise ParseError("Player row without /jogador/ link")
    raw_name = _text(link)
    is_captain = raw_name.endswith("(C)")
    name = raw_name.removesuffix("(C)").strip()
    number_txt = _text(node.css_first("div.number"))
    return LineupEntry(
        team_side=side,
        player_source_id=_id_from(link.attributes.get("href"), PLAYER_HREF_RE),
        player_name=name,
        shirt_number=int(number_txt) if number_txt.isdigit() else None,
        is_captain=is_captain,
        started=started,
        events=_parse_events(node.css_first("div.events")),
    )


def _parse_lineups(tree: LexborHTMLParser) -> list[LineupEntry]:
    report = tree.css_first("#game_report")
    if report is None:
        raise ParseError("No #game_report block (lineups missing)")
    # Columns: home XI, away XI, home bench, away bench, then coaches (ignored).
    cols = []
    for c in report.css("div.zz-tpl-col"):
        if _text(c.css_first(".subtitle")) == "Treinadores":
            break
        if c.css("div.player"):
            cols.append(c)
    if len(cols) != 4:
        raise ParseError(f"Expected 4 lineup columns, found {len(cols)}")
    layout = [("home", True), ("away", True), ("home", False), ("away", False)]
    entries: list[LineupEntry] = []
    for col, (side, started) in zip(cols, layout):
        for p in col.css("div.player"):
            entries.append(_parse_player(p, side, started))
    return entries


def _line_label(line_idx: int, n_lines: int) -> str:
    if line_idx == 0:
        return "GK"
    if line_idx == 1:
        return "DF"
    if line_idx == n_lines - 1:
        return "FW"
    return "MF"


def _parse_pitch_positions(tree: LexborHTMLParser) -> tuple[dict[int, str], list[str]]:
    """Map starter player_id -> GK/DF/MF/FW using the tactical pitch view.

    The pitch lists formation columns left to right: home GK first, then the
    home lines, then the away lines mirrored (away GK last).
    """
    formations = [_text(n) for n in tree.css("span.pitch_esquema")]
    pitch = tree.css_first("div.pitch_eleven_horizontal")
    if pitch is None or len(formations) != 2:
        return {}, formations
    cols = [
        [int(d.attributes["data-player-id"]) for d in c.css("[data-player-id]")]
        for c in pitch.css("div.coluna")
    ]
    n_home = 1 + len(formations[0].split("-"))
    home_cols, away_cols = cols[:n_home], list(reversed(cols[n_home:]))
    # Sweeper notation ("1-4-3-2"): the pitch draws the sweeper's column on the
    # outside of the goalkeeper's, so the GK is the second column, not the first.
    for formation, team_cols in zip(formations, (home_cols, away_cols)):
        if formation.startswith("1-") and len(team_cols) > 1:
            team_cols[0], team_cols[1] = team_cols[1], team_cols[0]
    positions: dict[int, str] = {}
    for team_cols in (home_cols, away_cols):
        for i, col in enumerate(team_cols):
            if i == 0 and len(col) != 1:
                continue  # malformed pitch (e.g. 2 players in the GK column): leave NA
            for pid in col:
                positions[pid] = _line_label(i, len(team_cols))
    return positions, formations


def season_from_edition(edition_name: str) -> str | None:
    """'Liga 3 Série A 2023/24' or 'Liga 3 Série A 23/24' -> '2023/24'."""
    m = re.search(r"(\d{2,4})/(\d{2})\b", edition_name or "")
    if not m:
        return None
    start = m.group(1) if len(m.group(1)) == 4 else f"20{m.group(1)}"
    return f"{start}/{m.group(2)}"


def edition_stage_series(edition_name: str) -> tuple[str, str | None]:
    """'Liga 3 Série A 2023/24' -> ('first', 'A');
    'Liga 3 II Fase Ap.Campeão 2023/24' -> ('second', 'Apuramento Campeão');
    'Liga 3 Manutenção Série 1 2023/24' -> ('second', 'Manutenção 1')."""
    n = edition_name or ""
    serie = re.search(r"Série\s+(\w+)", n)
    suffix = f" {serie.group(1)}" if serie else ""
    if "Campe" in n:
        return "second", "Apuramento Campeão"
    if re.search(r"play-?off", n, re.I):
        return "second", "Playoff Subida"
    if "Subida" in n:
        return "second", f"Subida{suffix}"
    if "Manuten" in n:
        return "second", f"Manutenção{suffix}"
    return "first", serie.group(1) if serie else None


def _parse_header(tree: LexborHTMLParser) -> dict:
    title = tree.css_first("div.match-header-info-title a")
    subtitle = tree.css_first("div.match-header-info-subtitle a")
    team_links = tree.css("div.match-header-team-name a")
    score = _text(tree.css_first("div.match-header-vs a"))
    # "2-1", "1-1(5-3 g.p.)" (penalty shoot-out), "1-1 a.p." (after extra
    # time), or a letter code such as "DA" (decided off the pitch).
    score_m = re.fullmatch(
        r"(\d+)-(\d+)\s*(?:\((\d+)-(\d+)\s*g\.p\.\))?\s*(a\.p\.)?", score
    )
    if len(team_links) != 2 or not (score_m or re.fullmatch(r"[A-Z]{1,4}", score)):
        raise ParseError(f"Unexpected header (teams={len(team_links)}, score={score!r})")
    result_code = None if score_m else score
    result_note = _text(tree.css_first("div.zz_error_message")) or None
    home_pens = int(score_m.group(3)) if score_m and score_m.group(3) else None
    away_pens = int(score_m.group(4)) if score_m and score_m.group(4) else None
    after_extra_time_marker = bool(score_m and score_m.group(5))

    day = _text(tree.css_first("div.match-header-datetime-day-num"))
    month_year = _text(tree.css_first("div.match-header-datetime-day-month"))
    m = re.match(r"([A-Z]{3})\.?\s+(\d{4})", month_year)
    date = f"{m.group(2)}-{PT_MONTHS[m.group(1)]:02d}-{int(day):02d}" if m and day else None

    edition_title = title.text(deep=True, separator="\n").split("\n")[0].strip() if title else ""
    home_goals, away_goals = (
        (None, None) if result_code else (int(score_m.group(1)), int(score_m.group(2)))
    )
    stage, series = edition_stage_series(edition_title)
    return {
        "edition_id": _id_from(title.attributes.get("href") if title else None, EDITION_HREF_RE),
        "edition_name": edition_title,
        "stage": stage,
        "series": series,
        "round": _text(subtitle) or None,
        "date": date,
        "home_team": _text(team_links[0]),
        "away_team": _text(team_links[1]),
        "home_team_id": _id_from(team_links[0].attributes.get("href"), TEAM_HREF_RE),
        "away_team_id": _id_from(team_links[1].attributes.get("href"), TEAM_HREF_RE),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result_code": result_code,
        "result_note": result_note,
        "home_penalties": home_pens,
        "away_penalties": away_pens,
        "_after_extra_time_marker": after_extra_time_marker,
    }


def _parse_header_goals(tree: LexborHTMLParser) -> list[dict]:
    """Goals from the header scorer lists; the side is the team credited.

    Own goals are listed on the side that benefited, so `scoring_team` is right
    even for own goals. Any extra text next to the minute (e.g. a penalty or
    own-goal marker) is kept in `note` so we can learn the notation.
    """
    goals: list[dict] = []
    for side, cls in (("home", "right"), ("away", "left")):
        box = tree.css_first(f"div.match-header-scorers.{cls}")
        if box is None:
            continue
        current: dict | None = None
        for child in box.iter(include_text=True):
            if child.tag == "a":
                current = {
                    "side": side,
                    "scorer_source_id": _id_from(child.attributes.get("href"), PLAYER_HREF_RE),
                    "scorer_name": _text(child),
                    "note": "",
                }
            elif child.tag == "span" and "time" in (child.attributes.get("class") or ""):
                # One <span class="time"> may hold several minutes ("21' (g.p.) 90+5'");
                # each goal's note is the text between its minute and the next one.
                txt = child.text()
                found = list(MINUTE_RE.finditer(txt))
                for k, mm in enumerate(found):
                    g = dict(current)
                    g["minute"], g["added"] = int(mm.group(1)), (int(mm.group(2)) if mm.group(2) else None)
                    end = found[k + 1].start() if k + 1 < len(found) else len(txt)
                    g["note"] = re.sub(r"[\s,]+", " ", txt[mm.end():end]).strip()
                    goals.append(g)
            elif child.tag == "-text" and current is not None and child.text().strip():
                if goals and goals[-1]["scorer_source_id"] == current["scorer_source_id"]:
                    goals[-1]["note"] = (goals[-1]["note"] + " " + child.text().strip()).strip()
    return goals


def parse_match(html: str, match_id: int) -> dict:
    tree = LexborHTMLParser(html)
    header = _parse_header(tree)
    lineups = _parse_lineups(tree)
    positions, formations = _parse_pitch_positions(tree)
    for e in lineups:
        if e.started:
            e.position = positions.get(e.player_source_id)
    # Extra time: "a.p." marker, or any event at minute >= 106 (stoppage time
    # in regulation is written "90+x", occasionally "93'", never past 105).
    max_minute = max(
        (ev.minute for e in lineups for ev in e.events if ev.minute is not None), default=0
    )
    extra_time = header.pop("_after_extra_time_marker") or max_minute >= 106
    return {
        "match_id": match_id,
        **header,
        "extra_time": extra_time,
        "match_length": EXTRA_TIME_LENGTH if extra_time else MATCH_LENGTH,
        "home_formation": formations[0] if len(formations) == 2 else None,
        "away_formation": formations[1] if len(formations) == 2 else None,
        "lineups": lineups,
        "header_goals": _parse_header_goals(tree),
    }


# --------------------------------------------------------------------------
# Calendar page
# --------------------------------------------------------------------------
def parse_calendar(html: str) -> list[dict]:
    tree = LexborHTMLParser(html)
    rows = []
    for tr in tree.css("#team_games tr.parent"):
        result = tr.css_first("td.result a") or tr.css_first("td.result")
        edition = tr.css_first("td.edition a")
        home = tr.css_first("td.home a[href^='/equipa/']")
        away = tr.css_first("td.away a[href^='/equipa/']")
        away_suffix = _text(tr.css_first("td.away span.small_faded"))
        home_suffix = _text(tr.css_first("td.home span.small_faded"))
        rows.append({
            "match_id": int(tr.attributes["id"]),
            "date": _text(tr.css_first("td.date")),
            "phase": _text(tr.css_first("td.phase")),
            "edition_id": _id_from(edition.attributes.get("href") if edition else None, EDITION_HREF_RE),
            "edition_name": _text(edition),
            "home_team_id": _id_from(home.attributes.get("href") if home else None, TEAM_HREF_RE),
            "home_team": f"{_text(home)} {home_suffix}".strip(),
            "away_team_id": _id_from(away.attributes.get("href") if away else None, TEAM_HREF_RE),
            "away_team": f"{_text(away)} {away_suffix}".strip(),
            "result": _text(result),
        })
    return rows


def calendar_last_page(html: str) -> int:
    pages = [int(p) for p in re.findall(r"[?&]page=(\d+)", html)]
    return max(pages, default=1)


# --------------------------------------------------------------------------
# Player page
# --------------------------------------------------------------------------
def parse_player(html: str, player_id: int) -> dict:
    tree = LexborHTMLParser(html)
    text = tree.body.text(separator="\n") if tree.body else ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    def after(label: str) -> str | None:
        for i, line in enumerate(lines[:-1]):
            if line == label:
                return lines[i + 1]
        return None

    birth = after("Data de Nascimento")
    birth_m = re.match(r"(\d{4}-\d{2}-\d{2})", birth or "")
    name_node = tree.css_first("h1")
    return {
        "player_source_id": player_id,
        # The <h1> is prefixed with the current shirt number, e.g. "5.<player name>".
        "player_name": re.sub(r"^\d+\.\s*", "", _text(name_node)) or None,
        "full_name": after("Nome"),
        "birth_date": birth_m.group(1) if birth_m else None,
        "position": after("Posição"),
        "nationality": after("Nacionalidade"),
    }


# --------------------------------------------------------------------------
# Player career table ("table.career" on the player profile)
# --------------------------------------------------------------------------
SEASON_ID_RE = re.compile(r"epoca_id=(\d+)")
TEAM_ID_ONCLICK_RE = re.compile(r"equipa_id=(\d+)")
GAMES_PATH_RE = re.compile(r"/jogador/[^/']+/\d+/jogos\?[^']+")
FLAG_RE = re.compile(r"flag:(\w+)")
# UK home nations are drawn with an image instead of a "flag:XX" class.
IMG_FLAG_RE = re.compile(r"bandeiras/flag_\d+_([a-z]+)_")
IMG_FLAG_CODES = {"eng": "GB-ENG", "sco": "GB-SCT", "wal": "GB-WLS", "nir": "GB-NIR"}


def _int_or_none(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdigit() else None


def parse_player_career(html: str, player_id: int) -> list[dict]:
    """One row per season and team listed on the profile.

    The profile gives season, team, matches, goals and assists only (all
    competitions together); competition, tier and minutes are not on it.
    Team cell labels: "[B]", "[Jun.A S19]", ... = squad; "(E) [Club]" = loan,
    where the last label is the parent club.
    """
    tree = LexborHTMLParser(html)
    table = tree.css_first("table.career")
    if table is None:
        return []
    rows: list[dict] = []
    season = None
    for tr in table.css("tr"):
        tds = tr.css("td")
        if len(tds) < 4:
            continue  # header row
        season = _text(tds[1]) or season  # several teams in one season: blank cell
        cell = tds[2]
        link = cell.css_first("a[href^='/equipa/']")
        labels = [
            c.text().strip().strip("[]").strip()
            for c in cell.iter() if c.tag == "span" and c.text().strip().startswith("[")
        ]
        is_loan = "(E)" in cell.text()
        parent = labels.pop() if is_loan and labels else None
        flag = FLAG_RE.search(cell.html or "")
        img_flag = IMG_FLAG_RE.search(cell.html or "")
        season_id = SEASON_ID_RE.search(tr.attributes.get("onclick") or "")
        rows.append({
            "player_source_id": player_id,
            "season": season,
            "season_source_id": int(season_id.group(1)) if season_id else None,
            "club": _text(link),
            # Big clubs link by slug only (/equipa/hnk-rijeka?epoca_id=...): the id
            # is then only in the row's onclick (equipa_id=...).
            "club_source_id": _id_from(link.attributes.get("href") if link else None, TEAM_HREF_RE)
            or _id_from(tr.attributes.get("onclick"), TEAM_ID_ONCLICK_RE),
            "squad": " ".join(labels) or None,  # None = first team
            "country": flag.group(1) if flag else (
                IMG_FLAG_CODES.get(img_flag.group(1), img_flag.group(1).upper()) if img_flag else None
            ),
            "is_loan": is_loan,
            "loan_from": parent,
            # Player's games for this club-season (per-competition minutes).
            "games_path": (GAMES_PATH_RE.search(tr.attributes.get("onclick") or "") or [None])[0],
            "matches": _int_or_none(_text(tds[3])),
            "goals": _int_or_none(_text(tds[4])) if len(tds) > 4 else None,
            "assists": _int_or_none(_text(tds[5])) if len(tds) > 5 else None,
        })
    return rows


# --------------------------------------------------------------------------
# Team page for one season ("Resumo" table: competitions played)
# --------------------------------------------------------------------------
CUP_RE = re.compile(
    r"(\b(cup|copa|taça|taca|coupe|coppa|pokal|beker|kupa|kupası|puchar|kypello|trophy|"
    r"supercopa|supercup|supertaça|shield|troféu|trofeo|play-?offs?)\b"
    r"|uefa|champions league|europa league|conference league|libertadores|sudamericana|"
    r"concacaf|\bafc\b)",
    re.I,
)


def parse_team_season(html: str) -> dict:
    """Competitions in the team's season summary and the league among them.

    The league is the non-cup competition with the most matches.
    """
    tree = LexborHTMLParser(html)
    comps = []
    for tr in tree.css("#entity_season tbody tr"):
        link = tr.css_first("td.edition a[href^='/edicao/']")
        stats = tr.css("td.stat")
        if link is None or not stats:
            continue
        comps.append({
            "edition_id": _id_from(link.attributes.get("href"), EDITION_HREF_RE),
            "competition": _text(link),
            "matches": _int_or_none(_text(stats[0])) or 0,
        })
    leagues = [c for c in comps if not CUP_RE.search(c["competition"])]
    league = max(leagues, key=lambda c: c["matches"]) if leagues else None
    return {"competitions": comps, "league": league}


LOGO_TEAM_ID_RE = re.compile(r"logos/equipas/(\d+)_")


def parse_edition_teams(html: str) -> list[dict]:
    """Teams in an edition's classification table (#edition_table).

    Big clubs link by slug only; their id is then taken from the logo file
    name (logos/equipas/<id>_...). B teams always carry their own id in the
    link (their logo is the parent club's), so the link id wins when present.
    """
    tree = LexborHTMLParser(html)
    table = tree.css_first("#edition_table")
    teams = []
    for tr in (table.css("tbody tr") if table else []):
        link = tr.css_first("td a[href^='/equipa/']")
        if link is None:
            continue
        img = tr.css_first("img")
        team_id = _id_from(link.attributes.get("href"), TEAM_HREF_RE) or _id_from(
            img.attributes.get("src") if img else None, LOGO_TEAM_ID_RE)
        name = (img.attributes.get("title") if img else None) or _text(link)
        teams.append({"team_id": team_id, "team": name})
    return teams


# --------------------------------------------------------------------------
# Player games page for one club-season (#player_summary)
# --------------------------------------------------------------------------
COMPETITION_ID_RE = re.compile(r"competicao_id=(\d+)")
# Not league football: cups, European/continental competitions, promotion /
# relegation play-offs ("Liguilha"), friendlies and pre-season.
# Not senior league football: cups, continental club competitions, friendlies,
# youth/U23 leagues (owner, 2026-11-14: Liga Next Gen and continental
# competitions do not count as league minutes).
NON_LEAGUE_RE = re.compile(
    CUP_RE.pattern
    + r"|liguilha|amig|friendl|pré-época|pre-season"
    + r"|next gen|revelação|revelacao|aspirantes|juniores|juvenis|youth|primavera|reserve"
    + r"|\bsub-?\d{2}\b|\bu-?\d{2}\b"
    + r"|canadian championship|club championship|african football league|\bcaf\b|asean|arab club"
    + r"|club world|recopa|leagues cup|intercontinental|\bacl\b|\bafc\b"
    + r"|kubo[gk]|kuboku|\bcupa\b|\bkup\b|bikar|cupen|\bpiala\b",
    re.I,
)
TEAM_ID_RE = re.compile(r"/equipa/[^/?]+/(\d+)")
EDITION_ID_RE = re.compile(r"edi(?:cao|tion)[^0-9]*(\d+)")


def parse_player_games_summary(html: str) -> list[dict]:
    """Per-competition totals: matches (J) and minutes (M), with is_league."""
    tree = LexborHTMLParser(html)
    table = tree.css_first("#player_summary table")
    head = [_text(th) for th in table.css("thead th")] if table else []
    rows = []
    for tr in (table.css("tbody tr") if table else []):
        link = tr.css_first("td.text a")
        tds = tr.css("td")
        if link is None or len(tds) != len(head):
            continue
        cells = dict(zip(head, (_text(td) for td in tds)))
        name = _text(link)
        rows.append({
            "competition": name,
            "competition_id": _id_from(link.attributes.get("href"), COMPETITION_ID_RE),
            "matches": _int_or_none(cells.get("J", "")) or 0,
            "minutes": _int_or_none(cells.get("M", "")) or 0,
            "is_league": not NON_LEAGUE_RE.search(name),
        })
    return rows


def parse_player_games_matches(html: str) -> list[dict]:
    """Per-match rows of a player games page: edition id, team id (None for
    the slug-only links of big clubs' first teams), team name and minutes.

    zerozero ignores the club filter of the games URL: the page lists every
    match of the player in that epoca, for all his clubs."""
    tree = LexborHTMLParser(html)
    table = tree.css_first("table.zz-table")
    out = []
    for tr in (table.css("tbody tr") if table else []):
        tds = tr.css("td")
        if len(tds) < 9:
            continue
        edition = None
        for a in tds[2].css("a"):
            m = EDITION_ID_RE.search(a.attributes.get("href") or "")
            if m:
                edition = int(m.group(1))
        link = tds[4].css_first("a")
        m = TEAM_ID_RE.search(link.attributes.get("href") or "") if link else None
        mins = _text(tds[8])
        out.append({"edition_id": edition, "team_id": int(m.group(1)) if m else None,
                    "team": _text(tds[4]), "minutes": int(mins) if mins.isdigit() else 0,
                    "played": mins.isdigit()})
    return out


def assign_editions(groups: dict, competitions: list[dict]) -> dict | None:
    """Map each edition id to the index of a competition of the summary so
    that (matches, minutes) add up per competition (a competition can have
    several editions, e.g. the two phases of Liga 3). None if impossible."""
    eids = list(groups)
    targets = [(c["matches"], c["minutes"]) for c in competitions]
    acc = [(0, 0)] * len(targets)
    mapping: dict = {}

    def rec(i):
        if i == len(eids):
            return all(a == t for a, t in zip(acc, targets))
        n, m = groups[eids[i]]
        for j, (tn, tm) in enumerate(targets):
            if acc[j][0] + n <= tn and acc[j][1] + m <= tm:
                acc[j] = (acc[j][0] + n, acc[j][1] + m)
                mapping[eids[i]] = j
                if rec(i + 1):
                    return True
                acc[j] = (acc[j][0] - n, acc[j][1] - m)
                del mapping[eids[i]]
        return False

    if len(eids) > 14:
        return None
    return dict(mapping) if rec(0) else None


def club_league_minutes(html: str, club_id: int, club_name: str | None = None) -> dict | None:
    """League minutes at one club from a games page: {competition: minutes}.

    Matches count when the team link has the club id, or (slug-only links of
    big clubs' first teams) the team name equals the club name. None when the
    page has no per-match rows or the editions cannot be matched to the
    summary (the caller then falls back to the summary)."""
    rows = [r for r in parse_player_games_matches(html) if r["played"]]
    comps = parse_player_games_summary(html)
    if not rows:
        return None
    groups: dict = {}
    for r in rows:
        n, m = groups.get(r["edition_id"], (0, 0))
        groups[r["edition_id"]] = (n + 1, m + r["minutes"])
    mapping = assign_editions(groups, comps)
    if mapping is None:
        return None
    out: dict = {}
    name = (club_name or "").casefold()
    for r in rows:
        mine = r["team_id"] == club_id or (r["team_id"] is None and name and r["team"].casefold() == name)
        comp = comps[mapping[r["edition_id"]]]
        if mine and comp["is_league"]:
            out[comp["competition"]] = out.get(comp["competition"], 0) + r["minutes"]
    return out


def in_main_squad(html: str, player_id: int) -> bool:
    """Player listed in the squad block (#team_squad) of a club's team page.
    Team pages of the main club list the main-team squad only (reserves and
    youth squads have their own team ids)."""
    node = LexborHTMLParser(html).css_first("#team_squad")
    return node is not None and any(
        re.search(rf"/{int(player_id)}(?:\?|$)", a.attributes.get("href") or "") for a in node.css("a"))
