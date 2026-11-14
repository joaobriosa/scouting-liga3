from scouting.parsing.zerozero import parse_player_career

TEAM = ('<div class="micrologo_and_text"><div class="image"><span class="flag:{flag}"></span></div>'
        '<div class="text"><a href="/equipa/{slug}/{tid}?epoca_id={sid}">{name}</a></div></div>')


def _row(season, sid, team_html, j, g, a):
    return (f'<tr onclick="document.location=\'/jogador/x/1/jogos?epoca_id={sid}\';">'
            f"<td></td><td>{season}</td><td>{team_html}</td><td>{j}</td><td><a>{g}</a></td><td><a>{a}</a></td></tr>")


HTML = (
    '<table class="career"><tr><th></th><th>ÉPOCA</th><th>EQUIPA</th><th>J</th><th>G</th><th>AST</th></tr>'
    + _row("2023/24", 153, TEAM.format(flag="PT", slug="varzim", tid=36, sid=153, name="Varzim"), 22, 0, 1)
    + _row("2022/23", 152, TEAM.format(flag="PT", slug="fafe", tid=2191, sid=152, name="Fafe")
           + " (E)<span> [Varzim]</span>", 11, 1, 0)
    + _row("", 152, TEAM.format(flag="PT", slug="varzim", tid=84965, sid=152, name="Varzim")
           + "<span> [B]</span>", 2, 0, 0)
    + _row("2021/22", 151, TEAM.format(flag="ES", slug="x", tid=7, sid=151, name="Club X")
           + " (E)<span> [Jun.A S19]</span><span> [Porto]</span>", "-", "-", "-")
    + "</table>"
)


def test_parse_player_career():
    rows = parse_player_career(HTML, 1)
    assert [r["season"] for r in rows] == ["2023/24", "2022/23", "2022/23", "2021/22"]  # blank season carried
    first, loan, b_team, youth_loan = rows
    assert (first["club_source_id"], first["matches"], first["assists"], first["is_loan"]) == (36, 22, 1, False)
    assert (loan["is_loan"], loan["loan_from"], loan["squad"]) == (True, "Varzim", None)
    assert (b_team["squad"], b_team["is_loan"], b_team["club_source_id"]) == ("B", False, 84965)
    assert (youth_loan["squad"], youth_loan["loan_from"], youth_loan["country"]) == ("Jun.A S19", "Porto", "ES")
    assert youth_loan["matches"] is None  # "-" in the source


def _pitch(formations, columns):
    esq = "".join(f'<span class="pitch_esquema">{f}</span>' for f in formations)
    cols = "".join(
        '<div class="coluna">' + "".join(f'<div data-player-id="{p}"></div>' for p in col) + "</div>"
        for col in columns
    )
    return f'<div class="pitch_esquemas">{esq}</div><div class="pitch_eleven_horizontal">{cols}</div>'


def test_pitch_positions_sweeper_and_malformed_gk():
    from selectolax.lexbor import LexborHTMLParser

    from scouting.parsing.zerozero import _parse_pitch_positions

    # Home 4-4-2 (GK=1). Away "1-4-3-2" drawn mirrored with the sweeper (99)
    # outside the goalkeeper (98): GK must be 98.
    html = _pitch(["4-4-2", "1-4-3-2"], [
        [1], [2, 3, 4, 5], [6, 7, 8, 9], [10, 11],
        [90, 91], [92, 93, 94], [95, 96, 97, 100], [98], [99],
    ])
    pos, _ = _parse_pitch_positions(LexborHTMLParser(html))
    assert (pos[1], pos[2], pos[6], pos[10]) == ("GK", "DF", "MF", "FW")
    assert (pos[98], pos[99], pos[90]) == ("GK", "DF", "FW")

    # Two players in the GK column: those two are left without position.
    html = _pitch(["4-4-2", "4-2-3"], [
        [1], [2, 3, 4, 5], [6, 7, 8, 9], [10, 11],
        [20, 21, 22], [23, 24], [25, 26, 27, 28], [29, 30],
    ])
    pos, _ = _parse_pitch_positions(LexborHTMLParser(html))
    assert 29 not in pos and 30 not in pos and pos[25] == "DF"


def test_career_uk_image_flag():
    team = ('<div class="micrologo_and_text"><div class="image"><img src="https://x/img/bandeiras/'
            'flag_16_eng_30x20.png" title="Inglaterra"></div><div class="text">'
            '<a href="/equipa/charlton-athletic/80">Charlton Athletic</a></div></div>')
    html = ('<table class="career"><tr><th></th></tr>'
            + _row("2024/25", 154, team, 12, 1, 0) + "</table>")
    assert parse_player_career(html, 1)[0]["country"] == "GB-ENG"


def test_career_club_id_from_onclick_when_link_has_no_id():
    team = ('<div class="micrologo_and_text"><div class="image"><span class="flag:HR"></span></div>'
            '<div class="text"><a href="/equipa/hnk-rijeka?epoca_id=153">HNK Rijeka</a></div></div>')
    row = ('<tr onclick="document.location=\'/jogador/x/1/jogos?tpstats=club&amp;epoca_id=153&amp;equipa_id=1855\';">'
           f"<td></td><td>2023/24</td><td>{team}</td><td>19</td><td><a>0</a></td><td><a>2</a></td></tr>")
    html = '<table class="career"><tr><th></th></tr>' + row + "</table>"
    r = parse_player_career(html, 1)[0]
    assert (r["club_source_id"], r["season_source_id"], r["matches"]) == (1855, 153, 19)


def test_normalize_career_seasons_calendar_year_and_epoca():
    import pandas as pd
    from scouting.tables import normalize_career_seasons
    df = pd.DataFrame([
        {"season": "2024/25", "country": "PT", "season_source_id": 154, "games_path": "/j?epoca_id=154&equipa_id=1"},
        {"season": "2025", "country": "BR", "season_source_id": None, "games_path": "/j?epoca_id=154&equipa_id=2"},
        {"season": "2025", "country": "PT", "season_source_id": None, "games_path": "/j?epoca_id=154&equipa_id=3"},
    ])
    out = normalize_career_seasons(df, {"BR"})
    assert list(out["season"]) == ["2024/25", "2025/26", "2024/25"]
    assert list(out["season_source_id"]) == [154, 154, 154]   # games pages keep the zerozero epoca
    assert list(out["calendar_year_row"]) == [False, True, False]


def _games_page(summary, matches):
    head = "<thead><tr><th>Competição</th><th>J</th><th>M</th></tr></thead>"
    srows = "".join(
        f'<tr><td class="text"><a href="/jogador/x/1/jogos?competicao_id={cid}&epoca_id=154">{name}</a></td>'
        f"<td>{j}</td><td>{m}</td></tr>" for cid, name, j, m in summary)
    mrows = "".join(
        f'<tr><td>V</td><td>2024/09/01</td><td><a href="/pais/x">x</a><a href="/edition.php?id_edicao={ed}">D1</a></td>'
        f'<td>J1</td><td class="text"><a href="{team}">T</a></td><td>(C)</td><td>o</td><td>1-0</td><td>{mins}</td></tr>'
        for ed, team, mins in matches)
    return (f'<div id="player_summary"><table>{head}<tbody>{srows}</tbody></table></div>'
            f'<table class="zztable zz-table stats"><tbody>{mrows}</tbody></table>')


def test_club_league_minutes_splits_clubs_and_skips_youth_and_cups():
    from scouting.parsing.zerozero import club_league_minutes
    html = _games_page(
        [(1, "Segunda Liga", 2, 180), (2, "Liga 3", 1, 90), (3, "Liga Next Gen", 1, 90), (4, "Taça de Portugal", 1, 90)],
        [(10, "/equipa/leixoes/7?epoca_id=154", 90), (10, "/equipa/leixoes/7?epoca_id=154", 90),
         (20, "/equipa/other/8?epoca_id=154", 90), (30, "/equipa/leixoes/7?epoca_id=154", 90),
         (40, "/equipa/leixoes/7?epoca_id=154", 90)])
    assert club_league_minutes(html, 7) == {"Segunda Liga": 180}   # Liga 3 at club 8, Next Gen, cup excluded
    assert club_league_minutes(html, 8) == {"Liga 3": 90}


def test_club_league_minutes_slug_only_first_team_by_name():
    from scouting.parsing.zerozero import club_league_minutes
    html = _games_page([(1, "Liga Portuguesa", 1, 90)], [(10, "/equipa/gil-vicente?epoca_id=152", 90)])
    html = html.replace('">T</a>', '">Gil Vicente</a>')
    assert club_league_minutes(html, 11, "Gil Vicente") == {"Liga Portuguesa": 90}
