import httpx
import pytest

from scouting import http as shttp
from scouting.parsing.zerozero import in_main_squad
from scouting.report import DESTINATIONS_SECTION, write_main, write_section


def test_ranking_rewrite_keeps_destinations_section(tmp_path):
    f = tmp_path / "ranking.md"
    write_main(f, "# Ranking\n\nv1")
    write_section(f, f"{DESTINATIONS_SECTION}\n\nd1")
    write_main(f, "# Ranking\n\nv2")  # rank_season.py again
    text = f.read_text(encoding="utf-8")
    assert "v2" in text and "v1" not in text
    assert text.count(DESTINATIONS_SECTION) == 1 and "d1" in text
    assert text.index("v2") < text.index(DESTINATIONS_SECTION)


def test_destinations_rewrite_replaces_only_its_section(tmp_path):
    f = tmp_path / "ranking.md"
    write_main(f, "# Ranking\n\nv1")
    write_section(f, f"{DESTINATIONS_SECTION}\n\nd1")
    write_section(f, f"{DESTINATIONS_SECTION}\n\nd2")
    text = f.read_text(encoding="utf-8")
    assert "v1" in text and "d2" in text and "d1" not in text
    assert text.count(DESTINATIONS_SECTION) == 1


def test_in_main_squad():
    html = ('<div id="team_squad"><a href="/jogador/x/123?epoca_id=156">X</a></div>'
            '<table><a href="/jogador/y/456">Y</a></table>')
    assert in_main_squad(html, 123)
    assert not in_main_squad(html, 456)  # elsewhere on the page only
    assert not in_main_squad(html, 12)   # id prefix must not match


def test_offline_mode_never_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("SCOUTING_OFFLINE", "1")
    c = shttp.PoliteClient("t", raw_dir=tmp_path, min_delay=0, max_delay=0)
    c._client = httpx.Client(transport=httpx.MockTransport(lambda req: pytest.fail("request made")))
    (tmp_path / "t" / "p").mkdir(parents=True)
    (tmp_path / "t" / "p" / "a.html").write_text("cached", encoding="utf-8")
    assert c.get("https://x/a", "p/a") == "cached"
    with pytest.raises(shttp.OfflineError):
        c.get("https://x/b", "p/b")
    assert c.requests_made == 0
