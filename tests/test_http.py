import httpx

from scouting import http as shttp


def _client(tmp_path, handler, sleeps):
    c = shttp.PoliteClient("t", raw_dir=tmp_path, min_delay=0, max_delay=0,
                           long_pause_every=3, long_pause_s=180)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_long_pause_every_n_requests_and_cache(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(shttp.time, "sleep", lambda s: sleeps.append(s))
    c = _client(tmp_path, lambda req: httpx.Response(200, text="ok"), sleeps)
    for i in range(7):
        c.get(f"https://x/{i}", f"p/{i}")
    c.get("https://x/0", "p/0")  # cached: no request
    assert c.requests_made == 7
    assert sleeps.count(180) == 2  # before request 4 and before request 7


def test_stops_after_second_consecutive_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(shttp.time, "sleep", lambda s: None)

    def handler(req):
        raise httpx.ReadError("reset")

    c = _client(tmp_path, handler, [])
    try:
        c.get("https://x/a", "p/a")
    except shttp.BlockedError:
        pass
    else:
        raise AssertionError("expected BlockedError")
    assert c.requests_made == 2 and not c.is_cached("p/a")
