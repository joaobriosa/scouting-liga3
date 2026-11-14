"""Polite, cached HTTP client shared by all sources.

Every page is stored under data/raw/<source>/<cache_key>.html before being
returned. A page that is already cached is never requested again, which also
makes every collection script resumable: re-running it skips what is on disk.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from scouting import config

log = logging.getLogger(__name__)

BLOCK_MARKERS = ("cf-challenge", "Just a moment...", "cf_chl_opt")


class BlockedError(RuntimeError):
    """Raised when the source looks like it is blocking or rate-limiting us."""


class PageError(RuntimeError):
    """A single page keeps failing on the server side (HTTP 500/502/504).
    Not a block: callers may skip the page and carry on."""


class OfflineError(RuntimeError):
    """SCOUTING_OFFLINE=1 and the page is not in the cache (no request is made)."""


class PoliteClient:
    def __init__(
        self,
        source: str,
        raw_dir: Path = config.RAW_DIR,
        min_delay: float = config.MIN_DELAY_S,
        max_delay: float = config.MAX_DELAY_S,
        long_pause_every: int = config.LONG_PAUSE_EVERY,
        long_pause_s: float = config.LONG_PAUSE_S,
        user_agent: str = config.USER_AGENT,
    ) -> None:
        self.source = source
        self.cache_dir = raw_dir / source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "_index.jsonl"
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.long_pause_every = long_pause_every
        self.long_pause_s = long_pause_s
        self.requests_made = 0
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Language": "pt-PT,pt;q=0.9"},
            follow_redirects=True,
            timeout=30.0,
        )

    # -- cache -------------------------------------------------------------
    def cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.html"

    def is_cached(self, cache_key: str) -> bool:
        return self.cache_path(cache_key).exists()

    def _write_cache(self, cache_key: str, url: str, final_url: str, text: str) -> None:
        path = self.cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)  # atomic: an interrupted run never leaves half a page
        entry = {
            "cache_key": cache_key,
            "url": url,
            "final_url": final_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- network -----------------------------------------------------------
    def _wait_turn(self) -> None:
        if self._last_request_at is None:
            return
        # Long rest every N network requests (the source resets connections
        # after ~100 requests in a row).
        if self.requests_made and self.requests_made % self.long_pause_every == 0:
            log.info("Long pause: %d s after %d requests", self.long_pause_s, self.requests_made)
            time.sleep(self.long_pause_s)
            return
        delay = random.uniform(self.min_delay, self.max_delay)
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def get(self, url: str, cache_key: str) -> str:
        """Return the page text, from cache if available, else from the network."""
        path = self.cache_path(cache_key)
        if path.exists():
            return path.read_text(encoding="utf-8")
        if os.environ.get("SCOUTING_OFFLINE") == "1":
            raise OfflineError(f"Not cached and SCOUTING_OFFLINE=1: {url} ({cache_key})")

        for attempt in range(2):
            self._wait_turn()
            log.info("GET %s", url)
            self.requests_made += 1
            try:
                resp = self._client.get(url)
            except httpx.TransportError as exc:
                # Connection reset/timeout: back off once, then stop (never hammer).
                self._last_request_at = time.monotonic()
                if attempt == 0:
                    log.warning("%s on %s, backing off 60 s", type(exc).__name__, url)
                    time.sleep(60)
                    continue
                raise BlockedError(f"{type(exc).__name__} twice for {url}") from exc
            self._last_request_at = time.monotonic()

            if resp.status_code in (429, 503) and attempt == 0:
                log.warning("HTTP %s on %s, backing off 60 s", resp.status_code, url)
                time.sleep(60)
                continue
            if resp.status_code in (403, 429, 503):
                raise BlockedError(f"HTTP {resp.status_code} for {url}")
            if resp.status_code in (500, 502, 504):
                if attempt == 0:
                    log.warning("HTTP %s on %s, backing off 60 s", resp.status_code, url)
                    time.sleep(60)
                    continue
                raise PageError(f"HTTP {resp.status_code} twice for {url}")
            resp.raise_for_status()
            if any(m in resp.text for m in BLOCK_MARKERS):
                raise BlockedError(f"Challenge page returned for {url}")

            self._write_cache(cache_key, url, str(resp.url), resp.text)
            return resp.text
        raise BlockedError(f"Gave up on {url}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
