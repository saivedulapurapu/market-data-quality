"""Polite, cached HTTP fetching.

- Honest User-Agent (see config.USER_AGENT); never impersonates a browser.
- Responses cached on disk in data/raw/<source>/; a cached file is never
  re-downloaded unless refresh=True.
- Fixed pause between network requests; one request at a time.
- A non-200 response or a non-JSON body (e.g. a bot challenge page) raises
  SourceRejected so the caller stops instead of retrying around a block.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from mdq import config


class SourceRejected(RuntimeError):
    """The source refused the request (status, challenge page, bad payload)."""


_last_request = [0.0]


def fetch_json(url: str, cache_path: Path, refresh: bool = False,
               session: requests.Session | None = None) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    wait = config.REQUEST_PAUSE_SECONDS - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    sess = session or requests
    resp = sess.get(url, headers={"User-Agent": config.USER_AGENT,
                                  "Accept": "application/json"},
                    timeout=config.REQUEST_TIMEOUT_SECONDS)
    _last_request[0] = time.monotonic()
    if resp.status_code != 200:
        raise SourceRejected(f"{url} -> HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise SourceRejected(f"{url} -> non-JSON response (possible bot challenge)") from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload
