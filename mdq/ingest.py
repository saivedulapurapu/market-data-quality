"""Fetch raw payloads (cached) and load them into the DuckDB store."""
from __future__ import annotations

import logging

import pandas as pd

from mdq import adjust, config, store
from mdq.issues import concat_issues
from mdq.sources import nasdaq, yahoo
from mdq.sources.http import SourceRejected

log = logging.getLogger("mdq.ingest")


def securities_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"security_id": t, "ticker": t, "name": n, "exchange": ex, "asset_class": ac,
         "yahoo_symbol": t, "nasdaq_symbol": t, "currency": "USD"}
        for t, n, ex, ac in config.UNIVERSE
    ])


def sources_frame() -> pd.DataFrame:
    return pd.DataFrame(config.SOURCES)


def fetch_all(refresh: bool = False, tickers: list[str] | None = None) -> dict:
    """Download every (source, ticker) not yet cached. Stops on the first
    rejection so that a block is reported instead of worked around."""
    done = {"yahoo": 0, "nasdaq": 0}
    for t, _, _, ac in config.UNIVERSE:
        if tickers and t not in tickers:
            continue
        for src, fn in (("yahoo", lambda: yahoo.fetch(t, refresh)),
                        ("nasdaq", lambda: nasdaq.fetch(t, ac, refresh))):
            try:
                fn()
                done[src] += 1
            except SourceRejected as exc:
                log.error("source rejected request, stopping: %s", exc)
                raise
        log.info("fetched %s", t)
    return done


def build_frames(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Parse cached payloads into store-shaped frames (no network)."""
    price_frames, events = [], []
    for t, _, _, ac in config.UNIVERSE:
        if tickers and t not in tickers:
            continue
        ypay = yahoo.fetch(t) if yahoo.cache_path(t).exists() else None
        npay = nasdaq.fetch(t, ac) if nasdaq.cache_path(t).exists() else None
        ev = yahoo.parse_events(ypay, t) if ypay else pd.DataFrame()
        if len(ev):
            events.append(ev)
        splits = ev[ev["action_type"] == "split"] if len(ev) else ev
        for src, pay, parser in (("yahoo", ypay, yahoo.parse_prices),
                                 ("nasdaq", npay, nasdaq.parse_prices)):
            if pay is None:
                continue
            p = parser(pay, t)
            p["source_id"] = src
            p["adjustment"] = "split"
            price_frames.append(p)
            price_frames.append(adjust.to_raw(p, splits))
    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    ca = pd.concat(events, ignore_index=True) if events else pd.DataFrame(
        columns=["security_id", "action_type", "ex_date", "ratio", "amount", "source_id", "details"])
    return {"securities": securities_frame(), "sources": sources_frame(),
            "prices": prices, "corporate_actions": ca}


def load(con, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Insert frames (Pandera-validated). Returns schema issues logged."""
    issues = []
    for table in ["securities", "sources", "prices", "corporate_actions"]:
        df = frames[table]
        if table == "prices" and len(df):
            for (_, _), g in df.groupby(["source_id", "adjustment"], sort=False):
                issues.append(store.insert(con, "prices", g.reset_index(drop=True)))
        else:
            issues.append(store.insert(con, table, df))
    return concat_issues(issues)
