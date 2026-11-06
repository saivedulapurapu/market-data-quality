"""Source B: Nasdaq.com historical quotes API (OHLCV only, no events).

Quirk found in Stage 0: short date windows return rows=null, so the full
5-year window is always requested in one call.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mdq import config
from mdq.sources.http import fetch_json, SourceRejected

SOURCE_ID = "nasdaq"


def url_for(ticker: str, asset_class: str, start: str = config.START_DATE,
            end: str = config.END_DATE) -> str:
    ac = "etf" if asset_class == "etf" else "stocks"
    return (f"https://api.nasdaq.com/api/quote/{ticker}/historical"
            f"?assetclass={ac}&fromdate={start}&todate={end}&limit=9999")


def cache_path(ticker: str) -> Path:
    return config.DATA_RAW / SOURCE_ID / f"{ticker}.json"


def fetch(ticker: str, asset_class: str, refresh: bool = False) -> dict:
    return fetch_json(url_for(ticker, asset_class), cache_path(ticker), refresh=refresh)


def _num(s) -> float:
    if s is None:
        return np.nan
    s = str(s).replace("$", "").replace(",", "").strip()
    if s in ("", "N/A", "--"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_prices(payload: dict, ticker: str) -> pd.DataFrame:
    data = payload.get("data") or {}
    table = data.get("tradesTable") or {}
    rows = table.get("rows")
    if rows is None:
        status = payload.get("status") or {}
        raise SourceRejected(f"nasdaq: no rows for {ticker} ({status})")
    df = pd.DataFrame({
        "security_id": ticker,
        "date": pd.to_datetime([r.get("date") for r in rows], format="%m/%d/%Y", errors="coerce"),
        "open": [_num(r.get("open")) for r in rows],
        "high": [_num(r.get("high")) for r in rows],
        "low": [_num(r.get("low")) for r in rows],
        "close": [_num(r.get("close")) for r in rows],
        "volume": [_num(r.get("volume")) for r in rows],
    })
    df["adj_close"] = np.nan
    df["date"] = df["date"].astype("datetime64[ns]")
    return df.sort_values("date").reset_index(drop=True)
