"""Source A: Yahoo Finance chart API (prices + split/dividend events)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mdq import config
from mdq.sources.http import fetch_json, SourceRejected

SOURCE_ID = "yahoo"
_TZ = "America/New_York"


def _epoch(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def url_for(ticker: str, start: str = config.START_DATE, end: str = config.END_DATE) -> str:
    p1 = _epoch(start)
    p2 = _epoch(end) + 86400  # period2 is exclusive
    return (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")


def cache_path(ticker: str) -> Path:
    return config.DATA_RAW / SOURCE_ID / f"{ticker}.json"


def fetch(ticker: str, refresh: bool = False) -> dict:
    return fetch_json(url_for(ticker), cache_path(ticker), refresh=refresh)


def _result(payload: dict) -> dict:
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise SourceRejected(f"yahoo error: {chart['error']}")
    res = chart.get("result") or []
    if not res:
        raise SourceRejected("yahoo: empty result")
    return res[0]


def parse_prices(payload: dict, ticker: str) -> pd.DataFrame:
    """Return one row per session: date, open, high, low, close, adj_close, volume.

    Yahoo timestamps are the session open in UTC; convert to the exchange
    date. OHLC are split-adjusted; adj_close is split+dividend adjusted.
    """
    res = _result(payload)
    ts = res.get("timestamp") or []
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    n = len(ts)

    def col(name, src=quote):
        vals = src.get(name) if isinstance(src, dict) else src
        if vals is None:
            return np.full(n, np.nan)
        return np.array([np.nan if v is None else v for v in vals], dtype="float64")

    df = pd.DataFrame({
        "security_id": ticker,
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(_TZ).tz_localize(None).normalize(),
        "open": col("open"), "high": col("high"), "low": col("low"), "close": col("close"),
        "adj_close": col(None, adj) if adj is not None else np.full(n, np.nan),
        "volume": col("volume"),
    })
    df["date"] = df["date"].astype("datetime64[ns]")
    return df


def parse_events(payload: dict, ticker: str) -> pd.DataFrame:
    """Split and dividend events -> corporate_actions rows."""
    res = _result(payload)
    ev = res.get("events") or {}
    rows = []
    for item in (ev.get("splits") or {}).values():
        num, den = float(item["numerator"]), float(item["denominator"])
        rows.append({"security_id": ticker, "action_type": "split",
                     "ex_date": item["date"], "ratio": num / den, "amount": np.nan,
                     "source_id": SOURCE_ID, "details": item.get("splitRatio", f"{num}:{den}")})
    for item in (ev.get("dividends") or {}).values():
        rows.append({"security_id": ticker, "action_type": "dividend",
                     "ex_date": item["date"], "ratio": np.nan, "amount": float(item["amount"]),
                     "source_id": SOURCE_ID, "details": ""})
    cols = ["security_id", "action_type", "ex_date", "ratio", "amount", "source_id", "details"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df["ex_date"] = (pd.to_datetime(df["ex_date"], unit="s", utc=True)
                         .dt.tz_convert(_TZ).dt.tz_localize(None).dt.normalize()
                         .astype("datetime64[ns]"))
    else:
        df["ex_date"] = pd.Series(dtype="datetime64[ns]")
    return df.sort_values(["ex_date", "action_type"]).reset_index(drop=True)


def parse_meta(payload: dict) -> dict:
    meta = _result(payload).get("meta", {})
    return {"yahoo_exchange": meta.get("exchangeName"), "currency": meta.get("currency"),
            "instrument_type": meta.get("instrumentType"), "long_name": meta.get("longName")}
