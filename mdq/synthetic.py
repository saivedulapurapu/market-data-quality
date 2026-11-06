"""Synthetic market data with known properties (all test fixtures use this;
no real vendor data is committed to the repository).

Source "a" is a geometric Brownian motion with internally consistent OHLC
bars; source "b" is the same tape rounded to cents with tiny independent
noise, mimicking two vendors of the same consolidated prints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.calendar import sessions


def make_series(security_id: str, start: str = "2022-01-03", end: str = "2023-12-29",
                seed: int = 0, s0: float = 100.0, vol: float = 0.02,
                drift: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = sessions(start, end)
    n = len(dates)
    r = rng.normal(drift, vol, n)
    r[0] = 0.0
    close = s0 * np.exp(np.cumsum(r))
    prev = np.concatenate([[s0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, vol / 3, n))
    hi_noise = np.abs(rng.normal(0, vol / 2, n))
    lo_noise = np.abs(rng.normal(0, vol / 2, n))
    high = np.maximum(open_, close) * np.exp(hi_noise)
    low = np.minimum(open_, close) * np.exp(-lo_noise)
    volume = np.round(rng.lognormal(np.log(5e6), 0.35, n))
    return pd.DataFrame({
        "security_id": security_id, "date": dates.astype("datetime64[ns]"),
        "open": open_, "high": high, "low": low, "close": close,
        "adj_close": close, "volume": volume,
    })


def second_source(df: pd.DataFrame, seed: int = 1, noise_bps: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in ["open", "high", "low", "close"]:
        out[c] = np.round(out[c] * (1 + rng.normal(0, noise_bps / 1e4, len(out))), 2)
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = np.round(out["volume"] * rng.uniform(0.97, 1.03, len(out)))
    out["adj_close"] = np.nan
    return out


def make_panel(n_securities: int = 4, start: str = "2022-01-03", end: str = "2023-12-29",
               seed: int = 0) -> dict[str, pd.DataFrame]:
    """Two-source panel: {"a": prices, "b": prices} with source_id/adjustment set."""
    a_frames, b_frames = [], []
    rng = np.random.default_rng(seed)
    for i in range(n_securities):
        s0 = float(rng.uniform(20, 400))
        a = make_series(f"SYN{i}", start, end, seed=seed * 1000 + i, s0=s0,
                        vol=float(rng.uniform(0.01, 0.03)))
        a_frames.append(a)
        b_frames.append(second_source(a, seed=seed * 1000 + i + 500))
    a = pd.concat(a_frames, ignore_index=True)
    b = pd.concat(b_frames, ignore_index=True)
    a["source_id"], b["source_id"] = "a", "b"
    a["adjustment"] = b["adjustment"] = "split"
    return {"a": a, "b": b}


def populate_store(con, seed: int = 21) -> dict:
    """Fill an empty store with a small synthetic two-source dataset that has
    known defects, then run the checks. Used by tests and the dashboard smoke
    test. Returns the dates of the planted defects."""
    from mdq import adjust, pipeline, store
    panel = make_panel(n_securities=3, start="2022-01-03", end="2022-12-30", seed=seed)
    a, b = panel["a"].copy(), panel["b"].copy()
    a["source_id"], b["source_id"] = "yahoo", "nasdaq"
    ex = a.loc[a.security_id == "SYN2", "date"].iloc[150]
    ev = pd.DataFrame({"security_id": ["SYN2", "SYN0"], "action_type": ["split", "dividend"],
                       "ex_date": [ex, a.date.iloc[40]], "ratio": [4.0, np.nan],
                       "amount": [np.nan, 0.0], "source_id": "yahoo", "details": ["4:1", ""]})
    i = b.index[b.security_id == "SYN1"][100]
    b.loc[i, "close"] *= 1.3                      # bad print in one source
    j = b.index[b.security_id == "SYN0"][60]
    b.loc[j, "open"] = b.loc[j, "high"] * 1.05    # OHLC violation
    a["adj_close"] = a["close"]
    ids = ["SYN0", "SYN1", "SYN2"]
    secs = pd.DataFrame({"security_id": ids, "ticker": ids, "name": "Synthetic", "exchange": "NYSE",
                         "asset_class": "stock", "yahoo_symbol": ids, "nasdaq_symbol": ids,
                         "currency": "USD"})
    store.insert(con, "securities", secs)
    for f in (a, b):
        store.insert(con, "prices", f)
        sp = ev[ev.action_type == "split"]
        store.insert(con, "prices", pd.concat([adjust.to_raw(g, sp[sp.security_id == s])
                                               for s, g in f.groupby("security_id")]))
    store.insert(con, "corporate_actions", ev)
    summary = pipeline.process(con)
    return {"summary": summary, "bad_print": b.loc[i, "date"], "ohlc": b.loc[j, "date"], "split": ex}
