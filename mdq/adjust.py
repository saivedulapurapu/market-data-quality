"""Split adjustment helpers.

Both sources deliver split-adjusted history. An "as traded" (raw) series is
reconstructed by undoing the split adjustment with the declared split events:

    raw_price(t)  = adj_price(t)  * F(t)
    raw_volume(t) = adj_volume(t) / F(t)
    F(t) = product of split ratios with ex_date > t
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]


def split_factor(dates: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """Cumulative split factor F(t) for each date (splits: ex_date, ratio)."""
    d = pd.to_datetime(dates).to_numpy()
    f = np.ones(len(d))
    if splits is None or len(splits) == 0:
        return f
    for ex, ratio in zip(pd.to_datetime(splits["ex_date"]).to_numpy(), splits["ratio"].to_numpy()):
        f = np.where(d < ex, f * float(ratio), f)
    return f


def to_raw(prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the as-traded series for one security's split-adjusted prices."""
    out = prices.copy()
    f = split_factor(out["date"], splits)
    for c in PRICE_COLS:
        out[c] = out[c] * f
    out["volume"] = out["volume"] / f
    out["adj_close"] = np.nan
    out["adjustment"] = "raw"
    return out


def apply_splits(raw: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Inverse of to_raw: split-adjust an as-traded series."""
    out = raw.copy()
    f = split_factor(out["date"], splits)
    for c in PRICE_COLS:
        out[c] = out[c] / f
    out["volume"] = out["volume"] * f
    out["adjustment"] = "split"
    return out
