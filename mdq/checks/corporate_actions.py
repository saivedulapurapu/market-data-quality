"""Corporate actions: unadjusted splits and dividend-adjustment verification.

Split check (on a series that is supposed to be split-adjusted)
  For each day, r_t = log(close_t / close_{t-1}). A split of ratio k left
  unadjusted shows r_t ~ -log(k) on the ex-date, and the new level persists:
  the median log close over [t, t+P) minus the median over [t-P, t) is also
  ~ -log(k). A day is a candidate when both are within tol of -log(k)
  (or +log(k) for a reverse split) for one of the candidate ratios.
    - candidate on a declared split ex-date with matching ratio -> unadjusted_split (critical)
    - candidate with no declared split                         -> suspected_unadjusted_split (error)

Dividend verification (Yahoo, which delivers adj_close)
  Adjustment factor f_t = adj_close_t / close_t. On a dividend ex-date
  f_{t-1} / f_t should equal 1 - D / close_{t-1}. A mismatch beyond tol is
  dividend_adjustment_mismatch. A factor step on a day with no dividend
  and no split is unexplained_adjustment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.issues import concat_issues, make_issues


def _median_shift(lc: np.ndarray, p: int) -> np.ndarray:
    s = pd.Series(lc)
    after = s[::-1].rolling(p, min_periods=max(2, p // 2)).median()[::-1].to_numpy()  # [t, t+p)
    before = s.shift(1).rolling(p, min_periods=max(2, p // 2)).median().to_numpy()  # [t-p, t)
    return after - before


def check_splits(prices: pd.DataFrame, events: pd.DataFrame | None,
                 ratios=(2, 3, 4, 5, 8, 10, 15, 20, 25, 50), tol: float = 0.10,
                 persist: int = 5) -> pd.DataFrame:
    splits = events[events["action_type"] == "split"] if events is not None and len(events) else None
    targets = np.log(np.array(sorted(set(ratios)), dtype="float64"))
    out = []
    for (sec, _), g in prices.groupby(["security_id", "source_id"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        close = g["close"].to_numpy(dtype="float64")
        with np.errstate(all="ignore"):
            lc = np.log(np.where(close > 0, close, np.nan))
        r = np.concatenate([[np.nan], np.diff(lc)])
        shift = _median_shift(lc, persist)
        declared = {}
        if splits is not None:
            s = splits[splits["security_id"] == sec]
            declared = dict(zip(pd.to_datetime(s["ex_date"]), s["ratio"].astype(float)))
        # distance of r and of the level shift to the nearest -log(k) / +log(k)
        cand_rows = []
        for sign in (-1.0, 1.0):
            d_r = np.abs(r[:, None] - sign * targets[None, :])
            d_s = np.abs(shift[:, None] - sign * targets[None, :])
            ok = (d_r < tol) & (d_s < tol)
            ok &= np.abs(r)[:, None] > tol  # must be a real jump, not noise
            for i, j in zip(*np.nonzero(ok)):
                k = float(np.exp(targets[j]))
                cand_rows.append((i, round(k if sign < 0 else 1.0 / k, 6), d_r[i, j]))
        best = {}
        for i, k, d in cand_rows:
            if i not in best or d < best[i][1]:
                best[i] = (k, d)
        for i, (k, _) in sorted(best.items()):
            date = g.loc[i, "date"]
            dk = declared.get(date)
            row = g.iloc[[i]]
            det = {"implied_ratio": round(float(np.exp(-r[i])), 4), "nearest_ratio": k,
                   "declared_ratio": dk, "prev_close": float(close[i - 1]), "close": float(close[i])}
            if dk is not None and abs(np.log(dk) - np.log(k)) < tol:
                out.append(make_issues(row, "unadjusted_split", "critical",
                                       value=float(np.exp(-r[i])), expected=1.0, details=[det]))
            else:
                out.append(make_issues(row, "suspected_unadjusted_split", "error",
                                       value=float(np.exp(-r[i])), expected=1.0, details=[det]))
    return concat_issues(out)


def check_dividend_adjustment(prices: pd.DataFrame, events: pd.DataFrame | None,
                              tol: float = 0.0005) -> pd.DataFrame:
    """prices must contain adj_close (rows without adj_close are skipped)."""
    p = prices.dropna(subset=["adj_close", "close"])
    if len(p) == 0:
        return concat_issues([])
    ev = events if events is not None else pd.DataFrame(columns=["security_id", "action_type",
                                                                  "ex_date", "amount", "ratio"])
    out = []
    for (sec, _), g in p.groupby(["security_id", "source_id"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        f = (g["adj_close"] / g["close"]).to_numpy(dtype="float64")
        step = np.concatenate([[np.nan], f[:-1] / f[1:]])  # f_{t-1}/f_t
        prev_close = np.concatenate([[np.nan], g["close"].to_numpy(dtype="float64")[:-1]])
        e = ev[ev["security_id"] == sec]
        divs = e[e["action_type"] == "dividend"].groupby("ex_date")["amount"].sum()
        split_dates = set(pd.to_datetime(e.loc[e["action_type"] == "split", "ex_date"]))
        dates = g["date"]
        is_div = dates.isin(divs.index).to_numpy()
        amount = dates.map(divs).to_numpy(dtype="float64")
        expected = np.where(is_div, 1 - amount / prev_close, 1.0)
        err = np.abs(step - expected)
        with np.errstate(invalid="ignore"):
            bad = err > tol
        bad[0] = False
        for i in np.flatnonzero(bad):
            if dates.iloc[i] in split_dates:
                continue
            name = "dividend_adjustment_mismatch" if is_div[i] else "unexplained_adjustment"
            det = {"factor_step": round(float(step[i]), 6), "expected_step": round(float(expected[i]), 6),
                   "dividend": None if np.isnan(amount[i]) else float(amount[i])}
            out.append(make_issues(g.iloc[[i]], name, "warning", value=float(step[i]),
                                   expected=float(expected[i]), details=[det]))
    return concat_issues(out)
