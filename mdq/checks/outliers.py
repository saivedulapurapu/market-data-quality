"""Return-based outlier detection: classic z-score, robust MAD, Isolation Forest.

All three work on log close-to-close returns per security/source.

Spike score (z-score and MAD). A bad print at t produces a large return
into t (r_in = r_t) and an opposite large return out of it
(r_out = r_{t+1}). Each return is standardised with location/scale estimated
from the trailing window r_{t-W..t-1} (so the suspect point never
contaminates its own reference), and

    score_t = min(|z_in|, |z_out|)   if z_in and z_out have opposite signs
            = 0                      otherwise

A genuine one-way move (earnings gap, split) has only one large return and
scores ~0; it is left to the corporate-action check. z-score and MAD differ
only in the estimator: mean/std vs median/(1.4826 * MAD). That makes the
comparison a direct test of robustness: a spike inflates the trailing std for
the next W days and masks nearby errors, but it barely moves the MAD.

Isolation Forest is fit per series on the features [r_in, r_out,
log(volume / rolling median volume), log(range / rolling median range)] and
gives anomaly score s = -score_samples in (0, 1]. It is unsupervised and is not
told about the spike structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest

from mdq.issues import concat_issues, make_issues


def _trailing(x: np.ndarray, window: int, min_periods: int, robust: bool):
    """Location/scale of x[t-window..t-1] for each t (NaN-aware), backfilled
    for the first points that lack min_periods of history."""
    n = len(x)
    pad = np.concatenate([np.full(window, np.nan), x])
    win = sliding_window_view(pad, window)[:n]  # win[t] = x[t-window..t-1]
    cnt = np.sum(~np.isnan(win), axis=1)
    with np.errstate(all="ignore"), _quiet():
        if robust:
            loc = np.nanmedian(win, axis=1)
            scale = 1.4826 * np.nanmedian(np.abs(win - loc[:, None]), axis=1)
        else:
            loc = np.nanmean(win, axis=1)
            scale = np.nanstd(win, axis=1, ddof=1)
    ok = cnt >= min_periods
    loc = np.where(ok, loc, np.nan)
    scale = np.where(ok & (scale > 0), scale, np.nan)
    loc = pd.Series(loc).bfill().to_numpy()
    scale = pd.Series(scale).bfill().to_numpy()
    return loc, scale


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def spike_scores(close: np.ndarray, window: int = 63, min_periods: int = 20,
                 robust: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (score, r_in, r_out) for one series sorted by date."""
    with np.errstate(all="ignore"):
        lc = np.log(np.where(close > 0, close, np.nan))
    r_in = np.concatenate([[np.nan], np.diff(lc)])
    r_out = np.concatenate([r_in[1:], [np.nan]])
    loc, scale = _trailing(r_in, window, min_periods, robust)
    z_in = (r_in - loc) / scale
    z_out = (r_out - loc) / scale
    score = np.where(np.sign(z_in) != np.sign(z_out), np.fmin(np.abs(z_in), np.abs(z_out)), 0.0)
    # first/last point: one of the two returns is unknown, so a spike cannot be
    # told apart from a genuine move; not scored (re-evaluated on the next run)
    score = np.where(np.isnan(z_in) | np.isnan(z_out), 0.0, score)
    score = np.nan_to_num(score, nan=0.0)
    return score, r_in, r_out


def _flag(prices, check_name, threshold, scorer) -> pd.DataFrame:
    out = []
    for (_, _), g in prices.groupby(["security_id", "source_id"], sort=False):
        g = g.sort_values("date")
        score, r_in, r_out = scorer(g)
        m = score > threshold
        if m.any():
            rows = g[m]
            det = [{"r_in": round(float(a), 6), "r_out": round(float(b), 6), "close": float(c)}
                   for a, b, c in zip(r_in[m], r_out[m], rows["close"])]
            sev = np.where(score[m] > 2 * threshold, "error", "warning")
            for s in ("warning", "error"):
                k = sev == s
                if k.any():
                    out.append(make_issues(rows[k], check_name, s, value=score[m][k],
                                           expected=threshold,
                                           details=[d for d, kk in zip(det, k) if kk]))
    return concat_issues(out)


def check_zscore(prices, threshold=6.0, window=63, min_periods=20) -> pd.DataFrame:
    return _flag(prices, "outlier_zscore", threshold,
                 lambda g: spike_scores(g["close"].to_numpy(dtype="float64"), window, min_periods,
                                        robust=False))


def check_mad(prices, threshold=6.0, window=63, min_periods=20) -> pd.DataFrame:
    return _flag(prices, "outlier_mad", threshold,
                 lambda g: spike_scores(g["close"].to_numpy(dtype="float64"), window, min_periods,
                                        robust=True))


def iforest_scores(g: pd.DataFrame, window: int = 63, n_estimators: int = 200,
                   seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = g["close"].to_numpy(dtype="float64")
    with np.errstate(all="ignore"):
        lc = np.log(np.where(close > 0, close, np.nan))
        r_in = np.concatenate([[np.nan], np.diff(lc)])
        r_out = np.concatenate([r_in[1:], [np.nan]])
        vol = g["volume"].to_numpy(dtype="float64")
        rng = (g["high"] - g["low"]).to_numpy(dtype="float64") / close
        vmed = pd.Series(vol).rolling(window, min_periods=5, center=True).median().to_numpy()
        rmed = pd.Series(rng).rolling(window, min_periods=5, center=True).median().to_numpy()
        f_vol = np.log((vol + 1) / (vmed + 1))
        f_rng = np.log((np.abs(rng) + 1e-6) / (np.abs(rmed) + 1e-6))
    X = np.column_stack([r_in, r_out, f_vol, f_rng])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < 10:
        return np.zeros(len(X)), r_in, r_out
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, contamination="auto")
    model.fit(X)
    return -model.score_samples(X), r_in, r_out


def check_iforest(prices, threshold=0.70, window=63, n_estimators=200, seed=7) -> pd.DataFrame:
    return _flag(prices, "outlier_iforest", threshold,
                 lambda g: iforest_scores(g, window, n_estimators, seed))
