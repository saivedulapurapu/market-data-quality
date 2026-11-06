"""Staleness: close unchanged for N consecutive sessions.

A run of k identical closes (k >= min_run) flags the k-1 repeats; the first
close of the run is a legitimate print and is not flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.issues import concat_issues, make_issues


def check_staleness(prices: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    out = []
    for (_, _), g in prices.groupby(["security_id", "source_id"], sort=False):
        g = g.sort_values("date")
        c = np.round(g["close"].to_numpy(dtype="float64"), 4)
        same = np.concatenate([[False], c[1:] == c[:-1]])
        run_id = np.cumsum(~same)
        run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
        flag = same & (run_len >= min_run)
        if flag.any():
            rows = g[flag]
            lens = run_len[flag]
            sev = np.where(lens >= 5, "error", "warning")
            for s in ("warning", "error"):
                m = sev == s
                if m.any():
                    out.append(make_issues(rows[m], "stale_close", s, value=c[flag][m],
                                           details=[{"run_length": int(x)} for x in lens[m]]))
    return concat_issues(out)
