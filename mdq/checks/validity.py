"""Validity: OHLC consistency, non-positive prices, negative / zero volume."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.issues import concat_issues, make_issues

REL_TOL = 1e-6  # float noise allowance for OHLC comparisons


def check_validity(prices: pd.DataFrame, zero_volume_severity: str = "warning") -> pd.DataFrame:
    p = prices
    o, h, l, c, v = (p[k].to_numpy(dtype="float64") for k in ["open", "high", "low", "close", "volume"])
    out = []

    px = np.vstack([o, h, l, c])
    with np.errstate(invalid="ignore"):
        nonpos = np.any(px <= 0, axis=0)
    if nonpos.any():
        rows = p[nonpos]
        out.append(make_issues(rows, "non_positive_price", "critical",
                               value=np.nanmin(px[:, nonpos], axis=0), expected=0.0,
                               details="price field <= 0"))

    with np.errstate(invalid="ignore"):
        tol_h = np.abs(h) * REL_TOL
        viol = {
            "low>high": l > h + tol_h,
            "open>high": o > h + tol_h,
            "close>high": c > h + tol_h,
            "open<low": o < l - tol_h,
            "close<low": c < l - tol_h,
        }
    any_v = np.zeros(len(p), dtype=bool)
    for m in viol.values():
        any_v |= m
    any_v &= ~nonpos
    if any_v.any():
        idx = np.flatnonzero(any_v)
        det = [{"violations": [k for k, m in viol.items() if m[i]],
                "open": o[i], "high": h[i], "low": l[i], "close": c[i]} for i in idx]
        # value = close, expected = the nearest bound of the [low, high] range
        exp = np.clip(c[idx], np.minimum(l[idx], h[idx]), np.maximum(l[idx], h[idx]))
        out.append(make_issues(p.iloc[idx], "ohlc_inconsistent", "error",
                               value=c[idx], expected=exp, details=det))

    with np.errstate(invalid="ignore"):
        neg = v < 0
        zero = v == 0
    if neg.any():
        out.append(make_issues(p[neg], "negative_volume", "critical", value=v[neg], expected=0.0))
    if zero.any():
        out.append(make_issues(p[zero], "zero_volume", zero_volume_severity, value=0.0,
                               details="zero volume on a session for a liquid name"))
    return concat_issues(out)
