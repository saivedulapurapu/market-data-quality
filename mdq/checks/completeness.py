"""Completeness: missing sessions vs the NYSE calendar, rows on non-sessions,
missing fields."""
from __future__ import annotations

import pandas as pd

from mdq.calendar import sessions
from mdq.issues import concat_issues, make_issues

FIELDS = ["open", "high", "low", "close", "volume"]


def check_completeness(prices: pd.DataFrame, bounds: dict | None = None) -> pd.DataFrame:
    """prices: one adjustment basis, any number of securities/sources.

    bounds: optional {security_id: (first, last)} expected coverage window.
    Default is each series' own first..last date (so a gap in the middle is
    found, but a series that starts late is not; the pipeline passes the
    union of both sources as bounds).
    """
    out = []
    for (sec, src), g in prices.groupby(["security_id", "source_id"], sort=False):
        dates = pd.DatetimeIndex(g["date"].dropna().unique())
        if len(dates) == 0:
            continue
        lo, hi = bounds.get(sec, (dates.min(), dates.max())) if bounds else (dates.min(), dates.max())
        expected = sessions(lo, hi)
        missing = expected.difference(dates)
        if len(missing):
            rows = pd.DataFrame({"security_id": sec, "date": missing, "source_id": src})
            out.append(make_issues(rows, "missing_day", "error",
                                   details="expected NYSE session absent from source"))
        valid_all = sessions(min(dates.min(), lo), max(dates.max(), hi))
        extra = dates.difference(valid_all)
        if len(extra):
            rows = pd.DataFrame({"security_id": sec, "date": extra, "source_id": src})
            out.append(make_issues(rows, "non_trading_day", "error",
                                   details="row dated on a non-session (weekend/holiday)"))
    nulls = prices[FIELDS].isna()
    bad = prices[nulls.any(axis=1)]
    if len(bad):
        det = [{"missing": [f for f in FIELDS if nulls.at[i, f]]} for i in bad.index]
        out.append(make_issues(bad, "missing_field", "error",
                               value=nulls.loc[bad.index].sum(axis=1).to_numpy(), details=det))
    return concat_issues(out)
