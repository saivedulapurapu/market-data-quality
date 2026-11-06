"""Standard quality-issue record shared by every check."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

ISSUE_COLUMNS = ["security_id", "date", "source_id", "check_name", "category",
                 "severity", "value", "expected", "details"]

# check_name -> category. Categories are the six families in docs/CHECKS.md.
CHECK_CATEGORY = {
    "schema_duplicate_key": "schema",
    "schema_null_key": "schema",
    "schema_invalid_value": "schema",
    "schema_type": "schema",
    "missing_day": "completeness",
    "non_trading_day": "completeness",
    "missing_field": "completeness",
    "ohlc_inconsistent": "validity",
    "non_positive_price": "validity",
    "negative_volume": "validity",
    "zero_volume": "validity",
    "stale_close": "staleness",
    "outlier_zscore": "outlier",
    "outlier_mad": "outlier",
    "outlier_iforest": "outlier",
    "unadjusted_split": "corporate_action",
    "suspected_unadjusted_split": "corporate_action",
    "split_not_visible_in_raw": "corporate_action",
    "dividend_adjustment_mismatch": "corporate_action",
    "unexplained_adjustment": "corporate_action",
    "recon_close_mismatch": "reconciliation",
    "recon_missing_in_source": "reconciliation",
}

SEVERITIES = ["info", "warning", "error", "critical"]


def empty_issues() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in ISSUE_COLUMNS})
    df["date"] = pd.Series(dtype="datetime64[ns]")
    df["value"] = pd.Series(dtype="float64")
    df["expected"] = pd.Series(dtype="float64")
    return df


def make_issues(rows: pd.DataFrame, check_name: str, severity: str,
                value=None, expected=None, details=None, source_id=None) -> pd.DataFrame:
    """Build issues from a frame that has security_id, date (and source_id).

    value/expected may be column names, arrays or scalars. details may be a
    string, a list of dicts/strings (one per row) or None.
    """
    if rows is None or len(rows) == 0:
        return empty_issues()
    n = len(rows)

    def resolve(x):
        if x is None:
            return np.full(n, np.nan)
        if isinstance(x, str) and x in rows.columns:
            return rows[x].to_numpy(dtype="float64")
        if np.isscalar(x):
            return np.full(n, float(x))
        return np.asarray(x, dtype="float64")

    if details is None:
        det = [""] * n
    elif isinstance(details, str):
        det = [details] * n
    else:
        det = [d if isinstance(d, str) else json.dumps(d, default=str) for d in details]

    src = rows["source_id"].to_numpy() if source_id is None and "source_id" in rows.columns \
        else np.full(n, source_id, dtype=object)
    out = pd.DataFrame({
        "security_id": rows["security_id"].to_numpy(),
        "date": pd.to_datetime(rows["date"].to_numpy()),
        "source_id": src,
        "check_name": check_name,
        "category": CHECK_CATEGORY.get(check_name, "other"),
        "severity": severity,
        "value": resolve(value),
        "expected": resolve(expected),
        "details": det,
    })
    return out[ISSUE_COLUMNS]


def concat_issues(frames) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return empty_issues()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out[ISSUE_COLUMNS]
