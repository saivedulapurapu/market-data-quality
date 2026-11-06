"""Run every detector over a set of per-source frames (shared by the
pipeline and the benchmark)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq import schemas
from mdq.config import CheckParams, DEFAULT_PARAMS
from mdq.issues import concat_issues
from mdq.checks.completeness import check_completeness
from mdq.checks.validity import check_validity
from mdq.checks.staleness import check_staleness
from mdq.checks.outliers import check_zscore, check_mad, check_iforest
from mdq.checks.corporate_actions import check_splits, check_dividend_adjustment
from mdq.checks.reconciliation import reconcile

DETECTORS = ["schema", "completeness", "validity", "staleness", "zscore", "mad", "iforest",
             "split", "dividend", "reconciliation"]

# detector -> check names it emits
DETECTOR_CHECKS = {
    "schema": ["schema_duplicate_key", "schema_null_key", "schema_invalid_value", "schema_type"],
    "completeness": ["missing_day", "non_trading_day", "missing_field"],
    "validity": ["ohlc_inconsistent", "non_positive_price", "negative_volume", "zero_volume"],
    "staleness": ["stale_close"],
    "zscore": ["outlier_zscore"],
    "mad": ["outlier_mad"],
    "iforest": ["outlier_iforest"],
    "split": ["unadjusted_split", "suspected_unadjusted_split"],
    "dividend": ["dividend_adjustment_mismatch", "unexplained_adjustment"],
    "reconciliation": ["recon_close_mismatch", "recon_missing_in_source"],
}
CHECK_DETECTOR = {c: d for d, cs in DETECTOR_CHECKS.items() for c in cs}


def coverage_bounds(frames: dict[str, pd.DataFrame]) -> dict:
    allp = pd.concat([f[["security_id", "date"]] for f in frames.values()], ignore_index=True)
    g = allp.dropna().groupby("security_id")["date"]
    return {s: (lo, hi) for s, lo, hi in zip(g.min().index, g.min(), g.max())}


def run_checks(frames: dict[str, pd.DataFrame], events: pd.DataFrame | None = None,
               params: CheckParams = DEFAULT_PARAMS, detectors=None,
               validate_schema: bool = True,
               corroborate_sources: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """frames: {source_id: split-adjusted prices}. Returns (issues, recon comparison)."""
    detectors = set(detectors or DETECTORS)
    issues, clean = [], {}
    for src, f in frames.items():
        f = f.copy()
        f["source_id"] = src
        if "adjustment" not in f:
            f["adjustment"] = "split"
        if "adj_close" not in f:
            f["adj_close"] = float("nan")
        if validate_schema:
            f, si = schemas.validate(f.reset_index(drop=True), schemas.price_schema)
            if "schema" in detectors:
                issues.append(si)
        clean[src] = f.sort_values(["security_id", "date"]).reset_index(drop=True)

    bounds = coverage_bounds(clean) if clean else {}
    p = params
    for src, f in clean.items():
        if "completeness" in detectors:
            issues.append(check_completeness(f, bounds))
        if "validity" in detectors:
            issues.append(check_validity(f, p.zero_volume_severity))
        if "staleness" in detectors:
            issues.append(check_staleness(f, p.stale_min_run))
        if "zscore" in detectors:
            issues.append(check_zscore(f, p.zscore_threshold, p.outlier_window, p.outlier_min_periods))
        if "mad" in detectors:
            issues.append(check_mad(f, p.mad_threshold, p.outlier_window, p.outlier_min_periods))
        if "iforest" in detectors:
            issues.append(check_iforest(f, p.iforest_threshold, p.outlier_window,
                                        p.iforest_n_estimators, p.iforest_seed))
        if "split" in detectors:
            issues.append(check_splits(f, events, p.split_candidate_ratios, p.split_log_tolerance,
                                       p.split_persist_window))
        if "dividend" in detectors:
            issues.append(check_dividend_adjustment(f, events, p.dividend_factor_tolerance))
    comp = None
    if "reconciliation" in detectors and len(clean) >= 2:
        a, b = list(clean.values())[:2]
        ri, comp = reconcile(a, b, p.recon_tolerance_bps, p.recon_blame_margin)
        issues.append(ri)
    out = concat_issues(issues)
    if corroborate_sources and len(clean) >= 2:
        out = corroborate(out, n_sources=len(clean))
    return out, comp


# Checks whose firing on *every* independent source, with the sources agreeing
# on price, indicates a genuine market event rather than a vendor error.
CORROBORABLE = {"stale_close", "outlier_zscore", "outlier_mad", "outlier_iforest"}


def corroborate(issues: pd.DataFrame, n_sources: int) -> pd.DataFrame:
    """Downgrade corroborated flags to severity 'info' (kept, not dropped).

    A flag is corroborated when the same check fires on the same
    security/day in all n_sources sources and reconciliation found no
    close mismatch that day.
    """
    if issues.empty:
        return issues
    out = issues.copy()
    mism = out.loc[out.check_name == "recon_close_mismatch", ["security_id", "date"]]
    mism_keys = set(zip(mism.security_id, mism.date))
    m = out.check_name.isin(CORROBORABLE)
    cnt = out[m].groupby(["security_id", "date", "check_name"])["source_id"].transform("nunique")
    corr = pd.Series(False, index=out.index)
    corr.loc[cnt.index] = cnt >= n_sources
    corr &= np.array([(s, d) not in mism_keys for s, d in zip(out.security_id, out.date)], dtype=bool)
    out.loc[corr, "severity"] = "info"
    out.loc[corr, "details"] = [
        (d[:-1] + ', "corroborated": true}') if isinstance(d, str) and d.endswith("}")
        else '{"corroborated": true}' for d in out.loc[corr, "details"]]
    return out
