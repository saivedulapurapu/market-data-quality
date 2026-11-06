"""Quality score, 0-100 per security per session, and per dataset.

    deduction(sec, day) = sum over distinct checks firing that day of
                          max severity weight of that check
    score(sec, day)     = max(0, 100 - deduction)

Severity weights: critical 60, error 30, warning 10, info 0. Each check
counts once per security-day, whichever source(s) it fired on, so a
problem seen by several sources is not double counted, while two different
problems on the same day are.

Only one outlier detector is scored (the primary one: MAD, chosen on the
benchmark), so the three detectors do not triple-count the same spike.
The others are still stored in quality_issues for comparison.

Dataset score per day = mean over securities. Overall dataset score = mean
over all security-days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.calendar import sessions
from mdq.config import SEVERITY_WEIGHTS

NOT_SCORED = {"outlier_zscore", "outlier_iforest"}


def score(issues: pd.DataFrame, universe_bounds: dict, weights: dict = SEVERITY_WEIGHTS,
          not_scored: set = NOT_SCORED) -> pd.DataFrame:
    """universe_bounds: {security_id: (first, last)} sessions to score."""
    grid = pd.concat([pd.DataFrame({"security_id": s, "date": sessions(lo, hi)})
                      for s, (lo, hi) in universe_bounds.items()], ignore_index=True) \
        if universe_bounds else pd.DataFrame(columns=["security_id", "date"])
    iss = issues.dropna(subset=["security_id", "date"])
    iss = iss[~iss["check_name"].isin(not_scored)].copy()
    if len(iss):
        iss["w"] = iss["severity"].map(weights).fillna(0.0)
        per_check = iss.groupby(["security_id", "date", "check_name"])["w"].max().reset_index()
        agg = per_check.groupby(["security_id", "date"]).agg(
            deduction=("w", "sum"), n_issues=("check_name", "size")).reset_index()
    else:
        agg = pd.DataFrame(columns=["security_id", "date", "deduction", "n_issues"])
    agg["date"] = pd.to_datetime(agg["date"]).astype("datetime64[ns]")
    grid["date"] = pd.to_datetime(grid["date"]).astype("datetime64[ns]")
    out = grid.merge(agg, on=["security_id", "date"], how="left")
    out["deduction"] = out["deduction"].fillna(0.0).astype("float64")
    out["n_issues"] = out["n_issues"].fillna(0).astype("int64")
    out["score"] = np.clip(100.0 - out["deduction"], 0.0, 100.0)
    return out[["security_id", "date", "score", "n_issues", "deduction"]]


def dataset_score(scores: pd.DataFrame) -> dict:
    by_day = scores.groupby("date")["score"].mean()
    return {"overall": float(scores["score"].mean()) if len(scores) else float("nan"),
            "worst_day": None if by_day.empty else str(by_day.idxmin().date()),
            "worst_day_score": None if by_day.empty else float(by_day.min()),
            "pct_security_days_perfect": float((scores["score"] == 100).mean() * 100) if len(scores) else 0.0}
