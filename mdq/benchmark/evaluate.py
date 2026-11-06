"""Precision / recall / F1 of each detector against injected labels.

Matching is exact on (security_id, date, source_id) for per-source detectors,
and on (security_id, date) for reconciliation (a cross-source check) and for
the end-to-end "pipeline" alert set.

Each error type is evaluated in its own scenario (only that error injected),
so a detector's precision for an error type = flags that hit an injected
label / all flags it raised in that scenario. False positives therefore
include alerts on genuine market data in the clean slice.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from mdq.checks.runner import DETECTOR_CHECKS, run_checks
from mdq.checks.outliers import iforest_scores, spike_scores
from mdq.config import CheckParams
from mdq.scoring import NOT_SCORED

BENCH_DETECTORS = ["schema", "completeness", "validity", "staleness", "zscore", "mad",
                   "iforest", "split", "reconciliation"]
SOURCE_AGNOSTIC = {"reconciliation", "pipeline"}


def prf(tp_flags: int, n_flags: int, tp_labels: int, n_labels: int) -> dict:
    p = tp_flags / n_flags if n_flags else (1.0 if n_labels == 0 else 0.0)
    r = tp_labels / n_labels if n_labels else float("nan")
    f = 2 * p * r / (p + r) if (p + r) > 0 and not np.isnan(r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "flags": n_flags, "tp_flags": tp_flags,
            "labels": n_labels, "detected": tp_labels}


def _keys(df: pd.DataFrame, with_source: bool) -> set:
    cols = ["security_id", "date", "source_id"] if with_source else ["security_id", "date"]
    return set(map(tuple, df[cols].astype(object).itertuples(index=False, name=None)))


def match(flags: pd.DataFrame, labels: pd.DataFrame, with_source: bool) -> dict:
    """Precision: flags hitting any altered row. Recall: primary rows found."""
    fk = _keys(flags, with_source)
    altered = _keys(labels, with_source)
    primary = _keys(labels[labels["primary"].astype(bool)] if "primary" in labels else labels, with_source)
    return prf(len(fk & altered), len(fk), len(primary & fk), len(primary))


def detector_flags(issues: pd.DataFrame, detector: str) -> pd.DataFrame:
    if detector == "pipeline":
        m = (~issues.check_name.isin(NOT_SCORED)) & (issues.severity != "info")
        return issues[m]
    return issues[issues.check_name.isin(DETECTOR_CHECKS[detector])]


def evaluate_scenario(frames, labels, events, params: CheckParams) -> dict:
    """Run every detector once (no corroboration, so raw detector behaviour
    is measured) plus the end-to-end pipeline alert set (with corroboration)."""
    raw, _ = run_checks(frames, events, params, detectors=BENCH_DETECTORS,
                        corroborate_sources=False)
    piped, _ = run_checks(frames, events, params, detectors=BENCH_DETECTORS,
                          corroborate_sources=True)
    res = {}
    for d in BENCH_DETECTORS:
        res[d] = match(detector_flags(raw, d), labels, with_source=d not in SOURCE_AGNOSTIC)
    res["pipeline"] = match(detector_flags(piped, "pipeline"), labels, with_source=False)
    res["_blame"] = blame_accuracy(raw, labels)
    return res


def blame_accuracy(issues: pd.DataFrame, labels: pd.DataFrame) -> dict:
    rec = issues[issues.check_name == "recon_close_mismatch"]
    if "primary" in labels:
        labels = labels[labels["primary"].astype(bool)]
    lab = labels.drop_duplicates(["security_id", "date"]).set_index(["security_id", "date"])["source_id"]
    hit = rec[[k in lab.index for k in zip(rec.security_id, rec.date)]]
    if hit.empty:
        return {"n": 0, "correct": 0, "wrong": 0, "undetermined": 0}
    truth = [lab.loc[k] for k in zip(hit.security_id, hit.date)]
    blamed = hit["source_id"].tolist()
    correct = sum(b == t for b, t in zip(blamed, truth))
    und = sum(b is None or (isinstance(b, float) and np.isnan(b)) for b in blamed)
    return {"n": len(hit), "correct": correct, "undetermined": und, "wrong": len(hit) - correct - und}


# ---------- threshold sweeps (validation) ----------
def outlier_scores(frames, params: CheckParams) -> pd.DataFrame:
    """Per-point scores of the three outlier detectors (fit once, threshold later)."""
    rows = []
    for src, f in frames.items():
        f = f.drop_duplicates(["security_id", "date"]).sort_values(["security_id", "date"])
        for sec, g in f.groupby("security_id", sort=False):
            c = g["close"].to_numpy(dtype="float64")
            z, _, _ = spike_scores(c, params.outlier_window, params.outlier_min_periods, robust=False)
            m, _, _ = spike_scores(c, params.outlier_window, params.outlier_min_periods, robust=True)
            i, _, _ = iforest_scores(g, params.outlier_window, params.iforest_n_estimators,
                                     params.iforest_seed)
            rows.append(pd.DataFrame({"security_id": sec, "date": g["date"].to_numpy(),
                                      "source_id": src, "zscore": z, "mad": m, "iforest": i}))
    return pd.concat(rows, ignore_index=True)


def sweep_scores(scores: pd.DataFrame, labels: pd.DataFrame, col: str, grid) -> pd.DataFrame:
    out = []
    for t in grid:
        fl = scores[scores[col] > t]
        out.append({"threshold": t, **match(fl, labels, with_source=True)})
    return pd.DataFrame(out)


def sweep_param(scenarios, params: CheckParams, field: str, grid, detector: str) -> pd.DataFrame:
    """Re-run one detector over pooled scenarios for each candidate value."""
    out = []
    for v in grid:
        p = replace(params, **{field: v})
        tpf = nf = tpl = nl = 0
        for frames, labels, events in scenarios:
            iss, _ = run_checks(frames, events, p, detectors=["schema", detector],
                                corroborate_sources=False)
            r = match(detector_flags(iss, detector), labels, with_source=detector not in SOURCE_AGNOSTIC)
            tpf += r["tp_flags"]; nf += r["flags"]; tpl += r["detected"]; nl += r["labels"]
        out.append({"value": v, **prf(tpf, nf, tpl, nl)})
    return pd.DataFrame(out)


def best(sweep: pd.DataFrame, key: str):
    s = sweep.sort_values(["f1", key], ascending=[False, True])
    return s.iloc[0][key]
