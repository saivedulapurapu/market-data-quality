"""Benchmark driver: clean slice -> val/test split -> tune on val -> report on test.

Everything is seeded:
  SPLIT_SEED      securities shuffled into validation / test halves
  VAL_SEED_BASE   injection seed for validation scenarios (+ error-type index)
  TEST_SEED_BASE  injection seed for test scenarios
"""
from __future__ import annotations

import json
import time
from dataclasses import replace, asdict

import numpy as np
import pandas as pd

from mdq import config, store
from mdq.benchmark import evaluate as ev
from mdq.benchmark.inject import ERROR_TYPES, inject
from mdq.config import CheckParams, DEFAULT_PARAMS
from mdq.scoring import NOT_SCORED

SPLIT_SEED = 42
VAL_SEED_BASE = 1000
TEST_SEED_BASE = 2000

Z_GRID = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
IF_GRID = [0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80]
STALE_GRID = [2, 3, 4, 5]
RECON_GRID = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
SPLIT_GRID = [0.02, 0.05, 0.10, 0.15, 0.20]
SPIKE_TYPES = ["spike_x10", "spike_div100", "small_spike"]


# Fixed screening parameters that define the "clean" slice. They are kept
# separate from the tuned defaults so that tuning cannot change the slice it
# is evaluated on (no circularity).
SCREEN_PARAMS = CheckParams(stale_min_run=3, zscore_threshold=6.0, mad_threshold=6.0,
                            iforest_threshold=0.70, split_log_tolerance=0.10,
                            recon_tolerance_bps=10.0)


def load_clean_slice(con) -> tuple[dict, pd.DataFrame, dict]:
    """Securities with no scored, non-info issue under SCREEN_PARAMS."""
    from mdq.checks.runner import run_checks
    prices = store.read(con, "SELECT * FROM prices WHERE adjustment='split'")
    ca = store.read(con, "SELECT * FROM corporate_actions")
    by_src = {s: prices[prices.source_id == s].reset_index(drop=True) for s in ["yahoo", "nasdaq"]}
    issues, _ = run_checks(by_src, ca, SCREEN_PARAMS,
                           detectors=["schema", "completeness", "validity", "staleness", "mad",
                                      "split", "dividend", "reconciliation"])
    scored = issues[~issues.check_name.isin(NOT_SCORED) & (issues.severity != "info")]
    dirty = sorted(scored.security_id.dropna().unique())
    keep = sorted(set(prices.security_id) - set(dirty))
    p = prices[prices.security_id.isin(keep)].copy()
    p["adj_close"] = np.nan  # dividend check is not benchmarked; see BENCHMARK.md
    frames = {s: p[p.source_id == s].reset_index(drop=True) for s in ["yahoo", "nasdaq"]}
    events = store.read(con, "SELECT * FROM corporate_actions WHERE action_type='split'")
    events = events[events.security_id.isin(keep)]
    return frames, events, {"excluded": dirty, "included": keep}


def split_securities(secs, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    s = list(rng.permutation(sorted(secs)))
    h = len(s) // 2
    return sorted(s[:h]), sorted(s[h:])


def subset(frames, secs):
    return {k: f[f.security_id.isin(secs)].reset_index(drop=True) for k, f in frames.items()}


def scenarios(frames, events, seed_base, types=ERROR_TYPES):
    return {t: inject(frames, t, seed=seed_base + i, events=events)
            for i, t in enumerate(ERROR_TYPES) if t in types}


def tune(val: dict, params: CheckParams) -> tuple[CheckParams, dict]:
    sweeps = {}
    # outliers: one fit per scenario, thresholds swept on the scores, pooled
    parts = {k: [] for k in ["zscore", "mad", "iforest"]}
    for t in SPIKE_TYPES:
        frames, labels, _ = val[t]
        sc = ev.outlier_scores(frames, params)
        for det, grid in (("zscore", Z_GRID), ("mad", Z_GRID), ("iforest", IF_GRID)):
            parts[det].append(ev.sweep_scores(sc, labels, det, grid))
    counts = ["flags", "tp_flags", "labels", "detected"]
    for det, lst in parts.items():
        tot = pd.concat(lst).groupby("threshold")[counts].sum().reset_index()
        sweeps[det] = pd.DataFrame([{"threshold": r.threshold,
                                     **ev.prf(r.tp_flags, r.flags, r.detected, r.labels)}
                                    for r in tot.itertuples()])
    p = replace(params,
                zscore_threshold=float(ev.best(sweeps["zscore"], "threshold")),
                mad_threshold=float(ev.best(sweeps["mad"], "threshold")),
                iforest_threshold=float(ev.best(sweeps["iforest"], "threshold")))
    sweeps["staleness"] = ev.sweep_param([val["stale_run"]], p, "stale_min_run", STALE_GRID, "staleness")
    p = replace(p, stale_min_run=int(ev.best(sweeps["staleness"], "value")))
    sweeps["reconciliation"] = ev.sweep_param([val["cross_source_drift"], val["small_spike"]], p,
                                              "recon_tolerance_bps", RECON_GRID, "reconciliation")
    p = replace(p, recon_tolerance_bps=float(ev.best(sweeps["reconciliation"], "value")))
    sweeps["split"] = ev.sweep_param([val["unapplied_split"]], p, "split_log_tolerance",
                                     SPLIT_GRID, "split")
    p = replace(p, split_log_tolerance=float(ev.best(sweeps["split"], "value")))
    return p, sweeps


def run_all(frames, events, params, seed_base) -> dict:
    out = {}
    for t, (f, labels, e) in scenarios(frames, events, seed_base).items():
        out[t] = ev.evaluate_scenario(f, labels, e, params)
    return out


def results_table(res: dict) -> pd.DataFrame:
    rows = []
    for t, by_det in res.items():
        for d, m in by_det.items():
            if d.startswith("_"):
                continue
            rows.append({"error_type": t, "detector": d, **m})
    return pd.DataFrame(rows)


def main():
    con = store.connect(config.DB_PATH, read_only=True)
    frames, events, info = load_clean_slice(con)
    con.close()
    payload = benchmark(frames, events, info)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "benchmark_results.json").write_text(json.dumps(payload, indent=2, default=str))
    from mdq.benchmark import report
    report.write(payload)
    t = payload["tuned_params"]
    print(f"benchmark done in {payload['runtime_seconds']}s; tuned: "
          f"z={t['zscore_threshold']} mad={t['mad_threshold']} if={t['iforest_threshold']} "
          f"stale={t['stale_min_run']} recon={t['recon_tolerance_bps']} split={t['split_log_tolerance']}")
    return payload


def benchmark(frames: dict, events: pd.DataFrame, info: dict,
              base_params: CheckParams = DEFAULT_PARAMS) -> dict:
    """Tune on the validation half, evaluate on the test half; returns the
    JSON-serialisable payload (no file I/O)."""
    t0 = time.time()
    val_secs, test_secs = split_securities(info["included"])
    val = scenarios(subset(frames, val_secs), events[events.security_id.isin(val_secs)], VAL_SEED_BASE)
    tuned, sweeps = tune(val, base_params)
    test_frames = subset(frames, test_secs)
    test_events = events[events.security_id.isin(test_secs)]
    test = run_all(test_frames, test_events, tuned, TEST_SEED_BASE)
    val_res = run_all(subset(frames, val_secs), events[events.security_id.isin(val_secs)], tuned,
                      VAL_SEED_BASE)
    table = results_table(test)
    payload = {
        "generated_by": "python -m mdq benchmark",
        "seeds": {"split": SPLIT_SEED, "val_base": VAL_SEED_BASE, "test_base": TEST_SEED_BASE},
        "slice": {"included": info["included"], "excluded": info["excluded"],
                  "validation": val_secs, "test": test_secs,
                  "rows_per_source": {s: int(len(f)) for s, f in frames.items()},
                  "test_rows_per_source": {s: int(len(f)) for s, f in test_frames.items()},
                  "date_range": [str(frames["yahoo"].date.min().date()), str(frames["yahoo"].date.max().date())]},
        "screen_params": SCREEN_PARAMS.to_dict(),
        "tuned_params": tuned.to_dict(),
        "test": table.to_dict(orient="records"),
        "validation": results_table(val_res).to_dict(orient="records"),
        "blame_test": {t: r["_blame"] for t, r in test.items()},
        "sweeps": {k: v.to_dict(orient="records") for k, v in sweeps.items()},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    return json.loads(json.dumps(payload, default=str))
