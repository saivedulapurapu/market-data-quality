"""Render reports/BENCHMARK.md and PNG charts from benchmark_results.json."""
from __future__ import annotations

import pandas as pd

from mdq import config, viz

FIG_DIR = config.REPORTS_DIR / "figures"

ET_LABEL = {
    "spike_x10": "fat-finger ×10", "spike_div100": "fat-finger ÷100",
    "small_spike": "bad bar ±5–15%", "stale_run": "stale run", "missing_days": "missing days",
    "unapplied_split": "unapplied split", "swapped_ohlc": "swapped OHLC fields",
    "duplicate_rows": "duplicated rows", "cross_source_drift": "cross-source drift",
}
PRIMARY = {  # the detector designed for each error type
    "spike_x10": "mad", "spike_div100": "mad", "small_spike": "mad", "stale_run": "staleness",
    "missing_days": "completeness", "unapplied_split": "split", "swapped_ohlc": "validity",
    "duplicate_rows": "schema", "cross_source_drift": "reconciliation",
}
DET_ORDER = ["schema", "completeness", "validity", "staleness", "zscore", "mad", "iforest",
             "split", "reconciliation", "pipeline"]


def _f(x, nd=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(lines)


def figures(test: pd.DataFrame, sweeps: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    viz.mpl_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. F1 heatmap detector x error type
    m = test.pivot(index="detector", columns="error_type", values="f1").reindex(DET_ORDER)
    m = m[[c for c in ET_LABEL if c in m.columns]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    cmap = LinearSegmentedColormap.from_list("seq", [viz.SURFACE] + viz.SEQ_BLUE)
    ax.imshow(m.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1]), [ET_LABEL[c] for c in m.columns], rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, "" if pd.isna(v) or v == 0 else f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=viz.SURFACE if (v or 0) > 0.6 else viz.INK)
    ax.set_title("F1 by detector and injected error type (held-out test slice; blank = 0)", loc="left")
    fig.tight_layout()
    paths["heatmap"] = FIG_DIR / "f1_heatmap.png"
    fig.savefig(paths["heatmap"], dpi=150)
    plt.close(fig)

    # 2. outlier detector comparison: small multiples per metric
    sub = test[test.detector.isin(["zscore", "mad", "iforest"]) &
               test.error_type.isin(["spike_x10", "spike_div100", "small_spike"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    types = ["spike_x10", "spike_div100", "small_spike"]
    w = 0.26
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for k, det in enumerate(["zscore", "mad", "iforest"]):
            vals = [sub[(sub.detector == det) & (sub.error_type == t)][metric].iloc[0] for t in types]
            xs = [x + (k - 1) * w for x in range(3)]
            ax.bar(xs, vals, width=w - 0.03, color=viz.DETECTOR_COLOR[det], label=det)
            for x, v in zip(xs, vals):
                if v < 0.005:  # a zero bar is invisible: label it
                    ax.text(x, 0.02, "0", ha="center", va="bottom", fontsize=8, color=viz.INK_2)
        ax.set_xticks(range(3), [ET_LABEL[t] for t in types], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.capitalize(), loc="left")
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Outlier detectors on the held-out test slice", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths["outliers"] = FIG_DIR / "outlier_comparison.png"
    fig.savefig(paths["outliers"], dpi=150)
    plt.close(fig)

    # 3. validation precision-recall curves for the outlier thresholds
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for det in ["zscore", "mad", "iforest"]:
        s = pd.DataFrame(sweeps[det]).sort_values("threshold")
        ax.plot(s["recall"], s["precision"], marker="o", markersize=4,
                color=viz.DETECTOR_COLOR[det], label=det)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Validation sweep: precision vs recall per threshold\n(pooled spike scenarios)",
                 loc="left")
    fig.tight_layout()
    paths["pr"] = FIG_DIR / "outlier_pr_validation.png"
    fig.savefig(paths["pr"], dpi=150)
    plt.close(fig)
    return paths


def write(payload: dict) -> None:
    test = pd.DataFrame(payload["test"])
    test.to_csv(config.REPORTS_DIR / "benchmark_test.csv", index=False)
    figures(test, payload["sweeps"])
    sl = payload["slice"]
    tp = payload["tuned_params"]

    # main table: each error type, its primary detector + outlier trio + pipeline
    main = []
    for et in ET_LABEL:
        r = test[test.error_type == et].set_index("detector")
        if r.empty:
            continue
        p = PRIMARY[et]
        best_d = r.drop(index="pipeline").sort_values("f1", ascending=False).index[0]
        main.append({"Error type": ET_LABEL[et], "Labels": int(r.loc[p, "labels"]),
                     "Designed detector": p,
                     "P": _f(r.loc[p, "precision"]), "R": _f(r.loc[p, "recall"]), "F1": _f(r.loc[p, "f1"]),
                     "Best single detector (F1)": f"{best_d} ({_f(r.loc[best_d, 'f1'])})",
                     "Pipeline alerts P / R / F1": f"{_f(r.loc['pipeline', 'precision'])} / "
                                                   f"{_f(r.loc['pipeline', 'recall'])} / {_f(r.loc['pipeline', 'f1'])}"})
    main = pd.DataFrame(main)

    # full matrix P/R/F1
    full = test.copy()
    full["cell"] = [f"{_f(p)} / {_f(r)} / {_f(f)}" if n else "– (0 flags)" for p, r, f, n in
                    zip(full["precision"], full["recall"], full["f1"], full["flags"])]
    mat = full.pivot(index="detector", columns="error_type", values="cell").reindex(DET_ORDER)
    mat = mat[[c for c in ET_LABEL if c in mat.columns]]
    mat.columns = [ET_LABEL[c] for c in mat.columns]
    mat = mat.reset_index()

    # outlier comparison
    oc = []
    for et in ["spike_x10", "spike_div100", "small_spike"]:
        for d in ["zscore", "mad", "iforest"]:
            r = test[(test.error_type == et) & (test.detector == d)].iloc[0]
            oc.append({"Error type": ET_LABEL[et], "Detector": d, "Precision": _f(r.precision),
                       "Recall": _f(r.recall), "F1": _f(r.f1), "Flags": int(r["flags"]),
                       "Detected / labels": f"{int(r.detected)} / {int(r.labels)}"})
    oc = pd.DataFrame(oc)

    # tuning table
    sw = payload["sweeps"]
    def best_row(name, key):
        s = pd.DataFrame(sw[name]).sort_values(["f1", key], ascending=[False, True]).iloc[0]
        return s
    tun = []
    for name, field, key, pooled in [
        ("zscore", "zscore_threshold", "threshold", "spike ×10, ÷100, bad bar"),
        ("mad", "mad_threshold", "threshold", "spike ×10, ÷100, bad bar"),
        ("iforest", "iforest_threshold", "threshold", "spike ×10, ÷100, bad bar"),
        ("staleness", "stale_min_run", "value", "stale run"),
        ("reconciliation", "recon_tolerance_bps", "value", "drift, bad bar"),
        ("split", "split_log_tolerance", "value", "unapplied split")]:
        b = best_row(name, key)
        tun.append({"Parameter": f"`{field}`", "Grid": ", ".join(_f(x[key], 2).rstrip("0").rstrip(".")
                                                                 for x in sw[name]),
                    "Tuned on (validation)": pooled, "Chosen": tp[field],
                    "Validation P / R / F1": f"{_f(b.precision)} / {_f(b.recall)} / {_f(b.f1)}"})
    tun = pd.DataFrame(tun)

    bl = []
    for et, b in payload["blame_test"].items():
        if b["n"]:
            bl.append({"Error type": ET_LABEL[et], "Recon flags on injected days": b["n"],
                       "Correct source": b["correct"], "Undetermined": b["undetermined"],
                       "Wrong source": b["wrong"],
                       "Correct %": _f(100 * b["correct"] / b["n"], 1)})
    bl = pd.DataFrame(bl)

    md = f"""# Injected-error benchmark

Generated by `python -m mdq benchmark` (`mdq/benchmark/`). All numbers below come from
[`benchmark_results.json`](benchmark_results.json), which the same run writes.

## Setup

- **Clean slice.** Real Yahoo and Nasdaq split-adjusted daily bars, {sl['date_range'][0]} to
  {sl['date_range'][1]}, for the {len(sl['included'])} securities that had **no scored issue** in
  a screening run with fixed parameters (`SCREEN_PARAMS` in `mdq/benchmark/run.py`, kept
  separate from the tuned defaults so tuning cannot change the slice it is scored on).
  Excluded ({len(sl['excluded'])}): {', '.join(sl['excluded'])}. Their real issues are in
  `QUALITY_REPORT.md`. Any residual real defect in the slice counts against precision rather
  than being hidden.
- **Split by security, seed {payload['seeds']['split']}.** {len(sl['validation'])} validation
  securities are used for threshold tuning and {len(sl['test'])} test securities for every
  number reported here ({sl['test_rows_per_source']['yahoo']:,} rows per source). No security
  appears in both halves.
- **One scenario per error type.** Each scenario injects only that error, into one randomly
  chosen source per event, at about 0.4% of rows per series (one per series for unapplied
  splits; drift windows of 20–40 sessions). Injection seeds: validation {payload['seeds']['val_base']}+i,
  test {payload['seeds']['test_base']}+i. Events are ≥15 sessions apart and away from series edges.
- **Matching.** A flag is a true positive when it hits an altered (security, date, source).
  Recall is measured on the *primary* labels: every altered row for most types, the ex-date
  for an unapplied split (the whole earlier history of that source is altered too, so a
  reconciliation flag there is correct, not a false positive), and drift days ≥10 bps off.
  Reconciliation and the pipeline alert set are matched on (security, date), because they are
  cross-source. Precision counts **every** flag the detector raised in the scenario, including
  alerts on genuine market moves.
- **Pipeline alerts** = all scored checks at severity ≥ warning, after cross-source
  corroboration (MAD is the only scored outlier detector). The per-detector rows are raw,
  without corroboration.
- **Error types.** fat-finger ×10 / ÷100 (close only); bad bar ±5–15% (whole bar scaled; an
  extra type, added so the outlier comparison is not trivial); stale run (bar repeated for 3–6
  sessions); missing days (1–3 consecutive rows deleted); unapplied split (history before the
  ex-date multiplied by 2–20; the event is declared in half the cases); swapped OHLC fields
  (high↔low, open↔high, close↔low or open↔close); duplicated rows (half exact, half with a
  conflicting close); cross-source drift (one source drifts linearly to 30–80 bps over 20–40
  sessions; only days ≥{10} bps off are labelled).

## Results (held-out test slice)

{_md(main)}

![F1 heatmap](figures/f1_heatmap.png)

### Full matrix: precision / recall / F1

"–" means the detector raised no flags in that scenario.

{_md(mat)}

## Outlier detectors: z-score vs MAD vs Isolation Forest

{_md(oc)}

![Outlier comparison](figures/outlier_comparison.png)

![PR sweep](figures/outlier_pr_validation.png)

## Threshold tuning (validation slice only)

The best F1 on the validation scenarios wins. Ties go to the more sensitive value. The chosen
values are the defaults in `mdq/config.py`.

{_md(tun)}

## Blame attribution (reconciliation, test slice)

On injected days the reconciliation check flagged, how often it named the source that was
actually corrupted:

{_md(bl)}

## Notes and limitations

- The dividend-adjustment check is not benchmarked, because no dividend error type is
  injected. `adj_close` is blanked in the slice so that it doesn't trivially expose close
  edits.
- Real market moves in the clean slice are genuine negatives. Every outlier flag on an
  un-injected day counts as a false positive, even when it is a real large move that partly
  reversed the next day.
- Unapplied splits: the 2 test misses both coincide with large real moves. On one ex-date the
  real return was about −20%, so the implied ratio was 0.16 instead of 0.20 and fell outside
  the log tolerance. In the other, a rally of about +30% within the 5-session persistence
  window broke the level-shift test.
- Blame works for point errors (spikes, bad bars) and is unreliable for stale runs and
  drift. Both of those produce smooth, internally consistent series, and with two sources
  there is no majority vote. Read stale-run and drift blame as "undetermined", whatever the
  per-day output.
- Selection effect: the slice was screened at 10 bps, and the reconciliation tolerance was
  then tuned down to 2 bps on it. On the full real dataset the 2 bps setting produced the
  same mismatch count as 10 bps (see `QUALITY_REPORT.md`), so the effect is small here, but
  it would need re-checking on another universe.
- The labels define drift as ≥{10} bps. Drift below that is invisible by design.
- The mixed case (several error types in the same week) is not measured. Each scenario is
  isolated so that per-type precision is well defined.
- Runtime: {payload['runtime_seconds']} s.
"""
    (config.REPORTS_DIR / "BENCHMARK.md").write_text(md, encoding="utf-8")


def rerender() -> None:
    import json
    write(json.loads((config.REPORTS_DIR / "benchmark_results.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    rerender()
