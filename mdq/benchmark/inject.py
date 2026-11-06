"""Controlled error injection with fixed seeds.

inject(frames, error_type, seed) -> (corrupted frames, labels, events)

frames: {source_id: split-adjusted prices} for a clean, reconciled slice.
Every injection goes into ONE source (picked at random per event), so the
other source stays a clean reference, as with a real vendor error. Events in
the same security are kept >= MIN_GAP sessions apart, and away from the series
edges, so that labels do not overlap.

Labels: one row per (security_id, date, source_id) that was altered, plus the
error_type and injected parameters. `primary` marks the rows a detector is
expected to find (recall is measured on these): for most types every altered
row; for an unapplied split only the ex-date (the earlier history is altered
too, so flags there are not false positives); for drift only days whose
offset is >= DRIFT_MATERIAL_BPS.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

ERROR_TYPES = ["spike_x10", "spike_div100", "small_spike", "stale_run", "missing_days",
               "unapplied_split", "swapped_ohlc", "duplicate_rows", "cross_source_drift"]

PRICE_COLS = ["open", "high", "low", "close"]
MIN_GAP = 15
EDGE = 25
DEFAULT_RATE = 0.004          # point errors per row per security (~5 per 5-year series)
DRIFT_MATERIAL_BPS = 10.0     # drift days at/above this offset are labelled as errors
SPLIT_RATIOS = [2, 3, 4, 5, 10, 20]
SWAP_PAIRS = [("high", "low"), ("open", "high"), ("close", "low"), ("open", "close")]


def _positions(rng, n, k, gap=MIN_GAP, edge=EDGE, span=1) -> list[int]:
    """Up to k start positions in [edge, n-edge-span), pairwise >= gap+span apart."""
    cand = rng.permutation(np.arange(edge, max(edge, n - edge - span)))
    chosen: list[int] = []
    for c in cand:
        if all(abs(c - x) >= gap + span for x in chosen):
            chosen.append(int(c))
            if len(chosen) == k:
                break
    return sorted(chosen)


def inject(frames: dict[str, pd.DataFrame], error_type: str, seed: int,
           rate: float = DEFAULT_RATE, events: pd.DataFrame | None = None):
    if error_type not in ERROR_TYPES:
        raise ValueError(error_type)
    rng = np.random.default_rng(seed)
    srcs = list(frames)
    out = {s: f.sort_values(["security_id", "date"]).reset_index(drop=True).copy()
           for s, f in frames.items()}
    labels, new_events, extra_rows = [], [], {s: [] for s in srcs}
    drop_idx = {s: [] for s in srcs}
    secs = sorted(set.intersection(*[set(f["security_id"]) for f in out.values()]))

    for sec in secs:
        idx = {s: out[s].index[out[s]["security_id"] == sec].to_numpy() for s in srcs}
        n = min(len(v) for v in idx.values())
        if error_type == "unapplied_split":
            k, span = 1, 1
        elif error_type == "cross_source_drift":
            k, span = max(1, int(round(rate * n / 4))), 40
        elif error_type == "stale_run":
            k, span = max(1, int(round(rate * n))), 7
        elif error_type == "missing_days":
            k, span = max(1, int(round(rate * n))), 3
        else:
            k, span = max(1, int(round(rate * n))), 1
        for pos in _positions(rng, n, k, span=span):
            src = srcs[int(rng.integers(len(srcs)))]
            f, ii = out[src], idx[src]
            i = ii[pos]
            date = f.at[i, "date"]

            def lab(d, primary=True, **params):
                labels.append({"security_id": sec, "date": d, "source_id": src,
                               "error_type": error_type, "primary": primary,
                               "params": json.dumps(params)})

            if error_type == "spike_x10":
                f.at[i, "close"] *= 10
                lab(date, factor=10)
            elif error_type == "spike_div100":
                f.at[i, "close"] /= 100
                lab(date, factor=0.01)
            elif error_type == "small_spike":
                m = float(rng.uniform(0.05, 0.15)) * (1 if rng.random() < 0.5 else -1)
                f.loc[i, PRICE_COLS] *= 1 + m
                lab(date, move=round(m, 4))
            elif error_type == "stale_run":
                L = int(rng.integers(3, 7))
                for j in ii[pos + 1: pos + 1 + L]:
                    f.loc[j, PRICE_COLS + ["volume"]] = f.loc[i, PRICE_COLS + ["volume"]].to_numpy()
                    lab(f.at[j, "date"], run=L)
            elif error_type == "missing_days":
                L = int(rng.integers(1, 4))
                for j in ii[pos: pos + L]:
                    drop_idx[src].append(j)
                    lab(f.at[j, "date"], length=L)
            elif error_type == "unapplied_split":
                ratio = int(rng.choice(SPLIT_RATIOS))
                before = ii[:pos]
                f.loc[before, PRICE_COLS] *= ratio
                f.loc[before, "volume"] /= ratio
                declared = bool(rng.random() < 0.5)
                if declared:
                    new_events.append({"security_id": sec, "action_type": "split", "ex_date": date,
                                       "ratio": float(ratio), "amount": np.nan,
                                       "source_id": "benchmark", "details": "injected"})
                lab(date, ratio=ratio, declared=declared)
                for j in before:  # every earlier row is wrong too, but the event is the ex-date
                    lab(f.at[j, "date"], primary=False, ratio=ratio)
            elif error_type == "swapped_ohlc":
                for _ in range(10):
                    a, b = SWAP_PAIRS[int(rng.integers(len(SWAP_PAIRS)))]
                    if not np.isclose(f.at[i, a], f.at[i, b]):
                        break
                f.loc[i, [a, b]] = f.loc[i, [b, a]].to_numpy().copy()  # copy: avoid view aliasing
                lab(date, swapped=f"{a}<->{b}")
            elif error_type == "duplicate_rows":
                row = f.loc[[i]].copy()
                conflicting = bool(rng.random() < 0.5)
                if conflicting:
                    row["close"] *= 1 + float(rng.uniform(0.01, 0.05))
                extra_rows[src].append(row)
                lab(date, conflicting=conflicting)
            elif error_type == "cross_source_drift":
                L = int(rng.integers(20, 41))
                D = float(rng.uniform(30, 80))
                win = ii[pos: pos + L]
                offs = np.linspace(D / len(win), D, len(win))
                sign = 1 if rng.random() < 0.5 else -1
                for j, o in zip(win, offs):
                    f.loc[j, PRICE_COLS] *= 1 + sign * o / 1e4
                    lab(f.at[j, "date"], primary=bool(o >= DRIFT_MATERIAL_BPS),
                        offset_bps=round(sign * o, 2), window=L)

    for s in srcs:
        if drop_idx[s]:
            out[s] = out[s].drop(index=drop_idx[s])
        if extra_rows[s]:
            out[s] = pd.concat([out[s]] + extra_rows[s], ignore_index=True)
        out[s] = out[s].reset_index(drop=True)
    lab_df = pd.DataFrame(labels, columns=["security_id", "date", "source_id", "error_type",
                                          "primary", "params"])
    lab_df["date"] = pd.to_datetime(lab_df["date"]).astype("datetime64[ns]")
    ev = events.copy() if events is not None else pd.DataFrame(
        columns=["security_id", "action_type", "ex_date", "ratio", "amount", "source_id", "details"])
    if new_events:
        ev = pd.concat([ev, pd.DataFrame(new_events)], ignore_index=True)
    if len(ev):
        ev["ex_date"] = pd.to_datetime(ev["ex_date"]).astype("datetime64[ns]")
    return out, lab_df, ev
