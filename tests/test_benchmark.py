"""Injection reproducibility and evaluation logic, on synthetic panels."""
import json

import numpy as np
import pandas as pd
import pytest

from mdq import synthetic
from mdq.benchmark import evaluate as ev
from mdq.benchmark.inject import ERROR_TYPES, inject
from mdq.benchmark.run import split_securities, subset
from mdq.config import CheckParams


@pytest.fixture(scope="module")
def bpanel():
    return synthetic.make_panel(n_securities=4, start="2022-01-03", end="2023-06-30", seed=5)


@pytest.mark.parametrize("etype", ERROR_TYPES)
def test_injection_reproducible(bpanel, etype):
    f1, l1, e1 = inject(bpanel, etype, seed=123)
    f2, l2, e2 = inject(bpanel, etype, seed=123)
    pd.testing.assert_frame_equal(l1, l2)
    for s in f1:
        pd.testing.assert_frame_equal(f1[s], f2[s])
    pd.testing.assert_frame_equal(e1.reset_index(drop=True), e2.reset_index(drop=True))
    assert len(l1) > 0
    f3, l3, _ = inject(bpanel, etype, seed=124)
    assert not l3[["security_id", "date", "source_id"]].equals(l1[["security_id", "date", "source_id"]])


def test_injection_does_not_mutate_input(bpanel):
    before = {s: f.copy() for s, f in bpanel.items()}
    inject(bpanel, "spike_x10", seed=1)
    for s in bpanel:
        pd.testing.assert_frame_equal(bpanel[s], before[s])


def test_spike_injection_values(bpanel):
    f, labels, _ = inject(bpanel, "spike_x10", seed=9)
    r = labels.iloc[0]
    src = bpanel[r.source_id]
    orig = src[(src.security_id == r.security_id) & (src.date == r.date)]["close"].iloc[0]
    new = f[r.source_id]
    val = new[(new.security_id == r.security_id) & (new.date == r.date)]["close"].iloc[0]
    assert val == pytest.approx(orig * 10)


def test_swap_actually_swaps(bpanel):
    f, labels, _ = inject(bpanel, "swapped_ohlc", seed=4)
    for r in labels.itertuples():
        a, b = json.loads(r.params)["swapped"].split("<->")
        o = bpanel[r.source_id].set_index(["security_id", "date"]).loc[(r.security_id, r.date)]
        n = f[r.source_id].set_index(["security_id", "date"]).loc[(r.security_id, r.date)]
        assert n[a] == pytest.approx(o[b]) and n[b] == pytest.approx(o[a])


def test_missing_and_duplicate_row_counts(bpanel):
    f, labels, _ = inject(bpanel, "missing_days", seed=2)
    assert sum(len(x) for x in f.values()) == sum(len(x) for x in bpanel.values()) - len(labels)
    f, labels, _ = inject(bpanel, "duplicate_rows", seed=2)
    assert sum(len(x) for x in f.values()) == sum(len(x) for x in bpanel.values()) + len(labels)


def test_unapplied_split_labels(bpanel):
    f, labels, events = inject(bpanel, "unapplied_split", seed=8)
    prim = labels[labels.primary]
    assert len(prim) == 4  # one per security
    assert (~labels.primary).sum() > 0  # earlier history marked as altered
    declared = [json.loads(p)["declared"] for p in prim.params]
    assert len(events[events.source_id == "benchmark"]) == sum(declared)


def test_drift_primary_only_material(bpanel):
    _, labels, _ = inject(bpanel, "cross_source_drift", seed=3)
    offs = labels.params.map(lambda p: abs(json.loads(p)["offset_bps"]))
    assert (offs[labels.primary] >= 10).all() and (offs[~labels.primary] < 10).all()


def test_prf_and_match():
    lab = pd.DataFrame({"security_id": ["A", "A", "A"], "date": pd.to_datetime(["2024-01-02"] * 3),
                        "source_id": ["x", "y", "z"], "primary": [True, True, False]})
    fl = pd.DataFrame({"security_id": ["A", "A", "B"], "date": pd.to_datetime(["2024-01-02"] * 3),
                       "source_id": ["x", "z", "x"]})
    m = ev.match(fl, lab, with_source=True)
    assert m["precision"] == pytest.approx(2 / 3)  # x and z hit altered rows
    assert m["recall"] == pytest.approx(1 / 2)     # only x of the primary {x, y}
    assert ev.prf(0, 0, 0, 0)["precision"] == 1.0


def test_end_to_end_benchmark_on_synthetic(bpanel):
    params = CheckParams(iforest_n_estimators=50)
    res = {}
    for i, t in enumerate(["spike_x10", "stale_run", "missing_days", "duplicate_rows",
                           "unapplied_split", "cross_source_drift"]):
        f, labels, e = inject(bpanel, t, seed=50 + i)
        res[t] = ev.evaluate_scenario(f, labels, e, params)
    assert res["spike_x10"]["mad"]["recall"] == 1.0
    assert res["stale_run"]["staleness"]["f1"] == 1.0
    assert res["missing_days"]["completeness"]["f1"] == 1.0
    assert res["duplicate_rows"]["schema"]["f1"] == 1.0
    assert res["unapplied_split"]["split"]["recall"] == 1.0
    assert res["cross_source_drift"]["reconciliation"]["recall"] > 0.95
    assert res["spike_x10"]["_blame"]["correct"] >= res["spike_x10"]["_blame"]["n"] * 0.9


def test_split_securities_deterministic():
    secs = [f"S{i}" for i in range(10)]
    a = split_securities(secs, seed=42)
    assert a == split_securities(secs, seed=42)
    assert set(a[0]).isdisjoint(a[1]) and len(a[0]) + len(a[1]) == 10


def test_threshold_sweep(bpanel):
    f, labels, _ = inject(bpanel, "spike_x10", seed=77)
    sc = ev.outlier_scores(subset(f, ["SYN0", "SYN1"]), CheckParams(iforest_n_estimators=30))
    sw = ev.sweep_scores(sc, labels[labels.security_id.isin(["SYN0", "SYN1"])], "mad", [3, 5, 1e6])
    assert sw.set_index("threshold").loc[1e6, "flags"] == 0
    assert sw["recall"].is_monotonic_decreasing


def test_full_benchmark_and_report_on_synthetic(tmp_path, monkeypatch):
    """Whole driver (tune on val, evaluate on test, render report) on synthetic data."""
    from mdq import config
    from mdq.benchmark import report, run
    panel = synthetic.make_panel(n_securities=4, start="2022-01-03", end="2022-12-30", seed=8)
    frames = {"yahoo": panel["a"].assign(source_id="yahoo"),
              "nasdaq": panel["b"].assign(source_id="nasdaq")}
    events = pd.DataFrame(columns=["security_id", "action_type", "ex_date", "ratio", "amount",
                                   "source_id", "details"])
    info = {"included": sorted(panel["a"].security_id.unique()), "excluded": []}
    payload = run.benchmark(frames, events, info, CheckParams(iforest_n_estimators=30))
    assert {"test", "validation", "tuned_params", "sweeps", "blame_test"} <= set(payload)
    t = pd.DataFrame(payload["test"])
    assert set(t.error_type) == set(ERROR_TYPES)
    assert t[(t.detector == "schema") & (t.error_type == "duplicate_rows")].f1.iloc[0] == 1.0
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(report, "FIG_DIR", tmp_path / "figures")
    report.write(payload)
    md = (tmp_path / "BENCHMARK.md").read_text(encoding="utf-8")
    assert "Injected-error benchmark" in md and "fat-finger" in md
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3
