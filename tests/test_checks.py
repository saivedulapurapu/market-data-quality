"""Each check on hand-built synthetic fixtures with known defects."""
import json

import numpy as np
import pandas as pd
import pytest

from mdq import synthetic
from mdq.checks.completeness import check_completeness
from mdq.checks.validity import check_validity
from mdq.checks.staleness import check_staleness
from mdq.checks.outliers import check_zscore, check_mad, check_iforest, spike_scores
from mdq.checks.corporate_actions import check_splits, check_dividend_adjustment
from mdq.checks.reconciliation import reconcile
from mdq.checks.runner import run_checks


def _dates(issues, name):
    return set(issues.loc[issues.check_name == name, "date"])


# ---------- completeness ----------
def test_clean_series_is_complete(series):
    assert len(check_completeness(series)) == 0


def test_missing_day_detected(series):
    drop = series["date"].iloc[[50, 51, 120]]
    iss = check_completeness(series[~series.date.isin(drop)])
    assert _dates(iss, "missing_day") == set(drop)


def test_missing_day_respects_bounds(series):
    late = series.iloc[10:]
    bounds = {"SYN": (series.date.min(), series.date.max())}
    iss = check_completeness(late, bounds)
    assert _dates(iss, "missing_day") == set(series.date.iloc[:10])


def test_non_trading_day(series):
    extra = series.iloc[[5]].copy()
    extra["date"] = pd.Timestamp("2022-07-04")  # Independence Day
    iss = check_completeness(pd.concat([series, extra]))
    assert _dates(iss, "non_trading_day") == {pd.Timestamp("2022-07-04")}


def test_missing_field(series):
    s = series.copy()
    s.loc[7, "volume"] = np.nan
    s.loc[9, "open"] = np.nan
    iss = check_completeness(s)
    mf = iss[iss.check_name == "missing_field"]
    assert set(mf.date) == {s.date[7], s.date[9]}
    assert json.loads(mf.iloc[0].details)["missing"] in (["volume"], ["open"])


# ---------- validity ----------
def test_clean_series_is_valid(series):
    assert len(check_validity(series)) == 0


def test_validity_violations(series):
    s = series.copy()
    s.loc[3, ["high", "low"]] = s.loc[3, ["low", "high"]].to_numpy()  # swapped
    s.loc[4, "close"] = s.loc[4, "high"] * 1.5
    s.loc[5, "low"] = -1.0
    s.loc[6, "volume"] = -10
    s.loc[8, "volume"] = 0
    iss = check_validity(s)
    assert _dates(iss, "ohlc_inconsistent") == {s.date[3], s.date[4]}
    assert _dates(iss, "non_positive_price") == {s.date[5]}
    assert _dates(iss, "negative_volume") == {s.date[6]}
    assert _dates(iss, "zero_volume") == {s.date[8]}
    row = iss[(iss.check_name == "ohlc_inconsistent") & (iss.date == s.date[4])].iloc[0]
    assert row.expected == pytest.approx(s.loc[4, "high"])


# ---------- staleness ----------
def test_stale_run(series):
    s = series.copy()
    s.loc[31:34, "close"] = s.loc[30, "close"]  # 5 identical closes: 30..34
    s.loc[80, "close"] = s.loc[79, "close"]      # run of 2: below threshold
    iss = check_staleness(s, min_run=3)
    assert _dates(iss, "stale_close") == set(s.date[31:35])
    assert set(iss.severity) == {"error"}  # run length 5 -> error


# ---------- outliers ----------
def test_spike_score_ignores_level_shift(series):
    c = series["close"].to_numpy().copy()
    c[150:] *= 0.5  # one-way jump (like an unadjusted split)
    score, _, _ = spike_scores(c, robust=True)
    # score = the ordinary next-day |z|, far below any threshold, while the
    # raw return z is ~35
    assert score[150] < 4.0


@pytest.mark.parametrize("check", [check_zscore, check_mad])
def test_spike_detected(series, check):
    s = series.copy()
    s.loc[100, "close"] *= 10
    s.loc[180, "close"] /= 100
    iss = check(s, threshold=6.0)
    assert {s.date[100], s.date[180]} <= _dates(iss, iss.check_name.iloc[0])
    assert len(iss) == 2


def test_mad_robust_to_masking(series):
    """A big spike inflates the trailing std; a moderate spike 5 days later is
    masked for the z-score but still found by MAD."""
    s = series.copy()
    s.loc[100, "close"] *= 10
    s.loc[105, "close"] *= 1.2
    z = check_zscore(s, threshold=6.0)
    m = check_mad(s, threshold=6.0)
    assert s.date[105] in set(m.date)
    assert s.date[105] not in set(z.date)


def test_iforest_scores_spike_highest(series):
    s = series.copy()
    s.loc[100, "close"] *= 10
    iss = check_iforest(s, threshold=0.6)
    top = iss.sort_values("value", ascending=False).iloc[0]
    assert top.date == s.date[100]


# ---------- corporate actions ----------
def _split_events(date, ratio, sec="SYN"):
    return pd.DataFrame({"security_id": [sec], "action_type": ["split"], "ex_date": [date],
                         "ratio": [ratio], "amount": [np.nan], "source_id": ["a"], "details": [""]})


def test_adjusted_series_passes_split_check(series):
    ev = _split_events(series.date[120], 4.0)
    assert len(check_splits(series, ev)) == 0


def test_unadjusted_split_with_event(series):
    s = series.copy()
    s.loc[:119, ["open", "high", "low", "close"]] *= 4  # history not adjusted
    iss = check_splits(s, _split_events(s.date[120], 4.0))
    assert list(iss.check_name) == ["unadjusted_split"]
    assert iss.iloc[0].date == s.date[120] and iss.iloc[0].severity == "critical"


def test_undeclared_split(series):
    s = series.copy()
    s.loc[:119, ["open", "high", "low", "close"]] *= 3
    iss = check_splits(s, None)
    assert list(iss.check_name) == ["suspected_unadjusted_split"]
    assert json.loads(iss.iloc[0].details)["nearest_ratio"] == 3.0


def test_reverse_split_detected(series):
    s = series.copy()
    s.loc[:119, ["open", "high", "low", "close"]] /= 10
    iss = check_splits(s, None)
    assert len(iss) == 1 and json.loads(iss.iloc[0].details)["nearest_ratio"] == pytest.approx(0.1)


def test_spike_is_not_a_split(series):
    s = series.copy()
    s.loc[100, "close"] /= 10
    assert len(check_splits(s, None)) == 0


def test_dividend_adjustment(series):
    s = series.copy()
    # build adj_close with a correct dividend at row 60 and a wrong one at row 150
    f = np.ones(len(s))
    d1, d2 = 1.0, 0.5
    f[:60] *= 1 - d1 / s.close[59]
    f[:150] *= 1 - d2 / s.close[149]
    s["adj_close"] = s["close"] * f
    ev = pd.DataFrame({"security_id": "SYN", "action_type": "dividend",
                       "ex_date": [s.date[60], s.date[150]], "ratio": np.nan,
                       "amount": [d1, d2 * 3], "source_id": "a", "details": ""})
    iss = check_dividend_adjustment(s, ev)
    assert _dates(iss, "dividend_adjustment_mismatch") == {s.date[150]}
    # unexplained factor step with no event
    s2 = s.copy()
    s2.loc[:200, "adj_close"] *= 0.99
    iss2 = check_dividend_adjustment(s2, ev.iloc[[0]])
    assert s.date[201] in _dates(iss2, "unexplained_adjustment")


# ---------- reconciliation ----------
def test_reconcile_clean(panel):
    iss, comp = reconcile(panel["a"], panel["b"], tol_bps=10)
    assert len(iss) == 0
    assert comp["diff_bps"].abs().max() < 10 + 1e4 * 0.005 / comp["close_a"].min()


def test_reconcile_blames_spiking_source(panel):
    a, b = panel["a"].copy(), panel["b"].copy()
    i = a.index[(a.security_id == "SYN1")][100]
    b.loc[i, "close"] *= 1.15  # bad print in source b, outside both bars
    iss, _ = reconcile(a, b, tol_bps=10)
    mm = iss[iss.check_name == "recon_close_mismatch"]
    assert len(mm) == 1 and mm.iloc[0].source_id == "b"
    assert json.loads(mm.iloc[0].details)["blame"] == "b"


def test_reconcile_drift_is_undetermined(panel):
    a, b = panel["a"].copy(), panel["b"].copy()
    idx = b.index[b.security_id == "SYN0"][100:130]
    b.loc[idx, "close"] *= np.exp(np.linspace(0.0, 0.006, len(idx)))
    iss, _ = reconcile(a, b, tol_bps=10)
    mm = iss[iss.check_name == "recon_close_mismatch"]
    assert len(mm) > 5
    assert mm["source_id"].isna().mean() > 0.8


def test_reconcile_missing_in_source(panel):
    a, b = panel["a"], panel["b"].drop(panel["b"].index[[10]])
    iss, _ = reconcile(a, b)
    miss = iss[iss.check_name == "recon_missing_in_source"]
    assert len(miss) == 1 and miss.iloc[0].source_id == "b"


# ---------- runner ----------
def test_run_checks_on_clean_panel(panel):
    iss, comp = run_checks(panel, detectors=["schema", "completeness", "validity", "staleness",
                                              "split", "reconciliation"])
    assert len(iss) == 0 and comp is not None


def test_run_checks_catches_duplicates(panel):
    a = pd.concat([panel["a"], panel["a"].iloc[[5]]], ignore_index=True)
    iss, _ = run_checks({"a": a, "b": panel["b"]}, detectors=["schema", "reconciliation"])
    assert list(iss.check_name) == ["schema_duplicate_key"]


def test_last_point_not_scored(series):
    s = series.copy()
    s.loc[s.index[-1], "close"] *= 10
    assert len(check_mad(s, threshold=6.0)) == 0


def test_corroborated_flags_downgraded(panel):
    a, b = panel["a"].copy(), panel["b"].copy()
    ia = a.index[a.security_id == "SYN0"][100]
    ib = b.index[b.security_id == "SYN0"][100]
    a.loc[ia, "close"] *= 1.3   # same genuine-looking move on both sources
    b.loc[ib, "close"] = round(a.loc[ia, "close"], 2)
    ic = b.index[b.security_id == "SYN1"][100]
    b.loc[ic, "close"] *= 1.3   # vendor error on one source only
    iss, _ = run_checks({"a": a, "b": b}, detectors=["mad", "reconciliation"])
    mad = iss[iss.check_name == "outlier_mad"].set_index(["security_id", "source_id"])
    assert set(mad.loc["SYN0", "severity"]) == {"info"}
    assert mad.loc[("SYN1", "b"), "severity"] in ("warning", "error")
    assert '"corroborated": true' in mad.loc[("SYN0", "a"), "details"]
