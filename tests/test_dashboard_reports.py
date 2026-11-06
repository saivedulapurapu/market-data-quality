"""Dashboard data functions and report generation on a synthetic DuckDB."""
import json

import numpy as np
import pandas as pd
import pytest

from mdq import config, dashboard_data as dd, reports, store, synthetic


@pytest.fixture(scope="module")
def db():
    con = store.connect(":memory:")
    t = synthetic.populate_store(con)
    yield con, t["summary"], t
    con.close()


def test_process_summary(db):
    con, summary, _ = db
    assert summary["securities"] == 3
    assert summary["issues_by_check"]["recon_close_mismatch"] >= 1
    # the injected open>high, plus the 1.3x bad print which also exceeds its own high
    assert summary["issues_by_check"]["ohlc_inconsistent"] == 2


def test_scores_and_dataset(db):
    con, _, truth = db
    ds = dd.dataset_scores(con)
    assert ds.score.min() < 100 and ds.score.max() == 100
    s1 = dd.security_scores(con, "SYN1").set_index("date")
    assert s1.loc[truth["bad_print"], "score"] < 100
    agg = dd.security_scores(con)
    assert set(agg.security_id) == {"SYN0", "SYN1", "SYN2"}


def test_issue_queries(db):
    con, _, truth = db
    c = dd.issue_counts(con, include_info=True)
    assert {"check_name", "severity", "n"} <= set(c.columns)
    rec = dd.issues(con, check_name="recon_close_mismatch")
    blamed = rec[rec.date == truth["bad_print"]]
    assert blamed.source_id.iloc[0] == "nasdaq"
    fl = dd.flagged_points(con, "SYN0")
    assert truth["ohlc"] in set(fl.date)
    assert fl.close.notna().all()


def test_prices_and_source_diff(db):
    con, _, truth = db
    p = dd.prices(con, "SYN1")
    assert set(p.source_id) == {"yahoo", "nasdaq"}
    d = dd.source_diff(con, "SYN1").set_index("date")
    assert abs(d.loc[truth["bad_print"], "diff_bps"]) > 1000
    assert d.diff_bps.abs().median() < 2


def test_split_verification(db):
    con, _, truth = db
    sv = dd.split_verification(con)
    assert len(sv) == 2  # one split x two sources
    assert (sv.adjusted_move - 1).abs().max() < 0.1
    assert (sv.raw_move - 0.25).abs().max() < 0.05


def test_corporate_actions_filter(db):
    con, _, _ = db
    assert len(dd.corporate_actions(con, action_type="split")) == 1
    assert len(dd.corporate_actions(con, security_id="SYN0")) == 1


def test_benchmark_results_loader(tmp_path):
    assert dd.benchmark_results(tmp_path / "missing.json") is None
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"test": [{"detector": "mad", "error_type": "spike_x10", "f1": 1.0}]}))
    assert dd.benchmark_table(dd.benchmark_results(p)).iloc[0].f1 == 1.0


def test_reports_render(db, tmp_path, monkeypatch):
    con, _, _ = db
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    prof = reports.write_profile(con)
    assert "Rows by source" in prof and "SYN" not in prof.split("## Scope")[0]
    q = reports.write_quality(con)
    assert "Issues by type" in q and "ohlc_inconsistent" in q
    eps = reports.recon_episodes(con)
    assert len(eps) >= 1 and {"start", "end", "days"} <= set(eps.columns)
