"""Headless smoke test of every dashboard page on a synthetic DuckDB file."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from mdq import config, store, synthetic

PAGES = ["Overview", "Issues", "Ticker", "Sources", "Corporate actions", "Benchmark"]


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("app")
    path = d / "demo.duckdb"
    con = store.connect(path)
    synthetic.populate_store(con)
    con.close()
    (d / "benchmark_results.json").write_text(json.dumps({
        "tuned_params": {"mad_threshold": 5.0},
        "test": [{"detector": det, "error_type": et, "precision": 1.0, "recall": 0.5, "f1": 0.67}
                 for det in ["zscore", "mad", "iforest"] for et in ["spike_x10", "small_spike"]]}))
    return d, path


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(demo_db, monkeypatch, page):
    d, path = demo_db
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "REPORTS_DIR", d)
    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert len(at.title) == 1
