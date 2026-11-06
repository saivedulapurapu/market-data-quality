"""ingest.build_frames/load from synthetic cached payloads (no network)."""
import json

import pandas as pd
import pytest

from mdq import config, ingest, store
from mdq.sources import nasdaq, yahoo


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(config, "UNIVERSE", [("SYNA", "Synthetic A", "NYSE", "stock")])
    monkeypatch.setattr(config, "START_DATE", "2024-01-01")
    monkeypatch.setattr(config, "END_DATE", "2024-01-31")
    dates = pd.bdate_range("2024-01-02", "2024-01-05")  # 2024-01-02..05 are NYSE sessions
    ts = [int((d + pd.Timedelta(hours=14, minutes=30)).timestamp()) for d in dates]
    # as delivered (split-adjusted); a 2:1 split on the 3rd session
    adj_closes = [5.0, 5.1, 5.1, 5.2]
    ypay = {"chart": {"error": None, "result": [{
        "meta": {}, "timestamp": ts,
        "events": {"splits": {str(ts[2]): {"date": ts[2], "numerator": 2.0, "denominator": 1.0,
                                           "splitRatio": "2:1"}}},
        "indicators": {"quote": [{"open": adj_closes, "high": [c * 1.01 for c in adj_closes],
                                  "low": [c * 0.99 for c in adj_closes], "close": adj_closes,
                                  "volume": [100, 100, 200, 200]}],
                       "adjclose": [{"adjclose": adj_closes}]}}]}}
    npay = {"data": {"tradesTable": {"rows": [
        {"date": d.strftime("%m/%d/%Y"), "close": f"${c:.2f}", "volume": "100", "open": f"{c:.2f}",
         "high": f"{c * 1.01:.2f}", "low": f"{c * 0.99:.2f}"} for d, c in zip(dates, adj_closes)]}}}
    (tmp_path / "yahoo").mkdir()
    (tmp_path / "nasdaq").mkdir()
    yahoo.cache_path("SYNA").write_text(json.dumps(ypay))
    nasdaq.cache_path("SYNA").write_text(json.dumps(npay))
    return tmp_path


def test_build_and_load(raw_dir):
    frames = ingest.build_frames()
    p = frames["prices"]
    assert set(zip(p.source_id, p.adjustment)) == {("yahoo", "split"), ("yahoo", "raw"),
                                                    ("nasdaq", "split"), ("nasdaq", "raw")}
    raw = p[(p.source_id == "yahoo") & (p.adjustment == "raw")].sort_values("date")
    assert raw.close.tolist() == pytest.approx([10.0, 10.2, 5.1, 5.2])  # split undone before ex-date
    assert len(frames["corporate_actions"]) == 1
    con = store.connect(":memory:")
    issues = ingest.load(con, frames)
    assert len(issues) == 0
    assert con.execute("select count(*) from prices").fetchone()[0] == 16
    assert con.execute("select count(*) from securities").fetchone()[0] == 1


def test_fetch_all_uses_cache(raw_dir):
    assert ingest.fetch_all() == {"yahoo": 1, "nasdaq": 1}  # served from cache, no network
