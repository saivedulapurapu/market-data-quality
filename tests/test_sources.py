"""Parsers and HTTP layer, exercised with synthetic payloads (no network)."""
import numpy as np
import pandas as pd
import pytest

from mdq import config
from mdq.sources import http, nasdaq, yahoo


def _yahoo_payload():
    # 3 sessions at 14:30 UTC (09:30 New York), one split, one dividend
    ts = [1704292200, 1704378600, 1704465000]  # 2024-01-03..05
    return {"chart": {"error": None, "result": [{
        "meta": {"currency": "USD", "exchangeName": "NMS", "instrumentType": "EQUITY",
                 "longName": "Synthetic Co"},
        "timestamp": ts,
        "events": {"splits": {"1704378600": {"date": 1704378600, "numerator": 4.0,
                                              "denominator": 1.0, "splitRatio": "4:1"}},
                   "dividends": {"1704465000": {"amount": 0.25, "date": 1704465000}}},
        "indicators": {"quote": [{"open": [10.0, 10.5, None], "high": [11.0, 11.0, 10.9],
                                  "low": [9.5, 10.1, 10.0], "close": [10.8, 10.6, 10.2],
                                  "volume": [1000, 2000, 1500]}],
                       "adjclose": [{"adjclose": [10.6, 10.4, 10.2]}]}}]}}


def test_yahoo_parse_prices():
    df = yahoo.parse_prices(_yahoo_payload(), "SYN")
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04", "2024-01-05"]
    assert np.isnan(df.loc[2, "open"])
    assert df.loc[0, "adj_close"] == 10.6
    assert df["date"].dtype == "datetime64[ns]"


def test_yahoo_parse_events():
    ev = yahoo.parse_events(_yahoo_payload(), "SYN")
    split = ev[ev.action_type == "split"].iloc[0]
    assert split.ratio == 4.0 and split.ex_date == pd.Timestamp("2024-01-04")
    div = ev[ev.action_type == "dividend"].iloc[0]
    assert div.amount == 0.25


def test_yahoo_meta():
    assert yahoo.parse_meta(_yahoo_payload())["currency"] == "USD"


def test_yahoo_error_raises():
    with pytest.raises(http.SourceRejected):
        yahoo.parse_prices({"chart": {"error": {"code": "Not Found"}, "result": None}}, "X")


def test_nasdaq_parse_prices():
    payload = {"data": {"tradesTable": {"rows": [
        {"date": "01/04/2024", "close": "$1,010.50", "volume": "1,234", "open": "$1,000.00",
         "high": "$1,020.00", "low": "$990.00"},
        {"date": "01/03/2024", "close": "N/A", "volume": "N/A", "open": "10", "high": "11",
         "low": "9"}]}}}
    df = nasdaq.parse_prices(payload, "SYN")
    assert df.loc[0, "date"] == pd.Timestamp("2024-01-03")  # sorted ascending
    assert np.isnan(df.loc[0, "close"])
    assert df.loc[1, "close"] == 1010.5 and df.loc[1, "volume"] == 1234


def test_nasdaq_null_rows_rejected():
    with pytest.raises(http.SourceRejected):
        nasdaq.parse_prices({"data": {"tradesTable": {"rows": None}}, "status": {}}, "SYN")


def test_urls():
    assert "assetclass=etf" in nasdaq.url_for("SPY", "etf")
    assert "events=div%2Csplit" in yahoo.url_for("AAPL")


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class _Session:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, headers, timeout):
        self.calls.append(headers)
        return self.resp


def test_http_cache_and_honest_user_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REQUEST_PAUSE_SECONDS", 0)
    sess = _Session(_Resp(200, {"ok": 1}))
    p = tmp_path / "x.json"
    assert http.fetch_json("http://example.invalid", p, session=sess) == {"ok": 1}
    assert sess.calls[0]["User-Agent"] == "market-data-quality/1.0 (personal research)"
    # second call is served from cache: no new request
    assert http.fetch_json("http://example.invalid", p, session=sess) == {"ok": 1}
    assert len(sess.calls) == 1


@pytest.mark.parametrize("resp", [_Resp(403, {}), _Resp(200, "<html>challenge</html>")])
def test_http_rejections(tmp_path, monkeypatch, resp):
    monkeypatch.setattr(config, "REQUEST_PAUSE_SECONDS", 0)
    with pytest.raises(http.SourceRejected):
        http.fetch_json("http://example.invalid", tmp_path / "y.json", session=_Session(resp))
    assert not (tmp_path / "y.json").exists()
