"""Read-only query functions used by the Streamlit dashboard and the reports.

Kept free of Streamlit so they can be unit tested against a synthetic DuckDB.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mdq import config, store


def securities(con) -> pd.DataFrame:
    return store.read(con, "SELECT * FROM securities ORDER BY security_id")


def dataset_scores(con) -> pd.DataFrame:
    return store.read(con, "SELECT * FROM dataset_scores ORDER BY date")


def security_scores(con, security_id: str | None = None) -> pd.DataFrame:
    if security_id:
        return store.read(con, "SELECT * FROM quality_scores WHERE security_id = ? ORDER BY date",
                          [security_id])
    return store.read(con, """SELECT security_id, avg(score) AS avg_score, min(score) AS min_score,
                                     sum(CAST(score < 100 AS INT)) AS days_below_100
                              FROM quality_scores GROUP BY 1 ORDER BY avg_score, security_id""")


def issue_counts(con, include_info: bool = False) -> pd.DataFrame:
    where = "" if include_info else "WHERE severity <> 'info'"
    return store.read(con, f"""SELECT category, check_name, severity, count(*) AS n,
                                      count(DISTINCT security_id) AS securities
                               FROM quality_issues {where}
                               GROUP BY ALL ORDER BY n DESC, check_name""")


def issues(con, security_id: str | None = None, check_name: str | None = None,
           include_info: bool = True, limit: int | None = None) -> pd.DataFrame:
    cond, params = [], []
    if security_id:
        cond.append("security_id = ?"); params.append(security_id)
    if check_name:
        cond.append("check_name = ?"); params.append(check_name)
    if not include_info:
        cond.append("severity <> 'info'")
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    lim = f"LIMIT {int(limit)}" if limit else ""
    return store.read(con, f"""SELECT security_id, date, source_id, check_name, category, severity,
                                      value, expected, details
                               FROM quality_issues {where} ORDER BY date, security_id {lim}""", params)


def prices(con, security_id: str, adjustment: str = "split") -> pd.DataFrame:
    return store.read(con, """SELECT * FROM prices WHERE security_id = ? AND adjustment = ?
                              ORDER BY source_id, date""", [security_id, adjustment])


def flagged_points(con, security_id: str) -> pd.DataFrame:
    """Issues joined to the close of the source they were raised on (for chart markers)."""
    return store.read(con, """
        SELECT i.date, coalesce(i.source_id, 'yahoo') AS source_id, i.check_name, i.severity,
               p.close, i.details
        FROM quality_issues i
        LEFT JOIN prices p ON p.security_id = i.security_id AND p.date = i.date
             AND p.source_id = coalesce(i.source_id, 'yahoo') AND p.adjustment = 'split'
        WHERE i.security_id = ? ORDER BY i.date""", [security_id])


def source_diff(con, security_id: str) -> pd.DataFrame:
    return store.read(con, """
        SELECT y.date, y.close AS close_yahoo, n.close AS close_nasdaq,
               (y.close - n.close) / ((y.close + n.close) / 2) * 1e4 AS diff_bps
        FROM prices y JOIN prices n
          ON y.security_id = n.security_id AND y.date = n.date AND y.adjustment = n.adjustment
        WHERE y.source_id = 'yahoo' AND n.source_id = 'nasdaq' AND y.adjustment = 'split'
          AND y.security_id = ?
        ORDER BY y.date""", [security_id])


def corporate_actions(con, security_id: str | None = None, action_type: str | None = None) -> pd.DataFrame:
    cond, params = [], []
    if security_id:
        cond.append("security_id = ?"); params.append(security_id)
    if action_type:
        cond.append("action_type = ?"); params.append(action_type)
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    return store.read(con, f"SELECT * FROM corporate_actions {where} ORDER BY ex_date DESC, security_id",
                      params)


def split_verification(con) -> pd.DataFrame:
    """For every declared split: close-to-close move on the ex-date in the
    as-delivered (split-adjusted) and reconstructed raw series, per source."""
    return store.read(con, """
        WITH s AS (SELECT security_id, ex_date, ratio FROM corporate_actions WHERE action_type = 'split'),
        p AS (SELECT security_id, source_id, adjustment, date, close,
                     lag(close) OVER (PARTITION BY security_id, source_id, adjustment ORDER BY date) AS prev
              FROM prices)
        SELECT s.security_id, s.ex_date, s.ratio, p.source_id,
               max(CASE WHEN p.adjustment = 'split' THEN p.close / p.prev END) AS adjusted_move,
               max(CASE WHEN p.adjustment = 'raw' THEN p.close / p.prev END) AS raw_move
        FROM s JOIN p ON p.security_id = s.security_id AND p.date = s.ex_date
        GROUP BY ALL ORDER BY s.ex_date, s.security_id, p.source_id""")


def benchmark_results(path: Path | None = None) -> dict | None:
    path = Path(path) if path else config.REPORTS_DIR / "benchmark_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_table(results: dict, split: str = "test") -> pd.DataFrame:
    return pd.DataFrame(results[split])
