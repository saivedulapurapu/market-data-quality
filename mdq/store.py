"""DuckDB store (no ORM) with Parquet export.

The DuckDB file is the working store; every table is also exported to
Parquet (data/processed/parquet/<table>.parquet) as the interchange format.
All inserts go through Pandera validation first (see mdq.schemas).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from mdq import config, schemas
from mdq.issues import concat_issues

DDL = """
CREATE TABLE IF NOT EXISTS securities (
    security_id   VARCHAR PRIMARY KEY,
    ticker        VARCHAR NOT NULL,
    name          VARCHAR NOT NULL,
    exchange      VARCHAR NOT NULL,
    asset_class   VARCHAR NOT NULL,
    yahoo_symbol  VARCHAR NOT NULL,
    nasdaq_symbol VARCHAR NOT NULL,
    currency      VARCHAR
);
CREATE TABLE IF NOT EXISTS sources (
    source_id        VARCHAR PRIMARY KEY,
    name             VARCHAR NOT NULL,
    url              VARCHAR,
    adjustment       VARCHAR,
    provides_events  BOOLEAN,
    notes            VARCHAR
);
CREATE TABLE IF NOT EXISTS prices (
    security_id VARCHAR NOT NULL,
    date        DATE NOT NULL,
    source_id   VARCHAR NOT NULL,
    adjustment  VARCHAR NOT NULL,   -- 'split' = split-adjusted as delivered, 'raw' = as traded (reconstructed)
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    adj_close   DOUBLE,             -- split+dividend adjusted (yahoo only)
    volume      DOUBLE,
    PRIMARY KEY (security_id, date, source_id, adjustment)
);
CREATE TABLE IF NOT EXISTS corporate_actions (
    security_id VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,   -- 'split' | 'dividend'
    ex_date     DATE NOT NULL,
    ratio       DOUBLE,             -- split: new shares per old share
    amount      DOUBLE,             -- dividend per share
    source_id   VARCHAR NOT NULL,
    details     VARCHAR,
    PRIMARY KEY (security_id, action_type, ex_date, source_id)
);
CREATE SEQUENCE IF NOT EXISTS issue_seq;
CREATE TABLE IF NOT EXISTS quality_issues (
    issue_id    BIGINT DEFAULT nextval('issue_seq') PRIMARY KEY,
    security_id VARCHAR,
    date        DATE,
    source_id   VARCHAR,
    check_name  VARCHAR NOT NULL,
    category    VARCHAR NOT NULL,
    severity    VARCHAR NOT NULL,
    value       DOUBLE,
    expected    DOUBLE,
    details     VARCHAR
);
CREATE TABLE IF NOT EXISTS quality_scores (
    security_id VARCHAR NOT NULL,
    date        DATE NOT NULL,
    score       DOUBLE NOT NULL,
    n_issues    BIGINT NOT NULL,
    deduction   DOUBLE NOT NULL,
    PRIMARY KEY (security_id, date)
);
CREATE OR REPLACE VIEW dataset_scores AS
    SELECT date, avg(score) AS score, count(*) AS n_securities,
           sum(CASE WHEN score < 100 THEN 1 ELSE 0 END) AS n_affected
    FROM quality_scores GROUP BY date;
"""

TABLES = ["securities", "sources", "prices", "corporate_actions", "quality_issues", "quality_scores"]

_SCHEMAS = {
    "securities": schemas.security_schema,
    "prices": schemas.price_schema,
    "corporate_actions": schemas.corporate_action_schema,
    "quality_issues": schemas.issue_schema,
    "quality_scores": schemas.score_schema,
}

_COLUMNS = {
    "securities": ["security_id", "ticker", "name", "exchange", "asset_class",
                   "yahoo_symbol", "nasdaq_symbol", "currency"],
    "sources": ["source_id", "name", "url", "adjustment", "provides_events", "notes"],
    "prices": ["security_id", "date", "source_id", "adjustment", "open", "high", "low",
               "close", "adj_close", "volume"],
    "corporate_actions": ["security_id", "action_type", "ex_date", "ratio", "amount",
                          "source_id", "details"],
    "quality_issues": ["security_id", "date", "source_id", "check_name", "category",
                       "severity", "value", "expected", "details"],
    "quality_scores": ["security_id", "date", "score", "n_issues", "deduction"],
}


def connect(path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = str(path) if path is not None else str(config.DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path, read_only=read_only)
    if not read_only:
        con.execute(DDL)
    return con


def reset(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP VIEW IF EXISTS dataset_scores")
    for t in reversed(TABLES):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("DROP SEQUENCE IF EXISTS issue_seq")
    con.execute(DDL)


def insert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame,
           validate: bool = True) -> pd.DataFrame:
    """Validate with Pandera, insert clean rows, return schema issues.

    Schema issues from tables other than quality_issues are written to
    quality_issues as well, so nothing is dropped silently.
    """
    issues = None
    if validate and table in _SCHEMAS:
        df, issues = schemas.validate(df, _SCHEMAS[table])
    cols = _COLUMNS[table]
    df = df[cols]
    if len(df):
        con.register("_incoming", df)
        con.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _incoming")
        con.unregister("_incoming")
    if issues is not None and len(issues) and table != "quality_issues":
        insert(con, "quality_issues", issues, validate=True)
    return issues if issues is not None else concat_issues([])


def read(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    df = con.execute(sql, params or []).df()
    for c in ("date", "ex_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    return df


def export_parquet(con: duckdb.DuckDBPyConnection, out_dir: str | Path | None = None) -> list[Path]:
    out_dir = Path(out_dir) if out_dir else config.DATA_PROCESSED / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in TABLES:
        p = out_dir / f"{t}.parquet"
        con.execute(f"COPY {t} TO '{p.as_posix()}' (FORMAT PARQUET)")
        paths.append(p)
    return paths
