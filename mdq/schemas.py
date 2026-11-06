"""Pandera schemas for every DataFrame entering the store.

Structural problems (null keys, duplicate keys, invalid categorical values,
uncoercible types) are *quarantined*: the offending rows are removed before
loading and each one is logged as a quality issue (category "schema").
Value-level problems (negative prices, OHLC inconsistencies, ...) are NOT
schema failures; they pass through and are caught by the quality checks,
so that they are scored rather than silently dropped.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from mdq.issues import ISSUE_COLUMNS, SEVERITIES, make_issues, empty_issues

ADJUSTMENTS = ["raw", "split"]
PRICE_KEY = ["security_id", "date", "source_id", "adjustment"]

_nullable_float = dict(dtype="float64", nullable=True, coerce=True)

price_schema = pa.DataFrameSchema(
    {
        "security_id": pa.Column(str, nullable=False),
        "date": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "source_id": pa.Column(str, nullable=False),
        "adjustment": pa.Column(str, pa.Check.isin(ADJUSTMENTS), nullable=False),
        "open": pa.Column(**_nullable_float),
        "high": pa.Column(**_nullable_float),
        "low": pa.Column(**_nullable_float),
        "close": pa.Column(**_nullable_float),
        "adj_close": pa.Column(**_nullable_float),
        "volume": pa.Column(**_nullable_float),
    },
    unique=PRICE_KEY,
    report_duplicates="exclude_first",
    strict="filter",
    name="prices",
)

security_schema = pa.DataFrameSchema(
    {
        "security_id": pa.Column(str, nullable=False, unique=True),
        "ticker": pa.Column(str, nullable=False),
        "name": pa.Column(str, nullable=False),
        "exchange": pa.Column(str, pa.Check.isin(["NASDAQ", "NYSE", "NYSEARCA"])),
        "asset_class": pa.Column(str, pa.Check.isin(["stock", "etf"])),
        "yahoo_symbol": pa.Column(str),
        "nasdaq_symbol": pa.Column(str),
        "currency": pa.Column(str, nullable=True),
    },
    strict="filter",
    name="securities",
)

corporate_action_schema = pa.DataFrameSchema(
    {
        "security_id": pa.Column(str, nullable=False),
        "action_type": pa.Column(str, pa.Check.isin(["split", "dividend"])),
        "ex_date": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "ratio": pa.Column("float64", pa.Check.gt(0), nullable=True, coerce=True),
        "amount": pa.Column("float64", pa.Check.ge(0), nullable=True, coerce=True),
        "source_id": pa.Column(str, nullable=False),
        "details": pa.Column(str, nullable=True),
    },
    unique=["security_id", "action_type", "ex_date", "source_id"],
    report_duplicates="exclude_first",
    strict="filter",
    name="corporate_actions",
)

issue_schema = pa.DataFrameSchema(
    {
        "security_id": pa.Column(str, nullable=True),
        "date": pa.Column("datetime64[ns]", nullable=True, coerce=True),
        "source_id": pa.Column(str, nullable=True),
        "check_name": pa.Column(str, nullable=False),
        "category": pa.Column(str, nullable=False),
        "severity": pa.Column(str, pa.Check.isin(SEVERITIES)),
        "value": pa.Column("float64", nullable=True, coerce=True),
        "expected": pa.Column("float64", nullable=True, coerce=True),
        "details": pa.Column(str, nullable=True),
    },
    strict="filter",
    name="quality_issues",
)

score_schema = pa.DataFrameSchema(
    {
        "security_id": pa.Column(str, nullable=False),
        "date": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "score": pa.Column("float64", pa.Check.in_range(0, 100), coerce=True),
        "n_issues": pa.Column("int64", pa.Check.ge(0), coerce=True),
        "deduction": pa.Column("float64", pa.Check.ge(0), coerce=True),
    },
    unique=["security_id", "date"],
    strict="filter",
    name="quality_scores",
)


def _classify(check: str) -> str:
    c = str(check)
    if "uniqueness" in c or "duplicate" in c:
        return "schema_duplicate_key"
    if "not_nullable" in c or "null" in c:
        return "schema_null_key"
    if "coerce" in c or "dtype" in c:
        return "schema_type"
    return "schema_invalid_value"


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema,
             source_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate lazily. Returns (clean_rows, schema_issues).

    Rows referenced by any failure case are dropped from clean_rows and
    reported once per (row, failure type).
    """
    try:
        return schema.validate(df, lazy=True), empty_issues()
    except SchemaErrors as err:
        fc = err.failure_cases
        bad_idx = fc["index"].dropna()
        if len(fc) and len(bad_idx) == 0:
            # whole-frame failure (missing column etc.) cannot be quarantined
            raise
        records = []
        for (idx, check), grp in fc.dropna(subset=["index"]).groupby(["index", "check"], sort=False):
            if idx not in df.index:
                continue
            row = df.loc[idx]
            if isinstance(row, pd.DataFrame):  # duplicated index labels
                row = row.iloc[0]
            records.append({
                "security_id": None if pd.isna(row.get("security_id")) else str(row.get("security_id")),
                "date": row.get("date", row.get("ex_date")),
                "source_id": source_id or (None if pd.isna(row.get("source_id", None))
                                           else row.get("source_id")),
                "check_name": _classify(check),
                "detail": {"table": schema.name, "pandera_check": str(check),
                           "columns": sorted(set(map(str, grp["column"].dropna())))},
            })
        drop = set(bad_idx.tolist())
        clean = df.loc[[i not in drop for i in df.index]]
        clean = schema.validate(clean, lazy=True)
        if not records:
            return clean, empty_issues()
        rec = pd.DataFrame(records)
        rec["date"] = pd.to_datetime(rec["date"], errors="coerce")
        issues = []
        for name, g in rec.groupby("check_name"):
            sev = "error" if name == "schema_duplicate_key" else "critical"
            issues.append(make_issues(g, name, sev, details=g["detail"].tolist(),
                                      source_id=None))
        out = pd.concat(issues, ignore_index=True)[ISSUE_COLUMNS]
        return clean, out
