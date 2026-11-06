import numpy as np
import pandas as pd

from mdq import adjust, ingest, schemas, store


def _prices(series):
    return series[list(schemas.price_schema.columns.keys())].copy()


def test_clean_prices_pass(series):
    clean, issues = schemas.validate(_prices(series), schemas.price_schema)
    assert len(clean) == len(series) and len(issues) == 0


def test_duplicate_and_null_key_quarantined(series):
    df = _prices(series).head(10)
    dup = df.iloc[[3]]
    nullkey = df.iloc[[5]].copy()
    nullkey["date"] = pd.NaT
    bad = pd.concat([df, dup, nullkey], ignore_index=True)
    clean, issues = schemas.validate(bad, schemas.price_schema)
    assert len(clean) == 10
    assert set(issues["check_name"]) == {"schema_duplicate_key", "schema_null_key"}
    dup_issue = issues[issues.check_name == "schema_duplicate_key"].iloc[0]
    assert dup_issue["date"] == df.iloc[3]["date"]
    assert dup_issue["severity"] == "error"


def test_invalid_adjustment_value(series):
    df = _prices(series).head(5)
    df.loc[2, "adjustment"] = "dividend"
    clean, issues = schemas.validate(df, schemas.price_schema)
    assert len(clean) == 4
    assert issues.iloc[0]["check_name"] == "schema_invalid_value"


def test_value_problems_are_not_schema_failures(series):
    df = _prices(series).head(5)
    df.loc[1, "close"] = -5.0
    df.loc[2, "volume"] = -1
    clean, issues = schemas.validate(df, schemas.price_schema)
    assert len(clean) == 5 and len(issues) == 0


def test_score_schema_range():
    df = pd.DataFrame({"security_id": ["A", "B"], "date": [pd.Timestamp("2024-01-02")] * 2,
                       "score": [120.0, 50.0], "n_issues": [0, 1], "deduction": [0.0, 50.0]})
    clean, issues = schemas.validate(df, schemas.score_schema)
    assert len(issues) == 1 and len(clean) == 1


def test_store_insert_logs_schema_issues(con, series):
    df = _prices(series).head(10)
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    store.insert(con, "prices", bad)
    assert con.execute("select count(*) from prices").fetchone()[0] == 10
    qi = store.read(con, "select * from quality_issues")
    assert list(qi["check_name"]) == ["schema_duplicate_key"]


def test_reference_tables_load(con):
    store.insert(con, "securities", ingest.securities_frame())
    store.insert(con, "sources", ingest.sources_frame())
    n = con.execute("select count(*), sum(CAST(asset_class='etf' AS INT)) from securities").fetchone()
    assert n == (52, 2)


def test_export_parquet(con, series, tmp_path):
    store.insert(con, "prices", _prices(series))
    paths = store.export_parquet(con, tmp_path)
    back = pd.read_parquet(tmp_path / "prices.parquet")
    assert len(back) == len(series) and len(paths) == len(store.TABLES)


def test_reset_clears(con, series):
    store.insert(con, "prices", _prices(series))
    store.reset(con)
    assert con.execute("select count(*) from prices").fetchone()[0] == 0


def test_raw_reconstruction_roundtrip(series):
    splits = pd.DataFrame({"ex_date": [series["date"].iloc[100]], "ratio": [4.0]})
    raw = adjust.to_raw(series, splits)
    # before the ex-date raw = 4x adjusted, from the ex-date on unchanged
    assert np.allclose(raw["close"].iloc[:100], series["close"].iloc[:100] * 4)
    assert np.allclose(raw["close"].iloc[100:], series["close"].iloc[100:])
    back = adjust.apply_splits(raw, splits)
    assert np.allclose(back["close"], series["close"])
    assert np.allclose(back["volume"], series["volume"])
