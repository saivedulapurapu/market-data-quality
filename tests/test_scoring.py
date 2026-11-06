import pandas as pd
import pytest

from mdq import scoring
from mdq.issues import make_issues


def _iss(rows, name, sev):
    return make_issues(pd.DataFrame(rows), name, sev)


def test_score_weights_and_dedup():
    d1, d2 = pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")
    issues = pd.concat([
        _iss([{"security_id": "A", "date": d1, "source_id": "x"},
              {"security_id": "A", "date": d1, "source_id": "y"}], "missing_day", "error"),  # same check, 2 sources -> once
        _iss([{"security_id": "A", "date": d1, "source_id": "x"}], "stale_close", "warning"),
        _iss([{"security_id": "A", "date": d2, "source_id": "x"}], "non_positive_price", "critical"),
        _iss([{"security_id": "A", "date": d2, "source_id": "x"}], "ohlc_inconsistent", "error"),
        _iss([{"security_id": "A", "date": d2, "source_id": "x"}], "outlier_zscore", "error"),  # not scored
    ])
    s = scoring.score(issues, {"A": (d1, pd.Timestamp("2024-01-05"))}).set_index("date")
    assert s.loc[d1, "score"] == 100 - 30 - 10
    assert s.loc[d2, "score"] == 100 - 60 - 30
    assert s.loc[pd.Timestamp("2024-01-05"), "score"] == 100
    assert s.loc[d1, "n_issues"] == 2


def test_score_floor_and_dataset():
    d = pd.Timestamp("2024-01-03")
    issues = pd.concat([_iss([{"security_id": "A", "date": d, "source_id": "x"}], n, "critical")
                        for n in ["non_positive_price", "negative_volume"]])
    s = scoring.score(issues, {"A": (d, d), "B": (d, d)})
    assert s.set_index("security_id").loc["A", "score"] == 0
    ds = scoring.dataset_score(s)
    assert ds["overall"] == pytest.approx(50.0)
    assert ds["pct_security_days_perfect"] == pytest.approx(50.0)
