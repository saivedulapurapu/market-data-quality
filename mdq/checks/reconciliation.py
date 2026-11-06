"""Cross-source reconciliation of closes, with blame attribution.

diff_bps = (close_a - close_b) / mid * 1e4. A day is a mismatch when
|diff_bps| > tol_bps + rounding allowance (half a cent relative to price,
because one vendor rounds to cents).

Blame: which source is likely wrong. With only two sources there is no
majority vote, so each source s gets a penalty from evidence that its close
is inconsistent:

  z_s    = e_s / robust_scale(e_s), where e_s = |log c_s,t - (log c_s,t-1 + log c_s,t+1)/2|
           (distance from the midpoint of its own neighbours, in units of that
           series' typical value): consistency with its own history
  +2     if close_s lies outside its own [low, high] bar
  +2     if close_s lies outside the other source's [low, high] bar
           (consistency with the other source)

The source whose penalty exceeds the other's by >= blame_margin is blamed.
Otherwise the result is "undetermined" (typical for slow drift, where both
series are smooth and within each other's ranges).

Persistence rule: if the day's difference equals the previous and next
day's difference (within tolerance), the mismatch is a level or
adjustment-basis difference (e.g. two vendors adjusting history by different
spin-off factors), not a bad print on that day, and blame is "undetermined".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mdq.issues import concat_issues, make_issues

KEY = ["security_id", "date"]


def _self_z(df: pd.DataFrame) -> pd.Series:
    lc = np.log(df["close"].where(df["close"] > 0))
    g = lc.groupby(df["security_id"])
    e = (lc - (g.shift(1) + g.shift(-1)) / 2).abs()
    scale = e.groupby(df["security_id"]).transform(lambda x: 1.4826 * x.median())
    return (e / scale.replace(0, np.nan)).fillna(0.0)


def reconcile(a: pd.DataFrame, b: pd.DataFrame, tol_bps: float = 10.0,
              blame_margin: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (issues, comparison frame with diff_bps for every common day)."""
    src_a = a["source_id"].iloc[0] if len(a) else "a"
    src_b = b["source_id"].iloc[0] if len(b) else "b"
    cols = KEY + ["close", "high", "low", "z"]
    a = a.drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
    b = b.drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
    a["z"], b["z"] = _self_z(a), _self_z(b)
    m = a[cols].merge(b[cols], on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    both_secs = set(a["security_id"]) & set(b["security_id"])
    m = m[m["security_id"].isin(both_secs)]
    out = []
    for side, src, other in (("left_only", src_b, src_a), ("right_only", src_a, src_b)):
        rows = m[m["_merge"] == side]
        if len(rows):
            out.append(make_issues(rows, "recon_missing_in_source", "warning", source_id=src,
                                   details=f"present in {other}, absent in {src}"))
    c = m[m["_merge"] == "both"].copy()
    mid = (c["close_a"] + c["close_b"]) / 2
    c["diff_bps"] = (c["close_a"] - c["close_b"]) / mid * 1e4
    c["allowed_bps"] = tol_bps + 0.005 / mid * 1e4
    # persistent difference: the day's diff matches both neighbouring days' diff
    # (within tolerance) -> a level / adjustment-basis difference, not a bad print
    c = c.sort_values(KEY)
    gd = c.groupby("security_id")["diff_bps"]
    c["persistent"] = (((c["diff_bps"] - gd.shift(1)).abs() <= c["allowed_bps"])
                       & ((c["diff_bps"] - gd.shift(-1)).abs() <= c["allowed_bps"]))
    # a NaN close on one side is a missing_field issue, not a mismatch
    bad = c[c["diff_bps"].abs() > c["allowed_bps"]].copy()
    if len(bad):
        def outside(x, lo, hi):
            return ((x < lo * (1 - 1e-6)) | (x > hi * (1 + 1e-6))).astype(float)
        pen_a = (bad["z_a"] + 2 * outside(bad["close_a"], bad["low_a"], bad["high_a"])
                 + 2 * outside(bad["close_a"], bad["low_b"], bad["high_b"]))
        pen_b = (bad["z_b"] + 2 * outside(bad["close_b"], bad["low_b"], bad["high_b"])
                 + 2 * outside(bad["close_b"], bad["low_a"], bad["high_a"]))
        blame = np.where(pen_a - pen_b >= blame_margin, src_a,
                         np.where(pen_b - pen_a >= blame_margin, src_b, None))
        blame = np.where(bad["persistent"].to_numpy(), None, blame)
        bad["source_id"] = blame
        bad["sev"] = np.where(bad["diff_bps"].abs() >= 100, "error", "warning")
        bad["det"] = [{"close_" + src_a: float(x), "close_" + src_b: float(y),
                       "blame": z or "undetermined", "persistent": bool(ps),
                       "penalty_" + src_a: round(float(pa), 3), "penalty_" + src_b: round(float(pb), 3)}
                      for x, y, z, ps, pa, pb in zip(bad["close_a"], bad["close_b"], blame,
                                                     bad["persistent"], pen_a, pen_b)]
        for s, grp in bad.groupby("sev"):
            out.append(make_issues(grp, "recon_close_mismatch", s, value="diff_bps",
                                   expected="allowed_bps", details=grp["det"].tolist(),
                                   source_id=None))
    comp = c[KEY + ["close_a", "close_b", "diff_bps", "allowed_bps"]].rename(
        columns={"close_a": f"close_{src_a}", "close_b": f"close_{src_b}"})
    return concat_issues(out), comp.reset_index(drop=True)
