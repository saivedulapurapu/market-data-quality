# Quality checks

All parameters live in `mdq/config.py` (`CheckParams`). The defaults are the values tuned on
the benchmark validation slice (see `reports/BENCHMARK.md`). Every check emits rows in the
same shape (`security_id, date, source_id, check_name, category, severity, value, expected,
details`).

Checks run on the **split-adjusted** series of each source, because that is what both vendors
deliver. The reconstructed as-traded series (`adjustment='raw'`) is stored for inspection and
for the dashboard.

| # | Check (`check_name`) | Category | Rule | Parameters | Severity | Rationale |
|---|---|---|---|---|---|---|
| 1 | `schema_duplicate_key`, `schema_null_key`, `schema_invalid_value`, `schema_type` | schema | Pandera validation before insert. The row is quarantined. | key = (security, date, source, adjustment) | error (dup), critical (others) | Duplicates or null keys break every downstream join, so they're kept out of the store but logged. |
| 2 | `missing_day` | completeness | A NYSE session (pandas_market_calendars `XNYS`) inside the security's coverage window with no row from that source | window = union of both sources' first..last date | error | A gap silently becomes a multi-day return in any downstream calculation. |
| 3 | `non_trading_day` | completeness | A row dated on a weekend or exchange holiday | — | error | Usually a timezone or date-shift bug in the feed. |
| 4 | `missing_field` | completeness | Any of OHLCV is null | — | error | Partial bars. |
| 5 | `ohlc_inconsistent` | validity | Not (low ≤ open, close ≤ high), with 1e-6 relative tolerance | — | error | Physically impossible bar, which points to a field swap or a bad print. |
| 6 | `non_positive_price` | validity | Any OHLC ≤ 0 | — | critical | Impossible for an equity. Breaks log returns. |
| 7 | `negative_volume` | validity | volume < 0 | — | critical | Impossible. |
| 8 | `zero_volume` | validity | volume = 0 on a session | — | warning | Implausible for this liquid universe. Usually a stale or placeholder bar. |
| 9 | `stale_close` | staleness | A run of ≥ N identical consecutive closes (rounded to 1e-4). The N−1 repeats are flagged. | `stale_min_run` | warning (error if run ≥ 5) | A feed that stopped updating repeats the last value. |
| 10 | `outlier_zscore` | outlier | Spike score with trailing mean/std | `zscore_threshold`, `outlier_window`=63, `outlier_min_periods`=20 | warning (error if > 2× threshold) | Classic baseline. |
| 11 | `outlier_mad` | outlier | Spike score with trailing median / 1.4826·MAD | `mad_threshold` | as above | Robust: one spike doesn't inflate the scale and mask its neighbours. **This is the scored outlier detector.** |
| 12 | `outlier_iforest` | outlier | Isolation Forest per series on [r_in, r_out, log volume ratio, log range ratio]. Score = −score_samples. | `iforest_threshold`, 200 trees, seed 7 | as above | Multivariate and unsupervised, so it can use volume and range too. |
| 13 | `unadjusted_split` | corporate_action | On a declared split ex-date, the return matches −log(ratio) within tol and the new level persists for 5 sessions | `split_log_tolerance`, `split_persist_window`, `split_candidate_ratios` | critical | The vendor knows the split but didn't back-adjust the history. |
| 14 | `suspected_unadjusted_split` | corporate_action | Same jump and persistence pattern for a common ratio (2, 3, 4, 5, 8, 10, 15, 20, 25, 50 or a reverse split) with no declared event | as above | error | Missing corporate-action event, or an unadjusted series. |
| 15 | `dividend_adjustment_mismatch` | corporate_action | Yahoo adjustment factor f = adj_close/close. On a dividend ex-date, f(t−1)/f(t) ≠ 1 − D/close(t−1) beyond tol. | `dividend_factor_tolerance` = 5e-4 | warning | Verifies the vendor's total-return adjustment against the declared dividend. |
| 16 | `unexplained_adjustment` | corporate_action | The factor steps on a day with no dividend and no split | as above | warning | An adjustment applied with no event to explain it. |
| 17 | `recon_close_mismatch` | reconciliation | \|close_yahoo − close_nasdaq\| / mid > tol + half a cent / mid | `recon_tolerance_bps`, `recon_blame_margin` | warning (error if ≥ 100 bps) | Independent vendors should agree. A disagreement means at least one of them is wrong. |
| 18 | `recon_missing_in_source` | reconciliation | The day is present in one source only | — | warning | Coverage disagreement. |

## Spike score (z-score and MAD)

A bad print at t creates a large return into t (r_in) and an opposite large return out of it
(r_out). Both are standardised with location and scale from the trailing window
r(t−W..t−1), so the suspect point never contaminates its own reference:

    score_t = min(|z_in|, |z_out|)  if z_in and z_out have opposite signs, else 0

A genuine one-way move (an earnings gap, or a split) has only one large return, so it scores
near 0. The first and last points are not scored, because one of the two returns is unknown.
The two detectors differ **only** in the estimator, mean/std vs median/MAD, which makes the
benchmark a clean test of robustness.

## Blame attribution (reconciliation)

With two sources there is no majority vote. Each source collects a penalty:

- **z of its distance from its own neighbours' midpoint**, measuring consistency with its own history;
- **+2 if its close lies outside its own [low, high] bar**;
- **+2 if its close lies outside the other source's bar**, measuring consistency with the other source.

The source whose penalty is higher by at least `recon_blame_margin` is blamed. Otherwise the
result is `undetermined`, which is typical for slow drift and for constant adjustment-factor
differences.

## Cross-source corroboration

A `stale_close` or outlier flag that fires on **both** independent sources on the same day,
while reconciliation finds no mismatch, is downgraded to `info` (kept, not deleted). Two
independent vendors printing the same unusual value is evidence of a real market event, such
as a −22% earnings day or three identical closes, not a vendor error. Limitation: if both
vendors take the same upstream error, corroboration will hide it.

## Quality score

    deduction(sec, day) = Σ over distinct checks firing that day of max severity weight
    score(sec, day)     = max(0, 100 − deduction)

| Severity | Weight |
|---|---|
| critical | 60 |
| error | 30 |
| warning | 10 |
| info | 0 |

- Each check counts once per security-day, whatever the number of sources it fired on.
- Only the primary outlier detector (MAD) is scored, so the three outlier detectors don't triple-count one spike.
- **Dataset score per day** = mean over securities (the `dataset_scores` view).
- **Overall dataset score** = mean over all security-days.
