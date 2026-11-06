"""End-to-end pipeline: cached raw -> store -> checks -> scores -> Parquet."""
from __future__ import annotations

import json
import logging

from mdq import config, ingest, scoring, store
from mdq.checks.runner import coverage_bounds, run_checks
from mdq.config import DEFAULT_PARAMS

log = logging.getLogger("mdq.pipeline")


def process(con, params=DEFAULT_PARAMS) -> dict:
    """Run all checks on what is in the store, write issues, scores and the
    reconciliation comparison table. Works on any populated store."""
    prices = store.read(con, "SELECT * FROM prices WHERE adjustment = 'split'")
    events = store.read(con, "SELECT * FROM corporate_actions")
    srcs = [s for s in ["yahoo", "nasdaq"] if (prices.source_id == s).any()]
    by_src = {s: prices[prices.source_id == s].reset_index(drop=True) for s in srcs}
    issues, comp = run_checks(by_src, events, params, validate_schema=False)
    store.insert(con, "quality_issues", issues)
    all_issues = store.read(con, "SELECT * FROM quality_issues")
    scores = scoring.score(all_issues, coverage_bounds(by_src))
    store.insert(con, "quality_scores", scores)
    if comp is not None:
        con.register("_comp", comp)
        con.execute("CREATE OR REPLACE TABLE recon_comparison AS SELECT * FROM _comp")
        con.unregister("_comp")
    return {
        "price_rows": int(len(prices)),
        "securities": int(prices.security_id.nunique()),
        "issues": int(len(all_issues)),
        "dataset_score": scoring.dataset_score(scores),
        "issues_by_check": all_issues.groupby("check_name").size().to_dict(),
    }


def run(db_path=None, tickers=None, params=DEFAULT_PARAMS) -> dict:
    con = store.connect(db_path or config.DB_PATH)
    store.reset(con)
    frames = ingest.build_frames(tickers)
    schema_issues = ingest.load(con, frames)
    log.info("loaded; %d schema issues", len(schema_issues))
    summary = process(con, params)
    store.export_parquet(con)
    (config.DATA_PROCESSED / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    con.close()
    return summary
