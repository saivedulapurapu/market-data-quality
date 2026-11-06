"""Command line: python -m mdq {fetch,pipeline,benchmark,reports,all}."""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mdq")
    ap.add_argument("command", choices=["fetch", "pipeline", "benchmark", "reports", "all"])
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.command in ("fetch", "all"):
        from mdq import ingest
        print(ingest.fetch_all(refresh=args.refresh, tickers=args.tickers))
    if args.command in ("pipeline", "all"):
        from mdq import pipeline
        print(pipeline.run(tickers=args.tickers))
    if args.command in ("benchmark", "all"):
        from mdq.benchmark import run as bench
        bench.main()
    if args.command in ("reports", "all"):
        from mdq import reports
        reports.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
