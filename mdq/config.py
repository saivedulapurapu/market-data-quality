"""Central configuration: paths, universe, period and check parameters.

Every tunable number used by the checks lives here so that docs, reports and
the benchmark all read the same values.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = DATA_PROCESSED / "mdq.duckdb"
REPORTS_DIR = ROOT / "reports"

START_DATE = "2021-11-08"
END_DATE = "2026-11-06"

# Honest, identifying User-Agent. No browser impersonation.
USER_AGENT = "market-data-quality/1.0 (personal research)"
REQUEST_PAUSE_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30

# (ticker, name, primary exchange, asset class)
UNIVERSE: list[tuple[str, str, str, str]] = [
    # Nasdaq-listed
    ("AAPL", "Apple Inc.", "NASDAQ", "stock"),
    ("MSFT", "Microsoft Corporation", "NASDAQ", "stock"),
    ("NVDA", "NVIDIA Corporation", "NASDAQ", "stock"),
    ("AMZN", "Amazon.com, Inc.", "NASDAQ", "stock"),
    ("GOOGL", "Alphabet Inc. Class A", "NASDAQ", "stock"),
    ("META", "Meta Platforms, Inc.", "NASDAQ", "stock"),
    ("TSLA", "Tesla, Inc.", "NASDAQ", "stock"),
    ("AVGO", "Broadcom Inc.", "NASDAQ", "stock"),
    ("COST", "Costco Wholesale Corporation", "NASDAQ", "stock"),
    ("PEP", "PepsiCo, Inc.", "NASDAQ", "stock"),
    ("ADBE", "Adobe Inc.", "NASDAQ", "stock"),
    ("CSCO", "Cisco Systems, Inc.", "NASDAQ", "stock"),
    ("INTC", "Intel Corporation", "NASDAQ", "stock"),
    ("AMD", "Advanced Micro Devices, Inc.", "NASDAQ", "stock"),
    ("NFLX", "Netflix, Inc.", "NASDAQ", "stock"),
    ("QCOM", "QUALCOMM Incorporated", "NASDAQ", "stock"),
    ("TXN", "Texas Instruments Incorporated", "NASDAQ", "stock"),
    ("AMGN", "Amgen Inc.", "NASDAQ", "stock"),
    ("SBUX", "Starbucks Corporation", "NASDAQ", "stock"),
    ("INTU", "Intuit Inc.", "NASDAQ", "stock"),
    ("CMCSA", "Comcast Corporation", "NASDAQ", "stock"),
    ("GILD", "Gilead Sciences, Inc.", "NASDAQ", "stock"),
    ("MDLZ", "Mondelez International, Inc.", "NASDAQ", "stock"),
    ("ISRG", "Intuitive Surgical, Inc.", "NASDAQ", "stock"),
    ("BKNG", "Booking Holdings Inc.", "NASDAQ", "stock"),
    # NYSE-listed
    ("JPM", "JPMorgan Chase & Co.", "NYSE", "stock"),
    ("V", "Visa Inc.", "NYSE", "stock"),
    ("MA", "Mastercard Incorporated", "NYSE", "stock"),
    ("JNJ", "Johnson & Johnson", "NYSE", "stock"),
    ("PG", "The Procter & Gamble Company", "NYSE", "stock"),
    ("XOM", "Exxon Mobil Corporation", "NYSE", "stock"),
    ("CVX", "Chevron Corporation", "NYSE", "stock"),
    ("UNH", "UnitedHealth Group Incorporated", "NYSE", "stock"),
    ("HD", "The Home Depot, Inc.", "NYSE", "stock"),
    ("KO", "The Coca-Cola Company", "NYSE", "stock"),
    ("MRK", "Merck & Co., Inc.", "NYSE", "stock"),
    ("PFE", "Pfizer Inc.", "NYSE", "stock"),
    ("ABBV", "AbbVie Inc.", "NYSE", "stock"),
    ("WMT", "Walmart Inc.", "NYSE", "stock"),
    ("DIS", "The Walt Disney Company", "NYSE", "stock"),
    ("BAC", "Bank of America Corporation", "NYSE", "stock"),
    ("WFC", "Wells Fargo & Company", "NYSE", "stock"),
    ("CRM", "Salesforce, Inc.", "NYSE", "stock"),
    ("ORCL", "Oracle Corporation", "NYSE", "stock"),
    ("MCD", "McDonald's Corporation", "NYSE", "stock"),
    ("NKE", "NIKE, Inc.", "NYSE", "stock"),
    ("CAT", "Caterpillar Inc.", "NYSE", "stock"),
    ("LLY", "Eli Lilly and Company", "NYSE", "stock"),
    ("CMG", "Chipotle Mexican Grill, Inc.", "NYSE", "stock"),
    ("BA", "The Boeing Company", "NYSE", "stock"),
    # ETFs
    ("SPY", "SPDR S&P 500 ETF Trust", "NYSEARCA", "etf"),
    ("QQQ", "Invesco QQQ Trust", "NASDAQ", "etf"),
]

SOURCES: list[dict] = [
    {
        "source_id": "yahoo",
        "name": "Yahoo Finance chart API (unofficial)",
        "url": "https://query1.finance.yahoo.com/v8/finance/chart/",
        "adjustment": "split",
        "provides_events": True,
        "notes": "OHLC split-adjusted; adjclose split+dividend adjusted; split/dividend events.",
    },
    {
        "source_id": "nasdaq",
        "name": "Nasdaq.com historical quotes API (unofficial)",
        "url": "https://api.nasdaq.com/api/quote/",
        "adjustment": "split",
        "provides_events": False,
        "notes": "OHLCV split-adjusted, not dividend-adjusted; prices rounded to cents.",
    },
]

SEVERITY_WEIGHTS = {"critical": 60, "error": 30, "warning": 10, "info": 0}


@dataclass
class CheckParams:
    """Parameters for all checks. Defaults are the values tuned on the
    benchmark validation slice (see reports/BENCHMARK.md)."""

    # staleness: flag runs of >= stale_min_run identical consecutive closes
    stale_min_run: int = 3
    # outliers
    outlier_window: int = 63
    outlier_min_periods: int = 20
    zscore_threshold: float = 6.0
    mad_threshold: float = 6.0
    iforest_threshold: float = 0.70  # anomaly score in [0, 1]
    iforest_n_estimators: int = 200
    iforest_seed: int = 7
    # corporate actions
    split_log_tolerance: float = 0.10
    split_persist_window: int = 5
    split_candidate_ratios: tuple = (2, 3, 4, 5, 8, 10, 15, 20, 25, 50)
    dividend_factor_tolerance: float = 0.0005
    # reconciliation
    recon_tolerance_bps: float = 10.0
    recon_blame_ratio: float = 3.0
    # zero volume is suspicious for liquid names
    zero_volume_severity: str = "warning"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = CheckParams()
