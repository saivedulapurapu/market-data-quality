"""NYSE trading calendar (pandas_market_calendars, works offline)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=32)
def _sessions(start: str, end: str) -> tuple:
    cal = mcal.get_calendar("XNYS")
    days = cal.valid_days(start_date=start, end_date=end)
    return tuple(pd.DatetimeIndex(days).tz_localize(None).normalize())


def sessions(start, end) -> pd.DatetimeIndex:
    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = pd.Timestamp(end).strftime("%Y-%m-%d")
    return pd.DatetimeIndex(_sessions(s, e)).astype("datetime64[ns]")
