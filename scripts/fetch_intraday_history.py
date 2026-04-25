"""Build the intraday bar cache from Alpaca, Schwab, yfinance, or combinations.

Usage (from repo root):

    # Recommended: Alpaca SIP 1-min bars, from 2023-01-01 to today (free account)
    python3 scripts/fetch_intraday_history.py --source alpaca

    # Custom start date or symbol list
    python3 scripts/fetch_intraday_history.py --source alpaca --start 2022-01-01
    python3 scripts/fetch_intraday_history.py --source alpaca --symbols SPYM,NVDA

    # Legacy sources (Schwab ~48d, yfinance ~60d, merged = both blended)
    python3 scripts/fetch_intraday_history.py --source schwab
    python3 scripts/fetch_intraday_history.py --source yfinance
    python3 scripts/fetch_intraday_history.py --source merged

Output: one parquet file per symbol at strategy/data_cache/{SYMBOL}_1min.parquet.

Credentials:
- Alpaca: set ALPACA_API_KEY and ALPACA_API_SECRET in a .env file at the
  repo root (see .env.example). A free paper-trading account is sufficient.
- Schwab: requires token.json + .env from the companion schwab_app repo.

Notes:
- Alpaca returns 1-min SIP (consolidated tape) bars for historical data
  (>15 min old) on the free plan. The SDK paginates large ranges automatically.
- Schwab priceHistory returns ~48 calendar days of 1-min bars per request.
  We chunk the requested range in 45-day windows.
- yfinance caps sub-hourly history at ~60 calendar days (Yahoo-side limit).
- Merge mode (schwab + yfinance) is additive: Schwab wins on overlapping days.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Make repo root importable whether invoked as `python scripts/...` or `python -m`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from strategy.data import (  # noqa: E402
    AlpacaBarsProvider,
    SchwabBarsProvider,
    YFinanceBarsProvider,
    merge_bars,
    write_cache,
)
from strategy.risk import DEFAULT_UNIVERSE  # noqa: E402

log = logging.getLogger(__name__)

# Default start date for Alpaca — covers 2+ full years of training data.
_ALPACA_DEFAULT_START = date(2023, 1, 1)

# Schwab priceHistory minute window (conservative; broker allows ~48 days).
_SCHWAB_CHUNK_DAYS = 45
_SCHWAB_WINDOW_DAYS = 48

# Yahoo returns ~60 trading SESSIONS of 5-min bars, spanning ~85 calendar days.
_YFINANCE_WINDOW_DAYS = 100


def _chunked_ranges(start: date, end: date, chunk: int = _SCHWAB_CHUNK_DAYS):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=chunk - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def _fetch_alpaca(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch 1-min SIP bars from Alpaca. SDK paginates automatically."""
    provider = AlpacaBarsProvider()
    return provider.get_bars(symbol, start, end)


def _fetch_schwab(symbol: str, start: date, end: date) -> pd.DataFrame:
    provider = SchwabBarsProvider()
    frames: list[pd.DataFrame] = []
    for a, b in _chunked_ranges(start, end):
        log.info("schwab: fetching %s %s..%s", symbol, a, b)
        df = provider.get_bars(symbol, a, b)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _fetch_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
    provider = YFinanceBarsProvider(interval="5m")
    log.info("yfinance: fetching %s %s..%s (5m)", symbol, start, end)
    return provider.get_bars(symbol, start, end)


def _fetch_merged(symbol: str, end: date) -> pd.DataFrame:
    """Legacy: yfinance (~100d, 5-min) blended with Schwab (~48d, 1-min).

    Both calls use their respective max windows; overlap resolved by
    merge_bars keeping Schwab (1-min) over yfinance (5-min) on shared days.
    """
    yf_start = end - timedelta(days=_YFINANCE_WINDOW_DAYS)
    yf_df = _fetch_yfinance(symbol, yf_start, end)

    sch_start = end - timedelta(days=_SCHWAB_WINDOW_DAYS)
    sch_df = _fetch_schwab(symbol, sch_start, end)

    return merge_bars(older=yf_df, newer=sch_df)


def main():
    parser = argparse.ArgumentParser(
        description="Build local 1-min bar cache for backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/fetch_intraday_history.py --source alpaca\n"
            "  python3 scripts/fetch_intraday_history.py --source alpaca --start 2022-01-01\n"
            "  python3 scripts/fetch_intraday_history.py --source alpaca --symbols TSLA,NVDA\n"
            "  python3 scripts/fetch_intraday_history.py --source merged   # legacy\n"
        ),
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_UNIVERSE),
        help=f"Comma-separated tickers. Default: {','.join(DEFAULT_UNIVERSE)}",
    )
    parser.add_argument(
        "--source",
        choices=["alpaca", "merged", "schwab", "yfinance"],
        default="alpaca",
        help=(
            "Data source. "
            "'alpaca' = 1-min SIP bars from 2023-01-01 (recommended, free). "
            "'merged' = legacy Schwab 1-min + yfinance 5-min blend (~60d). "
            "Default: alpaca"
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Start date YYYY-MM-DD. "
            "Alpaca default: 2023-01-01. "
            "Schwab/yfinance/merged: ignored (window-based, use --days instead)."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Trailing calendar days (alternative to --start for Schwab/yfinance). "
            "Ignored when --start is set. "
            "Defaults: schwab=48, yfinance=100, merged=100."
        ),
    )
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write parquet files into. "
            "Defaults to strategy/data_cache/. "
            "Set to strategy/schwab_cache/ to keep Schwab bars in a separate cache."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    end = date.fromisoformat(args.end) if args.end else date.today()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and not output_dir.is_absolute():
        output_dir = _ROOT / output_dir

    # Resolve start date.
    if args.start:
        start = date.fromisoformat(args.start)
    elif args.source == "alpaca":
        start = _ALPACA_DEFAULT_START
    else:
        days = args.days or {"schwab": _SCHWAB_WINDOW_DAYS, "yfinance": _YFINANCE_WINDOW_DAYS,
                             "merged": _YFINANCE_WINDOW_DAYS}[args.source]
        start = end - timedelta(days=days)

    log.info(
        "source=%s symbols=%s start=%s end=%s output_dir=%s",
        args.source, symbols, start, end, output_dir or "default (strategy/data_cache/)",
    )

    for sym in symbols:
        if args.source == "alpaca":
            df = _fetch_alpaca(sym, start, end)
        elif args.source == "schwab":
            df = _fetch_schwab(sym, start, end)
        elif args.source == "yfinance":
            df = _fetch_yfinance(sym, start, end)
        else:  # merged (legacy)
            df = _fetch_merged(sym, end)

        if df.empty:
            log.warning("no bars for %s, skipping cache write", sym)
            continue

        path = write_cache(sym, df, cache_dir=output_dir)
        n_days = len({dt.date() for dt in df.index.tz_convert("America/New_York")})
        log.info(
            "wrote %s: %d bars across %d sessions (%s..%s) → %s",
            sym, len(df), n_days,
            df.index.min().date(), df.index.max().date(), path,
        )


if __name__ == "__main__":
    main()
