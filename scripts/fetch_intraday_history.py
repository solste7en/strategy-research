"""Build the intraday bar cache from Alpaca or Schwab.

Usage (from repo root):

    # Default: Alpaca SIP 1-min bars, from 2023-01-01 to today (free account)
    python3 scripts/fetch_intraday_history.py

    # Custom start date or symbol list
    python3 scripts/fetch_intraday_history.py --start 2022-01-01
    python3 scripts/fetch_intraday_history.py --symbols TSLA,NVDA

    # Schwab: top up with the last 48 days of fresh 1-min bars
    python3 scripts/fetch_intraday_history.py --source schwab

Output: one parquet file per symbol at strategy/data_cache/{SYMBOL}_1min.parquet.

Credentials:
- Alpaca: set ALPACA_API_KEY and ALPACA_API_SECRET in a .env file at the
  repo root (see .env.example). A free paper-trading account is sufficient.
- Schwab: requires token.json + credentials from the companion schwab_app repo.

Notes:
- Alpaca returns 1-min SIP (consolidated tape) bars for historical data
  (>15 min old). The SDK paginates large date ranges automatically.
- Schwab priceHistory covers ~48 calendar days of 1-min bars. Requests are
  chunked in 45-day windows when the range is larger.
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


def _chunked_ranges(start: date, end: date, chunk: int = _SCHWAB_CHUNK_DAYS):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=chunk - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def _fetch_alpaca(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch 1-min SIP bars from Alpaca. SDK paginates automatically."""
    return AlpacaBarsProvider().get_bars(symbol, start, end)


def _fetch_schwab(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch 1-min bars from Schwab in 45-day chunks."""
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
    return out[~out.index.duplicated(keep="first")]


def main():
    parser = argparse.ArgumentParser(
        description="Build local 1-min bar cache for backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full Alpaca load (default)\n"
            "  python3 scripts/fetch_intraday_history.py\n\n"
            "  # Custom start or symbols\n"
            "  python3 scripts/fetch_intraday_history.py --start 2022-01-01\n"
            "  python3 scripts/fetch_intraday_history.py --symbols TSLA,NVDA\n\n"
            "  # Top up with recent Schwab bars (last 48 days)\n"
            "  python3 scripts/fetch_intraday_history.py --source schwab\n"
        ),
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_UNIVERSE),
        help=f"Comma-separated tickers. Default: {','.join(DEFAULT_UNIVERSE)}",
    )
    parser.add_argument(
        "--source",
        choices=["alpaca", "schwab"],
        default="alpaca",
        help="Data source. 'alpaca' = primary (default). 'schwab' = backup, last ~48 days.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD. Alpaca default: 2023-01-01. Ignored for --source schwab.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Trailing calendar days (schwab only, alternative to --start). Default: 48.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write parquet files into. Default: strategy/data_cache/.",
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
    if args.source == "schwab":
        days = args.days or _SCHWAB_WINDOW_DAYS
        start = end - timedelta(days=days)
    else:  # alpaca
        start = date.fromisoformat(args.start) if args.start else _ALPACA_DEFAULT_START

    log.info(
        "source=%s symbols=%s start=%s end=%s output_dir=%s",
        args.source, symbols, start, end,
        output_dir or "default (strategy/data_cache/)",
    )

    for sym in symbols:
        df = _fetch_alpaca(sym, start, end) if args.source == "alpaca" \
            else _fetch_schwab(sym, start, end)

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
