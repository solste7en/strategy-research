"""Build the intraday bar cache from Schwab, yfinance, or both (merged).

Usage (from repo root):

    # Default: merge — yfinance 5-min (older ~60 days) + Schwab 1-min (recent ~48d)
    python3 scripts/fetch_intraday_history.py
    python3 scripts/fetch_intraday_history.py --source merged
    python3 scripts/fetch_intraday_history.py --source schwab
    python3 scripts/fetch_intraday_history.py --source yfinance

    python3 scripts/fetch_intraday_history.py --symbols SPYM,NVDA

Output: one parquet file per symbol at strategy/data_cache/{SYMBOL}_1min.parquet.
(The filename suffix stays `_1min` for historical compatibility even when the
bars are 5-min yfinance data — the strategy is granularity-agnostic.)

Notes:
- Schwab priceHistory returns ~48 calendar days of 1-min bars per request.
  We chunk the requested range when needed.
- yfinance caps sub-hourly history at ~60 calendar days (Yahoo-side limit).
- Merge mode is strictly additive: Schwab wins for overlapping days
  (1-min is finer than 5-min).
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
    SchwabBarsProvider,
    YFinanceBarsProvider,
    merge_bars,
    write_cache,
)
from strategy.risk import DEFAULT_UNIVERSE  # noqa: E402

log = logging.getLogger(__name__)

# Schwab priceHistory minute window (conservative; broker allows ~48 days).
_SCHWAB_CHUNK_DAYS = 45
_SCHWAB_WINDOW_DAYS = 48
# Yahoo returns ~60 trading SESSIONS of 5-min bars, which span ~85 calendar days.
# We pass a 100-day lookback so the provider's slice keeps every session it can.
_YFINANCE_WINDOW_DAYS = 100


def _chunked_ranges(start: date, end: date, chunk: int = _SCHWAB_CHUNK_DAYS):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=chunk - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


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
    """Build the widest possible window: yfinance (~60d) extended by Schwab (~48d).

    Both calls use their respective max windows; overlap is resolved in
    merge_bars by keeping Schwab (1-min) over yfinance (5-min) on shared days.
    """
    yf_start = end - timedelta(days=_YFINANCE_WINDOW_DAYS)
    yf_df = _fetch_yfinance(symbol, yf_start, end)

    sch_start = end - timedelta(days=_SCHWAB_WINDOW_DAYS)
    sch_df = _fetch_schwab(symbol, sch_start, end)

    merged = merge_bars(older=yf_df, newer=sch_df)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build local bar cache.")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_UNIVERSE),
        help=f"Comma-separated tickers. Default: {','.join(DEFAULT_UNIVERSE)}",
    )
    parser.add_argument(
        "--source",
        choices=["merged", "schwab", "yfinance"],
        default="merged",
        help="Data source. 'merged' uses yfinance 5m for old days + Schwab 1m for recent.",
    )
    parser.add_argument("--days", type=int, default=120,
                        help="Trailing calendar days (schwab-only mode). Default 120.")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default today).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write parquet files into. "
            "Defaults to strategy/data_cache/. "
            "Set to strategy/schwab_cache/ to keep Schwab 1-min bars separate from "
            "the Yahoo 5-min training cache."
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

    log.info("source=%s symbols=%s end=%s output_dir=%s", args.source, symbols, end, output_dir or "default")

    for sym in symbols:
        if args.source == "schwab":
            start = end - timedelta(days=args.days)
            df = _fetch_schwab(sym, start, end)
        elif args.source == "yfinance":
            start = end - timedelta(days=_YFINANCE_WINDOW_DAYS)
            df = _fetch_yfinance(sym, start, end)
        else:  # merged
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
