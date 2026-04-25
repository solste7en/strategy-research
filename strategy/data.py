"""Intraday bar providers.

Three implementations share a common `BarsProvider` interface:

    AlpacaBarsProvider  — Alpaca Market Data API (1-min SIP bars, ~7yr history;
                          free account sufficient). Primary source for backtesting.
    SchwabBarsProvider  — Schwab priceHistory endpoint (1-min bars, ~48 calendar
                          days back). Backup / live-data source only.
    ParquetBarsProvider — reads a local parquet cache (fast replays for backtest;
                          no network calls).

Workflow: build the cache once with ``scripts/fetch_intraday_history.py
--source alpaca`` (default start 2023-01-01), then run all backtests against
the local parquet cache via ParquetBarsProvider. Use ``--source schwab`` only
to top up the cache with bars from the last 48 days if needed.

The ``merge_bars`` helper stitches two providers into one continuous timeseries,
preferring the ``newer`` frame on overlapping days (useful for blending Alpaca
history with recent Schwab bars).

Cache layout: one parquet file per symbol under ``strategy/data_cache/``.

All bars are **regular session only (09:30–16:00 America/New_York)** with a
timezone-aware DatetimeIndex. Core columns: open, high, low, close, volume.
AlpacaBarsProvider additionally populates ``vwap`` and ``trade_count`` per bar.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data_cache"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class BarsProvider(Protocol):
    def get_bars(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """Return 1-minute bars for ``symbol`` between ``start`` and ``end``
        inclusive (regular session only, NY tz-aware index, OHLCV columns)."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30:00 ≤ t < 16:00:00 ET bars."""
    if df.empty:
        return df
    idx_local = df.index.tz_convert(NY)
    mask = (idx_local.time >= SESSION_OPEN) & (idx_local.time < SESSION_CLOSE)
    return df.loc[mask]


def split_by_session_date(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Group bars by NY session date (09:30 ET boundaries)."""
    if df.empty:
        return {}
    idx_local = df.index.tz_convert(NY)
    return {
        d: df.loc[idx_local.date == d]
        for d in sorted({dt.date() for dt in idx_local})
    }


# ---------------------------------------------------------------------------
# Alpaca provider (primary — 1-min SIP bars, ~7yr history, free account)
# ---------------------------------------------------------------------------


class AlpacaBarsProvider:
    """Pulls 1-minute bars from Alpaca's Market Data API.

    Uses the SIP (consolidated tape) feed, which covers all US exchanges and
    is available for historical data (older than 15 minutes) on the free plan.
    A free paper-trading account at alpaca.markets is sufficient — no funded
    account is needed.

    Credentials are resolved in this order:
      1. Explicit ``api_key`` / ``api_secret`` constructor arguments.
      2. ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` environment variables
         (loaded from a ``.env`` file in the repo root if python-dotenv is
         installed).

    Returns bars with columns: open, high, low, close, volume, vwap,
    trade_count. The extra ``vwap`` (bar-level VWAP from the consolidated
    tape) and ``trade_count`` (number of individual trades in the bar) exceed
    the base BarsProvider contract but are available to strategies that want
    them and are preserved in the parquet cache.

    Prices are split-adjusted (``adjustment="split"``).
    """

    def __init__(self, api_key: str | None = None, api_secret: str | None = None):
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import os
            from alpaca.data.historical import StockHistoricalDataClient  # lazy

            # Try loading .env from the repo root so credentials don't need
            # to be exported in the shell manually.
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            except ImportError:
                pass

            key = self._api_key or os.environ.get("ALPACA_API_KEY")
            secret = self._api_secret or os.environ.get("ALPACA_API_SECRET")
            if not key or not secret:
                raise RuntimeError(
                    "Alpaca credentials not found. Set ALPACA_API_KEY and "
                    "ALPACA_API_SECRET in your .env file or as environment "
                    "variables. A free paper-trading account at "
                    "alpaca.markets is sufficient."
                )
            self._client = StockHistoricalDataClient(api_key=key, secret_key=secret)
        return self._client

    def get_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest          # lazy
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # lazy

        start_dt = datetime.combine(start, SESSION_OPEN, tzinfo=NY)
        # End at 16:00 on the last day so the final session is fully included.
        end_dt = datetime.combine(end, SESSION_CLOSE, tzinfo=NY)

        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_dt,
            end=end_dt,
            feed="sip",          # consolidated tape, all exchanges
            adjustment="split",  # split-adjusted prices
        )
        log.info("alpaca: fetching %s %s..%s (SIP, 1-min)", symbol, start, end)
        bars = self.client.get_stock_bars(request)
        df: pd.DataFrame = bars.df

        if df.empty:
            log.warning("Alpaca returned no bars for %s %s..%s", symbol, start, end)
            return _empty_bars()

        # Response has a MultiIndex (symbol, timestamp) — drop the symbol level.
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)

        # Normalize timezone to NY.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(NY)
        df.index.name = "datetime"
        df = df.sort_index()

        # Select and type-cast available columns.
        want = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
        cols = [c for c in want if c in df.columns]
        dtype_map: dict[str, str] = {
            "open": "float64", "high": "float64", "low": "float64",
            "close": "float64", "volume": "int64",
            "vwap": "float64", "trade_count": "int64",
        }
        df = df[cols].astype({c: dtype_map[c] for c in cols})

        return _filter_regular_session(df)


# ---------------------------------------------------------------------------
# Schwab provider (backup — 1-min bars, ~48 calendar days back)
# ---------------------------------------------------------------------------


class SchwabBarsProvider:
    """Pulls 1-minute bars directly from Schwab's priceHistory API.

    Backup / live-data source. Use when you need bars from the last 48 days
    that are fresher than Alpaca's 15-minute historical delay, or to top up
    the cache with sessions not yet covered by Alpaca.

    Requires token.json + Schwab credentials from the companion schwab_app
    repo (core.auth.get_client). Not needed for pure backtesting from cache.

    Schwab's minute-bar window is approximately 48 calendar days; we chunk
    requests in 45-day windows when the requested range is larger.
    """

    def __init__(self, client=None):
        """client: an authenticated schwab-py client. If None, resolves lazily
        via core.auth.get_client() so callers don't have to import it."""
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from core.auth import get_client  # local import: avoid cycles in tests
            self._client = get_client()
        return self._client

    def get_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        start_dt = datetime.combine(start, SESSION_OPEN, tzinfo=NY)
        # +1 day so the exclusive end on the Schwab side includes the full close
        end_dt = datetime.combine(end + timedelta(days=1), SESSION_OPEN, tzinfo=NY)

        resp = self.client.get_price_history_every_minute(
            symbol,
            start_datetime=start_dt,
            end_datetime=end_dt,
            need_extended_hours_data=False,
        )
        resp.raise_for_status()
        payload = resp.json()
        candles = payload.get("candles", [])
        if not candles:
            log.warning("No candles returned for %s %s..%s", symbol, start, end)
            return _empty_bars()

        df = pd.DataFrame(candles)
        # Schwab returns `datetime` as epoch-ms
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert(NY)
        df = df.set_index("datetime").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": "int64"}
        )
        return _filter_regular_session(df)


# ---------------------------------------------------------------------------
# Local parquet cache provider (fast backtest — no network)
# ---------------------------------------------------------------------------


class ParquetBarsProvider:
    """Reads 1-minute bars from local parquet cache.

    The cache is built once via ``scripts/fetch_intraday_history.py``. Files
    live at ``strategy/data_cache/{SYMBOL}_1min.parquet``. Missing files
    raise FileNotFoundError with a hint.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._loaded: dict[str, pd.DataFrame] = {}

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_1min.parquet"

    def _load(self, symbol: str) -> pd.DataFrame:
        if symbol in self._loaded:
            return self._loaded[symbol]
        p = self._path(symbol)
        if not p.exists():
            raise FileNotFoundError(
                f"No cache for {symbol} at {p}. Run "
                f"`python scripts/fetch_intraday_history.py --symbols {symbol}` first."
            )
        df = pd.read_parquet(p)
        # Normalize tz — parquet round-trips as UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(NY)
        df = df.sort_index()
        self._loaded[symbol] = df
        return df

    def get_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        df = self._load(symbol)
        idx_local = df.index.tz_convert(NY)
        mask = (idx_local.date >= start) & (idx_local.date <= end)
        return _filter_regular_session(df.loc[mask])

    def trading_dates(self, symbol: str) -> list[date]:
        """Return the distinct session dates available for this symbol."""
        df = self._load(symbol)
        return sorted({dt.date() for dt in df.index.tz_convert(NY)})


def write_cache(symbol: str, df: pd.DataFrame, cache_dir: Path | None = None) -> Path:
    """Persist a bars DataFrame to the parquet cache. Overwrites existing file."""
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol.upper()}_1min.parquet"
    # Store in UTC to avoid tz ambiguity in parquet; reader converts to NY.
    out = df.copy()
    out.index = out.index.tz_convert("UTC")
    out.to_parquet(path)
    return path


def _empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz=NY, name="datetime")
    return pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=idx,
    ).astype({"volume": "int64"})


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_bars(older: pd.DataFrame, newer: pd.DataFrame) -> pd.DataFrame:
    """Stitch two bar DataFrames into one continuous timeseries.

    For days present in BOTH frames, ``newer`` wins (higher fidelity or more
    recent source). For days only in ``older``, those bars are kept as-is.
    Output is sorted and deduplicated on the datetime index.

    Typical use: blend Alpaca history (older) with recent Schwab bars (newer)
    to extend the cache up to the current session.
    """
    if older.empty and newer.empty:
        return _empty_bars()
    if older.empty:
        return newer.sort_index()
    if newer.empty:
        return older.sort_index()

    newer_dates = {dt.date() for dt in newer.index.tz_convert(NY)}
    older_idx_local = older.index.tz_convert(NY)
    older_only_mask = ~pd.Index([d in newer_dates for d in older_idx_local.date])
    older_kept = older.loc[older_only_mask]

    merged = pd.concat([older_kept, newer]).sort_index()
    # Final dedupe (same timestamp from both sources is rare but possible).
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged
