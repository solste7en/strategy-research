"""Intraday bar providers.

Three implementations share a common `BarsProvider` interface:

    SchwabBarsProvider   — hits Schwab's priceHistory endpoint (1-min bars,
                           ~48 calendar days back)
    YFinanceBarsProvider — hits Yahoo via yfinance (5-min bars, ~60 days back
                           — that's the Yahoo-side max for sub-hourly)
    ParquetBarsProvider  — reads a local parquet cache (fast replays for
                           backtest)

The ``merge_bars`` helper stitches Schwab (recent, 1-min) + yfinance
(pre-Schwab window, 5-min) into one continuous timeseries. The strategy
treats all bars uniformly by looking up "closest bar at or before target
time", so mixed 1m/5m granularity works without special handling, at the
cost of slightly coarser fill prices on the yfinance portion.

The cache layout is one parquet file per symbol under ``strategy/data_cache/``.

All bars are **regular session only (09:30–16:00 America/New_York)** with a
timezone-aware DatetimeIndex. Column order: open, high, low, close, volume.
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
# Schwab live / cache-building provider
# ---------------------------------------------------------------------------


class SchwabBarsProvider:
    """Pulls 1-minute bars directly from Schwab's priceHistory API.

    Used by ``scripts/fetch_intraday_history.py`` to build the parquet cache,
    and by the live runner to fetch today's bars up to the current minute.

    Schwab's minute-bar window on `get_price_history_every_minute` is
    approximately 48 calendar days when called without explicit start/end
    — we pass datetimes to extend. If a requested range exceeds the broker's
    window, the call returns a truncated payload; caller can chunk.
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
# Local parquet cache provider (fast backtest)
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
# yfinance provider (5-min bars, ~60 days back)
# ---------------------------------------------------------------------------


class YFinanceBarsProvider:
    """Pulls 5-minute bars from Yahoo via yfinance.

    Yahoo caps sub-hourly history at ~60 calendar days regardless of the
    requested range. For our theory (15/30/60-min entry windows) 5-min
    granularity is sufficient: decision and entry/exit fills snap to the
    nearest 5-min boundary.

    Used only to *extend* history backwards beyond Schwab's ~48-day minute
    window. For the most recent sessions prefer SchwabBarsProvider's 1-min
    bars when available.
    """

    def __init__(self, interval: str = "5m"):
        if interval not in {"1m", "2m", "5m", "15m", "30m", "60m", "1h"}:
            raise ValueError(f"unsupported yfinance interval: {interval}")
        self.interval = interval

    def get_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf  # local import — only needed when actually fetching

        # yfinance's `period="60d"` fetches the maximum window in one call and
        # is more reliable than explicit start/end for sub-hourly intervals.
        # We fetch the max and then slice by caller's range.
        t = yf.Ticker(symbol)
        df = t.history(period="60d", interval=self.interval, auto_adjust=False)
        if df.empty:
            log.warning("yfinance returned no bars for %s", symbol)
            return _empty_bars()

        # yfinance may index in UTC or localtime depending on version — normalize.
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df.index = idx.tz_convert(NY)

        # Standardize column names to lower-case OHLCV.
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]].astype({
            "open": float, "high": float, "low": float, "close": float,
            "volume": "int64",
        })
        df.index.name = "datetime"
        df = _filter_regular_session(df.sort_index())

        # Slice to caller's date range inclusive.
        idx_local = df.index.tz_convert(NY)
        mask = (idx_local.date >= start) & (idx_local.date <= end)
        return df.loc[mask]


# ---------------------------------------------------------------------------
# Merge helpers (build combined cache from Schwab + yfinance)
# ---------------------------------------------------------------------------


def merge_bars(
    older: pd.DataFrame, newer: pd.DataFrame
) -> pd.DataFrame:
    """Stitch two bar DataFrames into one.

    For days present in BOTH frames, prefer ``newer`` (typically Schwab 1-min
    = higher fidelity than yfinance 5-min). For days only in ``older``,
    keep the yfinance bars. Output is sorted and deduped on the datetime index.
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
