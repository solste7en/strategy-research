"""Synthetic-bar fixtures for deterministic strategy tests.

No network. Builds a regular-session (09:30–16:00 ET) 1-minute DataFrame
with a configurable open-to-decision move and a configurable close.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

NY = ZoneInfo("America/New_York")


def make_day_bars(
    session: date,
    open_price: float = 100.0,
    move_pct_at_minute: dict[int, float] | None = None,
    close_price: float | None = None,
    noise_bps: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a single regular-session 1-minute bar DataFrame (390 bars).

    Price path:
      * 09:30 open = open_price
      * At each minute key in move_pct_at_minute, the price is pinned to
        open_price * (1 + move_pct).
      * The path linearly interpolates between pinned points.
      * If close_price is given, the 15:59 bar's close is pinned there.
      * noise_bps adds uniform noise to each bar's high/low (not close).
    """
    rng = np.random.default_rng(seed)
    start_dt = datetime.combine(session, time(9, 30), tzinfo=NY)
    minutes = [start_dt + timedelta(minutes=i) for i in range(390)]

    pinned = {0: 0.0}
    if move_pct_at_minute:
        pinned.update(move_pct_at_minute)
    if close_price is not None:
        pinned[389] = (close_price - open_price) / open_price

    keys = sorted(pinned)
    # Linear interpolation of pct-move per minute
    pct_series = np.zeros(390)
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        pct_series[a:b + 1] = np.linspace(pinned[a], pinned[b], b - a + 1)
    # Fill tail if last key < 389
    if keys[-1] < 389:
        pct_series[keys[-1] + 1:] = pinned[keys[-1]]

    closes = open_price * (1.0 + pct_series)
    # Open of bar n = close of bar n-1 (typical minute-bar convention).
    opens = np.empty_like(closes)
    opens[0] = open_price
    opens[1:] = closes[:-1]

    noise = rng.uniform(-noise_bps, noise_bps, size=390) / 10000.0
    highs = np.maximum(opens, closes) * (1.0 + np.abs(noise))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(noise))
    volumes = rng.integers(10_000, 100_000, size=390)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(minutes, tz=NY, name="datetime"),
    )


@pytest.fixture
def sample_session() -> date:
    return date(2026, 3, 4)


@pytest.fixture
def up_overextension_day(sample_session):
    """A day where the stock is +2% by 10:00 ET and closes -1% from open.

    Expected: short-entry signal for a symmetric strategy with threshold ≤ 2%
    and entry_window ≥ 30m; P&L should be positive.
    """
    return make_day_bars(
        sample_session,
        open_price=100.0,
        move_pct_at_minute={30: 0.02, 389: -0.01},
    )


@pytest.fixture
def down_overextension_day(sample_session):
    """A day where the stock is -2% by 10:00 ET and closes +1% from open."""
    return make_day_bars(
        sample_session,
        open_price=100.0,
        move_pct_at_minute={30: -0.02, 389: 0.01},
    )


@pytest.fixture
def quiet_day(sample_session):
    """A day that moves only ±0.3% all session — below any threshold."""
    return make_day_bars(
        sample_session,
        open_price=100.0,
        move_pct_at_minute={30: 0.003, 180: -0.003, 389: 0.001},
    )


def make_market_context_day(
    session: date,
    morning_pct: float,
    open_price: float = 500.0,
) -> pd.DataFrame:
    """Build a market-index context day (e.g. SPYM) with a clean morning move.

    Pinned to ``morning_pct`` at minute 30 (10:00 ET) and held flat through
    the rest of the session, so the IMC market filter sees a stable r_open.
    """
    return make_day_bars(
        session,
        open_price=open_price,
        move_pct_at_minute={30: morning_pct, 389: morning_pct},
    )
