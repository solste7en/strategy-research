"""Entry/exit logic tests for MFIDivergenceStrategy."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategy.strategies.mfi_divergence import (
    MFIDivergenceParams,
    MFIDivergenceStrategy,
    compute_mfi,
)

NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Synthetic-day builder for divergence scenarios
# ---------------------------------------------------------------------------


def _five_min_session(
    session: date,
    price_path: list[float],
    volume_path: list[int],
) -> pd.DataFrame:
    """Build a 5-minute bar DataFrame (78 bars from 09:30 to 15:55 ET).

    price_path: list of 78 close prices (one per bar). Opens are set to the
    prior close; high/low are +/- 5 bps around the close.
    volume_path: list of 78 volumes aligned to the bars.
    """
    n_bars = 78
    if len(price_path) != n_bars or len(volume_path) != n_bars:
        raise ValueError(f"paths must have length {n_bars}")

    start_dt = datetime.combine(session, time(9, 30), tzinfo=NY)
    index = pd.DatetimeIndex(
        [start_dt + timedelta(minutes=5 * i) for i in range(n_bars)],
        tz=NY,
        name="datetime",
    )
    closes = np.asarray(price_path, dtype=float)
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    # Small high/low band so divergence logic has clean extremes from the close path.
    highs = np.maximum(opens, closes) * 1.0005
    lows = np.minimum(opens, closes) * 0.9995
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume_path},
        index=index,
    )


@pytest.fixture
def sample_session() -> date:
    return date(2026, 3, 4)


@pytest.fixture
def bullish_divergence_day(sample_session) -> pd.DataFrame:
    """Construct a session where:

      * First low at bar 35 on HEAVY volume (selling climax).
      * Rally to bar 47.
      * Deeper low at bar 55 on LIGHT volume (selling exhausted).

    Both lows fall within a 24-bar lookback from bar 55 (window = bars 31..54).
    Bar 35's low is the min of that window; bar 55 breaks it.
    MFI at bar 55 (rolling 14 = bars 42..55) has positive flow from the rally
    and a light-volume negative flow from the slow decline, so MFI ≈ 30-40.
    MFI at bar 35 (bars 22..35) is nearly pure heavy-volume selling, so MFI ≈ 0-10.
    That's the bullish divergence: deeper low, HIGHER MFI.

    Expected: bullish divergence → LONG.
    """
    n = 78
    prices = [100.0] * n
    for i in range(n):
        if i <= 15:
            prices[i] = 100.0
        elif i <= 35:
            prices[i] = 100.0 - 2.5 * ((i - 15) / 20.0)   # 100 → 97.5 (first low)
        elif i <= 47:
            prices[i] = 97.5 + 2.0 * ((i - 35) / 12.0)    # 97.5 → 99.5 (rally)
        elif i <= 55:
            prices[i] = 99.5 - 2.2 * ((i - 47) / 8.0)     # 99.5 → 97.3 (deeper low)
        else:
            prices[i] = 97.3 + 2.5 * ((i - 55) / (n - 55 - 1))  # bounce toward close

    vols = [20_000] * n
    for i in range(n):
        if 25 <= i <= 40:
            vols[i] = 200_000    # heavy selling into first low
        elif 48 <= i <= 58:
            vols[i] = 25_000     # exhausted selling into deeper low
    return _five_min_session(sample_session, prices, vols)


@pytest.fixture
def bearish_divergence_day(sample_session) -> pd.DataFrame:
    """Mirror image: higher high on lighter volume → bearish divergence → SHORT."""
    n = 78
    prices = [100.0] * n
    for i in range(n):
        if i <= 15:
            prices[i] = 100.0
        elif i <= 35:
            prices[i] = 100.0 + 2.5 * ((i - 15) / 20.0)   # 100 → 102.5 (first high)
        elif i <= 47:
            prices[i] = 102.5 - 2.0 * ((i - 35) / 12.0)   # 102.5 → 100.5 (pullback)
        elif i <= 55:
            prices[i] = 100.5 + 2.2 * ((i - 47) / 8.0)    # 100.5 → 102.7 (new high)
        else:
            prices[i] = 102.7 - 2.5 * ((i - 55) / (n - 55 - 1))

    vols = [20_000] * n
    for i in range(n):
        if 25 <= i <= 40:
            vols[i] = 200_000    # heavy buying into first high
        elif 48 <= i <= 58:
            vols[i] = 25_000     # exhausted buying into new high
    return _five_min_session(sample_session, prices, vols)


@pytest.fixture
def quiet_day(sample_session) -> pd.DataFrame:
    """Flat session — no divergence, no trade."""
    n = 78
    prices = [100.0 + 0.05 * np.sin(i / 5.0) for i in range(n)]
    vols = [50_000] * n
    return _five_min_session(sample_session, prices, vols)


# ---------------------------------------------------------------------------
# MFI math sanity
# ---------------------------------------------------------------------------


def test_mfi_bounded_0_to_100(bullish_divergence_day):
    mfi = compute_mfi(bullish_divergence_day, period=14)
    valid = mfi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()
    # Warmup: the first (period - 1) = 13 bars have no rolling window and so
    # must be NaN. Later bars must contain at least some valid MFI readings.
    assert mfi.iloc[:13].isna().all()
    assert mfi.notna().sum() > 40  # plenty of valid readings across the day


def test_mfi_empty_bars_returns_empty_series():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY, name="datetime"),
    )
    mfi = compute_mfi(empty, period=14)
    assert mfi.empty


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------


def test_bullish_divergence_triggers_long(bullish_divergence_day):
    params = MFIDivergenceParams(
        mfi_period=14,
        divergence_lookback=24,
        oversold_threshold=40,   # loose to guarantee signal fires on synthetic data
        overbought_threshold=60,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    trades = strat.generate_trades_for_day("TEST", bullish_divergence_day)
    assert len(trades) == 1, f"expected exactly one trade, got {len(trades)}"
    t = trades[0]
    assert t.side == "long"
    # Entered AFTER we saw a deeper low with lighter volume → late in the day.
    assert t.entry_time > t.entry_time.replace(hour=11, minute=0)
    assert "bullish div" in t.entry_reason


def test_bearish_divergence_triggers_short(bearish_divergence_day):
    params = MFIDivergenceParams(
        mfi_period=14,
        divergence_lookback=24,
        oversold_threshold=40,
        overbought_threshold=60,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    trades = strat.generate_trades_for_day("TEST", bearish_divergence_day)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "short"
    assert "bearish div" in t.entry_reason


def test_long_only_ignores_bearish_divergence(bearish_divergence_day):
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=40, overbought_threshold=60,
        exit_window_minutes=30, direction_mode="long_only",
    )
    strat = MFIDivergenceStrategy(params)
    assert strat.generate_trades_for_day("TEST", bearish_divergence_day) == []


def test_short_only_ignores_bullish_divergence(bullish_divergence_day):
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=40, overbought_threshold=60,
        exit_window_minutes=30, direction_mode="short_only",
    )
    strat = MFIDivergenceStrategy(params)
    assert strat.generate_trades_for_day("TEST", bullish_divergence_day) == []


def test_quiet_day_no_trade(quiet_day):
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=30, overbought_threshold=70,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    assert strat.generate_trades_for_day("TEST", quiet_day) == []


def test_strict_oversold_threshold_blocks_signal(bullish_divergence_day):
    """With oversold_threshold=5 (very strict), the MFI at the deeper low
    won't be low enough → no trade."""
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=5, overbought_threshold=95,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    assert strat.generate_trades_for_day("TEST", bullish_divergence_day) == []


def test_empty_bars_returns_no_trades():
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=30, overbought_threshold=70,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY, name="datetime"),
    )
    assert strat.generate_trades_for_day("TEST", empty) == []


def test_shares_is_zero_sentinel_until_harness_fills(bullish_divergence_day):
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=40, overbought_threshold=60,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    trades = strat.generate_trades_for_day("TEST", bullish_divergence_day)
    assert trades
    assert trades[0].shares == 0


def test_as_tag_round_trips():
    p = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=12,
        oversold_threshold=20, overbought_threshold=80,
        exit_window_minutes=30, direction_mode="symmetric",
    )
    assert p.as_tag() == "mfi_p14_lb12_os20_ob80_x30_sym"


def test_multi_day_input_raises(sample_session):
    """Strategy must reject inputs that span >1 session."""
    # Build two tiny sessions and concat them
    d1 = _five_min_session(sample_session, [100.0] * 78, [50_000] * 78)
    d2 = _five_min_session(date(2026, 3, 5), [100.0] * 78, [50_000] * 78)
    mixed = pd.concat([d1, d2])
    params = MFIDivergenceParams(
        mfi_period=14, divergence_lookback=24,
        oversold_threshold=30, overbought_threshold=70,
        exit_window_minutes=30,
    )
    strat = MFIDivergenceStrategy(params)
    with pytest.raises(ValueError):
        strat.generate_trades_for_day("TEST", mixed)
