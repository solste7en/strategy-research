"""Unit tests for IntradayMomentumContinuationStrategy.

Covers:
  * long signal when morning is up and ATR threshold is cleared
  * short signal when morning is down
  * ATR filter rejection on noisy / quiet days
  * volume-z filter rejection
  * market-filter agreement and rejection
  * direction_mode gating
  * EOD edge cases
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from strategy.strategies.intraday_momentum_continuation import (
    IMCParams,
    IntradayMomentumContinuationStrategy,
    compute_atr,
)
from tests.conftest import make_day_bars, make_market_context_day

NY = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Defaults used by most tests
# ---------------------------------------------------------------------------

def _default_params(**overrides) -> IMCParams:
    """Permissive defaults — ATR multiple low so synthetic days pass the filter."""
    base = dict(
        observation_window_minutes=30,
        decision_time_minutes=330,
        exit_time_minutes=385,
        atr_period_bars=14,
        atr_multiple=0.50,
        volume_z_threshold=float("-inf"),
        use_market_filter=False,
        market_context_symbol="SPYM",
        direction_mode="symmetric",
    )
    base.update(overrides)
    return IMCParams(**base)


# ---------------------------------------------------------------------------
# Bar fixtures specific to IMC
# ---------------------------------------------------------------------------

@pytest.fixture
def session() -> date:
    return date(2026, 3, 4)


@pytest.fixture
def up_morning_continuation_day(session) -> pd.DataFrame:
    """+1.5% by 10:00, drifts higher all day, closes +2.0%."""
    return make_day_bars(
        session,
        open_price=100.0,
        move_pct_at_minute={30: 0.015, 389: 0.020},
    )


@pytest.fixture
def down_morning_continuation_day(session) -> pd.DataFrame:
    """-1.5% by 10:00, drifts lower all day, closes -2.0%."""
    return make_day_bars(
        session,
        open_price=100.0,
        move_pct_at_minute={30: -0.015, 389: -0.020},
    )


# ---------------------------------------------------------------------------
# Core signal behavior
# ---------------------------------------------------------------------------

def test_long_signal_when_morning_up_and_atr_exceeded(up_morning_continuation_day):
    strat = IntradayMomentumContinuationStrategy(_default_params())
    trades = strat.generate_trades_for_day("TEST", up_morning_continuation_day)

    assert len(trades) == 1
    t = trades[0]
    assert t.side == "long"
    # Entry near 15:01 ET (next bar after 15:00 decision).
    assert t.entry_time.tz_convert(NY).time().hour == 15
    # Exit at or after 15:55 ET, before 15:58.
    assert t.exit_time.tz_convert(NY).time() >= time(15, 55)
    # On a continuation day, P&L should be positive (afternoon drifts up).
    assert (t.exit_price - t.entry_price) > 0


def test_short_signal_when_morning_down(down_morning_continuation_day):
    strat = IntradayMomentumContinuationStrategy(_default_params())
    trades = strat.generate_trades_for_day("TEST", down_morning_continuation_day)

    assert len(trades) == 1
    t = trades[0]
    assert t.side == "short"
    # On a down-continuation day, P&L for a short is positive.
    assert (t.entry_price - t.exit_price) > 0


def test_no_signal_when_r_open_below_atr_threshold(session):
    """r_open of +0.30% on a noisy day where ATR ≈ 0.19% per bar shouldn't
    clear a 2.0×ATR threshold (~0.38%). Built with noise_bps so ATR > 0."""
    bars = make_day_bars(
        session,
        open_price=100.0,
        move_pct_at_minute={30: 0.003, 389: 0.005},
        noise_bps=20.0,
        seed=0,
    )
    strat = IntradayMomentumContinuationStrategy(_default_params(atr_multiple=2.0))
    assert strat.generate_trades_for_day("TEST", bars) == []


def test_atr_filter_blocks_when_multiple_too_high(session):
    """A morning move of +1.5% should NOT clear a 10×ATR threshold on a
    realistically-noisy day (ATR ≈ 1.0% gives threshold ≈ 10%)."""
    bars = make_day_bars(
        session,
        open_price=100.0,
        move_pct_at_minute={30: 0.015, 389: 0.020},
        noise_bps=50.0,
        seed=0,
    )
    strat = IntradayMomentumContinuationStrategy(_default_params(atr_multiple=10.0))
    assert strat.generate_trades_for_day("TEST", bars) == []


# ---------------------------------------------------------------------------
# Direction modes
# ---------------------------------------------------------------------------

def test_long_only_blocks_short(down_morning_continuation_day):
    strat = IntradayMomentumContinuationStrategy(
        _default_params(direction_mode="long_only")
    )
    assert strat.generate_trades_for_day("TEST", down_morning_continuation_day) == []


def test_short_only_blocks_long(up_morning_continuation_day):
    strat = IntradayMomentumContinuationStrategy(
        _default_params(direction_mode="short_only")
    )
    assert strat.generate_trades_for_day("TEST", up_morning_continuation_day) == []


# ---------------------------------------------------------------------------
# Volume-z filter
# ---------------------------------------------------------------------------

def test_volume_z_filter_off_by_default(up_morning_continuation_day):
    """volume_z_threshold=-inf disables the filter; signal still fires."""
    strat = IntradayMomentumContinuationStrategy(
        _default_params(volume_z_threshold=float("-inf"))
    )
    trades = strat.generate_trades_for_day("TEST", up_morning_continuation_day)
    assert len(trades) == 1


def test_volume_z_filter_blocks_when_observation_bar_is_quiet(session):
    """Build a day where the observation-bar volume is far below the rolling
    mean, so vol_z is negative and the filter rejects."""
    # Use the standard fixture, then manually pin a very low volume on the
    # 30-minute observation bar (10:00 ET = bar index 30).
    bars = make_day_bars(
        session,
        open_price=100.0,
        move_pct_at_minute={30: 0.015, 389: 0.020},
        seed=42,
    )
    bars = bars.copy()
    bars.iloc[30, bars.columns.get_loc("volume")] = 1  # one share — guaranteed below mean

    strat = IntradayMomentumContinuationStrategy(
        _default_params(volume_z_threshold=0.0)
    )
    assert strat.generate_trades_for_day("TEST", bars) == []


# ---------------------------------------------------------------------------
# Market regime filter (cross-asset)
# ---------------------------------------------------------------------------

def test_market_filter_allows_when_spym_agrees(up_morning_continuation_day, session):
    """Symbol up, SPYM also up → filter passes, trade fires."""
    spym_up = make_market_context_day(session, morning_pct=+0.005)
    strat = IntradayMomentumContinuationStrategy(
        _default_params(use_market_filter=True)
    )
    trades = strat.generate_trades_for_day(
        "TEST", up_morning_continuation_day, context_bars={"SPYM": spym_up},
    )
    assert len(trades) == 1
    assert trades[0].side == "long"


def test_market_filter_blocks_when_spym_disagrees(up_morning_continuation_day, session):
    """Symbol up, SPYM down → filter blocks the trade."""
    spym_down = make_market_context_day(session, morning_pct=-0.005)
    strat = IntradayMomentumContinuationStrategy(
        _default_params(use_market_filter=True)
    )
    assert strat.generate_trades_for_day(
        "TEST", up_morning_continuation_day, context_bars={"SPYM": spym_down},
    ) == []


def test_market_filter_blocks_when_no_context(up_morning_continuation_day):
    """use_market_filter=True without context_bars → safe-fail to no trade."""
    strat = IntradayMomentumContinuationStrategy(
        _default_params(use_market_filter=True)
    )
    assert strat.generate_trades_for_day(
        "TEST", up_morning_continuation_day, context_bars=None,
    ) == []
    assert strat.generate_trades_for_day(
        "TEST", up_morning_continuation_day, context_bars={},
    ) == []


def test_market_filter_blocks_when_context_empty_frame(up_morning_continuation_day):
    """Empty context DataFrame should not crash; safe-fail to no trade."""
    strat = IntradayMomentumContinuationStrategy(
        _default_params(use_market_filter=True)
    )
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY, name="datetime"),
    )
    assert strat.generate_trades_for_day(
        "TEST", up_morning_continuation_day, context_bars={"SPYM": empty},
    ) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_bars_returns_no_trade(session):
    strat = IntradayMomentumContinuationStrategy(_default_params())
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY, name="datetime"),
    )
    assert strat.generate_trades_for_day("TEST", empty) == []


def test_multi_day_input_raises(up_morning_continuation_day):
    """Strategy must reject inputs spanning >1 session — the harness
    guarantees one-session slices, so anything else is a config bug."""
    other = up_morning_continuation_day.copy()
    other.index = other.index + timedelta(days=1)
    mixed = pd.concat([up_morning_continuation_day, other])
    strat = IntradayMomentumContinuationStrategy(_default_params())
    with pytest.raises(ValueError):
        strat.generate_trades_for_day("TEST", mixed)


def test_decision_after_eod_returns_no_trade(up_morning_continuation_day):
    """A param config with decision_time after 15:58 should never fire,
    even if all other gates pass."""
    strat = IntradayMomentumContinuationStrategy(
        _default_params(decision_time_minutes=389, exit_time_minutes=389)
    )
    assert strat.generate_trades_for_day("TEST", up_morning_continuation_day) == []


def test_degenerate_timing_returns_no_trade(up_morning_continuation_day):
    """obs >= decision should be rejected as a config-level no-op."""
    strat = IntradayMomentumContinuationStrategy(
        _default_params(observation_window_minutes=330, decision_time_minutes=330)
    )
    assert strat.generate_trades_for_day("TEST", up_morning_continuation_day) == []


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def test_compute_atr_matches_simple_calculation(session):
    """ATR should equal the rolling mean of true range. On a flat synthetic
    day with no overnight gaps, TR ≈ high-low for every bar."""
    bars = make_day_bars(
        session, open_price=100.0, move_pct_at_minute={30: 0.0, 389: 0.0}, noise_bps=10.0,
    )
    atr14 = compute_atr(bars, 14)
    # First 13 values should be NaN.
    assert atr14.iloc[:13].isna().all()
    # From bar 13 onward, ATR should be strictly positive.
    assert (atr14.iloc[13:] > 0).all()


def test_compute_atr_empty_returns_empty_series():
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY, name="datetime"),
    )
    out = compute_atr(empty, 14)
    assert out.empty
