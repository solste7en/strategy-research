"""Entry/exit logic tests for IntradayOverextensionStrategy."""
from datetime import time
from zoneinfo import ZoneInfo

import pytest

from strategy.strategies.intraday_overextension import (
    IntradayOverextensionParams,
    IntradayOverextensionStrategy,
)

NY = ZoneInfo("America/New_York")


def test_up_overextension_triggers_short(up_overextension_day):
    """+2% in first 30m → symmetric strategy should enter short."""
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01, exit_window_minutes=15
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", up_overextension_day)

    assert len(trades) == 1
    t = trades[0]
    assert t.side == "short"
    # Entry at 10:01 (bar after decision bar at 10:00)
    assert t.entry_time.time() == time(10, 1)
    # Exit at 15:45 (close − 15 min), fills on 15:45 bar's close
    assert t.exit_time.time() == time(15, 45)
    # Short wins when exit < entry — day closed below the extension
    assert t.exit_price < t.entry_price


def test_down_overextension_triggers_long_when_symmetric(down_overextension_day):
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01,
        exit_window_minutes=15, direction_mode="symmetric",
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", down_overextension_day)

    assert len(trades) == 1
    t = trades[0]
    assert t.side == "long"
    # Long wins when exit > entry — day bounced off the extension low
    assert t.exit_price > t.entry_price


def test_short_only_ignores_down_move(down_overextension_day):
    """direction_mode='short_only' means we only fade UP moves. A -2% move must NOT fire."""
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01,
        exit_window_minutes=15, direction_mode="short_only",
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", down_overextension_day)
    assert trades == []


def test_long_only_ignores_up_move(up_overextension_day):
    """direction_mode='long_only' means we only fade DOWN moves. A +2% move must NOT fire."""
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01,
        exit_window_minutes=15, direction_mode="long_only",
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", up_overextension_day)
    assert trades == []


def test_below_threshold_no_trade(quiet_day):
    """A sub-threshold move produces no trades."""
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01, exit_window_minutes=15
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", quiet_day)
    assert trades == []


def test_threshold_boundary_inclusive_exclusive(up_overextension_day):
    """At the exact threshold, we do *not* fire — only a strict overshoot."""
    # Up move is 2%; threshold 2% should be exclusive and therefore not fire
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.02, exit_window_minutes=15
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", up_overextension_day)
    assert trades == []


def test_empty_bars_returns_no_trades():
    import pandas as pd
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01, exit_window_minutes=15
    )
    strat = IntradayOverextensionStrategy(params)
    empty = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz=NY, name="datetime"),
    )
    assert strat.generate_trades_for_day("TEST", empty) == []


def test_multi_day_input_raises(up_overextension_day, down_overextension_day):
    """Strategy must reject inputs that span >1 session."""
    import pandas as pd
    mixed = pd.concat([up_overextension_day, down_overextension_day])
    # Only true if fixtures used different dates — our fixtures share a date,
    # so we build a true multi-day frame inline:
    from tests.conftest import make_day_bars
    from datetime import date
    d1 = make_day_bars(date(2026, 3, 4), move_pct_at_minute={30: 0.02})
    d2 = make_day_bars(date(2026, 3, 5), move_pct_at_minute={30: -0.02})
    mixed = pd.concat([d1, d2])
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01, exit_window_minutes=15
    )
    strat = IntradayOverextensionStrategy(params)
    with pytest.raises(ValueError):
        strat.generate_trades_for_day("TEST", mixed)


def test_shares_is_zero_sentinel_until_harness_fills(up_overextension_day):
    """Strategy doesn't know sizing — it emits shares=0, harness fills in."""
    params = IntradayOverextensionParams(
        entry_window_minutes=30, threshold_pct=0.01, exit_window_minutes=15
    )
    strat = IntradayOverextensionStrategy(params)
    trades = strat.generate_trades_for_day("TEST", up_overextension_day)
    assert trades[0].shares == 0
