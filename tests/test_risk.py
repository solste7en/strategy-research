"""RiskManager and sizing tests."""
from datetime import date

import pytest

from strategy.risk import DEFAULT_UNIVERSE, RiskConfig, RiskManager, size_trade


def test_size_trade_returns_fractional_shares():
    # $100 budget, $49.99 price → 100/49.99 ≈ 2.0004 shares (fractional OK)
    assert size_trade(100.0, 49.99) == pytest.approx(100.0 / 49.99)


def test_size_trade_fractional_for_high_priced_stock():
    # $100 budget, $500 stock → 0.2 shares (no minimum-1-share rule; brokers support fractional)
    assert size_trade(100.0, 500.0) == pytest.approx(0.2)


def test_size_trade_rejects_nonpositive_price():
    with pytest.raises(ValueError):
        size_trade(100.0, 0)
    with pytest.raises(ValueError):
        size_trade(100.0, -1.5)


def test_risk_config_universe_size_cap():
    # max_universe_size defaults to 20; 21 tickers should raise
    with pytest.raises(ValueError):
        RiskConfig(universe=tuple(f"T{i}" for i in range(21)))


def test_can_trade_rejects_non_universe_symbols():
    rm = RiskManager()
    allowed, reason = rm.can_trade(date(2026, 4, 22), "GOOGL")
    assert not allowed
    assert "universe" in reason


def test_daily_trade_cap():
    rm = RiskManager(config=RiskConfig(max_trades_per_day=3))
    d = date(2026, 4, 22)
    for _ in range(3):
        allowed, _ = rm.can_trade(d, "SPYM")
        assert allowed
        rm.register_trade(d)
    allowed, reason = rm.can_trade(d, "SPYM")
    assert not allowed
    assert "daily trade cap" in reason


def test_trade_counter_resets_across_days():
    rm = RiskManager(config=RiskConfig(max_trades_per_day=1))
    rm.register_trade(date(2026, 4, 22))
    allowed, _ = rm.can_trade(date(2026, 4, 23), "SPYM")
    assert allowed


def test_default_universe():
    assert len(DEFAULT_UNIVERSE) == 12
    assert set(DEFAULT_UNIVERSE) == {
        "SPYM", "TQQQ", "TSLA", "NVDA", "COIN",
        "LYFT", "UBER", "HIMS", "RBLX", "HOOD", "PDD", "NFLX",
    }
