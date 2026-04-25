"""RiskManager and sizing tests."""
from datetime import date

import pytest

from strategy.risk import DEFAULT_UNIVERSE, RiskConfig, RiskManager, size_trade


def test_size_trade_floors_to_whole_shares():
    # $100 budget, $49.99 price → floor(100/49.99) = 2 shares
    assert size_trade(100.0, 49.99) == 2


def test_size_trade_always_at_least_one_share():
    # $100 budget, $500 price → floor would be 0, rule bumps to 1
    assert size_trade(100.0, 500.0) == 1


def test_size_trade_rejects_nonpositive_price():
    with pytest.raises(ValueError):
        size_trade(100.0, 0)
    with pytest.raises(ValueError):
        size_trade(100.0, -1.5)


def test_risk_config_universe_size_cap():
    with pytest.raises(ValueError):
        RiskConfig(universe=("A", "B", "C", "D", "E", "F"))


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


def test_default_universe_is_five_tickers():
    assert len(DEFAULT_UNIVERSE) == 5
    assert set(DEFAULT_UNIVERSE) == {"SPYM", "TQQQ", "TSLA", "NVDA", "COIN"}
