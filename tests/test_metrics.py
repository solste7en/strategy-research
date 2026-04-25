"""Metrics calculation tests."""
from datetime import datetime, timezone

from strategy.metrics import compute_metrics
from strategy.strategies.base import Trade


def _t(side, entry_price, exit_price, day=1, shares=1):
    et = datetime(2026, 3, day, 10, 0, tzinfo=timezone.utc)
    xt = datetime(2026, 3, day, 15, 55, tzinfo=timezone.utc)
    return Trade(
        symbol="TEST", side=side, shares=shares,
        entry_time=et, entry_price=entry_price,
        exit_time=xt, exit_price=exit_price,
        entry_reason="x", exit_reason="y",
    )


def test_empty_trades_returns_zeros():
    m = compute_metrics([])
    assert m.n_trades == 0
    assert m.total_pnl_dollars == 0


def test_long_pnl_sign_and_win_rate():
    trades = [
        _t("long", 100.0, 101.0),   # +$1 win
        _t("long", 100.0, 99.0),    # -$1 loss
        _t("long", 100.0, 102.0),   # +$2 win
    ]
    m = compute_metrics(trades)
    assert m.n_trades == 3
    assert m.n_wins == 2
    assert m.n_losses == 1
    assert m.win_rate == pytest_approx_2_over_3()
    assert m.total_pnl_dollars == 2.0
    assert m.best_trade_dollars == 2.0
    assert m.worst_trade_dollars == -1.0


def test_short_pnl_inverted():
    trades = [
        _t("short", 100.0, 99.0),   # short wins when price falls → +$1
        _t("short", 100.0, 101.0),  # short loses when price rises → -$1
    ]
    m = compute_metrics(trades)
    assert m.total_pnl_dollars == 0.0
    assert m.n_wins == 1


def test_max_drawdown_peak_to_trough():
    # P&L sequence: +5, -3, +2, -6, +4 → cum: 5, 2, 4, -2, 2
    # Peak 5 at t1, trough -2 at t4 → drawdown = 7
    trades = [
        _t("long", 100.0, 105.0, day=1),   # +5
        _t("long", 100.0, 97.0,  day=2),   # -3
        _t("long", 100.0, 102.0, day=3),   # +2
        _t("long", 100.0, 94.0,  day=4),   # -6
        _t("long", 100.0, 104.0, day=5),   # +4
    ]
    m = compute_metrics(trades)
    assert m.max_drawdown_dollars == 7.0


def pytest_approx_2_over_3():
    # Tiny helper rather than pulling pytest.approx
    class _A:
        def __eq__(self, other): return abs(other - 2 / 3) < 1e-9
    return _A()
