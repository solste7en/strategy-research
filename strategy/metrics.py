"""P&L and risk-adjusted metrics over a list of completed Trades."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, pstdev

from strategy.strategies.base import Trade


@dataclass
class TradeMetrics:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float              # 0–1, fraction of trades with positive P&L
    avg_pnl_dollars: float
    avg_pnl_pct: float           # per-trade % of notional
    total_pnl_dollars: float
    best_trade_dollars: float
    worst_trade_dollars: float
    sharpe_daily: float          # naive: daily-return mean / stdev × sqrt(252)
    max_drawdown_dollars: float  # peak-to-trough of cumulative P&L curve

    def as_row(self) -> dict:
        return asdict(self)


def compute_metrics(trades: list[Trade]) -> TradeMetrics:
    if not trades:
        return TradeMetrics(
            n_trades=0, n_wins=0, n_losses=0, win_rate=0.0,
            avg_pnl_dollars=0.0, avg_pnl_pct=0.0, total_pnl_dollars=0.0,
            best_trade_dollars=0.0, worst_trade_dollars=0.0,
            sharpe_daily=0.0, max_drawdown_dollars=0.0,
        )

    pnls = [t.pnl_dollars for t in trades]
    pnls_pct = [t.pnl_pct for t in trades]
    n_wins = sum(1 for p in pnls if p > 0)
    n_losses = sum(1 for p in pnls if p < 0)
    total_pnl = sum(pnls)

    # Drawdown over the chronological cumulative P&L curve.
    ordered = sorted(trades, key=lambda t: t.exit_time)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in ordered:
        cum += t.pnl_dollars
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Sharpe: bucket by trading day, compute daily P&L / avg capital allocated
    # that day (or just daily $ P&L if we want the simplest Sharpe).
    daily_pnl: dict = {}
    for t in ordered:
        d = t.exit_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t.pnl_dollars
    daily_values = list(daily_pnl.values())
    if len(daily_values) >= 2 and pstdev(daily_values) > 0:
        sharpe = (mean(daily_values) / pstdev(daily_values)) * math.sqrt(252)
    else:
        sharpe = 0.0

    return TradeMetrics(
        n_trades=len(trades),
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=n_wins / len(trades),
        avg_pnl_dollars=mean(pnls),
        avg_pnl_pct=mean(pnls_pct),
        total_pnl_dollars=total_pnl,
        best_trade_dollars=max(pnls),
        worst_trade_dollars=min(pnls),
        sharpe_daily=sharpe,
        max_drawdown_dollars=max_dd,
    )
