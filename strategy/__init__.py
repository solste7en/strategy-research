"""Automated trading strategy framework.

Modules:
    data         — historical and live intraday bar providers
    executor     — pluggable order executors (simulated for backtest, Schwab for live)
    risk         — position sizing and daily/sleeve budget limits
    metrics      — P&L, win rate, Sharpe, drawdown
    backtest     — grid-search backtest harness
    strategies.* — concrete strategy implementations

The same Strategy class is meant to run in both backtest and live modes by
swapping the BarsProvider and OrderExecutor implementations.
"""
