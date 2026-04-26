"""Strategy base types.

A Strategy consumes intraday bars and emits entry / exit Signals. The same
Strategy is meant to run in the backtest harness (fed historical bars) and the
live runner (fed Schwab realtime bars), so it must be stateless w.r.t. I/O.
Per-day, per-symbol state is passed in explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

import pandas as pd


Side = Literal["long", "short"]
Action = Literal["enter_long", "enter_short", "exit"]


@dataclass(frozen=True)
class Signal:
    """A decision emitted by a strategy for one symbol at one point in time."""

    action: Action
    symbol: str
    # Reference price the strategy observed. Executor decides the actual fill
    # (e.g. next bar's open for market orders).
    reference_price: float
    # Human-readable reason, logged for audit.
    reason: str


@dataclass
class Trade:
    """A completed round-trip trade (entry + exit).

    Used by the backtest harness to report P&L and by the live runner to persist
    to the strategy_trades table. A Trade is always flat by EOD per the
    project's `flat-by-EOD` rule.
    """

    symbol: str
    side: Side
    shares: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    entry_reason: str
    exit_reason: str

    @property
    def notional(self) -> float:
        return self.shares * self.entry_price

    @property
    def pnl_dollars(self) -> float:
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.shares
        return (self.entry_price - self.exit_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        """P&L as a fraction of notional entered. Positive = win."""
        if self.notional == 0:
            return 0.0
        return self.pnl_dollars / self.notional

    @property
    def won(self) -> bool:
        return self.pnl_dollars > 0


class Strategy(Protocol):
    """A strategy generates at most one round-trip trade per symbol per day.

    Implementations must not do I/O. They are handed a DataFrame of regular-
    session bars for one symbol on one day and return the Trade list for that
    day (empty if no signal fired).

    ``context_bars`` is an optional dict ``{symbol -> day_bars}`` for cross-
    asset context (e.g. SPYM as a market-regime filter). The harness slices
    each context symbol's bars to the same trading session as ``day_bars``
    before calling. Strategies that don't need cross-asset context simply
    ignore the kwarg, which keeps single-symbol implementations unchanged.
    """

    name: str

    def generate_trades_for_day(
        self,
        symbol: str,
        day_bars: pd.DataFrame,
        context_bars: dict[str, pd.DataFrame] | None = None,
    ) -> list[Trade]:
        """Return 0 or 1 Trade(s) for ``symbol`` on the day represented by
        ``day_bars`` (a chronologically sorted, regular-session DataFrame
        indexed by timezone-aware datetime with columns open/high/low/close/volume).

        A ``shares`` value of ``0`` is a sentinel — the harness fills in the
        correct size from the RiskManager before persisting the Trade. The
        strategy itself should not know about sizing.

        ``context_bars`` carries per-context-symbol bars sliced to the same
        session date. ``None`` (default) means no context was requested, which
        is the normal case for single-symbol strategies.
        """
        ...
