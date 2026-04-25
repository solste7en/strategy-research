"""Pluggable order executors.

Two implementations share the ``OrderExecutor`` protocol:

    SimulatedExecutor   — applies a slippage model to hypothetical fills, used
                          by the backtest harness. Records every fill.
    LiveSchwabExecutor  — stub wired up to ``services/orders.py`` for live
                          runs. Raises NotImplementedError today — will be
                          filled in after backtest validation.

Keeping the live executor behind the same interface means the live runner is
just a thin scheduler around the same Strategy + RiskManager that the backtest
already validated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from strategy.strategies.base import Side, Trade

log = logging.getLogger(__name__)


@dataclass
class Fill:
    """A single executed leg (entry or exit)."""

    symbol: str
    side: Side  # the side of the *position*, not the action
    action: Literal["open", "close"]
    shares: int
    price: float
    requested_price: float
    slippage_dollars: float
    timestamp: datetime


class OrderExecutor(Protocol):
    def open_position(
        self, symbol: str, side: Side, shares: int, ref_price: float, timestamp: datetime
    ) -> Fill:
        ...

    def close_position(
        self, symbol: str, side: Side, shares: int, ref_price: float, timestamp: datetime
    ) -> Fill:
        ...


# ---------------------------------------------------------------------------
# Simulated executor for backtests
# ---------------------------------------------------------------------------


@dataclass
class SimulatedExecutor:
    """Applies a simple proportional slippage model.

    A long entry pays ``ref_price * (1 + slippage_bps/10000)``.
    A short entry receives ``ref_price * (1 - slippage_bps/10000)``.
    Exits use the opposite sign: long exit sells at
    ``ref_price * (1 - slippage_bps/10000)`` and short cover pays
    ``ref_price * (1 + slippage_bps/10000)``. So the slippage always penalizes
    the trader — matching typical marketable-order experience.

    Default 2bp is a conservative estimate for liquid large caps during RTH.
    Crank it higher (e.g. 5–10bp) when stress-testing the edge on COIN.
    """

    slippage_bps: float = 2.0
    fills: list[Fill] = field(default_factory=list)

    def _slipped(self, ref: float, direction: Literal["pay", "receive"]) -> float:
        mult = 1.0 + self.slippage_bps / 10000.0 if direction == "pay" else 1.0 - self.slippage_bps / 10000.0
        return ref * mult

    def open_position(
        self, symbol: str, side: Side, shares: int, ref_price: float, timestamp: datetime
    ) -> Fill:
        direction = "pay" if side == "long" else "receive"
        fill_price = self._slipped(ref_price, direction)
        slip = (fill_price - ref_price) * shares * (1 if side == "long" else -1)
        f = Fill(
            symbol=symbol, side=side, action="open", shares=shares,
            price=fill_price, requested_price=ref_price,
            slippage_dollars=abs(slip), timestamp=timestamp,
        )
        self.fills.append(f)
        return f

    def close_position(
        self, symbol: str, side: Side, shares: int, ref_price: float, timestamp: datetime
    ) -> Fill:
        # Long close = sell (receive); short close = buy to cover (pay).
        direction = "receive" if side == "long" else "pay"
        fill_price = self._slipped(ref_price, direction)
        slip = (ref_price - fill_price) * shares * (1 if side == "long" else -1)
        f = Fill(
            symbol=symbol, side=side, action="close", shares=shares,
            price=fill_price, requested_price=ref_price,
            slippage_dollars=abs(slip), timestamp=timestamp,
        )
        self.fills.append(f)
        return f


def realize_trade(trade: Trade, executor: OrderExecutor) -> Trade:
    """Run an ideal-price Trade through an executor and return a new Trade
    with slipped entry/exit prices. The harness uses this so backtested P&L
    reflects slippage without the strategy needing to know about it."""
    entry_fill = executor.open_position(
        trade.symbol, trade.side, trade.shares, trade.entry_price, trade.entry_time
    )
    exit_fill = executor.close_position(
        trade.symbol, trade.side, trade.shares, trade.exit_price, trade.exit_time
    )
    return Trade(
        symbol=trade.symbol,
        side=trade.side,
        shares=trade.shares,
        entry_time=trade.entry_time,
        entry_price=entry_fill.price,
        exit_time=trade.exit_time,
        exit_price=exit_fill.price,
        entry_reason=trade.entry_reason,
        exit_reason=trade.exit_reason,
    )


# ---------------------------------------------------------------------------
# Live executor — stub until backtest confirms edge
# ---------------------------------------------------------------------------


class LiveSchwabExecutor:
    """Placeholder. Will call services/orders.py to place live market orders.

    Intentionally unimplemented until the backtest validates the strategy —
    we should not be wiring up live order placement before we have data that
    the edge exists.
    """

    def __init__(self, account_hash: str | None = None):
        self.account_hash = account_hash

    def open_position(self, *args, **kwargs) -> Fill:
        raise NotImplementedError(
            "LiveSchwabExecutor is not wired up yet. Run the backtest first; "
            "we'll implement live order placement once a configuration shows edge."
        )

    def close_position(self, *args, **kwargs) -> Fill:
        raise NotImplementedError(
            "LiveSchwabExecutor is not wired up yet. Run the backtest first; "
            "we'll implement live order placement once a configuration shows edge."
        )
