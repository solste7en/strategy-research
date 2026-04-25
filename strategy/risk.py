"""Sleeve budget and per-day trade caps.

Encodes the project's hardcoded rules:

* $1000 total sleeve (informational, not a hard stop — we track exposure)
* $100 per-trade target
* ≥1 share always (if 1 share > $100, trade 1 share)
* Max 10 trades/day across the whole universe
* Up to 20-ticker universe (enforced at construction)
* Flat by EOD (enforced by the strategy, not by this class)

Same logic is used by the backtest harness and the live runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


DEFAULT_UNIVERSE = (
    "SPYM", "TQQQ", "TSLA", "NVDA", "COIN",
    "LYFT", "UBER", "HIMS", "RBLX", "HOOD", "PDD", "NFLX",
)
# SPYM = SPDR Portfolio S&P 500 ETF (cheap proxy for SPY, ~$83 vs SPY ~$710)
# TQQQ = ProShares UltraPro QQQ, 3x-leveraged Nasdaq-100 (~$60 vs QQQ ~$620);
#        thresholds should be scaled ~3x QQQ's to match the leveraged move.
# LYFT/UBER = rideshare names, moderate intraday vol (~$15–$85)
# HIMS/RBLX/HOOD = high-beta mid-caps, wider intraday swings
# PDD = China ADR (Temu parent), can gap/spike aggressively on macro news
# NFLX = large-cap tech, tighter moves than small/mid names


@dataclass
class RiskConfig:
    total_capital: float = 1000.0
    per_trade_dollars: float = 100.0
    max_trades_per_day: int = 10
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    max_universe_size: int = 20

    def __post_init__(self):
        if len(self.universe) > self.max_universe_size:
            raise ValueError(
                f"universe exceeds max size {self.max_universe_size}: {self.universe}"
            )


def size_trade(per_trade_dollars: float, reference_price: float) -> float:
    """Return share count for one trade, allowing fractional shares.

    Rule: exactly per_trade_dollars / price shares (fractional OK — most
    brokers support fractional trading for equities). This ensures every trade
    risks exactly $100 of notional regardless of the stock price.
    """
    if reference_price <= 0:
        raise ValueError(f"reference_price must be > 0, got {reference_price}")
    return per_trade_dollars / reference_price


@dataclass
class RiskManager:
    config: RiskConfig = field(default_factory=RiskConfig)
    # Per-day trade counter
    _trades_by_day: dict[date, int] = field(default_factory=dict)

    def can_trade(self, day: date, symbol: str) -> tuple[bool, str]:
        """Return (allowed, reason). Reason is empty string when allowed."""
        if symbol.upper() not in {s.upper() for s in self.config.universe}:
            return False, f"{symbol} not in universe {self.config.universe}"
        used = self._trades_by_day.get(day, 0)
        if used >= self.config.max_trades_per_day:
            return False, (
                f"daily trade cap reached ({used}/{self.config.max_trades_per_day}) for {day}"
            )
        return True, ""

    def register_trade(self, day: date) -> None:
        self._trades_by_day[day] = self._trades_by_day.get(day, 0) + 1

    def size_for(self, reference_price: float) -> int:
        return size_trade(self.config.per_trade_dollars, reference_price)

    def trades_today(self, day: date) -> int:
        return self._trades_by_day.get(day, 0)

    def reset(self) -> None:
        self._trades_by_day.clear()
