"""Intraday overextension mean-reversion strategy.

Thesis: if a ticker moves more than ``threshold_pct`` (absolute) from the
session open within the first ``entry_window_minutes``, the move tends to
fade, so we take the opposite side and exit in the last
``exit_window_minutes`` of the session.

One trade per symbol per day. All positions flat by EOD per project rules.

Entry fill model: market order at the **next bar's open** after the decision
bar (mimics what a live market-on-touch order would experience).
Exit fill model: market order at the **close price of the first bar at or
after (session_close − exit_window_minutes)**.

Sizing is intentionally *not* the strategy's concern — the harness fills
in share count via the RiskManager before writing the Trade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Literal

import pandas as pd

from strategy.data import SESSION_CLOSE, SESSION_OPEN
from strategy.strategies.base import Trade

log = logging.getLogger(__name__)


DirectionMode = Literal["symmetric", "short_only", "long_only"]


@dataclass(frozen=True)
class IntradayOverextensionParams:
    """Parameters for IntradayOverextensionStrategy.

    Attributes:
        entry_window_minutes: Observation window from 09:30 ET. Entry decision
            happens on the first bar at or after (open + entry_window_minutes).
        threshold_pct: Absolute % change from session open that triggers entry.
            Expressed as a decimal: 0.01 = 1.0%.
        exit_window_minutes: Minutes before 16:00 ET to exit. Exit fills on
            the first bar at or after (close − exit_window_minutes).
        direction_mode: Which side(s) of overextension to trade.
            * "symmetric"  — fade both up and down moves (original theory)
            * "short_only" — only fade up-moves (short when price overshoots up)
            * "long_only"  — only fade down-moves (long when price overshoots down)
            Useful because some tickers exhibit the fade pattern only on one side.
    """

    entry_window_minutes: int
    threshold_pct: float
    exit_window_minutes: int
    direction_mode: DirectionMode = "symmetric"

    def as_tag(self) -> str:
        short_tag = {"symmetric": "sym", "short_only": "short", "long_only": "long"}[self.direction_mode]
        return (
            f"w{self.entry_window_minutes}"
            f"_t{int(round(self.threshold_pct * 10000))}bp"
            f"_x{self.exit_window_minutes}"
            f"_{short_tag}"
        )


class IntradayOverextensionStrategy:
    name = "intraday_overextension"

    def __init__(self, params: IntradayOverextensionParams):
        self.params = params

    # ---- core logic ----

    def generate_trades_for_day(
        self,
        symbol: str,
        day_bars: pd.DataFrame,
        context_bars: dict[str, pd.DataFrame] | None = None,
    ) -> list[Trade]:
        """Return at most one Trade for this symbol today. Empty list = no-trade day.

        ``context_bars`` is accepted for harness compatibility but unused —
        IOE is a single-symbol strategy.
        """
        del context_bars
        if day_bars.empty:
            return []
        p = self.params

        # Defensive: ensure we're looking at a single NY session date.
        idx_local = day_bars.index.tz_convert("America/New_York")
        session_dates = {dt.date() for dt in idx_local}
        if len(session_dates) != 1:
            raise ValueError(
                f"generate_trades_for_day expects a single session; got {session_dates}"
            )
        session_date = next(iter(session_dates))
        open_ts = pd.Timestamp.combine(session_date, SESSION_OPEN).tz_localize("America/New_York")
        decision_ts = open_ts + timedelta(minutes=p.entry_window_minutes)
        exit_deadline_ts = (
            pd.Timestamp.combine(session_date, SESSION_CLOSE).tz_localize("America/New_York")
            - timedelta(minutes=p.exit_window_minutes)
        )

        # Pre-decision window must include the opening bar (09:30) AND the decision bar.
        obs_window = day_bars.loc[day_bars.index <= decision_ts]
        if obs_window.empty or obs_window.index[0] > open_ts + timedelta(minutes=1):
            # Either no opening bar available or market opened late; skip.
            log.debug("%s %s: no opening bar, skip", symbol, session_date)
            return []

        open_price = float(obs_window.iloc[0]["open"])
        # Price the decision is made on = close of the decision bar (i.e., last bar ≤ decision_ts)
        decision_bar = obs_window.iloc[-1]
        decision_price = float(decision_bar["close"])
        move = (decision_price - open_price) / open_price

        # Decide direction. `shares=0` sentinel — harness fills in from RiskManager.
        short_ok = p.direction_mode in ("symmetric", "short_only")
        long_ok = p.direction_mode in ("symmetric", "long_only")
        if short_ok and move > p.threshold_pct:
            side = "short"
            entry_reason = (
                f"up {move*100:.2f}% from open by {p.entry_window_minutes}m (> {p.threshold_pct*100:.2f}%)"
            )
        elif long_ok and move < -p.threshold_pct:
            side = "long"
            entry_reason = (
                f"down {move*100:.2f}% from open by {p.entry_window_minutes}m (< -{p.threshold_pct*100:.2f}%)"
            )
        else:
            return []

        # Entry fill: next bar's open after the decision bar.
        post = day_bars.loc[day_bars.index > decision_bar.name]
        if post.empty:
            log.debug("%s %s: no bars after decision, skip", symbol, session_date)
            return []
        entry_bar = post.iloc[0]
        entry_price = float(entry_bar["open"])
        entry_time = entry_bar.name

        # Exit fill: first bar at or after exit deadline.
        exit_region = day_bars.loc[day_bars.index >= exit_deadline_ts]
        if exit_region.empty:
            # No bar past deadline (short day?) — exit on last available bar.
            exit_region = day_bars.tail(1)
        exit_bar = exit_region.iloc[0]
        exit_price = float(exit_bar["close"])
        exit_time = exit_bar.name
        exit_reason = f"EOD flat @ close − {p.exit_window_minutes}m"

        if exit_time <= entry_time:
            # Degenerate case: entry window pushed past exit window. Config-level bug.
            log.warning(
                "%s %s: exit_time %s <= entry_time %s (params=%s); skipping",
                symbol, session_date, exit_time, entry_time, p,
            )
            return []

        trade = Trade(
            symbol=symbol,
            side=side,
            shares=0,  # harness fills in
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
        )
        return [trade]
