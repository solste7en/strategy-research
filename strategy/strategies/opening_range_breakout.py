"""Opening Range Breakout (ORB) strategy.

Thesis: The high and low printed in the first N minutes of the session form a
support/resistance "map" for the rest of the day. A decisive close above the
range high signals continuation momentum (buy). A decisive close below the range
low signals downside continuation (short). We ride the move for a fixed window
then flatten, always closing well before the session end.

One trade per symbol per day. All positions flat by EOD.

Entry fill model: next bar's open after the breakout-confirmation bar.
Exit fill model: close of the first bar at or after (entry_time + exit_window_minutes),
with a hard backstop at 15:45 ET.
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

DirectionMode = Literal["symmetric", "long_only", "short_only"]

# Hard EOD stop: never hold past this to avoid accidental overnight.
_EOD_STOP = time(15, 45)


@dataclass(frozen=True)
class ORBParams:
    """Parameters for the Opening Range Breakout strategy.

    Attributes:
        range_window_minutes: Duration of the opening range (from 09:30 ET).
            Common values: 15, 30, 60.
        breakout_buffer_bps: Extra distance beyond the range edge required for
            a confirmed breakout, in basis points. Prevents whipsaw entries on
            a bare touch of the range boundary.
        exit_window_minutes: Minutes to hold after entry before exiting.
        direction_mode: Which breakout direction(s) to trade.
            "symmetric"  — take both long (upside break) and short (downside break).
            "long_only"  — only trade upside breakouts.
            "short_only" — only trade downside breakouts.
    """

    range_window_minutes: int
    breakout_buffer_bps: int          # e.g. 20 = 0.20% beyond the range edge
    exit_window_minutes: int
    direction_mode: DirectionMode = "symmetric"

    def as_tag(self) -> str:
        dm = {"symmetric": "sym", "long_only": "long", "short_only": "short"}[self.direction_mode]
        return (
            f"orb_r{self.range_window_minutes}"
            f"_b{self.breakout_buffer_bps}bp"
            f"_x{self.exit_window_minutes}"
            f"_{dm}"
        )


class OpeningRangeBreakoutStrategy:
    name = "opening_range_breakout"

    def __init__(self, params: ORBParams):
        self.params = params

    def generate_trades_for_day(
        self, symbol: str, day_bars: pd.DataFrame
    ) -> list[Trade]:
        """Return at most one Trade for this symbol today."""
        if day_bars.empty:
            return []
        p = self.params

        # --- Resolve session timestamps ---
        idx_local = day_bars.index.tz_convert("America/New_York")
        session_dates = {dt.date() for dt in idx_local}
        if len(session_dates) != 1:
            raise ValueError(f"Expected single session date; got {session_dates}")
        session_date = next(iter(session_dates))

        open_ts = pd.Timestamp.combine(session_date, SESSION_OPEN).tz_localize("America/New_York")
        range_end_ts = open_ts + timedelta(minutes=p.range_window_minutes)
        eod_stop_ts = pd.Timestamp.combine(session_date, _EOD_STOP).tz_localize("America/New_York")

        # --- Build opening range ---
        range_bars = day_bars.loc[day_bars.index <= range_end_ts]
        if range_bars.empty or range_bars.index[0] > open_ts + timedelta(minutes=5):
            log.debug("%s %s: no opening bar for ORB, skip", symbol, session_date)
            return []

        or_high = float(range_bars["high"].max())
        or_low = float(range_bars["low"].min())
        buffer = p.breakout_buffer_bps / 10_000.0
        buy_trigger = or_high * (1.0 + buffer)
        sell_trigger = or_low * (1.0 - buffer)

        long_ok = p.direction_mode in ("symmetric", "long_only")
        short_ok = p.direction_mode in ("symmetric", "short_only")

        # --- Scan post-range bars for first breakout ---
        post_bars = day_bars.loc[day_bars.index > range_end_ts]

        for idx in range(len(post_bars)):
            bar = post_bars.iloc[idx]
            ts = bar.name

            if ts >= eod_stop_ts:
                break  # too late to safely enter and exit same day

            close = float(bar["close"])
            side: str | None = None

            if long_ok and close > buy_trigger:
                side = "long"
                entry_reason = (
                    f"ORB long: close {close:.2f} > range_high {or_high:.2f} + {p.breakout_buffer_bps}bp buffer"
                )
            elif short_ok and close < sell_trigger:
                side = "short"
                entry_reason = (
                    f"ORB short: close {close:.2f} < range_low {or_low:.2f} - {p.breakout_buffer_bps}bp buffer"
                )

            if side is None:
                continue

            # Entry: next bar's open
            remaining = post_bars.iloc[idx + 1:]
            if remaining.empty:
                log.debug("%s %s ORB: no bar after breakout bar, skip", symbol, session_date)
                break

            entry_bar = remaining.iloc[0]
            entry_price = float(entry_bar["open"])
            entry_time = entry_bar.name

            # Exit: first bar at or after (entry + exit_window), capped at EOD stop
            exit_target = entry_time + timedelta(minutes=p.exit_window_minutes)
            exit_candidates = day_bars.loc[day_bars.index >= exit_target]
            if exit_candidates.empty or exit_candidates.index[0] > eod_stop_ts:
                exit_candidates = day_bars.loc[day_bars.index <= eod_stop_ts].tail(1)
            if exit_candidates.empty:
                exit_candidates = day_bars.tail(1)

            exit_bar = exit_candidates.iloc[0]
            exit_price = float(exit_bar["close"])
            exit_time = exit_bar.name

            if exit_time <= entry_time:
                log.debug("%s %s ORB: exit <= entry after breakout, skip", symbol, session_date)
                break

            return [Trade(
                symbol=symbol,
                side=side,
                shares=0,   # harness fills in via RiskManager
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                entry_reason=entry_reason,
                exit_reason=f"time_exit +{p.exit_window_minutes}m",
            )]

        return []
