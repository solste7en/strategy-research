"""Volume Surge Momentum strategy.

Thesis: Institutions can't hide large orders — they show up as anomalous volume
spikes. When a bar's volume is significantly higher than the recent baseline AND
price moves meaningfully in one direction on that bar, it signals conviction
from a large participant. We trade WITH that momentum rather than against it.

We compute a rolling N-bar average volume (our "baseline") and look for a bar
where: (1) volume >= multiplier × baseline, and (2) the bar's close-to-open
move is >= min_price_move_bps in one direction. We enter in the direction of
the surge bar, hold for exit_window_minutes, then exit flat.

One trade per symbol per day. All positions flat by EOD.

Entry fill model: next bar's open after the surge bar.
Exit fill model: close of first bar at or after (entry_time + exit_window_minutes),
with a 15:50 ET hard backstop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Literal

import pandas as pd

from strategy.data import SESSION_OPEN
from strategy.strategies.base import Trade

log = logging.getLogger(__name__)

DirectionMode = Literal["symmetric", "long_only", "short_only"]

_EOD_STOP = time(15, 50)


@dataclass(frozen=True)
class VolumeSurgeMomentumParams:
    """Parameters for the Volume Surge Momentum strategy.

    Attributes:
        lookback_bars: Number of prior bars used to compute the volume baseline.
            With 5-min bars: 6 = 30-min baseline, 12 = 60-min baseline.
        volume_multiplier: Surge bar must have volume >= this × the baseline.
            e.g. 3.0 means 3× normal volume.
        min_price_move_bps: The surge bar's (close - open) / open must be at
            least this many basis points in magnitude (direction determines side).
            e.g. 30 = 0.30%.
        exit_window_minutes: Minutes to hold after entry.
        direction_mode: Which direction of surge to trade.
            "symmetric"  — trade both up-surges (long) and down-surges (short).
            "long_only"  — only ride upward surges.
            "short_only" — only ride downward surges.
        min_start_bar: Minimum bar index (from session open) before scanning,
            so we have a meaningful baseline. Defaults to lookback_bars + 2.
    """

    lookback_bars: int
    volume_multiplier: float
    min_price_move_bps: int
    exit_window_minutes: int
    direction_mode: DirectionMode = "symmetric"

    def as_tag(self) -> str:
        dm = {"symmetric": "sym", "long_only": "long", "short_only": "short"}[self.direction_mode]
        return (
            f"vsm_lb{self.lookback_bars}"
            f"_v{self.volume_multiplier:.0f}x"
            f"_m{self.min_price_move_bps}bp"
            f"_x{self.exit_window_minutes}"
            f"_{dm}"
        )


class VolumeSurgeMomentumStrategy:
    name = "volume_surge_momentum"

    def __init__(self, params: VolumeSurgeMomentumParams):
        self.params = params

    def generate_trades_for_day(
        self,
        symbol: str,
        day_bars: pd.DataFrame,
        context_bars: dict[str, pd.DataFrame] | None = None,
    ) -> list[Trade]:
        """Return at most one Trade for this symbol today.

        ``context_bars`` is accepted for harness compatibility but unused.
        """
        del context_bars
        if day_bars.empty:
            return []
        p = self.params

        idx_local = day_bars.index.tz_convert("America/New_York")
        session_dates = {dt.date() for dt in idx_local}
        if len(session_dates) != 1:
            raise ValueError(f"Expected single session date; got {session_dates}")
        session_date = next(iter(session_dates))

        eod_stop_ts = pd.Timestamp.combine(session_date, _EOD_STOP).tz_localize("America/New_York")

        min_price_move = p.min_price_move_bps / 10_000.0
        long_ok = p.direction_mode in ("symmetric", "long_only")
        short_ok = p.direction_mode in ("symmetric", "short_only")

        # We need at least lookback_bars + 1 to have a baseline
        min_start = p.lookback_bars + 1

        bars_list = list(day_bars.itertuples())

        for i in range(min_start, len(bars_list)):
            bar = bars_list[i]
            ts = bar.Index
            if ts >= eod_stop_ts:
                break

            # Compute rolling baseline volume from prior lookback_bars bars
            baseline_bars = [bars_list[j].volume for j in range(i - p.lookback_bars, i)]
            avg_volume = sum(baseline_bars) / len(baseline_bars)

            if avg_volume <= 0:
                continue

            current_volume = float(bar.volume)
            if current_volume < p.volume_multiplier * avg_volume:
                continue  # not a surge bar

            # Check price direction on this surge bar
            bar_open = float(bar.open)
            bar_close = float(bar.close)
            if bar_open <= 0:
                continue
            bar_move = (bar_close - bar_open) / bar_open

            side: str | None = None
            if long_ok and bar_move >= min_price_move:
                side = "long"
                entry_reason = (
                    f"VSM long: vol {current_volume:.0f} = {current_volume/avg_volume:.1f}× avg, "
                    f"bar move +{bar_move*100:.2f}%"
                )
            elif short_ok and bar_move <= -min_price_move:
                side = "short"
                entry_reason = (
                    f"VSM short: vol {current_volume:.0f} = {current_volume/avg_volume:.1f}× avg, "
                    f"bar move {bar_move*100:.2f}%"
                )

            if side is None:
                continue

            # Entry: next bar's open
            if i + 1 >= len(bars_list):
                break
            entry_bar = bars_list[i + 1]
            entry_price = float(entry_bar.open)
            entry_time = entry_bar.Index

            if entry_time >= eod_stop_ts:
                break

            # Exit: first bar at or after (entry + exit_window), capped at EOD
            exit_target = entry_time + timedelta(minutes=p.exit_window_minutes)
            exit_price: float | None = None
            exit_time = None

            for j in range(i + 2, len(bars_list)):
                future_bar = bars_list[j]
                if future_bar.Index >= exit_target or future_bar.Index >= eod_stop_ts:
                    exit_price = float(future_bar.close)
                    exit_time = future_bar.Index
                    break

            if exit_price is None:
                last = bars_list[-1]
                exit_price = float(last.close)
                exit_time = last.Index

            if exit_time <= entry_time:
                break

            return [Trade(
                symbol=symbol,
                side=side,
                shares=0,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                entry_reason=entry_reason,
                exit_reason=f"time_exit +{p.exit_window_minutes}m",
            )]

        return []
