"""VWAP Mean Reversion strategy.

Thesis: The Volume-Weighted Average Price (VWAP) represents the "fair value"
price that institutional desks benchmark against throughout the day. When price
drifts far above VWAP, sell-side pressure tends to push it back. When it drops
far below, buy-side demand tends to support a rebound.

We wait for VWAP to "stabilize" (at least entry_start_minutes into the session)
then fade extreme deviations: short when price > VWAP × (1 + deviation_pct),
long when price < VWAP × (1 - deviation_pct). Exit when price reverts to VWAP
or after exit_window_minutes, whichever comes first, with a hard EOD stop.

One trade per symbol per day. All positions flat by EOD.

Entry fill model: next bar's open after the deviation-trigger bar.
Exit fill model: earliest of (a) price touches VWAP, (b) time window expires,
(c) 15:50 ET hard stop.
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
class VWAPReversionParams:
    """Parameters for the VWAP Mean Reversion strategy.

    Attributes:
        entry_start_minutes: Minutes from open before we start scanning for
            signals. Allows VWAP to stabilize before we trade it.
            Common values: 30, 60.
        deviation_bps: How far price must be from VWAP (in basis points) to
            trigger an entry. e.g. 100 = 1.0%.
        exit_window_minutes: Max time to hold. We also exit early if price
            reverts to within revert_band_bps of VWAP.
        revert_band_bps: Within this many bps of VWAP we consider "reverted"
            and exit early. Default 10 bps (≈ transaction cost threshold).
        direction_mode: Which side of deviation to trade.
            "symmetric"  — both long (below VWAP) and short (above VWAP).
            "long_only"  — only buy dips below VWAP.
            "short_only" — only short rips above VWAP.
    """

    entry_start_minutes: int
    deviation_bps: int
    exit_window_minutes: int
    revert_band_bps: int = 10
    direction_mode: DirectionMode = "symmetric"

    def as_tag(self) -> str:
        dm = {"symmetric": "sym", "long_only": "long", "short_only": "short"}[self.direction_mode]
        return (
            f"vwap_s{self.entry_start_minutes}"
            f"_d{self.deviation_bps}bp"
            f"_x{self.exit_window_minutes}"
            f"_{dm}"
        )


class VWAPReversionStrategy:
    name = "vwap_reversion"

    def __init__(self, params: VWAPReversionParams):
        self.params = params

    @staticmethod
    def _running_vwap(bars: pd.DataFrame) -> pd.Series:
        """Compute bar-by-bar VWAP from session open."""
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        cumulative_tp_vol = (typical * bars["volume"]).cumsum()
        cumulative_vol = bars["volume"].cumsum()
        # Guard against zero-volume prefixes (e.g. pre-open stubs)
        vwap = cumulative_tp_vol / cumulative_vol.replace(0, float("nan"))
        return vwap

    def generate_trades_for_day(
        self, symbol: str, day_bars: pd.DataFrame
    ) -> list[Trade]:
        """Return at most one Trade for this symbol today."""
        if day_bars.empty:
            return []
        p = self.params

        idx_local = day_bars.index.tz_convert("America/New_York")
        session_dates = {dt.date() for dt in idx_local}
        if len(session_dates) != 1:
            raise ValueError(f"Expected single session date; got {session_dates}")
        session_date = next(iter(session_dates))

        open_ts = pd.Timestamp.combine(session_date, SESSION_OPEN).tz_localize("America/New_York")
        entry_start_ts = open_ts + timedelta(minutes=p.entry_start_minutes)
        eod_stop_ts = pd.Timestamp.combine(session_date, _EOD_STOP).tz_localize("America/New_York")

        # Compute VWAP across all day bars (running from first bar forward)
        vwap_series = self._running_vwap(day_bars)

        dev = p.deviation_bps / 10_000.0
        revert_band = p.revert_band_bps / 10_000.0

        long_ok = p.direction_mode in ("symmetric", "long_only")
        short_ok = p.direction_mode in ("symmetric", "short_only")

        # --- Scan bars from entry_start onward ---
        scan_bars = day_bars.loc[day_bars.index >= entry_start_ts]

        for idx in range(len(scan_bars)):
            bar = scan_bars.iloc[idx]
            ts = bar.name
            if ts >= eod_stop_ts:
                break

            vwap = float(vwap_series.loc[ts])
            if pd.isna(vwap) or vwap <= 0:
                continue

            close = float(bar["close"])
            deviation = (close - vwap) / vwap

            side: str | None = None
            if short_ok and deviation > dev:
                side = "short"
                entry_reason = (
                    f"VWAP short: price {close:.2f} is +{deviation*100:.2f}% "
                    f"above VWAP {vwap:.2f} (threshold {p.deviation_bps}bp)"
                )
            elif long_ok and deviation < -dev:
                side = "long"
                entry_reason = (
                    f"VWAP long: price {close:.2f} is {deviation*100:.2f}% "
                    f"below VWAP {vwap:.2f} (threshold -{p.deviation_bps}bp)"
                )

            if side is None:
                continue

            # Entry: next bar's open
            remaining_scan = scan_bars.iloc[idx + 1:]
            all_remaining = day_bars.loc[day_bars.index > ts]
            if all_remaining.empty:
                break

            entry_bar = all_remaining.iloc[0]
            entry_price = float(entry_bar["open"])
            entry_time = entry_bar.name

            # Exit: scan forward for early VWAP reversion or time expiry
            exit_deadline = entry_time + timedelta(minutes=p.exit_window_minutes)
            post_entry = day_bars.loc[day_bars.index > entry_time]
            exit_price: float | None = None
            exit_time = None
            exit_reason = ""

            for _, future_bar in post_entry.iterrows():
                if future_bar.name > eod_stop_ts:
                    # EOD hard stop
                    exit_price = float(future_bar["close"])
                    exit_time = future_bar.name
                    exit_reason = "EOD stop"
                    break
                future_vwap = float(vwap_series.loc[future_bar.name]) if future_bar.name in vwap_series.index else float("nan")
                future_close = float(future_bar["close"])

                # Early exit: price has reverted to within revert_band of VWAP
                if not pd.isna(future_vwap) and future_vwap > 0:
                    future_dev = (future_close - future_vwap) / future_vwap
                    if side == "short" and future_dev <= revert_band:
                        exit_price = future_close
                        exit_time = future_bar.name
                        exit_reason = f"VWAP reversion (dev={future_dev*100:.2f}%)"
                        break
                    elif side == "long" and future_dev >= -revert_band:
                        exit_price = future_close
                        exit_time = future_bar.name
                        exit_reason = f"VWAP reversion (dev={future_dev*100:.2f}%)"
                        break

                # Time exit
                if future_bar.name >= exit_deadline:
                    exit_price = float(future_bar["close"])
                    exit_time = future_bar.name
                    exit_reason = f"time_exit +{p.exit_window_minutes}m"
                    break

            if exit_price is None or exit_time is None:
                # Fell through — use last bar of day
                last_bar = day_bars.iloc[-1]
                exit_price = float(last_bar["close"])
                exit_time = last_bar.name
                exit_reason = "EOD last bar"

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
                exit_reason=exit_reason,
            )]

        return []
