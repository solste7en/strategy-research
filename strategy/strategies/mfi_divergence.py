"""MFI Divergence strategy.

Thesis: the Money Flow Index (MFI) is essentially a volume-weighted RSI. When
price pushes to a new session extreme but MFI fails to confirm — i.e. a lower
low in price paired with a HIGHER low in MFI, or a higher high in price paired
with a LOWER high in MFI — the volume behind the move is drying up. That's
the classic institutional-footprint signature: distribution at the top or
accumulation at the bottom. Combined with an overbought / oversold zone
filter, the divergence becomes a mean-reversion trigger.

We scan each session for a bar that:
  (a) sets a new local extreme vs. the prior ``divergence_lookback`` bars,
  (b) whose MFI does NOT confirm the extreme (divergence),
  (c) whose MFI is beyond the oversold/overbought zone.

Entry fills at the next bar's open. Exit fills at the first of:
  * MFI mean-reverts through 50 (primary, "thesis played out")
  * time expiry (``exit_window_minutes``)
  * 15:50 ET EOD hard stop.

MFI reference formula (standard, 14-period default):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    positive_flow  = sum(rmf where typical_price > prev typical_price)  over window
    negative_flow  = sum(rmf where typical_price < prev typical_price)  over window
    money_ratio    = positive_flow / negative_flow
    MFI            = 100 - (100 / (1 + money_ratio))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from strategy.data import SESSION_OPEN
from strategy.strategies.base import Trade

log = logging.getLogger(__name__)

DirectionMode = Literal["symmetric", "long_only", "short_only"]

_EOD_STOP = time(15, 50)


@dataclass(frozen=True)
class MFIDivergenceParams:
    """Parameters for the MFI Divergence strategy.

    Attributes:
        mfi_period: Lookback (in bars) for the Money Flow Index. 14 is the
            textbook default; 21 is a slower / smoother variant. With 5-min
            bars, 14 = 70-min warmup and 21 = 105-min.
        divergence_lookback: Number of prior bars to search for the pivot
            extreme against which we test for divergence. 12 or 24 five-min
            bars = 60 / 120 minutes of context.
        oversold_threshold: MFI must be at or below this for a bullish-
            divergence long signal. 20 = textbook oversold; 25 / 30 are
            looser filters that fire more often.
        overbought_threshold: MFI must be at or above this for a bearish-
            divergence short signal. 80 / 75 / 70.
        exit_window_minutes: Maximum hold time after entry. Exit fires early
            if MFI mean-reverts through 50 first.
        direction_mode: Which side to trade.
            "symmetric"  — both long (bullish divergence) and short (bearish).
            "long_only"  — only trade bullish divergence.
            "short_only" — only trade bearish divergence.
    """

    mfi_period: int
    divergence_lookback: int
    oversold_threshold: int
    overbought_threshold: int
    exit_window_minutes: int
    direction_mode: DirectionMode = "symmetric"

    def as_tag(self) -> str:
        dm = {"symmetric": "sym", "long_only": "long", "short_only": "short"}[self.direction_mode]
        return (
            f"mfi_p{self.mfi_period}"
            f"_lb{self.divergence_lookback}"
            f"_os{self.oversold_threshold}"
            f"_ob{self.overbought_threshold}"
            f"_x{self.exit_window_minutes}"
            f"_{dm}"
        )


def compute_mfi(bars: pd.DataFrame, period: int) -> pd.Series:
    """Compute the standard Money Flow Index over ``bars``.

    Returns a pandas Series aligned to ``bars.index``. The first ``period``
    values are NaN (insufficient warmup).
    """
    if bars.empty:
        return pd.Series(dtype=float, index=bars.index)

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    raw_money_flow = typical * bars["volume"]

    # Classify each bar as positive / negative vs. the prior typical price.
    delta = typical.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0)
    neg_flow = raw_money_flow.where(delta < 0, 0.0)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    # When neg_sum == 0, money_ratio is infinite → MFI = 100.
    # When pos_sum == 0 and neg_sum > 0, MFI = 0.
    # When both are 0 (flat period), MFI is undefined → NaN.
    mfi = pd.Series(index=bars.index, dtype=float)
    both_zero = (pos_sum == 0) & (neg_sum == 0)
    neg_zero = (neg_sum == 0) & (pos_sum > 0)
    normal = ~both_zero & ~neg_zero & pos_sum.notna() & neg_sum.notna()

    money_ratio = pos_sum[normal] / neg_sum[normal]
    mfi.loc[normal] = 100.0 - (100.0 / (1.0 + money_ratio))
    mfi.loc[neg_zero] = 100.0
    mfi.loc[both_zero] = np.nan
    return mfi


class MFIDivergenceStrategy:
    name = "mfi_divergence"

    def __init__(self, params: MFIDivergenceParams):
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

        long_ok = p.direction_mode in ("symmetric", "long_only")
        short_ok = p.direction_mode in ("symmetric", "short_only")

        mfi_series = compute_mfi(day_bars, p.mfi_period)

        # We need at least mfi_period (for MFI warmup) + divergence_lookback
        # bars of prior context before we can scan.
        min_start = p.mfi_period + p.divergence_lookback

        bars_list = list(day_bars.itertuples())
        mfi_values = mfi_series.values

        for i in range(min_start, len(bars_list)):
            bar = bars_list[i]
            ts = bar.Index
            if ts >= eod_stop_ts:
                break

            mfi_now = mfi_values[i]
            if pd.isna(mfi_now):
                continue

            # --- Look for divergence against the prior-window extreme ---
            window_start = i - p.divergence_lookback
            prior_lows = [bars_list[j].low for j in range(window_start, i)]
            prior_highs = [bars_list[j].high for j in range(window_start, i)]

            side: str | None = None
            entry_reason = ""

            # Bullish divergence: new low in price, higher low in MFI, oversold zone.
            if long_ok and mfi_now <= p.oversold_threshold:
                prior_low_min = min(prior_lows)
                prior_low_idx = window_start + prior_lows.index(prior_low_min)
                prior_low_mfi = mfi_values[prior_low_idx]
                cur_low = float(bar.low)

                if (
                    not pd.isna(prior_low_mfi)
                    and cur_low < prior_low_min
                    and mfi_now > prior_low_mfi
                ):
                    side = "long"
                    entry_reason = (
                        f"MFI bullish div: price low {cur_low:.2f} < prior low "
                        f"{prior_low_min:.2f}; MFI {mfi_now:.1f} > prior-low MFI "
                        f"{prior_low_mfi:.1f} (oversold ≤ {p.oversold_threshold})"
                    )

            # Bearish divergence: new high in price, lower high in MFI, overbought zone.
            if side is None and short_ok and mfi_now >= p.overbought_threshold:
                prior_high_max = max(prior_highs)
                prior_high_idx = window_start + prior_highs.index(prior_high_max)
                prior_high_mfi = mfi_values[prior_high_idx]
                cur_high = float(bar.high)

                if (
                    not pd.isna(prior_high_mfi)
                    and cur_high > prior_high_max
                    and mfi_now < prior_high_mfi
                ):
                    side = "short"
                    entry_reason = (
                        f"MFI bearish div: price high {cur_high:.2f} > prior high "
                        f"{prior_high_max:.2f}; MFI {mfi_now:.1f} < prior-high MFI "
                        f"{prior_high_mfi:.1f} (overbought ≥ {p.overbought_threshold})"
                    )

            if side is None:
                continue

            # --- Entry: next bar's open ---
            if i + 1 >= len(bars_list):
                break
            entry_bar = bars_list[i + 1]
            entry_price = float(entry_bar.open)
            entry_time = entry_bar.Index
            if entry_time >= eod_stop_ts:
                break

            # --- Exit: first of MFI mean-reversion (>= 50 for long, <= 50 for short),
            #         time expiry, or EOD hard stop ---
            exit_deadline = entry_time + timedelta(minutes=p.exit_window_minutes)
            exit_price: float | None = None
            exit_time = None
            exit_reason = ""

            for j in range(i + 2, len(bars_list)):
                future_bar = bars_list[j]
                future_ts = future_bar.Index
                future_mfi = mfi_values[j]

                if future_ts >= eod_stop_ts:
                    exit_price = float(future_bar.close)
                    exit_time = future_ts
                    exit_reason = "EOD stop"
                    break

                # Primary exit: MFI mean-reverts through 50
                if not pd.isna(future_mfi):
                    if side == "long" and future_mfi >= 50.0:
                        exit_price = float(future_bar.close)
                        exit_time = future_ts
                        exit_reason = f"MFI mean reversion ({future_mfi:.1f} ≥ 50)"
                        break
                    if side == "short" and future_mfi <= 50.0:
                        exit_price = float(future_bar.close)
                        exit_time = future_ts
                        exit_reason = f"MFI mean reversion ({future_mfi:.1f} ≤ 50)"
                        break

                # Time exit
                if future_ts >= exit_deadline:
                    exit_price = float(future_bar.close)
                    exit_time = future_ts
                    exit_reason = f"time_exit +{p.exit_window_minutes}m"
                    break

            if exit_price is None or exit_time is None:
                last = bars_list[-1]
                exit_price = float(last.close)
                exit_time = last.Index
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
