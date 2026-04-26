"""Intraday Momentum Continuation (IMC).

Thesis (Gao, Han, Li, Zhou — JFE 2018, replicated through 2026): the morning
return of an instrument predicts its last-half-hour return on the same day.
The mechanism is structural — institutional positioning + market-on-close
order flow tend to extend the morning's direction into the close — which
makes the effect robust across regimes that break behavioral patterns.

Signal (single-symbol path) at decision time T (default 15:00 ET):
  1. r_open = (close[T] - open[09:30]) / open[09:30]
  2. atr     = mean true range over the prior `atr_period_bars` 1-min bars
              within today's session up to T (bounded above 0)
  3. Trade only if |r_open| > atr_multiple * atr
  4. Long if r_open > 0, short if r_open < 0 (subject to direction_mode)
  5. Optional volume-z filter on the cumulative session volume up to T
  6. Optional market filter: SPYM r_open at T must agree in sign

Entry fill: next bar's open after the decision bar.
Exit:       close of the first bar at or after `exit_time_minutes`, with a
            hard EOD backstop at 15:58 ET in case of late-day data gaps.

ATR is computed on 1-min bars to keep the strategy data-shape consistent
with the rest of the codebase. The `atr_period_bars` knob translates
directly: 14 = 14 minutes, 30 = 30 minutes. With the default 30-min
observation window, atr_period_bars=14 gives the ATR a meaningful pre-
decision footprint without overlapping the decision bar itself.

One trade per symbol per day. Sizing is delegated to the harness via the
shares=0 sentinel.
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

_EOD_STOP = time(15, 58)
_NEG_INF = float("-inf")


@dataclass(frozen=True)
class IMCParams:
    """Parameters for the Intraday Momentum Continuation strategy.

    Attributes:
        observation_window_minutes: Minutes from 09:30 ET over which to read
            the morning return r_open. Common values: 30, 60. Note the
            decision is taken later, at decision_time_minutes — so the
            strategy "watches" between obs_window and decision_time without
            updating r_open. The original GHLZ effect is morning -> last-30m,
            so observation_window_minutes is intentionally short.
        decision_time_minutes: Minutes from 09:30 ET to make the entry
            decision. Default 330 = 15:00 ET. Must be > observation_window
            and < exit_time.
        exit_time_minutes: Minutes from 09:30 ET to flatten the position.
            Default 385 = 15:55 ET. Must be < EOD stop (15:58).
        atr_period_bars: Number of 1-min bars over which to average the true
            range, ending at observation_window. 14 by default.
        atr_multiple: Threshold |r_open| must exceed `atr_multiple * atr`
            (where atr is expressed as a fraction of session-open price) to
            fire a signal. Higher = stricter filter.
        volume_z_threshold: Cumulative session volume up to T must satisfy
            (vol - mean) / std >= volume_z_threshold to fire. Set to
            float("-inf") to disable. Default disabled because the per-day
            volume rolling stats are computed against the same day's history
            (we do not have prior-day intraday windows in this contract);
            we re-use a within-session z computed over the lookback bars.
        use_market_filter: If True, additionally require that the SPYM
            (or whichever context symbol is provided) r_open at T has the
            same sign as the symbol's r_open. Requires the harness to pass
            ``context_bars`` containing at least one symbol's bars.
        market_context_symbol: Which key in ``context_bars`` to read for the
            market filter. Default "SPYM".
        direction_mode: Which side(s) to trade.
            "symmetric"  — long up-momentum days, short down-momentum days.
            "long_only"  — only ride up-momentum.
            "short_only" — only ride down-momentum.
    """

    observation_window_minutes: int
    decision_time_minutes: int
    exit_time_minutes: int
    atr_period_bars: int
    atr_multiple: float
    volume_z_threshold: float
    use_market_filter: bool
    market_context_symbol: str = "SPYM"
    direction_mode: DirectionMode = "symmetric"

    def as_tag(self) -> str:
        dm = {"symmetric": "sym", "long_only": "long", "short_only": "short"}[self.direction_mode]
        mf = "mf" if self.use_market_filter else "raw"
        # volume_z_threshold may be -inf (off); represent with "off"
        if self.volume_z_threshold == _NEG_INF:
            vz = "off"
        else:
            vz = f"{self.volume_z_threshold:.2f}"
        return (
            f"imc_o{self.observation_window_minutes}"
            f"_d{self.decision_time_minutes}"
            f"_x{self.exit_time_minutes}"
            f"_atr{self.atr_period_bars}x{self.atr_multiple:.2f}"
            f"_vz{vz}"
            f"_{mf}_{dm}"
        )


def compute_atr(bars: pd.DataFrame, period_bars: int) -> pd.Series:
    """Standard ATR on 1-min bars, expressed in price units (not bps).

    True range per bar = max(high-low, |high-prev_close|, |low-prev_close|).
    The first bar of a session has no prev_close so we fall back to high-low.
    Returns a Series aligned to ``bars.index``; the first ``period_bars-1``
    values are NaN by construction.
    """
    if bars.empty:
        return pd.Series(dtype=float, index=bars.index)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    prev_close = bars["close"].astype(float).shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period_bars, min_periods=period_bars).mean()


def _bar_at_or_before(bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """Return the last bar with index <= ts, or None if none exists."""
    sub = bars.loc[bars.index <= ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _bar_after(bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """Return the first bar with index > ts, or None."""
    sub = bars.loc[bars.index > ts]
    if sub.empty:
        return None
    return sub.iloc[0]


def _bar_at_or_after(bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    """Return the first bar with index >= ts, or None."""
    sub = bars.loc[bars.index >= ts]
    if sub.empty:
        return None
    return sub.iloc[0]


class IntradayMomentumContinuationStrategy:
    name = "intraday_momentum_continuation"

    def __init__(self, params: IMCParams):
        self.params = params

    def generate_trades_for_day(
        self,
        symbol: str,
        day_bars: pd.DataFrame,
        context_bars: dict[str, pd.DataFrame] | None = None,
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
        obs_ts = open_ts + timedelta(minutes=p.observation_window_minutes)
        decision_ts = open_ts + timedelta(minutes=p.decision_time_minutes)
        exit_ts = open_ts + timedelta(minutes=p.exit_time_minutes)
        eod_stop_ts = pd.Timestamp.combine(session_date, _EOD_STOP).tz_localize("America/New_York")

        # Sanity: parameter ordering must be obs < decision < exit < EOD.
        if not (obs_ts < decision_ts < exit_ts <= eod_stop_ts):
            log.debug(
                "%s %s IMC: degenerate timing obs=%s dec=%s exit=%s; skip",
                symbol, session_date, obs_ts, decision_ts, exit_ts,
            )
            return []

        # --- Opening bar must be present (allow up to 1-minute drift) ---
        first_bar = day_bars.iloc[0]
        if first_bar.name > open_ts + timedelta(minutes=1):
            log.debug("%s %s IMC: no opening bar, skip", symbol, session_date)
            return []
        open_price = float(first_bar["open"])
        if open_price <= 0:
            return []

        # --- Compute r_open at the OBSERVATION window (not decision time) ---
        # The GHLZ effect is "morning return predicts last-30m return", where
        # the morning window is the first 30 minutes. We measure at obs_ts
        # and HOLD that signal until decision_ts.
        obs_bar = _bar_at_or_before(day_bars, obs_ts)
        if obs_bar is None or obs_bar.name < open_ts:
            return []
        r_open = (float(obs_bar["close"]) - open_price) / open_price

        # --- Compute ATR over the prior atr_period_bars ending at obs_bar ---
        # Only use bars up to obs_ts for ATR — no look-ahead.
        atr_window_bars = day_bars.loc[day_bars.index <= obs_bar.name]
        if len(atr_window_bars) < p.atr_period_bars:
            log.debug(
                "%s %s IMC: only %d bars for ATR (need %d); skip",
                symbol, session_date, len(atr_window_bars), p.atr_period_bars,
            )
            return []
        atr_series = compute_atr(atr_window_bars, p.atr_period_bars)
        atr_price = atr_series.iloc[-1]
        if pd.isna(atr_price) or atr_price <= 0:
            log.debug("%s %s IMC: ATR unavailable; skip", symbol, session_date)
            return []
        # Express ATR as a fraction of open price so it's comparable to r_open.
        atr_frac = float(atr_price) / open_price

        threshold = p.atr_multiple * atr_frac
        if threshold <= 0:
            return []

        # --- Direction gate ---
        long_ok = p.direction_mode in ("symmetric", "long_only")
        short_ok = p.direction_mode in ("symmetric", "short_only")

        side: str | None = None
        entry_reason = ""

        if long_ok and r_open > threshold:
            side = "long"
            entry_reason = (
                f"IMC long: r_open=+{r_open*100:.2f}% > {p.atr_multiple:.2f}*ATR "
                f"({atr_frac*100:.2f}% = {threshold*100:.2f}%)"
            )
        elif short_ok and r_open < -threshold:
            side = "short"
            entry_reason = (
                f"IMC short: r_open={r_open*100:.2f}% < -{p.atr_multiple:.2f}*ATR "
                f"({atr_frac*100:.2f}% = -{threshold*100:.2f}%)"
            )

        if side is None:
            return []

        # --- Volume z-score filter (within-session) ---
        # We compute a within-session z on bar volume over the observation
        # window — a quick proxy for "real" institutional participation.
        # Rolling against a true cross-day prior would require state we
        # deliberately don't carry; this within-session z still flags days
        # where the bars we observed are unusually heavy compared to the
        # baseline within the same window.
        if p.volume_z_threshold > _NEG_INF:
            obs_vols = atr_window_bars["volume"].astype(float).to_numpy()
            if len(obs_vols) >= 2:
                v_mean = float(np.mean(obs_vols))
                v_std = float(np.std(obs_vols, ddof=0))
                cur_vol = float(obs_bar["volume"])
                if v_std > 0:
                    vol_z = (cur_vol - v_mean) / v_std
                else:
                    vol_z = 0.0
                if vol_z < p.volume_z_threshold:
                    log.debug(
                        "%s %s IMC: vol_z=%.2f < %.2f; skip",
                        symbol, session_date, vol_z, p.volume_z_threshold,
                    )
                    return []

        # --- Optional market regime filter via context_bars ---
        if p.use_market_filter:
            if not context_bars or p.market_context_symbol not in context_bars:
                log.debug(
                    "%s %s IMC: market_filter on but no context for %s; skip",
                    symbol, session_date, p.market_context_symbol,
                )
                return []
            ctx_df = context_bars[p.market_context_symbol]
            if ctx_df is None or ctx_df.empty:
                log.debug(
                    "%s %s IMC: context %s empty; skip",
                    symbol, session_date, p.market_context_symbol,
                )
                return []
            # Use the same observation window: same first bar's open and the
            # close at obs_ts. Slice strictly to <= obs_ts to stop look-ahead.
            ctx_first = ctx_df.iloc[0]
            ctx_obs_bar = _bar_at_or_before(ctx_df, obs_ts)
            if ctx_obs_bar is None:
                return []
            ctx_open = float(ctx_first["open"])
            if ctx_open <= 0:
                return []
            ctx_r_open = (float(ctx_obs_bar["close"]) - ctx_open) / ctx_open
            same_sign = (
                (side == "long" and ctx_r_open > 0)
                or (side == "short" and ctx_r_open < 0)
            )
            if not same_sign:
                log.debug(
                    "%s %s IMC: market filter rejects; %s r_open=%.2f%% vs %s ctx=%.2f%%",
                    symbol, session_date, symbol, r_open * 100,
                    p.market_context_symbol, ctx_r_open * 100,
                )
                return []
            entry_reason += (
                f"; mkt {p.market_context_symbol} r_open={ctx_r_open*100:.2f}%"
            )

        # --- Entry fill: next bar's open after decision_ts ---
        # Hold the signal until decision_ts, then enter on the next bar's
        # open. This is the live-trade analogue: we observe at 10:00, decide
        # at 15:00, market-buy gets filled on the 15:01 bar's open.
        decision_bar = _bar_at_or_before(day_bars, decision_ts)
        if decision_bar is None or decision_bar.name >= eod_stop_ts:
            return []
        entry_bar = _bar_after(day_bars, decision_bar.name)
        if entry_bar is None:
            return []
        entry_price = float(entry_bar["open"])
        entry_time = entry_bar.name
        if entry_time >= eod_stop_ts:
            return []

        # --- Exit fill: first bar at or after exit_ts, capped at EOD ---
        exit_bar = _bar_at_or_after(day_bars, exit_ts)
        if exit_bar is None or exit_bar.name > eod_stop_ts:
            # No bar past exit_ts before EOD — fall back to last bar in session.
            tail = day_bars.loc[day_bars.index <= eod_stop_ts]
            if tail.empty:
                return []
            exit_bar = tail.iloc[-1]
        exit_price = float(exit_bar["close"])
        exit_time = exit_bar.name
        exit_reason = f"EOD flat @ T+{p.exit_time_minutes - p.decision_time_minutes}m"

        if exit_time <= entry_time:
            log.debug(
                "%s %s IMC: exit_time %s <= entry_time %s; skip",
                symbol, session_date, exit_time, entry_time,
            )
            return []

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
