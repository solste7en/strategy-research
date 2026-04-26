"""Backtest harness — grid-search an IntradayOverextensionStrategy over a
historical window of cached bars and report per-ticker, per-config metrics.

Usage (from code):

    from strategy.backtest import BacktestRunner, build_default_grid
    from strategy.data import ParquetBarsProvider
    from strategy.risk import RiskConfig

    runner = BacktestRunner(
        bars_provider=ParquetBarsProvider(),
        universe=("SPY", "QQQ", "TSLA", "NVDA", "COIN"),
        start=date(2026, 1, 1),
        end=date(2026, 4, 22),
        risk_config=RiskConfig(),
    )
    results_df = runner.run(build_default_grid())

For the command-line interface see ``scripts/run_backtest.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

from strategy.data import BarsProvider, split_by_session_date
from strategy.executor import SimulatedExecutor, realize_trade
from strategy.metrics import compute_metrics
from strategy.risk import RiskConfig, RiskManager
from strategy.strategies.base import Trade
from strategy.strategies.intraday_overextension import (
    IntradayOverextensionParams,
    IntradayOverextensionStrategy,
)
from strategy.strategies.opening_range_breakout import (
    ORBParams,
    OpeningRangeBreakoutStrategy,
)
from strategy.strategies.vwap_reversion import (
    VWAPReversionParams,
    VWAPReversionStrategy,
)
from strategy.strategies.volume_surge_momentum import (
    VolumeSurgeMomentumParams,
    VolumeSurgeMomentumStrategy,
)
from strategy.strategies.mfi_divergence import (
    MFIDivergenceParams,
    MFIDivergenceStrategy,
)
from strategy.strategies.intraday_momentum_continuation import (
    IMCParams,
    IntradayMomentumContinuationStrategy,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------


def build_default_grid(
    direction_modes: tuple[str, ...] = ("symmetric", "short_only", "long_only"),
) -> list[IntradayOverextensionParams]:
    """Default universe-wide grid. Used only when no per-ticker grid is given.

    3 entry windows × 4 thresholds × 3 exit windows × 3 direction modes = 108 combos.
    """
    entry_windows = [15, 30, 60]
    thresholds = [0.005, 0.010, 0.015, 0.020]
    exit_windows = [5, 15, 30]
    return [
        IntradayOverextensionParams(
            entry_window_minutes=ew,
            threshold_pct=th,
            exit_window_minutes=xw,
            direction_mode=dm,
        )
        for ew, th, xw, dm in product(entry_windows, thresholds, exit_windows, direction_modes)
    ]


# Per-ticker threshold ranges, chosen by character:
#   - SPYM is an S&P proxy with typical first-30-min moves of 10-50 bps → tight grid
#   - TQQQ is 3x leveraged QQQ → thresholds ~3x an index ETF's
#   - TSLA / NVDA: high-beta single names, moderate overextension happens often
#   - COIN: highest volatility, widest thresholds
PER_TICKER_THRESHOLDS_BPS: dict[str, tuple[int, ...]] = {
    # --- Original universe ---
    "SPYM": (10, 25, 50, 75),        # S&P proxy ETF, tight intraday moves
    "TQQQ": (30, 75, 150, 225),      # 3x Nasdaq ETF, thresholds ~3x index
    "TSLA": (50, 100, 150, 200),     # high-beta single name
    "NVDA": (50, 100, 150, 200),     # high-beta single name
    "COIN": (100, 150, 200, 250, 300),  # highest vol — crypto-correlated

    # --- Expanded universe ---
    # Rideshare duopoly: moderate intraday vol, thinner than TSLA/NVDA
    "LYFT": (50, 100, 150, 200),
    "UBER": (50,  75, 125, 175),     # slightly larger cap, tighter swings

    # High-beta mid-caps: health/beauty, gaming, fintech
    "HIMS": (75, 125, 200, 275),     # small-cap health; sharp gap-and-run days
    "RBLX": (75, 125, 175, 250),     # gaming/metaverse; volatile on user data
    "HOOD": (75, 125, 200, 275),     # retail-fintech; vol spikes on meme days

    # China ADR: wide bands for macro/regulatory gap risk
    "PDD":  (100, 175, 250, 350),

    # Large-cap streaming: narrower moves than small/mid names
    "NFLX": (50,  75, 100, 150),
}
ENTRY_WINDOWS_DEFAULT = (15, 30, 60)
EXIT_WINDOWS_DEFAULT = (5, 15, 30)
DIRECTION_MODES_DEFAULT = ("symmetric", "short_only", "long_only")


def build_per_ticker_grids(
    universe: tuple[str, ...],
    thresholds_bps_by_ticker: dict[str, tuple[int, ...]] | None = None,
    entry_windows: tuple[int, ...] = ENTRY_WINDOWS_DEFAULT,
    exit_windows: tuple[int, ...] = EXIT_WINDOWS_DEFAULT,
    direction_modes: tuple[str, ...] = DIRECTION_MODES_DEFAULT,
) -> dict[str, list[IntradayOverextensionParams]]:
    """Build per-ticker parameter grids.

    Returns a dict {symbol -> list[params]}. Falls back to the 50/100/150/200bp
    set for any symbol missing from ``thresholds_bps_by_ticker``.
    """
    thresholds_bps_by_ticker = thresholds_bps_by_ticker or PER_TICKER_THRESHOLDS_BPS
    grids: dict[str, list[IntradayOverextensionParams]] = {}
    for sym in universe:
        bps = thresholds_bps_by_ticker.get(sym, (50, 100, 150, 200))
        thresholds = [b / 10000.0 for b in bps]
        grids[sym] = [
            IntradayOverextensionParams(
                entry_window_minutes=ew,
                threshold_pct=th,
                exit_window_minutes=xw,
                direction_mode=dm,
            )
            for ew, th, xw, dm in product(entry_windows, thresholds, exit_windows, direction_modes)
        ]
    return grids


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Raw trades and aggregated metrics for one (params × symbol) cell."""

    params: IntradayOverextensionParams
    symbol: str
    trades: list[Trade]

    def as_summary_row(self) -> dict:
        m = compute_metrics(self.trades)
        row: dict = {
            "symbol": self.symbol,
            "params": self.params.as_tag(),
            **m.as_row(),
        }
        # Include strategy-specific breakdown columns when present.
        # These are optional so the same harness works for all strategy types.
        for attr, col in [
            ("entry_window_minutes", "entry_window_min"),
            ("threshold_pct",        "threshold_pct"),
            ("exit_window_minutes",  "exit_window_min"),
            ("direction_mode",       "direction_mode"),
        ]:
            val = getattr(self.params, attr, None)
            if val is not None:
                row[col] = val
        return row


@dataclass
class BacktestRunner:
    bars_provider: BarsProvider
    universe: tuple[str, ...]
    start: date
    end: date
    risk_config: RiskConfig = None  # defaults filled at __post_init__
    slippage_bps: float = 2.0
    # Optional cross-asset context (e.g. SPYM as a market-regime filter for
    # IMC). Symbols here are loaded once and sliced per day before being
    # passed into ``generate_trades_for_day(... context_bars=)``. These are
    # NOT part of the trading universe and do not consume risk budget.
    context_symbols: tuple[str, ...] = ()

    def __post_init__(self):
        if self.risk_config is None:
            self.risk_config = RiskConfig(universe=self.universe)

    def run(
        self,
        grid: Iterable | dict[str, list],
        strategy_factory=None,
    ) -> pd.DataFrame:
        """Execute the backtest. Returns a summary DataFrame.

        ``grid`` accepts either:
          * a flat iterable — applied to every symbol (universe-wide grid), or
          * a dict keyed by symbol → list[params] — per-ticker grids, so each
            symbol can sweep different thresholds.

        ``strategy_factory`` is a callable that accepts a params object and
        returns a strategy instance. Defaults to IntradayOverextensionStrategy
        for backward compatibility.

        Per-trade detail is attached via the ``_trades_by_cell`` attr (dict
        keyed by (params_tag, symbol) → list[Trade]) so callers that want to
        inspect individual trades can drill in without rerunning.
        """
        if strategy_factory is None:
            strategy_factory = IntradayOverextensionStrategy

        # Normalize grid shape → per-ticker dict.
        if isinstance(grid, dict):
            per_ticker_grid: dict[str, list] = {
                s: list(grid[s]) for s in self.universe if s in grid
            }
            missing = [s for s in self.universe if s not in grid]
            if missing:
                raise ValueError(f"grid dict missing entries for symbols: {missing}")
        else:
            flat = list(grid)
            per_ticker_grid = {s: list(flat) for s in self.universe}

        self._trades_by_cell: dict[tuple[str, str], list[Trade]] = {}
        rows: list[dict] = []

        # Preload bars per symbol once; we slice by day inside the loop.
        by_symbol_by_day: dict[str, dict[date, pd.DataFrame]] = {}
        for sym in self.universe:
            full = self.bars_provider.get_bars(sym, self.start, self.end)
            by_symbol_by_day[sym] = split_by_session_date(full)
            log.info(
                "loaded %s: %d days between %s and %s",
                sym, len(by_symbol_by_day[sym]), self.start, self.end,
            )

        # Preload context symbols (not part of the trading universe). Skip
        # any context symbol that is already in the trading universe to
        # avoid redundant I/O.
        context_by_day: dict[str, dict[date, pd.DataFrame]] = {}
        for ctx_sym in self.context_symbols:
            if ctx_sym in by_symbol_by_day:
                context_by_day[ctx_sym] = by_symbol_by_day[ctx_sym]
                continue
            full_ctx = self.bars_provider.get_bars(ctx_sym, self.start, self.end)
            context_by_day[ctx_sym] = split_by_session_date(full_ctx)
            log.info(
                "loaded context %s: %d days between %s and %s",
                ctx_sym, len(context_by_day[ctx_sym]), self.start, self.end,
            )

        total_cells = sum(len(per_ticker_grid[s]) for s in self.universe)
        log.info(
            "running per-ticker grids: %d total cells across %d symbols",
            total_cells, len(self.universe),
        )

        for sym in self.universe:
            for params in per_ticker_grid[sym]:
                strat = strategy_factory(params)
                executor = SimulatedExecutor(slippage_bps=self.slippage_bps)
                risk = RiskManager(config=self.risk_config)
                realized_trades: list[Trade] = []

                for d, day_bars in sorted(by_symbol_by_day[sym].items()):
                    allowed, reason = risk.can_trade(d, sym)
                    if not allowed:
                        log.debug("skip %s %s: %s", sym, d, reason)
                        continue

                    if context_by_day:
                        ctx_for_day: dict[str, pd.DataFrame] | None = {
                            cs: context_by_day[cs].get(d, pd.DataFrame())
                            for cs in self.context_symbols
                        }
                    else:
                        ctx_for_day = None

                    candidate_trades = strat.generate_trades_for_day(
                        sym, day_bars, context_bars=ctx_for_day,
                    )
                    for ct in candidate_trades:
                        shares = risk.size_for(ct.entry_price)
                        sized = Trade(
                            symbol=ct.symbol, side=ct.side, shares=shares,
                            entry_time=ct.entry_time, entry_price=ct.entry_price,
                            exit_time=ct.exit_time, exit_price=ct.exit_price,
                            entry_reason=ct.entry_reason, exit_reason=ct.exit_reason,
                        )
                        filled = realize_trade(sized, executor)
                        realized_trades.append(filled)
                        risk.register_trade(d)

                cell_key = (params.as_tag(), sym)
                self._trades_by_cell[cell_key] = realized_trades
                rows.append(
                    BacktestResult(params=params, symbol=sym, trades=realized_trades).as_summary_row()
                )

        df = pd.DataFrame(rows)
        # Stable ordering: symbol then best total P&L first
        if not df.empty:
            df = df.sort_values(
                ["symbol", "total_pnl_dollars"], ascending=[True, False]
            ).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def top_configs_per_symbol(
    results: pd.DataFrame, n: int = 3, min_trades: int = 10
) -> pd.DataFrame:
    """Return the top-n configs per symbol by total P&L, filtering out
    configs with too-few trades (statistically noisy)."""
    filtered = results[results["n_trades"] >= min_trades]
    return (
        filtered.sort_values(["symbol", "total_pnl_dollars"], ascending=[True, False])
        .groupby("symbol", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def write_report(
    results: pd.DataFrame,
    out_dir: Path,
    tag: str,
    start: date,
    end: date,
) -> tuple[Path, Path]:
    """Write CSV of all results and a markdown summary. Returns (csv, md) paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"backtest_{tag}.csv"
    md_path = out_dir / f"backtest_{tag}.md"

    results.to_csv(csv_path, index=False)

    top = top_configs_per_symbol(results, n=3, min_trades=10)

    lines = [
        f"# Backtest Report — {tag}",
        "",
        f"- Window: **{start}** to **{end}**",
        f"- Configs tested: **{results['params'].nunique()}**",
        f"- Symbols: **{', '.join(sorted(results['symbol'].unique()))}**",
        f"- Slippage model: applied per SimulatedExecutor default",
        "",
        "## Top 3 configs per symbol (min 10 trades)",
        "",
    ]
    if top.empty:
        lines.append("_No (symbol × config) cell had ≥10 trades. Widen thresholds or the date window._")
    else:
        for sym in sorted(top["symbol"].unique()):
            lines.append(f"### {sym}")
            lines.append("")
            sub = top[top["symbol"] == sym][[
                "params", "n_trades", "win_rate", "avg_pnl_dollars",
                "total_pnl_dollars", "sharpe_daily", "max_drawdown_dollars",
            ]].copy()
            sub["win_rate"] = (sub["win_rate"] * 100).round(1).astype(str) + "%"
            sub["avg_pnl_dollars"] = sub["avg_pnl_dollars"].round(2)
            sub["total_pnl_dollars"] = sub["total_pnl_dollars"].round(2)
            sub["sharpe_daily"] = sub["sharpe_daily"].round(2)
            sub["max_drawdown_dollars"] = sub["max_drawdown_dollars"].round(2)
            lines.append(sub.to_markdown(index=False))
            lines.append("")

    # Also show the single best-per-symbol at a glance:
    if not top.empty:
        lines.append("## Best config per symbol")
        lines.append("")
        best = top.groupby("symbol").head(1)[[
            "symbol", "params", "n_trades", "win_rate", "total_pnl_dollars", "sharpe_daily"
        ]].copy()
        best["win_rate"] = (best["win_rate"] * 100).round(1).astype(str) + "%"
        best["total_pnl_dollars"] = best["total_pnl_dollars"].round(2)
        best["sharpe_daily"] = best["sharpe_daily"].round(2)
        lines.append(best.to_markdown(index=False))
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


# ---------------------------------------------------------------------------
# ORB grid builders
# ---------------------------------------------------------------------------

# Per-ticker breakout buffer bands (bps). Wider for more volatile names.
_ORB_BUFFERS_BPS: dict[str, tuple[int, ...]] = {
    "SPYM": (10, 20),
    "TQQQ": (30, 50),
    "TSLA": (30, 50),
    "NVDA": (30, 50),
    "COIN": (50, 80),
    "LYFT": (20, 40),
    "UBER": (20, 40),
    "HIMS": (30, 50),
    "RBLX": (30, 50),
    "HOOD": (30, 50),
    "PDD":  (50, 80),
    "NFLX": (20, 30),
}
_ORB_RANGE_WINDOWS = (15, 30, 60)
_ORB_EXIT_WINDOWS  = (30, 60, 120)
_ORB_DIRECTION_MODES = ("symmetric", "long_only", "short_only")


def build_orb_grids(
    universe: tuple[str, ...],
    buffers_bps_by_ticker: dict[str, tuple[int, ...]] | None = None,
    range_windows: tuple[int, ...] = _ORB_RANGE_WINDOWS,
    exit_windows: tuple[int, ...] = _ORB_EXIT_WINDOWS,
    direction_modes: tuple[str, ...] = _ORB_DIRECTION_MODES,
) -> dict[str, list[ORBParams]]:
    """Build per-ticker ORB parameter grids.

    Returns {symbol → list[ORBParams]}. Falls back to (20, 40) bps buffer
    for any symbol not in buffers_bps_by_ticker.
    """
    bps_map = buffers_bps_by_ticker or _ORB_BUFFERS_BPS
    grids: dict[str, list[ORBParams]] = {}
    for sym in universe:
        buffers = bps_map.get(sym, (20, 40))
        grids[sym] = [
            ORBParams(
                range_window_minutes=rw,
                breakout_buffer_bps=buf,
                exit_window_minutes=xw,
                direction_mode=dm,
            )
            for rw, buf, xw, dm in product(range_windows, buffers, exit_windows, direction_modes)
        ]
    return grids


# ---------------------------------------------------------------------------
# VWAP Reversion grid builders
# ---------------------------------------------------------------------------

_VWAP_DEVIATION_BPS: dict[str, tuple[int, ...]] = {
    "SPYM": (30, 50, 80),
    "TQQQ": (80, 120, 180),
    "TSLA": (80, 120, 180),
    "NVDA": (80, 120, 180),
    "COIN": (100, 150, 200),
    "LYFT": (60, 100, 150),
    "UBER": (60, 100, 150),
    "HIMS": (100, 150, 200),
    "RBLX": (100, 150, 200),
    "HOOD": (100, 150, 200),
    "PDD":  (120, 180, 250),
    "NFLX": (50,  80, 120),
}
_VWAP_ENTRY_START   = (30, 60)
_VWAP_EXIT_WINDOWS  = (15, 30, 60)
_VWAP_DIRECTION_MODES = ("symmetric", "long_only", "short_only")


def build_vwap_grids(
    universe: tuple[str, ...],
    deviation_bps_by_ticker: dict[str, tuple[int, ...]] | None = None,
    entry_start_minutes: tuple[int, ...] = _VWAP_ENTRY_START,
    exit_windows: tuple[int, ...] = _VWAP_EXIT_WINDOWS,
    direction_modes: tuple[str, ...] = _VWAP_DIRECTION_MODES,
) -> dict[str, list[VWAPReversionParams]]:
    """Build per-ticker VWAP reversion parameter grids."""
    dev_map = deviation_bps_by_ticker or _VWAP_DEVIATION_BPS
    grids: dict[str, list[VWAPReversionParams]] = {}
    for sym in universe:
        devs = dev_map.get(sym, (80, 120, 180))
        grids[sym] = [
            VWAPReversionParams(
                entry_start_minutes=es,
                deviation_bps=dev,
                exit_window_minutes=xw,
                direction_mode=dm,
            )
            for es, dev, xw, dm in product(entry_start_minutes, devs, exit_windows, direction_modes)
        ]
    return grids


# ---------------------------------------------------------------------------
# Volume Surge Momentum grid builders
# ---------------------------------------------------------------------------

_VSM_PRICE_MOVE_BPS: dict[str, tuple[int, ...]] = {
    "SPYM": (10, 20, 30),
    "TQQQ": (30, 50, 80),
    "TSLA": (30, 50, 80),
    "NVDA": (30, 50, 80),
    "COIN": (50, 80, 120),
    "LYFT": (20, 40, 60),
    "UBER": (20, 40, 60),
    "HIMS": (30, 60, 100),
    "RBLX": (30, 60, 100),
    "HOOD": (30, 60, 100),
    "PDD":  (50, 80, 120),
    "NFLX": (20, 40, 60),
}
_VSM_LOOKBACK_BARS   = (6, 12)        # 30-min or 60-min of 5-min bars
_VSM_MULTIPLIERS     = (2.0, 3.0, 4.0)
_VSM_EXIT_WINDOWS    = (15, 30, 60)
_VSM_DIRECTION_MODES = ("symmetric", "long_only", "short_only")


# ---------------------------------------------------------------------------
# MFI Divergence grid builders
# ---------------------------------------------------------------------------

# MFI divergence thresholds are mostly universal across tickers (MFI is a
# 0-100 bounded oscillator, not a price-scale indicator), so the per-ticker
# knob here is the divergence_lookback — wider/thinner names benefit from
# longer context because noise drowns short lookbacks.
_MFI_DIV_LOOKBACKS: dict[str, tuple[int, ...]] = {
    # Tight / liquid indices — shorter lookbacks catch cleaner pivots
    "SPYM": (8, 16),
    "TQQQ": (8, 16),
    "NFLX": (8, 16),
    # Single-name high-beta — medium lookbacks
    "TSLA": (12, 24),
    "NVDA": (12, 24),
    "UBER": (12, 24),
    "LYFT": (12, 24),
    # Very noisy small/mid-caps and ADRs — longer lookbacks
    "COIN": (16, 24),
    "HIMS": (16, 24),
    "RBLX": (16, 24),
    "HOOD": (16, 24),
    "PDD":  (16, 24),
}
_MFI_PERIODS            = (14,)                          # 14 = textbook; 21 adds little for 5-min intraday
_MFI_OVERSOLD_PAIRS     = ((20, 80), (25, 75), (30, 70))  # (oversold, overbought)
_MFI_EXIT_WINDOWS       = (15, 30)                        # longer holds rarely help MFI mean-reversion
_MFI_DIRECTION_MODES    = ("symmetric", "long_only", "short_only")


def build_mfi_divergence_grids(
    universe: tuple[str, ...],
    lookbacks_by_ticker: dict[str, tuple[int, ...]] | None = None,
    mfi_periods: tuple[int, ...] = _MFI_PERIODS,
    threshold_pairs: tuple[tuple[int, int], ...] = _MFI_OVERSOLD_PAIRS,
    exit_windows: tuple[int, ...] = _MFI_EXIT_WINDOWS,
    direction_modes: tuple[str, ...] = _MFI_DIRECTION_MODES,
) -> dict[str, list[MFIDivergenceParams]]:
    """Build per-ticker MFI Divergence parameter grids."""
    lb_map = lookbacks_by_ticker or _MFI_DIV_LOOKBACKS
    grids: dict[str, list[MFIDivergenceParams]] = {}
    for sym in universe:
        lookbacks = lb_map.get(sym, (12, 24))
        grids[sym] = [
            MFIDivergenceParams(
                mfi_period=mp,
                divergence_lookback=lb,
                oversold_threshold=pair[0],
                overbought_threshold=pair[1],
                exit_window_minutes=xw,
                direction_mode=dm,
            )
            for mp, lb, pair, xw, dm in product(
                mfi_periods, lookbacks, threshold_pairs, exit_windows, direction_modes
            )
        ]
    return grids


def build_vsm_grids(
    universe: tuple[str, ...],
    price_move_bps_by_ticker: dict[str, tuple[int, ...]] | None = None,
    lookback_bars: tuple[int, ...] = _VSM_LOOKBACK_BARS,
    volume_multipliers: tuple[float, ...] = _VSM_MULTIPLIERS,
    exit_windows: tuple[int, ...] = _VSM_EXIT_WINDOWS,
    direction_modes: tuple[str, ...] = _VSM_DIRECTION_MODES,
) -> dict[str, list[VolumeSurgeMomentumParams]]:
    """Build per-ticker Volume Surge Momentum parameter grids."""
    move_map = price_move_bps_by_ticker or _VSM_PRICE_MOVE_BPS
    grids: dict[str, list[VolumeSurgeMomentumParams]] = {}
    for sym in universe:
        moves = move_map.get(sym, (30, 50, 80))
        grids[sym] = [
            VolumeSurgeMomentumParams(
                lookback_bars=lb,
                volume_multiplier=vm,
                min_price_move_bps=mp,
                exit_window_minutes=xw,
                direction_mode=dm,
            )
            for lb, vm, mp, xw, dm in product(lookback_bars, volume_multipliers, moves, exit_windows, direction_modes)
        ]
    return grids


# ---------------------------------------------------------------------------
# IMC (Intraday Momentum Continuation) grid builders
# ---------------------------------------------------------------------------

# ATR multiples per ticker. Higher-volatility names need a tighter multiple
# because a fixed ATR fraction maps to a wider $-move; lower-vol names need
# a looser multiple or they never trigger. Calibrated so ~30–50% of sessions
# clear the threshold on the 2025 sample.
_IMC_ATR_MULTIPLES: dict[str, tuple[float, ...]] = {
    "SPYM": (0.50, 0.75, 1.00),  # tight index ETF, frequent small moves
    "TQQQ": (0.50, 0.75, 1.00),  # 3x leveraged, looser ATR but bigger gross move
    "TSLA": (0.50, 0.75, 1.00),
    "NVDA": (0.50, 0.75, 1.00),
    "COIN": (0.50, 0.75, 1.00),
    "LYFT": (0.50, 0.75, 1.00),
    "UBER": (0.50, 0.75, 1.00),
    "HIMS": (0.50, 0.75, 1.00),
    "RBLX": (0.50, 0.75, 1.00),
    "HOOD": (0.50, 0.75, 1.00),
    "PDD":  (0.50, 0.75, 1.00),
    "NFLX": (0.50, 0.75, 1.00),
}

_IMC_OBSERVATION_WINDOWS = (30, 60)            # minutes from open
_IMC_DECISION_TIME       = (330,)              # 15:00 ET
_IMC_EXIT_TIME           = (385,)              # 15:55 ET
_IMC_ATR_PERIODS         = (14, 30)            # 1-min bars
_IMC_VOLUME_Z_THRESHOLDS = (float("-inf"), 0.5)  # off / require above-mean volume
_IMC_USE_MARKET_FILTER   = (False, True)
_IMC_DIRECTION_MODES     = ("symmetric", "long_only", "short_only")


def build_imc_grids(
    universe: tuple[str, ...],
    atr_multiples_by_ticker: dict[str, tuple[float, ...]] | None = None,
    observation_windows: tuple[int, ...] = _IMC_OBSERVATION_WINDOWS,
    decision_times: tuple[int, ...] = _IMC_DECISION_TIME,
    exit_times: tuple[int, ...] = _IMC_EXIT_TIME,
    atr_periods: tuple[int, ...] = _IMC_ATR_PERIODS,
    volume_z_thresholds: tuple[float, ...] = _IMC_VOLUME_Z_THRESHOLDS,
    use_market_filter: tuple[bool, ...] = _IMC_USE_MARKET_FILTER,
    direction_modes: tuple[str, ...] = _IMC_DIRECTION_MODES,
) -> dict[str, list[IMCParams]]:
    """Build per-ticker IMC parameter grids.

    Default cell count per ticker:
      2 obs × 1 dec × 1 exit × 2 atr_period × 3 atr_mult × 2 vol_z × 2 mkt × 3 dir
      = 144 cells/ticker. Stays under the ~150-cell guideline in the plan.
    """
    mult_map = atr_multiples_by_ticker or _IMC_ATR_MULTIPLES
    grids: dict[str, list[IMCParams]] = {}
    for sym in universe:
        atr_multiples = mult_map.get(sym, (0.50, 0.75, 1.00))
        grids[sym] = [
            IMCParams(
                observation_window_minutes=ow,
                decision_time_minutes=dt,
                exit_time_minutes=xt,
                atr_period_bars=ap,
                atr_multiple=am,
                volume_z_threshold=vz,
                use_market_filter=mf,
                direction_mode=dm,
            )
            for ow, dt, xt, ap, am, vz, mf, dm in product(
                observation_windows,
                decision_times,
                exit_times,
                atr_periods,
                atr_multiples,
                volume_z_thresholds,
                use_market_filter,
                direction_modes,
            )
        ]
    return grids


def grid_uses_market_filter(grid: dict[str, list]) -> bool:
    """Return True if any IMC param in the per-ticker grid sets use_market_filter.

    The CLI uses this to decide whether to wire context_symbols=("SPYM",)
    into the BacktestRunner.
    """
    for params_list in grid.values():
        for p in params_list:
            if isinstance(p, IMCParams) and p.use_market_filter:
                return True
    return False
