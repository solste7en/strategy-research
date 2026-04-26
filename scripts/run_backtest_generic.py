"""Generic train / walk-forward backtest runner.

Supports all five strategies with per-ticker parameter grids.
Train on one date range, pick the best config per ticker, then test
on a separate out-of-sample window — no look-ahead.

Usage (from repo root):

    # IOE — Intraday Overextension
    python3 scripts/run_backtest_generic.py --strategy ioe

    # VSM — Volume Surge Momentum
    python3 scripts/run_backtest_generic.py --strategy vsm

    # MFI — MFI Divergence
    python3 scripts/run_backtest_generic.py --strategy mfi

    # ORB — Opening Range Breakout
    python3 scripts/run_backtest_generic.py --strategy orb

    # VWAP — VWAP Mean Reversion
    python3 scripts/run_backtest_generic.py --strategy vwap

    # Custom date ranges
    python3 scripts/run_backtest_generic.py --strategy vsm \\
        --train-start 2024-01-01 --train-end 2024-12-31 \\
        --test-start  2025-01-01 --test-end  2025-12-31

    # Subset of tickers
    python3 scripts/run_backtest_generic.py --strategy mfi --symbols TSLA,NVDA,COIN

Defaults:
    --train-start  2025-01-01
    --train-end    2025-12-31
    --test-start   2026-01-01
    --test-end     2026-04-24
    --per-trade    1000
    --max-trades   10
    --min-trades   10     (min trades for a config to qualify)
    --top-n        3      (top configs shown per ticker in training report)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from strategy.backtest import (  # noqa: E402
    BacktestRunner,
    build_per_ticker_grids,
    build_orb_grids,
    build_vwap_grids,
    build_vsm_grids,
    build_mfi_divergence_grids,
    build_imc_grids,
    grid_uses_market_filter,
    top_configs_per_symbol,
    write_report,
)
from strategy.data import ParquetBarsProvider  # noqa: E402
from strategy.risk import DEFAULT_UNIVERSE, RiskConfig  # noqa: E402
from strategy.strategies.intraday_overextension import IntradayOverextensionStrategy  # noqa: E402
from strategy.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy  # noqa: E402
from strategy.strategies.vwap_reversion import VWAPReversionStrategy  # noqa: E402
from strategy.strategies.volume_surge_momentum import VolumeSurgeMomentumStrategy  # noqa: E402
from strategy.strategies.mfi_divergence import MFIDivergenceStrategy  # noqa: E402
from strategy.strategies.intraday_momentum_continuation import IntradayMomentumContinuationStrategy  # noqa: E402

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "ioe":  (build_per_ticker_grids,      IntradayOverextensionStrategy,         "IOE — Intraday Overextension"),
    "orb":  (build_orb_grids,             OpeningRangeBreakoutStrategy,          "ORB — Opening Range Breakout"),
    "vwap": (build_vwap_grids,            VWAPReversionStrategy,                 "VWAP — Mean Reversion"),
    "vsm":  (build_vsm_grids,             VolumeSurgeMomentumStrategy,           "VSM — Volume Surge Momentum"),
    "mfi":  (build_mfi_divergence_grids,  MFIDivergenceStrategy,                 "MFI — Divergence"),
    "imc":  (build_imc_grids,             IntradayMomentumContinuationStrategy,  "IMC — Intraday Momentum Continuation"),
}

OUT_DIR = _ROOT / "strategy" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    """Format a results DataFrame for console printing."""
    cols = ["symbol", "params", "n_trades", "win_rate",
            "avg_pnl_dollars", "total_pnl_dollars", "sharpe_daily", "max_drawdown_dollars"]
    d = df[[c for c in cols if c in df.columns]].copy()
    if "win_rate"            in d: d["win_rate"]            = (d["win_rate"]*100).round(1).astype(str)+"%"
    if "avg_pnl_dollars"     in d: d["avg_pnl_dollars"]     = d["avg_pnl_dollars"].round(1)
    if "total_pnl_dollars"   in d: d["total_pnl_dollars"]   = d["total_pnl_dollars"].round(0)
    if "sharpe_daily"        in d: d["sharpe_daily"]        = d["sharpe_daily"].round(2)
    if "max_drawdown_dollars"in d: d["max_drawdown_dollars"]= d["max_drawdown_dollars"].round(0)
    return d


def _build_risk(per_trade: float, max_trades: int, universe: tuple[str, ...]) -> RiskConfig:
    return RiskConfig(
        total_capital     = per_trade * len(universe),
        per_trade_dollars = per_trade,
        max_trades_per_day= max_trades,
        universe          = universe,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train/walk-forward backtest for any strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Strategies: ioe, orb, vwap, vsm, mfi\n\n"
            "Examples:\n"
            "  python3 scripts/run_backtest_generic.py --strategy vsm\n"
            "  python3 scripts/run_backtest_generic.py --strategy mfi --symbols TSLA,NVDA,COIN\n"
            "  python3 scripts/run_backtest_generic.py --strategy orb "
            "--train-start 2024-01-01 --train-end 2024-12-31 "
            "--test-start 2025-01-01 --test-end 2025-12-31\n"
        ),
    )
    parser.add_argument("--strategy",     required=True, choices=list(STRATEGIES),
                        help="Strategy to backtest.")
    parser.add_argument("--train-start",  default="2025-01-01", help="Training start date (YYYY-MM-DD).")
    parser.add_argument("--train-end",    default="2025-12-31", help="Training end date (YYYY-MM-DD).")
    parser.add_argument("--test-start",   default="2026-01-01", help="Walk-forward start date (YYYY-MM-DD).")
    parser.add_argument("--test-end",     default="2026-04-24", help="Walk-forward end date (YYYY-MM-DD).")
    parser.add_argument("--symbols",      default=",".join(DEFAULT_UNIVERSE),
                        help=f"Comma-separated tickers. Default: all 12.")
    parser.add_argument("--per-trade",    type=float, default=1_000.0,
                        help="Notional dollars per trade (fractional shares). Default: 1000.")
    parser.add_argument("--max-trades",   type=int, default=10,
                        help="Max trades per day across the universe. Default: 10.")
    parser.add_argument("--min-trades",   type=int, default=10,
                        help="Min trades for a config to qualify as top-N. Default: 10.")
    parser.add_argument("--top-n",        type=int, default=3,
                        help="Top configs shown per ticker in training summary. Default: 3.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    strategy_name = args.strategy
    grid_builder, strategy_factory, strategy_label = STRATEGIES[strategy_name]

    train_start = date.fromisoformat(args.train_start)
    train_end   = date.fromisoformat(args.train_end)
    test_start  = date.fromisoformat(args.test_start)
    test_end    = date.fromisoformat(args.test_end)

    universe = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    risk     = _build_risk(args.per_trade, args.max_trades, universe)
    provider = ParquetBarsProvider()

    # ── Build grid ────────────────────────────────────────────────────────────
    grid = grid_builder(universe)
    total_cells = sum(len(v) for v in grid.values())

    # IMC's optional SPYM market-regime filter requires the harness to load
    # SPYM bars as cross-asset context. Auto-detect from grid contents so the
    # CLI works for both filtered and unfiltered IMC runs.
    context_syms: tuple[str, ...] = ()
    if grid_uses_market_filter(grid):
        context_syms = ("SPYM",)
        log.info("IMC market-filter cells present — loading SPYM as context")

    # ── TRAINING ──────────────────────────────────────────────────────────────
    log.info("=== %s | TRAINING %s → %s | %d cells across %d tickers ===",
             strategy_label, train_start, train_end, total_cells, len(universe))

    train_runner = BacktestRunner(
        bars_provider=provider, universe=universe,
        start=train_start, end=train_end,
        risk_config=risk, slippage_bps=2.0,
        context_symbols=context_syms,
    )
    train_results = train_runner.run(grid, strategy_factory=strategy_factory)

    tag_train = f"{strategy_name}_{train_start.year}_train"
    csv_t, md_t = write_report(train_results, OUT_DIR, tag_train, train_start, train_end)
    log.info("Training report → %s", md_t)

    top_train = top_configs_per_symbol(train_results, n=args.top_n, min_trades=args.min_trades)

    print("\n" + "="*70)
    print(f"{strategy_label}  |  TRAINING  {train_start} → {train_end}")
    print("="*70)
    print(f"  Cells evaluated : {len(train_results)}")
    print(f"  Tickers with ≥{args.min_trades}-trade configs: {top_train['symbol'].nunique()}/{len(universe)}")

    if top_train.empty:
        print(f"\n  [!] No ticker had a config with ≥{args.min_trades} trades.")
        print("      Try lowering --min-trades or widening the date range.")
        sys.exit(1)

    print(f"\n── Top {args.top_n} configs per ticker (training) ──")
    print(_fmt(top_train).to_string(index=False))

    print("\n── Best config per ticker ──")
    best_row = top_train.groupby("symbol", sort=False).head(1)
    print(
        _fmt(best_row)
        .sort_values("total_pnl_dollars", ascending=False)
        .to_string(index=False)
    )

    # ── Extract best params per ticker ────────────────────────────────────────
    best_params: dict[str, object] = {}
    for _, row in best_row.iterrows():
        sym, tag = row["symbol"], row["params"]
        for p in grid[sym]:
            if p.as_tag() == tag:
                best_params[sym] = p
                break

    qualified = tuple(best_params.keys())
    missing   = [s for s in universe if s not in best_params]
    if missing:
        log.warning("No qualifying config for %s — excluded from walk-forward.", missing)

    # ── WALK-FORWARD ──────────────────────────────────────────────────────────
    log.info("=== %s | WALK-FORWARD %s → %s | %d tickers ===",
             strategy_label, test_start, test_end, len(qualified))

    test_runner = BacktestRunner(
        bars_provider=provider, universe=qualified,
        start=test_start, end=test_end,
        risk_config=_build_risk(args.per_trade, args.max_trades, qualified),
        slippage_bps=2.0,
        context_symbols=context_syms,
    )
    test_results = test_runner.run(
        {sym: [p] for sym, p in best_params.items()},
        strategy_factory=strategy_factory,
    )

    tag_test = f"{strategy_name}_{test_start.year}_walkforward"
    csv_w, md_w = write_report(test_results, OUT_DIR, tag_test, test_start, test_end)
    log.info("Walk-forward report → %s", md_w)

    print("\n" + "="*70)
    print(f"{strategy_label}  |  WALK-FORWARD  {test_start} → {test_end}")
    print("="*70)
    wf = _fmt(test_results).sort_values("total_pnl_dollars", ascending=False)
    print(wf.to_string(index=False))

    n_trades   = int(test_results["n_trades"].sum())
    total_pnl  = test_results["total_pnl_dollars"].sum()
    n_positive = int((test_results["total_pnl_dollars"] > 0).sum())

    print(f"\n── Portfolio summary (walk-forward) ──")
    print(f"  Total trades      : {n_trades}")
    print(f"  Total P&L         : ${total_pnl:,.0f}")
    print(f"  Profitable tickers: {n_positive}/{len(test_results)}")
    print(f"\nReports saved to strategy/results/")
    print(f"  Training     → {md_t.name}")
    print(f"  Walk-forward → {md_w.name}")


if __name__ == "__main__":
    main()
