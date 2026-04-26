"""Legacy single-window backtest runner (grid-search only, no walk-forward split).

NOTE: For the recommended train / walk-forward workflow use
``scripts/run_backtest_generic.py`` instead. This script is retained for
quick single-window explorations with a rolling-date-range interface.

Supports: ioe (overextension), orb, vwap, vsm (volume_surge), mfi.

Usage (from repo root):

    # Strategy selection (default: overextension)
    python3 scripts/run_backtest.py --strategy overextension
    python3 scripts/run_backtest.py --strategy orb
    python3 scripts/run_backtest.py --strategy vwap
    python3 scripts/run_backtest.py --strategy volume_surge
    python3 scripts/run_backtest.py --strategy mfi

    python3 scripts/run_backtest.py --symbols NVDA,COIN
    python3 scripts/run_backtest.py --start 2026-01-28 --end 2026-04-24
    python3 scripts/run_backtest.py --slippage-bps 5

    # Walk-forward validation: grid on train sessions, verify top config on test
    python3 scripts/run_backtest.py --walk-forward
    python3 scripts/run_backtest.py --walk-forward --n-train 45 --n-test 15

Outputs files in strategy/results/:
    backtest_<strategy>_<tag>.csv  — all (symbol × config) rows with metrics
    backtest_<strategy>_<tag>.md   — top-3 configs per symbol
    backtest_<strategy>_<tag>_walkforward.md  — side-by-side train vs test P&L
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy.backtest import (  # noqa: E402
    BacktestRunner,
    build_orb_grids,
    build_per_ticker_grids,
    build_vwap_grids,
    build_vsm_grids,
    build_mfi_divergence_grids,
    write_report,
)
from strategy.data import ParquetBarsProvider  # noqa: E402
from strategy.risk import DEFAULT_UNIVERSE, RiskConfig  # noqa: E402
from strategy.strategies.intraday_overextension import IntradayOverextensionParams  # noqa: E402
from strategy.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy  # noqa: E402
from strategy.strategies.vwap_reversion import VWAPReversionStrategy  # noqa: E402
from strategy.strategies.volume_surge_momentum import VolumeSurgeMomentumStrategy  # noqa: E402
from strategy.strategies.mfi_divergence import MFIDivergenceStrategy  # noqa: E402

log = logging.getLogger(__name__)

_DEFAULT_RESULTS_DIR = _ROOT / "strategy" / "results"


# ---------------------------------------------------------------------------
# Walk-forward helpers
# ---------------------------------------------------------------------------


def collect_session_dates(
    bars_provider: ParquetBarsProvider,
    symbols: tuple[str, ...],
    start: date,
    end: date,
) -> list[date]:
    """Union of all cached session dates across symbols, filtered to [start, end]."""
    all_dates: set[date] = set()
    for sym in symbols:
        try:
            dates = bars_provider.trading_dates(sym)
            all_dates.update(d for d in dates if start <= d <= end)
        except FileNotFoundError:
            log.warning("No cache for %s — skipping in session count", sym)
    return sorted(all_dates)


def walk_forward_split(
    session_dates: list[date],
    n_train: int,
    n_test: int,
) -> tuple[list[date], list[date]]:
    """Chronological split: first n_train sessions = train, next n_test = test."""
    total = len(session_dates)
    if total < n_train + n_test:
        log.warning(
            "Only %d sessions available; requested %d train + %d test = %d. "
            "Will use what is available.",
            total, n_train, n_test, n_train + n_test,
        )
        n_train = max(1, total - n_test)
        n_test = min(n_test, total - n_train)
    train = session_dates[:n_train]
    test = session_dates[n_train: n_train + n_test]
    return train, test


def find_best_params_per_symbol(
    train_results,
    grids: dict[str, list[IntradayOverextensionParams]],
    symbols: tuple[str, ...],
    min_trades: int = 1,
) -> dict[str, IntradayOverextensionParams | None]:
    """Return the highest-total-P&L params per symbol from the train result frame.

    Builds a tag→params lookup per symbol, then finds the top-ranked row.
    Returns None for any symbol with no qualifying rows.
    """
    best: dict[str, IntradayOverextensionParams | None] = {}
    for sym in symbols:
        tag_to_params = {p.as_tag(): p for p in grids.get(sym, [])}
        sym_df = train_results[train_results["symbol"] == sym]
        qualifying = sym_df[sym_df["n_trades"] >= min_trades].copy()
        if qualifying.empty:
            log.warning(
                "%s: no configs with >= %d trades on train set — will skip test", sym, min_trades
            )
            best[sym] = None
            continue
        # Frame is already sorted by total_pnl_dollars desc for each symbol.
        top_tag = qualifying.iloc[0]["params"]
        best[sym] = tag_to_params.get(top_tag)
        if best[sym] is None:
            log.warning("%s: could not resolve params for tag %r", sym, top_tag)
    return best


def write_walk_forward_report(
    train_results,
    test_results,
    best_params: dict[str, "IntradayOverextensionParams | None"],
    train_dates: list[date],
    test_dates: list[date],
    symbols: tuple[str, ...],
    out_dir: Path,
    tag: str,
) -> Path:
    """Write the side-by-side train vs test P&L markdown file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"backtest_{tag}_walkforward.md"

    train_map = (
        {row["symbol"]: row for _, row in train_results.iterrows()}
        if not train_results.empty else {}
    )
    test_map = (
        {row["symbol"]: row for _, row in test_results.iterrows()}
        if not test_results.empty else {}
    )

    lines = [
        f"# Walk-Forward Validation — {tag}",
        "",
        f"- Train: **{train_dates[0]}** to **{train_dates[-1]}** ({len(train_dates)} sessions)",
        f"- Test:  **{test_dates[0]}** to **{test_dates[-1]}** ({len(test_dates)} sessions)",
        f"- Split: {len(train_dates)} train / {len(test_dates)} test",
        "",
        "## Side-by-Side Results",
        "",
        "| Symbol | Best Config (train) | Train Trades | Train P&L | Train Win% "
        "| Test Trades | Test P&L | Test Win% | Verdict |",
        "|--------|---------------------|:------------:|----------:|:---------:"
        "|:-----------:|---------:|:---------:|---------|",
    ]

    passes = 0
    fails = 0
    skips = 0

    for sym in sorted(symbols):
        params = best_params.get(sym)
        if params is None:
            lines.append(
                f"| {sym} | — | — | — | — | — | — | — | SKIP (no train data) |"
            )
            skips += 1
            continue

        cfg_tag = params.as_tag()

        # Train row (single-config test_results was run with this config)
        tr = train_map.get(sym)
        train_trades = int(tr["n_trades"]) if tr is not None else 0
        train_pnl = float(tr["total_pnl_dollars"]) if tr is not None else 0.0
        train_wr = float(tr["win_rate"]) * 100 if tr is not None else 0.0

        # Test row
        te = test_map.get(sym)
        test_trades = int(te["n_trades"]) if te is not None else 0
        test_pnl = float(te["total_pnl_dollars"]) if te is not None else 0.0
        test_wr = float(te["win_rate"]) * 100 if te is not None else 0.0

        # Verdict: PASS if test P&L > 0 AND at least 1 trade fired
        if test_trades == 0:
            verdict = "SKIP (0 test trades)"
            skips += 1
        elif test_pnl > 0:
            verdict = "PASS"
            passes += 1
        else:
            verdict = "FAIL"
            fails += 1

        pnl_sign = "+" if train_pnl >= 0 else ""
        test_sign = "+" if test_pnl >= 0 else ""

        lines.append(
            f"| {sym} | `{cfg_tag}` "
            f"| {train_trades} | {pnl_sign}${train_pnl:.2f} | {train_wr:.0f}% "
            f"| {test_trades} | {test_sign}${test_pnl:.2f} | {test_wr:.0f}% "
            f"| {verdict} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- **PASS**: {passes} / {passes + fails + skips} symbols "
        f"({passes}/{passes+fails} excluding skips)",
        f"- **FAIL**: {fails}",
        f"- **SKIP**: {skips} (no train data or zero test trades)",
        "",
        "> A config PASSES if it finishes the test window with positive P&L.",
        "> Zero test trades means the signal never fired in the test period — "
        "not necessarily a strategy failure, but worth noting.",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_strategy(
    strategy_name: str,
    symbols: tuple[str, ...],
    direction_modes: tuple[str, ...],
) -> tuple[dict, type, str]:
    """Return (grids, strategy_factory, human_label) for the chosen strategy."""
    if strategy_name == "orb":
        from strategy.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
        grids = build_orb_grids(universe=symbols, direction_modes=direction_modes)
        return grids, OpeningRangeBreakoutStrategy, "Opening Range Breakout"
    elif strategy_name == "vwap":
        from strategy.strategies.vwap_reversion import VWAPReversionStrategy
        grids = build_vwap_grids(universe=symbols, direction_modes=direction_modes)
        return grids, VWAPReversionStrategy, "VWAP Mean Reversion"
    elif strategy_name == "volume_surge":
        from strategy.strategies.volume_surge_momentum import VolumeSurgeMomentumStrategy
        grids = build_vsm_grids(universe=symbols, direction_modes=direction_modes)
        return grids, VolumeSurgeMomentumStrategy, "Volume Surge Momentum"
    elif strategy_name == "mfi_divergence":
        from strategy.strategies.mfi_divergence import MFIDivergenceStrategy
        grids = build_mfi_divergence_grids(universe=symbols, direction_modes=direction_modes)
        return grids, MFIDivergenceStrategy, "MFI Divergence"
    else:  # default: overextension
        from strategy.strategies.intraday_overextension import IntradayOverextensionStrategy
        grids = build_per_ticker_grids(universe=symbols, direction_modes=direction_modes)
        return grids, IntradayOverextensionStrategy, "Intraday Overextension"


def main():
    parser = argparse.ArgumentParser(description="Grid-search backtest for intraday strategies.")
    parser.add_argument(
        "--strategy",
        choices=["overextension", "orb", "vwap", "volume_surge", "mfi_divergence"],
        default="overextension",
        help=(
            "Strategy to backtest. "
            "overextension=IntradayOverextension (default), "
            "orb=Opening Range Breakout, "
            "vwap=VWAP Mean Reversion, "
            "volume_surge=Volume Surge Momentum, "
            "mfi_divergence=Money Flow Index Divergence."
        ),
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_UNIVERSE),
        help=f"Comma-separated tickers. Default: {','.join(DEFAULT_UNIVERSE)}",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 120 days before end).")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today).")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Simulated slippage per fill in basis points.")
    parser.add_argument(
        "--direction-modes",
        default="symmetric,short_only,long_only",
        help="Comma-separated direction modes to sweep. Subset of: symmetric,short_only,long_only",
    )
    parser.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR))
    parser.add_argument("-v", "--verbose", action="store_true")

    # Walk-forward flags
    wf = parser.add_argument_group("walk-forward validation")
    wf.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Split the available sessions into train and test windows. "
            "Run the full grid on train only, then re-run the best config per symbol "
            "on the untouched test window and print a side-by-side P&L comparison."
        ),
    )
    wf.add_argument(
        "--n-train",
        type=int,
        default=45,
        help="Number of sessions in the train window (default: 45).",
    )
    wf.add_argument(
        "--n-test",
        type=int,
        default=15,
        help="Number of sessions in the test window (default: 15).",
    )
    wf.add_argument(
        "--test-cache-dir",
        default=None,
        help=(
            "Optional path to a separate bar cache for the test phase "
            "(e.g. strategy/schwab_cache with 1-min Schwab bars). "
            "If omitted, the same cache as training is used."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=120)
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    direction_modes = tuple(m.strip() for m in args.direction_modes.split(",") if m.strip())
    results_dir = Path(args.results_dir)
    tag = f"{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    bars_provider = ParquetBarsProvider()

    # Resolve strategy grid and factory based on --strategy flag
    grids, strategy_factory, strategy_label = _resolve_strategy(
        args.strategy, symbols, direction_modes
    )
    total_cells = sum(len(g) for g in grids.values())

    # ------------------------------------------------------------------
    # Walk-forward mode
    # ------------------------------------------------------------------
    if args.walk_forward:
        log.info("=== Walk-Forward Mode: %d train / %d test sessions ===", args.n_train, args.n_test)

        session_dates = collect_session_dates(bars_provider, symbols, start, end)
        if not session_dates:
            log.error(
                "No session dates found in cache for %s between %s and %s. "
                "Run fetch_intraday_history.py first.",
                symbols, start, end,
            )
            sys.exit(1)

        train_dates, test_dates = walk_forward_split(session_dates, args.n_train, args.n_test)
        if not train_dates or not test_dates:
            log.error("Not enough sessions for a train/test split. Got %d total.", len(session_dates))
            sys.exit(1)

        train_start, train_end = train_dates[0], train_dates[-1]
        test_start, test_end = test_dates[0], test_dates[-1]

        log.info(
            "Train: %s → %s (%d sessions) | Test: %s → %s (%d sessions)",
            train_start, train_end, len(train_dates),
            test_start, test_end, len(test_dates),
        )
        log.info(
            "Grid: %d total cells across %d symbols (slippage=%.1fbp, modes=%s)",
            total_cells, len(symbols), args.slippage_bps, direction_modes,
        )

        # --- Phase 1: full grid search on train sessions ---
        log.info("Phase 1: running full grid on train sessions (%s)...", strategy_label)
        train_runner = BacktestRunner(
            bars_provider=bars_provider,
            universe=symbols,
            start=train_start,
            end=train_end,
            risk_config=RiskConfig(universe=symbols),
            slippage_bps=args.slippage_bps,
        )
        train_results = train_runner.run(grids, strategy_factory=strategy_factory)

        # Write the standard train report
        csv_path, md_path = write_report(train_results, results_dir, f"{tag}_train", train_start, train_end)
        log.info("Train report: %s", md_path)

        # --- Phase 2: identify best config per symbol from train ---
        best_params = find_best_params_per_symbol(train_results, grids, symbols, min_trades=1)

        # Summarize selections
        log.info("Phase 1 complete. Best configs selected per symbol:")
        for sym in sorted(symbols):
            p = best_params.get(sym)
            if p:
                train_row = train_results[
                    (train_results["symbol"] == sym) & (train_results["params"] == p.as_tag())
                ]
                pnl = train_row["total_pnl_dollars"].iloc[0] if not train_row.empty else float("nan")
                log.info("  %-6s  %s  (train P&L: $%.2f)", sym, p.as_tag(), pnl)
            else:
                log.info("  %-6s  no qualifying config", sym)

        # --- Phase 3: run only the selected configs on the test window ---
        log.info("Phase 2: running best configs on test sessions...")
        test_grids: dict[str, list] = {
            sym: [p] for sym, p in best_params.items() if p is not None
        }
        tested_symbols = tuple(s for s in symbols if best_params.get(s) is not None)

        if not tested_symbols:
            log.error("No symbols had a qualifying train config — nothing to test.")
            sys.exit(1)

        # Allow a separate high-fidelity cache (e.g. Schwab 1-min) for the test phase
        if args.test_cache_dir:
            test_cache_path = Path(args.test_cache_dir)
            if not test_cache_path.is_absolute():
                test_cache_path = _ROOT / test_cache_path
            log.info("Test phase using separate bar cache: %s", test_cache_path)
            test_bars_provider = ParquetBarsProvider(cache_dir=test_cache_path)
        else:
            test_bars_provider = bars_provider

        test_runner = BacktestRunner(
            bars_provider=test_bars_provider,
            universe=tested_symbols,
            start=test_start,
            end=test_end,
            risk_config=RiskConfig(universe=tested_symbols),
            slippage_bps=args.slippage_bps,
        )
        test_results = test_runner.run(test_grids, strategy_factory=strategy_factory)

        # Trim train_results to just the winning configs for the side-by-side table
        train_best_rows = []
        for sym, p in best_params.items():
            if p is None:
                continue
            mask = (train_results["symbol"] == sym) & (train_results["params"] == p.as_tag())
            rows = train_results[mask]
            if not rows.empty:
                train_best_rows.append(rows.iloc[0])
        import pandas as pd
        train_best_df = pd.DataFrame(train_best_rows) if train_best_rows else pd.DataFrame()

        # --- Write walk-forward report ---
        wf_path = write_walk_forward_report(
            train_results=train_best_df,
            test_results=test_results,
            best_params=best_params,
            train_dates=train_dates,
            test_dates=test_dates,
            symbols=symbols,
            out_dir=results_dir,
            tag=tag,
        )
        log.info("Walk-forward report: %s", wf_path)

        # --- Console summary ---
        _print_walk_forward_console(
            train_best_df, test_results, best_params, symbols, train_dates, test_dates
        )

        return

    # ------------------------------------------------------------------
    # Standard (non-walk-forward) mode
    # ------------------------------------------------------------------
    runner = BacktestRunner(
        bars_provider=bars_provider,
        universe=symbols,
        start=start,
        end=end,
        risk_config=RiskConfig(universe=symbols),
        slippage_bps=args.slippage_bps,
    )
    _ = strategy_label  # used in log above

    log.info(
        "running %d total %s configs across %d symbols over %s..%s (slippage=%.1fbp, modes=%s)",
        total_cells, strategy_label, len(symbols), start, end, args.slippage_bps, direction_modes,
    )
    for sym, g in grids.items():
        log.info("  %s: %d configs", sym, len(g))

    results = runner.run(grids, strategy_factory=strategy_factory)

    csv_path, md_path = write_report(results, results_dir, tag, start, end)
    log.info("wrote %s", csv_path)
    log.info("wrote %s", md_path)


# ---------------------------------------------------------------------------
# Console pretty-print for walk-forward
# ---------------------------------------------------------------------------


def _print_walk_forward_console(
    train_best_df,
    test_results,
    best_params: dict,
    symbols: tuple[str, ...],
    train_dates: list[date],
    test_dates: list[date],
) -> None:
    """Print a compact side-by-side table to stdout."""
    SEP = "=" * 110
    print()
    print(SEP)
    print(f"  WALK-FORWARD VALIDATION  |  "
          f"Train: {train_dates[0]} → {train_dates[-1]} ({len(train_dates)} sessions)  |  "
          f"Test: {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} sessions)")
    print(SEP)

    header = (
        f"{'Symbol':<6}  {'Best Config (train)':<40}  "
        f"{'Tr Trades':>9}  {'Train P&L':>10}  {'Tr Win%':>7}  "
        f"{'Te Trades':>9}  {'Test P&L':>10}  {'Te Win%':>7}  {'Verdict'}"
    )
    print(header)
    print("-" * 110)

    import pandas as pd
    train_map = (
        {row["symbol"]: row for _, row in train_best_df.iterrows()}
        if not train_best_df.empty else {}
    )
    test_map = (
        {row["symbol"]: row for _, row in test_results.iterrows()}
        if not test_results.empty else {}
    )

    passes = fails = skips = 0

    for sym in sorted(symbols):
        p = best_params.get(sym)
        if p is None:
            print(f"{sym:<6}  {'(no qualifying train config)':<40}  {'—':>9}  {'—':>10}  {'—':>7}  {'—':>9}  {'—':>10}  {'—':>7}  SKIP")
            skips += 1
            continue

        tr = train_map.get(sym)
        te = test_map.get(sym)

        tr_n = int(tr["n_trades"]) if tr is not None else 0
        tr_pnl = float(tr["total_pnl_dollars"]) if tr is not None else 0.0
        tr_wr = float(tr["win_rate"]) * 100 if tr is not None else 0.0

        te_n = int(te["n_trades"]) if te is not None else 0
        te_pnl = float(te["total_pnl_dollars"]) if te is not None else 0.0
        te_wr = float(te["win_rate"]) * 100 if te is not None else 0.0

        if te_n == 0:
            verdict = "SKIP (0 trades)"
            skips += 1
        elif te_pnl > 0:
            verdict = "PASS"
            passes += 1
        else:
            verdict = "FAIL"
            fails += 1

        tr_pnl_str = f"+${tr_pnl:.2f}" if tr_pnl >= 0 else f"-${abs(tr_pnl):.2f}"
        te_pnl_str = f"+${te_pnl:.2f}" if te_pnl >= 0 else f"-${abs(te_pnl):.2f}"

        print(
            f"{sym:<6}  {p.as_tag():<40}  "
            f"{tr_n:>9}  {tr_pnl_str:>10}  {tr_wr:>6.0f}%  "
            f"{te_n:>9}  {te_pnl_str:>10}  {te_wr:>6.0f}%  {verdict}"
        )

    print("-" * 110)
    total_sym = len(symbols)
    print(
        f"  PASS {passes}/{total_sym}   FAIL {fails}/{total_sym}   SKIP {skips}/{total_sym}"
        f"  (PASS rate excl. skips: "
        f"{passes}/{passes+fails} = {100*passes/max(passes+fails,1):.0f}%)"
    )
    print(SEP)
    print()


if __name__ == "__main__":
    main()
