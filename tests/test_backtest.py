"""End-to-end backtest test on synthetic bars.

Uses an in-memory BarsProvider so the harness never touches disk or network.
Verifies that the grid search produces one summary row per (params, symbol)
cell and that metrics reflect realistic outcomes on crafted synthetic days.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from strategy.backtest import (
    BacktestRunner,
    build_default_grid,
    build_per_ticker_grids,
    top_configs_per_symbol,
)
from strategy.risk import RiskConfig
from strategy.strategies.intraday_overextension import IntradayOverextensionParams
from tests.conftest import make_day_bars


class InMemoryBarsProvider:
    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame]):
        self._bars = bars_by_symbol

    def get_bars(self, symbol, start, end):
        df = self._bars[symbol]
        idx_local = df.index.tz_convert("America/New_York")
        mask = (idx_local.date >= start) & (idx_local.date <= end)
        return df.loc[mask]


def _build_20_day_dataset(symbols=("AAA",), up_days=15, down_days=5):
    """15 days where the overextension theory works for a short on a +2% open move,
    5 days where it doesn't (continuation). Total 20 days."""
    days = []
    d0 = date(2026, 3, 2)
    for i in range(up_days):
        d = d0 + timedelta(days=i)
        # Up 2% at 10:00, down 1% by close → short wins
        days.append(make_day_bars(d, 100.0, {30: 0.02, 389: -0.01}, seed=i))
    for j in range(down_days):
        d = d0 + timedelta(days=up_days + j)
        # Up 2% at 10:00, up 3% by close → short loses
        days.append(make_day_bars(d, 100.0, {30: 0.02, 389: 0.03}, seed=100 + j))
    full = pd.concat(days)
    return {s: full for s in symbols}


def test_harness_runs_grid_and_produces_summary():
    data = _build_20_day_dataset(symbols=("AAA", "BBB"))
    runner = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA", "BBB"),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA", "BBB"), max_trades_per_day=10),
        slippage_bps=0.0,
    )
    grid = [
        IntradayOverextensionParams(30, 0.01, 15),
        IntradayOverextensionParams(30, 0.005, 15),
    ]
    df = runner.run(grid)
    # 2 configs × 2 symbols = 4 cells
    assert len(df) == 4
    assert set(df["symbol"].unique()) == {"AAA", "BBB"}
    # With 15/20 days winning by $1-ish, 5/20 losing $3, expect total P&L positive overall
    for _, row in df.iterrows():
        assert row["n_trades"] == 20  # strategy fires on every day (+2% always crosses)


def test_daily_trade_cap_applies_in_harness():
    data = _build_20_day_dataset(symbols=("AAA",))
    runner = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA",),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA",), max_trades_per_day=1),
        slippage_bps=0.0,
    )
    grid = [IntradayOverextensionParams(30, 0.01, 15)]
    df = runner.run(grid)
    # Strategy fires at most 1 per symbol per day → daily cap of 1 matches trades
    assert df.iloc[0]["n_trades"] == 20


def test_slippage_reduces_pnl():
    data = _build_20_day_dataset(symbols=("AAA",), up_days=20, down_days=0)
    grid = [IntradayOverextensionParams(30, 0.01, 15)]

    runner_no_slip = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA",),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA",)),
        slippage_bps=0.0,
    )
    df_clean = runner_no_slip.run(grid)

    runner_slippy = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA",),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA",)),
        slippage_bps=10.0,  # 10 bps is heavy
    )
    df_slip = runner_slippy.run(grid)

    assert df_slip.iloc[0]["total_pnl_dollars"] < df_clean.iloc[0]["total_pnl_dollars"]


def test_top_configs_filters_by_min_trades():
    data = _build_20_day_dataset(symbols=("AAA",), up_days=20, down_days=0)
    runner = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA",),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA",)),
    )
    df = runner.run(build_default_grid(direction_modes=("symmetric",)))
    # With threshold 2%+, the strategy never fires on a 2% move (inclusive check)
    # so those rows should be filtered out by min_trades=10.
    top = top_configs_per_symbol(df, n=3, min_trades=10)
    for _, row in top.iterrows():
        assert row["n_trades"] >= 10


def test_per_ticker_grid_applies_correct_thresholds():
    """A per-ticker dict grid applies each symbol's own threshold list."""
    data = _build_20_day_dataset(symbols=("AAA", "BBB"))
    runner = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA", "BBB"),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA", "BBB")),
        slippage_bps=0.0,
    )
    grids = build_per_ticker_grids(
        universe=("AAA", "BBB"),
        thresholds_bps_by_ticker={"AAA": (50,), "BBB": (100,)},
        entry_windows=(30,),
        exit_windows=(15,),
        direction_modes=("symmetric",),
    )
    assert len(grids["AAA"]) == 1
    assert len(grids["BBB"]) == 1
    df = runner.run(grids)
    # 1 config per symbol × 2 symbols = 2 rows
    assert len(df) == 2
    aaa_row = df[df["symbol"] == "AAA"].iloc[0]
    bbb_row = df[df["symbol"] == "BBB"].iloc[0]
    assert aaa_row["threshold_pct"] == 0.005
    assert bbb_row["threshold_pct"] == 0.010


def test_grid_dict_missing_symbol_raises():
    data = _build_20_day_dataset(symbols=("AAA", "BBB"))
    runner = BacktestRunner(
        bars_provider=InMemoryBarsProvider(data),
        universe=("AAA", "BBB"),
        start=date(2026, 3, 2),
        end=date(2026, 3, 31),
        risk_config=RiskConfig(universe=("AAA", "BBB")),
    )
    # Dict missing BBB → should fail loudly rather than silently dropping
    incomplete = {"AAA": [IntradayOverextensionParams(30, 0.01, 15)]}
    with pytest.raises(ValueError, match="BBB"):
        runner.run(incomplete)
