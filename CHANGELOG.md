# Changelog

All notable changes to strategy-research are documented here.

---

## [0.2.0] — 2026-04-25

### Added — Intraday Momentum Continuation (IMC), the sixth strategy

- **`strategy/strategies/intraday_momentum_continuation.py`** — Implements IMC, based on the Gao-Han-Li-Zhou (JFE 2018, replicated through 2026) "morning-return predicts last-half-hour-return" effect. At a fixed `decision_time` (default 15:00 ET) for each session and symbol, computes a morning return `r_open` over the first `observation_window_minutes` and an ATR over the prior `atr_period_bars`; enters in the direction of `r_open` if `|r_open| > atr_multiple × atr` and exits at `exit_time_minutes` (default 15:55 ET). Supports `direction_mode` (symmetric / long_only / short_only), an optional within-session `volume_z_threshold` filter, and an optional cross-asset `use_market_filter` that requires SPYM's morning return to agree in sign before trading.
- **`build_imc_grids` in `strategy/backtest.py`** — Per-ticker IMC parameter grids (~144 cells/ticker by default) with per-ticker `atr_multiple` ranges and a coarse default sweep on observation window, ATR period, volume-z threshold, market-filter on/off, and direction mode.
- **`grid_uses_market_filter` helper** — Lets the CLI auto-detect when an IMC grid needs cross-asset context.
- **CLI registration** — `imc` is now a first-class strategy in `scripts/run_backtest_generic.py`. When the grid contains any market-filter cells, the CLI automatically wires `context_symbols=("SPYM",)` into the `BacktestRunner` so SPYM bars are loaded once and passed through to the strategy as cross-asset context.
- **Tests** — `tests/test_intraday_momentum_continuation.py` with 18 cases covering long/short signals, ATR threshold rejection, direction-mode gating, volume-z filtering, market-filter agreement and rejection, the empty/no-context safe-fail path, multi-day input rejection, and EOD edge cases.
- **Walk-forward report** — `strategy/results/backtest_imc_2025_train.md` and `strategy/results/backtest_imc_2026_walkforward.md`, plus a postmortem at `strategy/results/imc_walkforward_analysis.md` documenting why the strategy missed its 2026-Q1 acceptance bar and what changes would be needed before promoting it.

### Added — Optional cross-asset context (additive framework extension)

- **`Strategy` Protocol now accepts `context_bars`** — `generate_trades_for_day` gains an optional `context_bars: dict[str, pd.DataFrame] | None = None` keyword argument carrying per-context-symbol bars sliced to the same trading session as `day_bars`. Default is `None`, so all five legacy strategies are fully backward compatible — they accept the kwarg but ignore it. A strategy that needs market-regime / cross-asset context (like IMC's SPYM filter) reads it without violating the single-symbol decision contract.
- **`BacktestRunner` gains `context_symbols`** — A new optional tuple field. When non-empty, the runner preloads those symbols' bars once, slices them per session date, and passes the slice into every strategy call. Symbols already in the trading universe are reused without reloading. Empty default keeps existing scripts and the five incumbent strategies unaffected.
- **Look-ahead protection** — Context bars are sliced strictly to the same session date as `day_bars`, and the IMC market filter further enforces a `<= obs_ts` slice when reading the morning return. The harness never lets a context symbol leak future-bar information into a same-day decision.

### Changed

- **`strategy/strategies/__init__.py`** — Now exports all six strategies and their parameter dataclasses.

---

## [0.1.0] — 2026-04-25

Initial release. Full backtesting framework extracted from the `schwab_app` repo and published as a standalone research project.

### Core library (`strategy/`)

- **`data.py`** — Bar provider protocol and three concrete implementations: `SchwabBarsProvider` (1-min bars via Schwab priceHistory API, ~48-day window), `YFinanceBarsProvider` (5-min bars via yfinance, ~100-day window), and `ParquetBarsProvider` (reads locally cached parquet files, no network). `merge_bars` blends two providers with Schwab winning on overlapping days. `write_cache` / `split_by_session_date` helpers. Session boundary constants (`SESSION_OPEN`, `SESSION_CLOSE`) in US/Eastern.
- **`executor.py`** — `SimulatedExecutor`: tracks open positions, enforces one trade per symbol per day, applies the hard EOD flat rule. `realize_trade` converts an open position to a closed `Trade` with net P&L.
- **`metrics.py`** — `compute_metrics`: total trades, win rate, average trade P&L, total P&L, annualized Sharpe ratio, and max drawdown from a sequence of `Trade` objects.
- **`risk.py`** — `RiskConfig` dataclass ($1,000 sleeve, $100/trade, max 10 trades/day, max 20-ticker universe). `RiskManager`: per-day trade counter, universe membership check, `can_trade` / `register_trade` / `size_for`. `size_trade` helper for fractional-share sizing. `DEFAULT_UNIVERSE` (12 tickers: SPYM, TQQQ, TSLA, NVDA, COIN, LYFT, UBER, HIMS, RBLX, HOOD, PDD, NFLX).
- **`backtest.py`** — `BacktestRunner`: grid-searches strategy parameters over a date range, applies a configurable train/walk-forward split (default 70/30), outputs a `pd.DataFrame` with per-ticker per-config metrics. Supports all five strategies via a unified `StrategyFactory` protocol.

### Strategies (`strategy/strategies/`)

- **`base.py`** — `Trade` dataclass (symbol, side, entry/exit time and price, shares, P&L), `Side` enum (LONG / SHORT).
- **`opening_range_breakout.py`** — Opening Range Breakout (ORB). Captures the high/low of the first `range_window_minutes`, then trades breakouts with configurable `breakout_pct` confirmation threshold. Entry: next bar's open after confirmation. Exit: first bar after `exit_window_minutes`, hard stop at 15:45 ET. Direction: symmetric / long_only / short_only.
- **`vwap_reversion.py`** — VWAP Mean Reversion. Fades price deviations beyond `deviation_pct` from the session VWAP after `entry_start_minutes`. Exit: first of VWAP reversion, `exit_window_minutes` timeout, or 15:50 ET hard stop. Direction: symmetric / long_only / short_only.
- **`volume_surge_momentum.py`** — Volume Surge Momentum (VSM). Triggers on volume spikes exceeding `volume_multiplier` × session average within `entry_window_minutes`. Trades in the direction of the accompanying price move. Exit: first bar after `exit_window_minutes`, hard stop at 15:50 ET.
- **`mfi_divergence.py`** — MFI Divergence. Scans each session for price/MFI divergence at new session extremes while MFI is in the overbought/oversold zone. Standard 14-period MFI formula (volume-weighted RSI). Entry: next bar's open. Exit: first of MFI reversion through 50, `exit_window_minutes` timeout, or 15:50 ET hard stop.
- **`intraday_overextension.py`** — Intraday Overextension (IOE). Fades moves exceeding `threshold_pct` from the session open within `entry_window_minutes`. Entry: next bar's open. Exit: first bar after `session_close - exit_window_minutes`, hard stop at 15:45 ET. Direction: symmetric / long_only / short_only.

### Scripts

- **`scripts/fetch_intraday_history.py`** — CLI to build and refresh the local parquet cache. Supports `--source` (merged / schwab / yfinance), `--symbols`, `--days`, `--end`, `--output-dir`. Merged mode chunks Schwab requests in 45-day windows to stay within the API's ~48-day limit, then blends with yfinance's wider window.
- **`scripts/run_backtest.py`** — CLI to run a grid-search backtest for any strategy. Supports `--strategy`, `--start`, `--end`, `--symbols`, `--train-ratio`. Writes results to `strategy/results/` as CSV (full trade log) and Markdown (summary report).

### Tests

- `tests/test_backtest.py` — BacktestRunner integration tests with synthetic bar data.
- `tests/test_metrics.py` — Unit tests for all metrics (Sharpe edge cases, drawdown, win rate).
- `tests/test_risk.py` — RiskConfig, RiskManager, and size_trade unit tests.
- `tests/test_intraday_overextension.py` — IOE strategy signal and exit logic.
- `tests/test_mfi_divergence.py` — MFI calculation and divergence detection.

### Infrastructure

- `.gitignore` — excludes parquet files (`*.parquet`), `strategy/data_cache/`, `strategy/schwab_cache/`, `strategy/results/*.csv` and `*.md`, credentials (`.env`, `token.json`), Python bytecode, and common tooling artifacts.
- `requirements.txt` — pandas, pyarrow, yfinance, schwab-py (live fetch only), pytest, pytest-cov, ruff.
