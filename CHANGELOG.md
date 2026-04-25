# Changelog

All notable changes to strategy-research are documented here.

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
