# strategy-research

A personal intraday strategy research framework: data acquisition, backtesting, walk-forward validation, and per-strategy implementations. Everything runs locally against cached 1-minute bars.

**Current version: 0.2.0**

---

## Overview

The framework is built around a 12-ticker universe of high-activity names and a fixed set of risk rules: $1,000 notional per trade (fractional shares), max 10 trades per day across the full universe, and all positions flat by end of day. Strategies are evaluated with a train / walk-forward split to keep out-of-sample results honest.

Data comes primarily from **Alpaca's Market Data API** (1-min SIP consolidated-tape bars, free paper-trading account, ~3+ year history). **Schwab's priceHistory API** (1-min bars, ~48-day window) is a backup source for topping up the cache with the most recent sessions. All backtests run against the local parquet cache — no network calls at backtest time.

---

## Project structure

```
strategy_research/
├── strategy/                   # Core library
│   ├── data.py                 # Bar providers: Alpaca (primary), Schwab (backup), Parquet; merge + cache helpers
│   ├── backtest.py             # BacktestRunner: grid-search, train/walk-forward split, reports
│   ├── executor.py             # SimulatedExecutor: fill model, slippage, P&L realization
│   ├── metrics.py              # compute_metrics: Sharpe, win rate, avg trade, max drawdown
│   ├── risk.py                 # RiskConfig, RiskManager, DEFAULT_UNIVERSE, size_trade
│   └── strategies/
│       ├── base.py             # Trade dataclass, Strategy Protocol (with optional context_bars)
│       ├── opening_range_breakout.py          # ORB: first-N-minute range, breakout continuation
│       ├── vwap_reversion.py                  # VWAP: fade extreme deviations from session VWAP
│       ├── volume_surge_momentum.py           # VSM: momentum on abnormal volume spike
│       ├── mfi_divergence.py                  # MFI: price/volume divergence mean-reversion
│       ├── intraday_overextension.py          # IOE: fade large opening moves
│       └── intraday_momentum_continuation.py  # IMC: morning return predicts last-30m (experimental)
│
├── scripts/
│   ├── fetch_intraday_history.py   # Build and refresh the local parquet cache
│   └── run_backtest_generic.py     # CLI: train / walk-forward for any strategy
│
├── tests/                      # pytest suite (all synthetic data, no network)
│   ├── conftest.py
│   ├── test_backtest.py
│   ├── test_metrics.py
│   ├── test_risk.py
│   ├── test_intraday_overextension.py
│   ├── test_mfi_divergence.py
│   └── test_intraday_momentum_continuation.py
│
├── strategy/data_cache/        # Parquet files (Alpaca 1-min SIP) — git-ignored
└── strategy/results/           # Backtest output CSVs, Markdown reports — git-ignored
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Alpaca credentials (required for fetching data)

Create a `.env` file in the repo root (copy from `.env.example`) and add:

```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

A free paper-trading account at [alpaca.markets](https://alpaca.markets) is sufficient — no funded account needed. Schwab credentials are only needed if you use `--source schwab` to top up with the most recent 48 days.

---

## Fetching data

Build the local bar cache before running backtests. Parquet files land in `strategy/data_cache/` and are git-ignored.

```bash
# Default: Alpaca SIP 1-min bars from 2023-01-01 to today
python3 scripts/fetch_intraday_history.py

# Custom start date
python3 scripts/fetch_intraday_history.py --start 2022-01-01

# Specific symbols only
python3 scripts/fetch_intraday_history.py --symbols TSLA,NVDA,COIN

# Top up with fresh Schwab 1-min bars (last 48 days) — requires Schwab credentials
python3 scripts/fetch_intraday_history.py --source schwab
```

Alpaca's SIP feed covers all US exchanges on the consolidated tape. The SDK paginates large date ranges automatically.

---

## Running backtests

Use `run_backtest_generic.py` for all strategy backtests. It runs a full train / walk-forward split: grid-search on the training window, pick the best config per ticker, then replay only that config on the hold-out window.

```bash
# Default: IOE strategy, 2025 train, 2026 walk-forward, full 12-ticker universe
python3 scripts/run_backtest_generic.py --strategy ioe

# Any of the six strategies
python3 scripts/run_backtest_generic.py --strategy orb
python3 scripts/run_backtest_generic.py --strategy vwap
python3 scripts/run_backtest_generic.py --strategy vsm
python3 scripts/run_backtest_generic.py --strategy mfi
python3 scripts/run_backtest_generic.py --strategy imc

# Custom date range
python3 scripts/run_backtest_generic.py --strategy vsm \
    --train-start 2024-01-01 --train-end 2024-12-31 \
    --test-start  2025-01-01 --test-end  2025-12-31

# Subset of tickers
python3 scripts/run_backtest_generic.py --strategy mfi --symbols TSLA,NVDA,COIN

# Verbose (per-day debug output)
python3 scripts/run_backtest_generic.py --strategy ioe -v
```

Results are written to `strategy/results/` as `.md` (human-readable summary) and `.csv` (full per-trade log).

---

## Strategies

All strategies share the same contract: one trade per symbol per day, all positions flat by EOD, sizing delegated to the `RiskManager` ($1,000 notional per trade, max 10 trades/day across the universe).

### Opening Range Breakout (ORB)

The high and low printed in the first N minutes of the session form a support/resistance map. A decisive close above the range high triggers a long (momentum); a close below the range low triggers a short. Position held for a fixed exit window, hard backstop at 15:45 ET.

Key parameters: `range_window_minutes`, `exit_window_minutes`, `breakout_pct`, `direction_mode` (symmetric / long_only / short_only).

### VWAP Mean Reversion

VWAP represents the intraday "fair value" institutional desks benchmark against. When price drifts far above VWAP we short (fade the extension); far below, we go long. Exit triggers when price reverts to VWAP, when the exit time window expires, or at the 15:50 ET hard stop.

Key parameters: `deviation_pct`, `entry_start_minutes`, `exit_window_minutes`, `direction_mode`.

### Volume Surge Momentum (VSM)

When volume spikes well above the session average early in the day, it often signals institutional accumulation or distribution. The strategy trades in the direction of the price move accompanying the surge.

Key parameters: `volume_multiplier`, `entry_window_minutes`, `exit_window_minutes`, `direction_mode`.

### MFI Divergence

The Money Flow Index (MFI) is a volume-weighted RSI. When price makes a new session extreme but MFI fails to confirm — lower low in price paired with a higher MFI low, or higher high in price paired with a lower MFI high — the volume behind the move is drying up. That divergence, combined with an overbought/oversold filter, is a mean-reversion trigger.

Key parameters: `mfi_period`, `overbought_threshold`, `oversold_threshold`, `divergence_lookback`, `exit_window_minutes`.

### Intraday Overextension (IOE)

If a ticker moves more than N% from its session open within the first `entry_window_minutes`, the move tends to fade. The strategy takes the opposite side and exits near the close.

Key parameters: `threshold_pct`, `entry_window_minutes`, `exit_window_minutes`, `direction_mode`.

### Intraday Momentum Continuation (IMC) — *experimental*

Based on Gao-Han-Li-Zhou (JFE 2018, replicated through 2026): the morning return predicts the last-half-hour return. At a fixed decision time (default 15:00 ET) the strategy measures the morning return `r_open` over the first `observation_window_minutes` and computes an ATR over the prior `atr_period_bars`. If `|r_open| > atr_multiple × atr`, it enters in the direction of `r_open` and exits at `exit_time_minutes` (default 15:55 ET). ATR normalization replaces the fixed-bps thresholds used by the other strategies.

Optionally requires the SPYM morning return to agree in sign (`use_market_filter=True`) — implemented via an additive `context_bars` extension on the `Strategy` Protocol and the `context_symbols` field on `BacktestRunner`, which enables any future strategy to read cross-asset context without breaking the single-symbol contract.

Key parameters: `observation_window_minutes`, `decision_time_minutes`, `exit_time_minutes`, `atr_period_bars`, `atr_multiple`, `volume_z_threshold`, `use_market_filter`, `direction_mode`.

**Status:** training-set Sharpe is strong (12/12 tickers profitable on 2025) but the 2026-Q1 walk-forward came in at −$330 / 2-of-12 profitable, missing the strategy's own acceptance bar. See [`strategy/results/imc_walkforward_analysis.md`](strategy/results/imc_walkforward_analysis.md) for the full postmortem and proposed next steps.

---

## Risk rules

Defined in `strategy/risk.py` and shared across backtesting and any live runner:

| Rule | Value |
|------|-------|
| Notional per trade | $1,000 (fractional shares — exact notional regardless of price) |
| Max trades per day | 10 across the full universe |
| Max universe size | 20 tickers |
| EOD flat | enforced by each strategy (hard stop between 15:45–15:58 ET) |

**Default universe (12 tickers):** `SPYM, TQQQ, TSLA, NVDA, COIN, LYFT, UBER, HIMS, RBLX, HOOD, PDD, NFLX`

---

## Data providers

| Provider | Granularity | Window | Auth required |
|----------|-------------|--------|---------------|
| `AlpacaBarsProvider` | 1-min (SIP) | ~3+ years (free account) | Yes (`ALPACA_API_KEY` / `ALPACA_API_SECRET` in `.env`) |
| `SchwabBarsProvider` | 1-min | ~48 calendar days | Yes (`token.json` + Schwab credentials from `schwab_app`) |
| `ParquetBarsProvider` | as cached | whatever is on disk | No (reads local cache) |

The backtest harness reads exclusively from the local parquet cache via `ParquetBarsProvider` — no network calls at backtest time. Use `AlpacaBarsProvider` (primary) to build the initial cache from 2023 onward, and `SchwabBarsProvider` (backup) to top up with the freshest sessions if needed.

---

## Metrics

`strategy/metrics.py` computes the following for each ticker × parameter configuration:

- **Total trades** — number of completed round-trips
- **Win rate** — fraction of trades with positive P&L
- **Average trade P&L** — mean net P&L per trade (after simulated slippage)
- **Total P&L** — sum across all trades
- **Sharpe ratio** — annualized ratio of mean daily P&L to its standard deviation
- **Max drawdown** — largest peak-to-trough drop in the cumulative P&L series

---

## Testing

```bash
source venv/bin/activate
python3 -m pytest tests/ -v
```

Tests use synthetic bar data built by `tests/conftest.py:make_day_bars` and do not require Alpaca/Schwab credentials or the local parquet cache.

---

## Relation to schwab_app

`SchwabBarsProvider` in `strategy/data.py` uses `core.auth.get_client()` from the companion `schwab_app` repo for live Schwab API access. It is only needed when you run `fetch_intraday_history.py --source schwab`. All other workflows (Alpaca fetch, backtesting from cache) work without `schwab_app` in your Python path.
