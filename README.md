# strategy-research

A personal intraday strategy research framework: data acquisition, backtesting, walk-forward validation, and per-strategy implementations. Everything runs locally against cached 1-minute bars.

**Current version: 0.1.0**

---

## Overview

The framework is built around a small universe of high-activity tickers and a fixed set of risk rules: $100 notional per trade, max 10 trades per day across the full universe, and all positions flat by end of day. Strategies are evaluated with a train / walk-forward split to keep out-of-sample results honest.

Data comes from two sources: **Schwab's priceHistory API** (1-min bars, ~48-day window) and **yfinance** (5-min bars, ~100-day window). In merged mode the fetcher blends both, letting Schwab's finer resolution win on overlapping days, so the effective history extends to roughly 100 calendar days.

---

## Project structure

```
strategy_research/
├── strategy/                   # Core library
│   ├── data.py                 # Bar providers: Schwab, yfinance, Parquet; merge + cache helpers
│   ├── backtest.py             # BacktestRunner: grid-search, train/walk-forward split, output CSV
│   ├── executor.py             # SimulatedExecutor: fill model, P&L realization
│   ├── metrics.py              # compute_metrics: Sharpe, win rate, avg trade, max drawdown
│   ├── risk.py                 # RiskConfig, RiskManager, DEFAULT_UNIVERSE, size_trade
│   └── strategies/
│       ├── base.py             # Trade dataclass, Side enum
│       ├── opening_range_breakout.py   # ORB: first-N-minute range, breakout continuation
│       ├── vwap_reversion.py           # VWAP: fade extreme deviations from VWAP
│       ├── volume_surge_momentum.py    # VSM: momentum on abnormal volume spike
│       ├── mfi_divergence.py           # MFI: price/volume divergence mean-reversion
│       └── intraday_overextension.py   # IOE: fade large opening moves
│
├── scripts/
│   ├── fetch_intraday_history.py   # Build the local parquet cache
│   └── run_backtest.py             # CLI: run + report any strategy
│
├── tests/                      # pytest suite
│   ├── conftest.py
│   ├── test_backtest.py
│   ├── test_metrics.py
│   ├── test_risk.py
│   ├── test_backtest.py
│   ├── test_intraday_overextension.py
│   └── test_mfi_divergence.py
│
├── strategy/data_cache/        # Merged (yfinance + Schwab) parquet files — git-ignored
├── strategy/schwab_cache/      # Schwab-only 1-min parquet files — git-ignored
└── strategy/results/           # Backtest output CSVs and reports — git-ignored
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

### 3. (Optional) Schwab credentials — only needed for live data fetching

Copy `.env.example` from `schwab_app` and fill in your App Key and App Secret, or ensure `token.json` is present (shared with `schwab_app`). yfinance data works without any credentials.

---

## Fetching data

Build the local bar cache before running backtests. All parquet files land in `strategy/data_cache/` and are git-ignored.

```bash
# Default: merge yfinance 5-min (older ~100d) + Schwab 1-min (recent ~48d)
python3 scripts/fetch_intraday_history.py

# yfinance only — no Schwab credentials needed
python3 scripts/fetch_intraday_history.py --source yfinance

# Specific symbols
python3 scripts/fetch_intraday_history.py --symbols TSLA,NVDA,AAPL

# Write Schwab-only bars to a separate directory
python3 scripts/fetch_intraday_history.py --source schwab --output-dir strategy/schwab_cache
```

The `--source merged` mode (default) blends both providers: yfinance extends the window to ~100 days, Schwab's 1-min resolution wins on overlapping days.

---

## Running backtests

```bash
# Default: Intraday Overextension, full universe, last 90 days, train+walk-forward split
python3 scripts/run_backtest.py

# Specific strategy
python3 scripts/run_backtest.py --strategy orb
python3 scripts/run_backtest.py --strategy vwap
python3 scripts/run_backtest.py --strategy volume_surge
python3 scripts/run_backtest.py --strategy mfi_divergence
python3 scripts/run_backtest.py --strategy overextension

# Custom date range and symbols
python3 scripts/run_backtest.py --strategy orb --start 2026-01-01 --end 2026-04-25 --symbols TSLA,NVDA,COIN

# Verbose output
python3 scripts/run_backtest.py --strategy vwap -v
```

Results are written to `strategy/results/` as both a `.csv` (full per-trade log) and a `.md` (human-readable summary). The runner always performs a **train / walk-forward split** (default 70/30) so the final numbers are out-of-sample.

---

## Strategies

All strategies share the same contract: one trade per symbol per day, all positions flat by EOD, sizing delegated to the `RiskManager` ($100 notional per trade, max 10 trades/day).

### Opening Range Breakout (ORB)

The high and low printed in the first N minutes of the session form a support/resistance map. A decisive close above the range high triggers a long (momentum); a close below the range low triggers a short. Position held for a fixed exit window, hard backstop at 15:45 ET.

Key parameters: `range_window_minutes`, `exit_window_minutes`, `breakout_pct`, `direction` (symmetric / long_only / short_only).

### VWAP Mean Reversion

VWAP represents the intraday "fair value" institutional desks benchmark against. When price drifts far above VWAP we short (fade the extension); far below, we go long. Exit triggers when price reverts to VWAP, when the exit time window expires, or at the 15:50 ET hard stop.

Key parameters: `deviation_pct`, `entry_start_minutes`, `exit_window_minutes`, `direction`.

### Volume Surge Momentum (VSM)

When volume spikes well above the session average early in the day, it often signals institutional accumulation or distribution. The strategy trades in the direction of the price move accompanying the surge.

Key parameters: `volume_multiplier`, `entry_window_minutes`, `exit_window_minutes`, `direction`.

### MFI Divergence

The Money Flow Index (MFI) is a volume-weighted RSI. When price makes a new session extreme but MFI fails to confirm — lower low in price paired with a higher MFI low, or higher high in price paired with a lower MFI high — the volume behind the move is drying up. That divergence, combined with an overbought/oversold filter, is a mean-reversion trigger.

Key parameters: `mfi_period`, `overbought`, `oversold`, `divergence_lookback`, `exit_window_minutes`.

### Intraday Overextension (IOE)

If a ticker moves more than N% from its session open within the first `entry_window_minutes`, the move tends to fade. The strategy takes the opposite side and exits near the close.

Key parameters: `threshold_pct`, `entry_window_minutes`, `exit_window_minutes`, `direction`.

### Intraday Momentum Continuation (IMC) — *experimental*

Based on Gao-Han-Li-Zhou (JFE 2018, replicated through 2026): the morning return predicts the last-half-hour return. At a fixed decision time (default 15:00 ET) we measure the morning return `r_open` and compute an ATR over the prior `atr_period_bars`. If `|r_open| > atr_multiple × atr` we enter in the direction of `r_open` and exit at `exit_time_minutes` (default 15:55 ET). ATR-normalization replaces the fixed-bps thresholds that overfit in the other strategies.

Optionally requires the SPYM morning return to agree in sign (`use_market_filter=True`) — this is the first strategy in the framework to read cross-asset context, exposed via the additive `context_bars` extension on the `Strategy` Protocol and the `context_symbols` field on `BacktestRunner`.

Key parameters: `observation_window_minutes`, `decision_time_minutes`, `exit_time_minutes`, `atr_period_bars`, `atr_multiple`, `volume_z_threshold`, `use_market_filter`, `direction_mode`.

**Status:** training-set Sharpe is strong (12/12 tickers profitable on 2025) but the 2026-Q1 walk-forward came in at −$330 / 2-of-12 profitable, missing the strategy's own acceptance bar. See [`strategy/results/imc_walkforward_analysis.md`](strategy/results/imc_walkforward_analysis.md) for the full postmortem and proposed next steps.

---

## Risk rules

Defined in `strategy/risk.py` and shared across backtesting and any live runner:

| Rule | Value |
|------|-------|
| Capital sleeve | $1,000 |
| Notional per trade | $100 (fractional shares supported) |
| Max trades per day | 10 across the full universe |
| Max universe size | 20 tickers |
| EOD flat | enforced by each strategy (hard stop between 15:45–15:50 ET) |

**Default universe:** `SPYM, TQQQ, TSLA, NVDA, COIN, LYFT, UBER, HIMS, RBLX, HOOD, PDD, NFLX`

---

## Data providers

| Provider | Granularity | Window | Auth required |
|----------|-------------|--------|---------------|
| `SchwabBarsProvider` | 1-min | ~48 calendar days | Yes (`token.json` + `.env`) |
| `YFinanceBarsProvider` | 5-min | ~100 calendar days | No |
| `ParquetBarsProvider` | as cached | whatever is on disk | No (reads local cache) |

In merged mode, Schwab's 1-min bars win on any day both providers cover. The backtest harness reads exclusively from the local parquet cache via `ParquetBarsProvider` — no network calls at backtest time.

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

Tests use synthetic bar data and do not require Schwab credentials or the local parquet cache.

---

## Relation to schwab_app

`SchwabBarsProvider` in `strategy/data.py` uses `core.auth.get_client()` from the companion `schwab_app` repo for live Schwab API access. If you only run backtests from the parquet cache (the typical workflow), `schwab_app` does not need to be in your Python path. It is only needed when you run `fetch_intraday_history.py --source schwab` or `--source merged`.
