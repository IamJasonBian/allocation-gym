# allocation-gym

Version - 0.1.0 - This is part of the [allocation-manager](https://github.com/OptimChain/allocation-manager/) system.

Variance-Kelly backtesting framework built on [Backtrader](https://www.backtrader.com/).

Ports the core components from the Variance-Kelly Trading Engine into a standalone backtesting framework:
- **VarianceMetrics** — Yang-Zhang variance, Variance Ratio, Efficiency Ratio, Vol-of-Vol, semivariance, regime classification
- **Kelly position sizing** — Quarter-Kelly with downside semivariance stop distances
- **3 strategies** — Momentum, Mean Reversion, Multi-asset Kelly Rebalancing
- **Performance analytics** — Sharpe, Sortino, max drawdown, CAGR, Calmar ratio

## Install

```bash
pip install -e ".[data,dev]"
```

## Usage

```bash
# Momentum strategy on SPY/QQQ/NVDA (Yahoo Finance data)
python -m allocation_gym.runner --strategy momentum --symbols SPY QQQ NVDA \
    --start 2023-01-01 --end 2024-01-01

# Mean reversion on broad indices
python -m allocation_gym.runner --strategy mean_reversion --symbols SPY IWM EFA \
    --start 2023-01-01 --end 2024-01-01

# Multi-asset Kelly rebalancing
python -m allocation_gym.runner --strategy variance_kelly --symbols SPY GLD QQQ \
    --start 2023-01-01 --end 2024-01-01

# With Alpaca historical data
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
python -m allocation_gym.runner --strategy momentum --symbols SPY \
    --data-source alpaca --start 2024-02-15 --end 2025-02-15
```

## Strategies

| Strategy | Entry | Exit | Sizer |
|---|---|---|---|
| `momentum` | VR >= 1.1, ER >= 0.4, price > SMA50 | Price < SMA50 or CHOP regime | KellySizer |
| `mean_reversion` | VR < 0.9, RSI < 30 | RSI > 70 | KellySizer |
| `variance_kelly` | Kelly-optimal weights via inverse covariance | Rebalance when drift > 5% | Explicit sizing |

## Module Structure

```
allocation_gym/
├── runner.py              # CLI backtester entry-point
├── config.py              # BacktestConfig dataclass
├── enums.py               # SignalType, BandType, AssetClass, TradeState, EvalMetric
├── credentials.py         # Alpaca credential resolution (env vars → GitHub vars)
│
├── data/                  # Shared OHLCV data loaders
│   └── loaders.py         # load_ohlcv(), load_alpaca_data(), load_yfinance_data()
│
├── metrics/               # Pure-numpy computation layer (no Backtrader dependency)
│   └── variance_metrics.py# VarianceMetrics: 9 metrics + regime classification
│
├── indicators/            # Backtrader indicator wrappers
│   ├── variance.py        # VarianceIndicator (wraps VarianceMetrics)
│   ├── iv_zscore.py       # IV z-score indicator
│   └── etf_flow.py        # ETF flow indicator
│
├── strategies/            # Trading strategies
│   ├── momentum.py        # Trend-following (VR ≥ 1.1, ER ≥ 0.4, price > SMA50)
│   ├── mean_reversion.py  # Mean reversion (VR < 0.9, RSI < 30)
│   ├── variance_kelly.py  # Multi-asset Kelly rebalancing
│   └── momentum_dca.py    # Dollar-cost averaging variant
│
├── sizers/
│   └── kelly.py           # Quarter-Kelly with semivariance stops
│
├── analyzers/
│   └── performance.py     # Sharpe, Sortino, max DD, CAGR, Calmar, daily P&L
│
├── simulation/            # Monte Carlo forward testing
│   ├── engine.py          # MonteCarloGBM simulator
│   ├── calibrate.py       # GBM calibration from historical OHLCV
│   ├── forward_test.py    # Forward test harness
│   └── runner.py          # CLI simulator (python -m allocation_gym.simulation)
│
├── signals/
│   └── btc_dashboard.py   # BTC signal generation (python -m allocation_gym.signals)
│
├── blobs/                 # Netlify blob store client for options data
│   ├── client.py
│   ├── index.py
│   └── models.py
│
├── options/               # Options analysis tools (research-grade)
│   ├── black_scholes.py   # bs_put_price(), bs_call_price()
│   ├── metrics.py         # OptionsBacktestMetrics, compute_options_metrics()
│   └── ...                # Simulation, report generation, vol analysis
│
├── plotting.py            # Backtest result visualisation (Matplotlib)
├── evaluators.py          # Trade evaluation helpers
├── optimal_bands.py       # OU-process optimal band calibration (scipy)
└── weekend.py             # Weekend/holiday date adjustments
```

### Optional dependencies

| Extra | Packages | Required for |
|---|---|---|
| `data` | yfinance, alpaca-py | Live data loading |
| `plot` | matplotlib | Backtest charts |
| `dev` | pytest | Running tests |

### Environment variables

| Variable | Purpose |
|---|---|
| `ALPACA_API_KEY` | Alpaca API key (data loading) |
| `ALPACA_SECRET_KEY` | Alpaca secret key (data loading) |
| `TWELVE_DATA_API_KEY` | Twelve Data key (scripts/backtest_optimal_bands.py only) |

## Tests

```bash
pytest tests/ -v
```

