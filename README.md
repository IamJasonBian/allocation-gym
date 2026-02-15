# allocation-gym

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

## Tests

```bash
pytest tests/ -v
```
