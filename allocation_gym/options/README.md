# Options Module

Options hedging simulation and pricing analysis, ported from
[allocation-engine](https://github.com/IamJasonBian/allocation-engine).

## Structure

```
options/
├── black_scholes.py        # Pure-Python BS pricing (no scipy)
├── simulation.py           # Protective put & collar sim with monthly rolling
├── metrics.py              # Sharpe, drawdown, premium tracking for options
├── drawdown_analysis.py    # BTC drawdown backtest (120d, IV/strike sensitivity)
├── plot_drawdown.py        # 5-panel drawdown hedge visualization
├── backtest_forward.py     # BTC+IWM 2-week backtest + 30d MC forward projection
├── iwm_vol_compare.py      # IWM week-over-week vol & pricing comparison
├── historical_pricing.py   # 6-month rolling vol + historical options pricing
└── report.py               # PDF report generator (tables + embedded charts)
```

## Quick Start

### Generate the PDF report (price tables, vol charts, strike grid)

```bash
python -m allocation_gym.options.report
# -> docs/btc_iwm_options_report.pdf
```

### BTC drawdown analysis (how options handled the crash)

```bash
python -m allocation_gym.options.drawdown_analysis --iv 0.80 --otm-pct 0.05
# -> terminal output: roll breakdown, IV/strike sensitivity tables

python -m allocation_gym.options.plot_drawdown --iv 0.80
# -> docs/btc_options_drawdown.png (5-panel chart)
```

### BTC + IWM call backtest + forward test

```bash
python -m allocation_gym.options.backtest_forward --iv-btc 0.80 --iv-iwm 0.25
# -> docs/btc_iwm_call_forward.png (4-panel: backtest + MC fan chart)
```

### IWM vol & pricing week-over-week

```bash
python -m allocation_gym.options.iwm_vol_compare
# -> docs/iwm_vol_compare.png (price, rolling vol, call/put bar charts)
```

### Historical pricing (6-month time series)

```bash
python -m allocation_gym.options.historical_pricing --days 180
# -> docs/btc_iwm_historical_pricing.png (8-panel: price, vol, options %, ratio)
```

## Output Files

| File | Description |
|------|-------------|
| `docs/btc_iwm_options_report.pdf` | Full report: pricing tables, charts, observations |
| `docs/btc_options_drawdown.png` | BTC price + put strikes, returns, drawdown, premiums, vol |
| `docs/btc_iwm_call_forward.png` | 2-week backtest + 30-day MC forward projection |
| `docs/iwm_vol_compare.png` | IWM this-week vs last-week vol and option prices |
| `docs/btc_iwm_historical_pricing.png` | 6-month rolling vol + options pricing history |

## Core Components

### Black-Scholes Pricing (`black_scholes.py`)

Pure Python, no scipy dependency. Abramowitz & Stegun normal CDF approximation.

- `bs_call_price(S, K, T, r, sigma)` -- European call
- `bs_put_price(S, K, T, r, sigma)` -- European put

### Simulation Engine (`simulation.py`)

Monthly-rolling options overlay on any underlying:

- **Protective Put** -- long put at each roll, settled at expiry
- **Collar** -- long put + short call, net premium can be positive

Uses 365 trading days for crypto assets. Tracks cumulative premiums,
intrinsic recovery, and daily mark-to-market via BS.

### Metrics (`metrics.py`)

From an `OptionsSimulationResult`, computes:

- Total / annualized return, Sharpe ratio, max drawdown
- Premium paid, received, net cost
- Intrinsic recovered at settlement
- Protection cost as % of initial value

### PDF Report (`report.py`)

Generates a 4-page PDF with:

1. Price history charts (BTC + IWM)
2. Realized volatility panels (YZ-14d + CC-21d)
3. **Pricing snapshot tables** -- Now / 1mo / 3mo / 6mo with change column
4. Options premium as % of spot (historical)
5. Absolute $ cost over time
6. BTC/IWM vol ratio
7. **Current strike grid** -- calls & puts at -10% to +10% for both assets
8. Key observations

## Data Sources

- **BTC/USD**: Alpaca `CryptoHistoricalDataClient` (requires `ALPACA_API_KEY`)
- **IWM**: Yahoo Finance via `yfinance` (no API key needed)
