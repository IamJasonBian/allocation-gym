# allocation-gym

Local-only Python package for backtesting allocation strategies, options analysis, and generating research reports.

## docs/7 — IWN Volatility Analysis

**Report**: `docs/7/iwn_vol_analysis.tex` → compiled to `docs/7/iwn_vol_analysis.pdf` via `tectonic`.

### What it covers

A systematic scalping strategy on Grayscale BTC Mini Trust with IWN put hedges:

1. **Executive Summary** — single-paragraph overview with live P&L (+$15,770 BTC, +$4,195 options)
2. **Current Positions & Thesis** — BTC Mini Trust characteristics, index weaknesses, rolling return correlations with ETFs, and **confounding factor analysis** (partial correlations controlling for VIX, DXY, 2Y/10Y yields at 3M/6M/12M)
3. **Realized Volatility & Hedge Selection** — 21d vol comparison with avg/max weekly move columns; explains IWN vs IWM choice (more options volume)
4. **Put Cost Comparison** — Black-Scholes ATM put costs across tenors
5. **IWN Options Chain** — Mar 4 snapshot data from option chain snapshots (not Netlify/blob references); put-call parity derived underlying, expiry breakdown, ATM Greeks
6. **Implied Volatility Smile** — theory, Mar 4 IV smile table by moneyness bucket, hedging implications
7. **Put Trading** — executed protective put round-trips with P&L; scalping regime description (bear: 0.5% buy/-1.25% stop-loss, 10-min rebalance; bull: 3% both sides, sells fund puts)
8. **Backtest & Performance** — supporting analysis (63d correlations), strategy backtests (Baseline/Path A/Path B), filled orders tables, portfolio snapshot
9. **Change Log** — at the bottom

### Data generation scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_vol_report.py` | Fetches IWN options chains, computes IV smile, generates charts + LaTeX tables |
| `scripts/confounding_analysis.py` | Partial correlation analysis (BTC↔IWN controlling for VIX, DXY, yields) |
| `scripts/generate_ytd_report.py` | Daily BTC/share prices + IWN ATM IV → YTD performance report (`docs/9/`) |
| `docs/7/thesis_backtest.py` | Full-history backtests (Baseline, Path A, Path B) + runtime API actuals |

### Key data files

- `docs/7/report_data.json` — cached Mar 4 options chain data (prices, IV smile, Greeks)
- `docs/7/confounding_data.json` — partial correlation results
- `docs/7/thesis_backtest_summary.json` — backtest metrics for all strategies
- `docs/7/iwn_iv_smile.png` — IV smile chart (also referenced in `docs/9/ytd_performance.tex`)
- `docs/7/iwn_iv_term_structure.png` — IV term structure chart

### Compilation

```bash
cd docs/7 && tectonic iwn_vol_analysis.tex
```

Do **not** use `pdflatex` — it is not installed. `tectonic` auto-downloads packages.

### Frontend link

The PDF is served at a stable URL from the `IamJasonBian/audit-redis-proto` branch. After updating on any working branch, also push the PDF there:

```bash
git checkout IamJasonBian/audit-redis-proto
git checkout <working-branch> -- docs/7/iwn_vol_analysis.pdf docs/7/iwn_vol_analysis.tex
git add docs/7/iwn_vol_analysis.pdf docs/7/iwn_vol_analysis.tex
git commit -m "Update IWN vol report"
git push
git checkout <working-branch>
```

## docs/9 — YTD Performance Report

`docs/9/ytd_performance.tex` — daily BTC/USD, BTC share prices, IWN ATM IV, options round-trips, combined P&L (+$19,965). Includes the IV smile chart from docs/7.

## Blob store access

Options chain and market quotes data comes from Netlify blob stores via `allocation_gym/blobs/client.py`. Requires env vars `ALLOC_ENGINE_SITE_ID` and `NETLIFY_AUTH_TOKEN`.
