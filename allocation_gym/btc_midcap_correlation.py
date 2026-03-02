#!/usr/bin/env python3
"""
BTC vs Midcap & Commodity Correlation Analysis
===============================================
Fetches ~200 midcap stocks and commodity ETFs, computes 3/6/12-month
rolling correlations with BTC, prices protective puts via Black-Scholes,
and ranks the top 10 reallocation candidates for a 25% portfolio shift
under both long-BTC and short-BTC thesis.

Outputs:
  - Console tables with top-10 picks per thesis/horizon
  - PNG charts: price performance, correlation heatmap, put cost comparison

Usage:
    python -m allocation_gym.btc_midcap_correlation
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance required: pip install yfinance")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
except ImportError:
    sys.exit("matplotlib required: pip install matplotlib")

# Black-Scholes from this project
from allocation_gym.options.black_scholes import bs_put_price

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Universe: ~200 midcap stocks + commodity / commodity-linked ETFs
# ---------------------------------------------------------------------------

MIDCAP_TICKERS = [
    # --- Commodity ETFs & Futures Proxies (30) ---
    "GLD", "SLV", "PPLT", "PALL",       # precious metals
    "USO", "BNO", "UNG", "UGA",          # energy
    "DBA", "DBC", "PDBC", "COMT",        # broad commodity
    "CORN", "WEAT", "SOYB", "CANE",      # agriculture
    "COPX", "REMX", "LIT", "URA",        # metals & mining
    "WOOD", "MOO", "COW", "NIB",         # soft/agriculture
    "GDX", "GDXJ", "SIL", "SILJ",       # miners
    "XME", "PICK",                        # metals & mining ETFs

    # --- Energy Midcaps (20) ---
    "DVN", "FANG", "OVV", "CTRA", "CHRD",
    "MRO", "APA", "SM", "MTDR", "NOV",
    "RRC", "AR", "CNX", "EQT", "SWN",
    "TRGP", "AM", "HESM", "WES", "DTM",

    # --- Materials / Mining Midcaps (20) ---
    "CLF", "X", "AA", "ATI", "STLD",
    "CMC", "RS", "WOR", "HAYN", "CRS",
    "MP", "LAC", "ALB", "LTHM", "SQM",
    "VALE", "RIO", "BHP", "SCCO", "TECK",

    # --- Industrials Midcaps (25) ---
    "GNRC", "RBC", "ESAB", "MWA", "ATKR",
    "UFPI", "BLDR", "PATK", "AWI", "TREX",
    "GGG", "RRX", "MIDD", "WTS", "CW",
    "TTC", "SWK", "ALLE", "MAS", "FBHS",
    "AZEK", "SITE", "POOL", "WSC", "APG",

    # --- Tech Midcaps (25) ---
    "NET", "CRWD", "ZS", "OKTA", "DDOG",
    "MDB", "ESTC", "PATH", "CFLT", "S",
    "GTLB", "DOCN", "BRZE", "DT", "NEWR",
    "RPD", "TENB", "QLYS", "VRNS", "CYBR",
    "BILL", "HUBS", "PCOR", "APPF", "NCNO",

    # --- Consumer Midcaps (20) ---
    "BROS", "SHAK", "CAVA", "TXRH", "EAT",
    "DIN", "JACK", "CAKE", "DENN", "PLAY",
    "SKX", "DECK", "CROX", "ONON", "BIRK",
    "LULU", "GPS", "ANF", "AEO", "URBN",

    # --- Healthcare Midcaps (20) ---
    "EXAS", "NTRA", "NVST", "INSP", "GMED",
    "TNDM", "PODD", "IRTC", "SILK", "AVTR",
    "MEDP", "ICLR", "CRL", "PRCT", "NVCR",
    "RVMD", "PCVX", "ROIV", "IONS", "SRPT",

    # --- Financial Midcaps (15) ---
    "COIN", "HOOD", "SOFI", "LPLA", "IBKR",
    "MKTX", "CBOE", "VIRT", "PIPR", "EVR",
    "RJF", "SF", "HLI", "PJT", "MC",

    # --- Real Assets / REITs (15) ---
    "AMT", "CCI", "SBAC", "EQIX", "DLR",
    "VNO", "SLG", "BXP", "KRC", "HIW",
    "IRM", "COLD", "REXR", "STAG", "FR",

    # --- Crypto-Adjacent (10) ---
    "MSTR", "MARA", "RIOT", "CLSK", "HUT",
    "CIFR", "IREN", "BTBT", "BITF", "WULF",

    # --- Index / Sector / Asset-Class ETFs (40) ---
    "SPY", "QQQ", "IWM", "DIA", "MDY",       # US broad
    "IWO", "IWN", "VXF", "IJH", "IWR",       # US style / midcap
    "VTV", "VUG",                              # value / growth
    "EFA", "EEM", "VWO", "FXI", "EWJ", "EWZ", # international
    "XLF", "XLK", "XLE", "XLV", "XLI",        # SPDR sectors
    "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC", # SPDR sectors cont.
    "TLT", "IEF", "HYG", "LQD",               # fixed income
    "ARKK", "SOXX",                            # thematic
    "KWEB", "INDA", "EWT", "EWY", "VNQ",      # intl / RE
]

RISK_FREE_RATE = 0.045  # 4.5% (T-bill proxy)
BTC_TICKER = "BTC-USD"
PORTFOLIO_VALUE = 1_000_000  # $1M notional
REALLOC_PCT = 0.25  # 25% reallocation
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_prices(tickers: List[str], period: str = "2y") -> Dict[str, np.ndarray]:
    """Fetch daily close prices for a list of tickers via yfinance."""
    all_tickers = [BTC_TICKER] + tickers
    print(f"Fetching {len(all_tickers)} tickers ({period} lookback)...")

    data = yf.download(
        all_tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # Handle both MultiIndex and single-column DataFrames
    if isinstance(data.columns, __import__("pandas").MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    prices = {}
    for ticker in all_tickers:
        col = ticker
        if col in closes.columns:
            series = closes[col].dropna()
            if len(series) >= 60:  # need at least ~3 months
                prices[ticker] = series
            else:
                pass  # skip tickers with insufficient data
    print(f"  Got valid data for {len(prices)} tickers")
    return prices


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


def rolling_corr(ret_a: np.ndarray, ret_b: np.ndarray, window: int) -> float:
    """Trailing correlation over the last `window` trading days."""
    n = min(len(ret_a), len(ret_b))
    if n < window:
        return np.nan
    a = ret_a[-window:]
    b = ret_b[-window:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def realized_vol(returns: np.ndarray, window: int = 30) -> float:
    """Annualised realized vol from trailing daily returns."""
    n = min(len(returns), window)
    if n < 10:
        return 0.0
    r = returns[-n:]
    return float(np.std(r) * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Scoring & ranking
# ---------------------------------------------------------------------------

def score_long_btc(corr_3m: float, corr_6m: float, corr_12m: float,
                   put_cost_pct: float, vol: float) -> float:
    """
    Long-BTC thesis: want NEGATIVE correlation (hedge) + cheap puts.
    Score higher = better reallocation candidate.
    """
    # Weight: heavier on shorter-term correlation for tactical moves
    avg_corr = 0.5 * corr_3m + 0.3 * corr_6m + 0.2 * corr_12m
    # Negative corr is good → invert sign
    corr_score = -avg_corr
    # Cheaper puts are better (normalized, cap at 20%)
    cost_score = max(0, 1.0 - put_cost_pct / 0.20)
    # Moderate vol preferred (not too low, not too high)
    vol_score = 1.0 - abs(vol - 0.30) / 0.50
    vol_score = max(0, vol_score)
    return 0.50 * corr_score + 0.30 * cost_score + 0.20 * vol_score


def score_short_btc(corr_3m: float, corr_6m: float, corr_12m: float,
                    put_cost_pct: float, vol: float) -> float:
    """
    Short-BTC thesis: want POSITIVE correlation (move together when BTC drops)
    + cheap puts for downside protection on the alt.
    """
    avg_corr = 0.5 * corr_3m + 0.3 * corr_6m + 0.2 * corr_12m
    corr_score = avg_corr  # positive corr is good here
    cost_score = max(0, 1.0 - put_cost_pct / 0.20)
    vol_score = 1.0 - abs(vol - 0.35) / 0.50
    vol_score = max(0, vol_score)
    return 0.50 * corr_score + 0.30 * cost_score + 0.20 * vol_score


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "6")
    os.makedirs(docs_dir, exist_ok=True)

    # Fetch data
    prices = fetch_prices(MIDCAP_TICKERS, period="2y")

    if BTC_TICKER not in prices:
        sys.exit("ERROR: Could not fetch BTC-USD data")

    btc_prices = prices[BTC_TICKER]
    btc_returns = compute_log_returns(btc_prices.values)
    btc_spot = float(btc_prices.iloc[-1])
    btc_vol = realized_vol(btc_returns, window=30)

    print(f"\nBTC spot: ${btc_spot:,.0f}  |  30d realized vol: {btc_vol:.1%}")
    print(f"Analyzing {len(prices) - 1} assets across 3mo / 6mo / 12mo windows\n")

    # Windows in trading days
    windows = {"3mo": 63, "6mo": 126, "12mo": 252}

    results = []

    for ticker, price_series in prices.items():
        if ticker == BTC_TICKER:
            continue
        try:
            px = price_series.values
            rets = compute_log_returns(px)
            spot = float(price_series.iloc[-1])
            vol_30d = realized_vol(rets, window=30)

            # Correlations at each horizon
            corrs = {}
            for label, w in windows.items():
                corrs[label] = rolling_corr(btc_returns, rets, w)

            # Skip if we couldn't compute correlations
            if any(np.isnan(v) for v in corrs.values()):
                continue

            # Put pricing: ATM put, 3mo expiry, using 30d realized vol
            put_3m = bs_put_price(spot, spot, 0.25, RISK_FREE_RATE, vol_30d)
            put_6m = bs_put_price(spot, spot, 0.50, RISK_FREE_RATE, vol_30d)
            put_12m = bs_put_price(spot, spot, 1.00, RISK_FREE_RATE, vol_30d)

            # Put cost as % of spot
            put_pct_3m = put_3m / spot if spot > 0 else 0
            put_pct_6m = put_6m / spot if spot > 0 else 0
            put_pct_12m = put_12m / spot if spot > 0 else 0

            # 25% reallocation notional
            alloc_notional = PORTFOLIO_VALUE * REALLOC_PCT
            shares = alloc_notional / spot if spot > 0 else 0

            # Protective put cost for 25% allocation
            hedge_cost_3m = put_3m * shares
            hedge_cost_6m = put_6m * shares
            hedge_cost_12m = put_12m * shares

            # YTD return
            ytd_start_idx = 0
            for i, dt in enumerate(price_series.index):
                if dt.year == datetime.now().year:
                    ytd_start_idx = i
                    break
            ytd_ret = (spot / float(price_series.iloc[ytd_start_idx]) - 1) if ytd_start_idx > 0 else 0

            # 1Y return
            one_yr_ret = (spot / float(px[0]) - 1) if len(px) > 252 else (spot / float(px[0]) - 1)

            # Scoring
            s_long = score_long_btc(corrs["3mo"], corrs["6mo"], corrs["12mo"], put_pct_3m, vol_30d)
            s_short = score_short_btc(corrs["3mo"], corrs["6mo"], corrs["12mo"], put_pct_3m, vol_30d)

            results.append({
                "ticker": ticker,
                "spot": spot,
                "vol_30d": vol_30d,
                "corr_3m": corrs["3mo"],
                "corr_6m": corrs["6mo"],
                "corr_12m": corrs["12mo"],
                "put_pct_3m": put_pct_3m,
                "put_pct_6m": put_pct_6m,
                "put_pct_12m": put_pct_12m,
                "hedge_cost_3m": hedge_cost_3m,
                "hedge_cost_6m": hedge_cost_6m,
                "hedge_cost_12m": hedge_cost_12m,
                "ytd_ret": ytd_ret,
                "one_yr_ret": one_yr_ret,
                "score_long": s_long,
                "score_short": s_short,
                "price_series": price_series,
            })
        except Exception:
            continue

    if not results:
        sys.exit("ERROR: No valid results computed")

    print(f"Successfully analyzed {len(results)} assets\n")

    # Sort and pick top 10
    top_long = sorted(results, key=lambda x: x["score_long"], reverse=True)[:10]
    top_short = sorted(results, key=lambda x: x["score_short"], reverse=True)[:10]

    # ---------------------------------------------------------------------------
    # Console output
    # ---------------------------------------------------------------------------
    alloc_notional = PORTFOLIO_VALUE * REALLOC_PCT

    def print_table(title: str, rows: list, thesis: str):
        print("=" * 120)
        print(f"  {title}")
        print(f"  Portfolio: ${PORTFOLIO_VALUE:,.0f}  |  Reallocation: {REALLOC_PCT:.0%} (${alloc_notional:,.0f})  |  BTC: ${btc_spot:,.0f}")
        print("=" * 120)
        hdr = (
            f"{'Rank':<5} {'Ticker':<8} {'Spot':>10} {'30d Vol':>8} "
            f"{'Corr 3M':>8} {'Corr 6M':>8} {'Corr 12M':>9} "
            f"{'Put 3M%':>8} {'Put 6M%':>8} {'Put 12M%':>9} "
            f"{'Hedge$ 3M':>11} {'1Y Ret':>8} {'Score':>7}"
        )
        print(hdr)
        print("-" * 120)
        for i, r in enumerate(rows):
            score_key = "score_long" if thesis == "long" else "score_short"
            print(
                f"{i + 1:<5} {r['ticker']:<8} ${r['spot']:>8,.2f} {r['vol_30d']:>7.1%} "
                f"{r['corr_3m']:>+8.3f} {r['corr_6m']:>+8.3f} {r['corr_12m']:>+9.3f} "
                f"{r['put_pct_3m']:>7.1%} {r['put_pct_6m']:>7.1%} {r['put_pct_12m']:>8.1%} "
                f"${r['hedge_cost_3m']:>9,.0f} {r['one_yr_ret']:>+7.1%} {r[score_key]:>7.3f}"
            )
        print()

    print_table(
        "TOP 10 REALLOCATION CANDIDATES — BTC LONG THESIS (hedge with negative correlation)",
        top_long, "long"
    )
    print_table(
        "TOP 10 REALLOCATION CANDIDATES — BTC SHORT THESIS (correlated downside plays)",
        top_short, "short"
    )

    # ---------------------------------------------------------------------------
    # Detailed per-horizon tables
    # ---------------------------------------------------------------------------
    for horizon, window_key in [("3-Month", "3m"), ("6-Month", "6m"), ("12-Month", "12m")]:
        corr_key = f"corr_{window_key}"
        put_key = f"put_pct_{window_key}"
        hedge_key = f"hedge_cost_{window_key}"

        # Sort by absolute correlation descending
        by_corr = sorted(results, key=lambda x: abs(x[corr_key]), reverse=True)[:15]

        print(f"\n{'─' * 90}")
        print(f"  {horizon} Horizon — Top 15 by Absolute Correlation with BTC")
        print(f"{'─' * 90}")
        print(f"{'Rank':<5} {'Ticker':<8} {'Correlation':>12} {'Direction':>10} {'ATM Put%':>10} {'Hedge Cost':>12} {'30d Vol':>8}")
        print("-" * 90)
        for i, r in enumerate(by_corr):
            direction = "POSITIVE" if r[corr_key] > 0 else "NEGATIVE"
            print(
                f"{i + 1:<5} {r['ticker']:<8} {r[corr_key]:>+12.4f} {direction:>10} "
                f"{r[put_key]:>9.2%} ${r[hedge_key]:>10,.0f} {r['vol_30d']:>7.1%}"
            )

    # ---------------------------------------------------------------------------
    # CSV exports
    # ---------------------------------------------------------------------------
    print("\nExporting CSV data...")
    _export_csvs(results, top_long, top_short, btc_spot, btc_vol, docs_dir)

    # ---------------------------------------------------------------------------
    # IWN & CRWD deep-dive: put options + portfolio sizing
    # ---------------------------------------------------------------------------
    print("\n\nRunning IWN & CRWD put options / portfolio sizing deep-dive...")
    _deep_dive_put_sizing(
        focus_tickers=["IWN", "CRWD"],
        results=results,
        prices=prices,
        btc_prices=btc_prices,
        btc_returns=btc_returns,
        btc_spot=btc_spot,
        btc_vol=btc_vol,
        docs_dir=docs_dir,
    )

    # ---------------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------------
    print("\nGenerating charts...")

    # Chart 1: Top-10 Long Thesis — Price Performance
    _plot_price_chart(top_long, btc_prices, "BTC Long Thesis — Top 10 Reallocation Candidates",
                      os.path.join(docs_dir, "btc_long_thesis_prices.png"))

    # Chart 2: Top-10 Short Thesis — Price Performance
    _plot_price_chart(top_short, btc_prices, "BTC Short Thesis — Top 10 Reallocation Candidates",
                      os.path.join(docs_dir, "btc_short_thesis_prices.png"))

    # Chart 3: Correlation heatmap for all top picks
    _plot_correlation_heatmap(top_long, top_short, btc_returns,
                              os.path.join(docs_dir, "btc_correlation_heatmap.png"))

    # Chart 4: Put cost comparison
    _plot_put_cost_comparison(top_long, top_short,
                              os.path.join(docs_dir, "btc_put_cost_comparison.png"))

    # Chart 5: Comprehensive dashboard
    _plot_dashboard(top_long, top_short, results, btc_prices, btc_spot, btc_vol,
                    os.path.join(docs_dir, "btc_reallocation_dashboard.png"))

    print("\nDone! All outputs saved to docs/6/")


# ---------------------------------------------------------------------------
# IWN & CRWD deep-dive: put options + portfolio sizing
# ---------------------------------------------------------------------------

def _deep_dive_put_sizing(
    focus_tickers: List[str],
    results: list,
    prices: Dict,
    btc_prices,
    btc_returns: np.ndarray,
    btc_spot: float,
    btc_vol: float,
    docs_dir: str,
):
    """Full put-option strike ladder and portfolio sizing for focus assets."""
    import csv
    from allocation_gym.options.black_scholes import bs_put_price as _bs_put

    result_map = {r["ticker"]: r for r in results}

    # Strike offsets: ATM, 5% OTM, 10% OTM, 15% OTM, 20% OTM
    strike_offsets = [1.00, 0.95, 0.90, 0.85, 0.80]
    tenors = [("1mo", 1 / 12), ("3mo", 0.25), ("6mo", 0.50), ("9mo", 0.75), ("12mo", 1.00)]

    # Portfolio sizing: what % of $1M to allocate at different hedge budgets
    hedge_budgets_pct = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]  # 0.5% to 5% of portfolio

    all_rows = []  # for consolidated CSV

    for ticker in focus_tickers:
        if ticker not in result_map:
            print(f"  WARNING: {ticker} not in results, skipping")
            continue

        r = result_map[ticker]
        spot = r["spot"]
        vol = r["vol_30d"]
        corr_3m = r["corr_3m"]
        corr_6m = r["corr_6m"]
        corr_12m = r["corr_12m"]

        print(f"\n{'=' * 100}")
        print(f"  {ticker} DEEP DIVE — Put Options & Portfolio Sizing")
        print(f"  Spot: ${spot:,.2f}  |  30d Vol: {vol:.1%}  |  BTC Corr: 3M={corr_3m:+.3f}  6M={corr_6m:+.3f}  12M={corr_12m:+.3f}")
        print(f"{'=' * 100}")

        # ---- Strike Ladder ----
        print(f"\n  {'Strike Ladder — ATM Put Pricing':^80}")
        print(f"  {'':>10}", end="")
        for label, _ in tenors:
            print(f"{'':>4}{label:>8} {'($/sh)':>8}", end="")
        print()

        hdr = f"  {'Strike':>10}"
        for _ in tenors:
            hdr += f"  {'%Spot':>8} {'Price':>8}"
        print(hdr)
        print("  " + "-" * (10 + len(tenors) * 18))

        ladder_rows = []
        for offset in strike_offsets:
            strike = spot * offset
            moneyness_label = "ATM" if offset == 1.0 else f"{(1 - offset) * 100:.0f}% OTM"
            row_str = f"  ${strike:>8,.2f}"
            row_data = {"ticker": ticker, "strike": strike, "moneyness": moneyness_label}
            for label, T in tenors:
                price = _bs_put(spot, strike, T, RISK_FREE_RATE, vol)
                pct = price / spot * 100
                row_str += f"  {pct:>7.2f}% ${price:>7.2f}"
                row_data[f"put_{label}_pct"] = pct / 100
                row_data[f"put_{label}_price"] = price
            print(row_str + f"  ({moneyness_label})")
            ladder_rows.append(row_data)

        # ---- Breakeven Analysis ----
        print(f"\n  {'Breakeven Levels (spot must fall below this to profit on put)':^80}")
        print(f"  {'Strike':>10}", end="")
        for label, _ in tenors:
            print(f"  {label + ' BE':>10}", end="")
        print()
        print("  " + "-" * (10 + len(tenors) * 12))

        for offset in strike_offsets:
            strike = spot * offset
            moneyness_label = "ATM" if offset == 1.0 else f"{(1 - offset) * 100:.0f}% OTM"
            row_str = f"  ${strike:>8,.2f}"
            for label, T in tenors:
                premium = _bs_put(spot, strike, T, RISK_FREE_RATE, vol)
                breakeven = strike - premium
                be_pct = (breakeven / spot - 1) * 100
                row_str += f"  {be_pct:>+9.1f}%"
            print(row_str + f"  ({moneyness_label})")

        # ---- Portfolio Sizing ----
        print(f"\n  {'Portfolio Sizing — Shares & Puts for $1M Portfolio':^80}")
        print(f"  {'Alloc%':>8} {'Notional':>12} {'Shares':>10} {'ATM 3M Put':>12} {'Hedge Cost':>12} {'Hedge/Port':>12} {'Net Exp':>12}")
        print("  " + "-" * 82)

        sizing_rows = []
        for alloc_pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            notional = PORTFOLIO_VALUE * alloc_pct
            shares = notional / spot
            atm_put = _bs_put(spot, spot, 0.25, RISK_FREE_RATE, vol)
            hedge_cost = atm_put * shares
            hedge_port_pct = hedge_cost / PORTFOLIO_VALUE
            net_exposure = notional - hedge_cost
            print(
                f"  {alloc_pct:>7.0%} ${notional:>10,.0f} {shares:>10,.1f} "
                f"${atm_put:>10,.2f} ${hedge_cost:>10,.0f} {hedge_port_pct:>11.2%} ${net_exposure:>10,.0f}"
            )
            sizing_rows.append({
                "ticker": ticker, "alloc_pct": alloc_pct, "notional": notional,
                "shares": shares, "atm_3m_put": atm_put, "hedge_cost": hedge_cost,
                "hedge_port_pct": hedge_port_pct, "net_exposure": net_exposure,
            })

        # ---- Budget-constrained sizing ----
        print(f"\n  {'Budget-Constrained Sizing — Max allocation given hedge budget':^80}")
        print(f"  {'Budget':>10} {'Budget $':>10} {'Max Alloc%':>12} {'Max Notional':>14} {'Shares':>10} {'Hedged?':>8}")
        print("  " + "-" * 68)

        budget_rows = []
        for budget_pct in hedge_budgets_pct:
            budget_dollar = PORTFOLIO_VALUE * budget_pct
            atm_put = _bs_put(spot, spot, 0.25, RISK_FREE_RATE, vol)
            put_per_share = atm_put
            max_shares = budget_dollar / put_per_share if put_per_share > 0 else 0
            max_notional = max_shares * spot
            max_alloc = max_notional / PORTFOLIO_VALUE
            print(
                f"  {budget_pct:>9.1%} ${budget_dollar:>8,.0f} {max_alloc:>11.1%} "
                f"${max_notional:>12,.0f} {max_shares:>10,.1f} {'YES':>8}"
            )
            budget_rows.append({
                "ticker": ticker, "hedge_budget_pct": budget_pct,
                "budget_dollar": budget_dollar, "max_alloc_pct": max_alloc,
                "max_notional": max_notional, "max_shares": max_shares,
            })

        # ---- BTC correlation-adjusted sizing recommendation ----
        print(f"\n  {'Correlation-Adjusted Recommendation':^80}")
        abs_corr = abs(corr_3m)
        if corr_3m < -0.05:
            thesis = "HEDGE (negative BTC correlation)"
            # Negative corr = good diversifier, size up
            rec_alloc = min(0.25 + abs_corr * 0.5, 0.40)
        elif corr_3m > 0.15:
            thesis = "CORRELATED (positive BTC correlation)"
            # Positive corr = amplifies BTC risk, size down or hedge fully
            rec_alloc = max(0.25 - abs_corr * 0.3, 0.10)
        else:
            thesis = "NEUTRAL (low BTC correlation)"
            rec_alloc = 0.20
        rec_notional = PORTFOLIO_VALUE * rec_alloc
        rec_shares = rec_notional / spot
        rec_put = _bs_put(spot, spot, 0.25, RISK_FREE_RATE, vol)
        rec_hedge = rec_put * rec_shares

        print(f"  Thesis:            {thesis}")
        print(f"  3M BTC Corr:       {corr_3m:+.3f}")
        print(f"  Recommended Alloc: {rec_alloc:.0%} (${rec_notional:,.0f})")
        print(f"  Shares:            {rec_shares:,.1f}")
        print(f"  3M ATM Put Hedge:  ${rec_hedge:,.0f} ({rec_hedge / PORTFOLIO_VALUE:.2%} of portfolio)")

        # ---- Export per-ticker CSV ----
        csv_path = os.path.join(docs_dir, f"btc_{ticker.lower()}_put_sizing.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)

            # Section 1: Summary
            w.writerow(["# SUMMARY"])
            w.writerow(["ticker", "spot", "vol_30d", "corr_3m", "corr_6m", "corr_12m", "thesis", "rec_alloc_pct"])
            w.writerow([ticker, f"{spot:.4f}", f"{vol:.6f}", f"{corr_3m:.6f}", f"{corr_6m:.6f}", f"{corr_12m:.6f}", thesis, f"{rec_alloc:.4f}"])
            w.writerow([])

            # Section 2: Strike ladder
            w.writerow(["# STRIKE LADDER"])
            tenor_labels = [t[0] for t in tenors]
            header = ["strike", "moneyness"]
            for t in tenor_labels:
                header += [f"put_{t}_pct", f"put_{t}_price"]
            w.writerow(header)
            for lr in ladder_rows:
                row = [f"{lr['strike']:.4f}", lr["moneyness"]]
                for t in tenor_labels:
                    row += [f"{lr[f'put_{t}_pct']:.6f}", f"{lr[f'put_{t}_price']:.4f}"]
                w.writerow(row)
            w.writerow([])

            # Section 3: Portfolio sizing
            w.writerow(["# PORTFOLIO SIZING"])
            w.writerow(["alloc_pct", "notional", "shares", "atm_3m_put", "hedge_cost", "hedge_port_pct", "net_exposure"])
            for sr in sizing_rows:
                w.writerow([
                    f"{sr['alloc_pct']:.4f}", f"{sr['notional']:.2f}", f"{sr['shares']:.4f}",
                    f"{sr['atm_3m_put']:.4f}", f"{sr['hedge_cost']:.2f}",
                    f"{sr['hedge_port_pct']:.6f}", f"{sr['net_exposure']:.2f}",
                ])
            w.writerow([])

            # Section 4: Budget-constrained
            w.writerow(["# BUDGET-CONSTRAINED SIZING"])
            w.writerow(["hedge_budget_pct", "budget_dollar", "max_alloc_pct", "max_notional", "max_shares"])
            for br in budget_rows:
                w.writerow([
                    f"{br['hedge_budget_pct']:.4f}", f"{br['budget_dollar']:.2f}",
                    f"{br['max_alloc_pct']:.6f}", f"{br['max_notional']:.2f}", f"{br['max_shares']:.4f}",
                ])

        print(f"\n  -> {csv_path}")

        # Collect for combined chart
        all_rows.append({
            "ticker": ticker, "spot": spot, "vol": vol,
            "corr_3m": corr_3m, "corr_6m": corr_6m, "corr_12m": corr_12m,
            "ladder": ladder_rows, "sizing": sizing_rows, "budget": budget_rows,
            "rec_alloc": rec_alloc, "rec_hedge": rec_hedge, "thesis": thesis,
            "price_series": prices.get(ticker),
        })

    # Generate combined chart
    if all_rows:
        _plot_deep_dive(all_rows, btc_prices, btc_spot, docs_dir)


def _plot_deep_dive(focus_data: list, btc_prices, btc_spot: float, docs_dir: str):
    """4-panel chart for IWN/CRWD deep-dive."""
    n = len(focus_data)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("IWN & CRWD — Put Options & Portfolio Sizing Deep Dive",
                 fontsize=18, fontweight="bold", y=0.98)

    colors_map = {"IWN": "#1565C0", "CRWD": "#C62828"}
    default_colors = ["#1565C0", "#C62828", "#2E7D32", "#F57F17"]

    # Panel 1: Strike ladder heatmap — put cost % by tenor and strike
    ax = axes[0, 0]
    tenors_labels = ["1mo", "3mo", "6mo", "9mo", "12mo"]
    strike_labels = ["ATM", "5% OTM", "10% OTM", "15% OTM", "20% OTM"]

    bar_width = 0.35
    for idx, fd in enumerate(focus_data):
        tk = fd["ticker"]
        color = colors_map.get(tk, default_colors[idx])
        # Show 3mo put cost across strikes
        costs = [lr[f"put_3mo_pct"] * 100 for lr in fd["ladder"]]
        x = np.arange(len(strike_labels))
        offset = -bar_width / 2 + idx * bar_width
        bars = ax.bar(x + offset, costs, bar_width * 0.9, label=tk, color=color, alpha=0.8)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{cost:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(np.arange(len(strike_labels)))
    ax.set_xticklabels(strike_labels, fontsize=12)
    ax.set_ylabel("3-Month Put Cost (% of Spot)", fontsize=13)
    ax.set_title("Put Cost by Strike", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Put cost across tenors (ATM only)
    ax = axes[0, 1]
    for idx, fd in enumerate(focus_data):
        tk = fd["ticker"]
        color = colors_map.get(tk, default_colors[idx])
        atm_row = fd["ladder"][0]  # ATM
        costs = [atm_row[f"put_{t}_pct"] * 100 for t in tenors_labels]
        ax.plot(tenors_labels, costs, marker="o", markersize=10, linewidth=2.5,
                label=f"{tk} (\\${fd['spot']:,.0f})", color=color)
        for i, c in enumerate(costs):
            ax.annotate(f"{c:.1f}%", (tenors_labels[i], c),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=10, ha="center", color=color, fontweight="bold")

    ax.set_ylabel("ATM Put Cost (% of Spot)", fontsize=13)
    ax.set_xlabel("Tenor", fontsize=13)
    ax.set_title("ATM Put Term Structure", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: Portfolio sizing — hedge cost vs allocation
    ax = axes[1, 0]
    for idx, fd in enumerate(focus_data):
        tk = fd["ticker"]
        color = colors_map.get(tk, default_colors[idx])
        allocs = [s["alloc_pct"] * 100 for s in fd["sizing"]]
        hedges = [s["hedge_port_pct"] * 100 for s in fd["sizing"]]
        ax.plot(allocs, hedges, marker="s", markersize=8, linewidth=2.5,
                label=f"{tk} (vol={fd['vol']:.0%})", color=color)
        # Mark recommended allocation
        rec_x = fd["rec_alloc"] * 100
        rec_y = fd["rec_hedge"] / PORTFOLIO_VALUE * 100
        ax.scatter([rec_x], [rec_y], s=200, color=color, edgecolors="black",
                   linewidths=2, zorder=10, marker="*")
        ax.annotate(f"Rec: {rec_x:.0f}%", (rec_x, rec_y),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("Portfolio Allocation (%)", fontsize=13)
    ax.set_ylabel("Hedge Cost (% of Portfolio)", fontsize=13)
    ax.set_title("Allocation vs Hedge Cost (3M ATM Put)", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: Normalized price vs BTC
    ax = axes[1, 1]
    btc_norm = (btc_prices / btc_prices.iloc[0]) * 100
    ax.plot(btc_norm.index, btc_norm.values, color="orange", linewidth=2.5,
            label=f"BTC (\\${btc_spot:,.0f})", zorder=10)
    for idx, fd in enumerate(focus_data):
        tk = fd["ticker"]
        color = colors_map.get(tk, default_colors[idx])
        ps = fd["price_series"]
        if ps is not None and len(ps) > 10:
            common_start = max(ps.index[0], btc_prices.index[0])
            ps_a = ps[ps.index >= common_start]
            norm = (ps_a / ps_a.iloc[0]) * 100
            ax.plot(norm.index, norm.values, linewidth=2, color=color, alpha=0.85,
                    label=f"{tk} ({chr(961)}={fd['corr_3m']:+.2f})")

    ax.set_ylabel("Normalized Price (Start=100)", fontsize=13)
    ax.set_title("Price Performance vs BTC", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(docs_dir, "btc_iwn_crwd_deep_dive.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {out_path}")


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def _export_csvs(results: list, top_long: list, top_short: list,
                 btc_spot: float, btc_vol: float, docs_dir: str):
    """Write all analysis data to CSV files."""
    import csv

    # 1. Full universe — all assets with correlations, vol, put pricing, scores
    path_all = os.path.join(docs_dir, "btc_correlation_all_assets.csv")
    sorted_all = sorted(results, key=lambda x: abs(x["corr_3m"]), reverse=True)
    with open(path_all, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker", "spot", "vol_30d",
            "corr_3m", "corr_6m", "corr_12m",
            "put_pct_3m", "put_pct_6m", "put_pct_12m",
            "hedge_cost_3m", "hedge_cost_6m", "hedge_cost_12m",
            "ytd_return", "1y_return",
            "score_long", "score_short",
        ])
        for r in sorted_all:
            w.writerow([
                r["ticker"], f"{r['spot']:.4f}", f"{r['vol_30d']:.6f}",
                f"{r['corr_3m']:.6f}", f"{r['corr_6m']:.6f}", f"{r['corr_12m']:.6f}",
                f"{r['put_pct_3m']:.6f}", f"{r['put_pct_6m']:.6f}", f"{r['put_pct_12m']:.6f}",
                f"{r['hedge_cost_3m']:.2f}", f"{r['hedge_cost_6m']:.2f}", f"{r['hedge_cost_12m']:.2f}",
                f"{r['ytd_ret']:.6f}", f"{r['one_yr_ret']:.6f}",
                f"{r['score_long']:.6f}", f"{r['score_short']:.6f}",
            ])
    print(f"  -> {path_all}")

    # 2. Top-10 long thesis
    path_long = os.path.join(docs_dir, "btc_top10_long_thesis.csv")
    with open(path_long, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "ticker", "spot", "vol_30d",
            "corr_3m", "corr_6m", "corr_12m",
            "put_pct_3m", "put_pct_6m", "put_pct_12m",
            "hedge_cost_3m", "1y_return", "score_long",
        ])
        for i, r in enumerate(top_long):
            w.writerow([
                i + 1, r["ticker"], f"{r['spot']:.4f}", f"{r['vol_30d']:.6f}",
                f"{r['corr_3m']:.6f}", f"{r['corr_6m']:.6f}", f"{r['corr_12m']:.6f}",
                f"{r['put_pct_3m']:.6f}", f"{r['put_pct_6m']:.6f}", f"{r['put_pct_12m']:.6f}",
                f"{r['hedge_cost_3m']:.2f}", f"{r['one_yr_ret']:.6f}",
                f"{r['score_long']:.6f}",
            ])
    print(f"  -> {path_long}")

    # 3. Top-10 short thesis
    path_short = os.path.join(docs_dir, "btc_top10_short_thesis.csv")
    with open(path_short, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "ticker", "spot", "vol_30d",
            "corr_3m", "corr_6m", "corr_12m",
            "put_pct_3m", "put_pct_6m", "put_pct_12m",
            "hedge_cost_3m", "1y_return", "score_short",
        ])
        for i, r in enumerate(top_short):
            w.writerow([
                i + 1, r["ticker"], f"{r['spot']:.4f}", f"{r['vol_30d']:.6f}",
                f"{r['corr_3m']:.6f}", f"{r['corr_6m']:.6f}", f"{r['corr_12m']:.6f}",
                f"{r['put_pct_3m']:.6f}", f"{r['put_pct_6m']:.6f}", f"{r['put_pct_12m']:.6f}",
                f"{r['hedge_cost_3m']:.2f}", f"{r['one_yr_ret']:.6f}",
                f"{r['score_short']:.6f}",
            ])
    print(f"  -> {path_short}")

    # 4. Metadata / run info
    from datetime import datetime as _dt
    path_meta = os.path.join(docs_dir, "btc_correlation_metadata.csv")
    with open(path_meta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        w.writerow(["run_date", _dt.now().strftime("%Y-%m-%d %H:%M")])
        w.writerow(["btc_spot", f"{btc_spot:.2f}"])
        w.writerow(["btc_30d_vol", f"{btc_vol:.6f}"])
        w.writerow(["universe_size", len(results)])
        w.writerow(["portfolio_value", PORTFOLIO_VALUE])
        w.writerow(["reallocation_pct", REALLOC_PCT])
        w.writerow(["risk_free_rate", RISK_FREE_RATE])
        w.writerow(["lookback_period", "2y"])
        w.writerow(["correlation_windows", "63d / 126d / 252d"])
    print(f"  -> {path_meta}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_price_chart(top10: list, btc_prices, title: str, out_path: str):
    """Normalized price chart for top-10 candidates vs BTC."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), height_ratios=[2, 1])
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Normalize all to start at 100
    btc_norm = (btc_prices / btc_prices.iloc[0]) * 100
    ax1.plot(btc_norm.index, btc_norm.values, color="orange", linewidth=2.5,
             label=f"BTC (\\${float(btc_prices.iloc[-1]):,.0f})", zorder=10, alpha=0.9)

    for i, r in enumerate(top10):
        ps = r["price_series"]
        # Align to BTC date range
        common_start = max(ps.index[0], btc_prices.index[0])
        ps_aligned = ps[ps.index >= common_start]
        if len(ps_aligned) < 10:
            continue
        norm = (ps_aligned / ps_aligned.iloc[0]) * 100
        ax1.plot(norm.index, norm.values, color=colors[i], linewidth=1.3, alpha=0.8,
                 label=f"{r['ticker']} ({chr(961)}={r['corr_3m']:+.2f})")

    ax1.set_ylabel("Normalized Price (Start = 100)", fontsize=14)
    ax1.legend(loc="upper left", fontsize=11, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Price Performance (Normalized)", fontsize=15)
    ax1.tick_params(axis="both", labelsize=12)

    # Bottom panel: correlation bars by horizon
    tickers = [r["ticker"] for r in top10]
    x = np.arange(len(tickers))
    width = 0.25

    bars_3m = [r["corr_3m"] for r in top10]
    bars_6m = [r["corr_6m"] for r in top10]
    bars_12m = [r["corr_12m"] for r in top10]

    ax2.bar(x - width, bars_3m, width, label="3-Month", color="#2196F3", alpha=0.8)
    ax2.bar(x, bars_6m, width, label="6-Month", color="#FF9800", alpha=0.8)
    ax2.bar(x + width, bars_12m, width, label="12-Month", color="#4CAF50", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45, ha="right", fontsize=12)
    ax2.set_ylabel("Correlation with BTC", fontsize=14)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title("BTC Correlation by Horizon", fontsize=15)
    ax2.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def _plot_correlation_heatmap(top_long: list, top_short: list,
                              btc_returns: np.ndarray, out_path: str):
    """Correlation heatmap across all unique top picks."""
    # Combine unique tickers
    seen = set()
    all_picks = []
    for r in top_long + top_short:
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            all_picks.append(r)

    n = len(all_picks)
    if n < 2:
        return

    # Compute pairwise correlations using 3-month window
    corr_matrix = np.zeros((n, n))
    returns_map = {}
    for r in all_picks:
        returns_map[r["ticker"]] = compute_log_returns(r["price_series"].values)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                ri = returns_map[all_picks[i]["ticker"]]
                rj = returns_map[all_picks[j]["ticker"]]
                min_len = min(len(ri), len(rj))
                if min_len > 20:
                    corr_matrix[i, j] = float(
                        np.corrcoef(ri[-min_len:], rj[-min_len:])[0, 1]
                    )

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    tickers = [r["ticker"] for r in all_picks]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=90, fontsize=11)
    ax.set_yticklabels(tickers, fontsize=11)

    # Add correlation values
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Correlation", fontsize=13)
    ax.set_title("Pairwise Correlation Matrix — Top Reallocation Candidates",
                 fontsize=17, fontweight="bold", pad=20)

    # Mark long vs short picks
    for i, r in enumerate(all_picks):
        in_long = r["ticker"] in {x["ticker"] for x in top_long}
        in_short = r["ticker"] in {x["ticker"] for x in top_short}
        if in_long and in_short:
            marker = " [L+S]"
        elif in_long:
            marker = " [L]"
        else:
            marker = " [S]"
        current_label = ax.get_yticklabels()[i].get_text()
        ax.get_yticklabels()[i].set_text(current_label + marker)

    ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def _plot_put_cost_comparison(top_long: list, top_short: list, out_path: str):
    """Put cost bar chart comparison across horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("ATM Protective Put Cost (% of Spot) — 25% Reallocation",
                 fontsize=18, fontweight="bold")

    for ax, picks, label in [(axes[0], top_long, "Long BTC Thesis"),
                              (axes[1], top_short, "Short BTC Thesis")]:
        tickers = [r["ticker"] for r in picks]
        x = np.arange(len(tickers))
        width = 0.25

        costs_3m = [r["put_pct_3m"] * 100 for r in picks]
        costs_6m = [r["put_pct_6m"] * 100 for r in picks]
        costs_12m = [r["put_pct_12m"] * 100 for r in picks]

        ax.bar(x - width, costs_3m, width, label="3-Month Put", color="#E91E63", alpha=0.8)
        ax.bar(x, costs_6m, width, label="6-Month Put", color="#9C27B0", alpha=0.8)
        ax.bar(x + width, costs_12m, width, label="12-Month Put", color="#3F51B5", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=13)
        ax.set_ylabel("ATM Put Cost (% of Spot)", fontsize=14)
        ax.set_title(label, fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=12)

        # Annotate hedge cost in $ for 3mo puts
        alloc = PORTFOLIO_VALUE * REALLOC_PCT
        for i, r in enumerate(picks):
            cost_dollar = r["hedge_cost_3m"]
            ax.annotate(f"\\${cost_dollar:,.0f}", xy=(x[i] - width, costs_3m[i]),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=9, ha="center", color="#E91E63")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def _plot_dashboard(top_long: list, top_short: list, all_results: list,
                    btc_prices, btc_spot: float, btc_vol: float, out_path: str):
    """Comprehensive 6-panel dashboard."""
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(
        f"BTC Reallocation Analysis Dashboard\n"
        f"BTC: \\${btc_spot:,.0f}  |  30d Vol: {btc_vol:.1%}  |  "
        f"Universe: {len(all_results)} assets  |  Reallocation: 25% of \\$1M",
        fontsize=18, fontweight="bold", y=0.99
    )
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors_long = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
    colors_short = plt.cm.Reds(np.linspace(0.4, 0.9, 10))

    # Panel 1: Scatter — Correlation vs Vol (all assets)
    ax1 = fig.add_subplot(gs[0, 0])
    all_corr3 = [r["corr_3m"] for r in all_results]
    all_vol = [r["vol_30d"] for r in all_results]
    ax1.scatter(all_corr3, all_vol, alpha=0.3, s=40, color="gray", label="All assets")

    for r in top_long:
        ax1.scatter(r["corr_3m"], r["vol_30d"], s=160, color="blue", zorder=5,
                    edgecolors="black", linewidths=0.8)
        ax1.annotate(r["ticker"], (r["corr_3m"], r["vol_30d"]),
                     fontsize=11, fontweight="bold", ha="center", va="bottom", color="blue",
                     xytext=(0, 8), textcoords="offset points")
    for r in top_short:
        ax1.scatter(r["corr_3m"], r["vol_30d"], s=160, color="red", zorder=5,
                    edgecolors="black", linewidths=0.8)
        ax1.annotate(r["ticker"], (r["corr_3m"], r["vol_30d"]),
                     fontsize=11, fontweight="bold", ha="center", va="bottom", color="red",
                     xytext=(0, 8), textcoords="offset points")
    ax1.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("3M Correlation with BTC", fontsize=13)
    ax1.set_ylabel("30d Realized Vol", fontsize=13)
    ax1.set_title("Correlation vs Volatility Landscape", fontsize=15)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Top-10 Long — score decomposition
    ax2 = fig.add_subplot(gs[0, 1])
    tickers_l = [r["ticker"] for r in top_long]
    scores_l = [r["score_long"] for r in top_long]
    y_pos = np.arange(len(tickers_l))
    ax2.barh(y_pos, scores_l, color=colors_long, edgecolor="navy", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tickers_l, fontsize=12)
    ax2.set_xlabel("Composite Score", fontsize=13)
    ax2.set_title("Top 10 — Long BTC Thesis (hedge candidates)", fontsize=15)
    ax2.invert_yaxis()
    for i, (s, r) in enumerate(zip(scores_l, top_long)):
        ax2.text(s + 0.01, i, f"{chr(961)}={r['corr_3m']:+.2f}", va="center", fontsize=10)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel 3: Top-10 Short — score decomposition
    ax3 = fig.add_subplot(gs[1, 0])
    tickers_s = [r["ticker"] for r in top_short]
    scores_s = [r["score_short"] for r in top_short]
    y_pos = np.arange(len(tickers_s))
    ax3.barh(y_pos, scores_s, color=colors_short, edgecolor="darkred", linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(tickers_s, fontsize=12)
    ax3.set_xlabel("Composite Score", fontsize=13)
    ax3.set_title("Top 10 — Short BTC Thesis (correlated downside plays)", fontsize=15)
    ax3.invert_yaxis()
    for i, (s, r) in enumerate(zip(scores_s, top_short)):
        ax3.text(s + 0.01, i, f"{chr(961)}={r['corr_3m']:+.2f}", va="center", fontsize=10)
    ax3.tick_params(axis="x", labelsize=11)
    ax3.grid(True, alpha=0.3, axis="x")

    # Panel 4: Put cost comparison — top picks
    ax4 = fig.add_subplot(gs[1, 1])
    combined = []
    for r in top_long[:5]:
        combined.append((r["ticker"], r["put_pct_3m"], r["put_pct_6m"], r["put_pct_12m"], "Long"))
    for r in top_short[:5]:
        combined.append((r["ticker"], r["put_pct_3m"], r["put_pct_6m"], r["put_pct_12m"], "Short"))

    ctickers = [c[0] for c in combined]
    x = np.arange(len(ctickers))
    w = 0.25
    ax4.bar(x - w, [c[1] * 100 for c in combined], w, label="3M", color="#E91E63", alpha=0.8)
    ax4.bar(x, [c[2] * 100 for c in combined], w, label="6M", color="#9C27B0", alpha=0.8)
    ax4.bar(x + w, [c[3] * 100 for c in combined], w, label="12M", color="#3F51B5", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(ctickers, rotation=45, ha="right", fontsize=12)

    # Color-code labels
    for i, c in enumerate(combined):
        color = "blue" if c[4] == "Long" else "red"
        ax4.get_xticklabels()[i].set_color(color)

    ax4.set_ylabel("ATM Put (% of Spot)", fontsize=13)
    ax4.set_title("Put Pricing — Top 5 per Thesis", fontsize=15)
    ax4.legend(fontsize=11)
    ax4.tick_params(axis="y", labelsize=11)
    ax4.grid(True, alpha=0.3, axis="y")

    # Panel 5: BTC price + top-5 long normalized
    ax5 = fig.add_subplot(gs[2, 0])
    btc_norm = (btc_prices / btc_prices.iloc[0]) * 100
    ax5.plot(btc_norm.index, btc_norm.values, color="orange", linewidth=2, label="BTC", zorder=10)
    for i, r in enumerate(top_long[:5]):
        ps = r["price_series"]
        common_start = max(ps.index[0], btc_prices.index[0])
        ps_a = ps[ps.index >= common_start]
        if len(ps_a) > 10:
            norm = (ps_a / ps_a.iloc[0]) * 100
            ax5.plot(norm.index, norm.values, linewidth=1.2, alpha=0.7,
                     label=f"{r['ticker']}")
    ax5.set_ylabel("Normalized (Start=100)", fontsize=13)
    ax5.set_title("Long Thesis — Top 5 vs BTC Price", fontsize=15)
    ax5.legend(fontsize=11, loc="upper left")
    ax5.tick_params(axis="both", labelsize=11)
    ax5.grid(True, alpha=0.3)

    # Panel 6: BTC price + top-5 short normalized
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(btc_norm.index, btc_norm.values, color="orange", linewidth=2, label="BTC", zorder=10)
    for i, r in enumerate(top_short[:5]):
        ps = r["price_series"]
        common_start = max(ps.index[0], btc_prices.index[0])
        ps_a = ps[ps.index >= common_start]
        if len(ps_a) > 10:
            norm = (ps_a / ps_a.iloc[0]) * 100
            ax6.plot(norm.index, norm.values, linewidth=1.2, alpha=0.7,
                     label=f"{r['ticker']}")
    ax6.set_ylabel("Normalized (Start=100)", fontsize=13)
    ax6.set_title("Short Thesis — Top 5 vs BTC Price", fontsize=15)
    ax6.legend(fontsize=11, loc="upper left")
    ax6.tick_params(axis="both", labelsize=11)
    ax6.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_analysis()
