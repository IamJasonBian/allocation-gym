#!/usr/bin/env python3
"""
Thesis Decay Analysis & Strategy Backtests
==========================================
Part 1: Rolling correlation analysis — has Grayscale BTC always been equity-correlated?
Part 2: Backtest three strategies (Baseline, Path A, Path B) on historical prices
Part 3: Pull actual portfolio snapshots from the runtime service API
"""

import sys
import os
import math
import json
import urllib.request
import urllib.parse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from allocation_gym.options.black_scholes import bs_put_price
from allocation_gym.options.simulation import (
    run_options_simulation,
    OptionsStrategyType,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────
RISK_FREE = 0.045
TRADING_DAYS = 252
RUNTIME_API = "https://route-runtime-service.netlify.app"

# Portfolio from the LaTeX report (Table 3)
PORTFOLIO_VALUE = 194_917.0

BASELINE_WEIGHTS = {
    "GBTC_MINI": 0.461,  # Grayscale BTC Mini Trust
    "SPY":       0.047,
    "VOO":       0.043,
    "QQQ":       0.052,
    "MSFT":      0.071,
    "SPY_PROXY": 0.326,  # PGR, MA, BRK.B, etc. treated as equity proxy
}

PATH_A_WEIGHTS = {
    "IBIT":      0.461,  # Convert Grayscale → spot BTC ETF
    "IWN":       0.050,  # New decorrelated position
    "SPY":       0.047,
    "VOO":       0.043,
    "QQQ":       0.052,
    "MSFT":      0.071,
    "SPY_PROXY": 0.276,  # Reduced residual
    # Plus: 10% notional in rolling SPY ATM puts (handled separately)
}

PATH_B_WEIGHTS = {
    "GBTC_MINI": 0.461,  # Keep Grayscale BTC
    "GLD":       0.100,  # New hedge
    "SPY":       0.007,  # Trimmed from 4.7%
    "VOO":       0.003,  # Trimmed from 4.3%
    "QQQ":       0.052,
    "MSFT":      0.071,
    "SPY_PROXY": 0.306,
}


# ── Data Loading ───────────────────────────────────────────────────────────

def fetch_all_prices() -> pd.DataFrame:
    """Fetch daily closes for all needed tickers via yfinance."""
    tickers = ["BTC-USD", "SPY", "QQQ", "IWM", "GLD", "IWN", "IBIT",
               "VOO", "MSFT"]

    # Grayscale BTC Mini Trust — try BTC first (the actual ticker)
    grayscale_candidates = ["BTC", "GBTC"]

    print("Fetching price data from yfinance...")
    data = yf.download(tickers + grayscale_candidates,
                       period="max", interval="1d",
                       auto_adjust=True, progress=False, threads=True)

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    # Resolve Grayscale BTC ticker: should be ~$20-60 range
    gbtc_mini_col = None
    for candidate in grayscale_candidates:
        if candidate in closes.columns:
            last_price = closes[candidate].dropna().iloc[-1] if len(closes[candidate].dropna()) > 0 else None
            if last_price and 10 < last_price < 100:
                gbtc_mini_col = candidate
                print(f"  Grayscale BTC Mini Trust resolved: {candidate} (${last_price:.2f})")
                break
            elif last_price:
                print(f"  {candidate}: ${last_price:.2f} — not mini trust range")

    if gbtc_mini_col is None:
        print("  WARNING: Could not find Grayscale BTC Mini Trust on yfinance")
        print("  Using GBTC as proxy (original Grayscale trust)")
        gbtc_mini_col = "GBTC" if "GBTC" in closes.columns else None

    # Rename for consistency
    result = pd.DataFrame()
    col_map = {
        "BTC-USD": "BTC_SPOT",
        "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM",
        "GLD": "GLD", "IWN": "IWN", "IBIT": "IBIT",
        "VOO": "VOO", "MSFT": "MSFT",
    }
    if gbtc_mini_col:
        col_map[gbtc_mini_col] = "GBTC_MINI"

    for src, dst in col_map.items():
        if src in closes.columns:
            result[dst] = closes[src]

    result = result.dropna(how="all").sort_index()

    print(f"  Date range: {result.index[0].date()} to {result.index[-1].date()}")
    print(f"  Tickers available: {list(result.columns)}")
    for col in result.columns:
        valid = result[col].dropna()
        print(f"    {col}: {len(valid)} days, ${valid.iloc[-1]:.2f}" if len(valid) > 0 else f"    {col}: no data")

    return result


# ── Part 1: Thesis Decay ──────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def thesis_decay_analysis(prices: pd.DataFrame) -> dict:
    """Analyze how Grayscale BTC correlation with equities evolved over time."""
    print("\n" + "=" * 70)
    print("  PART 1: THESIS DECAY ANALYSIS")
    print("=" * 70)

    if "GBTC_MINI" not in prices.columns:
        print("  Grayscale BTC data unavailable — skipping thesis decay")
        return {}

    # Align all instruments that have data with GBTC_MINI
    peers = [c for c in ["SPY", "QQQ", "IWM", "GLD", "IWN", "BTC_SPOT", "IBIT"] if c in prices.columns]
    aligned = prices[["GBTC_MINI"] + peers].dropna()
    log_rets = compute_log_returns(aligned)

    print(f"\n  Aligned data: {len(aligned)} trading days ({aligned.index[0].date()} → {aligned.index[-1].date()})")

    results = {"peers": {}, "dates": {}}

    # ── Rolling correlations ──
    for window in [21, 63]:
        label = f"{window}d"
        results["dates"][label] = []

        for peer in peers:
            rolling = log_rets["GBTC_MINI"].rolling(window).corr(log_rets[peer])
            valid = rolling.dropna()
            if len(valid) == 0:
                continue

            if peer not in results["peers"]:
                results["peers"][peer] = {}

            results["peers"][peer][label] = {
                "current": float(valid.iloc[-1]),
                "earliest": float(valid.iloc[0]),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "max": float(valid.max()),
            }

            if not results["dates"][label]:
                results["dates"][label] = [str(d.date()) for d in valid.index]

    # ── Current correlations table ──
    print(f"\n  {'─' * 60}")
    print(f"  TABLE 1: Current Rolling Correlations vs Grayscale BTC")
    print(f"  {'─' * 60}")
    print(f"  {'Ticker':<12} {'21d Corr':>10} {'63d Corr':>10} {'21d Vol':>10}")
    print(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10}")

    for peer in peers:
        if peer not in results["peers"]:
            continue
        corr_21 = results["peers"][peer].get("21d", {}).get("current", float("nan"))
        corr_63 = results["peers"][peer].get("63d", {}).get("current", float("nan"))

        # Rolling vol
        vol = float(log_rets[peer].rolling(21).std().iloc[-1] * np.sqrt(TRADING_DAYS) * 100) if peer in log_rets else float("nan")

        print(f"  {peer:<12} {corr_21:>+10.3f} {corr_63:>+10.3f} {vol:>9.1f}%")

    # ── Correlation evolution table ──
    print(f"\n  {'─' * 60}")
    print(f"  TABLE 2: Correlation Evolution (was it always this high?)")
    print(f"  {'─' * 60}")
    print(f"  {'Ticker':<12} {'Earliest 63d':>12} {'Latest 63d':>12} {'Change':>10} {'Mean':>10} {'Std':>8}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 8}")

    for peer in peers:
        p = results["peers"].get(peer, {}).get("63d")
        if not p:
            continue
        delta = p["current"] - p["earliest"]
        print(f"  {peer:<12} {p['earliest']:>+12.3f} {p['current']:>+12.3f} {delta:>+10.3f} {p['mean']:>+10.3f} {p['std']:>8.3f}")

    # ── Regime shift detection ──
    if "SPY" in log_rets.columns:
        rolling_spy = log_rets["GBTC_MINI"].rolling(21).corr(log_rets["SPY"]).dropna()
        above_threshold = rolling_spy > 0.5

        # Find first date correlation exceeded +0.5
        first_equity_like = above_threshold[above_threshold].index
        if len(first_equity_like) > 0:
            regime_date = str(first_equity_like[0].date())
            results["regime_shift_date"] = regime_date
            # What % of time has it been above 0.5?
            pct_above = above_threshold.sum() / len(above_threshold) * 100
            results["pct_time_equity_like"] = float(pct_above)
            print(f"\n  Regime shift: Grayscale BTC first exceeded +0.5 SPY corr on {regime_date}")
            print(f"  Above +0.5 threshold: {pct_above:.0f}% of the time")

    # ── Grayscale vs Spot BTC divergence ──
    if "BTC_SPOT" in results["peers"]:
        print(f"\n  {'─' * 60}")
        print(f"  TABLE 3: Grayscale BTC vs Spot BTC Divergence")
        print(f"  {'─' * 60}")
        print(f"  {'Window':<10} {'Grayscale↔SPY':>15} {'Grayscale↔BTC':>15} {'Delta':>10}")
        print(f"  {'─' * 10} {'─' * 15} {'─' * 15} {'─' * 10}")
        for w in ["21d", "63d"]:
            gs = results["peers"].get("SPY", {}).get(w, {}).get("current", float("nan"))
            gb = results["peers"].get("BTC_SPOT", {}).get(w, {}).get("current", float("nan"))
            print(f"  {w:<10} {gs:>+15.3f} {gb:>+15.3f} {gs - gb:>+10.3f}")

    # ── Rolling vol comparison ──
    print(f"\n  {'─' * 60}")
    print(f"  TABLE 4: Annualized Realized Vol (current 21d)")
    print(f"  {'─' * 60}")
    vol_data = {}
    for col in ["GBTC_MINI"] + peers:
        if col in log_rets.columns:
            v = float(log_rets[col].rolling(21).std().iloc[-1] * np.sqrt(TRADING_DAYS) * 100)
            vol_data[col] = v
            print(f"  {col:<12} {v:>8.1f}%")

    results["vol_data"] = vol_data
    return results


# ── Part 2: Backtests ─────────────────────────────────────────────────────

def portfolio_metrics(equity_curve: np.ndarray) -> dict:
    """Compute performance metrics from a daily equity curve."""
    daily_rets = np.diff(np.log(equity_curve))
    total_ret = equity_curve[-1] / equity_curve[0] - 1
    n_days = len(daily_rets)
    ann_factor = TRADING_DAYS / n_days if n_days > 0 else 1

    ann_ret = (1 + total_ret) ** ann_factor - 1
    ann_vol = float(np.std(daily_rets) * np.sqrt(TRADING_DAYS))
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = float(np.min(dd))

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_return": float(total_ret),
        "ann_return": float(ann_ret),
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "n_days": n_days,
    }


def run_weighted_portfolio(prices: pd.DataFrame, weights: dict,
                           initial_value: float) -> np.ndarray:
    """
    Run a daily-rebalanced weighted portfolio backtest.
    Returns equity curve array.
    """
    # Map weight keys to price columns
    key_to_col = {
        "GBTC_MINI": "GBTC_MINI",
        "SPY": "SPY", "VOO": "VOO", "QQQ": "QQQ",
        "MSFT": "MSFT", "GLD": "GLD", "IWN": "IWN",
        "IBIT": "IBIT",
        "SPY_PROXY": "SPY",  # Residual positions proxied as SPY
    }

    log_rets = compute_log_returns(prices)

    # Compute daily portfolio return as weighted sum of log returns
    port_log_rets = pd.Series(0.0, index=log_rets.index)
    total_weight = 0

    for key, w in weights.items():
        col = key_to_col.get(key)
        if col and col in log_rets.columns:
            port_log_rets += w * log_rets[col]
            total_weight += w

    if total_weight < 0.99:
        # Assign missing weight to SPY proxy
        if "SPY" in log_rets.columns:
            port_log_rets += (1 - total_weight) * log_rets["SPY"]

    # Build equity curve
    cum_rets = port_log_rets.cumsum()
    equity = initial_value * np.exp(cum_rets.values)
    equity = np.insert(equity, 0, initial_value)

    return equity


def run_backtests(prices: pd.DataFrame) -> dict:
    """Run all three strategy backtests."""
    print("\n" + "=" * 70)
    print("  PART 2: STRATEGY BACKTESTS")
    print("=" * 70)

    # Find common date range where all needed tickers have data
    needed = ["SPY", "QQQ", "VOO", "MSFT"]
    has_gbtc = "GBTC_MINI" in prices.columns
    has_ibit = "IBIT" in prices.columns
    has_gld = "GLD" in prices.columns
    has_iwn = "IWN" in prices.columns

    cols = list(set(needed + (["GBTC_MINI"] if has_gbtc else []) +
                    (["IBIT"] if has_ibit else []) +
                    (["GLD"] if has_gld else []) +
                    (["IWN"] if has_iwn else [])))
    aligned = prices[cols].dropna()

    print(f"\n  Backtest period: {aligned.index[0].date()} → {aligned.index[-1].date()} ({len(aligned)} days)")
    dates = [str(d.date()) for d in aligned.index]
    initial = PORTFOLIO_VALUE

    results = {}

    # ── Strategy 1: Baseline ──
    print("\n  Running Baseline (current portfolio)...")
    eq_baseline = run_weighted_portfolio(aligned, BASELINE_WEIGHTS, initial)
    results["baseline"] = {
        "metrics": portfolio_metrics(eq_baseline),
        "equity_curve": eq_baseline.tolist(),
        "dates": dates,
    }

    # ── Strategy 2: Path A (IBIT + SPY puts + IWN) ──
    if has_ibit:
        print("  Running Path A (spot BTC ETF + SPY puts + IWN)...")
        eq_path_a_equity = run_weighted_portfolio(aligned, PATH_A_WEIGHTS, initial)

        # Options overlay: rolling SPY ATM puts on 10% of portfolio
        spy_prices = aligned["SPY"].values
        spy_bars = [{"date": str(d.date()), "close": float(p)} for d, p in zip(aligned.index, spy_prices)]
        notional_puts = initial * 0.10
        n_shares = max(1, int(notional_puts / spy_prices[0]))

        # Realized vol for IV estimate
        spy_rets = np.diff(np.log(spy_prices))
        realized_vol = float(np.std(spy_rets[:min(63, len(spy_rets))]) * np.sqrt(TRADING_DAYS))
        iv_estimate = realized_vol * 1.10  # 10% vol premium

        sim_result = run_options_simulation(
            bars=spy_bars,
            symbol="SPY",
            strategy_type=OptionsStrategyType.PROTECTIVE_PUT,
            initial_shares=n_shares,
            initial_price=spy_prices[0],
            iv=iv_estimate,
            otm_pct=0.0,  # ATM
            roll_period_days=63,  # quarterly
            risk_free_rate=RISK_FREE,
        )

        # Extract options P&L contribution (net_portfolio_value - equity_value per day)
        options_pnl = np.zeros(len(eq_path_a_equity))
        for i, snap in enumerate(sim_result.snapshots):
            if i + 1 < len(options_pnl):
                options_pnl[i + 1] = snap.net_portfolio_value - snap.equity_value

        eq_path_a = eq_path_a_equity + options_pnl[:len(eq_path_a_equity)]
        results["path_a"] = {
            "metrics": portfolio_metrics(eq_path_a),
            "equity_curve": eq_path_a.tolist(),
            "dates": dates,
            "options_cost_total": float(sim_result.snapshots[-1].cumulative_premium_paid) if sim_result.snapshots else 0,
            "n_rolls": len(sim_result.rolls),
        }
    else:
        print("  IBIT data unavailable — skipping Path A")
        results["path_a"] = None

    # ── Strategy 3: Path B (keep Grayscale + GLD) ──
    if has_gbtc and has_gld:
        print("  Running Path B (keep Grayscale BTC + add GLD)...")
        eq_path_b = run_weighted_portfolio(aligned, PATH_B_WEIGHTS, initial)
        results["path_b"] = {
            "metrics": portfolio_metrics(eq_path_b),
            "equity_curve": eq_path_b.tolist(),
            "dates": dates,
        }
    else:
        missing = []
        if not has_gbtc: missing.append("GBTC_MINI")
        if not has_gld: missing.append("GLD")
        print(f"  Missing {missing} — skipping Path B")
        results["path_b"] = None

    # ── Comparison table ──
    print(f"\n  {'─' * 75}")
    print(f"  {'Strategy':<20} {'Total Ret':>10} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>8} {'Max DD':>10} {'Calmar':>8}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 8}")

    for name, label in [("baseline", "Baseline"), ("path_a", "Path A (IBIT+puts)"), ("path_b", "Path B (BTC+GLD)")]:
        r = results.get(name)
        if r is None:
            print(f"  {label:<20} {'N/A':>10}")
            continue
        m = r["metrics"]
        print(f"  {label:<20} {m['total_return']:>+9.1%} {m['ann_return']:>+9.1%} {m['ann_vol']:>9.1%} "
              f"{m['sharpe']:>7.2f} {m['max_drawdown']:>+9.1%} {m['calmar']:>7.2f}")

    if results.get("path_a"):
        print(f"\n  Path A options: {results['path_a']['n_rolls']} put rolls, "
              f"total premium paid: ${results['path_a']['options_cost_total']:,.0f}")

    return results


# ── Part 3: Actuals ───────────────────────────────────────────────────────

def fetch_runtime_snapshots() -> list:
    """Fetch portfolio snapshots from the runtime service API."""
    print("\n" + "=" * 70)
    print("  PART 3: ACTUAL PORTFOLIO (Runtime Service)")
    print("=" * 70)

    try:
        # Get list of snapshot keys
        url = f"{RUNTIME_API}/api/snapshots"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        keys = data if isinstance(data, list) else data.get("keys", data.get("snapshots", []))
        print(f"\n  Found {len(keys)} snapshots")

        if not keys:
            return []

        # Fetch each snapshot (limit to avoid hammering)
        snapshots = []
        for i, key in enumerate(keys[:50]):
            key_str = key if isinstance(key, str) else key.get("key", str(key))
            snap_url = f"{RUNTIME_API}/api/snapshots?key={urllib.parse.quote(key_str)}"
            try:
                req = urllib.request.Request(snap_url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    snap = json.loads(resp.read().decode())
                    snapshots.append({"key": key_str, "data": snap})
            except Exception as e:
                print(f"    Warning: failed to fetch snapshot {key_str}: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"    Fetched {i + 1}/{min(len(keys), 50)} snapshots...")

        print(f"  Successfully fetched {len(snapshots)} snapshots")
        return snapshots

    except Exception as e:
        print(f"\n  WARNING: Could not reach runtime API: {e}")
        print(f"  Skipping actuals comparison")
        return []


def extract_actuals(snapshots: list) -> dict:
    """Extract equity curve and portfolio state from snapshots."""
    if not snapshots:
        return {}

    records = []
    for snap in snapshots:
        key = snap["key"]
        raw = snap["data"]

        # Try multiple paths to find equity and portfolio value
        equity = None
        portfolio_val = None
        timestamp = key

        # Navigate nested structure: {snapshot_key, data: {timestamp, account, positions, ...}}
        if isinstance(raw, dict):
            data = raw.get("data", raw)  # unwrap nested 'data' key
            if isinstance(data, dict):
                timestamp = data.get("timestamp", raw.get("timestamp", key))

                # Check account sub-object
                account = data.get("account", {})
                if isinstance(account, dict):
                    equity = account.get("equity")
                    portfolio_val = account.get("portfolio_value") or account.get("market_value")

                # Direct fields fallback
                equity = equity or data.get("equity")
                portfolio_val = portfolio_val or data.get("portfolio_value")

                # Nested under portfolio
                portfolio = data.get("portfolio", {})
                if isinstance(portfolio, dict):
                    equity = equity or portfolio.get("equity")
                    portfolio_val = portfolio_val or portfolio.get("market_value")
            else:
                timestamp = raw.get("timestamp", key)

        if equity is not None:
            try:
                records.append({
                    "timestamp": str(timestamp),
                    "equity": float(equity),
                    "portfolio_value": float(portfolio_val) if portfolio_val else None,
                })
            except (ValueError, TypeError):
                pass

    if not records:
        # Try to extract from the raw data structure
        print("  Could not extract equity values from standard paths")
        print("  Attempting to inspect snapshot structure...")

        # Show first snapshot structure for debugging
        if snapshots:
            first = snapshots[0]["data"]
            if isinstance(first, dict):
                print(f"  Top-level keys: {list(first.keys())[:15]}")
                for k, v in first.items():
                    if isinstance(v, dict):
                        print(f"    {k}: {list(v.keys())[:10]}")
                    elif isinstance(v, (int, float, str)):
                        print(f"    {k}: {v}")
        return {}

    # Sort by timestamp
    records.sort(key=lambda r: r["timestamp"])

    print(f"\n  Extracted {len(records)} equity data points")
    print(f"  Date range: {records[0]['timestamp']} → {records[-1]['timestamp']}")
    print(f"  Equity: ${records[0]['equity']:,.0f} → ${records[-1]['equity']:,.0f}")

    equity_curve = [r["equity"] for r in records]
    timestamps = [r["timestamp"] for r in records]

    if len(equity_curve) >= 2:
        total_ret = equity_curve[-1] / equity_curve[0] - 1
        print(f"  Total return: {total_ret:+.1%}")

    # ── Print timeline ──
    print(f"\n  {'─' * 50}")
    print(f"  Snapshot Timeline")
    print(f"  {'─' * 50}")
    for r in records[:5]:
        pv = f"  PV: ${r['portfolio_value']:,.0f}" if r['portfolio_value'] else ""
        print(f"  {r['timestamp'][:19]}  Equity: ${r['equity']:,.0f}{pv}")
    if len(records) > 10:
        print(f"  ... ({len(records) - 10} more) ...")
    for r in records[-5:]:
        pv = f"  PV: ${r['portfolio_value']:,.0f}" if r['portfolio_value'] else ""
        print(f"  {r['timestamp'][:19]}  Equity: ${r['equity']:,.0f}{pv}")

    return {
        "equity_curve": equity_curve,
        "timestamps": timestamps,
        "records": records,
        "total_return": equity_curve[-1] / equity_curve[0] - 1 if len(equity_curve) >= 2 else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  THESIS DECAY ANALYSIS & STRATEGY BACKTESTS")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1. Fetch prices
    prices = fetch_all_prices()

    # 2. Thesis decay
    thesis_results = thesis_decay_analysis(prices)

    # 3. Backtests
    backtest_results = run_backtests(prices)

    # 4. Actuals
    snapshots = fetch_runtime_snapshots()
    actuals = extract_actuals(snapshots)

    # 5. Summary JSON
    summary = {
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "thesis_decay": {},
        "backtests": {},
        "actuals": {},
    }

    # Thesis decay summary
    if thesis_results:
        td = summary["thesis_decay"]
        for peer in ["SPY", "QQQ", "IWM", "GLD", "BTC_SPOT"]:
            if peer in thesis_results.get("peers", {}):
                td[peer] = thesis_results["peers"][peer]
        td["regime_shift_date"] = thesis_results.get("regime_shift_date")
        td["pct_time_equity_like"] = thesis_results.get("pct_time_equity_like")
        td["vol_data"] = thesis_results.get("vol_data", {})

    # Backtest summary
    for name in ["baseline", "path_a", "path_b"]:
        r = backtest_results.get(name)
        if r:
            summary["backtests"][name] = r["metrics"]
            if "options_cost_total" in r:
                summary["backtests"][name]["options_cost_total"] = r["options_cost_total"]
                summary["backtests"][name]["n_rolls"] = r["n_rolls"]

    # Actuals summary
    if actuals:
        summary["actuals"] = {
            "total_return": actuals.get("total_return"),
            "n_snapshots": len(actuals.get("equity_curve", [])),
            "first_equity": actuals["equity_curve"][0] if actuals.get("equity_curve") else None,
            "last_equity": actuals["equity_curve"][-1] if actuals.get("equity_curve") else None,
        }

    # Write summary
    out_path = os.path.join(os.path.dirname(__file__), "thesis_backtest_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary written to {out_path}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
