"""
BTC Options-Drawdown Interaction Analysis.

Fetches recent BTC/USD daily data covering the Jan-Feb 2026 drawdown
(~$97K -> ~$63K) and simulates how protective puts and collars would
have performed. Computes realized vol, put payoff profiles, and
protection cost breakdowns.

Usage:
    python -m allocation_gym.options.drawdown_analysis
    python -m allocation_gym.options.drawdown_analysis --iv 0.80 --otm-pct 0.10
"""

import argparse
import math
import sys
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from allocation_gym.credentials import get_alpaca_keys
from allocation_gym.options.metrics import OptionsBacktestMetrics, compute_options_metrics
from allocation_gym.options.simulation import (
    OptionsSimulationResult,
    OptionsStrategyType,
    run_options_simulation,
)


# ── Data loading ─────────────────────────────────────────────────────────


def load_btc_daily(days: int = 120) -> pd.DataFrame:
    """Fetch daily BTC/USD bars from Alpaca."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key, secret_key = get_alpaca_keys()
    if not api_key or not secret_key:
        raise RuntimeError("Alpaca credentials required")

    end = datetime.utcnow()
    start = end - timedelta(days=int(days * 1.5))

    client = CryptoHistoricalDataClient(api_key, secret_key)
    request = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        start=start,
        end=end,
        timeframe=TimeFrame.Day,
    )
    bars = client.get_crypto_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level="symbol")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(days)


def df_to_bars(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list-of-dicts for simulation."""
    bars = []
    for idx, row in df.iterrows():
        bars.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        })
    return bars


# ── Realized vol computation ─────────────────────────────────────────────


def compute_realized_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rolling annualized realized volatility (close-to-close)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(365)


def compute_yang_zhang_vol(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Yang-Zhang volatility estimator using OHLC data."""
    log_oc = np.log(df["Close"] / df["Open"])
    log_co = np.log(df["Open"] / df["Close"].shift(1))
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])

    # Rogers-Satchell
    rs = log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)

    close_var = log_co.rolling(window).var()
    open_var = log_oc.rolling(window).var()
    rs_var = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = close_var + k * open_var + (1 - k) * rs_var
    return np.sqrt(yz_var.clip(lower=0) * 365)


# ── Drawdown analysis ────────────────────────────────────────────────────


def find_drawdowns(df: pd.DataFrame, threshold: float = 0.10) -> List[Dict]:
    """Identify drawdown periods exceeding threshold."""
    prices = df["Close"].values
    dates = df.index
    peak = prices[0]
    peak_idx = 0
    drawdowns = []
    in_drawdown = False
    dd_start = 0

    for i in range(len(prices)):
        if prices[i] > peak:
            if in_drawdown:
                drawdowns.append({
                    "start_date": str(dates[dd_start].date()),
                    "trough_date": str(dates[trough_idx].date()),
                    "end_date": str(dates[i].date()),
                    "peak_price": peak,
                    "trough_price": prices[trough_idx],
                    "drawdown_pct": (peak - prices[trough_idx]) / peak * 100,
                    "duration_days": (dates[i] - dates[dd_start]).days,
                })
                in_drawdown = False
            peak = prices[i]
            peak_idx = i

        dd = (peak - prices[i]) / peak
        if dd >= threshold and not in_drawdown:
            in_drawdown = True
            dd_start = peak_idx
            trough_idx = i
        elif in_drawdown and prices[i] < prices[trough_idx]:
            trough_idx = i

    # Capture ongoing drawdown
    if in_drawdown:
        drawdowns.append({
            "start_date": str(dates[dd_start].date()),
            "trough_date": str(dates[trough_idx].date()),
            "end_date": "ongoing",
            "peak_price": peak,
            "trough_price": prices[trough_idx],
            "drawdown_pct": (peak - prices[trough_idx]) / peak * 100,
            "duration_days": (dates[-1] - dates[dd_start]).days,
        })

    return drawdowns


# ── Main analysis ────────────────────────────────────────────────────────


def run_drawdown_analysis(
    iv: float = 0.80,
    otm_pct: float = 0.05,
    roll_days: int = 21,
    lookback_days: int = 120,
):
    """
    Full analysis of BTC options interaction during the latest drawdown.

    Args:
        iv: Implied volatility for BS pricing (BTC IV is typically 60-100%)
        otm_pct: OTM strike distance (0.05 = 5%)
        roll_days: Trading days between rolls
        lookback_days: How far back to fetch data
    """
    print("=" * 70)
    print("BTC OPTIONS-DRAWDOWN INTERACTION ANALYSIS")
    print("=" * 70)
    print(f"IV: {iv:.0%}  |  OTM: {otm_pct:.0%}  |  Roll: {roll_days}d")
    print()

    # ── 1. Load data ─────────────────────────────────────────────────
    print("Loading BTC/USD daily data from Alpaca...")
    df = load_btc_daily(days=lookback_days)
    bars = df_to_bars(df)
    print(f"  Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Price range: ${df['Close'].min():,.0f} - ${df['Close'].max():,.0f}")
    print(f"  Current: ${df['Close'].iloc[-1]:,.0f}")
    print()

    # ── 2. Identify drawdowns ────────────────────────────────────────
    print("Drawdown periods (>10%):")
    drawdowns = find_drawdowns(df, threshold=0.10)
    for i, dd in enumerate(drawdowns):
        status = f"to {dd['end_date']}" if dd["end_date"] != "ongoing" else "(ONGOING)"
        print(
            f"  [{i+1}] {dd['start_date']} -> {dd['trough_date']} {status}"
        )
        print(
            f"      Peak ${dd['peak_price']:,.0f} -> Trough ${dd['trough_price']:,.0f} "
            f"({dd['drawdown_pct']:.1f}%, {dd['duration_days']}d)"
        )
    print()

    # ── 3. Realized volatility ───────────────────────────────────────
    rv_21 = compute_realized_vol(df, window=21)
    yz_14 = compute_yang_zhang_vol(df, window=14)
    print("Realized Volatility (annualized):")
    print(f"  Close-to-close (21d): {rv_21.iloc[-1]:.1%}")
    print(f"  Yang-Zhang (14d):     {yz_14.iloc[-1]:.1%}")
    print(f"  21d vol range:        {rv_21.dropna().min():.1%} - {rv_21.dropna().max():.1%}")
    print(f"  Input IV assumption:  {iv:.1%}")

    # IV vs realized spread
    rv_latest = rv_21.iloc[-1]
    iv_spread = iv - rv_latest
    print(f"  IV - RV spread:       {iv_spread:+.1%} ({'rich' if iv_spread > 0 else 'cheap'})")
    print()

    # ── 4. Options simulation ────────────────────────────────────────
    initial_price = bars[0]["close"]
    initial_shares = 1  # Per-unit analysis

    strategies = [
        (OptionsStrategyType.PROTECTIVE_PUT, "Protective Put"),
        (OptionsStrategyType.COLLAR, "Collar"),
    ]

    results = {}
    print("Options Strategy Simulations:")
    print("-" * 70)

    for strat_type, label in strategies:
        result = run_options_simulation(
            bars=bars,
            symbol="BTC/USD",
            strategy_type=strat_type,
            initial_shares=initial_shares,
            initial_price=initial_price,
            iv=iv,
            otm_pct=otm_pct,
            roll_period_days=roll_days,
        )
        metrics = compute_options_metrics(result)
        results[label] = {"result": result, "metrics": metrics}

        print(f"\n  {label} (BTC/USD, {otm_pct:.0%} OTM, {roll_days}d rolls):")
        print(f"    Total Return:         {metrics.total_return_pct:+.2f}%")
        print(f"    Annualized Return:    {metrics.annualized_return_pct:+.2f}%")
        print(f"    Sharpe Ratio:         {metrics.sharpe_ratio:.3f}")
        print(f"    Max Drawdown:         {metrics.max_drawdown_pct:.2f}%")
        print(f"      DD period:          {metrics.max_drawdown_start} to {metrics.max_drawdown_end}")
        print(f"    Premium Paid:         ${metrics.total_premium_paid:,.2f}/share")
        print(f"    Premium Received:     ${metrics.total_premium_received:,.2f}/share")
        print(f"    Net Premium Cost:     ${metrics.net_premium_cost:,.2f}/share")
        print(f"    Put Intrinsic Rcvd:   ${metrics.total_intrinsic_recovered:,.2f}/share")
        print(f"    Protection Cost:      {metrics.protection_cost_pct:.2f}% of initial")
        print(f"    Option Rolls:         {metrics.num_rolls}")

    # ── 5. Unhedged comparison ───────────────────────────────────────
    print()
    print("-" * 70)
    final_price = bars[-1]["close"]
    unhedged_return = (final_price - initial_price) / initial_price * 100
    unhedged_peak = max(b["close"] for b in bars)
    unhedged_dd = (unhedged_peak - min(b["close"] for b in bars)) / unhedged_peak * 100

    print(f"\n  Unhedged BTC (buy & hold):")
    print(f"    Total Return:         {unhedged_return:+.2f}%")
    print(f"    Max Drawdown:         {unhedged_dd:.2f}%")
    print(f"    Peak:                 ${unhedged_peak:,.0f}")
    print(f"    Current:              ${final_price:,.0f}")

    # ── 6. Roll-by-roll breakdown ────────────────────────────────────
    print()
    print("=" * 70)
    print("ROLL-BY-ROLL BREAKDOWN (Protective Put)")
    print("=" * 70)

    put_result = results["Protective Put"]["result"]
    for i, roll in enumerate(put_result.rolls):
        # Find price at expiry
        expiry_idx = min(roll.expiry_bar_idx, len(bars) - 1)
        expiry_price = bars[expiry_idx]["close"]
        roll_price = bars[min(max(0, roll.expiry_bar_idx - roll_days), len(bars) - 1)]["close"]

        status = "ITM" if expiry_price < roll.put_strike else "OTM"
        pnl = roll.put_intrinsic_at_expiry - roll.put_premium_paid
        print(
            f"  Roll {i+1}: {roll.roll_date} | "
            f"Strike ${roll.put_strike:,.0f} | "
            f"Prem ${roll.put_premium_paid:,.2f} | "
            f"Intrinsic ${roll.put_intrinsic_at_expiry:,.2f} | "
            f"{status} | Net ${pnl:+,.2f}"
        )

    # ── 7. IV sensitivity ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("IV SENSITIVITY ANALYSIS (Protective Put, 5% OTM)")
    print("=" * 70)
    print(f"  {'IV':>6s}  {'Return':>10s}  {'MaxDD':>8s}  {'Sharpe':>8s}  {'Cost':>10s}  {'Prem Paid':>12s}")
    print(f"  {'---':>6s}  {'------':>10s}  {'-----':>8s}  {'------':>8s}  {'----':>10s}  {'---------':>12s}")

    for test_iv in [0.40, 0.60, 0.80, 1.00, 1.20]:
        r = run_options_simulation(
            bars=bars,
            symbol="BTC/USD",
            strategy_type=OptionsStrategyType.PROTECTIVE_PUT,
            initial_shares=1,
            initial_price=initial_price,
            iv=test_iv,
            otm_pct=otm_pct,
            roll_period_days=roll_days,
        )
        m = compute_options_metrics(r)
        print(
            f"  {test_iv:>5.0%}  {m.total_return_pct:>+9.2f}%  {m.max_drawdown_pct:>7.2f}%  "
            f"{m.sharpe_ratio:>7.3f}  {m.protection_cost_pct:>+9.2f}%  "
            f"${m.total_premium_paid:>10,.2f}"
        )

    # ── 8. Strike sensitivity ────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"STRIKE SENSITIVITY (Protective Put, IV={iv:.0%})")
    print("=" * 70)
    print(f"  {'OTM%':>6s}  {'Return':>10s}  {'MaxDD':>8s}  {'Sharpe':>8s}  {'Cost':>10s}  {'Put Rcvd':>12s}")
    print(f"  {'----':>6s}  {'------':>10s}  {'-----':>8s}  {'------':>8s}  {'----':>10s}  {'--------':>12s}")

    for test_otm in [0.02, 0.05, 0.10, 0.15, 0.20]:
        r = run_options_simulation(
            bars=bars,
            symbol="BTC/USD",
            strategy_type=OptionsStrategyType.PROTECTIVE_PUT,
            initial_shares=1,
            initial_price=initial_price,
            iv=iv,
            otm_pct=test_otm,
            roll_period_days=roll_days,
        )
        m = compute_options_metrics(r)
        print(
            f"  {test_otm:>5.0%}  {m.total_return_pct:>+9.2f}%  {m.max_drawdown_pct:>7.2f}%  "
            f"{m.sharpe_ratio:>7.3f}  {m.protection_cost_pct:>+9.2f}%  "
            f"${m.total_intrinsic_recovered:>10,.2f}"
        )

    # ── 9. Key findings ──────────────────────────────────────────────
    put_m = results["Protective Put"]["metrics"]
    collar_m = results["Collar"]["metrics"]

    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()

    # DD reduction
    put_dd_reduction = unhedged_dd - put_m.max_drawdown_pct
    collar_dd_reduction = unhedged_dd - collar_m.max_drawdown_pct
    print(f"  Drawdown reduction:")
    print(f"    Protective put:  {put_dd_reduction:+.1f}pp  (from {unhedged_dd:.1f}% to {put_m.max_drawdown_pct:.1f}%)")
    print(f"    Collar:          {collar_dd_reduction:+.1f}pp  (from {unhedged_dd:.1f}% to {collar_m.max_drawdown_pct:.1f}%)")
    print()

    # Return drag
    put_drag = put_m.total_return_pct - unhedged_return
    collar_drag = collar_m.total_return_pct - unhedged_return
    print(f"  Return drag from hedging:")
    print(f"    Protective put:  {put_drag:+.2f}pp")
    print(f"    Collar:          {collar_drag:+.2f}pp")
    print()

    # Cost efficiency
    if put_dd_reduction > 0:
        cost_per_dd_pt = abs(put_m.net_premium_cost) / put_dd_reduction
        print(f"  Cost efficiency (put):")
        print(f"    Net premium per 1pp DD reduction: ${cost_per_dd_pt:,.2f}/share")
    print()

    # IV regime observation
    print(f"  Volatility regime:")
    print(f"    Current 21d realized vol: {rv_latest:.1%}")
    print(f"    BTC IV assumption:        {iv:.1%}")
    if iv_spread > 0.10:
        print(f"    -> Options appear EXPENSIVE (IV {iv_spread:.0%} above RV)")
        print(f"    -> Collar strategy preferred to offset premium via short call")
    elif iv_spread < -0.05:
        print(f"    -> Options appear CHEAP (IV {abs(iv_spread):.0%} below RV)")
        print(f"    -> Protective puts offer good value")
    else:
        print(f"    -> IV roughly at fair value relative to realized")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="BTC Options-Drawdown Analysis")
    parser.add_argument("--iv", type=float, default=0.80, help="Implied vol (default: 0.80)")
    parser.add_argument("--otm-pct", type=float, default=0.05, help="OTM pct (default: 0.05)")
    parser.add_argument("--roll-days", type=int, default=21, help="Roll period (default: 21)")
    parser.add_argument("--days", type=int, default=120, help="Lookback days (default: 120)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_drawdown_analysis(
        iv=args.iv,
        otm_pct=args.otm_pct,
        roll_days=args.roll_days,
        lookback_days=args.days,
    )
