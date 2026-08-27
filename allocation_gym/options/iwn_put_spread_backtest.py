"""
IWN PUT spread backtest: Daily spread orders vs immediate execution.

Strategy:
- Each day at market open, submit a spread of PUT limit orders at different strikes
- Also submit limit buy orders for underlying at various price levels
- Compare 50% fill rate for spread orders vs 50% fill rate for immediate execution
- Backtest over last 3 weeks

Usage:
    python -m allocation_gym.options.iwn_put_spread_backtest
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from allocation_gym.options.black_scholes import bs_put_price


# ── Configuration ────────────────────────────────────────────────────────


INITIAL_CAPITAL = 50000.0  # Need ~$19k per contract, so $50k allows for 2-3 positions
POSITION_SIZE = 0.10  # 10% of capital per trade
FILL_RATE = 0.50  # 50% fill rate
SPREAD_STRIKES = [-0.05, -0.02, 0.0, 0.02, 0.05]  # OTM %
MAX_POSITIONS = 3  # Maximum number of open positions (limited by capital)
RISK_FREE_RATE = 0.045
DTE = 7  # Days to expiration for PUTs
CONTRACTS_PER_ORDER = 1  # Number of contracts per order


# ── Data Loading ─────────────────────────────────────────────────────────


def load_iwn_recent(days: int = 30) -> pd.DataFrame:
    """Load recent IWN data for backtest"""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.Ticker("IWN").history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise RuntimeError("No IWN data from yfinance")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ── Strategy Logic ───────────────────────────────────────────────────────


def calculate_implied_vol(df: pd.DataFrame, window: int = 21) -> float:
    """Estimate IV from realized volatility"""
    returns = np.log(df["Close"] / df["Close"].shift(1))
    vol = returns.tail(window).std() * np.sqrt(252)
    # Add vol premium (typically IV > realized)
    return float(vol * 1.3)


def simulate_spread_orders(
    df: pd.DataFrame,
    start_idx: int,
    num_days: int = 15,
) -> Dict:
    """
    Simulate spread order strategy:
    - Submit limit orders at market open each day
    - 50% fill rate based on intraday price action
    """
    results = {
        "dates": [],
        "positions": [],
        "pnl": [],
        "cash": INITIAL_CAPITAL,
        "total_value": [],
        "fills": [],
    }

    positions = []  # List of (entry_date, entry_price, strike, premium, shares)

    for day_offset in range(num_days):
        idx = start_idx + day_offset
        if idx >= len(df):
            break

        date = df.index[idx]
        open_price = df["Open"].iloc[idx]
        high = df["High"].iloc[idx]
        low = df["Low"].iloc[idx]
        close = df["Close"].iloc[idx]

        # Calculate IV for option pricing
        iv = calculate_implied_vol(df[:idx])

        # Submit spread of PUT limit orders at open
        filled_orders = []
        for strike_pct in SPREAD_STRIKES:
            strike = round(open_price * (1 + strike_pct), 2)

            # Price PUT option
            T = DTE / 252
            put_premium = bs_put_price(open_price, strike, T, RISK_FREE_RATE, iv)

            # Check if order would fill (50% chance, based on whether low touched strike)
            # Simplification: if intraday low <= strike, higher chance of fill
            fill_prob = FILL_RATE
            if low <= strike:
                fill_prob = min(0.8, FILL_RATE + 0.3)

            # Limit total positions
            if np.random.random() < fill_prob and len(positions) < MAX_POSITIONS:
                # Fixed contract size
                contracts = CONTRACTS_PER_ORDER
                collateral_per_contract = strike * 100  # Cash-secured put requires full strike value
                total_collateral = contracts * collateral_per_contract
                premium_collected = contracts * put_premium * 100  # 100 shares per contract

                if results["cash"] >= total_collateral:
                    results["cash"] -= total_collateral
                    results["cash"] += premium_collected

                    filled_orders.append({
                        "strike": strike,
                        "premium": put_premium,
                        "contracts": contracts,
                        "collateral": total_collateral,
                        "entry_date": date,
                        "expiration": date + timedelta(days=DTE),
                    })

        positions.extend(filled_orders)

        # Mark-to-market existing positions
        mtm_value = 0
        expired_positions = []
        for i, pos in enumerate(positions):
            if date >= pos["expiration"]:
                # Position expired
                if close < pos["strike"]:
                    # Assigned: buy shares at strike, then sell immediately
                    assignment_cost = pos["contracts"] * pos["strike"] * 100
                    sale_proceeds = pos["contracts"] * close * 100
                    pnl = sale_proceeds - assignment_cost + (pos["contracts"] * pos["premium"] * 100)
                    results["cash"] += pos["collateral"]  # Return collateral
                else:
                    # Expired worthless, keep premium
                    pnl = pos["contracts"] * pos["premium"] * 100
                    results["cash"] += pos["collateral"]  # Return collateral

                results["pnl"].append(pnl)
                expired_positions.append(i)
            else:
                # Mark-to-market
                T_remaining = max((pos["expiration"] - date).days / 252, 0.001)
                current_put_value = bs_put_price(close, pos["strike"], T_remaining, RISK_FREE_RATE, iv)
                mtm_value -= pos["contracts"] * current_put_value * 100  # Short puts are liability

        # Remove expired positions
        for i in sorted(expired_positions, reverse=True):
            positions.pop(i)

        total_value = results["cash"] + mtm_value

        results["dates"].append(date)
        results["positions"].append(len(positions))
        results["total_value"].append(total_value)
        results["fills"].append(len(filled_orders))

    return results


def simulate_immediate_execution(
    df: pd.DataFrame,
    start_idx: int,
    num_days: int = 15,
) -> Dict:
    """
    Simulate immediate execution strategy:
    - Buy PUTs at market price each day
    - 50% fill rate (randomly skip half the days)
    """
    results = {
        "dates": [],
        "positions": [],
        "pnl": [],
        "cash": INITIAL_CAPITAL,
        "total_value": [],
        "fills": [],
    }

    positions = []

    for day_offset in range(num_days):
        idx = start_idx + day_offset
        if idx >= len(df):
            break

        date = df.index[idx]
        open_price = df["Open"].iloc[idx]
        close = df["Close"].iloc[idx]

        # Calculate IV
        iv = calculate_implied_vol(df[:idx])

        # 50% chance of execution (simulating fill rate)
        if np.random.random() < FILL_RATE and len(positions) < MAX_POSITIONS:
            # Execute at market (use ATM put as representative)
            strike = round(open_price, 2)
            T = DTE / 252
            put_premium = bs_put_price(open_price, strike, T, RISK_FREE_RATE, iv)

            contracts = CONTRACTS_PER_ORDER
            collateral_per_contract = strike * 100
            total_collateral = contracts * collateral_per_contract
            premium_collected = contracts * put_premium * 100

            if results["cash"] >= total_collateral:
                results["cash"] -= total_collateral
                results["cash"] += premium_collected

                positions.append({
                    "strike": strike,
                    "premium": put_premium,
                    "contracts": contracts,
                    "collateral": total_collateral,
                    "entry_date": date,
                    "expiration": date + timedelta(days=DTE),
                })
                results["fills"].append(1)
            else:
                results["fills"].append(0)
        else:
            results["fills"].append(0)

        # Mark-to-market
        mtm_value = 0
        expired_positions = []
        for i, pos in enumerate(positions):
            if date >= pos["expiration"]:
                if close < pos["strike"]:
                    assignment_cost = pos["contracts"] * pos["strike"] * 100
                    sale_proceeds = pos["contracts"] * close * 100
                    pnl = sale_proceeds - assignment_cost + (pos["contracts"] * pos["premium"] * 100)
                    results["cash"] += pos["collateral"]
                else:
                    pnl = pos["contracts"] * pos["premium"] * 100
                    results["cash"] += pos["collateral"]

                results["pnl"].append(pnl)
                expired_positions.append(i)
            else:
                T_remaining = max((pos["expiration"] - date).days / 252, 0.001)
                current_put_value = bs_put_price(close, pos["strike"], T_remaining, RISK_FREE_RATE, iv)
                mtm_value -= pos["contracts"] * current_put_value * 100

        for i in sorted(expired_positions, reverse=True):
            positions.pop(i)

        total_value = results["cash"] + mtm_value

        results["dates"].append(date)
        results["positions"].append(len(positions))
        results["total_value"].append(total_value)

    return results


# ── Main Analysis ────────────────────────────────────────────────────────


def run_backtest():
    print("=" * 70)
    print("IWN PUT SPREAD BACKTEST (Last 3 Weeks)")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    print("\nLoading IWN data...")
    df = load_iwn_recent(days=30)
    print(f"  {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")

    # Get last 3 weeks (15 trading days)
    start_idx = len(df) - 15

    print(f"\n  Backtest period: {df.index[start_idx].date()} to {df.index[-1].date()}")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Fill rate: {FILL_RATE:.0%}")
    print(f"  Position size: {POSITION_SIZE:.0%} per trade")
    print(f"  PUT DTE: {DTE} days")

    # Run simulations
    print("\n" + "=" * 70)
    print("STRATEGY 1: SPREAD ORDERS (limit orders at multiple strikes)")
    print("=" * 70)

    spread_results = simulate_spread_orders(df, start_idx, num_days=15)

    spread_total_return = (spread_results["total_value"][-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    spread_total_pnl = sum(spread_results["pnl"])
    spread_avg_positions = np.mean(spread_results["positions"])
    spread_total_fills = sum(spread_results["fills"])

    print(f"\n  Total fills: {spread_total_fills}")
    print(f"  Avg open positions: {spread_avg_positions:.1f}")
    print(f"  Total P&L: ${spread_total_pnl:,.2f}")
    print(f"  Final value: ${spread_results['total_value'][-1]:,.2f}")
    print(f"  Total return: {spread_total_return:+.2f}%")

    print("\n" + "=" * 70)
    print("STRATEGY 2: IMMEDIATE EXECUTION (market orders at open)")
    print("=" * 70)

    immediate_results = simulate_immediate_execution(df, start_idx, num_days=15)

    immediate_total_return = (immediate_results["total_value"][-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    immediate_total_pnl = sum(immediate_results["pnl"])
    immediate_avg_positions = np.mean(immediate_results["positions"])
    immediate_total_fills = sum(immediate_results["fills"])

    print(f"\n  Total fills: {immediate_total_fills}")
    print(f"  Avg open positions: {immediate_avg_positions:.1f}")
    print(f"  Total P&L: ${immediate_total_pnl:,.2f}")
    print(f"  Final value: ${immediate_results['total_value'][-1]:,.2f}")
    print(f"  Total return: {immediate_total_return:+.2f}%")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    better_strategy = "SPREAD ORDERS" if spread_total_return > immediate_total_return else "IMMEDIATE EXECUTION"
    diff = abs(spread_total_return - immediate_total_return)

    print(f"\n  Winner: {better_strategy}")
    print(f"  Return difference: {diff:.2f}%")
    print(f"  Spread vs Immediate P&L: ${spread_total_pnl - immediate_total_pnl:+,.2f}")

    # Generate visualization
    _plot_backtest_results(df, start_idx, spread_results, immediate_results)

    print("\n" + "=" * 70)
    print("Backtest complete!")
    print("=" * 70)


def _plot_backtest_results(df, start_idx, spread_results, immediate_results):
    """Generate backtest visualization"""
    plt.close("all")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    spread_return = (spread_results["total_value"][-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    immediate_return = (immediate_results["total_value"][-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    fig.suptitle(
        f"IWN PUT Spread Backtest (3 Weeks) | Spread: {spread_return:+.2f}% | Immediate: {immediate_return:+.2f}%",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: IWN price
    ax = axes[0]
    backtest_df = df.iloc[start_idx:start_idx+15]
    ax.plot(backtest_df.index, backtest_df["Close"], color="#2196F3", linewidth=2, label="IWN Close")
    ax.fill_between(backtest_df.index, backtest_df["Low"], backtest_df["High"], alpha=0.2, color="#2196F3")

    ax.set_ylabel("IWN Price ($)", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("IWN Price Action (Backtest Period)", fontsize=11, fontweight="bold")

    # Panel 2: Portfolio value comparison
    ax = axes[1]
    ax.plot(spread_results["dates"], spread_results["total_value"],
            color="#4CAF50", linewidth=2, label="Spread Orders", marker="o", markersize=4)
    ax.plot(immediate_results["dates"], immediate_results["total_value"],
            color="#FF5722", linewidth=2, label="Immediate Execution", marker="s", markersize=4)
    ax.axhline(INITIAL_CAPITAL, color="black", linewidth=1, linestyle="--", alpha=0.5, label="Initial Capital")

    ax.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("Portfolio Value Over Time", fontsize=11, fontweight="bold")

    # Panel 3: Fill rate comparison
    ax = axes[2]
    x = np.arange(len(spread_results["fills"]))
    width = 0.35

    ax.bar(x - width/2, spread_results["fills"], width, color="#4CAF50", alpha=0.7, label="Spread Fills")
    ax.bar(x + width/2, immediate_results["fills"], width, color="#FF5722", alpha=0.7, label="Immediate Fills")

    ax.set_ylabel("Number of Fills", fontsize=10)
    ax.set_xlabel("Trading Day", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_title("Daily Fill Rate Comparison", fontsize=11, fontweight="bold")

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iwn_put_spread_backtest.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved backtest visualization to {out_path}")


if __name__ == "__main__":
    run_backtest()
