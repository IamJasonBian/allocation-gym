"""
BTC + IWM call options: 2-week backtest + 30-day forward projection.

Backtests the last 14 days of actual data, then uses GBM Monte Carlo
to project 30 days forward with options overlays. Generates a combined
plot showing historical + projected performance.

Usage:
    python -m allocation_gym.options.backtest_forward
    python -m allocation_gym.options.backtest_forward --iv-btc 0.80 --iv-iwm 0.25
"""

import argparse
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from allocation_gym.credentials import get_alpaca_keys
from allocation_gym.options.black_scholes import bs_call_price, bs_put_price
from allocation_gym.options.simulation import (
    OptionsStrategyType,
    run_options_simulation,
)
from allocation_gym.options.metrics import compute_options_metrics
from allocation_gym.simulation.engine import MonteCarloGBM


# ── Data ─────────────────────────────────────────────────────────────────


def load_asset_daily(symbol: str, days: int = 45) -> pd.DataFrame:
    """Fetch daily bars: Alpaca for crypto, yfinance for stocks."""
    is_crypto = "/" in symbol or symbol.upper() in {"BTC", "BTC/USD", "ETH/USD"}

    end = datetime.now(tz=None)
    start = end - timedelta(days=int(days * 1.8))

    if is_crypto:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key, secret_key = get_alpaca_keys()
        if not api_key or not secret_key:
            raise RuntimeError("Alpaca credentials required for crypto")

        client = CryptoHistoricalDataClient(api_key, secret_key)
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol, start=start, end=end,
            timeframe=TimeFrame.Day,
        )
        bars = client.get_crypto_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
    else:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"))
        if df.empty:
            raise RuntimeError(f"No data from yfinance for {symbol}")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(days)


def df_to_bars(df: pd.DataFrame) -> List[Dict]:
    out = []
    for idx, row in df.iterrows():
        out.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        })
    return out


# ── Calibration ──────────────────────────────────────────────────────────


def calibrate_from_df(df: pd.DataFrame, trading_days: int = 365) -> dict:
    """Quick GBM calibration from a DataFrame."""
    closes = df["Close"].values
    log_ret = np.log(closes[1:] / closes[:-1])
    mu = float(np.mean(log_ret)) * trading_days
    sigma = float(np.std(log_ret, ddof=1)) * np.sqrt(trading_days)
    return {"mu": mu, "sigma": sigma, "S0": float(closes[-1])}


# ── Forward projection with options ─────────────────────────────────────


def project_with_options(
    S0: float,
    mu: float,
    sigma: float,
    iv: float,
    otm_pct: float,
    n_days: int = 30,
    n_paths: int = 500,
    strategy: str = "call",
    trading_days: int = 365,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo forward projection with call/put overlay.

    Returns dict with paths, option_pnl, hedged_paths, and percentile bands.
    """
    mc = MonteCarloGBM(mu=mu, sigma=sigma, initial_price=S0)
    result = mc.simulate(n_paths=n_paths, n_days=n_days, seed=seed)
    paths = result.paths  # (n_paths, n_days+1)

    # For each path, compute option P&L over the period
    # Strategy: buy a call (or put) at inception, hold to end
    T = n_days / trading_days
    r = 0.05

    if strategy == "call":
        strike = S0 * (1 + otm_pct)
        premium = bs_call_price(S0, strike, T, r, iv)
        # Terminal payoff per path
        payoffs = np.maximum(paths[:, -1] - strike, 0) - premium
    else:  # put
        strike = S0 * (1 - otm_pct)
        premium = bs_put_price(S0, strike, T, r, iv)
        payoffs = np.maximum(strike - paths[:, -1], 0) - premium

    # Daily mark-to-market of options along each path
    option_marks = np.zeros_like(paths)
    for day in range(n_days + 1):
        T_rem = max((n_days - day) / trading_days, 1 / trading_days)
        prices_at_day = paths[:, day]
        if strategy == "call":
            for p in range(n_paths):
                option_marks[p, day] = bs_call_price(
                    prices_at_day[p], strike, T_rem, r, iv
                ) - premium
        else:
            for p in range(n_paths):
                option_marks[p, day] = bs_put_price(
                    prices_at_day[p], strike, T_rem, r, iv
                ) - premium

    # Hedged value = underlying + option P&L (per unit)
    hedged_paths = paths + option_marks

    # Percentiles
    pcts = [5, 10, 25, 50, 75, 90, 95]
    underlying_bands = {p: np.percentile(paths, p, axis=0) for p in pcts}
    hedged_bands = {p: np.percentile(hedged_paths, p, axis=0) for p in pcts}

    return {
        "paths": paths,
        "option_marks": option_marks,
        "hedged_paths": hedged_paths,
        "payoffs": payoffs,
        "strike": strike,
        "premium": premium,
        "underlying_bands": underlying_bands,
        "hedged_bands": hedged_bands,
        "strategy": strategy,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def run_backtest_forward(
    iv_btc: float = 0.80,
    iv_iwm: float = 0.25,
    otm_pct: float = 0.05,
    fwd_days: int = 30,
    n_paths: int = 500,
):
    print("=" * 70)
    print("BTC + IWM CALL OPTIONS: 2-WEEK BACKTEST + FORWARD PROJECTION")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    print("\nLoading data...")
    btc_df = load_asset_daily("BTC/USD", days=45)
    iwm_df = load_asset_daily("IWM", days=45)

    # Last 14 calendar days for backtest window
    cutoff = btc_df.index[-1] - pd.Timedelta(days=14)
    btc_bt = btc_df[btc_df.index >= cutoff]
    iwm_bt = iwm_df[iwm_df.index >= cutoff]

    # Full history for calibration
    btc_cal = calibrate_from_df(btc_df, trading_days=365)
    iwm_cal = calibrate_from_df(iwm_df, trading_days=252)

    print(f"  BTC: {len(btc_bt)} bars backtest, cal mu={btc_cal['mu']:+.1%} sigma={btc_cal['sigma']:.1%} S0=${btc_cal['S0']:,.0f}")
    print(f"  IWM: {len(iwm_bt)} bars backtest, cal mu={iwm_cal['mu']:+.1%} sigma={iwm_cal['sigma']:.1%} S0=${iwm_cal['S0']:.2f}")

    # ── Backtest: run call option simulation on last 2 weeks ─────────
    print("\nBacktesting last 2 weeks...")

    # BTC call backtest
    btc_bars = df_to_bars(btc_bt)
    btc_call_result = run_options_simulation(
        bars=btc_bars, symbol="BTC/USD",
        strategy_type=OptionsStrategyType.COLLAR,  # Use collar to get both call and put
        initial_shares=1, initial_price=btc_bars[0]["close"],
        iv=iv_btc, otm_pct=otm_pct, roll_period_days=max(len(btc_bars) - 1, 7),
    )

    # IWM call backtest
    iwm_bars = df_to_bars(iwm_bt)
    iwm_call_result = run_options_simulation(
        bars=iwm_bars, symbol="IWM",
        strategy_type=OptionsStrategyType.COLLAR,
        initial_shares=1, initial_price=iwm_bars[0]["close"],
        iv=iv_iwm, otm_pct=otm_pct, roll_period_days=max(len(iwm_bars) - 1, 7),
    )

    # Print backtest summaries
    for label, res in [("BTC/USD", btc_call_result), ("IWM", iwm_call_result)]:
        m = compute_options_metrics(res)
        print(f"\n  {label} (last 2wk):")
        print(f"    Return: {m.total_return_pct:+.2f}%  MaxDD: {m.max_drawdown_pct:.2f}%  Sharpe: {m.sharpe_ratio:.3f}")
        print(f"    Put prem: ${m.total_premium_paid:,.2f}  Call rcvd: ${m.total_premium_received:,.2f}")
        for roll in res.rolls:
            exp_idx = min(roll.expiry_bar_idx, len(res.snapshots) - 1)
            exp_price = res.snapshots[exp_idx].underlying_price
            call_status = ""
            if roll.call_strike:
                call_status = f"  Call K=${roll.call_strike:,.2f} ({'ITM' if exp_price > roll.call_strike else 'OTM'})"
            print(f"    Roll {roll.roll_date}: Put K=${roll.put_strike:,.2f} ({'ITM' if exp_price < roll.put_strike else 'OTM'}){call_status}")

    # ── Forward projection ───────────────────────────────────────────
    print(f"\nProjecting {fwd_days} days forward ({n_paths} MC paths)...")

    btc_fwd = project_with_options(
        S0=btc_cal["S0"], mu=btc_cal["mu"], sigma=btc_cal["sigma"],
        iv=iv_btc, otm_pct=otm_pct, n_days=fwd_days, n_paths=n_paths,
        strategy="call", trading_days=365, seed=42,
    )
    iwm_fwd = project_with_options(
        S0=iwm_cal["S0"], mu=iwm_cal["mu"], sigma=iwm_cal["sigma"],
        iv=iv_iwm, otm_pct=otm_pct, n_days=fwd_days, n_paths=n_paths,
        strategy="call", trading_days=252, seed=43,
    )

    # Print forward stats
    for label, fwd, iv_val in [("BTC/USD", btc_fwd, iv_btc), ("IWM", iwm_fwd, iv_iwm)]:
        final_underlying = fwd["paths"][:, -1]
        final_hedged = fwd["hedged_paths"][:, -1]
        S0 = fwd["paths"][0, 0]
        print(f"\n  {label} Forward ({fwd_days}d, IV={iv_val:.0%}):")
        print(f"    Strike: ${fwd['strike']:,.2f} ({otm_pct:.0%} OTM call)  Premium: ${fwd['premium']:,.2f}")
        print(f"    Underlying P50: ${np.median(final_underlying):,.2f} ({(np.median(final_underlying)/S0-1)*100:+.1f}%)")
        print(f"    With call P50:  ${np.median(final_hedged):,.2f} ({(np.median(final_hedged)/S0-1)*100:+.1f}%)")
        print(f"    Call payoff P50: ${np.median(fwd['payoffs']):,.2f}  P90: ${np.percentile(fwd['payoffs'], 90):,.2f}")
        print(f"    P(call profitable): {np.mean(fwd['payoffs'] > 0)*100:.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────
    print("\nGenerating plot...")
    _plot_combined(
        btc_bt=btc_bt, iwm_bt=iwm_bt,
        btc_call_result=btc_call_result, iwm_call_result=iwm_call_result,
        btc_fwd=btc_fwd, iwm_fwd=iwm_fwd,
        btc_cal=btc_cal, iwm_cal=iwm_cal,
        iv_btc=iv_btc, iv_iwm=iv_iwm,
        otm_pct=otm_pct, fwd_days=fwd_days,
    )


def _plot_combined(
    btc_bt, iwm_bt,
    btc_call_result, iwm_call_result,
    btc_fwd, iwm_fwd,
    btc_cal, iwm_cal,
    iv_btc, iv_iwm,
    otm_pct, fwd_days,
):
    plt.close("all")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"BTC + IWM CALL OPTIONS  |  2-Week Backtest + {fwd_days}d Forward  |  "
        f"BTC IV={iv_btc:.0%}  IWM IV={iv_iwm:.0%}  OTM={otm_pct:.0%}",
        fontsize=13, fontweight="bold",
    )

    C_UNDERLYING = "#F7931A"
    C_HEDGED = "#2196F3"
    C_IWM = "#4CAF50"
    C_IWM_HEDGED = "#9C27B0"

    # ── Top-left: BTC backtest + forward ─────────────────────────────
    ax = axes[0, 0]
    _plot_asset_panel(
        ax, "BTC/USD", btc_bt, btc_call_result, btc_fwd, btc_cal,
        C_UNDERLYING, C_HEDGED, fwd_days,
    )

    # ── Top-right: IWM backtest + forward ────────────────────────────
    ax = axes[0, 1]
    _plot_asset_panel(
        ax, "IWM", iwm_bt, iwm_call_result, iwm_fwd, iwm_cal,
        C_IWM, C_IWM_HEDGED, fwd_days,
    )

    # ── Bottom-left: Forward return distributions ────────────────────
    ax = axes[1, 0]
    btc_S0 = btc_fwd["paths"][0, 0]
    iwm_S0 = iwm_fwd["paths"][0, 0]

    btc_ret_underlying = (btc_fwd["paths"][:, -1] / btc_S0 - 1) * 100
    btc_ret_hedged = (btc_fwd["hedged_paths"][:, -1] / btc_S0 - 1) * 100
    iwm_ret_underlying = (iwm_fwd["paths"][:, -1] / iwm_S0 - 1) * 100
    iwm_ret_hedged = (iwm_fwd["hedged_paths"][:, -1] / iwm_S0 - 1) * 100

    bins = np.linspace(-50, 60, 60)
    ax.hist(btc_ret_underlying, bins=bins, alpha=0.3, color=C_UNDERLYING, label=f"BTC spot (med {np.median(btc_ret_underlying):+.1f}%)", density=True)
    ax.hist(btc_ret_hedged, bins=bins, alpha=0.4, color=C_HEDGED, label=f"BTC+call (med {np.median(btc_ret_hedged):+.1f}%)", density=True)
    ax.hist(iwm_ret_underlying, bins=bins, alpha=0.3, color=C_IWM, label=f"IWM spot (med {np.median(iwm_ret_underlying):+.1f}%)", density=True)
    ax.hist(iwm_ret_hedged, bins=bins, alpha=0.4, color=C_IWM_HEDGED, label=f"IWM+call (med {np.median(iwm_ret_hedged):+.1f}%)", density=True)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Return (%)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{fwd_days}-Day Forward Return Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── Bottom-right: Call option P&L distribution ───────────────────
    ax = axes[1, 1]
    btc_payoffs = btc_fwd["payoffs"]
    iwm_payoffs = iwm_fwd["payoffs"]

    bins_pnl = 50
    ax.hist(btc_payoffs, bins=bins_pnl, alpha=0.5, color=C_HEDGED,
            label=f"BTC call (med ${np.median(btc_payoffs):,.0f}, P(profit)={np.mean(btc_payoffs>0)*100:.0f}%)", density=True)
    ax.hist(iwm_payoffs, bins=bins_pnl, alpha=0.5, color=C_IWM_HEDGED,
            label=f"IWM call (med ${np.median(iwm_payoffs):,.2f}, P(profit)={np.mean(iwm_payoffs>0)*100:.0f}%)", density=True)

    ax.axvline(0, color="black", linewidth=1, alpha=0.6)
    ax.axvline(-btc_fwd["premium"], color=C_HEDGED, linewidth=1, linestyle="--", alpha=0.6, label=f"BTC prem -${btc_fwd['premium']:,.0f}")
    ax.axvline(-iwm_fwd["premium"], color=C_IWM_HEDGED, linewidth=1, linestyle="--", alpha=0.6, label=f"IWM prem -${iwm_fwd['premium']:,.2f}")

    ax.set_xlabel("Call Option P&L ($)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{fwd_days}-Day Call Payoff Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    # Add summary text
    summary = (
        f"BTC: ${btc_S0:,.0f} | Call K=${btc_fwd['strike']:,.0f} | Prem ${btc_fwd['premium']:,.0f}\n"
        f"IWM: ${iwm_S0:.2f} | Call K=${iwm_fwd['strike']:.2f} | Prem ${iwm_fwd['premium']:.2f}"
    )
    ax.annotate(
        summary, xy=(0.98, 0.98), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#FBC02D", alpha=0.95),
    )

    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "docs", "btc_iwm_call_forward.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    plt.show(block=True)


def _plot_asset_panel(ax, label, bt_df, call_result, fwd, cal, c_spot, c_hedged, fwd_days):
    """Plot one asset: backtest period + forward fan chart."""
    # Backtest period
    bt_dates = bt_df.index
    bt_prices = bt_df["Close"].values
    bt_hedged = np.array([s.net_portfolio_value for s in call_result.snapshots])

    # Normalize both to percentage return from start of backtest
    bt_base = bt_prices[0]
    bt_ret = (bt_prices / bt_base - 1) * 100
    bt_hedged_ret = (bt_hedged / bt_hedged[0] - 1) * 100

    ax.plot(bt_dates, bt_ret, color=c_spot, linewidth=2, label=f"{label} spot")
    ax.plot(bt_dates, bt_hedged_ret, color=c_hedged, linewidth=2, linestyle="--", label=f"{label} + collar")

    # Forward period: fan chart
    last_date = bt_dates[-1]
    fwd_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=fwd_days + 1)
    if len(fwd_dates) > fwd_days + 1:
        fwd_dates = fwd_dates[:fwd_days + 1]

    # Pad to match fwd path length
    S0 = fwd["paths"][0, 0]
    fwd_len = fwd["underlying_bands"][50].shape[0]
    if len(fwd_dates) < fwd_len:
        extra = pd.bdate_range(start=fwd_dates[-1] + pd.Timedelta(days=1), periods=fwd_len - len(fwd_dates))
        fwd_dates = fwd_dates.append(extra)
    fwd_dates = fwd_dates[:fwd_len]

    # Convert forward bands to return % from backtest start
    for pct_lo, pct_hi, alpha in [(5, 95, 0.08), (10, 90, 0.12), (25, 75, 0.18)]:
        lo = (fwd["underlying_bands"][pct_lo] / bt_base - 1) * 100
        hi = (fwd["underlying_bands"][pct_hi] / bt_base - 1) * 100
        ax.fill_between(fwd_dates, lo, hi, alpha=alpha, color=c_spot)

    # Median forward path
    med = (fwd["underlying_bands"][50] / bt_base - 1) * 100
    ax.plot(fwd_dates, med, color=c_spot, linewidth=1.5, alpha=0.6, linestyle=":")

    # Hedged forward bands
    for pct_lo, pct_hi, alpha in [(10, 90, 0.1), (25, 75, 0.15)]:
        lo = (fwd["hedged_bands"][pct_lo] / bt_base - 1) * 100
        hi = (fwd["hedged_bands"][pct_hi] / bt_base - 1) * 100
        ax.fill_between(fwd_dates, lo, hi, alpha=alpha, color=c_hedged)

    med_h = (fwd["hedged_bands"][50] / bt_base - 1) * 100
    ax.plot(fwd_dates, med_h, color=c_hedged, linewidth=1.5, alpha=0.6, linestyle=":")

    # Divider line
    ax.axvline(last_date, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.annotate("BACKTEST | FORWARD", xy=(last_date, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 5),
                fontsize=7, alpha=0.6, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Strike line
    strike_ret = (fwd["strike"] / bt_base - 1) * 100
    ax.axhline(strike_ret, color=c_hedged, linewidth=0.8, linestyle=":", alpha=0.4)
    ax.annotate(f"Call K=${fwd['strike']:,.0f}" if fwd["strike"] > 1000 else f"Call K=${fwd['strike']:.2f}",
                xy=(fwd_dates[-1], strike_ret), fontsize=7, color=c_hedged, alpha=0.7, ha="right")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Return from 2wk start (%)", fontsize=10)
    ax.set_title(f"{label}: Backtest + {fwd_days}d Forward (Call {otm_pct*100:.0f}% OTM)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Annotate current price
    ax.annotate(
        f"Now ${cal['S0']:,.0f}" if cal["S0"] > 1000 else f"Now ${cal['S0']:.2f}",
        xy=(last_date, bt_ret[-1]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.9),
    )


# allow reference from _plot_asset_panel closure
otm_pct = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="BTC + IWM Call Backtest + Forward")
    parser.add_argument("--iv-btc", type=float, default=0.80, help="BTC IV (default: 0.80)")
    parser.add_argument("--iv-iwm", type=float, default=0.25, help="IWM IV (default: 0.25)")
    parser.add_argument("--otm-pct", type=float, default=0.05, help="OTM pct (default: 0.05)")
    parser.add_argument("--fwd-days", type=int, default=30, help="Forward days (default: 30)")
    parser.add_argument("--n-paths", type=int, default=500, help="MC paths (default: 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    otm_pct = args.otm_pct
    run_backtest_forward(
        iv_btc=args.iv_btc,
        iv_iwm=args.iv_iwm,
        otm_pct=args.otm_pct,
        fwd_days=args.fwd_days,
        n_paths=args.n_paths,
    )
