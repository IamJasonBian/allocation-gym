"""
IWN 6-month analysis with rolling correlations to BTC, QQQ, SPY.

Generates two plots:
1. 6-month IWN price zoom with volatility
2. Rolling correlation analysis (30-day and 60-day windows)

Usage:
    python -m allocation_gym.options.iwn_correlation
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from allocation_gym.options.black_scholes import bs_call_price, bs_put_price


# ── Data Loading ─────────────────────────────────────────────────────────


def load_iwn(days: int = 180) -> pd.DataFrame:
    """Load IWN data for last 6 months"""
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


def load_btc(days: int = 180) -> pd.DataFrame:
    """Load BTC data"""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.Ticker("BTC-USD").history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise RuntimeError("No BTC data from yfinance")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Close"]].rename(columns={"Close": "BTC"})


def load_qqq(days: int = 180) -> pd.DataFrame:
    """Load QQQ data"""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.Ticker("QQQ").history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise RuntimeError("No QQQ data from yfinance")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Close"]].rename(columns={"Close": "QQQ"})


def load_spy(days: int = 180) -> pd.DataFrame:
    """Load SPY (S&P 500) data"""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.Ticker("SPY").history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise RuntimeError("No SPY data from yfinance")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Close"]].rename(columns={"Close": "SPY"})


# ── Volatility & Correlation ─────────────────────────────────────────────


def realized_vol(closes: np.ndarray, trading_days: int = 252) -> float:
    """Annualized close-to-close realized vol."""
    log_ret = np.log(closes[1:] / closes[:-1])
    return float(np.std(log_ret, ddof=1) * np.sqrt(trading_days))


def rolling_correlation(df: pd.DataFrame, col1: str, col2: str, window: int) -> pd.Series:
    """Compute rolling correlation between two columns"""
    return df[col1].pct_change().rolling(window).corr(df[col2].pct_change())


# ── Main Analysis ────────────────────────────────────────────────────────


def run_analysis():
    print("=" * 70)
    print("IWN 6-MONTH ANALYSIS + ROLLING CORRELATIONS")
    print("=" * 70)

    # Load data
    print("\nLoading market data (last 6 months)...")
    iwn_df = load_iwn(days=180)
    btc_df = load_btc(days=180)
    qqq_df = load_qqq(days=180)
    spy_df = load_spy(days=180)

    # Merge all data on date index
    combined = iwn_df[["Close"]].rename(columns={"Close": "IWN"})
    combined = combined.join(btc_df, how="inner")
    combined = combined.join(qqq_df, how="inner")
    combined = combined.join(spy_df, how="inner")

    print(f"  {len(combined)} trading days from {combined.index[0].date()} to {combined.index[-1].date()}")
    print(f"\n  Current prices:")
    print(f"    IWN: ${combined['IWN'].iloc[-1]:.2f}")
    print(f"    BTC: ${combined['BTC'].iloc[-1]:,.2f}")
    print(f"    QQQ: ${combined['QQQ'].iloc[-1]:.2f}")
    print(f"    SPY: ${combined['SPY'].iloc[-1]:.2f}")

    # Calculate rolling correlations
    print("\n" + "=" * 70)
    print("ROLLING CORRELATIONS (30-day and 60-day windows)")
    print("=" * 70)

    corr_30d_btc = rolling_correlation(combined, "IWN", "BTC", 30)
    corr_30d_qqq = rolling_correlation(combined, "IWN", "QQQ", 30)
    corr_30d_spy = rolling_correlation(combined, "IWN", "SPY", 30)

    corr_60d_btc = rolling_correlation(combined, "IWN", "BTC", 60)
    corr_60d_qqq = rolling_correlation(combined, "IWN", "QQQ", 60)
    corr_60d_spy = rolling_correlation(combined, "IWN", "SPY", 60)

    # Latest correlations
    print(f"\n  Current 30-day correlations:")
    print(f"    IWN vs BTC: {corr_30d_btc.iloc[-1]:.3f}")
    print(f"    IWN vs QQQ: {corr_30d_qqq.iloc[-1]:.3f}")
    print(f"    IWN vs SPY: {corr_30d_spy.iloc[-1]:.3f}")

    print(f"\n  Current 60-day correlations:")
    print(f"    IWN vs BTC: {corr_60d_btc.iloc[-1]:.3f}")
    print(f"    IWN vs QQQ: {corr_60d_qqq.iloc[-1]:.3f}")
    print(f"    IWN vs SPY: {corr_60d_spy.iloc[-1]:.3f}")

    # Calculate rolling volatility for IWN
    rolling_vol_30d = combined["IWN"].pct_change().rolling(30).std() * np.sqrt(252) * 100
    rolling_vol_60d = combined["IWN"].pct_change().rolling(60).std() * np.sqrt(252) * 100

    print(f"\n  Current IWN volatility:")
    print(f"    30-day: {rolling_vol_30d.iloc[-1]:.1f}%")
    print(f"    60-day: {rolling_vol_60d.iloc[-1]:.1f}%")

    # Generate plots
    _plot_6month_zoom(iwn_df, combined, rolling_vol_30d, rolling_vol_60d)
    _plot_correlations(
        combined,
        corr_30d_btc, corr_30d_qqq, corr_30d_spy,
        corr_60d_btc, corr_60d_qqq, corr_60d_spy,
    )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


def _plot_6month_zoom(iwn_df, combined, rolling_vol_30d, rolling_vol_60d):
    """Plot 1: 6-month IWN price and volatility zoom"""
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"IWN 6-Month Analysis | Price: ${combined['IWN'].iloc[-1]:.2f}",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: Price
    ax = axes[0]
    dates = combined.index
    ax.plot(dates, combined["IWN"], color="#2196F3", linewidth=2, label="IWN Close")
    ax.fill_between(dates, combined["IWN"], alpha=0.2, color="#2196F3")

    ax.set_ylabel("IWN Price ($)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("IWN Price (Last 6 Months)", fontsize=12, fontweight="bold")

    # Annotate current price
    ax.annotate(
        f"${combined['IWN'].iloc[-1]:.2f}",
        xy=(dates[-1], combined["IWN"].iloc[-1]),
        fontsize=9, color="#2196F3", fontweight="bold",
        ha="right", va="bottom",
    )

    # Panel 2: Rolling Volatility
    ax = axes[1]
    ax.plot(dates, rolling_vol_30d, color="#FF9800", linewidth=2, label="30-day Vol", alpha=0.8)
    ax.plot(dates, rolling_vol_60d, color="#9C27B0", linewidth=2, label="60-day Vol")

    ax.set_ylabel("Annualized Vol (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("IWN Rolling Realized Volatility", fontsize=12, fontweight="bold")

    # Annotate current vol
    if not rolling_vol_30d.empty and not np.isnan(rolling_vol_30d.iloc[-1]):
        ax.annotate(
            f"{rolling_vol_30d.iloc[-1]:.1f}%",
            xy=(dates[-1], rolling_vol_30d.iloc[-1]),
            fontsize=8, fontweight="bold", color="#FF9800",
            xytext=(-40, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#FF9800", lw=0.8),
        )

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iwn_6month_zoom.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved 6-month zoom plot to {out_path}")


def _plot_correlations(
    combined,
    corr_30d_btc, corr_30d_qqq, corr_30d_spy,
    corr_60d_btc, corr_60d_qqq, corr_60d_spy,
):
    """Plot 2: Rolling correlations with BTC, QQQ, SPY"""
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "IWN Rolling Correlations (BTC, QQQ, SPY)",
        fontsize=14, fontweight="bold",
    )

    dates = combined.index

    # Panel 1: 30-day correlations
    ax = axes[0]
    ax.plot(dates, corr_30d_btc, color="#F7931A", linewidth=2, label="IWN vs BTC", alpha=0.8)
    ax.plot(dates, corr_30d_qqq, color="#00AA00", linewidth=2, label="IWN vs QQQ")
    ax.plot(dates, corr_30d_spy, color="#0066CC", linewidth=2, label="IWN vs SPY")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_ylim(-1, 1)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("30-Day Rolling Correlation", fontsize=12, fontweight="bold")

    # Panel 2: 60-day correlations
    ax = axes[1]
    ax.plot(dates, corr_60d_btc, color="#F7931A", linewidth=2, label="IWN vs BTC", alpha=0.8)
    ax.plot(dates, corr_60d_qqq, color="#00AA00", linewidth=2, label="IWN vs QQQ")
    ax.plot(dates, corr_60d_spy, color="#0066CC", linewidth=2, label="IWN vs SPY")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylim(-1, 1)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("60-Day Rolling Correlation", fontsize=12, fontweight="bold")

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iwn_rolling_correlations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved correlation plot to {out_path}")


if __name__ == "__main__":
    run_analysis()
