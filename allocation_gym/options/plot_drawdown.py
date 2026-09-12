"""
Plot BTC options hedging performance during the Jan-Feb 2026 drawdown.

Generates a multi-panel figure:
  1. BTC price with option roll periods + strike levels
  2. Portfolio value: unhedged vs protective put vs collar
  3. Drawdown curves for all three strategies
  4. Cumulative premium flows (paid, received, intrinsic recovered)
  5. Realized vol vs IV with vol regime shading

Usage:
    python -m allocation_gym.options.plot_drawdown
    python -m allocation_gym.options.plot_drawdown --iv 0.80 --otm-pct 0.10
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from allocation_gym.options.drawdown_analysis import (
    compute_realized_vol,
    compute_yang_zhang_vol,
    df_to_bars,
    find_drawdowns,
    load_btc_daily,
)
from allocation_gym.options.metrics import compute_options_metrics
from allocation_gym.options.simulation import OptionsStrategyType, run_options_simulation


def plot_drawdown_analysis(
    iv: float = 0.80,
    otm_pct: float = 0.05,
    roll_days: int = 21,
    lookback_days: int = 120,
):
    plt.close("all")

    # ── Load data ────────────────────────────────────────────────────
    print("Loading BTC/USD data...")
    df = load_btc_daily(days=lookback_days)
    bars = df_to_bars(df)
    dates = df.index
    prices = df["Close"].values
    initial_price = bars[0]["close"]

    # ── Run simulations ──────────────────────────────────────────────
    print("Running options simulations...")
    put_result = run_options_simulation(
        bars=bars, symbol="BTC/USD",
        strategy_type=OptionsStrategyType.PROTECTIVE_PUT,
        initial_shares=1, initial_price=initial_price,
        iv=iv, otm_pct=otm_pct, roll_period_days=roll_days,
    )
    collar_result = run_options_simulation(
        bars=bars, symbol="BTC/USD",
        strategy_type=OptionsStrategyType.COLLAR,
        initial_shares=1, initial_price=initial_price,
        iv=iv, otm_pct=otm_pct, roll_period_days=roll_days,
    )
    put_m = compute_options_metrics(put_result)
    collar_m = compute_options_metrics(collar_result)

    # Extract series
    put_values = np.array([s.net_portfolio_value for s in put_result.snapshots])
    collar_values = np.array([s.net_portfolio_value for s in collar_result.snapshots])
    unhedged_values = prices.copy()

    # Normalize to percentage returns
    put_ret = (put_values / put_values[0] - 1) * 100
    collar_ret = (collar_values / collar_values[0] - 1) * 100
    unhedged_ret = (unhedged_values / unhedged_values[0] - 1) * 100

    # Drawdown curves
    def drawdown_series(vals):
        peak = np.maximum.accumulate(vals)
        return (vals - peak) / peak * 100

    put_dd = drawdown_series(put_values)
    collar_dd = drawdown_series(collar_values)
    unhedged_dd = drawdown_series(unhedged_values)

    # Volatility
    rv_21 = compute_realized_vol(df, window=21)
    yz_14 = compute_yang_zhang_vol(df, window=14)

    # Premium flows
    cum_paid = np.array([s.cumulative_premium_paid for s in put_result.snapshots])
    cum_intrinsic = np.array([s.cumulative_put_intrinsic for s in put_result.snapshots])
    collar_received = np.array([s.cumulative_premium_received for s in collar_result.snapshots])
    collar_paid = np.array([s.cumulative_premium_paid for s in collar_result.snapshots])

    # ── Create figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(
        5, 1, figsize=(16, 22),
        height_ratios=[2.0, 1.5, 1.2, 1.2, 1.0],
        sharex=True,
    )
    fig.suptitle(
        f"BTC OPTIONS HEDGING vs DRAWDOWN  |  "
        f"IV={iv:.0%}  OTM={otm_pct:.0%}  Roll={roll_days}d  |  "
        f"{dates[0].strftime('%b %d')} – {dates[-1].strftime('%b %d, %Y')}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    C_BTC = "#F7931A"      # Bitcoin orange
    C_PUT = "#2196F3"      # Blue
    C_COLLAR = "#4CAF50"   # Green
    C_UNHEDGED = "#9E9E9E" # Grey
    C_STRIKE = "#E91E63"   # Pink
    C_CALL_STRIKE = "#FF9800"

    # ── Panel 1: BTC Price + Roll Periods + Strikes ──────────────────
    ax1 = axes[0]
    ax1.plot(dates, prices, color=C_BTC, linewidth=2, label="BTC/USD", zorder=3)
    ax1.fill_between(dates, prices, prices.min() * 0.95, alpha=0.06, color=C_BTC)

    # Roll periods and strikes
    for i, roll in enumerate(put_result.rolls):
        roll_date = pd.Timestamp(roll.roll_date)
        expiry_idx = min(roll.expiry_bar_idx, len(dates) - 1)
        expiry_date = dates[expiry_idx]

        # Shade roll period
        ax1.axvspan(roll_date, expiry_date, alpha=0.07, color=C_PUT, zorder=1)

        # Put strike line
        ax1.hlines(
            roll.put_strike, roll_date, expiry_date,
            colors=C_STRIKE, linestyles="--", linewidth=1.2, alpha=0.7, zorder=2,
        )
        # Label strike
        ax1.annotate(
            f"${roll.put_strike:,.0f}",
            xy=(roll_date, roll.put_strike),
            fontsize=7, color=C_STRIKE, alpha=0.8,
            va="bottom", ha="left",
        )

        # Collar call strike
        if roll.call_strike:
            ax1.hlines(
                roll.call_strike, roll_date, expiry_date,
                colors=C_CALL_STRIKE, linestyles=":", linewidth=1.0, alpha=0.6, zorder=2,
            )

        # ITM indicator
        expiry_price = bars[expiry_idx]["close"]
        if roll.put_intrinsic_at_expiry > 0:
            ax1.scatter(
                [expiry_date], [expiry_price],
                marker="v", color=C_STRIKE, s=80, zorder=5,
                edgecolors="black", linewidths=0.5,
            )
        else:
            ax1.scatter(
                [expiry_date], [expiry_price],
                marker="o", color=C_UNHEDGED, s=40, zorder=5,
                edgecolors="black", linewidths=0.5, alpha=0.5,
            )

    ax1.set_ylabel("BTC/USD Price", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.2, linestyle="--")

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_BTC, linewidth=2, label="BTC Price"),
        Line2D([0], [0], color=C_STRIKE, linewidth=1.2, linestyle="--", label="Put Strike"),
        Line2D([0], [0], color=C_CALL_STRIKE, linewidth=1, linestyle=":", label="Call Strike (collar)"),
        mpatches.Patch(facecolor=C_PUT, alpha=0.15, label="Roll Period"),
        Line2D([0], [0], marker="v", color=C_STRIKE, markersize=8, linestyle="None", label="Put ITM at Expiry"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    # Annotate peak and trough
    peak_idx = np.argmax(prices)
    trough_idx = np.argmin(prices)
    ax1.annotate(
        f"Peak\n${prices[peak_idx]:,.0f}",
        xy=(dates[peak_idx], prices[peak_idx]),
        xytext=(20, 15), textcoords="offset points",
        fontsize=9, fontweight="bold", color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", lw=1),
    )
    ax1.annotate(
        f"Trough\n${prices[trough_idx]:,.0f}",
        xy=(dates[trough_idx], prices[trough_idx]),
        xytext=(20, -25), textcoords="offset points",
        fontsize=9, fontweight="bold", color=C_STRIKE,
        arrowprops=dict(arrowstyle="->", color=C_STRIKE, lw=1),
    )

    # ── Panel 2: Cumulative Returns ──────────────────────────────────
    ax2 = axes[1]
    ax2.plot(dates, unhedged_ret, color=C_UNHEDGED, linewidth=1.8, label=f"Unhedged ({unhedged_ret[-1]:+.1f}%)", alpha=0.7)
    ax2.plot(dates, put_ret, color=C_PUT, linewidth=1.8, label=f"Protective Put ({put_ret[-1]:+.1f}%)")
    ax2.plot(dates, collar_ret, color=C_COLLAR, linewidth=2, label=f"Collar ({collar_ret[-1]:+.1f}%)")
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax2.fill_between(dates, collar_ret, unhedged_ret, alpha=0.1, color=C_COLLAR,
                     where=collar_ret > unhedged_ret, label="_nolegend_")
    ax2.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2, linestyle="--")

    # Annotate the spread
    final_spread = collar_ret[-1] - unhedged_ret[-1]
    ax2.annotate(
        f"Collar outperformance\n{final_spread:+.1f}pp",
        xy=(dates[-1], (collar_ret[-1] + unhedged_ret[-1]) / 2),
        xytext=(-120, 0), textcoords="offset points",
        fontsize=9, fontweight="bold", color=C_COLLAR,
        arrowprops=dict(arrowstyle="->", color=C_COLLAR, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_COLLAR, alpha=0.9),
    )

    # ── Panel 3: Drawdown Curves ─────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, unhedged_dd, 0, alpha=0.15, color=C_UNHEDGED, label=f"Unhedged (max {unhedged_dd.min():.1f}%)")
    ax3.fill_between(dates, put_dd, 0, alpha=0.2, color=C_PUT, label=f"Put (max {put_dd.min():.1f}%)")
    ax3.fill_between(dates, collar_dd, 0, alpha=0.25, color=C_COLLAR, label=f"Collar (max {collar_dd.min():.1f}%)")
    ax3.plot(dates, unhedged_dd, color=C_UNHEDGED, linewidth=1, alpha=0.5)
    ax3.plot(dates, put_dd, color=C_PUT, linewidth=1.2)
    ax3.plot(dates, collar_dd, color=C_COLLAR, linewidth=1.5)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.2, linestyle="--")

    # ── Panel 4: Premium Flows ───────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(dates, -cum_paid, color=C_STRIKE, linewidth=1.5, label="Put Premium Paid")
    ax4.plot(dates, cum_intrinsic, color=C_PUT, linewidth=1.5, label="Put Intrinsic Recovered")
    ax4.plot(dates, collar_received, color=C_COLLAR, linewidth=1.5, linestyle="--", label="Call Premium Received (collar)")

    net_cost_put = cum_intrinsic - cum_paid
    net_cost_collar = cum_intrinsic - collar_paid + collar_received
    ax4.plot(dates, net_cost_put, color=C_PUT, linewidth=1, linestyle=":", alpha=0.7, label="Net Cost (put)")
    ax4.plot(dates, net_cost_collar, color=C_COLLAR, linewidth=1, linestyle=":", alpha=0.7, label="Net Cost (collar)")

    ax4.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax4.fill_between(dates, net_cost_collar, 0, alpha=0.08, color=C_COLLAR,
                     where=net_cost_collar > 0)
    ax4.set_ylabel("Cumulative $ / share", fontsize=11)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax4.grid(True, alpha=0.2, linestyle="--")

    # ── Panel 5: Realized Vol + IV ───────────────────────────────────
    ax5 = axes[4]
    ax5.plot(dates, rv_21.values * 100, color="#9C27B0", linewidth=1.5, label="21d Realized Vol")
    ax5.plot(dates, yz_14.values * 100, color="#FF5722", linewidth=1.2, alpha=0.7, label="14d Yang-Zhang Vol")
    ax5.axhline(iv * 100, color="black", linewidth=1, linestyle="--", alpha=0.6, label=f"IV Assumption ({iv:.0%})")

    # Shade IV-RV spread
    rv_vals = rv_21.values * 100
    iv_line = np.full_like(rv_vals, iv * 100)
    ax5.fill_between(
        dates, rv_vals, iv_line,
        where=rv_vals > iv_line, alpha=0.15, color="#F44336",
        label="RV > IV (options cheap)",
    )
    ax5.fill_between(
        dates, rv_vals, iv_line,
        where=rv_vals < iv_line, alpha=0.15, color="#4CAF50",
        label="RV < IV (options rich)",
    )

    ax5.set_ylabel("Annualized Vol (%)", fontsize=11)
    ax5.set_xlabel("Date", fontsize=11)
    ax5.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax5.grid(True, alpha=0.2, linestyle="--")
    ax5.set_ylim(0, max(rv_vals[~np.isnan(rv_vals)].max(), iv * 100) * 1.3)

    # Format x-axis
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax5.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    # ── Summary stats box ────────────────────────────────────────────
    stats_text = (
        f"SUMMARY\n"
        f"{'─' * 40}\n"
        f"BTC:  ${prices[0]:,.0f} → ${prices[-1]:,.0f}  ({unhedged_ret[-1]:+.1f}%)\n"
        f"Peak: ${prices.max():,.0f}  Trough: ${prices.min():,.0f}\n"
        f"{'─' * 40}\n"
        f"         {'Return':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}\n"
        f"Unhedged {unhedged_ret[-1]:>+7.1f}%  {unhedged_dd.min():>6.1f}%       —\n"
        f"Put      {put_ret[-1]:>+7.1f}%  {put_dd.min():>6.1f}%  {put_m.sharpe_ratio:>6.3f}\n"
        f"Collar   {collar_ret[-1]:>+7.1f}%  {collar_dd.min():>6.1f}%  {collar_m.sharpe_ratio:>6.3f}\n"
        f"{'─' * 40}\n"
        f"Collar DD reduction: {abs(unhedged_dd.min()) - abs(collar_dd.min()):+.1f}pp"
    )
    fig.text(
        0.015, 0.01, stats_text,
        fontsize=8, fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.99])

    # Save
    out_path = "docs/4/btc_options_drawdown.png"
    import os
    os.makedirs("docs/4", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    plt.show(block=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot BTC options vs drawdown")
    parser.add_argument("--iv", type=float, default=0.80, help="Implied vol (default: 0.80)")
    parser.add_argument("--otm-pct", type=float, default=0.05, help="OTM pct (default: 0.05)")
    parser.add_argument("--roll-days", type=int, default=21, help="Roll period (default: 21)")
    parser.add_argument("--days", type=int, default=120, help="Lookback days (default: 120)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_drawdown_analysis(
        iv=args.iv,
        otm_pct=args.otm_pct,
        roll_days=args.roll_days,
        lookback_days=args.days,
    )
