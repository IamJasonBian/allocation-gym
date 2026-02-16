"""
Backtest result plotting — equity curve, P&L, and trades on price.
Auto-shows on every run. Use --no-plot to disable.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_backtest(analyzer, cerebro_result, strategy_name="", symbols=None):
    """
    Plot 3-panel backtest results:
      1. Price chart with buy/sell fill markers per symbol
      2. Equity curve with drawdown shading
      3. Daily P&L bars + cumulative P&L line

    Args:
        analyzer: PerformanceAnalyzer instance
        cerebro_result: The strategy result object (results[0])
        strategy_name: Label for the title
        symbols: List of symbol names
    """
    dates, values = analyzer.get_equity_curve()
    orders = analyzer.get_orders()
    perf = analyzer.get_analysis()

    if not dates or not values:
        print("No data to plot.")
        return

    values = np.array(values, dtype=float)
    symbols = symbols or []

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2.5, 1, 1],
                             sharex=True)
    fig.suptitle(
        f"{strategy_name.upper()}  |  "
        f"Return: {perf.get('total_return_pct', 0):+.1f}%  "
        f"Sharpe: {perf.get('sharpe', 0):.2f}  "
        f"Max DD: {perf.get('max_drawdown_pct', 0):.1f}%  "
        f"Orders: {perf.get('total_orders', 0)} "
        f"({perf.get('buy_orders', 0)}B / {perf.get('sell_orders', 0)}S)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Price lines + order fill markers ──
    ax1 = axes[0]
    strat = cerebro_result
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]

    for i, data in enumerate(strat.datas):
        name = data._name
        price_dates = []
        price_vals = []
        for j in range(-len(data) + 1, 1):
            try:
                dt = data.datetime.date(j)
                cl = data.close[j]
                price_dates.append(dt)
                price_vals.append(cl)
            except IndexError:
                continue

        color = colors[i % len(colors)]
        ax1.plot(price_dates, price_vals, label=name, color=color,
                 linewidth=1.3, alpha=0.9)

    # Overlay order fills as markers
    buy_dates = [o["dt"] for o in orders if o["side"] == "buy"]
    buy_prices = [o["price"] for o in orders if o["side"] == "buy"]
    sell_dates = [o["dt"] for o in orders if o["side"] == "sell"]
    sell_prices = [o["price"] for o in orders if o["side"] == "sell"]

    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, marker="^", color="#00c853", s=50,
                    zorder=5, label="Buy", edgecolors="black", linewidths=0.4)
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, marker="v", color="#ff1744", s=50,
                    zorder=5, label="Sell", edgecolors="black", linewidths=0.4)

    ax1.set_ylabel("Price ($)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=min(len(symbols) + 2, 6),
               framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle="--")

    # ── Panel 2: Equity curve + drawdown ──
    ax2 = axes[1]
    ax2.plot(dates, values, color="steelblue", linewidth=1.5)
    ax2.fill_between(dates, values[0], values, where=values >= values[0],
                     color="steelblue", alpha=0.08)
    ax2.axhline(y=values[0], color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # Drawdown on twin axis
    peak = np.maximum.accumulate(values)
    dd_pct = (values - peak) / peak * 100
    ax2_dd = ax2.twinx()
    ax2_dd.fill_between(dates, dd_pct, 0, color="red", alpha=0.12)
    ax2_dd.set_ylabel("Drawdown %", color="red", fontsize=9)
    ax2_dd.tick_params(axis="y", labelcolor="red", labelsize=8)
    dd_floor = min(dd_pct) * 1.3 if min(dd_pct) < 0 else -5
    ax2_dd.set_ylim(dd_floor, 2)

    ax2.set_ylabel("Equity ($)", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.25, linestyle="--")

    # ── Panel 3: Daily P&L bars + cumulative line ──
    ax3 = axes[2]

    daily_pnl = np.diff(values)
    pnl_dates = dates[1:]

    bar_colors = ["#00c853" if p > 0 else "#ff1744" for p in daily_pnl]
    ax3.bar(pnl_dates, daily_pnl, color=bar_colors, alpha=0.5, width=1.5)

    cum_pnl = np.cumsum(daily_pnl)
    ax3_cum = ax3.twinx()
    ax3_cum.plot(pnl_dates, cum_pnl, color="navy", linewidth=1.5)
    ax3_cum.set_ylabel("Cumulative P&L ($)", fontsize=9)
    ax3_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Stats annotation
    pos_days = np.sum(daily_pnl > 0)
    neg_days = np.sum(daily_pnl <= 0)
    best_day = np.max(daily_pnl)
    worst_day = np.min(daily_pnl)
    ax3.annotate(
        f"Win days: {pos_days}/{len(daily_pnl)}  |  "
        f"Best: ${best_day:+,.0f}  |  Worst: ${worst_day:+,.0f}",
        xy=(0.98, 0.92), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85),
    )

    ax3.set_ylabel("Daily P&L ($)", fontsize=10)
    ax3.set_xlabel("Date", fontsize=10)
    ax3.grid(True, alpha=0.25, linestyle="--")

    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.show(block=True)
