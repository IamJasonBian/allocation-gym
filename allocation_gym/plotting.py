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
    Plot backtest results with separate subplots per symbol + equity + P&L.

    Args:
        analyzer: PerformanceAnalyzer instance
        cerebro_result: The strategy result object (results[0])
        strategy_name: Label for the title
        symbols: List of symbol names
    """
    plt.close("all")

    dates, values = analyzer.get_equity_curve()
    orders = analyzer.get_orders()
    perf = analyzer.get_analysis()

    if not dates or not values:
        print("No data to plot.")
        return

    values = np.array(values, dtype=float)
    symbols = symbols or []

    strat = cerebro_result
    n_symbols = len(strat.datas)
    n_panels = n_symbols + 2  # one per symbol + equity + P&L

    height_ratios = [1.5] * n_symbols + [1, 1]
    fig_height = 3 * n_symbols + 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_height),
                             height_ratios=height_ratios, sharex=True)

    # Handle single-symbol case where axes isn't an array
    if n_panels == 3:
        axes = [axes[0], axes[1], axes[2]]

    fig.suptitle(
        f"{strategy_name.upper()}  |  "
        f"Return: {perf.get('total_return_pct', 0):+.1f}%  "
        f"Sharpe: {perf.get('sharpe', 0):.2f}  "
        f"Max DD: {perf.get('max_drawdown_pct', 0):.1f}%  "
        f"Orders: {perf.get('total_orders', 0)} "
        f"({perf.get('buy_orders', 0)}B / {perf.get('sell_orders', 0)}S)",
        fontsize=11, fontweight="bold",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]

    # ── Per-symbol price panels with order markers ──
    for i, data in enumerate(strat.datas):
        ax = axes[i]
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
        ax.plot(price_dates, price_vals, label=name, color=color,
                linewidth=1.3, alpha=0.9)

        # Overlay order markers for this symbol only
        sym_portfolio = [o for o in orders if o["side"] == "portfolio" and o["symbol"] == name]
        sym_buys = [o for o in orders if o["side"] == "buy" and o["symbol"] == name]
        sym_sells = [o for o in orders if o["side"] == "sell" and o["symbol"] == name]

        if sym_portfolio:
            ax.scatter([o["dt"] for o in sym_portfolio],
                       [o["price"] for o in sym_portfolio],
                       marker="s", color="#FFD600", s=60,
                       zorder=6, label="Portfolio", edgecolors="black", linewidths=0.6)
        if sym_buys:
            ax.scatter([o["dt"] for o in sym_buys],
                       [o["price"] for o in sym_buys],
                       marker="^", color="#00c853", s=50,
                       zorder=5, label="Buy", edgecolors="black", linewidths=0.4)
        if sym_sells:
            ax.scatter([o["dt"] for o in sym_sells],
                       [o["price"] for o in sym_sells],
                       marker="v", color="#ff1744", s=50,
                       zorder=5, label="Sell", edgecolors="black", linewidths=0.4)

        ax.set_ylabel(f"{name} ($)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.margins(x=0.02)

    # ── Equity curve + drawdown ──
    ax_eq = axes[n_symbols]
    ax_eq.plot(dates, values, color="steelblue", linewidth=1.5)
    ax_eq.fill_between(dates, values[0], values, where=values >= values[0],
                       color="steelblue", alpha=0.08)
    ax_eq.axhline(y=values[0], color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    peak = np.maximum.accumulate(values)
    dd_pct = (values - peak) / peak * 100
    ax_dd = ax_eq.twinx()
    ax_dd.fill_between(dates, dd_pct, 0, color="red", alpha=0.12)
    ax_dd.set_ylabel("Drawdown %", color="red", fontsize=9)
    ax_dd.tick_params(axis="y", labelcolor="red", labelsize=8)
    dd_floor = min(dd_pct) * 1.3 if min(dd_pct) < 0 else -5
    ax_dd.set_ylim(dd_floor, 2)

    ax_eq.set_ylabel("Equity ($)", fontsize=10)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_eq.grid(True, alpha=0.25, linestyle="--")
    ax_eq.margins(x=0.02)

    # ── Daily P&L bars + cumulative line ──
    ax_pnl = axes[n_symbols + 1]

    daily_pnl = np.diff(values)
    pnl_dates = dates[1:]

    bar_colors = ["#00c853" if p > 0 else "#ff1744" for p in daily_pnl]
    ax_pnl.bar(pnl_dates, daily_pnl, color=bar_colors, alpha=0.5, width=1.5)

    cum_pnl = np.cumsum(daily_pnl)
    ax_cum = ax_pnl.twinx()
    ax_cum.plot(pnl_dates, cum_pnl, color="navy", linewidth=1.5)
    ax_cum.set_ylabel("Cumulative P&L ($)", fontsize=9)
    ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    pos_days = np.sum(daily_pnl > 0)
    best_day = np.max(daily_pnl)
    worst_day = np.min(daily_pnl)
    ax_pnl.annotate(
        f"Win days: {pos_days}/{len(daily_pnl)}  |  "
        f"Best: ${best_day:+,.0f}  |  Worst: ${worst_day:+,.0f}",
        xy=(0.98, 0.92), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85),
    )

    ax_pnl.set_ylabel("Daily P&L ($)", fontsize=10)
    ax_pnl.set_xlabel("Date", fontsize=10)
    ax_pnl.grid(True, alpha=0.25, linestyle="--")
    ax_pnl.margins(x=0.02)

    # Format x-axis dates
    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_pnl.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.show(block=True)
