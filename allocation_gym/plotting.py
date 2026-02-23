"""
Backtest result plotting — equity curve, P&L, trades on price, and signal overlays.
Auto-shows on every run. Use --no-plot to disable.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _extract_line_history(indicator, line_name):
    """Extract (dates, values) from a backtrader indicator line."""
    line = getattr(indicator.lines, line_name)
    dates, vals = [], []
    for j in range(-len(indicator) + 1, 1):
        try:
            dt = indicator.data.datetime.date(j)
            v = line[j]
            dates.append(dt)
            vals.append(v)
        except IndexError:
            continue
    return dates, vals


def _has_signals(strat):
    """Check if the strategy has signal indicators attached."""
    return getattr(strat, "signal_iv", None) is not None


def plot_backtest(analyzer, cerebro_result, strategy_name="", symbols=None):
    """
    Plot backtest results with separate subplots per symbol + equity + P&L.
    When --signals is enabled, adds IV z-score, ETF flows, and hist vol panels.

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
    has_sigs = _has_signals(strat)

    # Panel count: symbols + equity + PnL + (optional: IV, ETF flow, hist vol)
    n_signal_panels = 3 if has_sigs else 0
    n_panels = n_symbols + 2 + n_signal_panels

    height_ratios = [1.5] * n_symbols + [1, 1] + [0.8] * n_signal_panels
    fig_height = 3 * n_symbols + 5 + 2.5 * n_signal_panels
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_height),
                             height_ratios=height_ratios, sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = list(axes)

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
    ax_pnl.grid(True, alpha=0.25, linestyle="--")
    ax_pnl.margins(x=0.02)

    # ── Signal panels (when --signals is enabled) ──
    if has_sigs:
        base_idx = n_symbols + 2

        # Panel: IV Z-Score
        ax_iv = axes[base_idx]
        iv_dates, iv_dvol = _extract_line_history(strat.signal_iv, "dvol")
        _, iv_zs = _extract_line_history(strat.signal_iv, "dvol_zscore")

        ax_iv.plot(iv_dates, iv_dvol, color="purple", linewidth=1, label="DVOL")
        ax_iv.set_ylabel("DVOL (%)", fontsize=9, color="purple")
        ax_iv.tick_params(axis="y", labelcolor="purple", labelsize=8)
        ax_iv.grid(True, alpha=0.25, linestyle="--")
        ax_iv.margins(x=0.02)

        ax_zs = ax_iv.twinx()
        zs_colors = ["#ff1744" if z > 1 else ("#00c853" if z < -1 else "gray") for z in iv_zs]
        ax_zs.bar(iv_dates, iv_zs, color=zs_colors, alpha=0.3, width=1.5)
        ax_zs.axhline(0, color="black", linewidth=0.5)
        ax_zs.axhline(1, color="red", linestyle=":", linewidth=0.6, alpha=0.5)
        ax_zs.axhline(-1, color="green", linestyle=":", linewidth=0.6, alpha=0.5)
        ax_zs.set_ylabel("Z-Score", fontsize=9)
        ax_zs.tick_params(axis="y", labelsize=8)
        ax_iv.set_title("IV Z-Score (Deribit DVOL)", fontsize=9, loc="left")
        ax_iv.legend(loc="upper left", fontsize=7)

        # Panel: ETF Cumulative Flows
        ax_flow = axes[base_idx + 1]
        flow_dates, cum_flow = _extract_line_history(strat.signal_flow, "cumulative_flow")

        cum_arr = np.array(cum_flow)
        ax_flow.fill_between(flow_dates, 0, cum_arr, alpha=0.25,
                             color="green", where=cum_arr >= 0)
        ax_flow.fill_between(flow_dates, 0, cum_arr, alpha=0.25,
                             color="red", where=cum_arr < 0)
        ax_flow.plot(flow_dates, cum_arr, color="steelblue", linewidth=1)
        ax_flow.axhline(0, color="black", linewidth=0.5)
        ax_flow.set_ylabel("Cum. Flow ($)", fontsize=9)
        ax_flow.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B" if abs(x) >= 1e9 else f"${x/1e6:.0f}M")
        )
        ax_flow.set_title("BTC ETF Cumulative Net Flows (est.)", fontsize=9, loc="left")
        ax_flow.grid(True, alpha=0.25, linestyle="--")
        ax_flow.margins(x=0.02)

        # Panel: Historical Volatility (from first symbol's VarianceIndicator)
        ax_vol = axes[base_idx + 2]
        first_name = strat.datas[0]._name
        var_ind = strat.variance_indicators.get(first_name)
        if var_ind is not None:
            vol_dates, vol_vals = _extract_line_history(var_ind, "yz_vol_ann")
            vol_pct = [v * 100 for v in vol_vals]
            ax_vol.plot(vol_dates, vol_pct, color="darkorange", linewidth=1,
                        label="Yang-Zhang Vol (ann.)")
            ax_vol.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=0.7)
            ax_vol.axhline(80, color="red", linestyle=":", alpha=0.4, linewidth=0.7)
            ax_vol.set_ylabel("Vol %", fontsize=9)
            ax_vol.set_title(f"Historical Volatility — {first_name}", fontsize=9, loc="left")
            ax_vol.legend(loc="upper left", fontsize=7)
        ax_vol.grid(True, alpha=0.25, linestyle="--")
        ax_vol.margins(x=0.02)

    # Format x-axis dates on the bottom panel
    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.show(block=True)
