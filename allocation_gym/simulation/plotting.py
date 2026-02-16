"""Monte Carlo simulation visualization — fan chart + terminal distribution."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_simulation(stats: dict, result=None, symbol: str = "BTC/USD"):
    """
    Two-panel plot:
      1. Fan chart: percentile bands (P10-P90) over time
      2. Histogram: distribution of final prices
    """
    plt.close("all")

    fig, (ax_fan, ax_hist) = plt.subplots(
        2, 1, figsize=(14, 10), height_ratios=[2, 1],
    )

    percentile_paths = stats["percentile_paths"]
    n_days = stats["n_days"]
    time_days = np.arange(len(list(percentile_paths.values())[0]))
    initial = stats["initial_price"]

    fig.suptitle(
        f"{symbol} Forward Monte Carlo  |  "
        f"{stats['n_paths']:,} paths x {n_days} days  |  "
        f"mu={stats['mu']:.1%}  sigma={stats['sigma']:.1%}",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: Fan chart ──
    pairs = [(10, 90), (25, 75)]
    for i, (lo, hi) in enumerate(pairs):
        if lo in percentile_paths and hi in percentile_paths:
            ax_fan.fill_between(
                time_days,
                percentile_paths[lo],
                percentile_paths[hi],
                alpha=0.3 + i * 0.15,
                color="#3498db",
                label=f"P{lo}-P{hi}",
            )

    if 50 in percentile_paths:
        ax_fan.plot(
            time_days, percentile_paths[50],
            color="#2c3e50", linewidth=2, label="Median (P50)",
        )

    # Sample paths for texture
    if result is not None and result.paths.shape[0] >= 20:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(result.paths.shape[0], size=20, replace=False)
        for idx in sample_idx:
            ax_fan.plot(
                time_days, result.paths[idx],
                color="gray", alpha=0.08, linewidth=0.5,
            )

    ax_fan.axhline(
        y=initial, color="red", linestyle="--",
        linewidth=0.8, alpha=0.7, label=f"Start ${initial:,.0f}",
    )

    ax_fan.set_ylabel(f"{symbol} Price ($)", fontsize=10)
    ax_fan.set_xlabel("Days Forward", fontsize=10)
    ax_fan.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_fan.grid(True, alpha=0.25, linestyle="--")
    ax_fan.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax_fan.margins(x=0.02)

    # ── Panel 2: Terminal price distribution ──
    if result is not None:
        final_prices = result.paths[:, -1]

        ax_hist.hist(
            final_prices, bins=80, color="#3498db",
            alpha=0.7, edgecolor="white", linewidth=0.3,
        )

        for p in [10, 50, 90]:
            if f"P{p}" in stats:
                val = stats[f"P{p}"]
                ax_hist.axvline(
                    x=val, color="navy" if p == 50 else "gray",
                    linestyle="--" if p != 50 else "-",
                    linewidth=1.2 if p == 50 else 0.8,
                    label=f"P{p}: ${val:,.0f}",
                )

        ax_hist.axvline(
            x=initial, color="red", linestyle="--",
            linewidth=1, alpha=0.8, label=f"Start ${initial:,.0f}",
        )

        prob_profit = stats["prob_above_initial"] * 100
        median_ret = stats["expected_return_pct"]
        ax_hist.annotate(
            f"Prob(profit): {prob_profit:.1f}%\n"
            f"Median return: {median_ret:+.1f}%\n"
            f"P10: ${stats.get('P10', 0):,.0f}  |  P90: ${stats.get('P90', 0):,.0f}",
            xy=(0.98, 0.92), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
        )

    ax_hist.set_xlabel(f"Final {symbol} Price ($)", fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_hist.grid(True, alpha=0.25, linestyle="--")
    ax_hist.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax_hist.margins(x=0.02)

    plt.tight_layout()
    plt.show(block=True)
