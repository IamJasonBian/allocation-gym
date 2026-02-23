#!/usr/bin/env python3
"""
Gold vs BTC Asset Allocation Analysis
======================================
Two-asset efficient frontier, risk-return decomposition, and
rolling-correlation study using monthly close data (Jan 2020 – Dec 2025).

Generates SVG plots in docs/ — no external dependencies required.
"""

from __future__ import annotations

import math
import statistics
import os
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Historical monthly close prices (source: public market data)
# Gold = GLD ETF (USD), BTC = BTC/USD spot
# ---------------------------------------------------------------------------
DATES = [
    "2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06",
    "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12",
    "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06",
    "2021-07", "2021-08", "2021-09", "2021-10", "2021-11", "2021-12",
    "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06",
    "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12",
    "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
    "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12",
    "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
    "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
]

# GLD ETF monthly close (USD)
GOLD_PRICES = [
    148.95, 152.10, 151.00, 162.00, 163.50, 168.50,
    182.35, 187.20, 178.80, 177.50, 169.50, 174.60,
    174.10, 167.80, 164.00, 169.10, 175.70, 168.20,
    172.90, 170.10, 164.50, 170.00, 170.50, 169.40,
    171.20, 176.50, 183.20, 175.60, 172.30, 167.10,
    164.70, 160.50, 156.50, 154.00, 163.20, 168.30,
    178.60, 172.90, 183.50, 189.20, 184.50, 179.80,
    184.70, 179.10, 178.20, 185.10, 188.70, 191.80,
    189.30, 191.70, 203.40, 213.50, 222.60, 213.00,
    221.80, 228.40, 243.30, 250.20, 247.60, 242.10,
    253.10, 264.50, 278.80, 288.80, 303.20, 295.50,
    298.80, 310.60, 301.40, 295.40, 293.10, 290.80,
]

# BTC/USD monthly close
BTC_PRICES = [
    9350,  8780,  6425,  8625, 9450, 9150,
    11350, 11650, 10775, 13780, 19695, 29000,
    33100, 45230, 58800, 57700, 37300, 35040,
    41460, 47150, 43790, 61350, 57000, 46210,
    38470, 43200, 45540, 37640, 31790, 19785,
    23290, 20050, 19425, 20490, 17150, 16530,
    23140, 23170, 28475, 29250, 27220, 30480,
    29340, 26040, 27010, 34500, 37720, 42280,
    42580, 51800, 71290, 60670, 67520, 61510,
    66800, 59100, 63330, 72340, 96400, 93400,
    102400, 84300, 82500, 95200, 103800, 106500,
    97800, 96500, 84200, 87600, 95100, 98300,
]

# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def log_returns(prices: List[float]) -> List[float]:
    """Monthly log returns."""
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def annualised_return(monthly_rets: List[float]) -> float:
    return statistics.mean(monthly_rets) * 12


def annualised_vol(monthly_rets: List[float]) -> float:
    return statistics.stdev(monthly_rets) * math.sqrt(12)


def sharpe(monthly_rets: List[float], rf: float = 0.045) -> float:
    ar = annualised_return(monthly_rets)
    av = annualised_vol(monthly_rets)
    return (ar - rf) / av if av > 0 else 0.0


def max_drawdown(prices: List[float]) -> float:
    peak = prices[0]
    mdd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def correlation(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)
    sx = statistics.stdev(xs)
    sy = statistics.stdev(ys)
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def rolling_correlation(xs: List[float], ys: List[float], window: int = 12) -> List[float]:
    out = []
    for i in range(window - 1, len(xs)):
        chunk_x = xs[i - window + 1: i + 1]
        chunk_y = ys[i - window + 1: i + 1]
        out.append(correlation(chunk_x, chunk_y))
    return out


def portfolio_return(w_btc: float, ret_btc: float, ret_gold: float) -> float:
    return w_btc * ret_btc + (1 - w_btc) * ret_gold


def portfolio_vol(
    w_btc: float, vol_btc: float, vol_gold: float, corr: float
) -> float:
    v = (
        (w_btc * vol_btc) ** 2
        + ((1 - w_btc) * vol_gold) ** 2
        + 2 * w_btc * (1 - w_btc) * vol_btc * vol_gold * corr
    )
    return math.sqrt(max(v, 0))


def cumulative_returns(prices: List[float]) -> List[float]:
    return [p / prices[0] for p in prices]


def drawdown_series(prices: List[float]) -> List[float]:
    peak = prices[0]
    dd = []
    for p in prices:
        if p > peak:
            peak = p
        dd.append((p - peak) / peak)
    return dd


# ---------------------------------------------------------------------------
# SVG plotting helpers
# ---------------------------------------------------------------------------

@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: float = 800.0
    height: float = 450.0
    margin_l: float = 80.0
    margin_r: float = 30.0
    margin_t: float = 50.0
    margin_b: float = 60.0

    @property
    def plot_w(self) -> float:
        return self.width - self.margin_l - self.margin_r

    @property
    def plot_h(self) -> float:
        return self.height - self.margin_t - self.margin_b

    def tx(self, v: float) -> float:
        return self.margin_l + (v - self.x_min) / max(self.x_max - self.x_min, 1e-12) * self.plot_w

    def ty(self, v: float) -> float:
        return self.margin_t + self.plot_h - (v - self.y_min) / max(self.y_max - self.y_min, 1e-12) * self.plot_h


def svg_header(b: Bounds, bg: str = "#ffffff") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{b.width}" height="{b.height}" '
        f'viewBox="0 0 {b.width} {b.height}">\n'
        f'<rect width="{b.width}" height="{b.height}" fill="{bg}"/>\n'
        f'<style>text {{ font-family: "Helvetica Neue", Arial, sans-serif; }}</style>\n'
    )


def svg_grid(b: Bounds, x_ticks: List[Tuple[float, str]], y_ticks: List[Tuple[float, str]]) -> str:
    lines = []
    # plot area border
    lines.append(
        f'<rect x="{b.margin_l}" y="{b.margin_t}" '
        f'width="{b.plot_w}" height="{b.plot_h}" '
        f'fill="none" stroke="#cccccc" stroke-width="1"/>\n'
    )
    for val, label in y_ticks:
        y = b.ty(val)
        lines.append(
            f'<line x1="{b.margin_l}" y1="{y}" x2="{b.margin_l + b.plot_w}" y2="{y}" '
            f'stroke="#e0e0e0" stroke-width="0.5"/>\n'
        )
        lines.append(
            f'<text x="{b.margin_l - 8}" y="{y + 4}" text-anchor="end" '
            f'font-size="11" fill="#555">{label}</text>\n'
        )
    for val, label in x_ticks:
        x = b.tx(val)
        lines.append(
            f'<line x1="{x}" y1="{b.margin_t}" x2="{x}" y2="{b.margin_t + b.plot_h}" '
            f'stroke="#e0e0e0" stroke-width="0.5"/>\n'
        )
        lines.append(
            f'<text x="{x}" y="{b.margin_t + b.plot_h + 18}" text-anchor="middle" '
            f'font-size="10" fill="#555">{label}</text>\n'
        )
    return "".join(lines)


def svg_polyline(
    b: Bounds, xs: List[float], ys: List[float],
    color: str = "#1f77b4", width: float = 2.0, dash: str = ""
) -> str:
    pts = " ".join(f"{b.tx(xs[i]):.1f},{b.ty(ys[i]):.1f}" for i in range(len(xs)))
    sd = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{sd}/>\n'


def svg_circle(b: Bounds, x: float, y: float, r: float = 4, color: str = "#1f77b4") -> str:
    return (
        f'<circle cx="{b.tx(x):.1f}" cy="{b.ty(y):.1f}" r="{r}" '
        f'fill="{color}" stroke="white" stroke-width="1"/>\n'
    )


def svg_text(
    b: Bounds, x: float, y: float, text: str,
    size: int = 12, color: str = "#333", anchor: str = "start",
    dx: float = 0, dy: float = 0
) -> str:
    return (
        f'<text x="{b.tx(x) + dx:.1f}" y="{b.ty(y) + dy:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="{color}">{text}</text>\n'
    )


def svg_title(b: Bounds, text: str, subtitle: str = "") -> str:
    s = (
        f'<text x="{b.width / 2}" y="{b.margin_t - 20}" text-anchor="middle" '
        f'font-size="16" font-weight="bold" fill="#222">{text}</text>\n'
    )
    if subtitle:
        s += (
            f'<text x="{b.width / 2}" y="{b.margin_t - 5}" text-anchor="middle" '
            f'font-size="11" fill="#666">{subtitle}</text>\n'
        )
    return s


def svg_legend(b: Bounds, items: List[Tuple[str, str]], x_off: float = 0, y_off: float = 0) -> str:
    s = ""
    bx = b.margin_l + 12 + x_off
    by = b.margin_t + 18 + y_off
    for i, (label, color) in enumerate(items):
        yy = by + i * 20
        s += f'<rect x="{bx}" y="{yy - 8}" width="14" height="3" fill="{color}"/>\n'
        s += f'<text x="{bx + 20}" y="{yy - 2}" font-size="11" fill="#333">{label}</text>\n'
    return s


def svg_axis_labels(b: Bounds, x_label: str, y_label: str) -> str:
    s = ""
    if x_label:
        s += (
            f'<text x="{b.margin_l + b.plot_w / 2}" y="{b.height - 8}" '
            f'text-anchor="middle" font-size="12" fill="#444">{x_label}</text>\n'
        )
    if y_label:
        cx = 16
        cy = b.margin_t + b.plot_h / 2
        s += (
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'font-size="12" fill="#444" '
            f'transform="rotate(-90,{cx},{cy})">{y_label}</text>\n'
        )
    return s


def nice_ticks(lo: float, hi: float, n: int = 6) -> List[Tuple[float, str]]:
    """Generate nice axis ticks."""
    rng = hi - lo
    if rng <= 0:
        return [(lo, f"{lo:.2f}")]
    raw_step = rng / n
    mag = 10 ** math.floor(math.log10(raw_step))
    choices = [1, 2, 2.5, 5, 10]
    step = min(choices, key=lambda c: abs(c * mag - raw_step)) * mag
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.01:
        if abs(v) < 1:
            label = f"{v:.2f}"
        elif abs(v) < 100:
            label = f"{v:.1f}"
        else:
            label = f"{v:.0f}"
        ticks.append((v, label))
        v += step
    return ticks


def date_ticks(dates: List[str], step: int = 6) -> List[Tuple[float, str]]:
    return [(i, dates[i]) for i in range(0, len(dates), step)]


# ---------------------------------------------------------------------------
# SVG area fill helper
# ---------------------------------------------------------------------------
def svg_area(
    b: Bounds, xs: List[float], y_upper: List[float], y_lower: List[float],
    color: str = "#1f77b4", opacity: float = 0.15
) -> str:
    """Filled area between two y-series."""
    pts_upper = [f"{b.tx(xs[i]):.1f},{b.ty(y_upper[i]):.1f}" for i in range(len(xs))]
    pts_lower = [f"{b.tx(xs[i]):.1f},{b.ty(y_lower[i]):.1f}" for i in reversed(range(len(xs)))]
    pts = " ".join(pts_upper + pts_lower)
    return f'<polygon points="{pts}" fill="{color}" opacity="{opacity}" stroke="none"/>\n'


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def plot_normalized_prices(dates, gold_cum, btc_cum, out_path):
    """Plot 1: Normalized cumulative growth (base = 1.0)."""
    n = len(gold_cum)
    x_vals = list(range(n))
    all_y = gold_cum + btc_cum
    y_min = min(all_y) * 0.9
    y_max = max(all_y) * 1.05

    b = Bounds(x_min=0, x_max=n - 1, y_min=y_min, y_max=y_max, width=900, height=480)
    svg = svg_header(b)
    svg += svg_title(b, "Gold vs BTC: Normalized Growth", "Jan 2020 = 1.0  |  Monthly closes")
    svg += svg_grid(b, date_ticks(dates, 6), nice_ticks(y_min, y_max, 8))
    svg += svg_axis_labels(b, "", "Growth of $1")

    # Shade BTC area
    svg += svg_area(b, x_vals, btc_cum, [1.0] * n, color="#f7931a", opacity=0.08)

    svg += svg_polyline(b, x_vals, gold_cum, color="#FFD700", width=2.5)
    svg += svg_polyline(b, x_vals, btc_cum, color="#f7931a", width=2.5)

    # End labels
    svg += svg_text(b, n - 1, gold_cum[-1], f" Gold {gold_cum[-1]:.1f}x", size=11, color="#b8860b", dx=5)
    svg += svg_text(b, n - 1, btc_cum[-1], f" BTC {btc_cum[-1]:.1f}x", size=11, color="#f7931a", dx=5, dy=-2)

    svg += svg_legend(b, [("Gold (GLD)", "#FFD700"), ("Bitcoin", "#f7931a")])
    svg += "</svg>"

    with open(out_path, "w") as f:
        f.write(svg)
    print(f"  -> {out_path}")


def plot_efficient_frontier(ret_gold, vol_gold, ret_btc, vol_btc, corr_val, out_path):
    """Plot 2: Two-asset efficient frontier."""
    # Generate frontier points
    weights = [i / 100 for i in range(0, 101)]
    frontier_vol = []
    frontier_ret = []
    for w in weights:
        r = portfolio_return(w, ret_btc, ret_gold)
        v = portfolio_vol(w, vol_btc, vol_gold, corr_val)
        frontier_ret.append(r * 100)  # percent
        frontier_vol.append(v * 100)

    # Find min-vol and max-Sharpe
    rf = 4.5  # percent
    min_vol_idx = frontier_vol.index(min(frontier_vol))
    best_sharpe = -999
    best_sharpe_idx = 0
    for i, (v, r) in enumerate(zip(frontier_vol, frontier_ret)):
        s = (r - rf) / v if v > 0 else 0
        if s > best_sharpe:
            best_sharpe = s
            best_sharpe_idx = i

    all_v = frontier_vol
    all_r = frontier_ret
    v_lo = min(all_v) * 0.85
    v_hi = max(all_v) * 1.1
    r_lo = min(all_r) - 3
    r_hi = max(all_r) + 5

    b = Bounds(x_min=v_lo, x_max=v_hi, y_min=r_lo, y_max=r_hi, width=900, height=520,
               margin_l=80, margin_b=70, margin_t=60)
    svg = svg_header(b)
    svg += svg_title(b, "Gold / BTC Efficient Frontier", f"Correlation = {corr_val:.2f}  |  Monthly data Jan 2020 – Dec 2025")
    svg += svg_grid(b, nice_ticks(v_lo, v_hi, 8), nice_ticks(r_lo, r_hi, 8))
    svg += svg_axis_labels(b, "Annualised Volatility (%)", "Annualised Return (%)")

    # Frontier curve
    svg += svg_polyline(b, frontier_vol, frontier_ret, color="#4a90d9", width=2.5)

    # Colour-coded dots every 10%
    colors_gradient = [
        "#FFD700", "#e8c800", "#d1b900", "#baab00", "#a39c00",
        "#8c8e00", "#757f00", "#5e7100", "#476200", "#305400",
        "#f7931a",
    ]
    for i in range(0, 101, 10):
        ci = i // 10
        svg += svg_circle(b, frontier_vol[i], frontier_ret[i], r=5, color=colors_gradient[ci])
        label = f"{i}% BTC" if i in (0, 50, 100) else ""
        if label:
            dy_off = 14 if i == 0 else -10
            svg += svg_text(b, frontier_vol[i], frontier_ret[i], label, size=10, color="#555",
                            anchor="middle", dy=dy_off)

    # Highlight min-vol
    svg += svg_circle(b, frontier_vol[min_vol_idx], frontier_ret[min_vol_idx], r=7, color="#e74c3c")
    svg += svg_text(b, frontier_vol[min_vol_idx], frontier_ret[min_vol_idx],
                    f"Min Vol ({min_vol_idx}% BTC)", size=10, color="#e74c3c", dx=10, dy=-8)

    # Highlight max-Sharpe
    svg += svg_circle(b, frontier_vol[best_sharpe_idx], frontier_ret[best_sharpe_idx], r=7, color="#27ae60")
    svg += svg_text(b, frontier_vol[best_sharpe_idx], frontier_ret[best_sharpe_idx],
                    f"Max Sharpe ({best_sharpe_idx}% BTC, SR={best_sharpe:.2f})",
                    size=10, color="#27ae60", dx=10, dy=14)

    # Individual assets
    svg += svg_circle(b, vol_gold * 100, ret_gold * 100, r=7, color="#FFD700")
    svg += svg_text(b, vol_gold * 100, ret_gold * 100, "  100% Gold", size=11, color="#b8860b", dx=8, dy=-2)
    svg += svg_circle(b, vol_btc * 100, ret_btc * 100, r=7, color="#f7931a")
    svg += svg_text(b, vol_btc * 100, ret_btc * 100, "  100% BTC", size=11, color="#f7931a", dx=8, dy=-2)

    svg += "</svg>"
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"  -> {out_path}")


def plot_rolling_correlation(dates, gold_rets, btc_rets, window, out_path):
    """Plot 3: 12-month rolling correlation."""
    rc = rolling_correlation(gold_rets, btc_rets, window)
    n = len(rc)
    rc_dates = dates[window:]  # offset for rolling window
    x_vals = list(range(n))

    y_min = min(rc) - 0.1
    y_max = max(rc) + 0.1
    y_min = max(y_min, -1.0)
    y_max = min(y_max, 1.0)

    b = Bounds(x_min=0, x_max=n - 1, y_min=y_min, y_max=y_max, width=900, height=400)
    svg = svg_header(b)
    svg += svg_title(b, f"Gold / BTC Rolling {window}-Month Correlation",
                     "Monthly log returns")
    svg += svg_grid(b, date_ticks(rc_dates, 6), nice_ticks(y_min, y_max, 8))
    svg += svg_axis_labels(b, "", "Correlation")

    # Zero line
    zero_y = b.ty(0)
    svg += (
        f'<line x1="{b.margin_l}" y1="{zero_y}" '
        f'x2="{b.margin_l + b.plot_w}" y2="{zero_y}" '
        f'stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>\n'
    )

    # Shade positive/negative
    zeros = [0.0] * n
    pos_rc = [max(r, 0) for r in rc]
    neg_rc = [min(r, 0) for r in rc]
    svg += svg_area(b, x_vals, pos_rc, zeros, color="#27ae60", opacity=0.15)
    svg += svg_area(b, x_vals, zeros, neg_rc, color="#e74c3c", opacity=0.15)

    svg += svg_polyline(b, x_vals, rc, color="#8e44ad", width=2.0)

    # Stats annotation
    avg_corr = statistics.mean(rc)
    svg += (
        f'<rect x="{b.margin_l + b.plot_w - 180}" y="{b.margin_t + 8}" '
        f'width="170" height="50" rx="4" fill="white" stroke="#ccc" opacity="0.9"/>\n'
        f'<text x="{b.margin_l + b.plot_w - 170}" y="{b.margin_t + 28}" '
        f'font-size="11" fill="#333">Mean: {avg_corr:.3f}</text>\n'
        f'<text x="{b.margin_l + b.plot_w - 170}" y="{b.margin_t + 46}" '
        f'font-size="11" fill="#333">Current: {rc[-1]:.3f}</text>\n'
    )

    svg += "</svg>"
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"  -> {out_path}")


def plot_drawdowns(dates, gold_dd, btc_dd, out_path):
    """Plot 4: Drawdown comparison."""
    n = len(gold_dd)
    x_vals = list(range(n))
    all_dd = gold_dd + btc_dd
    y_min = min(all_dd) * 1.1
    y_max = 0.02

    b = Bounds(x_min=0, x_max=n - 1, y_min=y_min, y_max=y_max, width=900, height=400)
    svg = svg_header(b)
    svg += svg_title(b, "Drawdown Comparison: Gold vs BTC", "From peak, monthly closes")
    svg += svg_grid(b, date_ticks(dates, 6),
                    [(v, f"{v * 100:.0f}%") for v, _ in nice_ticks(y_min, y_max, 6)])
    svg += svg_axis_labels(b, "", "Drawdown")

    # Fill areas
    zeros = [0.0] * n
    svg += svg_area(b, x_vals, zeros, gold_dd, color="#FFD700", opacity=0.3)
    svg += svg_area(b, x_vals, zeros, btc_dd, color="#f7931a", opacity=0.2)

    svg += svg_polyline(b, x_vals, gold_dd, color="#b8860b", width=1.8)
    svg += svg_polyline(b, x_vals, btc_dd, color="#f7931a", width=1.8)

    svg += svg_legend(b, [
        (f"Gold (max DD {min(gold_dd) * 100:.1f}%)", "#b8860b"),
        (f"BTC (max DD {min(btc_dd) * 100:.1f}%)", "#f7931a"),
    ])

    svg += "</svg>"
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"  -> {out_path}")


def plot_risk_return_table(
    ret_gold, vol_gold, sharpe_gold, mdd_gold,
    ret_btc, vol_btc, sharpe_btc, mdd_btc,
    corr_val, out_path
):
    """Plot 5: Summary statistics table as SVG."""
    w, h = 720, 420
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">\n'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'<style>text {{ font-family: "Helvetica Neue", Arial, sans-serif; }}</style>\n'
    )
    svg += (
        f'<text x="{w / 2}" y="32" text-anchor="middle" font-size="16" '
        f'font-weight="bold" fill="#222">Asset Allocation Summary Statistics</text>\n'
        f'<text x="{w / 2}" y="50" text-anchor="middle" font-size="11" '
        f'fill="#666">Jan 2020 – Dec 2025  |  Monthly data  |  Risk-free rate = 4.5%</text>\n'
    )

    # Table
    cols = [60, 220, 420, 600]
    rows_y = [90, 120, 155, 190, 225, 260, 295, 330, 365]
    headers = ["", "Gold (GLD)", "Bitcoin (BTC)", "Correlation"]
    metrics = [
        ("Ann. Return", f"{ret_gold * 100:.1f}%", f"{ret_btc * 100:.1f}%", ""),
        ("Ann. Volatility", f"{vol_gold * 100:.1f}%", f"{vol_btc * 100:.1f}%", ""),
        ("Sharpe Ratio", f"{sharpe_gold:.2f}", f"{sharpe_btc:.2f}", ""),
        ("Max Drawdown", f"{mdd_gold * 100:.1f}%", f"{mdd_btc * 100:.1f}%", ""),
        ("Total Return", f"{(GOLD_PRICES[-1] / GOLD_PRICES[0] - 1) * 100:.0f}%",
         f"{(BTC_PRICES[-1] / BTC_PRICES[0] - 1) * 100:.0f}%", ""),
        ("Correlation", "", "", f"{corr_val:.3f}"),
    ]

    # Header row
    svg += f'<rect x="40" y="72" width="{w - 80}" height="28" fill="#f0f4f8" rx="4"/>\n'
    for ci, hdr in enumerate(headers):
        svg += (
            f'<text x="{cols[ci]}" y="91" font-size="12" font-weight="bold" fill="#333">'
            f'{hdr}</text>\n'
        )

    # Data rows
    for ri, (metric, g_val, b_val, c_val) in enumerate(metrics):
        y = rows_y[ri + 2]
        if ri % 2 == 0:
            svg += f'<rect x="40" y="{y - 16}" width="{w - 80}" height="30" fill="#fafafa" rx="2"/>\n'
        svg += f'<text x="{cols[0]}" y="{y}" font-size="12" fill="#333">{metric}</text>\n'
        svg += f'<text x="{cols[1]}" y="{y}" font-size="12" fill="#b8860b">{g_val}</text>\n'
        svg += f'<text x="{cols[2]}" y="{y}" font-size="12" fill="#f7931a">{b_val}</text>\n'
        if c_val:
            svg += f'<text x="{cols[3]}" y="{y}" font-size="13" font-weight="bold" fill="#8e44ad">{c_val}</text>\n'

    # Color squares
    svg += f'<rect x="{cols[1] - 16}" y="78" width="10" height="10" fill="#FFD700" rx="2"/>\n'
    svg += f'<rect x="{cols[2] - 16}" y="78" width="10" height="10" fill="#f7931a" rx="2"/>\n'

    svg += "</svg>"
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"  -> {out_path}")


def plot_annual_returns(dates, gold_rets, btc_rets, out_path):
    """Plot 6: Annual return bar chart comparison."""
    # Aggregate annual returns
    years = sorted(set(d[:4] for d in dates[1:]))  # skip first (no return)
    annual = {}
    for yr in years:
        annual[yr] = {"gold": [], "btc": []}

    for i, d in enumerate(dates[1:]):
        yr = d[:4]
        annual[yr]["gold"].append(gold_rets[i])
        annual[yr]["btc"].append(btc_rets[i])

    yr_labels = []
    gold_annual = []
    btc_annual = []
    for yr in years:
        yr_labels.append(yr)
        gold_annual.append((math.exp(sum(annual[yr]["gold"])) - 1) * 100)
        btc_annual.append((math.exp(sum(annual[yr]["btc"])) - 1) * 100)

    n = len(yr_labels)
    all_vals = gold_annual + btc_annual
    y_min = min(min(all_vals), 0) - 10
    y_max = max(all_vals) + 15

    b = Bounds(x_min=-0.5, x_max=n - 0.5, y_min=y_min, y_max=y_max,
               width=900, height=450, margin_l=80, margin_b=60)
    svg = svg_header(b)
    svg += svg_title(b, "Annual Returns: Gold vs BTC", "Calendar year, log-return compounded")

    # Y grid
    y_ticks = nice_ticks(y_min, y_max, 8)
    svg += svg_grid(b, [], [(v, f"{lbl}%") for v, lbl in y_ticks])

    # Zero line
    zero_y = b.ty(0)
    svg += (
        f'<line x1="{b.margin_l}" y1="{zero_y}" '
        f'x2="{b.margin_l + b.plot_w}" y2="{zero_y}" '
        f'stroke="#555" stroke-width="1"/>\n'
    )

    bar_w = b.plot_w / n * 0.35
    for i in range(n):
        cx = b.tx(i)
        # Gold bar
        g_top = b.ty(max(gold_annual[i], 0))
        g_bot = b.ty(min(gold_annual[i], 0))
        svg += (
            f'<rect x="{cx - bar_w - 2}" y="{g_top}" '
            f'width="{bar_w}" height="{max(g_bot - g_top, 1)}" '
            f'fill="#FFD700" stroke="#b8860b" stroke-width="0.5" rx="2"/>\n'
        )
        # BTC bar
        bt_top = b.ty(max(btc_annual[i], 0))
        bt_bot = b.ty(min(btc_annual[i], 0))
        svg += (
            f'<rect x="{cx + 2}" y="{bt_top}" '
            f'width="{bar_w}" height="{max(bt_bot - bt_top, 1)}" '
            f'fill="#f7931a" stroke="#c67300" stroke-width="0.5" rx="2"/>\n'
        )
        # Year label
        svg += (
            f'<text x="{cx}" y="{b.margin_t + b.plot_h + 18}" text-anchor="middle" '
            f'font-size="12" fill="#333">{yr_labels[i]}</text>\n'
        )
        # Value labels
        svg += (
            f'<text x="{cx - bar_w / 2 - 2}" y="{g_top - 4}" text-anchor="middle" '
            f'font-size="9" fill="#b8860b">{gold_annual[i]:.0f}%</text>\n'
        )
        svg += (
            f'<text x="{cx + bar_w / 2 + 2}" y="{bt_top - 4}" text-anchor="middle" '
            f'font-size="9" fill="#c67300">{btc_annual[i]:.0f}%</text>\n'
        )

    svg += svg_legend(b, [("Gold (GLD)", "#FFD700"), ("Bitcoin", "#f7931a")], x_off=b.plot_w - 180)
    svg += svg_axis_labels(b, "", "Return (%)")
    svg += "</svg>"
    with open(out_path, "w") as f:
        f.write(svg)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(docs_dir, exist_ok=True)

    print("=" * 65)
    print("  Gold vs BTC Asset Allocation Analysis")
    print("  Period: Jan 2020 – Dec 2025 (monthly closes)")
    print("=" * 65)

    # Compute returns
    gold_rets = log_returns(GOLD_PRICES)
    btc_rets = log_returns(BTC_PRICES)

    # Cumulative
    gold_cum = cumulative_returns(GOLD_PRICES)
    btc_cum = cumulative_returns(BTC_PRICES)

    # Annualised stats
    ret_gold = annualised_return(gold_rets)
    vol_gold = annualised_vol(gold_rets)
    sharpe_gold = sharpe(gold_rets)
    mdd_gold = max_drawdown(GOLD_PRICES)

    ret_btc = annualised_return(btc_rets)
    vol_btc = annualised_vol(btc_rets)
    sharpe_btc = sharpe(btc_rets)
    mdd_btc = max_drawdown(BTC_PRICES)

    corr_val = correlation(gold_rets, btc_rets)

    # Print summary
    print(f"\n{'Metric':<22} {'Gold (GLD)':>14} {'Bitcoin':>14}")
    print("-" * 52)
    print(f"{'Ann. Return':<22} {ret_gold * 100:>13.1f}% {ret_btc * 100:>13.1f}%")
    print(f"{'Ann. Volatility':<22} {vol_gold * 100:>13.1f}% {vol_btc * 100:>13.1f}%")
    print(f"{'Sharpe Ratio':<22} {sharpe_gold:>14.2f} {sharpe_btc:>14.2f}")
    print(f"{'Max Drawdown':<22} {mdd_gold * 100:>13.1f}% {mdd_btc * 100:>13.1f}%")
    print(f"{'Total Return':<22} {(GOLD_PRICES[-1] / GOLD_PRICES[0] - 1) * 100:>13.0f}% "
          f"{(BTC_PRICES[-1] / BTC_PRICES[0] - 1) * 100:>13.0f}%")
    print(f"\nCorrelation: {corr_val:.3f}")

    # Efficient frontier highlights
    print("\n--- Efficient Frontier Highlights ---")
    rf = 0.045
    min_vol = 999
    min_vol_w = 0
    best_sr = -999
    best_sr_w = 0
    for w_pct in range(0, 101):
        w = w_pct / 100
        r = portfolio_return(w, ret_btc, ret_gold)
        v = portfolio_vol(w, vol_btc, vol_gold, corr_val)
        sr = (r - rf) / v if v > 0 else 0
        if v < min_vol:
            min_vol = v
            min_vol_w = w_pct
        if sr > best_sr:
            best_sr = sr
            best_sr_w = w_pct

    mv_r = portfolio_return(min_vol_w / 100, ret_btc, ret_gold)
    ms_r = portfolio_return(best_sr_w / 100, ret_btc, ret_gold)
    ms_v = portfolio_vol(best_sr_w / 100, vol_btc, vol_gold, corr_val)

    print(f"  Min-Variance portfolio: {min_vol_w}% BTC / {100 - min_vol_w}% Gold")
    print(f"    Return: {mv_r * 100:.1f}%  Vol: {min_vol * 100:.1f}%")
    print(f"  Max-Sharpe portfolio:   {best_sr_w}% BTC / {100 - best_sr_w}% Gold")
    print(f"    Return: {ms_r * 100:.1f}%  Vol: {ms_v * 100:.1f}%  Sharpe: {best_sr:.2f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_normalized_prices(DATES, gold_cum, btc_cum,
                           os.path.join(docs_dir, "gold_btc_normalized_growth.svg"))
    plot_efficient_frontier(ret_gold, vol_gold, ret_btc, vol_btc, corr_val,
                            os.path.join(docs_dir, "gold_btc_efficient_frontier.svg"))
    plot_rolling_correlation(DATES, gold_rets, btc_rets, 12,
                             os.path.join(docs_dir, "gold_btc_rolling_correlation.svg"))
    plot_drawdowns(DATES, drawdown_series(GOLD_PRICES), drawdown_series(BTC_PRICES),
                   os.path.join(docs_dir, "gold_btc_drawdowns.svg"))
    plot_risk_return_table(ret_gold, vol_gold, sharpe_gold, mdd_gold,
                           ret_btc, vol_btc, sharpe_btc, mdd_btc,
                           corr_val,
                           os.path.join(docs_dir, "gold_btc_summary_table.svg"))
    plot_annual_returns(DATES, gold_rets, btc_rets,
                        os.path.join(docs_dir, "gold_btc_annual_returns.svg"))

    print("\nDone. All plots saved to docs/")


if __name__ == "__main__":
    main()
