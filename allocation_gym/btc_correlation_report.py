#!/usr/bin/env python3
"""
BTC Correlation & Put Sizing — PDF Report Generator
====================================================
Reads analysis outputs from docs/ and assembles a multi-page PDF report
in reports/.

Usage:
    python -m allocation_gym.btc_correlation_report
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs", "6")
REPORTS_DIR = os.path.join(BASE_DIR, "docs", "6")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(filename: str) -> list[dict]:
    path = os.path.join(DOCS_DIR, filename)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _add_title_page(pdf: PdfPages, metadata: list[dict]):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    meta = {r["parameter"]: r["value"] for r in metadata}

    ax.text(0.5, 0.82, "BTC Correlation & Reallocation Analysis",
            fontsize=28, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.72, "Put Options Pricing & Portfolio Sizing Report",
            fontsize=18, ha="center", va="center", color="#555",
            transform=ax.transAxes)

    # Divider
    ax.plot([0.15, 0.85], [0.65, 0.65], color="#ccc", linewidth=2,
            transform=ax.transAxes)

    details = [
        f"Report Date:  {datetime.now().strftime('%B %d, %Y')}",
        f"Data As-Of:   {meta.get('run_date', 'N/A')}",
        f"BTC Spot:     ${float(meta.get('btc_spot', 0)):,.2f}",
        f"BTC 30d Vol:  {float(meta.get('btc_30d_vol', 0)):.1%}",
        f"Universe:     {meta.get('universe_size', 'N/A')} assets (midcaps + commodities + indices)",
        f"Portfolio:    ${int(meta.get('portfolio_value', 0)):,}",
        f"Reallocation: {float(meta.get('reallocation_pct', 0)):.0%}",
        f"Risk-Free:    {float(meta.get('risk_free_rate', 0)):.1%}",
        f"Lookback:     {meta.get('lookback_period', 'N/A')}",
        f"Corr Windows: {meta.get('correlation_windows', 'N/A')}",
    ]

    for i, line in enumerate(details):
        ax.text(0.5, 0.56 - i * 0.045, line,
                fontsize=13, ha="center", va="center", fontfamily="monospace",
                transform=ax.transAxes)

    ax.text(0.5, 0.06, "allocation-gym  |  yangon-v1",
            fontsize=10, ha="center", color="#999", transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, img_path: str, title: str):
    """Full-page embed of a PNG chart."""
    if not os.path.exists(img_path):
        return
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(1, 1, figure=fig, left=0.02, right=0.98, top=0.92, bottom=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")

    img = mpimg.imread(img_path)
    ax.imshow(img, aspect="auto")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_page(pdf: PdfPages, title: str, subtitle: str,
                    headers: list[str], rows: list[list[str]],
                    col_widths: list[float] | None = None):
    """Render a data table as a full page."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    ax.text(0.5, 0.97, subtitle, fontsize=11, ha="center", va="top",
            color="#666", transform=ax.transAxes)

    n_cols = len(headers)
    n_rows = len(rows)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Alternate row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#F8F9FA")
            else:
                cell.set_facecolor("#FFFFFF")

    pdf.savefig(fig)
    plt.close(fig)


def _add_put_sizing_page(pdf: PdfPages, ticker: str):
    """Parse per-ticker put sizing CSV and render tables."""
    csv_path = os.path.join(DOCS_DIR, f"btc_{ticker.lower()}_put_sizing.csv")
    if not os.path.exists(csv_path):
        return

    with open(csv_path, newline="") as f:
        lines = list(csv.reader(f))

    # Parse sections
    sections = {}
    current_section = None
    current_rows = []
    for line in lines:
        if not line or not line[0]:
            if current_section and current_rows:
                sections[current_section] = current_rows
            current_section = None
            current_rows = []
            continue
        if line[0].startswith("#"):
            if current_section and current_rows:
                sections[current_section] = current_rows
            current_section = line[0].replace("# ", "")
            current_rows = []
        else:
            current_rows.append(line)
    if current_section and current_rows:
        sections[current_section] = current_rows

    # Summary info
    summary_header = sections.get("SUMMARY", [[]])[0] if "SUMMARY" in sections else []
    summary_data = sections["SUMMARY"][1] if "SUMMARY" in sections and len(sections["SUMMARY"]) > 1 else []

    spot = summary_data[1] if len(summary_data) > 1 else "?"
    vol = summary_data[2] if len(summary_data) > 2 else "?"
    thesis = summary_data[6] if len(summary_data) > 6 else "?"

    try:
        spot_str = f"${float(spot):,.2f}"
        vol_str = f"{float(vol):.1%}"
    except (ValueError, IndexError):
        spot_str = spot
        vol_str = vol

    sub = f"Spot: {spot_str}  |  30d Vol: {vol_str}  |  {thesis}"

    # Strike ladder table
    if "STRIKE LADDER" in sections:
        sl = sections["STRIKE LADDER"]
        if len(sl) > 1:
            headers_raw = sl[0]
            # Simplify headers
            headers_display = ["Strike", "Type"]
            for h in headers_raw[2:]:
                if "_pct" in h:
                    tenor = h.replace("put_", "").replace("_pct", "")
                    headers_display.append(f"{tenor} %")
                elif "_price" in h:
                    tenor = h.replace("put_", "").replace("_price", "")
                    headers_display.append(f"{tenor} $")

            table_rows = []
            for row in sl[1:]:
                formatted = [f"${float(row[0]):,.2f}", row[1]]
                for k, val in enumerate(row[2:]):
                    if k % 2 == 0:  # pct
                        formatted.append(f"{float(val) * 100:.1f}%")
                    else:  # price
                        formatted.append(f"${float(val):,.2f}")
                table_rows.append(formatted)

            _add_table_page(pdf,
                            f"{ticker} — Put Strike Ladder",
                            sub,
                            headers_display, table_rows)

    # Portfolio sizing table
    if "PORTFOLIO SIZING" in sections:
        ps = sections["PORTFOLIO SIZING"]
        if len(ps) > 1:
            table_rows = []
            for row in ps[1:]:
                table_rows.append([
                    f"{float(row[0]) * 100:.0f}%",
                    f"${float(row[1]):,.0f}",
                    f"{float(row[2]):,.1f}",
                    f"${float(row[3]):,.2f}",
                    f"${float(row[4]):,.0f}",
                    f"{float(row[5]) * 100:.2f}%",
                    f"${float(row[6]):,.0f}",
                ])
            _add_table_page(pdf,
                            f"{ticker} — Portfolio Sizing (3M ATM Put)",
                            sub,
                            ["Alloc%", "Notional", "Shares", "ATM Put", "Hedge$", "Hedge/Port", "Net Exp"],
                            table_rows)

    # Budget-constrained table
    if "BUDGET-CONSTRAINED SIZING" in sections:
        bc = sections["BUDGET-CONSTRAINED SIZING"]
        if len(bc) > 1:
            table_rows = []
            for row in bc[1:]:
                table_rows.append([
                    f"{float(row[0]) * 100:.1f}%",
                    f"${float(row[1]):,.0f}",
                    f"{float(row[2]) * 100:.1f}%",
                    f"${float(row[3]):,.0f}",
                    f"{float(row[4]):,.1f}",
                ])
            _add_table_page(pdf,
                            f"{ticker} — Budget-Constrained Sizing",
                            sub,
                            ["Budget %", "Budget $", "Max Alloc%", "Max Notional", "Shares"],
                            table_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "btc_correlation_report.pdf")

    print("=" * 60)
    print("  Generating BTC Correlation & Reallocation PDF Report")
    print("=" * 60)

    # Read CSVs
    metadata = _read_csv("btc_correlation_metadata.csv")
    top_long = _read_csv("btc_top10_long_thesis.csv")
    top_short = _read_csv("btc_top10_short_thesis.csv")
    all_assets = _read_csv("btc_correlation_all_assets.csv")

    with PdfPages(out_path) as pdf:
        # Page 1: Title
        print("  Page 1: Title page")
        _add_title_page(pdf, metadata)

        # Page 2: Dashboard
        print("  Page 2: Reallocation dashboard")
        _add_image_page(pdf,
                        os.path.join(DOCS_DIR, "btc_reallocation_dashboard.png"),
                        "BTC Reallocation Analysis Dashboard")

        # Page 3: Top-10 Long thesis table
        print("  Page 3: Top-10 long thesis table")
        long_rows = []
        for r in top_long:
            long_rows.append([
                r["rank"],
                r["ticker"],
                f"${float(r['spot']):,.2f}",
                f"{float(r['vol_30d']):.1%}",
                f"{float(r['corr_3m']):+.3f}",
                f"{float(r['corr_6m']):+.3f}",
                f"{float(r['corr_12m']):+.3f}",
                f"{float(r['put_pct_3m']):.1%}",
                f"${float(r['hedge_cost_3m']):,.0f}",
                f"{float(r['score_long']):.3f}",
            ])
        _add_table_page(pdf,
                        "Top 10 — BTC Long Thesis (Hedge Candidates)",
                        "Assets with negative BTC correlation — diversify BTC long exposure",
                        ["#", "Ticker", "Spot", "30d Vol", "Corr 3M", "Corr 6M", "Corr 12M", "Put 3M%", "Hedge$", "Score"],
                        long_rows)

        # Page 4: Long thesis price chart
        print("  Page 4: Long thesis price chart")
        _add_image_page(pdf,
                        os.path.join(DOCS_DIR, "btc_long_thesis_prices.png"),
                        "BTC Long Thesis — Top 10 Price Performance & Correlation")

        # Page 5: Top-10 Short thesis table
        print("  Page 5: Top-10 short thesis table")
        short_rows = []
        for r in top_short:
            short_rows.append([
                r["rank"],
                r["ticker"],
                f"${float(r['spot']):,.2f}",
                f"{float(r['vol_30d']):.1%}",
                f"{float(r['corr_3m']):+.3f}",
                f"{float(r['corr_6m']):+.3f}",
                f"{float(r['corr_12m']):+.3f}",
                f"{float(r['put_pct_3m']):.1%}",
                f"${float(r['hedge_cost_3m']):,.0f}",
                f"{float(r['score_short']):.3f}",
            ])
        _add_table_page(pdf,
                        "Top 10 — BTC Short Thesis (Correlated Downside Plays)",
                        "Assets with positive BTC correlation — amplify short BTC positioning",
                        ["#", "Ticker", "Spot", "30d Vol", "Corr 3M", "Corr 6M", "Corr 12M", "Put 3M%", "Hedge$", "Score"],
                        short_rows)

        # Page 6: Short thesis price chart
        print("  Page 6: Short thesis price chart")
        _add_image_page(pdf,
                        os.path.join(DOCS_DIR, "btc_short_thesis_prices.png"),
                        "BTC Short Thesis — Top 10 Price Performance & Correlation")

        # Page 7: Correlation heatmap
        print("  Page 7: Correlation heatmap")
        _add_image_page(pdf,
                        os.path.join(DOCS_DIR, "btc_correlation_heatmap.png"),
                        "Pairwise Correlation Matrix — All Top Picks")

        # Page 8: Put cost comparison
        print("  Page 8: Put cost comparison")
        _add_image_page(pdf,
                        os.path.join(DOCS_DIR, "btc_put_cost_comparison.png"),
                        "ATM Protective Put Cost Comparison (3M / 6M / 12M)")

        # Pages 9-11: IWN deep dive
        print("  Pages 9-11: IWN deep dive")
        _add_put_sizing_page(pdf, "IWN")

        # Pages 12-14: CRWD deep dive
        print("  Pages 12-14: CRWD deep dive")
        _add_put_sizing_page(pdf, "CRWD")

        # Page 15: IWN/CRWD comparison chart
        print("  Page 15: IWN/CRWD comparison chart")
        _add_image_page(pdf,
                        os.path.join(DOCS_DIR, "btc_iwn_crwd_deep_dive.png"),
                        "IWN & CRWD — Put Options & Portfolio Sizing Deep Dive")

        # Page 16: Full universe top-20 by absolute correlation
        print("  Page 16: Universe top-20 by correlation")
        sorted_assets = sorted(all_assets, key=lambda x: abs(float(x["corr_3m"])), reverse=True)[:20]
        univ_rows = []
        for i, r in enumerate(sorted_assets):
            corr = float(r["corr_3m"])
            direction = "+" if corr > 0 else "-"
            univ_rows.append([
                str(i + 1),
                r["ticker"],
                f"${float(r['spot']):,.2f}",
                f"{float(r['vol_30d']):.1%}",
                f"{corr:+.4f}",
                f"{float(r['corr_6m']):+.4f}",
                f"{float(r['corr_12m']):+.4f}",
                f"{float(r['put_pct_3m']):.1%}",
                f"{float(r['1y_return']):+.1%}",
            ])
        _add_table_page(pdf,
                        "Full Universe — Top 20 by 3M Absolute Correlation with BTC",
                        f"{len(all_assets)} assets scanned (midcaps + commodities + indices)",
                        ["#", "Ticker", "Spot", "Vol", "Corr 3M", "Corr 6M", "Corr 12M", "Put 3M%", "1Y Ret"],
                        univ_rows)

    print(f"\n  Report saved: {out_path}")
    print(f"  Pages: 16")


if __name__ == "__main__":
    main()
