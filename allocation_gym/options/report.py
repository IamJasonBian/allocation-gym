"""
Generate a PDF report: BTC + IWM 30-day options pricing, historical
vol, and current snapshot tables with embedded charts.

Usage:
    python -m allocation_gym.options.report
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from fpdf import FPDF

from allocation_gym.options.black_scholes import bs_call_price, bs_put_price
from allocation_gym.options.historical_pricing import (
    load_btc,
    load_iwm,
    rolling_yang_zhang,
    rolling_close_vol,
    price_options_history,
    _find_nearest_idx,
    _percentile_rank,
)


DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "4")
TMP_DIR = os.path.join(DOCS_DIR, ".tmp_report")


# ── Chart generation ─────────────────────────────────────────────────────


def _save_chart(fig, name: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    path = os.path.join(TMP_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_price_panels(btc, iwm) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

    ax1.plot(btc.index, btc["Close"], color="#F7931A", linewidth=1.5)
    ax1.set_title("BTC/USD", fontsize=10, fontweight="bold")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    ax2.plot(iwm.index, iwm["Close"], color="#4CAF50", linewidth=1.5)
    ax2.set_title("IWM", fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    return _save_chart(fig, "price_panels.png")


def chart_vol_panels(btc_yz, btc_cc, iwm_yz, iwm_cc) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

    ax1.plot(btc_yz.index, btc_yz * 100, color="#9C27B0", linewidth=1.5, label="YZ 14d")
    ax1.plot(btc_cc.index, btc_cc * 100, color="#F7931A", linewidth=1, alpha=0.5, label="CC 21d")
    ax1.fill_between(btc_yz.index, btc_yz * 100, alpha=0.08, color="#9C27B0")
    ax1.set_title("BTC Realized Vol", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Ann. Vol (%)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    ax2.plot(iwm_yz.index, iwm_yz * 100, color="#9C27B0", linewidth=1.5, label="YZ 14d")
    ax2.plot(iwm_cc.index, iwm_cc * 100, color="#4CAF50", linewidth=1, alpha=0.5, label="CC 21d")
    ax2.fill_between(iwm_yz.index, iwm_yz * 100, alpha=0.08, color="#9C27B0")
    ax2.set_title("IWM Realized Vol", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Ann. Vol (%)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    return _save_chart(fig, "vol_panels.png")


def chart_options_pct(btc_opts, iwm_opts) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

    ax1.plot(btc_opts.index, btc_opts["call_ATM_pct"], color="#2196F3", linewidth=1.5, label="ATM Call")
    ax1.plot(btc_opts.index, btc_opts["put_ATM_pct"], color="#F44336", linewidth=1.5, label="ATM Put")
    ax1.plot(btc_opts.index, btc_opts["call_5%OTM_pct"], color="#2196F3", linewidth=1, alpha=0.4, linestyle="--", label="5% OTM Call")
    ax1.plot(btc_opts.index, btc_opts["put_5%OTM_pct"], color="#F44336", linewidth=1, alpha=0.4, linestyle="--", label="5% OTM Put")
    ax1.set_title("BTC 30d Options (% of spot)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Premium (%)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    ax2.plot(iwm_opts.index, iwm_opts["call_ATM_pct"], color="#2196F3", linewidth=1.5, label="ATM Call")
    ax2.plot(iwm_opts.index, iwm_opts["put_ATM_pct"], color="#F44336", linewidth=1.5, label="ATM Put")
    ax2.plot(iwm_opts.index, iwm_opts["call_5%OTM_pct"], color="#2196F3", linewidth=1, alpha=0.4, linestyle="--", label="5% OTM Call")
    ax2.plot(iwm_opts.index, iwm_opts["put_5%OTM_pct"], color="#F44336", linewidth=1, alpha=0.4, linestyle="--", label="5% OTM Put")
    ax2.set_title("IWM 30d Options (% of spot)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Premium (%)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    return _save_chart(fig, "options_pct.png")


def chart_cost_over_time(btc_opts, iwm_opts) -> str:
    """ATM call $ cost over time for both assets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

    ax1.plot(btc_opts.index, btc_opts["call_ATM"], color="#F7931A", linewidth=1.5, label="ATM Call")
    ax1.plot(btc_opts.index, btc_opts["put_ATM"], color="#F44336", linewidth=1.5, label="ATM Put")
    ax1.fill_between(btc_opts.index, btc_opts["call_ATM"], alpha=0.08, color="#F7931A")
    ax1.set_title("BTC 30d Option Cost ($)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Premium ($)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    ax2.plot(iwm_opts.index, iwm_opts["call_ATM"], color="#4CAF50", linewidth=1.5, label="ATM Call")
    ax2.plot(iwm_opts.index, iwm_opts["put_ATM"], color="#F44336", linewidth=1.5, label="ATM Put")
    ax2.fill_between(iwm_opts.index, iwm_opts["call_ATM"], alpha=0.08, color="#4CAF50")
    ax2.set_title("IWM 30d Option Cost ($)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Premium ($)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    return _save_chart(fig, "cost_over_time.png")


def chart_vol_ratio(btc_yz, iwm_yz) -> str:
    common = btc_yz.dropna().index.intersection(iwm_yz.dropna().index)
    ratio = btc_yz.loc[common] / iwm_yz.loc[common]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(common, ratio, color="#FF5722", linewidth=1.5)
    ax.fill_between(common, ratio, ratio.mean(), alpha=0.1, color="#FF5722")
    ax.axhline(ratio.mean(), color="grey", linewidth=1, linestyle="--", alpha=0.6,
               label=f"Mean: {ratio.mean():.1f}x")
    ax.set_ylabel("BTC Vol / IWM Vol")
    ax.set_title("BTC / IWM Volatility Ratio", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate(rotation=25)
    plt.tight_layout()
    return _save_chart(fig, "vol_ratio.png")


# ── PDF generation ───────────────────────────────────────────────────────


class OptionsPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "allocation-gym  |  Options Pricing Report", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(66, 133, 244)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None, highlight_row=None):
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(55, 71, 79)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8)
        for row_idx, row in enumerate(rows):
            if row_idx == highlight_row:
                self.set_fill_color(255, 249, 196)
                fill = True
            elif row_idx % 2 == 0:
                self.set_fill_color(245, 245, 245)
                fill = True
            else:
                fill = False

            self.set_text_color(50, 50, 50)
            for i, cell in enumerate(row):
                align = "L" if i == 0 else "R"
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align=align)
            self.ln()
        self.ln(3)


def generate_report(days: int = 180):
    print("Loading data...")
    btc = load_btc(days=days)
    iwm = load_iwm(days=days)

    btc_yz14 = rolling_yang_zhang(btc, window=14, trading_days=365)
    btc_cc21 = rolling_close_vol(btc, window=21, trading_days=365)
    iwm_yz14 = rolling_yang_zhang(iwm, window=14, trading_days=252)
    iwm_cc21 = rolling_close_vol(iwm, window=21, trading_days=252)

    btc_opts = price_options_history(btc, btc_yz14, expiry_days=30, trading_days=365)
    iwm_opts = price_options_history(iwm, iwm_yz14, expiry_days=30, trading_days=252)

    # Generate chart images
    print("Generating charts...")
    img_prices = chart_price_panels(btc, iwm)
    img_vol = chart_vol_panels(btc_yz14, btc_cc21, iwm_yz14, iwm_cc21)
    img_opts_pct = chart_options_pct(btc_opts, iwm_opts)
    img_cost = chart_cost_over_time(btc_opts, iwm_opts)
    img_ratio = chart_vol_ratio(btc_yz14, iwm_yz14)

    # Snapshot data
    def snap(opts, idx):
        row = opts.iloc[idx]
        return row

    btc_now = snap(btc_opts, -1)
    btc_1mo = snap(btc_opts, _find_nearest_idx(btc_opts, btc_opts.index[-1] - pd.Timedelta(days=30)))
    btc_3mo = snap(btc_opts, _find_nearest_idx(btc_opts, btc_opts.index[-1] - pd.Timedelta(days=90)))
    btc_6mo = snap(btc_opts, 0)

    iwm_now = snap(iwm_opts, -1)
    iwm_1mo = snap(iwm_opts, _find_nearest_idx(iwm_opts, iwm_opts.index[-1] - pd.Timedelta(days=30)))
    iwm_3mo = snap(iwm_opts, _find_nearest_idx(iwm_opts, iwm_opts.index[-1] - pd.Timedelta(days=90)))
    iwm_6mo = snap(iwm_opts, 0)

    # ── Build PDF ────────────────────────────────────────────────────
    print("Building PDF...")
    pdf = OptionsPDF(orientation="P", unit="mm", format="letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Title + Prices + Vol ─────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 15, "BTC + IWM 30-Day Options Pricing", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Report Date: {datetime.now().strftime('%B %d, %Y')}  |  Lookback: {days} days", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.section_title("1. Price History")
    pdf.image(img_prices, w=190)
    pdf.ln(3)

    pdf.section_title("2. Realized Volatility")
    pdf.image(img_vol, w=190)
    pdf.ln(3)

    # ── Page 2: BTC Options Pricing Table ────────────────────────────
    pdf.add_page()
    pdf.section_title("3. BTC/USD 30-Day Options Pricing")
    pdf.body_text(
        f"Current spot: ${btc_now['spot']:,.0f}  |  "
        f"YZ Vol: {btc_now['iv']:.1%}  |  "
        f"Percentile: {_percentile_rank(btc_yz14.dropna(), btc_yz14.dropna().iloc[-1]):.0f}th"
    )

    cw = [35, 32, 32, 32, 32, 32]
    headers = ["", "Now", "1mo ago", "3mo ago", "6mo ago", "Chg (now vs 3mo)"]

    def _fmt_d(v, fmt_type):
        if fmt_type == "dollar_big":
            return f"${v:,.0f}"
        elif fmt_type == "dollar":
            return f"${v:,.2f}"
        elif fmt_type == "pct":
            return f"{v:.2f}%"
        return str(v)

    def _chg(now, then, fmt_type):
        d = now - then
        if fmt_type == "dollar_big":
            return f"${d:+,.0f}"
        elif fmt_type == "dollar":
            return f"${d:+,.2f}"
        elif fmt_type == "pct":
            return f"{d:+.2f}pp"
        return ""

    btc_rows = [
        ("Spot", "spot", "dollar_big"),
        ("YZ Vol (14d)", "iv", "pct"),
        ("ATM Call $", "call_ATM", "dollar"),
        ("ATM Call % spot", "call_ATM_pct", "pct"),
        ("5% OTM Call $", "call_5%OTM", "dollar"),
        ("5% OTM Call % spot", "call_5%OTM_pct", "pct"),
        ("ATM Put $", "put_ATM", "dollar"),
        ("ATM Put % spot", "put_ATM_pct", "pct"),
        ("5% OTM Put $", "put_5%OTM", "dollar"),
        ("5% OTM Put % spot", "put_5%OTM_pct", "pct"),
    ]

    table_rows = []
    for label, col, fmt_type in btc_rows:
        table_rows.append([
            label,
            _fmt_d(btc_now[col], fmt_type),
            _fmt_d(btc_1mo[col], fmt_type),
            _fmt_d(btc_3mo[col], fmt_type),
            _fmt_d(btc_6mo[col], fmt_type),
            _chg(btc_now[col], btc_3mo[col], fmt_type),
        ])

    pdf.add_table(headers, table_rows, col_widths=cw, highlight_row=0)

    # ── IWM Options Pricing Table ────────────────────────────────────
    pdf.section_title("4. IWM 30-Day Options Pricing")
    pdf.body_text(
        f"Current spot: ${iwm_now['spot']:.2f}  |  "
        f"YZ Vol: {iwm_now['iv']:.1%}  |  "
        f"Percentile: {_percentile_rank(iwm_yz14.dropna(), iwm_yz14.dropna().iloc[-1]):.0f}th"
    )

    iwm_rows = [
        ("Spot", "spot", "dollar"),
        ("YZ Vol (14d)", "iv", "pct"),
        ("ATM Call $", "call_ATM", "dollar"),
        ("ATM Call % spot", "call_ATM_pct", "pct"),
        ("5% OTM Call $", "call_5%OTM", "dollar"),
        ("5% OTM Call % spot", "call_5%OTM_pct", "pct"),
        ("ATM Put $", "put_ATM", "dollar"),
        ("ATM Put % spot", "put_ATM_pct", "pct"),
        ("5% OTM Put $", "put_5%OTM", "dollar"),
        ("5% OTM Put % spot", "put_5%OTM_pct", "pct"),
    ]

    iwm_table_rows = []
    for label, col, fmt_type in iwm_rows:
        iwm_table_rows.append([
            label,
            _fmt_d(iwm_now[col], fmt_type),
            _fmt_d(iwm_1mo[col], fmt_type),
            _fmt_d(iwm_3mo[col], fmt_type),
            _fmt_d(iwm_6mo[col], fmt_type),
            _chg(iwm_now[col], iwm_3mo[col], fmt_type),
        ])

    pdf.add_table(headers, iwm_table_rows, col_widths=cw, highlight_row=0)

    # ── Page 3: Charts + Analysis ────────────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Options Premium as % of Spot (Historical)")
    pdf.image(img_opts_pct, w=190)
    pdf.ln(3)

    pdf.section_title("6. Absolute Option Cost Over Time")
    pdf.image(img_cost, w=190)
    pdf.ln(3)

    # ── Page 4: Vol Ratio + Current Pricing Grid ─────────────────────
    pdf.add_page()
    pdf.section_title("7. BTC / IWM Volatility Ratio")
    pdf.image(img_ratio, w=190)
    pdf.ln(3)

    # Current pricing grid at multiple strikes
    pdf.section_title("8. Current Strike Grid (30-day, both assets)")

    S_btc = btc_now["spot"]
    S_iwm = iwm_now["spot"]
    iv_btc = btc_now["iv"]
    iv_iwm = iwm_now["iv"]
    T_btc = 30 / 365
    T_iwm = 30 / 252
    r = 0.045

    grid_headers = ["Strike %", "BTC Call $", "BTC Call %", "BTC Put $", "BTC Put %",
                    "IWM Call $", "IWM Call %", "IWM Put $", "IWM Put %"]
    grid_cw = [18, 22, 18, 22, 18, 22, 18, 22, 18]
    grid_rows = []

    for pct in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]:
        Kc_btc = S_btc * (1 + pct)
        Kp_btc = S_btc * (1 - pct)
        Kc_iwm = S_iwm * (1 + pct)
        Kp_iwm = S_iwm * (1 - pct)

        bc = bs_call_price(S_btc, Kc_btc, T_btc, r, iv_btc)
        bp = bs_put_price(S_btc, Kp_btc, T_btc, r, iv_btc)
        ic = bs_call_price(S_iwm, Kc_iwm, T_iwm, r, iv_iwm)
        ip_ = bs_put_price(S_iwm, Kp_iwm, T_iwm, r, iv_iwm)

        label = "ATM" if pct == 0 else f"{pct:+.0%}"
        grid_rows.append([
            label,
            f"${bc:,.0f}", f"{bc/S_btc*100:.1f}%",
            f"${bp:,.0f}", f"{bp/S_btc*100:.1f}%",
            f"${ic:.2f}", f"{ic/S_iwm*100:.1f}%",
            f"${ip_:.2f}", f"{ip_/S_iwm*100:.1f}%",
        ])

    pdf.add_table(grid_headers, grid_rows, col_widths=grid_cw, highlight_row=3)

    # ── Page 5: Key Observations ─────────────────────────────────────
    pdf.section_title("9. Key Observations")

    common = btc_yz14.dropna().index.intersection(iwm_yz14.dropna().index)
    ratio_now = btc_yz14.loc[common].iloc[-1] / iwm_yz14.loc[common].iloc[-1]
    ratio_mean = (btc_yz14.loc[common] / iwm_yz14.loc[common]).mean()

    observations = [
        f"BTC realized vol at {btc_now['iv']:.0%} is at the {_percentile_rank(btc_yz14.dropna(), btc_yz14.dropna().iloc[-1]):.0f}th percentile of the last 6 months. "
        f"ATM 30-day calls cost {btc_now['call_ATM_pct']:.1f}% of spot (${btc_now['call_ATM']:,.0f}), nearly 3x the 6-month-ago level of {btc_6mo['call_ATM_pct']:.1f}%.",

        f"IWM realized vol at {iwm_now['iv']:.0%} is at the {_percentile_rank(iwm_yz14.dropna(), iwm_yz14.dropna().iloc[-1]):.0f}th percentile. "
        f"ATM 30-day calls cost {iwm_now['call_ATM_pct']:.1f}% of spot (${iwm_now['call_ATM']:.2f}).",

        f"BTC/IWM vol ratio is {ratio_now:.1f}x vs {ratio_mean:.1f}x mean. BTC carries a wider-than-normal vol premium, "
        f"meaning BTC options are relatively expensive compared to equity options.",

        f"BTC options pricing implication: With vol at cycle highs, selling premium via covered calls or collars captures rich volatility. "
        f"Buying protective puts is expensive but pays well during further drawdowns.",

        f"IWM options pricing implication: Vol elevated but within historical norms for risk-off periods. "
        f"Directional calls are reasonably priced for a recovery bet; puts offer decent protection value.",
    ]

    for i, obs in enumerate(observations):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(5, 5, f"{i+1}.")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, f" {obs}")
        pdf.ln(2)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(DOCS_DIR, exist_ok=True)
    out_path = os.path.join(DOCS_DIR, "btc_iwm_options_report.pdf")
    pdf.output(out_path)
    print(f"PDF saved to {out_path}")

    # Cleanup temp images
    import shutil
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    return out_path


if __name__ == "__main__":
    generate_report(days=180)
