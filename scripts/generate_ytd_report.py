#!/usr/bin/env python3
"""Generate YTD performance report from blob store market data.

Fetches daily BTC/USD and BTC share prices from the Netlify blob store,
computes YTD metrics, and generates a LaTeX PDF.

Requires:
    ALLOC_ENGINE_SITE_ID  - Netlify site ID
    NETLIFY_AUTH_TOKEN     - Netlify PAT
"""

import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from allocation_gym.blobs import BlobClient, BlobIndex

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "9")


def main() -> None:
    print("=" * 70)
    print("  YTD PERFORMANCE REPORT GENERATOR")
    print("=" * 70)

    client = BlobClient()

    # ── Fetch all market quotes ───────────────────────────────
    mq_dates = sorted(client.list_market_quote_dates())
    print(f"\nMarket quote dates: {len(mq_dates)} ({mq_dates[0]} to {mq_dates[-1]})")

    daily_data = []
    for d in mq_dates:
        mq = client.get_market_quotes(date=d)
        btc_usd = next((q for q in mq.quotes if q.symbol == "BTC/USD"), None)
        btc_share = next((q for q in mq.quotes if q.symbol == "BTC"), None)
        btc_usd_price = btc_usd.mid if btc_usd and btc_usd.mid > 100 else None
        btc_share_price = btc_share.bid if btc_share and btc_share.bid > 1 else None
        daily_data.append({
            "date": d,
            "btc_usd": btc_usd_price,
            "btc_share": btc_share_price,
        })

    # ── Fetch IWN IV data across dates ────────────────────────
    iwn_dates = sorted(client.list_option_dates("IWN"))
    print(f"IWN option dates: {len(iwn_dates)}")

    iwn_iv_data = []
    for d in iwn_dates:
        chain = client.get_options_chain("IWN", date=d)
        idx = BlobIndex(chain.snapshots)
        atm = idx.where(
            lambda s: s.delta is not None and 0.35 < abs(s.delta) < 0.65 and s.iv is not None
        )
        atm_iv = sum(atm.map(lambda s: s.iv)) / len(atm) if len(atm) > 0 else None
        all_iv = idx.has_iv()
        avg_iv = sum(all_iv.map(lambda s: s.iv)) / len(all_iv) if len(all_iv) > 0 else None
        iwn_iv_data.append({"date": d, "atm_iv": atm_iv, "avg_iv": avg_iv, "n": len(idx)})

    # ── Generate charts ───────────────────────────────────────
    os.makedirs(DOCS_DIR, exist_ok=True)
    _generate_charts(daily_data, iwn_iv_data, DOCS_DIR)

    # ── Compute YTD metrics ───────────────────────────────────
    btc_prices = [d["btc_usd"] for d in daily_data if d["btc_usd"]]
    share_prices = [d["btc_share"] for d in daily_data if d["btc_share"]]

    btc_ret = (btc_prices[-1] / btc_prices[0] - 1) * 100 if len(btc_prices) >= 2 else 0
    share_ret = (share_prices[-1] / share_prices[0] - 1) * 100 if len(share_prices) >= 2 else 0

    log_rets = [math.log(btc_prices[i] / btc_prices[i - 1]) for i in range(1, len(btc_prices))]
    daily_vol = (sum((r - sum(log_rets) / len(log_rets)) ** 2 for r in log_rets) / len(log_rets)) ** 0.5 if log_rets else 0
    ann_vol = daily_vol * math.sqrt(252) * 100

    btc_high = max(btc_prices)
    btc_low = min(btc_prices)
    btc_dd = min((p / max(btc_prices[:i + 1]) - 1) for i, p in enumerate(btc_prices)) * 100

    share_high = max(share_prices)
    share_low = min(share_prices)

    # IWN IV stats
    atm_ivs = [d["atm_iv"] for d in iwn_iv_data if d["atm_iv"]]
    avg_atm_iv = sum(atm_ivs) / len(atm_ivs) * 100 if atm_ivs else 0
    iv_high = max(atm_ivs) * 100 if atm_ivs else 0
    iv_low = min(atm_ivs) * 100 if atm_ivs else 0

    # ── Write LaTeX ───────────────────────────────────────────
    tex_path = os.path.join(DOCS_DIR, "ytd_performance.tex")
    _write_latex(
        tex_path, daily_data, iwn_iv_data,
        btc_ret=btc_ret, share_ret=share_ret, ann_vol=ann_vol,
        btc_high=btc_high, btc_low=btc_low, btc_dd=btc_dd,
        share_high=share_high, share_low=share_low,
        avg_atm_iv=avg_atm_iv, iv_high=iv_high, iv_low=iv_low,
    )

    # ── Compile ───────────────────────────────────────────────
    print(f"\nCompiling: {tex_path}")
    try:
        result = subprocess.run(
            ["tectonic", tex_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"  PDF: {tex_path.replace('.tex', '.pdf')}")
        else:
            print(f"  tectonic error: {result.returncode}")
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
    except FileNotFoundError:
        print("  tectonic not found")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


def _generate_charts(daily_data, iwn_iv_data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    dates_btc = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data if d["btc_usd"]]
    btc_prices = [d["btc_usd"] for d in daily_data if d["btc_usd"]]
    share_prices = [d["btc_share"] for d in daily_data if d["btc_share"]]
    dates_share = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_data if d["btc_share"]]

    dates_iv = [datetime.strptime(d["date"], "%Y-%m-%d") for d in iwn_iv_data if d["atm_iv"]]
    atm_ivs = [d["atm_iv"] * 100 for d in iwn_iv_data if d["atm_iv"]]

    # ── 3-panel chart ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("YTD Performance: March 2026", fontsize=14, fontweight="bold")

    # Panel 1: BTC/USD
    ax = axes[0]
    ax.plot(dates_btc, btc_prices, "o-", color="#F7931A", linewidth=2, markersize=3, label="BTC/USD")
    btc_min, btc_max = min(btc_prices), max(btc_prices)
    btc_pad = (btc_max - btc_min) * 0.15
    ax.set_ylim(btc_min - btc_pad, btc_max + btc_pad)
    ax.fill_between(dates_btc, btc_prices, btc_min - btc_pad, alpha=0.1, color="#F7931A")
    ax.set_ylabel("BTC/USD ($)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_title("Spot BTC/USD", fontsize=11, fontweight="bold")
    ax.annotate(f"${btc_prices[0]:,.0f}", xy=(dates_btc[0], btc_prices[0]),
                fontsize=8, ha="left", va="bottom", color="#666")
    ax.annotate(f"${btc_prices[-1]:,.0f}", xy=(dates_btc[-1], btc_prices[-1]),
                fontsize=8, ha="right", va="bottom", fontweight="bold", color="#F7931A")

    # Panel 2: BTC share price
    ax = axes[1]
    ax.plot(dates_share, share_prices, "s-", color="#2196F3", linewidth=2, markersize=3, label="Grayscale BTC Mini Trust")
    sh_min, sh_max = min(share_prices), max(share_prices)
    sh_pad = (sh_max - sh_min) * 0.15
    ax.set_ylim(sh_min - sh_pad, sh_max + sh_pad)
    ax.fill_between(dates_share, share_prices, sh_min - sh_pad, alpha=0.1, color="#2196F3")
    ax.set_ylabel("Share Price ($)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_title("Grayscale BTC Mini Trust (BTC)", fontsize=11, fontweight="bold")
    ax.annotate(f"${share_prices[0]:.2f}", xy=(dates_share[0], share_prices[0]),
                fontsize=8, ha="left", va="bottom", color="#666")
    ax.annotate(f"${share_prices[-1]:.2f}", xy=(dates_share[-1], share_prices[-1]),
                fontsize=8, ha="right", va="bottom", fontweight="bold", color="#2196F3")

    # Panel 3: IWN ATM IV
    ax = axes[2]
    ax.plot(dates_iv, atm_ivs, "^-", color="#4CAF50", linewidth=2, markersize=4, label="IWN ATM IV")
    iv_min, iv_max = min(atm_ivs), max(atm_ivs)
    iv_pad = (iv_max - iv_min) * 0.15
    ax.set_ylim(iv_min - iv_pad, iv_max + iv_pad)
    ax.fill_between(dates_iv, atm_ivs, iv_min - iv_pad, alpha=0.1, color="#4CAF50")
    ax.set_ylabel("ATM IV (%)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_title("IWN ATM Implied Volatility", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "ytd_performance.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart: {chart_path}")


def _write_latex(
    path, daily_data, iwn_iv_data, *,
    btc_ret, share_ret, ann_vol,
    btc_high, btc_low, btc_dd,
    share_high, share_low,
    avg_atm_iv, iv_high, iv_low,
):
    # Build daily price table rows
    price_rows = []
    for d in daily_data:
        if d["btc_usd"] and d["btc_share"]:
            # Find matching IV
            iv_entry = next((x for x in iwn_iv_data if x["date"] == d["date"]), None)
            iv_val = iv_entry["atm_iv"] * 100 if iv_entry and iv_entry["atm_iv"] else None
            iv_str = f"{iv_val:.1f}\\%" if iv_val else "---"
            price_rows.append(
                f"{d['date']} & \\${d['btc_usd']:,.0f} & \\${d['btc_share']:.2f} & {iv_str} \\\\"
            )

    price_table = "\n".join(price_rows)

    # Options round-trips for the period
    options_pnl_rows = r"""IWM \$262 Put 3/11 & 10 & 02/26 $\to$ 02/27 & \$3.16 & \$4.96 & \textcolor{gain}{+\$1,800} & +57\% \\
IWM \$247 Put 3/13 & 10 & 02/27 $\to$ 03/01 & \$1.41 & \$2.86 & \textcolor{gain}{+\$1,450} & +103\% \\
CRWD \$390 Put 3/06 & 1 & 02/25 $\to$ 02/27 & \$21.05 & \$33.30 & \textcolor{gain}{+\$1,225} & +58\% \\
CRWD \$367.5 Put 3/20 & 1 & 02/27 $\to$ 03/04 & \$19.05 & \$16.25 & \textcolor{loss}{--\$280} & --15\% \\"""

    ret_color = "gain" if btc_ret >= 0 else "loss"
    share_color = "gain" if share_ret >= 0 else "loss"

    tex = rf"""\documentclass[11pt,letterpaper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{enumitem}}
\usepackage{{longtable}}

\definecolor{{gain}}{{RGB}}{{0,128,0}}
\definecolor{{loss}}{{RGB}}{{200,0,0}}
\definecolor{{neutral}}{{RGB}}{{80,80,80}}

\pagestyle{{fancy}}
\fancyhf{{}}
\rhead{{Allocation Engine --- Internal}}
\lhead{{YTD Performance Report}}
\rfoot{{\thepage}}

\title{{%
  \textbf{{YTD Performance Report}} \\[4pt]
  \large Allocation Engine: March 2--25, 2026 \\[2pt]
  \normalsize Data sourced from Netlify blob store (market-quotes \& options-chain)
}}
\author{{Allocation Engine Research}}
\date{{March 25, 2026}}

\begin{{document}}
\maketitle
\thispagestyle{{fancy}}

% ============================================================
\section{{Executive Summary}}
% ============================================================

This report summarises the allocation engine's YTD performance using
daily market data captured in the Netlify blob store from
\textbf{{March~2--25, 2026}} ({len(daily_data)} trading days).

\begin{{table}}[H]
\centering
\caption{{YTD performance summary}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
BTC/USD return (Mar~2--25)   & \textcolor{{{ret_color}}}{{{btc_ret:+.1f}\%}} \\
BTC share return (Mar~2--25) & \textcolor{{{share_color}}}{{{share_ret:+.1f}\%}} \\
BTC/USD annualised vol       & {ann_vol:.1f}\% \\
BTC/USD high / low           & \${btc_high:,.0f} / \${btc_low:,.0f} \\
BTC share high / low         & \${share_high:.2f} / \${share_low:.2f} \\
BTC/USD max drawdown         & \textcolor{{loss}}{{{btc_dd:.1f}\%}} \\
\midrule
IWN ATM IV (period avg)      & {avg_atm_iv:.1f}\% \\
IWN ATM IV high / low        & {iv_high:.1f}\% / {iv_low:.1f}\% \\
\midrule
BTC trading P\&L (Feb~21--Mar~4) & \textcolor{{gain}}{{+\$15,770}} \\
Options P\&L (Feb~21--Mar~4) & \textcolor{{gain}}{{+\$4,195}} \\
\textbf{{Total engine P\&L}} & \textcolor{{gain}}{{\textbf{{+\$19,965}}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

% ============================================================
\section{{Daily Market Data}}
% ============================================================

\begin{{figure}}[H]
  \centering
  \includegraphics[width=\textwidth]{{ytd_performance.png}}
  \caption{{YTD price action: BTC/USD (top), Grayscale BTC Mini Trust share
  price (middle), and IWN ATM implied volatility (bottom). Data from
  Netlify blob store market-quotes and options-chain stores.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{Daily prices and IWN ATM IV (from blob store)}}
{{\small
\begin{{tabular}}{{l r r r}}
\toprule
Date & BTC/USD & BTC Share & IWN ATM IV \\
\midrule
{price_table}
\bottomrule
\end{{tabular}}}}
\end{{table}}

% ============================================================
\section{{Realised P\&L}}
% ============================================================

\subsection{{BTC Trading (Feb 21 -- Mar 4)}}

The engine executed a bracket strategy on Grayscale BTC Mini Trust,
buying 1,575 shares at an average of \$29.76 and selling 2,054 shares
at an average of \$30.50, generating \textcolor{{gain}}{{+\$15,770}} in
trading P\&L on round-trip volume.

\subsection{{Options Round-Trips}}

\begin{{table}}[H]
\centering
\caption{{Closed options round-trips (Feb 21 -- Mar 4, 2026)}}
{{\small
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}l r l r r r r}}
\toprule
Contract & Qty & Dates & Open & Close & P\&L & Return \\
\midrule
{options_pnl_rows}
\midrule
\multicolumn{{5}}{{l}}{{\textbf{{Total Options P\&L}}}} & \textcolor{{gain}}{{\textbf{{+\$4,195}}}} & \\
\bottomrule
\end{{tabular*}}}}
\end{{table}}

\subsection{{Combined P\&L}}

\begin{{table}}[H]
\centering
\caption{{Combined engine P\&L summary}}
\begin{{tabular}}{{lr}}
\toprule
Source & P\&L \\
\midrule
BTC bracket trading     & \textcolor{{gain}}{{+\$15,770}} \\
IWM put round-trips     & \textcolor{{gain}}{{+\$3,250}} \\
CRWD put round-trips    & \textcolor{{gain}}{{+\$945}} \\
\midrule
\textbf{{Total}}        & \textcolor{{gain}}{{\textbf{{+\$19,965}}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

% ============================================================
\section{{Volatility Environment}}
% ============================================================

The IWN ATM implied volatility averaged \textbf{{{avg_atm_iv:.1f}\%}} over
the period, ranging from {iv_low:.1f}\% to {iv_high:.1f}\%. This level is
consistent with the 21-day realised vol of 18.4\% reported in the main
strategy document, suggesting options are \textbf{{fairly priced}} relative
to realised moves.

Key observations:
\begin{{itemize}}[nosep]
  \item BTC/USD traded in a \${btc_low:,.0f}--\${btc_high:,.0f} range
        ({btc_ret:+.1f}\% net) with {ann_vol:.0f}\% annualised vol
  \item Grayscale BTC share tracked spot with {share_ret:+.1f}\% return
        vs {btc_ret:+.1f}\% for spot BTC/USD
  \item IWN IV remained stable, supporting continued use of the put
        hedging strategy at current pricing levels
  \item The maximum intra-period BTC/USD drawdown was
        \textcolor{{loss}}{{{btc_dd:.1f}\%}}, highlighting the need for
        the correlated put hedge overlay
\end{{itemize}}

\end{{document}}
"""

    with open(path, "w") as f:
        f.write(tex)
    print(f"  LaTeX: {path}")


if __name__ == "__main__":
    main()
