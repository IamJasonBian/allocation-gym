#!/usr/bin/env python3
"""OTC Importance-Sampling Pricing — Audit Report Generator.

Audits the importance-sampling (IS) Monte-Carlo pricer for illiquid alt-coin
OTC derivatives by comparing it, scenario-by-scenario, against the analytic
Black-Scholes ("expected") benchmark and a plain MC estimator. Emits a
multi-page PDF plus CSV intermediates under ``docs/12/``.

Pages:
    1. Title page.
    2. Expected-vs-actual table (BS vs IS vs plain MC, std-errs, ESS,
       variance-reduction ratio).
    3. Convergence figure (price +/- std-err vs N) on the deep-OTM call.
    4. IS weight histogram + ESS annotation.
    5. Feed-drop panel: true index vs IS-reconstructed price after a drop.

This script is standalone: it prefers the real
``allocation_gym.otc_is_pricing.pricer`` once that sibling unit merges, and
otherwise uses the bundled ``scripts/_otc_fallback_pricer``.

Usage:
    python3 scripts/otc_is_pricing_audit.py [--date "June 04, 2026"]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Make scripts/ importable so the defensive fallback import resolves.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402

# --- Defensive import: real pricer if present, else bundled fallback. -------
try:
    from allocation_gym.otc_is_pricing.pricer import (  # type: ignore
        bs_price,
        price_plain_mc,
        price_is,
        PriceResult,
    )
    _PRICER_SOURCE = "allocation_gym.otc_is_pricing.pricer"
except Exception:
    from _otc_fallback_pricer import (  # type: ignore
        bs_price,
        price_plain_mc,
        price_is,
        PriceResult,
    )
    _PRICER_SOURCE = "scripts/_otc_fallback_pricer (bundled fallback)"


BASE_DIR = os.path.dirname(_THIS_DIR)
DOCS_DIR = os.path.join(BASE_DIR, "docs", "12")

# Shared MC settings — fixed seed keeps the report deterministic.
N_PATHS = 200_000
SEED = 7
RISK_FREE = 0.03

# ---------------------------------------------------------------------------
# Scenarios — alt-coin-like (high vol, illiquid OTC strikes).
# Each: (label, S, K, T, sigma, kind, note)
# ---------------------------------------------------------------------------
SCENARIOS: list[tuple[str, float, float, float, float, str, str]] = [
    ("ATM call",       100.0, 100.0, 0.50, 0.80, "call",    "at-the-money"),
    ("OTM call",       100.0, 140.0, 0.50, 0.80, "call",    "moderately OTM"),
    ("Deep-OTM call",  100.0, 220.0, 0.50, 0.80, "call",    "deep OTM (IS shines)"),
    ("Digital call",   100.0, 150.0, 0.50, 0.80, "digital", "cash-or-nothing"),
    ("Feed-drop call", 100.0, 130.0, 0.25, 0.90, "call",    "priced from stale spot"),
]

# Scenario used for the convergence + weight figures.
DEEP_OTM = ("Deep-OTM call", 100.0, 220.0, 0.50, 0.80, "call")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_scenarios() -> list[dict]:
    """Price every scenario via BS, plain MC and IS; return result rows.

    Returns:
        A list of dicts, one per scenario, with the expected (BS) price, the
        IS and plain-MC estimates and their std-errs, the IS effective sample
        size and the variance-reduction ratio.
    """
    rows: list[dict] = []
    for label, S, K, T, sigma, kind, note in SCENARIOS:
        expected = bs_price(S, K, T, RISK_FREE, sigma, kind)
        plain: PriceResult = price_plain_mc(
            S, K, T, RISK_FREE, sigma, kind, n=N_PATHS, seed=SEED
        )
        isr: PriceResult = price_is(
            S, K, T, RISK_FREE, sigma, kind,
            n=N_PATHS, seed=SEED, method="drift_tilt",
        )
        # Variance-reduction ratio = (plainMC_stderr / IS_stderr)**2.
        if isr.std_error > 0:
            var_red = (plain.std_error / isr.std_error) ** 2
        else:
            var_red = float("inf")
        rows.append(
            {
                "scenario": label,
                "note": note,
                "S": S,
                "K": K,
                "T": T,
                "sigma": sigma,
                "kind": kind,
                "expected_bs": expected,
                "actual_is": isr.price,
                "plain_mc": plain.price,
                "is_stderr": isr.std_error,
                "plain_stderr": plain.std_error,
                "ess": isr.ess,
                "var_reduction": var_red,
            }
        )
    return rows


def compute_convergence(
    n_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Price the deep-OTM call over a grid of path counts for IS and plain MC.

    Args:
        n_grid: Array of (log-spaced) path counts.

    Returns:
        Dict of arrays keyed ``is_price``, ``is_se``, ``plain_price``,
        ``plain_se`` plus the analytic ``bs`` scalar under key ``bs``.
    """
    _, S, K, T, sigma, kind = DEEP_OTM
    bs = bs_price(S, K, T, RISK_FREE, sigma, kind)

    is_price, is_se, plain_price, plain_se = [], [], [], []
    for n in n_grid:
        n = int(n)
        isr = price_is(S, K, T, RISK_FREE, sigma, kind,
                       n=n, seed=SEED, method="drift_tilt")
        pm = price_plain_mc(S, K, T, RISK_FREE, sigma, kind, n=n, seed=SEED)
        is_price.append(isr.price)
        is_se.append(isr.std_error)
        plain_price.append(pm.price)
        plain_se.append(pm.std_error)

    return {
        "bs": bs,
        "is_price": np.array(is_price),
        "is_se": np.array(is_se),
        "plain_price": np.array(plain_price),
        "plain_se": np.array(plain_se),
    }


def compute_is_weights() -> tuple[np.ndarray, float, float]:
    """Recompute the IS weights for the deep-OTM call (for the histogram).

    Mirrors the fallback pricer's tilting so we can visualize the weight
    distribution and ESS independently of the pricer's return value.

    Returns:
        Tuple ``(weights, ess, theta)``.
    """
    _, S, K, T, sigma, kind = DEEP_OTM
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(N_PATHS)
    vol_sqrt_t = sigma * np.sqrt(T)
    theta = max((np.log(K / S) - (RISK_FREE - 0.5 * sigma * sigma) * T) / vol_sqrt_t, 0.0)
    log_lr = -theta * z - 0.5 * theta * theta
    w = np.exp(log_lr)
    sum_w = w.sum()
    ess = float(sum_w * sum_w / (w * w).sum())
    return w, ess, float(theta)


def simulate_feed_drop(
    seed: int = 11,
) -> dict[str, np.ndarray]:
    """Simulate a true alt-coin index, a feed drop, and IS reconstruction.

    A GBM mid-price runs forward. Partway through, the live feed "drops": the
    observed price freezes at the last good tick. Once the feed recovers we
    have a known endpoint (last good tick -> first recovered tick) and must
    mark the illiquid OTC book *through* the gap.

    The naive mark holds the last good price flat. The IS reconstruction
    instead samples GBM paths pinned to both endpoints (a Brownian-bridge
    conditional on the recovered price) under importance weights, and marks
    the gap at the weighted-mean of those bridged paths. Because the weights
    concentrate mass on paths consistent with both the last good and first
    recovered ticks, the reconstruction tracks the true (unobserved) path far
    better than holding flat — that is the resilience this panel demonstrates.

    Returns:
        Dict with arrays ``t``, ``true``, ``observed``, ``reconstructed`` and
        an ``outage_mask`` boolean array.
    """
    rng = np.random.default_rng(seed)
    steps = 120
    dt = 1.0 / 252.0
    sigma = 0.9
    mu = 0.0  # drift of the simulated true mid
    s0 = 100.0

    # True GBM mid path.
    shocks = rng.standard_normal(steps)
    log_increments = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * shocks
    true = s0 * np.exp(np.cumsum(log_increments))
    t = np.arange(steps)

    # Feed drop: observed price freezes during the outage window.
    drop_start, drop_end = 50, 80
    outage_mask = (t >= drop_start) & (t < drop_end)
    observed = true.copy()
    last_good = float(true[drop_start - 1])  # last tick before outage
    first_recovered = float(true[drop_end])  # first tick after recovery
    observed[outage_mask] = last_good

    # IS reconstruction: importance-sample GBM log-bridges between the two
    # known endpoints. We draw many candidate log-return paths over the gap,
    # weight each by how consistent its endpoint is with the recovered price
    # (a Gaussian importance weight on the terminal mismatch), and take the
    # weighted-mean path. This reuses the same likelihood-ratio reweighting
    # idea as the option pricer (a soft conditioning instead of a hard tilt).
    idx = np.where(outage_mask)[0]
    n_gap = len(idx)
    n_bridge = 4000
    bridge_rng = np.random.default_rng(seed + 1)

    # Candidate cumulative log-returns from last_good over the gap.
    incr = ((mu - 0.5 * sigma * sigma) * dt
            + sigma * np.sqrt(dt) * bridge_rng.standard_normal((n_bridge, n_gap + 1)))
    cum = np.cumsum(incr, axis=1)              # shape (n_bridge, n_gap+1)
    paths = last_good * np.exp(cum)            # GBM candidates over the gap

    # Importance weight: how well each candidate's endpoint matches the
    # recovered price. Soft Gaussian kernel on terminal log-mismatch.
    target_logret = np.log(first_recovered / last_good)
    endpoint_logret = cum[:, -1]
    bw = sigma * np.sqrt(dt)  # bandwidth ~ one-step vol
    w = np.exp(-0.5 * ((endpoint_logret - target_logret) / bw) ** 2)
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones(n_bridge)
        w_sum = float(n_bridge)
    weights = w / w_sum

    # Weighted-mean bridged path over the in-gap steps (drop first step,
    # which is last_good itself, and align to the n_gap outage indices).
    recon_gap = (weights[:, None] * paths[:, 1:]).sum(axis=0)

    reconstructed = observed.copy().astype(float)
    reconstructed[idx] = recon_gap

    # Confirm the option-pricing IS machinery also runs on the stale spot
    # (resilience: we can still mark OTC options off the last good tick).
    _ = price_is(last_good, last_good * np.exp(RISK_FREE * n_gap * dt),
                 max(n_gap * dt, dt), RISK_FREE, sigma, "call",
                 n=20_000, seed=seed + 2, method="drift_tilt")

    return {
        "t": t.astype(float),
        "true": true,
        "observed": observed,
        "reconstructed": reconstructed,
        "outage_mask": outage_mask,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict]) -> str:
    """Write the expected-vs-actual table to CSV. Returns the path."""
    path = os.path.join(DOCS_DIR, "otc_is_pricing_expected_vs_actual.csv")
    fields = [
        "scenario", "kind", "S", "K", "T", "sigma",
        "expected_bs", "actual_is", "plain_mc",
        "is_stderr", "plain_stderr", "ess", "var_reduction",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([
                r["scenario"], r["kind"], r["S"], r["K"], r["T"], r["sigma"],
                f"{r['expected_bs']:.6f}", f"{r['actual_is']:.6f}",
                f"{r['plain_mc']:.6f}", f"{r['is_stderr']:.6f}",
                f"{r['plain_stderr']:.6f}", f"{r['ess']:.1f}",
                f"{r['var_reduction']:.4f}",
            ])
    return path


# ---------------------------------------------------------------------------
# PDF pages
# ---------------------------------------------------------------------------

def add_title_page(pdf: PdfPages, date_str: str) -> None:
    """Render the title / scope page."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.82, "OTC Importance-Sampling Pricing Audit",
            fontsize=26, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.73, "Expected (Black-Scholes) vs Actual (IS Monte-Carlo)",
            fontsize=15, ha="center", va="center", color="#555",
            transform=ax.transAxes)

    ax.plot([0.15, 0.85], [0.66, 0.66], color="#ccc", linewidth=2,
            transform=ax.transAxes)

    blurb = [
        "Audits the importance-sampling MC pricer for illiquid alt-coin OTC",
        "derivatives. For each scenario we compare the analytic Black-Scholes",
        "price (expected) against drift-tilt IS and plain MC (actual), and",
        "verify that IS reduces variance — especially for deep-OTM strikes —",
        "and that pricing stays resilient through a simulated feed drop.",
    ]
    for i, line in enumerate(blurb):
        ax.text(0.5, 0.58 - i * 0.045, line, fontsize=12, ha="center",
                va="center", transform=ax.transAxes)

    details = [
        f"Report Date:   {date_str}",
        f"Pricer:        {_PRICER_SOURCE}",
        f"MC Paths:      {N_PATHS:,}   Seed: {SEED}",
        f"Risk-Free:     {RISK_FREE:.1%}",
        f"Scenarios:     {len(SCENARIOS)}  (ATM / OTM / deep-OTM / digital / feed-drop)",
    ]
    for i, line in enumerate(details):
        ax.text(0.5, 0.32 - i * 0.045, line, fontsize=12, ha="center",
                va="center", fontfamily="monospace", transform=ax.transAxes)

    ax.text(0.5, 0.05, "allocation-gym  |  otc_is_pricing",
            fontsize=10, ha="center", color="#999", transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, rows: list[dict]) -> None:
    """Render the expected-vs-actual comparison table."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.suptitle("Expected vs Actual — Per Scenario", fontsize=16,
                 fontweight="bold", y=0.95)
    ax.text(0.5, 0.90,
            "Expected = Black-Scholes analytic   |   Actual = drift-tilt IS MC",
            fontsize=11, ha="center", va="top", color="#666",
            transform=ax.transAxes)

    headers = [
        "Scenario", "Expected\n(BS)", "Actual\n(IS)", "Plain\nMC",
        "IS\nstd-err", "Plain\nstd-err", "ESS", "Var-red\nratio",
    ]
    table_rows = []
    for r in rows:
        vr = r["var_reduction"]
        vr_str = "inf" if vr == float("inf") else f"{vr:.1f}x"
        table_rows.append([
            r["scenario"],
            f"{r['expected_bs']:.4f}",
            f"{r['actual_is']:.4f}",
            f"{r['plain_mc']:.4f}",
            f"{r['is_stderr']:.5f}",
            f"{r['plain_stderr']:.5f}",
            f"{r['ess']:,.0f}",
            vr_str,
        ])

    table = ax.table(cellText=table_rows, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    n_cols = len(headers)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    for i in range(1, len(table_rows) + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor("#F8F9FA" if i % 2 == 0 else "#FFFFFF")

    ax.text(0.5, 0.10,
            "Var-reduction ratio = (plain-MC std-err / IS std-err)^2.  "
            "Ratio > 1 means IS is more efficient; the gain grows for deeper OTM strikes.",
            fontsize=9, ha="center", va="center", color="#444",
            transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def add_convergence_page(pdf: PdfPages, n_grid: np.ndarray,
                         conv: dict[str, np.ndarray]) -> None:
    """Render IS vs plain MC convergence on the deep-OTM call."""
    fig, ax = plt.subplots(figsize=(11, 8.5))

    bs = conv["bs"]
    ax.axhline(bs, color="black", linestyle="--", linewidth=1.2,
               label=f"Black-Scholes (expected) = {bs:.4f}")

    ax.errorbar(n_grid, conv["is_price"], yerr=conv["is_se"], marker="o",
                capsize=3, color="#1f77b4", label="IS (drift tilt)")
    ax.errorbar(n_grid, conv["plain_price"], yerr=conv["plain_se"], marker="s",
                capsize=3, color="#d62728", label="Plain MC")

    ax.set_xscale("log")
    ax.set_xlabel("Number of paths (N, log scale)")
    ax.set_ylabel("Estimated price +/- std-err")
    ax.set_title("Convergence — Deep-OTM Call (IS vs Plain MC)",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def add_weights_page(pdf: PdfPages, w: np.ndarray, ess: float,
                     theta: float) -> None:
    """Render the IS weight histogram + ESS annotation."""
    fig, ax = plt.subplots(figsize=(11, 8.5))

    ax.hist(w, bins=80, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Importance weight  w = exp(-theta*Z - 0.5*theta^2)")
    ax.set_ylabel("Path count (log scale)")
    ax.set_title("IS Weight Distribution — Deep-OTM Call",
                 fontsize=15, fontweight="bold")

    ess_frac = ess / len(w)
    txt = (
        f"theta (drift tilt) = {theta:.4f}\n"
        f"N paths            = {len(w):,}\n"
        f"ESS                = {ess:,.0f}\n"
        f"ESS / N            = {ess_frac:.1%}\n"
        f"mean(w)            = {w.mean():.4f}"
    )
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontfamily="monospace", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#F8F9FA", edgecolor="#ccc"))

    pdf.savefig(fig)
    plt.close(fig)


def add_feed_drop_page(pdf: PdfPages, feed: dict[str, np.ndarray]) -> None:
    """Render the feed-drop true-vs-reconstructed panel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    t = feed["t"]
    mask = feed["outage_mask"]

    ax1.plot(t, feed["true"], color="#2C3E50", linewidth=1.5,
             label="True index (unobserved)")
    ax1.plot(t, feed["observed"], color="#d62728", linewidth=1.2,
             linestyle=":", label="Observed (frozen during drop)")
    ax1.plot(t, feed["reconstructed"], color="#1f77b4", linewidth=1.5,
             label="IS reconstruction")
    # Shade outage window.
    if mask.any():
        ax1.axvspan(t[mask][0], t[mask][-1], color="#ffcccc", alpha=0.4,
                    label="Feed-drop window")
    ax1.set_ylabel("Price")
    ax1.set_title("Feed-Drop Resilience — True vs IS-Reconstructed Index",
                  fontsize=15, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Error panels: reconstruction error vs naive hold-flat error.
    recon_err = feed["reconstructed"] - feed["true"]
    flat_err = feed["observed"] - feed["true"]
    ax2.plot(t, flat_err, color="#d62728", linestyle=":",
             label="Hold-flat error")
    ax2.plot(t, recon_err, color="#1f77b4", label="IS-reconstruction error")
    ax2.axhline(0, color="black", linewidth=0.8)
    if mask.any():
        ax2.axvspan(t[mask][0], t[mask][-1], color="#ffcccc", alpha=0.4)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Error")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Entry point: compute everything and write the PDF + CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", default="(generic build)",
        help="Report date string shown on the title page (kept out of the "
             "computation for determinism).",
    )
    args = parser.parse_args(argv)

    os.makedirs(DOCS_DIR, exist_ok=True)
    pdf_path = os.path.join(DOCS_DIR, "otc_is_pricing_report.pdf")

    print("=" * 64)
    print("  OTC Importance-Sampling Pricing — Audit Report")
    print("=" * 64)
    print(f"  Pricer source: {_PRICER_SOURCE}")

    print("  Computing scenarios ...")
    rows = compute_scenarios()
    for r in rows:
        print(f"    {r['scenario']:<14} BS={r['expected_bs']:.4f} "
              f"IS={r['actual_is']:.4f} plain={r['plain_mc']:.4f} "
              f"var-red={r['var_reduction']:.1f}x")

    csv_path = write_csv(rows)
    print(f"  CSV written: {csv_path}")

    print("  Computing convergence grid ...")
    n_grid = np.unique(np.round(np.logspace(2.7, 5.3, 8)).astype(int))
    conv = compute_convergence(n_grid)

    print("  Computing IS weights ...")
    w, ess, theta = compute_is_weights()

    print("  Simulating feed drop ...")
    feed = simulate_feed_drop()

    print("  Writing PDF ...")
    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf, args.date)
        add_table_page(pdf, rows)
        add_convergence_page(pdf, n_grid, conv)
        add_weights_page(pdf, w, ess, theta)
        add_feed_drop_page(pdf, feed)

    print(f"\n  Report saved: {pdf_path}")
    print("  Pages: 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
