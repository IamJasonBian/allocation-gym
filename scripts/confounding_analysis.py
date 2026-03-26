#!/usr/bin/env python3
"""
Confounding factor analysis: BTC ↔ IWN co-movement.

Checks whether VIX, DXY, and treasury yields (2Y, 10Y, 2s10s spread)
explain away the BTC-IWN correlation at 3M, 6M, and 12M horizons.

Outputs partial correlations and a LaTeX table.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


def fetch_data() -> pd.DataFrame:
    """Fetch daily closes for all tickers."""
    tickers = [
        "BTC",       # Grayscale BTC Mini Trust
        "IWN",       # iShares Russell 2000 Value
        "^VIX",      # CBOE VIX
        "DX-Y.NYB",  # US Dollar Index
        "^IRX",      # 3-month T-bill (proxy; we'll use 2Y/10Y below)
    ]

    print("Fetching price data...")
    data = yf.download(tickers, period="2y", interval="1d",
                       auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    # Fetch treasury yields separately (FRED-sourced via yfinance)
    # 2-Year: ^TWO doesn't exist on yfinance; use proxy tickers
    treasury_tickers = ["^TNX", "^TYX"]  # 10Y, 30Y
    t_data = yf.download(treasury_tickers, period="2y", interval="1d",
                         auto_adjust=True, progress=False, threads=True)
    if isinstance(t_data.columns, pd.MultiIndex):
        t_closes = t_data["Close"]
    else:
        t_closes = t_data

    # Try to get 2-year yield
    two_year_tickers = ["2YY=F", "^IRX"]
    t2_data = yf.download(two_year_tickers, period="2y", interval="1d",
                          auto_adjust=True, progress=False, threads=True)
    if isinstance(t2_data.columns, pd.MultiIndex):
        t2_closes = t2_data["Close"]
    else:
        t2_closes = t2_data

    result = pd.DataFrame()
    col_map = {
        "BTC": "BTC",
        "IWN": "IWN",
        "^VIX": "VIX",
        "DX-Y.NYB": "DXY",
    }

    for src, dst in col_map.items():
        if src in closes.columns:
            result[dst] = closes[src]

    # 10-year yield
    if "^TNX" in t_closes.columns:
        result["UST_10Y"] = t_closes["^TNX"]

    # 2-year yield proxy
    if "2YY=F" in t2_closes.columns:
        result["UST_2Y"] = t2_closes["2YY=F"]
    elif "^IRX" in t2_closes.columns:
        # 3-month as rough proxy
        result["UST_2Y"] = t2_closes["^IRX"]
    elif "^IRX" in closes.columns:
        result["UST_2Y"] = closes["^IRX"]

    result = result.dropna(how="all").sort_index()

    # Compute 2s10s spread if both available
    if "UST_2Y" in result.columns and "UST_10Y" in result.columns:
        result["SPREAD_2s10s"] = result["UST_10Y"] - result["UST_2Y"]

    print(f"  Date range: {result.index[0].date()} to {result.index[-1].date()}")
    print(f"  Columns: {list(result.columns)}")
    for col in result.columns:
        valid = result[col].dropna()
        if len(valid) > 0:
            print(f"    {col}: {len(valid)} days, last={valid.iloc[-1]:.2f}")

    return result


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def partial_correlation(x: pd.Series, y: pd.Series, z: pd.DataFrame) -> float:
    """
    Partial correlation between x and y, controlling for columns in z.
    Uses regression-based approach: residualize x and y on z, then correlate.
    """
    from numpy.linalg import lstsq

    # Drop NaN rows across all
    combined = pd.concat([x.rename("x"), y.rename("y"), z], axis=1).dropna()
    if len(combined) < 10:
        return float("nan")

    X = combined["x"].values
    Y = combined["y"].values
    Z = combined[z.columns].values

    # Add intercept
    Z_aug = np.column_stack([np.ones(len(Z)), Z])

    # Residualize x
    beta_x, _, _, _ = lstsq(Z_aug, X, rcond=None)
    resid_x = X - Z_aug @ beta_x

    # Residualize y
    beta_y, _, _, _ = lstsq(Z_aug, Y, rcond=None)
    resid_y = Y - Z_aug @ beta_y

    # Correlation of residuals
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return float("nan")

    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def run_analysis(prices: pd.DataFrame) -> dict:
    """Run confounding analysis at 3M, 6M, 12M windows."""
    windows = {
        "3M": 63,
        "6M": 126,
        "12M": 252,
    }

    log_rets = compute_log_returns(prices)

    confounders = ["VIX", "DXY", "UST_10Y", "UST_2Y", "SPREAD_2s10s"]
    available_confounders = [c for c in confounders if c in log_rets.columns]

    # Friendly names for display
    confounder_labels = {
        "VIX": "VIX",
        "DXY": "DXY (Dollar)",
        "UST_10Y": "10Y Yield",
        "UST_2Y": "2Y Yield",
        "SPREAD_2s10s": "2s10s Spread",
    }

    results = {}

    for label, days in windows.items():
        recent = log_rets.tail(days)

        if "BTC" not in recent.columns or "IWN" not in recent.columns:
            print(f"  {label}: BTC or IWN data missing, skipping")
            continue

        btc = recent["BTC"]
        iwn = recent["IWN"]

        # Raw correlation
        raw_corr = float(btc.corr(iwn))

        # Partial correlations controlling for each confounder individually
        individual = {}
        for cf in available_confounders:
            if cf in recent.columns:
                pcorr = partial_correlation(btc, iwn, recent[[cf]])
                individual[cf] = pcorr

        # Partial correlation controlling for ALL confounders
        all_cf_cols = [c for c in available_confounders if c in recent.columns]
        if all_cf_cols:
            pcorr_all = partial_correlation(btc, iwn, recent[all_cf_cols])
        else:
            pcorr_all = raw_corr

        results[label] = {
            "days": days,
            "actual_days": len(recent.dropna(subset=["BTC", "IWN"])),
            "raw_corr": raw_corr,
            "individual": individual,
            "partial_all": pcorr_all,
        }

        print(f"\n  {label} ({results[label]['actual_days']} days):")
        print(f"    Raw BTC↔IWN correlation: {raw_corr:+.3f}")
        for cf, pc in individual.items():
            delta = pc - raw_corr
            print(f"    Controlling for {confounder_labels.get(cf, cf)}: {pc:+.3f} (Δ={delta:+.3f})")
        print(f"    Controlling for ALL: {pcorr_all:+.3f} (Δ={pcorr_all - raw_corr:+.3f})")

    return results


def generate_latex(results: dict) -> str:
    """Generate LaTeX table from results."""
    confounder_labels = {
        "VIX": "VIX",
        "DXY": "DXY (Dollar)",
        "UST_10Y": "10Y Yield",
        "UST_2Y": "2Y Yield",
        "SPREAD_2s10s": "2s10s Spread",
    }

    windows = ["3M", "6M", "12M"]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{BTC $\leftrightarrow$ IWN partial correlations controlling for confounding factors}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Controlling For & 3M & 6M & 12M \\")
    lines.append(r"\midrule")

    # Raw correlation row
    vals = []
    for w in windows:
        if w in results:
            vals.append(f"{results[w]['raw_corr']:+.3f}")
        else:
            vals.append("---")
    lines.append(f"\\textbf{{None (raw)}} & {' & '.join(vals)} \\\\")

    # Individual confounder rows
    # Gather all confounders across windows
    all_cfs = []
    for w in windows:
        if w in results:
            for cf in results[w]["individual"]:
                if cf not in all_cfs:
                    all_cfs.append(cf)

    for cf in all_cfs:
        vals = []
        for w in windows:
            if w in results and cf in results[w]["individual"]:
                vals.append(f"{results[w]['individual'][cf]:+.3f}")
            else:
                vals.append("---")
        label = confounder_labels.get(cf, cf)
        lines.append(f"{label} & {' & '.join(vals)} \\\\")

    lines.append(r"\midrule")

    # All confounders row
    vals = []
    for w in windows:
        if w in results:
            vals.append(f"{results[w]['partial_all']:+.3f}")
        else:
            vals.append("---")
    lines.append(f"\\textbf{{All factors}} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("  CONFOUNDING ANALYSIS: BTC ↔ IWN CO-MOVEMENT")
    print("=" * 60)

    prices = fetch_data()
    results = run_analysis(prices)
    latex = generate_latex(results)

    print("\n" + "=" * 60)
    print("  LATEX OUTPUT")
    print("=" * 60)
    print(latex)

    # Save results
    import json
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "7")
    with open(os.path.join(out_dir, "confounding_data.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Data saved to docs/7/confounding_data.json")


if __name__ == "__main__":
    main()
