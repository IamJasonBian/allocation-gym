"""
Backtest simple Black-Scholes pricing models against a real option's daily closes.

Models (all price the call at each day's underlying close):
  * realized  — sigma = trailing N-day realized vol of the underlying
  * entry_iv  — sigma frozen at the IV implied on the first option bar
  * prev_iv   — sigma = IV implied by the previous day's option close
                (one-day-ahead: uses only information available yesterday)

Usage:
    python -m allocation_gym.options.pricer_backtest
    python -m allocation_gym.options.pricer_backtest \
        --symbol NBIS --strike 230 --expiry 2026-11-20 --start 2026-09-29
"""

import argparse
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from allocation_gym.options.black_scholes import (
    bs_call_implied_vol,
    bs_call_price,
)

MODELS = ("realized", "entry_iv", "prev_iv")


def year_fraction(day: str, expiry: str) -> float:
    """Calendar-time to expiry in years, close-to-close."""
    d0 = date.fromisoformat(day)
    d1 = date.fromisoformat(expiry)
    return max((d1 - d0).days, 0) / 365.0


def realized_vol(closes: List[float], window: int) -> Optional[float]:
    """Annualized close-to-close vol over the last `window` returns."""
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    rets = [math.log(b / a) for a, b in zip(tail, tail[1:])]
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var * 252)


def run_pricer_backtest(
    underlying: Dict[str, float],
    option: Dict[str, float],
    strike: float,
    expiry: str,
    r: float = 0.04,
    vol_window: int = 20,
    start: Optional[str] = None,
) -> List[dict]:
    """
    Price the call on every day that has both an underlying and option close.

    `underlying` and `option` map ISO date -> close. Underlying history before
    `start` is used only to seed realized vol.
    """
    days = sorted(d for d in option if d in underlying and (start is None or d >= start))
    u_days = sorted(underlying)
    rows: List[dict] = []
    entry_iv: Optional[float] = None
    prev_iv: Optional[float] = None

    for d in days:
        S = underlying[d]
        actual = option[d]
        T = year_fraction(d, expiry)
        hist = [underlying[x] for x in u_days if x <= d]
        rv = realized_vol(hist, vol_window)
        iv_today = bs_call_implied_vol(actual, S, strike, T, r)
        if entry_iv is None:
            entry_iv = iv_today

        row = {"date": d, "spot": S, "actual": actual, "iv": iv_today, "rv": rv}
        for name, sigma in (("realized", rv), ("entry_iv", entry_iv), ("prev_iv", prev_iv)):
            row[name] = bs_call_price(S, strike, T, r, sigma) if sigma else None
        rows.append(row)
        if iv_today is not None:
            prev_iv = iv_today
    return rows


def summarize(rows: List[dict]) -> Dict[str, dict]:
    """MAE ($), mean signed error ($), and mean absolute % error per model."""
    out = {}
    for m in MODELS:
        pairs = [(r[m], r["actual"]) for r in rows if r[m] is not None]
        if not pairs:
            continue
        errs = [p - a for p, a in pairs]
        out[m] = {
            "n": len(pairs),
            "mae": sum(abs(e) for e in errs) / len(errs),
            "bias": sum(errs) / len(errs),
            "mape": 100 * sum(abs(p - a) / a for p, a in pairs) / len(pairs),
        }
    return out


# ── Data fetch (Yahoo public chart endpoint) ─────────────────────────────


def fetch_daily_closes(symbol: str, range_: str = "6mo") -> Dict[str, float]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range={range_}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                res = json.load(resp)["chart"]["result"][0]
            break
        except (http.client.IncompleteRead, urllib.error.URLError):
            if attempt == 4:
                raise
            time.sleep(1 + attempt)
    out = {}
    for ts, c in zip(res.get("timestamp", []), res["indicators"]["quote"][0]["close"]):
        if c is not None:
            out[datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()] = c
    return out


def occ_symbol(symbol: str, expiry: str, strike: float, right: str = "C") -> str:
    return f"{symbol}{date.fromisoformat(expiry):%y%m%d}{right}{int(round(strike * 1000)):08d}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbol", default="NBIS")
    ap.add_argument("--strike", type=float, default=230.0)
    ap.add_argument("--expiry", default="2026-11-20")
    ap.add_argument("--start", default="2026-09-29")
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--vol-window", type=int, default=20)
    args = ap.parse_args(argv)

    contract = occ_symbol(args.symbol, args.expiry, args.strike)
    rows = run_pricer_backtest(
        fetch_daily_closes(args.symbol),
        fetch_daily_closes(contract),
        args.strike, args.expiry, args.rate, args.vol_window, args.start,
    )

    fmt = lambda v, w=7: f"{v:{w}.2f}" if v is not None else " " * (w - 1) + "-"
    print(f"{contract}  r={args.rate}  rv_window={args.vol_window}")
    print(f"{'date':10} {'spot':>7} {'actual':>7} {'iv%':>6} {'rv%':>6} "
          f"{'realized':>8} {'entry_iv':>8} {'prev_iv':>8}")
    for r in rows:
        iv = r["iv"] * 100 if r["iv"] else None
        rv = r["rv"] * 100 if r["rv"] else None
        print(f"{r['date']:10} {fmt(r['spot'])} {fmt(r['actual'])} {fmt(iv, 6)} {fmt(rv, 6)} "
              f"{fmt(r['realized'], 8)} {fmt(r['entry_iv'], 8)} {fmt(r['prev_iv'], 8)}")
    print()
    for m, s in summarize(rows).items():
        print(f"{m:9} n={s['n']:3}  MAE ${s['mae']:.2f}  bias ${s['bias']:+.2f}  MAPE {s['mape']:.1f}%")


if __name__ == "__main__":
    main()
