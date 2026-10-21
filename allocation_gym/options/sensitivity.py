"""
Sensitivity model for a European call: Greeks plus scenario grids.

Greeks (per-unit conventions, matching how brokers display them):
  * delta  — $ change per $1 move in the underlying
  * gamma  — change in delta per $1 move in the underlying
  * vega   — $ change per 1 vol point (0.01 sigma)
  * theta  — $ change per calendar day of time decay
  * rho    — $ change per 1% (0.01) move in rates

Scenario grid: reprices the call across spot moves x IV shifts x days forward,
so you can read off e.g. "NBIS +5%, IV -10 pts, 3 days later".

Usage:
    python -m allocation_gym.options.sensitivity \
        --spot 180 --price 3.80 --ref-spot 171.45 --days 30
"""

import argparse
import math
from typing import Dict, List, Optional, Sequence

from allocation_gym.options.black_scholes import (
    _norm_cdf,
    bs_call_implied_vol,
    bs_call_price,
)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_call_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """Analytic Black-Scholes call Greeks. At/after expiry returns intrinsic-only values."""
    if T <= 0 or sigma <= 0:
        itm = 1.0 if S > K else 0.0
        return {"price": max(S - K, 0.0), "delta": itm, "gamma": 0.0,
                "vega": 0.0, "theta": 0.0, "rho": 0.0}
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    disc = math.exp(-r * T)
    theta_yr = -S * _norm_pdf(d1) * sigma / (2 * sq) - r * K * disc * _norm_cdf(d2)
    return {
        "price": S * _norm_cdf(d1) - K * disc * _norm_cdf(d2),
        "delta": _norm_cdf(d1),
        "gamma": _norm_pdf(d1) / (S * sigma * sq),
        "vega": S * _norm_pdf(d1) * sq / 100.0,
        "theta": theta_yr / 365.0,
        "rho": K * T * disc * _norm_cdf(d2) / 100.0,
    }


def scenario_grid(
    S: float, K: float, days: int, r: float, sigma: float,
    spot_moves: Sequence[float], iv_shifts: Sequence[float], days_forward: int = 0,
) -> List[List[float]]:
    """
    Call prices on a grid: rows = spot moves (fractions, e.g. 0.05),
    cols = IV shifts (absolute vol, e.g. -0.10), `days_forward` days later.
    """
    T = max(days - days_forward, 0) / 365.0
    return [
        [bs_call_price(S * (1 + m), K, T, r, max(sigma + s, 0.01)) for s in iv_shifts]
        for m in spot_moves
    ]


def breakeven_spot(K: float, premium: float) -> float:
    """Underlying price at expiry where the long call breaks even."""
    return K + premium


def main(argv=None):
    ap = argparse.ArgumentParser(description="Call option Greeks + scenario grid")
    ap.add_argument("--spot", type=float, default=180.0, help="underlying to center the grid on")
    ap.add_argument("--strike", type=float, default=230.0)
    ap.add_argument("--days", type=int, default=30, help="calendar days to expiry")
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--iv", type=float, help="vol (e.g. 0.93); else implied from --price/--ref-spot")
    ap.add_argument("--price", type=float, default=3.80, help="last option price, to imply IV")
    ap.add_argument("--ref-spot", type=float, default=171.45, help="underlying when --price printed")
    ap.add_argument("--ref-days", type=int, default=31, help="days to expiry when --price printed")
    ap.add_argument("--cost", type=float, default=35.78, help="entry premium per share")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--days-forward", type=int, nargs="+", default=[0, 7, 14])
    args = ap.parse_args(argv)

    sigma: Optional[float] = args.iv or bs_call_implied_vol(
        args.price, args.ref_spot, args.strike, args.ref_days / 365.0, args.rate)
    if sigma is None:
        raise SystemExit("could not imply vol from --price; pass --iv")
    mult = 100 * args.contracts
    T = args.days / 365.0
    g = bs_call_greeks(args.spot, args.strike, T, args.rate, sigma)

    print(f"K={args.strike:g}  spot={args.spot:g}  {args.days}d  IV={sigma*100:.1f}%  r={args.rate}")
    print(f"price ${g['price']:.2f}  (position ${g['price']*mult:,.0f}, "
          f"P&L ${(g['price']-args.cost)*mult:+,.0f} vs ${args.cost:.2f} cost)")
    print(f"delta {g['delta']:.3f}  gamma {g['gamma']:.4f}  vega ${g['vega']:.3f}/pt  "
          f"theta ${g['theta']:.3f}/day  rho ${g['rho']:.3f}/1%")
    print(f"position: ${g['delta']*mult:,.0f} per $1 NBIS, ${g['vega']*mult:,.0f} per vol pt, "
          f"${g['theta']*mult:,.0f}/day decay")
    print(f"breakeven at expiry ${breakeven_spot(args.strike, args.cost):.2f} "
          f"({(breakeven_spot(args.strike, args.cost)/args.spot-1)*100:+.1f}% from spot)")

    moves = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
    shifts = [-0.30, -0.20, -0.10, 0.0, 0.10]
    for fwd in args.days_forward:
        grid = scenario_grid(args.spot, args.strike, args.days, args.rate, sigma, moves, shifts, fwd)
        print(f"\n+{fwd}d  rows=spot move, cols=IV shift (pts)")
        print(f"{'':>14}" + "".join(f"{s*100:+7.0f}" for s in shifts))
        for m, row in zip(moves, grid):
            print(f"{m*100:+4.0f}% {args.spot*(1+m):7.2f}  " + "".join(f"{p:7.2f}" for p in row))


if __name__ == "__main__":
    main()
