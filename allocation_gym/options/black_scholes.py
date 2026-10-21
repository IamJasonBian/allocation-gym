"""
Pure-Python Black-Scholes pricing (no scipy dependency).

Ported from allocation-engine trading_system/backtests/options_simulation.py.
"""

import math
from typing import Optional


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun (7.1.26). Max err ~1.5e-7."""
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    p = 0.3275911
    # 7.1.26 approximates erf(z); Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
    sign = 1.0 if x >= 0 else -1.0
    z = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
    return 0.5 * (1.0 + sign * y)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_call_implied_vol(
    price: float, S: float, K: float, T: float, r: float,
    lo: float = 0.01, hi: float = 5.0, tol: float = 1e-6,
) -> Optional[float]:
    """
    Implied volatility of a European call by bisection.

    Returns None when the price is outside the no-arbitrage band
    [intrinsic, S] or T <= 0 — there is no sigma that reproduces it.
    """
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price <= intrinsic or price >= S:
        return None
    if bs_call_price(S, K, T, r, hi) < price:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, r, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
