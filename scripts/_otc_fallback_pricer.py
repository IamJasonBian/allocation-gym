#!/usr/bin/env python3
"""Bundled fallback reference pricer for the OTC IS audit script.

This is a self-contained, correct reference implementation of the
importance-sampling (IS) Monte-Carlo pricer that the audit script relies on.

The audit script prefers the real module
``allocation_gym.otc_is_pricing.pricer`` when it is available (after the
sibling unit merges), and falls back to this module so the audit can run
standalone today.

Models a single-asset GBM under the risk-neutral measure::

    S_T = S * exp((r - 0.5*sigma**2)*T + sigma*sqrt(T)*Z),   Z ~ N(0, 1)

Payoffs supported (``kind``):

* ``"call"``   — vanilla European call, max(S_T - K, 0)
* ``"put"``    — vanilla European put,  max(K - S_T, 0)
* ``"digital"``— cash-or-nothing digital call paying 1 if S_T > K

Importance sampling uses *exponential drift tilting*: we sample
``Z + theta`` instead of ``Z`` so that out-of-the-money (OTM) call paths land
near / beyond the strike, then re-weight with the likelihood ratio
``w = exp(-theta*Z - 0.5*theta**2)``. The effective sample size
``ess = (sum w)**2 / sum(w**2)`` measures how many "effective" draws remain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PriceResult",
    "bs_price",
    "price_plain_mc",
    "price_is",
    "drift_tilt_theta",
]


@dataclass
class PriceResult:
    """Result of a Monte-Carlo (or analytic) price estimate.

    Attributes:
        price: The estimated option price (already discounted to today).
        std_error: Monte-Carlo standard error of ``price``. Zero for analytic.
        ess: Effective sample size of the estimator. Equals ``n_paths`` for a
            plain (unweighted) estimator.
        n_paths: Number of simulated paths.
        method: Short label describing the method, e.g. ``"plain_mc"`` or
            ``"is_drift_tilt"``.
    """

    price: float
    std_error: float
    ess: float
    n_paths: int
    method: str


# ---------------------------------------------------------------------------
# Analytic Black-Scholes — the "expected" benchmark
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
) -> float:
    """Analytic Black-Scholes price for call / put / digital(call).

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiry in years (must be > 0).
        r: Continuously-compounded risk-free rate.
        sigma: Volatility (annualized, must be > 0).
        kind: One of ``"call"``, ``"put"``, ``"digital"`` (cash-or-nothing
            digital call paying 1).

    Returns:
        The analytic, discounted option price.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    kind = kind.lower()
    disc = math.exp(-r * T)
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if kind == "call":
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    if kind == "put":
        return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    if kind == "digital":
        # Cash-or-nothing digital call: pays 1 unit if S_T > K.
        return disc * _norm_cdf(d2)
    raise ValueError(f"unknown kind: {kind!r}")


# ---------------------------------------------------------------------------
# Payoff
# ---------------------------------------------------------------------------

def _payoff(ST: np.ndarray, K: float, kind: str) -> np.ndarray:
    """Vectorized undiscounted payoff for an array of terminal prices."""
    kind = kind.lower()
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    if kind == "digital":
        return (ST > K).astype(np.float64)
    raise ValueError(f"unknown kind: {kind!r}")


# ---------------------------------------------------------------------------
# Plain Monte-Carlo
# ---------------------------------------------------------------------------

def price_plain_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    n: int = 100_000,
    seed: int | None = None,
    **kw,
) -> PriceResult:
    """Plain (unweighted) risk-neutral Monte-Carlo price.

    Args:
        S, K, T, r, sigma, kind: Standard option parameters (see ``bs_price``).
        n: Number of simulated paths.
        seed: RNG seed for reproducibility.
        **kw: Ignored (accepted for signature parity with ``price_is``).

    Returns:
        A :class:`PriceResult` with ``method="plain_mc"`` and
        ``ess == n_paths``.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    ST = S * np.exp(drift + diffusion * z)

    disc = math.exp(-r * T)
    discounted = disc * _payoff(ST, K, kind)

    price = float(discounted.mean())
    # Standard error of the mean.
    std_error = float(discounted.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    return PriceResult(
        price=price,
        std_error=std_error,
        ess=float(n),
        n_paths=n,
        method="plain_mc",
    )


# ---------------------------------------------------------------------------
# Importance sampling — exponential drift tilting
# ---------------------------------------------------------------------------

def drift_tilt_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Optimal drift-tilt ``theta`` that centers the GBM exponent on the strike.

    Solving ``(r - 0.5*sigma**2)*T + sigma*sqrt(T)*theta = log(K/S)`` for
    ``theta`` shifts the sampling distribution of ``Z`` so that, on average,
    ``S_T`` lands exactly on the strike ``K``. Clamped at >= 0 so we only ever
    tilt "upward" toward an OTM call strike (a no-op when ``K <= forward``).

    Args:
        S, K, T, r, sigma: Standard option parameters.

    Returns:
        Non-negative drift tilt in units of standard normals.
    """
    vol_sqrt_t = sigma * math.sqrt(T)
    theta = (math.log(K / S) - (r - 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return max(theta, 0.0)


def price_is(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    n: int = 100_000,
    seed: int | None = None,
    method: str = "drift_tilt",
    **kw,
) -> PriceResult:
    """Importance-sampling MC price via exponential drift tilting.

    We sample ``Z + theta`` instead of ``Z``, pushing mass toward (and beyond)
    the strike so OTM-call / digital payoffs are hit far more often, then
    correct the bias with the likelihood ratio ``w = exp(-theta*Z - 0.5*theta**2)``.
    The estimator ``mean(w * discounted_payoff)`` is unbiased for *any* payoff;
    for a deep-OTM call or digital its variance is far lower than plain MC.

    The tilt is non-negative (it shifts mass *upward* toward a high strike), so
    the variance reduction targets out-of-the-money calls / digital calls. For a
    put the estimate remains unbiased, but the upward tilt does not help (and an
    in-the-money put may even gain variance); price puts with ``price_plain_mc``
    instead. This audit only exercises calls and digitals.

    Args:
        S, K, T, r, sigma, kind: Standard option parameters.
        n: Number of simulated paths.
        seed: RNG seed for reproducibility.
        method: IS method label. Only ``"drift_tilt"`` is implemented; other
            values raise ``ValueError``.
        **kw: Ignored.

    Returns:
        A :class:`PriceResult` with ``method="is_<method>"`` and the realized
        effective sample size ``ess = (sum w)**2 / sum(w**2)``.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if method != "drift_tilt":
        raise ValueError(f"unsupported IS method: {method!r}")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)

    theta = drift_tilt_theta(S, K, T, r, sigma)

    # Tilted terminal prices: Z is replaced by (Z + theta).
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T)
    ST = S * np.exp(drift + diffusion * (z + theta))

    # Likelihood ratio (Radon-Nikodym derivative) of the original measure
    # w.r.t. the tilted one, expressed in terms of the *base* draw z.
    log_lr = -theta * z - 0.5 * theta * theta
    w = np.exp(log_lr)

    disc = math.exp(-r * T)
    payoff = _payoff(ST, K, kind)
    weighted = disc * w * payoff

    price = float(weighted.mean())
    std_error = float(weighted.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    # Effective sample size of the (importance) weights.
    sum_w = float(w.sum())
    sum_w2 = float((w * w).sum())
    ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0 else 0.0

    return PriceResult(
        price=price,
        std_error=std_error,
        ess=ess,
        n_paths=n,
        method=f"is_{method}",
    )
