"""Self-contained fallback Monte-Carlo pricer.

Minimal reference implementation conforming to the frozen pricer interfaces so
the HTTP API can boot and price standalone before the real ``pricer`` module is
merged. Implements Black-Scholes, plain MC, and importance-sampling MC for
European (and simple barrier) payoffs on a single GBM underlying.

Frozen interfaces:
    PriceResult, bs_price, price_plain_mc, price_is
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via :func:`math.erf`."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@dataclass
class PriceResult:
    """Result of a pricing call."""

    price: float
    std_error: float
    ess: float
    n_paths: int
    method: str


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """Black-Scholes price of a European call/put.

    Handles the degenerate ``T <= 0`` or ``sigma <= 0`` cases by returning the
    discounted intrinsic value.
    """
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"unsupported kind: {kind!r}")
    if T <= 0 or sigma <= 0:
        fwd = S * math.exp(r * T) if T > 0 else S
        intrinsic = max(fwd - K, 0.0) if kind == "call" else max(K - fwd, 0.0)
        return math.exp(-r * T) * intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _terminal_payoff(ST: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def _apply_barrier(payoff: np.ndarray, ST: np.ndarray, barrier: float | None,
                   kind: str) -> np.ndarray:
    """Apply a simple terminal knock-out using only the terminal price.

    This is a coarse approximation (no path monitoring) but keeps the fallback
    self-contained. An up-and-out knocks out when ST >= barrier (calls); a
    down-and-out knocks out when ST <= barrier (puts).
    """
    if barrier is None:
        return payoff
    if kind == "call":
        alive = ST < barrier
    else:
        alive = ST > barrier
    return np.where(alive, payoff, 0.0)


def price_plain_mc(S: float, K: float, T: float, r: float, sigma: float, kind: str,
                   n: int, seed: int, **kw) -> PriceResult:
    """Plain Monte-Carlo price under GBM with terminal sampling."""
    kind = kind.lower()
    barrier = kw.get("barrier")
    rng = np.random.default_rng(seed)
    n = int(n)
    z = rng.standard_normal(n)
    drift = (r - 0.5 * sigma * sigma) * T
    diff = sigma * math.sqrt(T)
    ST = S * np.exp(drift + diff * z)
    payoff = _terminal_payoff(ST, K, kind)
    payoff = _apply_barrier(payoff, ST, barrier, kind)
    disc = math.exp(-r * T)
    disc_payoff = disc * payoff
    price = float(np.mean(disc_payoff))
    std_error = float(np.std(disc_payoff, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return PriceResult(price=price, std_error=std_error, ess=float(n),
                       n_paths=n, method="plain_mc")


def price_is(S: float, K: float, T: float, r: float, sigma: float, kind: str,
             n: int, seed: int, method: str = "drift_tilt", **kw) -> PriceResult:
    """Importance-sampling MC price.

    The ``drift_tilt`` method shifts the sampling distribution of the terminal
    Brownian increment by a constant ``mu`` so that paths land near the payoff
    region (deep OTM options), reducing variance. The likelihood ratio
    re-weights each sample back to the physical measure. Other ``method``
    values fall back to plain MC.
    """
    kind = kind.lower()
    barrier = kw.get("barrier")
    n = int(n)
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)

    if method != "drift_tilt":
        logger.debug("IS method %r not specialised; falling back to plain MC", method)
        res = price_plain_mc(S, K, T, r, sigma, kind, n, seed, **kw)
        return PriceResult(price=res.price, std_error=res.std_error, ess=res.ess,
                           n_paths=res.n_paths, method=method)

    sqrtT = math.sqrt(T)
    drift = (r - 0.5 * sigma * sigma) * T
    diff = sigma * sqrtT

    # Choose the tilt so the drifted distribution is centred on the strike.
    # Solve S*exp(drift + diff*mu) = K  =>  mu = (log(K/S) - drift) / diff.
    if diff > 0:
        mu = (math.log(K / S) - drift) / diff
    else:
        mu = 0.0

    z = rng.standard_normal(n) + mu  # sample under shifted measure N(mu, 1)
    ST = S * np.exp(drift + diff * z)
    payoff = _terminal_payoff(ST, K, kind)
    payoff = _apply_barrier(payoff, ST, barrier, kind)

    # Likelihood ratio of N(0,1) vs N(mu,1): exp(-mu*z + mu^2/2).
    weights = np.exp(-mu * z + 0.5 * mu * mu)
    weighted = disc * payoff * weights

    price = float(np.mean(weighted))
    std_error = float(np.std(weighted, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    # Effective sample size from the importance weights.
    sw = float(np.sum(weights))
    sw2 = float(np.sum(weights * weights))
    ess = (sw * sw / sw2) if sw2 > 0 else float(n)

    return PriceResult(price=price, std_error=std_error, ess=ess,
                       n_paths=n, method="drift_tilt")
