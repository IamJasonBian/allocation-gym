"""Plain-MC and importance-sampling pricers for OTC crypto option payoffs.

The module exposes a frozen :class:`PriceResult` plus three entry points:

* :func:`bs_price`     -- closed-form Black-Scholes reference values.
* :func:`price_plain_mc` -- naive risk-neutral Monte Carlo (baseline variance).
* :func:`price_is`     -- importance-sampling MC with several proposal schemes
  (drift tilt, barrier rare-event tilt, digital tilt, and a generic lognormal
  ``l2_proposal`` self-normalised estimator).

All payoffs are discounted by ``exp(-r * T)``. Reported ``std_error`` is the
standard error of the discounted (weighted) payoff; ``ess`` is the effective
sample size of the importance weights (``ess == n_paths`` for plain MC).

Only the terminal price ``S_T`` is simulated, so path-dependent products are
priced on a *terminal-value* definition (documented per method).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from allocation_gym.otc_is_pricing.sampler import ess, gbm_terminal, self_normalized

logger = logging.getLogger(__name__)

_VALID_KINDS = ("call", "put", "digital")
_VALID_METHODS = ("drift_tilt", "barrier", "digital", "l2_proposal")


@dataclass
class PriceResult:
    """Outcome of a pricing run.

    Attributes:
        price: Discounted option price (present value).
        std_error: Standard error of the Monte Carlo / IS estimate.
        ess: Effective sample size of the importance weights (== ``n_paths``
            for plain MC).
        n_paths: Number of simulated paths.
        method: Identifier of the estimator used (e.g. ``"plain_mc"``,
            ``"drift_tilt"``).
    """

    price: float
    std_error: float
    ess: float
    n_paths: int
    method: str


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via ``math.erf`` (scalar)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """Analytic Black-Scholes price (the 'expected' reference value).

    Args:
        S: Spot price (> 0).
        K: Strike (> 0).
        T: Time to maturity in years (> 0).
        r: Risk-free rate.
        sigma: Volatility (> 0).
        kind: One of ``"call"``, ``"put"`` or ``"digital"``. ``"digital"`` is a
            cash-or-nothing call paying 1 unit when ``S_T > K``.

    Returns:
        The discounted Black-Scholes value.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    if S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        raise ValueError("S, K, T, sigma must all be positive")

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-r * T)

    if kind == "call":
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    if kind == "put":
        return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    # digital cash-or-nothing call: pays 1 if S_T > K
    return disc * _norm_cdf(d2)


def _payoff(ST: np.ndarray, K: float, kind: str) -> np.ndarray:
    """Vectorised undiscounted payoff for a vanilla ``kind``."""
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    # digital cash-or-nothing call
    return (ST > K).astype(float)


def price_plain_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str,
    n: int,
    seed: int,
    **kw: object,
) -> PriceResult:
    """Naive risk-neutral Monte Carlo price (no variance reduction).

    Args:
        S, K, T, r, sigma: Black-Scholes parameters.
        kind: ``"call"``, ``"put"`` or ``"digital"``.
        n: Number of paths.
        seed: Seed for ``np.random.default_rng``.
        **kw: Ignored (accepted for a uniform call signature).

    Returns:
        A :class:`PriceResult` with ``ess == n`` and the standard error of the
        discounted payoff.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    rng = np.random.default_rng(seed)
    ST, _log_lr = gbm_terminal(S, r, sigma, T, n, rng, drift_shift=0.0)

    disc = math.exp(-r * T)
    disc_payoff = disc * _payoff(ST, K, kind)

    price = float(np.mean(disc_payoff))
    std_error = float(np.std(disc_payoff, ddof=1) / math.sqrt(n))
    return PriceResult(
        price=price,
        std_error=std_error,
        ess=float(n),
        n_paths=n,
        method="plain_mc",
    )


def _drift_tilt_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Tilt that lands the tilted mean of ``Z'`` near ``log(K/S)``.

    We want the proposal mean of the driving normal to push the *median* terminal
    log-price toward the strike. Solving ``log(S) + (r - 0.5 sigma**2) T +
    sigma sqrt(T) * theta = log(K)`` for ``theta`` gives::

        theta = (log(K / S) - (r - 0.5 sigma**2) T) / (sigma sqrt(T))

    Tilting toward an OTM call only helps when ``theta > 0`` (we must push *up*),
    so the value is clamped to ``>= 0``. For an already-ITM strike ``theta``
    collapses to 0 and we fall back to plain sampling.
    """
    sqrt_t = math.sqrt(T)
    theta = (math.log(K / S) - (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return max(theta, 0.0)


def price_is(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str,
    n: int,
    seed: int,
    method: str,
    **kw: object,
) -> PriceResult:
    """Importance-sampling Monte Carlo price.

    Args:
        S, K, T, r, sigma: Black-Scholes parameters.
        kind: ``"call"``, ``"put"`` or ``"digital"`` (the vanilla payoff to
            price; ``barrier`` overlays a knock-in condition).
        n: Number of paths.
        seed: Seed for ``np.random.default_rng``.
        method: One of:

            * ``"drift_tilt"`` -- exponential drift tilt toward the strike for a
              deep-OTM ``kind`` payoff. Unbiased (un-self-normalised) estimator.
            * ``"barrier"``    -- terminal-value knock-in tilted toward the
              barrier. ``kw["barrier"]`` (float) and ``kw["barrier_type"]``
              (default ``"up-in"``) select the condition; payoff is a call.
            * ``"digital"``    -- drift tilt for a cash-or-nothing digital
              (forces ``kind == "digital"``).
            * ``"l2_proposal"`` -- generic self-normalised IS under a lognormal
              proposal centred near ``K`` (or a user sampler in
              ``kw["proposal_sampler"]``).

        **kw: Method-specific keyword arguments (see above).

    Returns:
        A :class:`PriceResult`.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if method == "l2_proposal":
        return _price_l2_proposal(S, K, T, r, sigma, kind, n, seed, **kw)
    if method == "barrier":
        return _price_barrier(S, K, T, r, sigma, n, seed, **kw)
    if method == "digital":
        kind = "digital"

    # drift_tilt / digital: exponential tilt toward the strike, unbiased IS.
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    rng = np.random.default_rng(seed)
    theta = _drift_tilt_theta(S, K, T, r, sigma)
    ST, log_lr = gbm_terminal(S, r, sigma, T, n, rng, drift_shift=theta)

    disc = math.exp(-r * T)
    weights = np.exp(log_lr)
    weighted_disc_payoff = disc * _payoff(ST, K, kind) * weights

    # Unbiased IS: the estimator is the plain mean of payoff * weight.
    price = float(np.mean(weighted_disc_payoff))
    std_error = float(np.std(weighted_disc_payoff, ddof=1) / math.sqrt(n))
    return PriceResult(
        price=price,
        std_error=std_error,
        ess=ess(weights),
        n_paths=n,
        method=method,
    )


def _price_barrier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n: int,
    seed: int,
    barrier: float | None = None,
    barrier_type: str = "up-in",
    **kw: object,
) -> PriceResult:
    """Terminal-value knock-in barrier call priced via rare-event drift tilt.

    Because only the terminal price ``S_T`` is simulated, the barrier is applied
    to ``S_T`` (a terminal-value knock-in), not continuously monitored. For an
    ``"up-in"`` call the payoff is ``max(S_T - K, 0)`` when ``S_T >= barrier``.
    The drift is tilted so the proposal median lands at the barrier (a rarer,
    higher level than the strike), which is exactly the deep-tail region the
    payoff needs.
    """
    if barrier is None:
        raise ValueError("barrier method requires kw['barrier']")
    if barrier <= 0.0:
        raise ValueError(f"barrier must be positive, got {barrier}")

    barrier_type = str(barrier_type)
    if barrier_type not in ("up-in", "down-in"):
        raise ValueError(
            f"barrier_type must be 'up-in' or 'down-in', got {barrier_type!r}"
        )

    rng = np.random.default_rng(seed)
    # Tilt toward the barrier level (the rare event driving the payoff).
    theta = _drift_tilt_theta(S, barrier, T, r, sigma)
    if barrier_type == "down-in":
        # Push paths *down* toward a low barrier.
        sqrt_t = math.sqrt(T)
        theta = (math.log(barrier / S) - (r - 0.5 * sigma * sigma) * T) / (
            sigma * sqrt_t
        )
        theta = min(theta, 0.0)

    ST, log_lr = gbm_terminal(S, r, sigma, T, n, rng, drift_shift=theta)

    if barrier_type == "up-in":
        knocked = ST >= barrier
    else:
        knocked = ST <= barrier

    disc = math.exp(-r * T)
    payoff = np.where(knocked, np.maximum(ST - K, 0.0), 0.0)
    weights = np.exp(log_lr)
    weighted_disc_payoff = disc * payoff * weights

    price = float(np.mean(weighted_disc_payoff))
    std_error = float(np.std(weighted_disc_payoff, ddof=1) / math.sqrt(n))
    return PriceResult(
        price=price,
        std_error=std_error,
        ess=ess(weights),
        n_paths=n,
        method="barrier",
    )


def _price_l2_proposal(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str,
    n: int,
    seed: int,
    proposal_sampler: object = None,
    **kw: object,
) -> PriceResult:
    """Generic self-normalised IS under a lognormal proposal centred near ``K``.

    The proposal draws ``S_T`` from a lognormal whose median sits at ``K`` with
    the same log-volatility ``sigma sqrt(T)`` as the target. The importance
    weight is the ratio of the target (risk-neutral) log-normal density to the
    proposal density evaluated at each sampled ``S_T``, and the price is the
    self-normalised weighted mean of the discounted payoff.

    If ``proposal_sampler`` is provided it must be a callable
    ``(n, rng) -> (ST, log_proposal_pdf)`` returning sampled terminal prices and
    the log proposal density at those points; the target density is supplied
    internally.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    rng = np.random.default_rng(seed)
    sqrt_t = math.sqrt(T)
    vol = sigma * sqrt_t  # log-vol of the terminal price

    # Target (risk-neutral) terminal log-price ~ N(mu_f, vol**2).
    mu_f = math.log(S) + (r - 0.5 * sigma * sigma) * T

    if proposal_sampler is not None:
        ST, log_g = proposal_sampler(n, rng)  # type: ignore[operator]
        ST = np.asarray(ST, dtype=float)
        log_g = np.asarray(log_g, dtype=float)
        x = np.log(ST)
        # log target density of log-price + Jacobian d(log ST)/d ST = 1/ST.
        log_f = (
            -0.5 * math.log(2.0 * math.pi)
            - math.log(vol)
            - 0.5 * ((x - mu_f) / vol) ** 2
            - np.log(ST)
        )
    else:
        # Lognormal proposal: log-price ~ N(mu_g, vol**2) with median at K.
        mu_g = math.log(K)
        x = rng.normal(loc=mu_g, scale=vol, size=n)
        ST = np.exp(x)
        # Densities of the *log-price* (the Jacobian cancels in the ratio).
        log_f = -0.5 * ((x - mu_f) / vol) ** 2
        log_g = -0.5 * ((x - mu_g) / vol) ** 2

    log_w = log_f - log_g
    # Stabilise against overflow before exponentiating.
    log_w -= float(np.max(log_w))
    weights = np.exp(log_w)

    disc = math.exp(-r * T)
    disc_payoff = disc * _payoff(ST, K, kind)

    price, std_error = self_normalized(disc_payoff, weights)
    return PriceResult(
        price=price,
        std_error=std_error,
        ess=ess(weights),
        n_paths=n,
        method="l2_proposal",
    )
