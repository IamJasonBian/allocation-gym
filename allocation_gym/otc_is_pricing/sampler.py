"""GBM terminal sampling under a tilted measure and importance-sampling helpers.

Importance sampling (IS) rewrites an expectation under the target density ``f``
as an expectation under a proposal density ``g``::

    E_f[h(X)] = E_g[h(X) * f(X) / g(X)]

We draw ``X`` from the proposal ``g`` and reweight every sample by the
likelihood ratio ``w = f(X) / g(X)``. For deep-OTM / barrier / digital crypto
payoffs the payoff region is rarely visited under the risk-neutral measure, so
the plain estimator has huge variance. Tilting the Brownian increment by a mean
shift ``theta`` (exponential / Esscher tilting) pushes paths toward the strike;
the Radon-Nikodym weight ``w`` un-tilts the estimate back to the risk-neutral
measure so the estimator stays unbiased.

Tilting math
------------
Under the risk-neutral measure ``f`` the terminal price of a GBM is::

    ST = S * exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * Z),  Z ~ N(0, 1)

Under the proposal ``g`` we shift the standard normal by ``theta`` so that the
driving variate is ``Z' = Z + theta ~ N(theta, 1)`` while still feeding ``Z'``
through the same GBM map. With ``phi`` the standard-normal pdf, the likelihood
ratio evaluated at the realised ``Z'`` is::

    w = f / g = phi(Z') / phi(Z' - theta)
      = exp(-0.5 * Z'**2 + 0.5 * (Z' - theta)**2)
      = exp(-theta * Z' + 0.5 * theta**2)

Writing it in terms of the pre-shift draw ``Z = Z' - theta``::

    w   = exp(-theta * Z - 0.5 * theta**2)
    log_lr = -theta * Z - 0.5 * theta**2

which is what :func:`gbm_terminal` returns. ``E_g[w] = 1`` by construction, so
the weights re-normalise correctly.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def gbm_terminal(
    S: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    rng: np.random.Generator,
    drift_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample GBM terminal prices under a (possibly) tilted measure.

    The driving standard normal ``Z`` is drawn, shifted by ``theta = drift_shift``
    to obtain ``Z' = Z + theta`` (the proposal variate), and pushed through the
    log-normal GBM terminal map. The per-path **log** likelihood ratio that
    un-tilts the sample back to the risk-neutral measure is returned alongside.

    Args:
        S: Current spot price (> 0).
        r: Risk-free rate (continuously compounded).
        sigma: Volatility (> 0).
        T: Time to maturity in years (> 0).
        n: Number of paths to draw (> 0).
        rng: A seeded ``numpy`` random generator.
        drift_shift: Mean shift ``theta`` applied to the driving normal. ``0.0``
            recovers plain risk-neutral sampling (all weights == 1).

    Returns:
        ``(ST, log_lr)`` where ``ST`` is the array of terminal prices and
        ``log_lr`` is the array of per-path log likelihood ratios. With
        ``drift_shift == 0`` every ``log_lr`` is exactly ``0`` (weight 1).
    """
    if S <= 0.0:
        raise ValueError(f"S must be positive, got {S}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    theta = float(drift_shift)
    z = rng.standard_normal(n)
    z_prime = z + theta  # proposal variate ~ N(theta, 1)

    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T)
    ST = S * np.exp(drift + diffusion * z_prime)

    # log w = -theta * Z - 0.5 * theta**2  (== 0 when theta == 0)
    log_lr = -theta * z - 0.5 * theta * theta
    return ST, log_lr


def ess(weights: np.ndarray) -> float:
    """Effective sample size of a set of importance weights.

    ``ESS = (sum w)**2 / sum(w**2)`` measures weight degeneracy: it equals ``n``
    when all weights are equal and collapses toward 1 when a single path
    dominates.

    Args:
        weights: Non-negative importance weights.

    Returns:
        The effective sample size in ``[0, n]``. Returns ``0.0`` if the weights
        sum of squares is zero (all weights zero).
    """
    w = np.asarray(weights, dtype=float)
    sum_sq = float(np.sum(w * w))
    if sum_sq <= 0.0:
        return 0.0
    s = float(np.sum(w))
    return (s * s) / sum_sq


def self_normalized(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Self-normalised importance-sampling estimate and its standard error.

    The estimator is the weighted mean ``mu = sum(w * v) / sum(w)``; it does not
    require the weights to sum to ``n``. The standard error is obtained via the
    delta method for a ratio estimator, which is equivalent to the weighted
    variance of the residuals divided by the effective sample size::

        se = sqrt( sum(w**2 * (v - mu)**2) ) / sum(w)

    This reduces to the weighted-variance / sqrt(ESS) form and to the ordinary
    ``std / sqrt(n)`` when all weights are equal.

    Args:
        values: Per-path payoff (or discounted payoff) realisations.
        weights: Per-path importance weights (need not sum to ``n``).

    Returns:
        ``(estimate, std_error)``. If the weights sum to zero, returns
        ``(0.0, 0.0)``.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    sum_w = float(np.sum(w))
    if sum_w == 0.0:
        return 0.0, 0.0

    mu = float(np.sum(w * v) / sum_w)
    resid = v - mu
    var_num = float(np.sum(w * w * resid * resid))
    se = float(np.sqrt(var_num)) / sum_w
    return mu, se
