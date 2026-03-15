"""
Optimal Band Selection for mean-reverting (OU) processes.

Implements Section 11.3 from "Pairs Trading and Statistical Arbitrage
Strategies" (Leung & Li).  Given Ornstein-Uhlenbeck parameters
(kappa, theta, sigma) and a discount rate rho with transaction cost c,
this module solves for:

  * epsilon_star  – optimal exit level for a long position  (Eq 11.2)
  * epsilon_star_entry – optimal entry level for a long position (Eq 11.3)
  * epsilon_star_short_exit – optimal exit for a short position (Eq 11.4)

The fundamental solutions F_+(eps) and F_-(eps) of
    (L - rho) F = 0
are evaluated via numerical quadrature, and the non-linear equations
are solved with scipy.optimize.brentq.

Public API
----------
    calibrate_ou(prices)  -> (kappa, theta, sigma)
    compute_optimal_bands(kappa, theta, sigma, rho, c) -> BandResult
    compute_bands_from_prices(prices, rho, c) -> BandResult
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import integrate, optimize


# ---------------------------------------------------------------------------
# OU calibration from price series
# ---------------------------------------------------------------------------

def calibrate_ou(
    prices: np.ndarray,
    dt: float = 1.0 / 252,
) -> Tuple[float, float, float]:
    """Estimate OU parameters from a price / spread series via OLS.

    Model: dX = kappa * (theta - X) dt + sigma dW

    We regress  X_{t+1} - X_t  on  X_t  using discrete observations:
        X_{t+1} = a + b * X_t + residual
    where a = kappa*theta*dt, b = 1 - kappa*dt.

    Returns (kappa, theta, sigma).
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 10:
        raise ValueError("Need at least 10 observations to calibrate OU")

    x = prices[:-1]
    y = prices[1:]

    # OLS: y = a + b*x
    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x * y)
    sx2 = np.sum(x ** 2)

    b = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
    a = (sy - b * sx) / n

    # Extract OU parameters
    kappa = -math.log(max(b, 1e-8)) / dt  # b = exp(-kappa*dt) ≈ 1 - kappa*dt
    kappa = max(kappa, 1e-6)

    theta = a / (1 - b) if abs(1 - b) > 1e-10 else np.mean(prices)

    # Residual variance -> sigma
    residuals = y - (a + b * x)
    var_resid = np.var(residuals, ddof=1)
    # Var(residuals) = sigma^2 * (1 - exp(-2*kappa*dt)) / (2*kappa)
    factor = (1 - math.exp(-2 * kappa * dt)) / (2 * kappa)
    sigma = math.sqrt(var_resid / factor) if factor > 0 else math.sqrt(var_resid / dt)

    return kappa, theta, sigma


# ---------------------------------------------------------------------------
# Fundamental solutions F_+ and F_-
# ---------------------------------------------------------------------------

def _alpha(kappa: float, rho: float) -> float:
    """Exponent alpha = kappa / rho used in the integral representation."""
    return kappa / rho


def _beta(kappa: float, sigma: float, theta: float, eps: float) -> float:
    """Coefficient sqrt(2*kappa/sigma^2) * (theta - eps)."""
    return math.sqrt(2 * kappa / (sigma ** 2)) * (theta - eps)


def _safe_exp_integrand(u: float, alpha: float, linear_coeff: float) -> float:
    """Compute u^(alpha-1) * exp(linear_coeff * u - u^2/2) in log-space.

    Args:
        u: Integration variable.
        alpha: kappa / rho.
        linear_coeff: The coefficient of u in the exponent.
    """
    if u < 1e-30:
        return 0.0
    log_val = (alpha - 1) * math.log(u) + linear_coeff * u - 0.5 * u * u
    if log_val < -700:
        return 0.0
    if log_val > 700:
        return 0.0  # integrand should decay; if not, we're outside valid range
    return math.exp(log_val)


def _find_integration_bounds(alpha: float, linear_coeff: float) -> float:
    """Find upper integration bound by locating the peak and adding margin.

    The peak of (alpha-1)*ln(u) + linear_coeff*u - u^2/2 is where:
        (alpha-1)/u + linear_coeff - u = 0  =>  u^2 - linear_coeff*u - (alpha-1) = 0
    """
    # Solve u^2 - linear_coeff*u - (alpha-1) = 0
    disc = linear_coeff ** 2 + 4 * max(alpha - 1, 0)
    if disc > 0:
        u_peak = max((linear_coeff + math.sqrt(disc)) / 2, 0.1)
    else:
        u_peak = max(math.sqrt(max(alpha - 1, 0.1)), 0.1)
    return u_peak + 10 * max(math.sqrt(max(alpha, 1)), 3.0)


def _F_plus(eps: float, kappa: float, theta: float, sigma: float, rho: float) -> float:
    """F_+(eps) = integral_0^inf u^(alpha-1) exp(-beta*u - u^2/2) du.

    Where beta = sqrt(2*kappa/sigma^2) * (theta - eps).
    Strictly positive, increasing, and convex in eps.
    """
    alpha = _alpha(kappa, rho)
    beta = _beta(kappa, sigma, theta, eps)
    # The linear coefficient in the exponent is -beta
    linear_coeff = -beta

    def integrand(u):
        return _safe_exp_integrand(u, alpha, linear_coeff)

    upper = _find_integration_bounds(alpha, linear_coeff)
    val, _ = integrate.quad(integrand, 0, upper, limit=200)
    return max(val, 1e-300)


def _F_minus(eps: float, kappa: float, theta: float, sigma: float, rho: float) -> float:
    """F_-(eps) = integral_0^inf u^(alpha-1) exp(+beta*u - u^2/2) du.

    Where beta = sqrt(2*kappa/sigma^2) * (theta - eps).
    Strictly positive, decreasing, and convex in eps.
    """
    alpha = _alpha(kappa, rho)
    beta = _beta(kappa, sigma, theta, eps)
    # The linear coefficient in the exponent is +beta
    linear_coeff = beta

    def integrand(u):
        return _safe_exp_integrand(u, alpha, linear_coeff)

    upper = _find_integration_bounds(alpha, linear_coeff)
    val, _ = integrate.quad(integrand, 0, upper, limit=200)
    return max(val, 1e-300)


def _F_plus_deriv(eps: float, kappa: float, theta: float, sigma: float, rho: float) -> float:
    """Numerical derivative of F_+ w.r.t. eps via central differences."""
    h = max(abs(eps) * 1e-5, 1e-7)
    return (_F_plus(eps + h, kappa, theta, sigma, rho) -
            _F_plus(eps - h, kappa, theta, sigma, rho)) / (2 * h)


def _F_minus_deriv(eps: float, kappa: float, theta: float, sigma: float, rho: float) -> float:
    """Numerical derivative of F_- w.r.t. eps via central differences."""
    h = max(abs(eps) * 1e-5, 1e-7)
    return (_F_minus(eps + h, kappa, theta, sigma, rho) -
            _F_minus(eps - h, kappa, theta, sigma, rho)) / (2 * h)


# ---------------------------------------------------------------------------
# Optimal band solvers
# ---------------------------------------------------------------------------

def _solve_exit_long(
    kappa: float, theta: float, sigma: float, rho: float, c: float,
) -> float:
    """Solve Eq 11.2: (eps* - c) F'_+(eps*) = F_+(eps*) for eps* > theta.

    This gives the optimal level at which to exit (close) a long position.
    """
    def objective(eps):
        fp = _F_plus(eps, kappa, theta, sigma, rho)
        fpd = _F_plus_deriv(eps, kappa, theta, sigma, rho)
        return (eps - c) * fpd - fp

    # Search above theta + c (the position must be profitable net of costs)
    lo = theta + c + 1e-4
    hi = theta + c + 10 * sigma / math.sqrt(kappa)

    # Extend hi if needed
    for _ in range(10):
        if objective(hi) * objective(lo) < 0:
            break
        hi *= 2

    try:
        return optimize.brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return theta + c + sigma / math.sqrt(2 * kappa)


def _solve_exit_short(
    kappa: float, theta: float, sigma: float, rho: float, c: float,
) -> float:
    """Solve Eq 11.4: (eps*_- + c) F'_-(eps*_-) = F_-(eps*_-) for eps*_- < theta.

    This gives the optimal level at which to exit (close) a short position.
    """
    def objective(eps):
        fm = _F_minus(eps, kappa, theta, sigma, rho)
        fmd = _F_minus_deriv(eps, kappa, theta, sigma, rho)
        return (eps + c) * fmd - fm

    hi = theta - c - 1e-4
    lo = theta - c - 10 * sigma / math.sqrt(kappa)

    for _ in range(10):
        if objective(lo) * objective(hi) < 0:
            break
        lo *= 2

    try:
        return optimize.brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return theta - c - sigma / math.sqrt(2 * kappa)


def _H_plus(
    eps: float, eps_star: float,
    kappa: float, theta: float, sigma: float, rho: float, c: float,
) -> float:
    """Value function H_+(eps) for exiting a long position.

    H_+(eps) = F_+(eps)/F_+(eps*) * (eps* - c)   if eps < eps*
             = (eps - c)                           if eps >= eps*
    """
    if eps >= eps_star:
        return eps - c
    fp_eps = _F_plus(eps, kappa, theta, sigma, rho)
    fp_star = _F_plus(eps_star, kappa, theta, sigma, rho)
    if fp_star < 1e-300:
        return eps_star - c  # degenerate case
    return (fp_eps / fp_star) * (eps_star - c)


def _H_plus_deriv(
    eps: float, eps_star: float,
    kappa: float, theta: float, sigma: float, rho: float, c: float,
) -> float:
    """Numerical derivative of H_+ w.r.t. eps."""
    h = max(abs(eps) * 1e-5, 1e-7)
    return (_H_plus(eps + h, eps_star, kappa, theta, sigma, rho, c) -
            _H_plus(eps - h, eps_star, kappa, theta, sigma, rho, c)) / (2 * h)


def _solve_entry_long(
    kappa: float, theta: float, sigma: float, rho: float, c: float,
    eps_star: float,
) -> float:
    """Solve Eq 11.3 for optimal entry into a long position.

    (H_+(eps_*) - eps_* - c) F'_-(eps_*) = (H'_+(eps_*) - 1) F_-(eps_*)

    The entry point eps_* < eps_star, typically below theta.
    """
    def objective(eps):
        h_val = _H_plus(eps, eps_star, kappa, theta, sigma, rho, c)
        h_der = _H_plus_deriv(eps, eps_star, kappa, theta, sigma, rho, c)
        fm = _F_minus(eps, kappa, theta, sigma, rho)
        fmd = _F_minus_deriv(eps, kappa, theta, sigma, rho)
        return (h_val - eps - c) * fmd - (h_der - 1) * fm

    hi = theta - 1e-4
    lo = theta - 10 * sigma / math.sqrt(kappa)

    for _ in range(10):
        if objective(lo) * objective(hi) < 0:
            break
        lo = lo - 5 * sigma / math.sqrt(kappa)

    try:
        return optimize.brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return theta - sigma / math.sqrt(2 * kappa)


# ---------------------------------------------------------------------------
# Public dataclass and main entry points
# ---------------------------------------------------------------------------

@dataclass
class BandResult:
    """Optimal entry/exit bands for an OU mean-reverting process."""

    # OU parameters
    kappa: float       # mean-reversion speed
    theta: float       # long-run mean
    sigma: float       # volatility

    # Optimization parameters
    rho: float         # discount rate
    c: float           # transaction cost (per unit)

    # Results — levels expressed in the same units as the OU process
    exit_long: float       # eps* — close long when spread >= this
    entry_long: float      # eps_* — enter long when spread <= this
    exit_short: float      # eps*_- — close short when spread <= this

    # Derived quantities for the DCA strategy
    stop_offset_pct: float   # (theta - entry_long) / theta as fraction
    buy_offset: float        # exit_long - entry_long  (absolute spread)
    band_width: float        # exit_long - entry_long


def compute_optimal_bands(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float = 0.05,
    c: float = 0.01,
) -> BandResult:
    """Compute optimal entry/exit bands given OU parameters.

    Args:
        kappa: Mean-reversion speed.
        theta: Long-run mean level.
        sigma: Volatility of the OU process.
        rho: Discount rate (urgency parameter, >0).
        c: Transaction cost per unit.

    Returns:
        BandResult with optimal levels and DCA-strategy parameters.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    exit_long = _solve_exit_long(kappa, theta, sigma, rho, c)
    entry_long = _solve_entry_long(kappa, theta, sigma, rho, c, exit_long)
    exit_short = _solve_exit_short(kappa, theta, sigma, rho, c)

    band_width = exit_long - entry_long
    stop_offset_pct = (theta - entry_long) / abs(theta) if abs(theta) > 1e-10 else 0.0

    return BandResult(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        c=c,
        exit_long=exit_long,
        entry_long=entry_long,
        exit_short=exit_short,
        stop_offset_pct=max(stop_offset_pct, 0.0),
        buy_offset=max(band_width, 0.0),
        band_width=band_width,
    )


def compute_bands_from_prices(
    prices: np.ndarray,
    rho: float = 0.05,
    c: float = 0.01,
    dt: float = 1.0 / 252,
) -> BandResult:
    """Calibrate OU from a price series and compute optimal bands.

    This is the main convenience function: pass in a price or spread
    series and get back the optimal entry/exit levels.

    Args:
        prices: 1-D array of prices or spread values.
        rho: Discount rate.
        c: Transaction cost per unit.
        dt: Time step between observations (default 1/252 for daily).

    Returns:
        BandResult with calibrated parameters and optimal bands.
    """
    kappa, theta, sigma = calibrate_ou(prices, dt=dt)
    return compute_optimal_bands(kappa, theta, sigma, rho, c)
