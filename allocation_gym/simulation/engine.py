"""
Geometric Brownian Motion Monte Carlo engine.

Model: dS = mu * S * dt + sigma * S * dW
Exact: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    paths: np.ndarray           # (n_paths, n_steps+1), includes S0
    time_days: np.ndarray       # (n_steps+1,)
    mu: float
    sigma: float
    initial_price: float
    n_paths: int
    n_days: int


class MonteCarloGBM:
    """Forward Monte Carlo simulation using Geometric Brownian Motion."""

    def __init__(self, mu: float, sigma: float, initial_price: float):
        if initial_price <= 0:
            raise ValueError(f"initial_price must be positive, got {initial_price}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self.initial_price = initial_price

    def simulate(
        self,
        n_paths: int = 1000,
        n_days: int = 90,
        seed: int | None = 42,
    ) -> SimulationResult:
        rng = np.random.default_rng(seed)

        dt_annual = 1.0 / 365.0
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt_annual
        diffusion = self.sigma * np.sqrt(dt_annual)

        Z = rng.standard_normal((n_paths, n_days))
        log_increments = drift + diffusion * Z

        log_paths = np.cumsum(log_increments, axis=1)
        log_paths = np.column_stack([np.zeros(n_paths), log_paths])

        paths = self.initial_price * np.exp(log_paths)
        time_days = np.arange(n_days + 1, dtype=float)

        return SimulationResult(
            paths=paths,
            time_days=time_days,
            mu=self.mu,
            sigma=self.sigma,
            initial_price=self.initial_price,
            n_paths=n_paths,
            n_days=n_days,
        )

    @staticmethod
    def summary_stats(
        result: SimulationResult,
        percentiles: tuple[int, ...] = (10, 25, 50, 75, 90),
    ) -> dict:
        final_prices = result.paths[:, -1]

        stats = {
            "initial_price": result.initial_price,
            "n_paths": result.n_paths,
            "n_days": result.n_days,
            "mu": result.mu,
            "sigma": result.sigma,
            "mean_final": float(np.mean(final_prices)),
            "median_final": float(np.median(final_prices)),
            "std_final": float(np.std(final_prices)),
            "min_final": float(np.min(final_prices)),
            "max_final": float(np.max(final_prices)),
            "prob_above_initial": float(np.mean(final_prices > result.initial_price)),
            "expected_return_pct": float(
                (np.median(final_prices) / result.initial_price - 1) * 100
            ),
        }

        for p in percentiles:
            stats[f"P{p}"] = float(np.percentile(final_prices, p))

        percentile_paths = {}
        for p in percentiles:
            percentile_paths[p] = np.percentile(result.paths, p, axis=0)
        stats["percentile_paths"] = percentile_paths

        return stats
