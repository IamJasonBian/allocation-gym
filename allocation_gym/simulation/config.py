"""Configuration for forward Monte Carlo simulation."""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    symbol: str = "BTC/USD"

    # Calibration
    calibration_days: int = 90
    trading_days: int = 365          # 365 for crypto, 252 for equities

    # Simulation
    n_paths: int = 1000
    n_days_forward: int = 90
    seed: int | None = 42

    # Overrides (None = auto-calibrate from data)
    mu_override: float | None = None
    sigma_override: float | None = None
    initial_price: float | None = None

    # Visualization
    percentiles: tuple[int, ...] = (10, 25, 50, 75, 90)
    no_plot: bool = False
