"""Tests for Monte Carlo simulation module."""
import numpy as np
import pytest

from allocation_gym.simulation.engine import MonteCarloGBM
from allocation_gym.simulation.calibrate import calibrate_gbm, CalibrationResult
from tests.conftest import generate_synthetic_ohlc


# ── Engine tests ──

def test_simulate_shape():
    mc = MonteCarloGBM(mu=0.30, sigma=0.65, initial_price=50000)
    result = mc.simulate(n_paths=100, n_days=30, seed=42)
    assert result.paths.shape == (100, 31)
    assert result.time_days.shape == (31,)


def test_starts_at_initial_price():
    mc = MonteCarloGBM(mu=0.10, sigma=0.50, initial_price=97000)
    result = mc.simulate(n_paths=50, n_days=10, seed=0)
    np.testing.assert_allclose(result.paths[:, 0], 97000.0)


def test_all_positive_prices():
    mc = MonteCarloGBM(mu=-0.50, sigma=1.0, initial_price=100)
    result = mc.simulate(n_paths=500, n_days=365, seed=42)
    assert np.all(result.paths > 0)


def test_seed_reproducibility():
    mc = MonteCarloGBM(mu=0.20, sigma=0.60, initial_price=50000)
    r1 = mc.simulate(n_paths=100, n_days=30, seed=123)
    r2 = mc.simulate(n_paths=100, n_days=30, seed=123)
    np.testing.assert_array_equal(r1.paths, r2.paths)


def test_different_seeds_differ():
    mc = MonteCarloGBM(mu=0.20, sigma=0.60, initial_price=50000)
    r1 = mc.simulate(n_paths=100, n_days=30, seed=1)
    r2 = mc.simulate(n_paths=100, n_days=30, seed=2)
    assert not np.allclose(r1.paths, r2.paths)


def test_summary_stats_keys():
    mc = MonteCarloGBM(mu=0.30, sigma=0.65, initial_price=50000)
    result = mc.simulate(n_paths=200, n_days=30, seed=42)
    stats = MonteCarloGBM.summary_stats(result)
    for key in ["median_final", "mean_final", "prob_above_initial",
                "expected_return_pct", "P10", "P50", "P90", "percentile_paths"]:
        assert key in stats


def test_prob_bounded():
    mc = MonteCarloGBM(mu=0.30, sigma=0.65, initial_price=50000)
    result = mc.simulate(n_paths=1000, n_days=90, seed=42)
    stats = MonteCarloGBM.summary_stats(result)
    assert 0.0 <= stats["prob_above_initial"] <= 1.0


def test_invalid_initial_price():
    with pytest.raises(ValueError, match="positive"):
        MonteCarloGBM(mu=0.1, sigma=0.5, initial_price=-100)


def test_invalid_sigma():
    with pytest.raises(ValueError, match="non-negative"):
        MonteCarloGBM(mu=0.1, sigma=-0.5, initial_price=100)


# ── Calibration tests ──

def test_calibrate_from_synthetic():
    df = generate_synthetic_ohlc(days=100, base_price=50000, seed=42)
    cal = calibrate_gbm(
        opens=df["Open"].values,
        highs=df["High"].values,
        lows=df["Low"].values,
        closes=df["Close"].values,
        trading_days=365,
    )
    assert isinstance(cal, CalibrationResult)
    assert cal.sigma > 0
    assert cal.initial_price > 0
    assert cal.n_days_used == 100


def test_calibrate_too_few_bars():
    with pytest.raises(ValueError, match="at least 5"):
        calibrate_gbm(
            opens=np.array([100.0, 101.0, 102.0]),
            highs=np.array([102.0, 103.0, 104.0]),
            lows=np.array([99.0, 100.0, 101.0]),
            closes=np.array([101.0, 102.0, 103.0]),
        )


def test_calibrate_returns_regime():
    df = generate_synthetic_ohlc(days=60, base_price=50000, seed=7)
    cal = calibrate_gbm(
        opens=df["Open"].values,
        highs=df["High"].values,
        lows=df["Low"].values,
        closes=df["Close"].values,
    )
    assert cal.variance_result.regime in {
        "RANDOM_WALK", "STRONG_TREND", "NOISY_TREND", "CHOP", "MEAN_REVERT"
    }
