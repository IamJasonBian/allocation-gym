"""Tests for the optimal band selection module (OU-based, Section 11.3)."""

import math

import numpy as np
import pytest

from allocation_gym.optimal_bands import (
    calibrate_ou,
    compute_optimal_bands,
    compute_bands_from_prices,
    BandResult,
    _F_plus,
    _F_minus,
)


# ---------------------------------------------------------------------------
# OU calibration
# ---------------------------------------------------------------------------

class TestCalibrateOU:

    def test_basic_calibration(self):
        """Calibrate from a synthetic OU path and check parameter signs."""
        np.random.seed(42)
        kappa_true, theta_true, sigma_true = 2.0, 100.0, 5.0
        dt = 1 / 252
        n = 1000
        x = np.zeros(n)
        x[0] = theta_true
        for i in range(1, n):
            x[i] = (x[i - 1]
                     + kappa_true * (theta_true - x[i - 1]) * dt
                     + sigma_true * math.sqrt(dt) * np.random.randn())

        kappa, theta, sigma = calibrate_ou(x, dt=dt)
        assert kappa > 0, "kappa should be positive"
        assert sigma > 0, "sigma should be positive"
        # theta should be near theta_true (within ~10%)
        assert abs(theta - theta_true) / theta_true < 0.15

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            calibrate_ou(np.array([1, 2, 3]))

    def test_constant_series(self):
        """Constant series should give near-zero sigma and theta ~ constant."""
        x = np.full(100, 50.0)
        kappa, theta, sigma = calibrate_ou(x)
        assert abs(theta - 50.0) < 1.0


# ---------------------------------------------------------------------------
# Fundamental solutions F_+ and F_-
# ---------------------------------------------------------------------------

class TestFundamentalSolutions:

    def test_F_plus_positive(self):
        val = _F_plus(0.5, kappa=2.0, theta=0.0, sigma=0.5, rho=0.05)
        assert val > 0

    def test_F_minus_positive(self):
        val = _F_minus(-0.5, kappa=2.0, theta=0.0, sigma=0.5, rho=0.05)
        assert val > 0

    def test_F_plus_increasing(self):
        """F_+ should be increasing in eps (more eps -> higher value)."""
        kw = dict(kappa=2.0, theta=0.0, sigma=0.5, rho=0.05)
        f1 = _F_plus(0.1, **kw)
        f2 = _F_plus(0.5, **kw)
        assert f2 > f1

    def test_F_minus_decreasing(self):
        """F_- should be decreasing in eps."""
        kw = dict(kappa=2.0, theta=0.0, sigma=0.5, rho=0.05)
        f1 = _F_minus(-0.5, **kw)
        f2 = _F_minus(-0.1, **kw)
        assert f1 > f2


# ---------------------------------------------------------------------------
# Optimal bands
# ---------------------------------------------------------------------------

class TestComputeOptimalBands:

    def test_basic_bands(self):
        """Bands should satisfy entry < theta < exit for long position."""
        bands = compute_optimal_bands(
            kappa=2.0, theta=0.0, sigma=0.5, rho=0.05, c=0.01,
        )
        assert isinstance(bands, BandResult)
        assert bands.entry_long < bands.theta
        assert bands.exit_long > bands.theta + bands.c
        assert bands.exit_short < bands.theta

    def test_higher_kappa_tighter_bands(self):
        """Higher mean-reversion -> tighter bands (closer to theta)."""
        b1 = compute_optimal_bands(kappa=1.0, theta=0.0, sigma=0.5, rho=0.05, c=0.01)
        b2 = compute_optimal_bands(kappa=4.0, theta=0.0, sigma=0.5, rho=0.05, c=0.01)
        assert b2.band_width < b1.band_width

    def test_bands_with_different_rho(self):
        """Different discount rates should produce valid bands."""
        for rho in [0.04, 0.08, 0.16]:
            bands = compute_optimal_bands(
                kappa=2.0, theta=0.0, sigma=0.5, rho=rho, c=0.01
            )
            assert bands.exit_long > bands.theta + bands.c
            assert bands.entry_long < bands.theta
            assert bands.band_width > 0

    def test_symmetric_around_theta(self):
        """Entry and exit should be roughly (but not exactly) symmetric."""
        bands = compute_optimal_bands(kappa=2.0, theta=0.0, sigma=0.5, rho=0.05, c=0.01)
        # They won't be exactly symmetric due to the discount factor
        assert abs(bands.exit_long) > 0
        assert abs(bands.entry_long) > 0

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            compute_optimal_bands(kappa=-1.0, theta=0.0, sigma=0.5, rho=0.05, c=0.01)

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            compute_optimal_bands(kappa=2.0, theta=0.0, sigma=0.5, rho=0.0, c=0.01)

    def test_dca_params_positive(self):
        """Derived DCA parameters should be non-negative."""
        bands = compute_optimal_bands(kappa=2.0, theta=100.0, sigma=5.0, rho=0.05, c=0.5)
        assert bands.stop_offset_pct >= 0
        assert bands.buy_offset >= 0
        assert bands.band_width >= 0


# ---------------------------------------------------------------------------
# End-to-end: from prices to bands
# ---------------------------------------------------------------------------

class TestComputeBandsFromPrices:

    def test_synthetic_ou_prices(self):
        """Full pipeline: synthetic OU -> calibrate -> bands."""
        np.random.seed(123)
        dt = 1 / 252
        kappa, theta, sigma = 3.0, 50.0, 2.0
        n = 500
        x = np.zeros(n)
        x[0] = theta
        for i in range(1, n):
            x[i] = (x[i - 1]
                     + kappa * (theta - x[i - 1]) * dt
                     + sigma * math.sqrt(dt) * np.random.randn())

        bands = compute_bands_from_prices(x, rho=0.05, c=0.1, dt=dt)

        assert bands.entry_long < bands.theta
        assert bands.exit_long > bands.theta
        assert bands.kappa > 0
        assert bands.sigma > 0

    def test_real_like_prices(self):
        """Price series with drift — should still produce valid bands."""
        np.random.seed(99)
        prices = 100 * np.cumprod(1 + np.random.normal(0.0002, 0.01, 300))
        bands = compute_bands_from_prices(prices, rho=0.05, c=0.01)
        assert isinstance(bands, BandResult)
        assert bands.band_width > 0


# ---------------------------------------------------------------------------
# Momentum DCA strategy smoke test
# ---------------------------------------------------------------------------

class TestMomentumDcaStrategy:

    def test_strategy_runs(self):
        """Strategy should run on synthetic data without errors."""
        import backtrader as bt
        from tests.conftest import generate_synthetic_ohlc
        from allocation_gym.strategies.momentum_dca import MomentumDcaStrategy

        cerebro = bt.Cerebro()
        df = generate_synthetic_ohlc(days=200, seed=42)
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name="TEST")
        cerebro.addstrategy(MomentumDcaStrategy, ou_lookback=30, recalib_period=10)
        cerebro.broker.setcash(100_000)
        cerebro.broker.setcommission(commission=0.001)

        results = cerebro.run()
        assert len(results) == 1

    def test_strategy_with_multiple_symbols(self):
        """Strategy should handle multiple symbols."""
        import backtrader as bt
        from tests.conftest import generate_synthetic_ohlc
        from allocation_gym.strategies.momentum_dca import MomentumDcaStrategy

        cerebro = bt.Cerebro()
        for sym in ["SYM_A", "SYM_B"]:
            df = generate_synthetic_ohlc(days=200, seed=hash(sym) % 2**31)
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name=sym)

        cerebro.addstrategy(MomentumDcaStrategy, ou_lookback=30, recalib_period=10)
        cerebro.broker.setcash(200_000)
        results = cerebro.run()
        assert len(results) == 1
