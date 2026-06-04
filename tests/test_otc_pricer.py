"""Tests for the importance-sampling OTC option pricer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from allocation_gym.otc_is_pricing.pricer import (
    PriceResult,
    bs_price,
    price_is,
    price_plain_mc,
)
from allocation_gym.otc_is_pricing.sampler import (
    ess,
    gbm_terminal,
    self_normalized,
)

# Deep-OTM crypto-like contract used across several tests.
OTM = dict(S=100.0, K=160.0, T=0.25, r=0.0, sigma=0.8)
ATM = dict(S=100.0, K=100.0, T=0.25, r=0.0, sigma=0.8)


# ── sampler ──


class TestSampler:
    def test_plain_sampling_has_unit_weights(self):
        rng = np.random.default_rng(0)
        ST, log_lr = gbm_terminal(100.0, 0.0, 0.8, 0.25, 1000, rng, drift_shift=0.0)
        assert ST.shape == (1000,)
        np.testing.assert_allclose(log_lr, 0.0)
        assert np.all(ST > 0)

    def test_log_lr_formula(self):
        # With a known rng we can reconstruct Z and check log_lr = -theta Z - 0.5 theta^2.
        theta = 0.7
        rng = np.random.default_rng(42)
        z = np.random.default_rng(42).standard_normal(500)
        _ST, log_lr = gbm_terminal(100.0, 0.0, 0.8, 0.25, 500, rng, drift_shift=theta)
        expected = -theta * z - 0.5 * theta * theta
        np.testing.assert_allclose(log_lr, expected, rtol=1e-12)

    def test_weights_unbias_the_drift(self):
        # E_g[w] == 1: weighted mean of any moment matches risk-neutral sampling.
        rng = np.random.default_rng(7)
        ST, log_lr = gbm_terminal(100.0, 0.05, 0.5, 1.0, 200_000, rng, drift_shift=1.2)
        w = np.exp(log_lr)
        # Risk-neutral E[ST] = S e^{rT}.
        est = np.sum(w * ST) / np.sum(w)
        assert est == pytest.approx(100.0 * math.exp(0.05 * 1.0), rel=0.02)

    def test_ess_equal_weights(self):
        w = np.ones(100)
        assert ess(w) == pytest.approx(100.0)

    def test_ess_degenerate(self):
        w = np.zeros(100)
        w[0] = 1.0
        assert ess(w) == pytest.approx(1.0)

    def test_ess_zero_weights(self):
        assert ess(np.zeros(10)) == 0.0

    def test_self_normalized_equal_weights(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        est, se = self_normalized(v, w)
        assert est == pytest.approx(2.5)
        # equal-weight SE reduces to population-style weighted std / sqrt(n).
        assert se > 0.0

    def test_sampler_rejects_bad_inputs(self):
        rng = np.random.default_rng(0)
        for bad in (dict(S=-1.0), dict(sigma=0.0), dict(T=-0.1), dict(n=0)):
            args = dict(S=100.0, r=0.0, sigma=0.8, T=0.25, n=10)
            args.update(bad)
            with pytest.raises(ValueError):
                gbm_terminal(args["S"], args["r"], args["sigma"], args["T"], args["n"], rng)


# ── analytic reference ──


class TestBlackScholes:
    def test_call_put_parity(self):
        c = bs_price(kind="call", **ATM)
        p = bs_price(kind="put", **ATM)
        # C - P = S - K e^{-rT}
        lhs = c - p
        rhs = ATM["S"] - ATM["K"] * math.exp(-ATM["r"] * ATM["T"])
        assert lhs == pytest.approx(rhs, abs=1e-10)

    def test_digital_in_unit_interval(self):
        d = bs_price(kind="digital", **OTM)
        assert 0.0 < d < 1.0

    def test_known_atm_call_positive(self):
        assert bs_price(kind="call", **ATM) > 0.0

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            bs_price(kind="straddle", **ATM)


# ── plain MC ──


class TestPlainMC:
    def test_ess_equals_n(self):
        res = price_plain_mc(kind="call", n=20_000, seed=1, **ATM)
        assert res.ess == pytest.approx(res.n_paths)
        assert res.ess == pytest.approx(20_000.0)

    def test_atm_close_to_bs(self):
        res = price_plain_mc(kind="call", n=200_000, seed=3, **ATM)
        ref = bs_price(kind="call", **ATM)
        assert abs(res.price - ref) < 3.0 * res.std_error

    def test_result_type(self):
        res = price_plain_mc(kind="call", n=1000, seed=0, **ATM)
        assert isinstance(res, PriceResult)
        assert res.method == "plain_mc"


# ── importance sampling ──


class TestImportanceSampling:
    def test_atm_call_within_3se(self):
        res = price_is(kind="call", n=50_000, seed=11, method="drift_tilt", **ATM)
        ref = bs_price(kind="call", **ATM)
        assert abs(res.price - ref) < 3.0 * res.std_error

    def test_deep_otm_call_within_3se(self):
        res = price_is(kind="call", n=50_000, seed=12, method="drift_tilt", **OTM)
        ref = bs_price(kind="call", **OTM)
        assert abs(res.price - ref) < 3.0 * res.std_error

    def test_variance_reduction_deep_otm(self):
        is_res = price_is(kind="call", n=50_000, seed=21, method="drift_tilt", **OTM)
        mc_res = price_plain_mc(kind="call", n=50_000, seed=21, **OTM)
        assert is_res.std_error < mc_res.std_error

    def test_ess_in_bounds(self):
        res = price_is(kind="call", n=50_000, seed=5, method="drift_tilt", **OTM)
        assert 0.0 < res.ess <= res.n_paths

    def test_digital_within_3se(self):
        res = price_is(kind="digital", n=50_000, seed=31, method="digital", **OTM)
        ref = bs_price(kind="digital", **OTM)
        assert res.price > 0.0
        assert math.isfinite(res.price)
        assert abs(res.price - ref) < 3.0 * res.std_error

    def test_digital_variance_reduction(self):
        is_res = price_is(kind="digital", n=50_000, seed=41, method="digital", **OTM)
        mc_res = price_plain_mc(kind="digital", n=50_000, seed=41, **OTM)
        assert is_res.std_error < mc_res.std_error

    def test_barrier_finite_positive(self):
        res = price_is(
            kind="call",
            n=50_000,
            seed=51,
            method="barrier",
            barrier=200.0,
            barrier_type="up-in",
            **OTM,
        )
        assert math.isfinite(res.price)
        assert res.price > 0.0
        assert 0.0 < res.ess <= res.n_paths

    def test_barrier_requires_barrier_kw(self):
        with pytest.raises(ValueError):
            price_is(kind="call", n=1000, seed=0, method="barrier", **OTM)

    def test_l2_proposal_within_3se(self):
        res = price_is(kind="call", n=50_000, seed=61, method="l2_proposal", **OTM)
        ref = bs_price(kind="call", **OTM)
        assert math.isfinite(res.price)
        assert res.price > 0.0
        assert 0.0 < res.ess <= res.n_paths
        assert abs(res.price - ref) < 3.0 * res.std_error

    def test_l2_proposal_custom_sampler(self):
        # A user-supplied lognormal proposal centred at K.
        S, K, T, r, sigma = OTM["S"], OTM["K"], OTM["T"], OTM["r"], OTM["sigma"]
        vol = sigma * math.sqrt(T)
        mu_g = math.log(K)

        def sampler(n, rng):
            x = rng.normal(mu_g, vol, size=n)
            ST = np.exp(x)
            log_g = (
                -0.5 * math.log(2 * math.pi)
                - math.log(vol)
                - 0.5 * ((x - mu_g) / vol) ** 2
                - x  # Jacobian d log ST / d ST
            )
            return ST, log_g

        res = price_is(
            kind="call",
            n=50_000,
            seed=71,
            method="l2_proposal",
            proposal_sampler=sampler,
            **OTM,
        )
        ref = bs_price(kind="call", **OTM)
        assert abs(res.price - ref) < 3.0 * res.std_error

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            price_is(kind="call", n=10, seed=0, method="nope", **OTM)
