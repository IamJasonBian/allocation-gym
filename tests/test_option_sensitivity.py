"""Greeks are checked against finite differences of bs_call_price."""

import pytest

from allocation_gym.options.black_scholes import bs_call_price
from allocation_gym.options.sensitivity import (
    breakeven_spot,
    bs_call_greeks,
    scenario_grid,
)

S, K, T, R, V = 180.0, 230.0, 30 / 365, 0.04, 0.93


def test_greeks_match_finite_differences():
    g = bs_call_greeks(S, K, T, R, V)
    h = 0.01
    p = lambda s=S, t=T, r=R, v=V: bs_call_price(s, K, t, r, v)
    assert g["price"] == pytest.approx(p())
    assert g["delta"] == pytest.approx((p(s=S + h) - p(s=S - h)) / (2 * h), rel=1e-4)
    assert g["gamma"] == pytest.approx((p(s=S + h) - 2 * p() + p(s=S - h)) / h ** 2, rel=1e-2)
    assert g["vega"] == pytest.approx((p(v=V + 1e-4) - p(v=V - 1e-4)) / 2e-4 / 100, rel=1e-4)
    assert g["theta"] == pytest.approx(p(t=T - 1 / 365) - p(), rel=2e-2)
    assert g["rho"] == pytest.approx((p(r=R + 1e-4) - p(r=R - 1e-4)) / 2e-4 / 100, rel=1e-4)


def test_expired_greeks():
    assert bs_call_greeks(240, K, 0, R, V)["delta"] == 1.0
    assert bs_call_greeks(200, K, 0, R, V)["price"] == 0.0


def test_scenario_grid_center_and_monotone():
    grid = scenario_grid(S, K, 30, R, V, [-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1])
    assert grid[1][1] == pytest.approx(bs_call_price(S, K, T, R, V))
    assert grid[0][1] < grid[1][1] < grid[2][1]      # higher spot -> higher call
    assert grid[1][0] < grid[1][1] < grid[1][2]      # higher IV -> higher call
    later = scenario_grid(S, K, 30, R, V, [0.0], [0.0], days_forward=7)
    assert later[0][0] < grid[1][1]                  # time decay


def test_breakeven():
    assert breakeven_spot(230, 35.78) == pytest.approx(265.78)


@pytest.mark.parametrize("x", [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
def test_norm_cdf_matches_erf(x):
    import math
    from allocation_gym.options.black_scholes import _norm_cdf
    assert _norm_cdf(x) == pytest.approx(0.5 * (1 + math.erf(x / math.sqrt(2))), abs=1e-6)
