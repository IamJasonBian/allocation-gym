"""Unit tests for the IV solver and the option pricer backtest (no network)."""

import pytest

from allocation_gym.options.black_scholes import bs_call_implied_vol, bs_call_price
from allocation_gym.options.pricer_backtest import (
    occ_symbol,
    realized_vol,
    run_pricer_backtest,
    summarize,
    year_fraction,
)


@pytest.mark.parametrize("sigma", [0.2, 0.6, 1.2])
def test_implied_vol_round_trip(sigma):
    p = bs_call_price(170, 230, 31 / 365, 0.04, sigma)
    assert bs_call_implied_vol(p, 170, 230, 31 / 365, 0.04) == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_rejects_out_of_band():
    assert bs_call_implied_vol(0.0, 170, 230, 0.1, 0.04) is None
    assert bs_call_implied_vol(200.0, 170, 230, 0.1, 0.04) is None  # above spot
    assert bs_call_implied_vol(1.0, 170, 230, 0.0, 0.04) is None    # expired


def test_occ_symbol():
    assert occ_symbol("NBIS", "2026-11-20", 230) == "NBIS261120C00230000"


def test_year_fraction():
    assert year_fraction("2026-10-20", "2026-11-20") == pytest.approx(31 / 365)
    assert year_fraction("2026-12-01", "2026-11-20") == 0.0


def test_realized_vol_flat_and_short():
    assert realized_vol([100.0] * 25, 20) == 0.0
    assert realized_vol([100.0] * 5, 20) is None


def test_constant_iv_market_is_priced_exactly():
    """If the market prices at one fixed sigma, entry_iv and prev_iv are exact."""
    sigma, K, exp = 0.9, 230.0, "2026-11-20"
    days = [f"2026-10-{d:02d}" for d in range(1, 11)]
    spots = [200 - 3 * i for i in range(len(days))]
    under = dict(zip(days, spots))
    opt = {d: bs_call_price(s, K, year_fraction(d, exp), 0.04, sigma) for d, s in under.items()}

    rows = run_pricer_backtest(under, opt, K, exp, vol_window=3)
    stats = summarize(rows)
    assert rows[0]["prev_iv"] is None  # no prior day yet
    assert stats["entry_iv"]["mae"] == pytest.approx(0, abs=1e-3)
    assert stats["prev_iv"]["mae"] == pytest.approx(0, abs=1e-3)
    assert stats["prev_iv"]["n"] == len(days) - 1
