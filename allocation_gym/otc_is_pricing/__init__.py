"""Importance-sampling Monte Carlo pricing for illiquid alt-coin OTC derivatives.

This package prices deep-OTM / barrier / digital crypto option payoffs where
plain Monte Carlo suffers from severe variance because few simulated paths reach
the payoff region. We push paths toward the strike via exponential tilting of the
GBM drift (a change of measure) and correct the resulting bias with the
Radon-Nikodym / likelihood-ratio weights.

Submodules:
    sampler -- GBM terminal sampling under a shifted (tilted) measure, the
        likelihood-ratio weights, effective sample size (ESS), and the
        self-normalised IS estimator.
    pricer  -- analytic Black-Scholes references plus plain-MC and IS pricers
        exposing a frozen ``PriceResult`` interface.

The package is pure math: it takes plain ``float`` inputs and a numpy ``rng`` and
has no dependency on data feeds or HTTP layers. ``numpy`` is the only third-party
dependency.
"""
from __future__ import annotations

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

__all__ = [
    "PriceResult",
    "bs_price",
    "price_is",
    "price_plain_mc",
    "ess",
    "gbm_terminal",
    "self_normalized",
]
