"""
Options hedging simulation tools ported from allocation-engine.

Core public API
---------------
* :func:`~allocation_gym.options.black_scholes.bs_put_price` —
  Black-Scholes European put pricing (no scipy dependency).
* :func:`~allocation_gym.options.black_scholes.bs_call_price` —
  Black-Scholes European call pricing.
* :class:`~allocation_gym.options.metrics.OptionsBacktestMetrics` —
  Aggregated metrics for an options hedging backtest.
* :func:`~allocation_gym.options.metrics.compute_options_metrics` —
  Compute hedging metrics from a simulation result.
"""

from allocation_gym.options.black_scholes import bs_put_price, bs_call_price

__all__ = [
    "bs_put_price",
    "bs_call_price",
]

