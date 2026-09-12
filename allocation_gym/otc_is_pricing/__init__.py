"""Importance-sampling pricing of illiquid alt-coin OTC derivatives.

Self-contained system for pricing illiquid alt-coin OTC derivatives via
importance-sampling Monte Carlo, resilient to upstream datafeed drops, with a
stdlib HTTP API hooked to real Binance L2 depth and a deterministic mock
fallback.

Import submodules directly:
    feeds   -- order-book L1/L2 snapshots + datafeed-drop-resilient index price.
    sampler -- tilted GBM sampling, likelihood-ratio weights, ESS, self-normalised IS.
    pricer  -- analytic Black-Scholes reference plus plain-MC and IS pricers.
    api     -- dependency-free stdlib HTTP pricing service.
"""
