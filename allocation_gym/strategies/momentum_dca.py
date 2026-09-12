"""
Momentum DCA Strategy with Optimal Band Selection.

Adapts the paired sell-stop / buy-limit DCA approach from allocation-engine's
MomentumDcaLongStrategy, replacing fixed stop/buy offsets with OU-calibrated
optimal bands from Section 11.3 (Leung & Li).

The strategy:
  1. Calibrates an OU process on the trailing price window
  2. Solves for optimal entry/exit bands (F_+, F_- fundamental solutions)
  3. Uses those bands to set stop-limit sell and paired limit-buy levels
  4. Maintains a coverage threshold on the position via Backtrader

Works as a Backtrader strategy so it integrates with the existing
allocation-gym backtesting infrastructure.
"""

import math

import backtrader as bt
import numpy as np

from allocation_gym.indicators.variance import VarianceIndicator
from allocation_gym.optimal_bands import calibrate_ou, compute_optimal_bands, BandResult


class MomentumDcaStrategy(bt.Strategy):
    """
    Momentum DCA with OU-optimal band selection.

    Holds a long position and places protective sell-stop orders at the
    optimal exit level, paired with buy-limit orders at the optimal entry
    level. Band levels are recalibrated every ``recalib_period`` bars.
    """

    params = (
        # OU calibration
        ("ou_lookback", 60),       # bars used for OU calibration
        ("recalib_period", 21),    # recalibrate bands every N bars
        ("rho", 0.05),            # discount rate (urgency)
        ("transaction_cost", 0.01),  # per-unit transaction cost for band calc

        # DCA coverage
        ("coverage_threshold", 0.20),  # minimum fraction of position covered
        ("lot_fraction", 0.10),        # fraction of position per paired order
        ("max_pairs", 5),              # max simultaneous pending pairs

        # Entry filter (from original momentum strategy)
        ("sma_period", 50),
        ("min_variance_ratio", 1.1),
        ("min_efficiency", 0.4),
        ("variance_lookback", 14),
        ("vr_k", 3),
        ("trading_days", 252),

        # Fallback fixed offsets (used when OU calibration fails)
        ("fallback_stop_pct", 0.015),
        ("fallback_buy_offset_pct", 0.005),
    )

    def __init__(self):
        self.variance_indicators = {}
        self.sma = {}
        self.order_refs = {}

        # Band state per symbol
        self._bands = {}           # symbol -> BandResult
        self._bar_count = {}       # symbol -> int
        self._pending_pairs = {}   # symbol -> list of (sell_order, buy_order)
        self._entry_done = {}      # symbol -> bool

        for data in self.datas:
            name = data._name
            self.variance_indicators[name] = VarianceIndicator(
                data,
                period=self.p.variance_lookback,
                vr_k=self.p.vr_k,
                trading_days=self.p.trading_days,
            )
            self.sma[name] = bt.indicators.SMA(data.close, period=self.p.sma_period)
            self._bands[name] = None
            self._bar_count[name] = 0
            self._pending_pairs[name] = []
            self._entry_done[name] = False

    def next(self):
        for data in self.datas:
            name = data._name
            self._bar_count[name] += 1

            var = self.variance_indicators[name]
            sma = self.sma[name]
            price = data.close[0]
            pos = self.getposition(data)

            # --- Recalibrate OU bands periodically ---
            if (self._bar_count[name] % self.p.recalib_period == 0
                    and len(data) >= self.p.ou_lookback):
                self._recalibrate(data, name)

            # --- Phase 1: Initial entry (momentum filter) ---
            if pos.size == 0 and not self._entry_done.get(name):
                trending = (
                    var.variance_ratio[0] >= self.p.min_variance_ratio
                    and var.efficiency_ratio[0] >= self.p.min_efficiency
                )
                above_sma = price > sma[0]

                if trending and above_sma:
                    self.buy(data=data)
                    self._entry_done[name] = True
                continue

            # --- Phase 2: DCA coverage management (when in position) ---
            if pos.size <= 0:
                continue

            bands = self._bands.get(name)
            if bands is None:
                # No bands yet — try to calibrate
                if len(data) >= self.p.ou_lookback:
                    self._recalibrate(data, name)
                    bands = self._bands.get(name)
                if bands is None:
                    continue

            # Clean up filled/cancelled pairs
            self._pending_pairs[name] = [
                (s, b) for s, b in self._pending_pairs[name]
                if s.status in (s.Submitted, s.Accepted)
            ]

            # Check coverage
            n_pending = len(self._pending_pairs[name])
            covered_qty = sum(
                abs(s.size) for s, _ in self._pending_pairs[name]
                if s.status in (s.Submitted, s.Accepted)
            )
            coverage = covered_qty / pos.size if pos.size > 0 else 1.0

            if coverage >= self.p.coverage_threshold:
                continue
            if n_pending >= self.p.max_pairs:
                continue

            # Place a new paired order
            lot_size = max(1, int(pos.size * self.p.lot_fraction))

            # Compute stop and buy levels from bands
            stop_price = self._compute_stop_price(price, bands)
            buy_price = self._compute_buy_price(price, bands)

            if stop_price >= price or buy_price >= stop_price:
                # Bands don't make sense at current price — use fallback
                stop_price = round(price * (1 - self.p.fallback_stop_pct), 2)
                buy_price = round(price * (1 - self.p.fallback_stop_pct - self.p.fallback_buy_offset_pct), 2)

            if buy_price <= 0 or stop_price <= 0:
                continue

            # Place sell-stop and paired buy-limit
            sell_order = self.sell(
                data=data,
                size=lot_size,
                exectype=bt.Order.Stop,
                price=stop_price,
            )
            buy_order = self.buy(
                data=data,
                size=lot_size,
                exectype=bt.Order.Limit,
                price=buy_price,
                valid=bt.num2date(data.datetime[0] + 30),
            )
            self._pending_pairs[name].append((sell_order, buy_order))

            # --- Phase 3: Full exit on regime change ---
            is_chop = var.regime[0] == VarianceIndicator.REGIME_MAP["CHOP"]
            below_sma = price < sma[0]

            if is_chop or below_sma:
                # Cancel all pending pairs and close position
                for s, b in self._pending_pairs[name]:
                    if s.status in (s.Submitted, s.Accepted):
                        self.cancel(s)
                    if b.status in (b.Submitted, b.Accepted):
                        self.cancel(b)
                self._pending_pairs[name] = []
                self.close(data=data)
                self._entry_done[name] = False

    def _recalibrate(self, data, name):
        """Recalibrate OU parameters from trailing prices."""
        closes = np.array(data.close.get(size=self.p.ou_lookback))
        if len(closes) < 10:
            return

        try:
            kappa, theta, sigma = calibrate_ou(closes)
            if kappa > 0 and sigma > 0:
                bands = compute_optimal_bands(
                    kappa, theta, sigma,
                    rho=self.p.rho,
                    c=self.p.transaction_cost,
                )
                self._bands[name] = bands
        except (ValueError, RuntimeError):
            pass  # keep previous bands

    def _compute_stop_price(self, current_price: float, bands: BandResult) -> float:
        """Map the optimal exit level to a stop price.

        The OU exit level is in spread-units. We convert to a price offset
        using the ratio of the band width to theta.
        """
        if abs(bands.theta) < 1e-10:
            return round(current_price * (1 - self.p.fallback_stop_pct), 2)

        # stop_offset_pct derived from optimal bands
        offset = bands.stop_offset_pct
        offset = max(offset, 0.002)  # floor at 0.2%
        offset = min(offset, 0.10)   # cap at 10%
        return round(current_price * (1 - offset), 2)

    def _compute_buy_price(self, current_price: float, bands: BandResult) -> float:
        """Map the optimal entry level to a buy-limit price.

        The buy price is set at the stop price minus the band width
        (converted to price-space).
        """
        stop_price = self._compute_stop_price(current_price, bands)

        if abs(bands.theta) < 1e-10:
            return round(stop_price * (1 - self.p.fallback_buy_offset_pct), 2)

        # band_width relative to theta, applied to price
        buy_offset_pct = bands.buy_offset / abs(bands.theta)
        buy_offset_pct = max(buy_offset_pct, 0.001)  # floor
        buy_offset_pct = min(buy_offset_pct, 0.05)    # cap
        return round(stop_price * (1 - buy_offset_pct), 2)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            name = order.data._name
            self.order_refs.pop(name, None)
