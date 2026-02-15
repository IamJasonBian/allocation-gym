"""
Variance Metrics — Pure numpy computation layer.
Ported verbatim from trading_engine.py. No Backtrader dependency.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class VarianceResult:
    """Output from variance metric calculations."""
    yang_zhang_var: float = 0.0
    yang_zhang_vol: float = 0.0
    yang_zhang_vol_ann: float = 0.0
    variance_ratio: float = 1.0
    efficiency_ratio: float = 0.5
    vol_of_vol: float = 0.0
    downside_semivol: float = 0.0
    upside_semivol: float = 0.0
    vol_skew: float = 1.0
    regime: str = "RANDOM_WALK"
    trading_days: int = 252


class VarianceMetrics:
    """
    Compute all 9 trailing variance metrics from OHLC data.
    Standalone — no framework dependency. Works with numpy arrays.
    """

    @staticmethod
    def compute(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
        vr_k: int = 3,
        trading_days: int = 252,
    ) -> VarianceResult:
        result = VarianceResult(trading_days=trading_days)

        n = min(period, len(closes) - 1)
        if n < 3:
            return result

        O = opens[-n:]
        H = highs[-n:]
        L = lows[-n:]
        C = closes[-n:]
        prev_C = closes[-(n + 1):-1]

        # Yang-Zhang Variance
        log_oc = np.log(C / O)
        log_overnight = np.log(O / prev_C)
        log_hc = np.log(H / C)
        log_ho = np.log(H / O)
        log_lc = np.log(L / C)
        log_lo = np.log(L / O)

        sigma2_overnight = np.var(log_overnight, ddof=1)
        sigma2_oc = np.var(log_oc, ddof=1)
        sigma2_rs = np.mean(log_hc * log_ho + log_lc * log_lo)

        k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34
        yz_var = sigma2_overnight + k * sigma2_oc + (1 - k) * sigma2_rs
        yz_var = max(yz_var, 1e-12)

        result.yang_zhang_var = yz_var
        result.yang_zhang_vol = math.sqrt(yz_var)
        result.yang_zhang_vol_ann = math.sqrt(yz_var * trading_days)

        # Variance Ratio
        log_returns = np.log(C / prev_C)
        var_1 = np.var(log_returns, ddof=1) if len(log_returns) > 1 else 1e-12

        if len(log_returns) >= vr_k:
            k_returns = np.array([
                sum(log_returns[i:i + vr_k])
                for i in range(len(log_returns) - vr_k + 1)
            ])
            var_k = np.var(k_returns, ddof=1) if len(k_returns) > 1 else var_1 * vr_k
            result.variance_ratio = var_k / (vr_k * var_1) if var_1 > 0 else 1.0

        # Efficiency Ratio
        if len(C) >= 2:
            net_move = abs(C[-1] - C[0])
            total_path = np.sum(np.abs(np.diff(C)))
            result.efficiency_ratio = net_move / total_path if total_path > 0 else 0.0

        # Vol of Vol
        if len(log_returns) >= 5:
            daily_vars = log_returns ** 2
            mean_var = np.mean(daily_vars)
            std_var = np.std(daily_vars, ddof=1)
            result.vol_of_vol = std_var / mean_var if mean_var > 0 else 0.0

        # Downside / Upside Semivariance
        neg_returns = log_returns[log_returns < 0]
        pos_returns = log_returns[log_returns > 0]

        down_var = np.mean(neg_returns ** 2) if len(neg_returns) > 0 else 1e-12
        up_var = np.mean(pos_returns ** 2) if len(pos_returns) > 0 else 1e-12

        result.downside_semivol = math.sqrt(down_var)
        result.upside_semivol = math.sqrt(up_var)
        result.vol_skew = down_var / up_var if up_var > 0 else 1.0

        # Regime Classification
        vr = result.variance_ratio
        er = result.efficiency_ratio

        if vr > 1.10 and er > 0.60:
            result.regime = "STRONG_TREND"
        elif vr > 1.10:
            result.regime = "NOISY_TREND"
        elif vr < 0.90 and er < 0.30:
            result.regime = "CHOP"
        elif vr < 0.90:
            result.regime = "MEAN_REVERT"
        else:
            result.regime = "RANDOM_WALK"

        return result
