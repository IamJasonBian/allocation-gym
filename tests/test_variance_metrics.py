import numpy as np
from allocation_gym.metrics.variance_metrics import VarianceMetrics, VarianceResult


def test_compute_returns_variance_result(synthetic_arrays):
    result = VarianceMetrics.compute(**synthetic_arrays)
    assert isinstance(result, VarianceResult)


def test_yang_zhang_vol_positive(synthetic_arrays):
    result = VarianceMetrics.compute(**synthetic_arrays)
    assert result.yang_zhang_var > 0
    assert result.yang_zhang_vol > 0
    assert result.yang_zhang_vol_ann > 0


def test_variance_ratio_reasonable(synthetic_arrays):
    result = VarianceMetrics.compute(**synthetic_arrays)
    assert 0.1 < result.variance_ratio < 5.0


def test_efficiency_ratio_bounded(synthetic_arrays):
    result = VarianceMetrics.compute(**synthetic_arrays)
    assert 0.0 <= result.efficiency_ratio <= 1.0


def test_regime_is_valid(synthetic_arrays):
    result = VarianceMetrics.compute(**synthetic_arrays)
    valid_regimes = {"RANDOM_WALK", "STRONG_TREND", "NOISY_TREND", "CHOP", "MEAN_REVERT"}
    assert result.regime in valid_regimes


def test_semivariance_positive(synthetic_arrays):
    result = VarianceMetrics.compute(**synthetic_arrays)
    assert result.downside_semivol > 0
    assert result.upside_semivol > 0


def test_short_data_returns_defaults():
    result = VarianceMetrics.compute(
        opens=np.array([100.0, 101.0]),
        highs=np.array([102.0, 103.0]),
        lows=np.array([99.0, 100.0]),
        closes=np.array([101.0, 102.0]),
    )
    assert result.yang_zhang_var == 0.0
    assert result.regime == "RANDOM_WALK"
