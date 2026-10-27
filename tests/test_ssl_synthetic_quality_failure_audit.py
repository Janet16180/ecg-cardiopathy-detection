"""Artificial failed-profile admission checks."""

import pytest

from scripts.validation.audit_ssl_synthetic_quality_failure_v1 import forecasts


def test_only_wall_forecast_stops_and_arithmetic_corruption_raises():
    value = {
        "cpu_seconds": 0.2,
        "wall_seconds": 5.8,
        "spent_cpu_seconds": 1,
        "spent_wall_seconds": 7,
        "forecast_cpu_seconds": 83,
        "forecast_wall_seconds": 1545,
        "peak_allocated_bytes": 45_000_000,
        "forecast_saved_bytes": 228_000_000,
    }
    assert forecasts(value) == {
        "wall_admitted": False,
        "cpu_admitted": True,
        "memory_admitted": True,
        "output_admitted": True,
    }
    value["forecast_wall_seconds"] = 1199
    with pytest.raises(ValueError, match="forecast arithmetic"):
        forecasts(value)
