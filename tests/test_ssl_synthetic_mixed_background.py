"""Artificial checks for matched-budget mixed-background learning and fixed scientific guards."""

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_background_editor_transfer as transfer
from ecg_experiment import ssl_synthetic_mixed_background as mixed
from ecg_experiment import ssl_synthetic_randomized_localization as original


def rows_fixture() -> list[dict]:
    """Create a complete fixture with heterogeneous records and fixed paired training effects.

    Returns
    -------
    list
        All768 model-record dictionaries with an exact synthetic0.02 training effect.
    """
    rows = []
    for arm in mixed.TRAINING_ARMS:
        for source in transfer.SOURCES:
            for slot in range(64):
                for fit in transfer.FITS:
                    positive = {}
                    for variant in transfer.POSITIVES:
                        edited = 0.05 + slot / 10000 + fit / 1000 + (0.02 if arm == "mixed" else 0)
                        positive[variant] = dict.fromkeys(mixed.METRICS, edited)
                    rows.append(
                        {
                            "training_arm": arm,
                            "source": source,
                            "slot": slot,
                            "fit_id": fit,
                            "arm": "B",
                            "positives": positive,
                            "controls": {
                                control: dict.fromkeys(
                                    ["mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5"],
                                    0.1 + (0.01 if arm == "mixed" else 0),
                                )
                                for control in ["sham", "nuisance"]
                            },
                        }
                    )
    return rows


def test_new_real_training_reuses_original_editor_and_scales_once() -> None:
    """Physical-native transformation and every original Hann/nuisance arithmetic step are unchanged."""
    physical = np.random.default_rng(61).normal(size=(12, 5000))
    actual = mixed.real_case(physical, 40)
    expected = original.make_case(physical[list(transfer.NATIVE)].astype(np.float32), 40)
    for left, right in zip(actual[:3], expected[:3], strict=True):
        np.testing.assert_array_equal(left, right)
    assert actual[3] == expected[3]
    with pytest.raises(ValueError, match="fixed real training geometry"):
        mixed.real_case(physical, 80)


def test_complete_shared_slot_bootstrap_retains_fixed_fits_and_differences() -> None:
    """All paired-arm effects share geometry draws and use every fixed seed rather than picking one."""
    rows = rows_fixture()
    config = {"bootstrap": {"seed": 2026106701, "draws": 2000}}
    statistics, draws = mixed.summarize(rows, config)
    assert len(statistics["scientific_gates"]) == 14
    assert statistics["scientific_passed"]
    assert len(statistics["per_fit_mode"]) == 48
    assert statistics["bootstrap"]["fits_resampled"] is False
    key = "mixed_minus_generated__real__biphasic__pooled__edited_ap"
    column = statistics["endpoint_order"].index(key)
    np.testing.assert_allclose(draws[:, column], 0.02, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="exactly once"):
        mixed.summarize(rows[:-1], config)
    with pytest.raises(ValueError, match="exactly once"):
        mixed.summarize(rows + [rows[0]], config)


def test_primary_gain_cannot_hide_an_unfavorable_repeated_mode() -> None:
    """A positive pooled result fails when the fixed repeated-mode safeguard fails."""
    rows = rows_fixture()
    for row in rows:
        if row["training_arm"] == "mixed" and row["source"] == "real":
            row["positives"]["biphasic_focal"]["edited_ap"] += 0.04
            row["positives"]["biphasic_repeated"]["edited_ap"] -= 0.04
    statistics, _ = mixed.summarize(rows, {"bootstrap": {"seed": 2026106701, "draws": 2000}})
    assert statistics["scientific_gates"]["primary_AP_gain_at_least_0p01"]
    assert not statistics["scientific_gates"]["real_biphasic_repeated_noninferiority"]
    assert not statistics["scientific_passed"]


def test_control_worsening_cannot_be_rescued_by_positive_ap() -> None:
    """The two real controls remain separate guards despite a favorable AP contrast."""
    rows = rows_fixture()
    for row in rows:
        if row["training_arm"] == "mixed" and row["source"] == "real":
            row["controls"]["nuisance"]["top600_mean"] = 0.14
    statistics, _ = mixed.summarize(rows, {"bootstrap": {"seed": 2026106701, "draws": 2000}})
    assert statistics["scientific_gates"]["real_sham_top600_preservation"]
    assert not statistics["scientific_gates"]["real_nuisance_top600_preservation"]
    assert not statistics["scientific_passed"]


def test_generated_control_cells_remain_exploratory() -> None:
    """The fourteen predeclared guards are not silently expanded to favorable/unfavorable secondary cells."""
    rows = rows_fixture()
    for row in rows:
        if row["training_arm"] == "mixed" and row["source"] == "generated":
            row["controls"]["nuisance"]["top600_mean"] = 0.9
    statistics, _ = mixed.summarize(rows, {"bootstrap": {"seed": 2026106701, "draws": 2000}})
    assert statistics["scientific_passed"]
    assert statistics["endpoints"]["mixed_minus_generated__generated__nuisance__top600_mean"]["mean"] > 0.7
