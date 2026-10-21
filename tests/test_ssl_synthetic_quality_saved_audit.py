"""Artificial tests for independent dimensionless fixture verification."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ecg_experiment.ssl_synthetic_quality_saved_audit import (
    average_precision,
    coherence_values,
    intervention_values,
    primary_values,
    reconstruct,
    spectral_values,
)


def raw_fixture():
    times = np.arange(5000) / 500
    return np.array([np.sin(2 * np.pi * (lead + 1) * times) / 2 for lead in range(8)], dtype=np.float32)


def test_tied_ap_agrees_with_reference():
    rng = np.random.default_rng(6)
    mask = rng.random((12, 100)) > 0.8
    scores = rng.integers(0, 5, (12, 100)).astype(np.float32)
    assert average_precision(mask, scores) == pytest.approx(
        average_precision_score(mask.ravel(), scores.ravel()), abs=1e-15
    )
    assert average_precision(mask, mask.astype(float)) == 1


def test_reconstruction_propagates_all_affected_limb_leads_and_exact_masks():
    bank, strict, support, params = reconstruct(raw_fixture(), 0)
    assert bank.dtype == np.float32
    assert len(params["median"]) == 8
    rows = intervention_values(bank, strict, support)
    for mode in rows.values():
        assert mode["retained_energy_fraction"] >= 0.99
        assert mode["paired_difference_oracle_ap"] == 1
        assert mode["outside_intended_hull_cells"] == 0
    assert set(np.flatnonzero(strict[0].any(axis=1))) == {1, 2, 3, 4, 5}
    assert np.count_nonzero(bank[0] != bank[1]) > 0
    strict[0, 1, 100] = True
    with pytest.raises(ValueError, match="strict numerical support"):
        intervention_values(bank, strict, support)


def test_unintended_change_detected_with_truthful_masks():
    bank, _, _, _ = reconstruct(raw_fixture(), 0)
    bank[2, 7, 22] += 0.1
    difference = bank[2:].astype(float) - bank[1].astype(float)
    with pytest.raises(ValueError, match="outside constructed hull"):
        intervention_values(bank, difference != 0, np.abs(difference) > 0.01)


def test_coherence_bootstrap_and_zero_spectral_denominator():
    row = coherence_values(np.repeat(raw_fixture()[:1], 8, axis=0), 0)
    assert row["native_jaccard"] == 1
    assert row["difference"] > 0.5
    primary = primary_values([row] * 8)
    assert primary["ci_low"] == primary["ci_high"]
    assert primary["draws"] == 2000
    assert primary["unit"] == "generator_seed"
    bands, zeros = spectral_values(np.zeros((8, 5000), dtype=np.float32))
    assert all(zeros)
    assert all(value == [None] * 8 for value in bands.values())


def test_spectral_100hz_endpoint_excluded():
    times = np.arange(5000) / 500
    signal = np.sin(2 * np.pi * times) + 2 * np.sin(2 * np.pi * 100 * times)
    bands, zeros = spectral_values(np.repeat(signal[None, :], 8, axis=0))
    assert not any(zeros)
    assert bands["0p5_to_5"][0] == pytest.approx(1, abs=1e-12)


def test_main_independent_artificial_integration():
    main = pytest.importorskip("ecg_experiment.ssl_synthetic_generated_quality")
    from ecg_experiment.ssl_synthetic_quality_saved_audit import morphology_values
    from scripts.validation.audit_ssl_synthetic_generated_quality import compare

    raw = np.random.default_rng(9).normal(0, 0.2, (8, 5000)).astype(np.float32)
    values, parameters = main.standardize(raw)
    bank, strict, support = main.intervention_bank(values, 0)
    expected, expected_strict, expected_support, expected_parameters = reconstruct(raw, 0)
    assert bank.tobytes() == expected.tobytes()
    assert strict.tobytes() == expected_strict.tobytes()
    assert support.tobytes() == expected_support.tobytes()
    compare(expected_parameters, parameters, "standardization")
    compare(coherence_values(values, 0), main.coherence(values, 0), "coherence")
    compare(morphology_values(values, raw), main.morphology(values, raw), "morphology")
    compare(
        intervention_values(bank, strict, support), main.intervention_metrics(bank, strict, support), "edit"
    )


def test_nested_compare_detects_changed_metric():
    from scripts.validation.audit_ssl_synthetic_generated_quality import compare

    assert compare({"score": [0.1, None, True]}, {"score": [0.1, None, True]}, "fixture") == 3
    with pytest.raises(ValueError, match="numeric"):
        compare({"score": [0.1]}, {"score": [0.1001]}, "fixture")
