"""Scientific guards for the clean cached linear-probe rerun."""

import numpy as np
import pytest

from ecg_experiment.clean_cached_probes import fit_probe, validate_selection


def test_clean_selection_rejects_patient_or_label_change() -> None:
    """Cleaning may exclude a row but cannot rewrite its supervised identity."""
    original = [{"ecg_id": "3", "patient_id": "12.0", "target": "1"}]
    clean = [{"record_id": "ptbxl:3", "patient_id": "ptbxl:12.0", "target": "1"}]
    assert validate_selection(clean, original) == original
    with pytest.raises(ValueError, match="identity differs"):
        validate_selection([{**clean[0], "target": "0"}], original)
    with pytest.raises(ValueError, match="identity differs"):
        validate_selection([{**clean[0], "patient_id": "ptbxl:13.0"}], original)


def test_probe_scaler_uses_training_rows_only() -> None:
    """A shifted development cohort must not affect fitted scaler statistics."""
    train_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = np.array([0, 0, 1, 1])
    dev_x = np.array([[100.0], [101.0]])
    dev_y = np.array([0, 1])
    parameters, _ = fit_probe(train_x, train_y, dev_x, dev_y)
    assert np.array_equal(parameters["mean"], np.array([0.0]))
    assert np.array_equal(parameters["scale"], np.array([np.sqrt(2.5)]))
