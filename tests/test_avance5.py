"""Check the delivery's scientific boundaries using synthetic administrative records."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.avance5 import conditional_projection, cv_splits, independence


def test_repeated_patients_stay_together():
    rows = pd.DataFrame({"source": ["ptbxl"] * 24,
                         "patient_id": np.repeat(np.arange(12).astype(str), 2),
                         "record_id": np.arange(24).astype(str), "target": np.tile([0, 1], 12)})
    for fit, validation in cv_splits(rows):
        assert not set(rows.patient_id.iloc[fit]) & set(rows.patient_id.iloc[validation])


def test_identity_check_handles_legacy_integral_patient_ids():
    train = pd.DataFrame({"source": ["ptbxl"], "patient_id": ["ptbxl:12.0"], "record_id": ["ptbxl:1"]})
    held = pd.DataFrame({"patient_id": ["12"], "ecg_id": ["2"]})
    with pytest.raises(ValueError, match="patient overlap"):
        independence(train, "ptb", held)


def test_low_prevalence_projection_does_not_reuse_hospital_class_balance():
    result = conditional_projection(0.8, 0.95, people=10000)
    assert (result.VPP_hipotetico < 0.1).all()
    assert np.allclose(result.detectados_hipoteticos + result.omitidos_hipoteticos,
                       result.positivos_hipoteticos)
