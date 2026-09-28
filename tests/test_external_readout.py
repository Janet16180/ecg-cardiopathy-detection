"""Checks of the Experiment 022 v3 head, cache-selection and scoring helpers."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.external_encoders import jepa_input, xecg_input
from ecg_experiment.external_readout import (
    ALL_HEADS,
    cache_positions,
    combine_features,
    fit_logistic_c,
    head_inputs,
)
from ecg_experiment.full_development import fit_logistic, predict
from ecg_experiment.xecg import preprocess_xecg
from scripts.experiments.run_sph_external022 import (
    EXPECTED_EVALUATION,
    check_counts,
    development_scores,
    median_residual,
    policy_scores,
    quality_counts,
)


def test_fit_logistic_c_at_001_reproduces_fit_logistic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    y = (x[:, 0] + rng.normal(0, 1, 300) > 0).astype(int)
    assert np.array_equal(predict(fit_logistic_c(x, y, 0.01), x), predict(fit_logistic(x, y), x))


def test_fit_logistic_c_uses_the_given_strength():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 5))
    y = (x[:, 0] > 0).astype(int)
    weak = fit_logistic_c(x, y, 0.01)[1].coef_
    strong = fit_logistic_c(x, y, 0.1)[1].coef_
    assert np.abs(strong).sum() > np.abs(weak).sum()


def test_cache_positions_find_rows_by_ecg_id():
    assert cache_positions(np.array([30, 10, 20]), np.array([20, 99, 30])).tolist() == [2, -1, 0]


def test_cache_positions_reject_a_repeated_id():
    with pytest.raises(ValueError, match="repeats"):
        cache_positions(np.array([1, 1]), np.array([1]))


def test_combine_features_fill_missing_rows_in_order():
    cache = np.arange(6, dtype=np.float32).reshape(3, 2)
    extracted = np.array([[-1, -1], [-2, -2]], dtype=np.float32)
    result = combine_features(np.array([5, 2, 6, 1]), cache, np.array([1, 2, 3]), np.array([5, 6]), extracted)
    assert result.tolist() == [[-1, -1], [2, 3], [-2, -2], [0, 1]]


def test_combine_features_reject_other_extracted_rows():
    cache = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="missing"):
        combine_features(np.array([7, 1]), cache, np.array([1]), np.array([8]), np.zeros((1, 2)))


def test_head_inputs_cover_every_head():
    frame = pd.DataFrame({"age": [50.0, 60.0], "male": [1.0, 0.0]})
    features = {"cpc": np.zeros((2, 4)), "jepa": np.ones((2, 3)), "xecg": np.full((2, 5), 2.0)}
    inputs = head_inputs(frame, 55.0, features)
    assert tuple(inputs) == ALL_HEADS
    assert inputs["cpc_age_sex_standard"].shape == (2, 7)
    assert inputs["jepa_standard_c01"] is features["jepa"]


def test_jepa_input_keeps_leads_i_ii_and_v1_to_v6():
    signal = np.repeat(np.arange(12, dtype=np.float32)[:, None], 5000, axis=1)
    reduced = jepa_input(signal)
    assert reduced.shape == (8, 2500)
    assert np.allclose(reduced[:, 0], [0, 1, 6, 7, 8, 9, 10, 11], atol=1e-5)


def test_xecg_input_widens_float16_exactly():
    rng = np.random.default_rng(2)
    stored = rng.normal(0, 0.5, (12, 5000)).astype(np.float16)
    assert np.array_equal(xecg_input(stored.astype(np.float32)), preprocess_xecg(stored.astype(np.float64)))


def test_check_counts_reject_a_changed_table():
    frame = pd.DataFrame({"primary": [0.0, 1.0, np.nan], "secondary": [0.0, 1.0, 0.0],
                          "age": [50.0, 60.0, 70.0], "patient_id": ["a", "b", "c"]})
    assert EXPECTED_EVALUATION["records"] == 25577
    with pytest.raises(ValueError, match="changed"):
        check_counts(frame)


def test_development_scores_split_original_and_added_labeled_rows():
    development = pd.DataFrame({"standard": [0.0, 1.0, np.nan, 0.0, 1.0],
                                "original": [True, True, True, False, False]})
    scores = {"head": np.array([0.1, 0.9, 0.5, 0.8, 0.2])}
    result = development_scores(development, scores)["head"]
    assert result["full"]["records"] == 4
    assert result["original"]["auroc"] == 1.0
    assert result["added"]["auroc"] == 0.0


def test_policy_scores_drop_excluded_and_undefined_records():
    frame = pd.DataFrame({"primary": [0.0, 1.0, 1.0, np.nan, 0.0],
                          "use_training": [True, True, False, True, True]})
    values = np.array([0.1, 0.9, 0.0, 0.5, 0.2])
    result = policy_scores(frame, dict.fromkeys(("cpc_standard", "jepa_standard", "xecg_standard"), values))
    assert result["jepa_standard"]["records"] == 3
    assert result["jepa_standard"]["auroc"] == 1.0


def test_quality_counts_tally_manifest_reasons():
    frame = pd.DataFrame({"exclusion_reasons": ["", "adc_rail", "adc_rail;extreme_amplitude"],
                          "review_flags": ["", "", "flat_lead"], "use_training": [True, False, False]})
    assert quality_counts(frame) == {"exclusion_reasons": {"adc_rail": 2, "extreme_amplitude": 1},
                                     "review_flags": {"flat_lead": 1}, "excluded_records": 2}


def test_median_residual_ignores_one_corrupted_sample_but_not_a_lead_swap():
    rng = np.random.default_rng(3)
    lead_i, lead_ii = rng.normal(0, 0.5, (2, 5000))
    signal = np.zeros((12, 5000))
    signal[:4] = lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2
    signal[3, 2059] *= -1
    assert median_residual(signal) < 1e-9
    signal[[0, 1]] = signal[[1, 0]]
    assert median_residual(signal) > 0.1
