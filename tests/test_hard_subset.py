"""Checks of the Experiment 035 label rules, weights, folds and verdicts."""

import math

import numpy as np
import pytest

from ecg_experiment.hard_subset import (
    ARM_SPECS,
    arm_design,
    arm_verdict,
    patient_folds,
    prevalence_weights,
    ptbxl_under_challenge,
    recovery,
    rule_label,
    share_weights,
)

SR, BRADY, PVC, LVHV, AF = "426783006", "426177001", "427172004", "55827005", "164889003"
T_ABNORMAL, IRBBB = "164934002", "713426002"


def test_primary_and_secondary_rules_match_the_mapping():
    assert rule_label([SR], "primary") == 0
    assert math.isnan(rule_label([SR, BRADY], "primary"))
    assert rule_label([SR, BRADY], "secondary") == 0
    assert rule_label([BRADY], "secondary") == 0
    assert rule_label([BRADY, IRBBB], "primary") == 1
    assert math.isnan(rule_label([SR, PVC], "secondary"))


def test_ptbxl_negative_allows_compatible_codes_with_a_sinus_rhythm():
    assert rule_label([BRADY, PVC, LVHV], "ptbxl_negative") == 0
    assert math.isnan(rule_label([PVC], "ptbxl_negative"))
    assert math.isnan(rule_label([SR, AF], "ptbxl_negative"))
    assert rule_label([SR, PVC, IRBBB], "ptbxl_negative") == 1


def test_form_codes_stop_counting_as_positive():
    assert rule_label([SR, T_ABNORMAL], "primary") == 1
    assert math.isnan(rule_label([SR, T_ABNORMAL], "form_ignored"))
    assert rule_label([SR, T_ABNORMAL, IRBBB], "form_ignored") == 1
    assert rule_label([SR, T_ABNORMAL], "ptbxl_rule") == 0
    assert rule_label([SR], "form_ignored") == 0


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="Unknown label rule"):
        rule_label([SR], "strict")


def test_ptbxl_records_under_the_challenge_rules():
    classes = {"NORM": "NORM", "IRBBB": "CD"}
    assert ptbxl_under_challenge({"NORM": 100, "SR": 0}, classes) == {"primary": 0.0, "secondary": 0.0}
    brady = ptbxl_under_challenge({"NORM": 80, "SBRAD": 0}, classes)
    assert math.isnan(brady["primary"])
    assert brady["secondary"] == 0
    assert ptbxl_under_challenge({"NORM": 80, "IRBBB": 100}, classes)["primary"] == 1
    assert ptbxl_under_challenge({"NORM": 50, "SR": 0, "LPR": 0}, classes)["primary"] == 1
    assert math.isnan(ptbxl_under_challenge({"NORM": 50, "SR": 0, "PVC": 0}, classes)["secondary"])


def test_prevalence_weights_keep_family_totals():
    families = np.array(["ptbxl", "ptbxl", "a", "a", "a", "a"])
    y = np.array([1, 0, 1, 1, 1, 0])
    weights = prevalence_weights(families, y, 0.5)
    assert weights[:2].tolist() == [1, 1]
    rows = families == "a"
    assert weights[rows].sum() == pytest.approx(4)
    assert weights[rows & (y == 1)].sum() == pytest.approx(2)
    with pytest.raises(ValueError, match="lacks a class"):
        prevalence_weights(families, np.array([1, 0, 1, 1, 1, 1]), 0.5)


def test_share_weights_average_one():
    selected = np.array([True, False, False, False])
    weights = share_weights(selected, 0.5)
    assert weights.mean() == pytest.approx(1)
    assert weights[selected].sum() / weights.sum() == pytest.approx(0.5)


def test_patient_folds_keep_patients_together():
    patients = np.array(["a", "b", "a", "c", "d", "b", "e"])
    folds = patient_folds(patients, 3, 1)
    for patient in np.unique(patients):
        assert len(set(folds[patients == patient])) == 1
    assert set(folds) <= {0, 1, 2}
    assert np.array_equal(folds, patient_folds(patients, 3, 1))


def test_recovery_and_verdicts():
    assert recovery(0.69, 0.65, 0.73) == pytest.approx(0.5)
    assert math.isnan(recovery(0.7, 0.7, 0.69))
    hard_up = {"difference": 0.05, "ci_low": 0.01, "ci_high": 0.09}
    hard_unclear = {"difference": 0.05, "ci_low": -0.01, "ci_high": 0.09}
    assert arm_verdict(hard_up, {"difference": -0.002}, 0.7) == "fix"
    assert arm_verdict(hard_up, {"difference": -0.006}, 0.7) == "trade_off"
    assert arm_verdict(hard_unclear, {"difference": 0.0}, 0.7) == "suggestive"
    assert arm_verdict(hard_up, {"difference": 0.0}, 0.3) == "no_recovery"


def stacked():
    families = np.array(["ptbxl", "ptbxl", "ptbxl", "chapman_ningbo", "chapman_ningbo", "georgia", "cpsc"])
    primary = np.array([1, 0, 1, 1, 0, 1, np.nan])
    secondary = np.array([1, 0, 1, 1, 0, 1, 0])
    labels = {"primary": primary, "secondary": secondary, "ptbxl_negative": secondary,
              "form_ignored": primary, "ptbxl_rule": secondary}
    dropped = np.array([False, True, False, False, False, False, False])
    return families, labels, dropped


def test_arm_designs_select_the_described_rows():
    families, labels, dropped = stacked()
    pooled = arm_design(ARM_SPECS["pooled"], families, labels, dropped)
    assert pooled["selected"].tolist() == [True] * 6 + [False]
    assert pooled["weights"] is None
    assert pooled["indicators"] is None
    assert arm_design(ARM_SPECS["ptbxl"], families, labels, dropped)["selected"].sum() == 3
    assert arm_design(ARM_SPECS["loso_georgia"], families, labels, dropped)["selected"].tolist() == \
        [True] * 5 + [False, False]
    assert arm_design(ARM_SPECS["secondary_negative"], families, labels, dropped)["selected"].all()
    without = arm_design(ARM_SPECS["without_dropped"], families, labels, dropped)
    assert not without["selected"][1]
    indicator = arm_design(ARM_SPECS["source_indicator"], families, labels, dropped)["indicators"]
    assert indicator.shape == (6, 3)
    assert indicator[:3].sum() == 0
    upweighted = arm_design(ARM_SPECS["dropped_upweighted"], families, labels, dropped)
    assert upweighted["weights"][1] / upweighted["weights"].sum() == pytest.approx(1 / 3)
