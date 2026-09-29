"""Checks of the Experiment 027b arm definitions and prespecified reading."""

import numpy as np
import pandas as pd
from scipy.special import expit

from ecg_experiment.multisource_calibration import fit_arm
from scripts.experiments.run_multisource_calibration027b import (
    ARMS,
    arm_members,
    fit_arms,
    heldout_sets,
    reading,
)


def pool(records: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    families = rng.choice(["ptbxl", "chapman_ningbo", "georgia", "cpsc"], records, p=[0.1, 0.6, 0.2, 0.1])
    logits = rng.normal(0, 2, records)
    y = (rng.random(records) < expit(logits + 0.5)).astype(int)
    return y, families, {"a_standard": expit(logits), "b_standard": expit(0.5 * logits)}


def test_arm_members_hold_out_one_family_each():
    families = np.array(["ptbxl", "chapman_ningbo", "georgia", "cpsc"])
    members = arm_members(families)
    assert set(members) == set(ARMS)
    assert members["ptbxl"].tolist() == [True, False, False, False]
    assert members["pooled"].all()
    assert members["balanced"].all()
    assert members["loso_georgia"].tolist() == [True, True, False, True]


def test_fit_arms_fits_each_arm_on_its_members():
    y, families, scores = pool()
    calibrators, thresholds = fit_arms(y, families, scores)
    selected = families != "cpsc"
    logits = np.log(scores["a_standard"][selected] / (1 - scores["a_standard"][selected]))
    reference, threshold = fit_arm(logits, y[selected])
    assert calibrators["loso_cpsc"]["a_standard"].coef_[0, 0] == reference.coef_[0, 0]
    assert thresholds["loso_cpsc"]["a_standard"] == threshold
    balanced, pooled = calibrators["balanced"]["a_standard"], calibrators["pooled"]["a_standard"]
    assert balanced.coef_[0, 0] != pooled.coef_[0, 0]


def sph_summary(pooled: list[tuple[float, float]]) -> dict[str, object]:
    heads = ("cpc_standard", "jepa_standard", "xecg_standard")
    differences = {"pooled": {head: {"deviation": {"ci_low": low, "ci_high": high}}
                              for head, (low, high) in zip(heads, pooled, strict=True)}}
    rates = {"ptbxl": {head: {"sensitivity": {"ci_low": 0.91, "ci_high": 0.93}} for head in heads},
             "pooled": {head: {"sensitivity": {"ci_low": 0.94, "ci_high": 0.96}} for head in heads}}
    return {"bootstrap": {"differences": differences, "rates": rates}}


def test_reading_adopts_pooled_calibration_only_by_the_rule():
    better, same, worse = (-0.03, -0.01), (-0.01, 0.01), (0.01, 0.02)
    assert reading(sph_summary([better, better, same]))["adopt_pooled_calibration"]
    assert not reading(sph_summary([better, same, same]))["adopt_pooled_calibration"]
    assert not reading(sph_summary([better, better, worse]))["adopt_pooled_calibration"]
    result = reading(sph_summary([better, better, better]))
    assert result["threshold_transfers_to_sph"]["pooled"]["jepa_standard"]
    assert not result["threshold_transfers_to_sph"]["ptbxl"]["jepa_standard"]


def test_heldout_sets_select_one_family_and_the_nonzero_ningbo_rows():
    challenge = pd.DataFrame({"family": ["chapman_ningbo", "chapman_ningbo", "georgia", "cpsc"],
                              "zero_lead": [True, False, False, False]})
    sets = heldout_sets(challenge)
    assert sets["georgia"][0].tolist() == [False, False, True, False]
    assert sets["chapman_ningbo_without_zero_leads"][1] == "chapman_ningbo"
    assert sets["chapman_ningbo_without_zero_leads"][0].tolist() == [False, True, False, False]
