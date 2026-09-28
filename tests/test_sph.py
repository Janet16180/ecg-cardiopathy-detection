"""Checks of the SPH label mapping and duplicate resolution."""

import math

import numpy as np
import pandas as pd

from ecg_experiment.sph import base_codes, duplicate_status, einthoven_residual, labels


def test_base_codes_drop_modifiers():
    assert base_codes("22;145+362") == {"22", "145"}
    assert base_codes("147+226+367") == {"147"}


def test_normal_only_is_negative_and_findings_are_positive():
    assert labels({"1"})["primary"] == 0
    result = labels({"22", "145"})
    assert result["primary"] == 1
    assert result["STTC"] == 1
    assert result["MI"] == 0


def test_rhythm_only_is_undefined_except_sinus_variants_in_the_secondary_label():
    atrial_fibrillation = labels({"50"})
    assert math.isnan(atrial_fibrillation["primary"])
    assert math.isnan(atrial_fibrillation["secondary"])
    bradycardia = labels({"22", "23"})
    assert math.isnan(bradycardia["primary"])
    assert bradycardia["secondary"] == 0


def test_flutter_conduction_codes_are_not_av_block():
    assert math.isnan(labels({"51", "85"})["primary"])


def test_duplicates_keep_the_first_copy_unless_the_labels_conflict():
    index = pd.Index(["A3", "A1", "A2", "A4", "A5"])
    sha = pd.Series(["x", "x", "y", "z", "z"], index=index)
    codes = pd.Series(["1", "1", "1", "1", "22"], index=index)
    status = duplicate_status(sha, codes)
    assert status.to_dict() == {"A3": "dropped_copy", "A1": "kept", "A2": "unique",
                                "A4": "dropped_conflict", "A5": "dropped_conflict"}


def test_einthoven_residual_is_zero_for_consistent_limb_leads():
    rng = np.random.default_rng(0)
    lead_i, lead_ii = rng.normal(size=(2, 100))
    signal = np.zeros((12, 100))
    signal[:4] = [lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2]
    assert einthoven_residual(signal) < 1e-12
