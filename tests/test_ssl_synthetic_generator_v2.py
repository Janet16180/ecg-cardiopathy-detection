"""Predeclared v2 narrowing retains backgrounds, algebra and engineering truth."""

import json
from dataclasses import replace

import numpy as np
import pytest

from ecg_experiment import ROOT
from ecg_experiment.ssl_synthetic_generator import generate_pair as generate_pair_v1
from ecg_experiment.ssl_synthetic_generator_pilot_qualification import pilot_seed
from ecg_experiment.ssl_synthetic_generator_qualification import _same_pair, gate_failures
from ecg_experiment.ssl_synthetic_generator_v2 import generate_pair


@pytest.mark.parametrize("profile", ["A", "B"])
@pytest.mark.parametrize("mode", ["focal", "persistent"])
@pytest.mark.parametrize("sample_rate", [250, 500])
def test_v2_preserves_exact_background_and_narrowing(profile, mode, sample_rate):
    args = {"profile": profile, "mode": mode, "sample_rate": sample_rate}
    old = generate_pair_v1(17, **args)
    new = generate_pair(17, **args)
    np.testing.assert_array_equal(old.baseline, new.baseline)
    np.testing.assert_array_equal(old.raw_baseline, new.raw_baseline)
    assert new.metadata["latent"] == old.metadata["latent"]
    assert new.metadata["nuisance"] == old.metadata["nuisance"]
    params = new.metadata["intervention"]
    assert 0.14 <= params["width_seconds"] <= 0.19
    assert params["amplitude_mv"] == old.metadata["intervention"]["amplitude_mv"]
    assert params["selected_beats"] == old.metadata["intervention"]["selected_beats"]
    primitive = np.array(params["primitive_coefficients"])
    assert sum(primitive[2:] != 0) == (
        1
        if primitive[:2].any()
        else sum(np.array(old.metadata["intervention"]["primitive_coefficients"])[2:] != 0)
    )
    assert np.count_nonzero(new.support.any(axis=1)) <= 6
    config = json.loads((ROOT / "configs/ssl_synthetic_generator_v1.json").read_text())
    assert not gate_failures(new, config)
    assert _same_pair(new, generate_pair(17, **args))


@pytest.mark.parametrize("nuisance", [False, True])
def test_v2_no_target_remains_identical_with_same_nuisance(nuisance):
    pair = generate_pair(21, target=False, nuisance=nuisance)
    positive = generate_pair(21, nuisance=nuisance)
    np.testing.assert_array_equal(pair.baseline, pair.edited)
    np.testing.assert_array_equal(pair.baseline, positive.baseline)
    assert pair.metadata["intervention"] == positive.metadata["intervention"]
    assert not pair.support.any()


def test_v2_substreams_and_profile_share_target_rule():
    pair = generate_pair(pilot_seed("validation", 3))
    noisy = generate_pair(pilot_seed("validation", 3), nuisance_seed=123)
    other_edit = generate_pair(pilot_seed("validation", 3), edit_seed=123)
    other_profile = generate_pair(pilot_seed("validation", 3), profile="B")
    assert pair.metadata["intervention"] == noisy.metadata["intervention"]
    assert pair.metadata["intervention"] == other_profile.metadata["intervention"]
    np.testing.assert_array_equal(pair.baseline, other_edit.baseline)
    assert pair.metadata["nuisance"] == other_edit.metadata["nuisance"]
    np.testing.assert_array_equal(pair.candidate, other_profile.candidate)


def test_v2_artifact_controls_are_still_rejected():
    pair = generate_pair(17, sample_rate=500)
    config = json.loads((ROOT / "configs/ssl_synthetic_generator_v1.json").read_text())
    assert {"occupancy", "exact_support"}.issubset(
        gate_failures(replace(pair, support=np.ones_like(pair.support)), config)
    )
    assert "exact_support" in gate_failures(replace(pair, support=np.zeros_like(pair.support)), config)
    rectangle = pair.baseline.copy()
    rectangle[6, 1000:1100] += 0.1
    assert "slope" in gate_failures(replace(pair, edited=rectangle, raw_edited=rectangle), config)
    corrupted = pair.edited.copy()
    corrupted[2] += 0.02
    assert "limb_algebra" in gate_failures(replace(pair, edited=corrupted), config)
