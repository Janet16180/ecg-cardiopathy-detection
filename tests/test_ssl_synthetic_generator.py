"""Engineering truth and artifact-control tests without patient payloads."""

import json
from dataclasses import replace

import numpy as np
import pytest

from ecg_experiment import ROOT
from ecg_experiment.resample import resample_full
from ecg_experiment.ssl_synthetic_generator import (
    SUPPORT_FLOOR_MV,
    couple_limb_leads,
    edit_profile,
    generate_pair,
)
from ecg_experiment.ssl_synthetic_generator_qualification import (
    _independent_streams,
    _invalid_controls,
    _same_pair,
    gate_failures,
    pair_metrics,
    qualify_generator,
)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/ssl_synthetic_generator_v1.json").read_text())


@pytest.mark.parametrize("profile", ["A", "B"])
@pytest.mark.parametrize("mode", ["focal", "persistent"])
@pytest.mark.parametrize("sample_rate", [250, 500])
def test_pair_exact_truth_and_gates(profile, mode, sample_rate, config):
    pair = generate_pair(17, profile=profile, mode=mode, sample_rate=sample_rate)
    assert pair.edited.shape == (12, sample_rate * 10)
    np.testing.assert_array_equal(pair.support, np.abs(pair.difference) > SUPPORT_FLOOR_MV)
    assert pair.edited.dtype == np.float32
    assert not gate_failures(pair, config), pair_metrics(pair)
    assert _same_pair(pair, generate_pair(17, profile=profile, mode=mode, sample_rate=sample_rate))
    assert _independent_streams(17, pair)


@pytest.mark.parametrize("nuisance", [False, True])
def test_shams_keep_paired_nuisance_identical(nuisance, config):
    pair = generate_pair(5, target=False, nuisance=nuisance)
    positive = generate_pair(5, nuisance=nuisance)
    np.testing.assert_array_equal(pair.baseline, pair.edited)
    np.testing.assert_array_equal(pair.baseline, positive.baseline)
    assert pair.metadata["nuisance"] == positive.metadata["nuisance"]
    assert not pair.support.any()
    assert not pair.nominal.any()
    assert not gate_failures(pair, config)


def test_polyphase_pair_and_candidate_prior_do_not_depend_on_target():
    raw = generate_pair(17, sample_rate=500)
    pair = generate_pair(17)
    np.testing.assert_array_equal(pair.edited, resample_full(raw.edited, 1, 2))
    np.testing.assert_array_equal(pair.baseline, resample_full(raw.baseline, 1, 2))
    other = generate_pair(17, profile="B", mode="persistent", edit_seed=100)
    np.testing.assert_array_equal(pair.candidate, other.candidate)
    assert pair.metadata["latent"]["beat_landmarks"] == other.metadata["latent"]["beat_landmarks"]
    assert all(
        set(landmark) == {"r", "j", "st_end"} for landmark in pair.metadata["latent"]["beat_landmarks"]
    )
    assert len(pair.metadata["latent"]["permissible_leads"]) == 12


def test_lead_algebra_preserves_all_coupled_edits():
    basis = np.eye(8)
    actual = couple_limb_leads(basis)
    np.testing.assert_array_equal(actual[:, 0], [1, 0, -1, -0.5, 1, -0.5, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(actual[:, 1], [0, 1, 1, -0.5, -0.5, 1, 0, 0, 0, 0, 0, 0])
    assert actual.shape == (12, 8)


@pytest.mark.parametrize("profile", ["A", "B"])
def test_profiles_have_compact_support_and_smooth_endpoints(profile):
    values = edit_profile(np.array([-1, 0, 1e-8, 0.5, 1 - 1e-8, 1, 2]), profile)
    np.testing.assert_array_equal(values[[0, 1, 5, 6]], np.zeros(4))
    assert values[3] == 1
    assert max(values[2], values[4]) / 1e-8 < 1e-5


def test_expected_invalid_controls_are_rejected(config):
    assert all(_invalid_controls(config).values())


def test_reported_support_cannot_silently_omit_changed_coupled_leads(config):
    pair = generate_pair(17)
    bad = replace(pair, support=pair.support & (np.arange(12)[:, None] >= 6))
    assert "exact_support" in gate_failures(bad, config)


@pytest.mark.parametrize(
    "kwargs", [{"sample_rate": 100}, {"mode": "unknown"}, {"profile": "C"}, {"seed": -1}]
)
def test_unsupported_inputs_raise(kwargs):
    arguments = {"seed": 3, **kwargs}
    with pytest.raises(ValueError, match="Expected|Profile|Seed"):
        generate_pair(**arguments)


def test_full_grid_is_prospective_and_mechanism_b_is_not_training_profile(config):
    assert config["seed_ids"] == list(range(64))
    assert len(config["profiles"]) * len(config["modes"]) * len(config["sample_rates"]) * 64 == 512
    assert config["support_floor_mv"] == SUPPORT_FLOOR_MV
    assert not np.array_equal(
        edit_profile(np.linspace(0, 1, 101), "A"), edit_profile(np.linspace(0, 1, 101), "B")
    )


def test_qualification_never_overwrites_existing_directory(tmp_path):
    sentinel = tmp_path / "retained.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        qualify_generator(tmp_path)
    assert sentinel.read_text() == "original"
    assert list(tmp_path.iterdir()) == [sentinel]
