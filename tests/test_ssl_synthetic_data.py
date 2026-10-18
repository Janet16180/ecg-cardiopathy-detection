"""Verify role and nuisance allocation without accessing a waveform source or real generator."""

from types import SimpleNamespace

import numpy as np
import pytest

from ecg_experiment.ssl_synthetic_data import (
    build_bank,
    case_specifications,
    latent_seed,
    load_bank,
    positive_class_weight,
    save_bank,
)


def fake_pair(seed, **kwargs):
    support = np.zeros((12, 2500), dtype=bool)
    if kwargs["target"]:
        support[0, :20] = True
    baseline = np.full(support.shape, 0.1, dtype=np.float32)
    return SimpleNamespace(
        baseline=baseline,
        edited=baseline + support.astype(np.float32),
        support=support,
        candidate=np.ones_like(support),
        metadata={"seed": seed, **kwargs},
    )


def test_roles_are_disjoint_and_training_budget_is_exact():
    seeds = [
        {latent_seed(role, i) for i in range(n)}
        for role, n in [("train", 500), ("validation", 128), ("evaluation", 256)]
    ]
    assert not seeds[0] & seeds[1]
    assert not seeds[0] & seeds[2]
    assert not seeds[1] & seeds[2]
    specs = case_specifications("train", 500)
    assert len(specs) == 500
    assert sum(s["target"] for s in specs) == 375
    assert sum(s["kind"] == "sham" for s in specs) == 62
    assert sum(s["kind"] == "nuisance" for s in specs) == 63
    for mode in ["focal", "persistent"]:
        positive = [s for s in specs if s["target"] and s["mode"] == mode]
        assert {s["nuisance"] for s in positive} == {False, True}


def test_variants_share_latent_identity_but_not_target_or_nuisance_labels():
    specs = case_specifications("validation", 2)
    assert len(specs) == 8
    for index in range(2):
        cases = [s for s in specs if s["subject"] == index]
        assert len({s["seed"] for s in cases}) == 1
        assert cases[0]["nuisance"] == cases[1]["nuisance"]
        assert not cases[2]["target"]
        assert not cases[2]["nuisance"]
        assert not cases[3]["target"]
        assert cases[3]["nuisance"]


def test_bad_case_aborts_instead_of_retrying_or_dropping():
    calls = []

    def qualify(pair):
        calls.append(pair)
        return ["bad support"]

    with pytest.raises(ValueError, match="failed qualification"):
        build_bank("train", 2, "A", fake_pair, qualify)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Held-out edit profile"):
        build_bank("train", 2, "B", fake_pair, qualify)


def test_bank_keeps_truth_separate_and_refuses_overwrite(tmp_path):
    bank = build_bank("validation", 2, "A", fake_pair, lambda pair: [])
    assert bank["signals"].shape == (8, 12, 2500)
    assert not np.array_equal(bank["signals"], bank["baselines"])
    target = tmp_path / "bank"
    hashes = save_bank(bank, target)
    loaded = load_bank(target)
    assert len(hashes) == 8
    assert np.array_equal(loaded["signals"], bank["signals"])
    assert loaded["signals"].flags.writeable is False
    with pytest.raises(FileExistsError):
        save_bank(bank, target)


def test_positive_weight_uses_training_schedule_frequencies():
    mask = np.zeros((2, 1, 100), dtype=bool)
    mask[0, :, :50] = True
    bank = {"support": mask}
    assert positive_class_weight(bank, np.array([[0, 1]])) == 3.0
    assert positive_class_weight(bank, np.array([[0, 0]])) == 1.0
    with pytest.raises(ValueError, match="support and background"):
        positive_class_weight(bank, np.array([[1, 1]]))
