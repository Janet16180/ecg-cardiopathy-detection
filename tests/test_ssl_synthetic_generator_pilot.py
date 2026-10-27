"""Fixed pilot identity and condition allocation tests without real data."""

import hashlib
from collections import Counter

import pytest

from ecg_experiment.ssl_synthetic_generator_pilot_qualification import (
    _role_integrity,
    pilot_cases,
    pilot_seed,
    qualify_pilot_domain,
)


def test_declared_seed_rule_and_role_independence():
    expected = int.from_bytes(hashlib.sha256(b"ssl_pilot_v1:synthetic:train:0").digest()[:8], "big")
    assert pilot_seed("train", 0) == expected
    assert pilot_seed("validation", 0) != expected
    assert pilot_seed("evaluation", 0) != expected
    integrity = _role_integrity(pilot_cases())
    assert integrity == {
        "counts": {"train": 500, "validation": 128, "evaluation": 256},
        "unique_latent_seeds": 884,
        "all_disjoint": True,
    }


def test_full_pilot_counts_balance_and_related_renders():
    cases = pilot_cases()
    assert len(cases) == 3060
    assert Counter(row["cohort"] for row in cases) == {
        "train/A": 500,
        "validation/A": 512,
        "evaluation/A": 1024,
        "evaluation/B": 1024,
    }
    training = [row for row in cases if row["role"] == "train"]
    assert sum(row["target"] for row in training) == 375
    assert sum(not row["target"] and row["nuisance"] for row in training) == 63
    assert sum(not row["target"] and not row["nuisance"] for row in training) == 62
    assert all(row["profile"] == "A" for row in cases if row["role"] != "evaluation")
    for row in training[:375]:
        assert row["mode"] == ("focal" if row["index"] % 2 == 0 else "persistent")
        assert row["nuisance"] == bool(row["seed"] & 1)
    selected = [row for row in cases if row["role"] == "evaluation" and row["index"] == 17]
    assert len(selected) == 8
    assert len({row["seed"] for row in selected}) == 1
    positives = [row for row in selected if row["target"]]
    assert len({row["nuisance"] for row in positives}) == 1
    assert {(row["profile"], row["mode"]) for row in positives} == {
        ("A", "focal"),
        ("A", "persistent"),
        ("B", "focal"),
        ("B", "persistent"),
    }


@pytest.mark.parametrize(("role", "index"), [("test", 0), ("train", -1)])
def test_bad_seed_requests_raise(role, index):
    with pytest.raises(ValueError, match="declared role"):
        pilot_seed(role, index)


def test_existing_admission_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        qualify_pilot_domain(tmp_path)
    assert not list(tmp_path.iterdir())
