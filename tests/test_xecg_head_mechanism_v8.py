"""Scientific invariants for the fixed xECG head-only mechanism study."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from ecg_experiment import xecg_head_mechanism_v8 as head
from scripts.experiments.run_xecg_head_mechanism016_v8 import load_state, replay_check, save_state


def synthetic():
    rng = np.random.default_rng(16)
    x = rng.normal(size=(130, 3)) * [0.4, 2.0, 4.0] + [3.0, -5.0, 0.2]
    y = rng.integers(0, 2, size=len(x)).astype(np.float64)
    mu, scale = head.train_scaler(x)
    z = (x - mu) / scale
    raw = np.array([0.2, -0.1, 0.05, 0.3])
    return x, z, y, mu, scale, raw


def test_raw_standardized_logits_objective_and_gradient_equivalence():
    x, z, y, mu, scale, raw = synthetic()
    standardized = head.to_standardized(raw, mu, scale)
    np.testing.assert_allclose(head.to_raw(standardized, mu, scale), raw, atol=1e-15)
    np.testing.assert_allclose(x @ raw[:-1] + raw[-1], z @ standardized[:-1] + standardized[-1], atol=1e-14)
    raw_value, raw_gradient, _, _ = head.objective_raw(raw, x, y, mu, scale, 0.13)
    std_value, std_gradient, _, _ = head.objective_standardized(standardized, z, y, 0.13)
    assert raw_value == pytest.approx(std_value, abs=1e-14)
    np.testing.assert_allclose(raw_gradient, head.raw_gradient(std_gradient, mu, scale), atol=1e-13)
    tensor = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    data = torch.tensor(x, dtype=torch.float64)
    labels = torch.tensor(y, dtype=torch.float64)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(data @ tensor[:-1] + tensor[-1], labels)
    penalty = 0.13 / 2 * torch.sum((torch.tensor(scale) * tensor[:-1]) ** 2)
    (bce + penalty).backward()
    np.testing.assert_allclose(raw_gradient, tensor.grad.detach().numpy(), atol=1e-12)


def test_penalty_scaling_and_raw_coordinate_clipping_equivalence():
    x, z, y, mu, scale, raw = synthetic()
    lam = 1 / (len(x) * 0.01)
    _, raw_grad, _, penalty = head.objective_raw(raw, x, y, mu, scale, lam)
    expected = lam / 2 * np.sum((scale * raw[:-1]) ** 2)
    assert penalty == pytest.approx(expected)
    std = head.to_standardized(raw, mu, scale)
    _, std_grad, _, _ = head.objective_standardized(std, z, y, lam)
    raw_from_std = head.raw_gradient(std_grad, mu, scale)
    np.testing.assert_allclose(raw_grad, raw_from_std, atol=1e-11)
    factor_raw = min(1.0, 3.0 / np.linalg.norm(raw_grad))
    factor_std = min(1.0, 3.0 / np.linalg.norm(raw_from_std))
    assert factor_raw == factor_std
    np.testing.assert_allclose(head.raw_gradient(std_grad * factor_std, mu, scale), raw_grad * factor_raw)


def test_train_only_statistics_and_constant_dimension():
    train = np.array([[0.0, 4.0], [2.0, 4.0], [4.0, 4.0]])
    mu, scale = head.train_scaler(train)
    np.testing.assert_array_equal(mu, [2.0, 4.0])
    np.testing.assert_allclose(scale, [np.sqrt(8 / 3), 1.0])
    dev = np.full((2, 2), 1000.0)
    assert not np.array_equal(mu, np.vstack([train, dev]).mean(axis=0))


def test_seeded_common_batches_exact_coverage_and_63_record_tail():
    x, z, y, mu, scale, raw = synthetic()
    a = head.initial_state(raw, mu, scale, "A", 44, len(x))
    b = head.initial_state(raw, mu, scale, "B", 44, len(x))
    c = head.initial_state(raw, mu, scale, "C", 44, len(x))
    other = head.initial_state(raw, mu, scale, "C", 45, len(x))
    np.testing.assert_array_equal(a.order, b.order)
    np.testing.assert_array_equal(a.order, c.order)
    assert not np.array_equal(a.order, other.order)
    for state, arm in ((a, "A"), (b, "B"), (c, "C")):
        seen = []
        for _ in range(3):
            index, _ = head.step(state, x, z, y, arm, mu, scale, 0.02)
            seen.append(index)
        np.testing.assert_array_equal(np.sort(np.concatenate(seen)), np.arange(len(x)))
        assert state.next_index == len(x)
        assert len(seen[-1]) == 2
        assert state.updates == 3
    np.testing.assert_array_equal(a.order, b.order)
    np.testing.assert_array_equal(a.order, c.order)
    full = head.initial_state(np.zeros(4), mu, scale, "A", 44, head.N_TRAIN)
    sizes = [len(head._next_batch(full, head.N_TRAIN)) for _ in range(head.UPDATES_PER_EPOCH)]
    assert sizes == [64] * 239 + [63]
    assert full.next_index == head.N_TRAIN


def test_checkpoint_restores_state_rng_and_next_update(tmp_path):
    x, z, y, mu, scale, raw = synthetic()
    state = head.initial_state(raw, mu, scale, "C", 44, len(x))
    for _ in range(3):
        head.step(state, x, z, y, "C", mu, scale, 0.02)
    path = tmp_path / "head_state.npz"
    save_state(path, state)
    restored = load_state(path)
    np.testing.assert_array_equal(state.theta, restored.theta)
    assert state.rng_state == restored.rng_state
    data = {"x": x, "z": z, "y": y, "mu": mu, "scale": scale}
    assert replay_check(state, path, data, "C")["exact"]


def test_clean_cohort_join_detects_patient_label_and_order_errors(monkeypatch):
    monkeypatch.setattr(head, "N_TRAIN", 2)
    full = [
        {"ecg_id": "1", "target": "0", "patient_id": "p1"},
        {"ecg_id": "12722", "target": "1", "patient_id": "bad"},
        {"ecg_id": "2", "target": "1", "patient_id": "p2"},
    ]
    overlay = [
        {"record_id": "ptbxl:1", "target": "0", "patient_id": "ptbxl:p1"},
        {"record_id": "ptbxl:2", "target": "1", "patient_id": "ptbxl:p2"},
    ]
    dev = [{"ecg_id": "3"}]
    digest = hashlib.sha256(np.array([1, 2, 3], dtype=np.int64).tobytes()).hexdigest()
    assert head.clean_cohort_join(full, overlay, dev, digest) == [full[0], full[2]]
    with pytest.raises(ValueError, match="identities/order"):
        head.clean_cohort_join(full, overlay, dev, "0" * 64)
    overlay[0]["patient_id"] = "wrong"
    with pytest.raises(ValueError, match="patient"):
        head.clean_cohort_join(full, overlay, dev, digest)


def test_common_patient_draws_average_seed_differences_not_ensemble_auc():
    labels = np.array([0, 1, 0, 1, 0, 1])
    patients = np.array(["a", "b", "c", "d", "e", "f"])
    logits = {
        "c44": np.array([0, 3, 1, 4, 2, 5.0]),
        "b44": np.array([0, 2, 1, 3, 4, 5.0]),
        "c45": np.array([0, 2, 1, 5, 3, 4.0]),
        "b45": np.array([0, 4, 1, 3, 2, 5.0]),
    }
    result = head.paired_bootstrap(
        labels,
        patients,
        logits,
        {
            "seed44": ("c44", "b44"),
            "seed45": ("c45", "b45"),
            "mean": (("c44", "b44"), ("c45", "b45")),
        },
        draws=100,
    )
    observed = result["contrasts"]
    assert observed["mean"]["observed"] == pytest.approx(
        (observed["seed44"]["observed"] + observed["seed45"]["observed"]) / 2
    )
    assert result["valid"] + result["single_class_invalid"] == 100
