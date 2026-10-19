"""Checks for NTT affinity geometry, label permutations and fitting-only cluster selection."""

import time
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ecg_experiment.clustering.reference_geometry import (
    affinity_views,
    candidate,
    fit_model,
    fit_view,
    primary_decision,
    support_prototypes,
)
from ecg_experiment.same_wave_ntt04 import FINDINGS, SEEDS


def test_affinity_aggregation_keeps_patient_and_bank_boundaries():
    """Region means/maxima/std are summarized per patient and bank before bank averaging."""
    cache = {
        "qrs": np.array([[1.0, 0], [0, 1.0]]),
        "region_vectors": np.array([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0], [0, 1.0], [0, 1.0]]),
        "region_offsets": np.array([0, 3, 6]),
    }
    bank = np.array([[1.0, 0], [0, 1.0], [-1.0, 0]])
    prototypes = np.stack([bank, bank, bank])
    record, regional, single = affinity_views(cache, prototypes)
    np.testing.assert_allclose(record, [[1, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(regional[0, :3], [2 / 3, 1 / 3, -2 / 3])
    np.testing.assert_allclose(regional[0, 3:6], [1, 1, 0])
    np.testing.assert_allclose(regional[0, 6:], np.array([2, 2, 2]) ** 0.5 / 3)
    np.testing.assert_allclose(regional[1], [0, 1, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(single[0], regional)


def test_support_label_shuffle_rebuilds_geometry_not_column_names():
    """Permuting individual support labels changes pairwise affinity distances."""
    selected = {"support_draws": []}
    features = {}
    rng = np.random.default_rng(12)
    for bank, seed in enumerate(SEEDS):
        for i, finding in enumerate(FINDINGS):
            rows = []
            for j in range(3):
                key = bank * 100 + i * 3 + j
                rows.append({"ecg_id": key, "patient_id": key})
                v = rng.normal(size=8)
                v /= np.linalg.norm(v)
                features[str(key)] = {"native_qrs": v}
            selected["support_draws"].append({"seed": seed, "finding": finding, "supports": rows})
    true = support_prototypes(selected, features)
    shuffled = support_prototypes(selected, features, 20261019)
    np.testing.assert_allclose(np.linalg.norm(shuffled, axis=2), 1)
    assert not np.allclose(true @ true.transpose(0, 2, 1), shuffled @ shuffled.transpose(0, 2, 1))
    np.testing.assert_allclose(shuffled, support_prototypes(selected, features, 20261019))
    x = rng.normal(size=(10, 8))
    original = x @ true[0].T
    rotated = original[:, [1, 2, 0]]
    np.testing.assert_allclose(
        np.linalg.norm(original[:, None] - original[None, :], axis=2),
        np.linalg.norm(rotated[:, None] - rotated[None, :], axis=2),
    )


def test_centering_and_pca_never_adapt_to_anchor_values():
    """Anchor outliers cannot affect the fitted geometry of unchanged fitting rows."""
    rng = np.random.default_rng(14)
    x = rng.normal(size=(50, 20))
    fitting = np.arange(40)
    changed = x.copy()
    changed[40:] += 1000
    for view in ("native_qrs", "ntt_regions", "physical"):
        _, first = fit_view(x, fitting, view)
        _, second = fit_view(changed, fitting, view)
        np.testing.assert_allclose(first[:40], second[:40], atol=1e-10)
    state, affinity = fit_view(x[:, :3], fitting, "ntt_affinity")
    assert state["dimensions"] == 3
    np.testing.assert_allclose(affinity[:40].mean(axis=0), 0, atol=1e-15)


def test_collapsed_anchor_fails_even_when_resampling_ari_is_perfect():
    """ARI=1 from a single anchor cluster must never admit a configuration."""
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(-5, 0.01, (40, 2)), rng.normal(5, 0.01, (40, 2)), rng.normal(-5, 0.01, (20, 2))]
    fitting = np.arange(80)
    anchors = np.arange(80, 100)
    samples = [fitting] * 9
    states = [{"coordinates": x}] * 9
    row, _ = candidate("ntt_regions", "kmeans", "none", 2, states, samples, anchors, time.perf_counter() + 30)
    assert row["stability"]["minimum"] == 1
    assert not row["eligible"]
    assert not row["conditions"]["all_anchors_five_percent_occupancy"]


def test_primary_gate_cannot_be_rescued_by_other_views():
    """Only the prespecified native and regional winners and bank checks can pass."""
    labels = np.array([0, 0, 1, 1])
    rows = {}
    assignments = {}
    for view in ("native_qrs", "native_global", "ntt_affinity", "ntt_regions", "physical"):
        rows[view] = {"key": view, "eligible": view != "ntt_regions"}
        assignments[view] = {"anchor": labels}
    checks = [{"converged": True, "fit_balanced": True, "anchor_balanced": True}] * 3
    result = primary_decision(rows, assignments, [labels, 1 - labels, labels], checks)
    assert adjusted_rand_score(labels, 1 - labels) == 1
    assert not result["primary_persistence_gate_passed"]
    assert not result["conditions"]["ntt_regions_eligible"]


def test_kmeans_iteration_exhaustion_is_not_convergence(monkeypatch):
    """A legacy helper's unconditional success cannot admit a capped k-means fit."""
    from ecg_experiment.clustering import reference_geometry as module

    model = SimpleNamespace(n_iter_=300, max_iter=300)
    monkeypatch.setattr(module, "fit_partition", lambda *args: (model, True))
    _, ok = fit_model(np.zeros((4, 2)), "kmeans", "none", 2, 1)
    assert not ok
    model.n_iter_ = 4
    _, ok = fit_model(np.zeros((4, 2)), "kmeans", "none", 2, 1)
    assert ok
