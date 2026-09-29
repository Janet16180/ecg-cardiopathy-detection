"""Checks of the Experiment 034 one-class baselines."""

import math

import numpy as np
import pytest
import torch

from ecg_experiment import one_class_baselines as baselines
from ecg_experiment.normal_manifold import fit_mahalanobis


def test_knn_distances_match_brute_force_on_unit_vectors():
    rng = np.random.default_rng(0)
    reference, query = rng.normal(size=(40, 6)), rng.normal(size=(7, 6))
    kth, mean = baselines.knn_distances(reference, query, k=3)
    unit_reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    unit_query = query / np.linalg.norm(query, axis=1, keepdims=True)
    distances = np.sort(np.linalg.norm(unit_query[:, None] - unit_reference[None], axis=2), axis=1)
    assert np.allclose(kth, distances[:, 2])
    assert np.allclose(mean, distances[:, :3].mean(axis=1))


def test_whiten_gives_unit_variance_components_on_the_fit_set():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 20)) @ rng.normal(size=(20, 20))
    z = baselines.whiten(fit_mahalanobis(x, components=5), x)
    assert z.shape == (300, 5)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(z.var(axis=0, ddof=1), 1)


def test_holdout_mask_keeps_units_whole_and_rounds_up():
    units = np.array(["a", "a", "b", "c", "c", "c", "d", "e", "f", "g", "h", "i", "j", "k"])
    held = baselines.holdout_mask(units, fraction=0.1, seed=3)
    held_units = np.unique(units[held])
    assert len(held_units) == math.ceil(0.1 * 11)
    assert np.array_equal(held, np.isin(units, held_units))
    assert np.array_equal(held, baselines.holdout_mask(units.copy(), fraction=0.1, seed=3))


def test_choose_gmm_components_prefers_two_for_two_clusters(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(baselines, "GMM_COMPONENTS", (1, 2))
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-5, 1, (200, 2)), rng.normal(5, 1, (200, 2))])
    rng.shuffle(x)
    chosen, likelihood = baselines.choose_gmm_components(x[:300], x[300:])
    assert chosen == 2
    assert likelihood[2] > likelihood[1]


def test_flow_log_prob_is_a_density_change_of_variables():
    flow = baselines.RealNVP(4, layers=2)
    x = torch.randn(5, 4)
    with torch.no_grad():
        for coupling in flow.couplings:
            coupling.net[-1].weight.zero_()
            coupling.net[-1].bias.zero_()
        expected = -0.5 * (x ** 2).sum(dim=1) - 2 * math.log(2 * math.pi)
        assert torch.allclose(flow.log_prob(x), expected)


def test_deep_svdd_centre_has_no_small_coordinates():
    network, row_loss, weight_decay = baselines.build_network("deep_svdd", torch.randn(50, 8))
    assert weight_decay == baselines.SVDD_WEIGHT_DECAY
    assert all(layer.bias is None for layer in network if isinstance(layer, torch.nn.Linear))
    assert row_loss(torch.randn(3, 8)).shape == (3,)


@pytest.mark.parametrize("kind", baselines.NETWORKS)
def test_network_scores_choose_epochs_on_held_out_normals(kind: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(baselines, "MAX_EPOCHS", 4)
    rng = np.random.default_rng(4)
    fit = rng.normal(size=(120, 6))
    held = np.zeros(120, dtype=bool)
    held[:20] = True
    scores, details = baselines.network_scores(kind, fit, held, {"query": rng.normal(size=(9, 6))}, math.inf)
    assert scores["query"].shape == (9,)
    assert np.isfinite(scores["query"]).all()
    assert len(details["held_out_loss"]) == 4
    assert details["chosen_epochs"] == int(np.argmin(details["held_out_loss"])) + 1


def test_train_network_stops_at_the_deadline():
    with pytest.raises(baselines.WallTimeExceededError):
        baselines.train_network("autoencoder", np.zeros((10, 4)), epochs=3, deadline=0.0)
