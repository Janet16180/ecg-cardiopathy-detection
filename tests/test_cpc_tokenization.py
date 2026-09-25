"""Scientific and resume invariants for the continued CPC tokenization arms."""

import numpy as np
import torch

from ecg_experiment.cpc import CPCPretrainer, cpc_loss
from ecg_experiment.cpc_tokenization import (
    CLUSTERS,
    TokenizationPretrainer,
    fit_kmeans,
    future_cluster_loss,
    nearest_cluster,
    snapshot_teacher_convs,
)
from ecg_experiment.reproducibility import cpu_state, seed_everything
from scripts.experiments.run_cpc_tokenization import (
    classifier_with_matched_head,
    load_bootstrap_weights,
    reset_cluster_heads,
)


def test_cluster_assignment_uses_frozen_teacher_and_euclidean_centers():
    centers = torch.zeros(CLUSTERS, 256)
    centers[1, 0] = 1
    centers[2, 0] = -1
    teacher_tokens = torch.zeros(1, 2, 79, 256)
    teacher_tokens[..., 0] = 1
    labels = nearest_cluster(teacher_tokens, centers)
    assert torch.all(labels == 1)
    teacher_tokens.requires_grad_()
    contexts = torch.randn(1, 2, 79, 256, requires_grad=True)
    heads = torch.nn.ModuleList(torch.nn.Linear(256, CLUSTERS) for _ in range(3))
    loss = future_cluster_loss(teacher_tokens, contexts, centers, heads)
    loss.backward()
    assert contexts.grad.abs().sum() > 0
    assert teacher_tokens.grad is None
    assert all(head.weight.grad.abs().sum() > 0 for head in heads)


def test_common_query_coverage_and_future_mask():
    tokens = torch.randn(2, 2, 79, 256, requires_grad=True)
    contexts = torch.randn_like(tokens, requires_grad=True)
    heads = torch.nn.ModuleList(torch.nn.Linear(256, 256, bias=False) for _ in range(3))
    loss = cpc_loss(tokens, contexts, heads, first_query=24)
    assert torch.isfinite(loss)
    loss.backward()
    assert contexts.grad[:, :, :24].abs().sum() == 0
    assert contexts.grad[:, :, 24:].abs().sum() > 0


def test_all_arms_restore_identical_cpc_heads_and_classifier_head():
    seed_everything(42)
    source = CPCPretrainer()
    enc_state, head_state = cpu_state(source.encoder), cpu_state(source.heads)
    beat_boundaries = torch.zeros(1, 2, 79, dtype=torch.bool)
    beat_boundaries[:, :, 15::16] = True
    beat_boundaries[:, :, -1] = True
    signal = torch.randn(1, 12, 2500)
    classifier_heads = []
    for variant in ("continuation", "clusteraux", "fixedchunk", "beatchunk", "learnedchunk"):
        model = TokenizationPretrainer(variant)
        load_bootstrap_weights(model, enc_state, head_state)
        for key, value in model.heads.state_dict().items():
            torch.testing.assert_close(value, head_state[key], atol=0, rtol=0)
        teacher = snapshot_teacher_convs(model.encoder) if variant == "clusteraux" else None
        centers = torch.randn(CLUSTERS, 256) if teacher is not None else None
        loss, details = model(signal, beat_boundaries, centers, teacher)
        assert torch.isfinite(loss)
        assert np.isfinite(list(details.values())).all()
        classifier = classifier_with_matched_head(variant, "cpu")
        classifier.encoder.load_state_dict(model.encoder.state_dict())
        assert classifier(signal, beat_boundaries).shape == (1,)
        classifier_heads.append(cpu_state(classifier.head))
    for state in classifier_heads[1:]:
        for name in state:
            torch.testing.assert_close(state[name], classifier_heads[0][name], atol=0, rtol=0)


def test_auxiliary_round_reset_preserves_main_optimizer_and_rng():
    seed_everything(42)
    model = TokenizationPretrainer("clusteraux")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(42)
    signal = torch.randn(2, 12, 2500)
    teacher = snapshot_teacher_convs(model.encoder)
    loss, _ = model(signal, centers=torch.randn(CLUSTERS, 256), teacher_convs=teacher)
    loss.backward()
    optimizer.step()
    before_main = [value.detach().clone() for value in model.heads.parameters()]
    before_main_state = [optimizer.state[parameter]["step"].clone() for parameter in model.heads.parameters()]
    expected_rng = torch.get_rng_state().clone()
    expected_loader = generator.get_state().clone()
    reset_cluster_heads(model, optimizer, generator, seed=43)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    torch.testing.assert_close(generator.get_state(), expected_loader, atol=0, rtol=0)
    for parameter, expected, step in zip(model.heads.parameters(), before_main, before_main_state,
                                         strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        torch.testing.assert_close(optimizer.state[parameter]["step"], step, atol=0, rtol=0)
    assert all(parameter not in optimizer.state for parameter in model.cluster_heads.parameters())


def test_kmeans_fit_is_deterministic_and_uses_unit_inputs():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(128, 256)).astype("float32")
    first, info = fit_kmeans(features, seed=42)
    second, _ = fit_kmeans(features.copy(), seed=42)
    np.testing.assert_allclose(first, second, atol=0)
    assert first.shape == (CLUSTERS, 256)
    assert info["samples"] == 128
    assert info["clusters"] == CLUSTERS
