"""Sampling, objective, and resume checks for the matched word2vec-style CPC arms."""

import math

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812 - conventional alias

from ecg_experiment.cpc import HORIZONS
from ecg_experiment.cpc_word2vec import SampledCPCPretrainer, sampled_negative_indices, sampled_objective
from ecg_experiment.reproducibility import seed_everything
from scripts.experiments.run_cpc_word2vec import resume_or_new, save_epoch


def test_sampled_negatives_are_distant_same_half_and_reproducible():
    before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(11)
    first, valid, positives = sampled_negative_indices(79, 4, 3, generator, "cpu")
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    second, second_valid, second_positives = sampled_negative_indices(
        79, 4, 3, torch.Generator().manual_seed(11), "cpu")
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(valid, second_valid, atol=0, rtol=0)
    torch.testing.assert_close(positives, second_positives, atol=0, rtol=0)
    assert first.shape == (6, int(valid.sum()), 16)
    assert torch.all((first - positives.reshape(1, -1, 1)).abs() > 3)
    assert torch.all((first >= 0) & (first < 79))
    # Every sampled column indexes the same record's half in the loss gather.
    assert first.shape[0] == 2 * 3


def test_both_losses_match_reference_on_exact_same_sampled_scores():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randn(2, 2, 32, 256)
    contexts = torch.randn_like(tokens)
    heads = torch.nn.ModuleList(torch.nn.Linear(256, 256, bias=False) for _ in HORIZONS)
    state = generator.get_state()
    sampled_info, _ = sampled_objective(tokens, contexts, heads, "sampled_info", generator)
    generator.set_state(state)
    sgns, _ = sampled_objective(tokens, contexts, heads, "sgns", generator)
    generator.set_state(state)
    targets = F.normalize(tokens.reshape(4, 32, 256), dim=-1)
    queries = contexts.reshape(4, 32, 256)
    reference_info, reference_sgns = [], []
    for horizon, head in zip(HORIZONS, heads, strict=True):
        indices, valid, positives = sampled_negative_indices(32, horizon, 2, generator, "cpu")
        predictions = F.normalize(head(queries[:, valid]), dim=-1)
        scores = torch.bmm(predictions, targets.transpose(1, 2)) / 0.1
        positive = scores[:, torch.arange(len(positives)), positives]
        negative = scores.gather(2, indices)
        logits = torch.cat((positive.unsqueeze(-1), negative), dim=-1)
        reference_info.append(F.cross_entropy(logits.reshape(-1, 17),
                                              torch.zeros(logits.numel() // 17, dtype=torch.long)))
        reference_sgns.append((F.softplus(-positive) + F.softplus(negative).sum(-1)).mean())
    torch.testing.assert_close(sampled_info, torch.stack(reference_info).mean())
    torch.testing.assert_close(sgns, torch.stack(reference_sgns).mean())


def test_zero_score_oracle_distinguishes_negative_sum_from_mean():
    tokens = torch.randn(1, 2, 32, 256)
    contexts = torch.randn_like(tokens)
    heads = torch.nn.ModuleList(torch.nn.Linear(256, 256, bias=False) for _ in HORIZONS)
    for head in heads:
        torch.nn.init.zeros_(head.weight)
    info, _ = sampled_objective(tokens, contexts, heads, "sampled_info",
                                torch.Generator().manual_seed(1))
    sgns, _ = sampled_objective(tokens, contexts, heads, "sgns",
                                torch.Generator().manual_seed(1))
    torch.testing.assert_close(info, torch.tensor(math.log(17)), atol=1e-6, rtol=0)
    torch.testing.assert_close(sgns, torch.tensor(17 * math.log(2)), atol=1e-6, rtol=0)


def test_same_initialization_finite_gradients_and_collapse_metrics():
    torch.set_num_threads(1)
    signal = torch.randn(2, 12, 2500)
    states = []
    for variant in ("sampled_info", "sgns"):
        seed_everything(42)
        model = SampledCPCPretrainer(variant)
        states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        loss, details = model(signal, torch.Generator().manual_seed(7))
        assert torch.isfinite(loss)
        assert np.isfinite(list(details.values())).all()
        loss.backward()
        assert model.encoder.convs[0].conv.weight.grad.abs().sum() > 0
        assert model.encoder.context.weight_ih_l0.grad.abs().sum() > 0
        assert all(head.weight.grad.abs().sum() > 0 for head in model.heads)
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], atol=0, rtol=0)


def test_resume_restores_private_sampler_and_loader_rng(tmp_path):
    seed_everything(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    loader_generator = torch.Generator().manual_seed(42)
    sampler_generator = torch.Generator().manual_seed(424242)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    save_epoch(tmp_path, "matching", 1, model, optimizer,
               loader_generator, sampler_generator, [{"epoch": 1}])
    expected_loader = torch.rand(8, generator=loader_generator)
    expected_sampler = torch.rand(8, generator=sampler_generator)
    expected_global = torch.rand(8)
    rebuilt = torch.nn.Linear(2, 1)
    rebuilt_optimizer = torch.optim.AdamW(rebuilt.parameters())
    rebuilt_loader = torch.Generator().manual_seed(0)
    rebuilt_sampler = torch.Generator().manual_seed(0)
    epoch, history = resume_or_new(tmp_path, "matching", rebuilt, rebuilt_optimizer,
                                   rebuilt_loader, rebuilt_sampler)
    assert epoch == 1
    assert history == [{"epoch": 1}]
    torch.testing.assert_close(torch.rand(8, generator=rebuilt_loader), expected_loader, atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(8, generator=rebuilt_sampler), expected_sampler, atol=0, rtol=0)
    torch.testing.assert_close(torch.rand(8), expected_global, atol=0, rtol=0)
    for expected, actual in zip(model.parameters(), rebuilt.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert rebuilt_optimizer.state_dict()["state"]
