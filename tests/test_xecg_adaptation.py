"""Loss semantics, train-only streams, and exact step-boundary SSL recovery."""

import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from ecg_experiment.xecg_adaptation import (
    AdaptationConfig,
    AdaptationModel,
    ShuffledStream,
    coding_rate,
    contiguous_masks,
    encode_tokens,
    gram_loss,
    rarity_weights,
    ssl_parameter_groups,
    update_ema,
    visible_loss,
)
from scripts.experiments.run_xecg_adaptation import (
    SSL_ARTIFACTS,
    diagnostic_indices,
    load_ssl_resume,
    save_ssl_resume,
    ssl_completion,
    ssl_update,
)

RELEASED_WEIGHTS = Path(__file__).resolve().parents[1] / "third_party/checkpoints/xecg/model.safetensors"

class TinyCore(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Context mixing lets masked input influence visible features as xECG does.
        return self.dropout(torch.tanh(self.linear(x + 0.2 * x.mean(dim=1, keepdim=True))))


class TinyEncoder(nn.Module):
    patch_size = 25
    embedding_size = 6
    cls_type = "avg"

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12 * 25, self.embedding_size)
        self.mask_token = nn.Parameter(torch.arange(self.embedding_size).float() / 10)
        self.core = TinyCore(self.embedding_size)

    def patch_embedding(self, x):
        return self.projection(x.reshape(len(x), -1, 25 * 12))

    def get_padding_mask(self, x):
        padding = (x.abs().sum(dim=-1) == 0).reshape(len(x), -1, 25)[:, :, 0, None]
        return padding.expand(-1, -1, self.embedding_size)

    def pooling(self, tokens, padding):
        return tokens.masked_fill(padding, 0).sum(dim=1) / (~padding).sum(dim=1).clamp_min(1), tokens

    def forward(self, signal):
        return self.pooling(self.core(self.patch_embedding(signal)), self.get_padding_mask(signal))


def test_masks_replace_embedding_and_preserve_real_positions():
    encoder = TinyEncoder().eval()
    signal = torch.randn(2, 1000, 12)
    masks = contiguous_masks(2, 40, 8, torch.Generator().manual_seed(42))
    assert torch.equal(masks.sum(-1), torch.full((2, 2), 8))
    for record in range(2):
        for view in range(2):
            positions = torch.where(masks[record, view])[0]
            assert torch.equal(positions[1:] - positions[:-1], torch.ones(7, dtype=torch.long))
    seen = []
    hook = encoder.core.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach().clone()))
    _, tokens = encode_tokens(encoder, signal, masks[:, 0])
    hook.remove()
    patches = encoder.patch_embedding(signal)
    torch.testing.assert_close(seen[0][masks[:, 0]], encoder.mask_token.expand(16, -1))
    torch.testing.assert_close(seen[0][~masks[:, 0]], patches[~masks[:, 0]])
    assert tokens.shape == (2, 40, 6)


def test_coding_rate_determinant_lemma_and_gradient():
    torch.manual_seed(5)
    features = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    epsilon = 0.05
    actual = coding_rate(features, epsilon)
    z = F.normalize(features, dim=-1, eps=1e-8)
    expected = -0.5 * epsilon * (4 / (7 * 4)) ** 0.5 * torch.linalg.slogdet(
        torch.eye(7, dtype=torch.float64) + 7 / (4 * epsilon) * z.T @ z)[1]
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(actual, features, retain_graph=True)[0],
                               torch.autograd.grad(expected, features)[0], rtol=1e-9, atol=1e-11)


def test_gram_and_visible_losses_exclude_hidden_positions_and_tie_weights():
    torch.manual_seed(8)
    student = torch.randn(2, 10, 6, requires_grad=True)
    frozen = torch.randn(2, 10, 6, requires_grad=True)
    visible = torch.ones(2, 10, dtype=torch.bool)
    visible[:, 3:5] = False
    loss = gram_loss(student, frozen, visible) + visible_loss(student, frozen, visible)
    gradient, = torch.autograd.grad(loss, student)
    assert gradient[~visible].count_nonzero() == 0
    assert gradient[visible].abs().sum() > 0
    assert frozen.grad is None
    changed = student.detach().clone()
    changed[~visible] += 1000
    torch.testing.assert_close(gram_loss(changed, frozen, visible), gram_loss(student, frozen, visible))
    equal_tokens = torch.ones(2, 10, 6, requires_grad=True)
    weights, rarity = rarity_weights(equal_tokens)
    assert not weights.requires_grad
    assert not rarity.requires_grad
    torch.testing.assert_close(weights, torch.full((2, 10), 1.5))
    torch.testing.assert_close(visible_loss(student, frozen, visible, weights),
                               visible_loss(student, frozen, visible))
    varied, _ = rarity_weights(frozen)
    assert varied.min() >= 1
    assert varied.max() <= 2


def test_all_arms_stop_teacher_gradients_and_ema_preserves_anchor():
    config = replace(AdaptationConfig(), mask_tokens=2, microbatch_size=2, effective_batch_size=2)
    for arm in ("a", "b", "c", "d"):
        torch.manual_seed(13)
        model = AdaptationModel(TinyEncoder()).train()
        anchor = {key: value.clone() for key, value in model.anchor.state_dict().items()}
        signal = torch.randn(2, 250, 12)
        masks = contiguous_masks(2, 10, 2, torch.Generator().manual_seed(77))
        loss, terms = model(signal, masks, arm, config)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.student.projection.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.ema.parameters())
        assert all(p.grad is None for p in model.anchor.parameters())
        assert set(terms) == {"masked", "pooled", "expansion", "gram", "visible_uniform", "visible_weighted"}
        with torch.no_grad():
            model.student.mask_token.add_(1)
        before = model.ema.mask_token.clone()
        update_ema(model.ema, model.student, 0.9)
        torch.testing.assert_close(model.ema.mask_token, before + 0.1)
        for key, value in model.anchor.state_dict().items():
            torch.testing.assert_close(value, anchor[key], rtol=0, atol=0)


def setup(seed, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = AdaptationModel(TinyEncoder())
    optimizer = torch.optim.AdamW(ssl_parameter_groups(model.student, config), lr=config.learning_rate)
    return model, optimizer, ShuffledStream(7, 42), torch.Generator().manual_seed(10043)


def test_exact_ssl_resume_including_ema_optimizer_stream_masks_and_rng(tmp_path):
    config = replace(AdaptationConfig(), updates=4, effective_batch_size=4, microbatch_size=2,
                     mask_tokens=2, warmup_updates=1, learning_rate=0.01)
    views = np.random.default_rng(45).normal(size=(7, 250, 12)).astype(np.float32)
    fingerprint = {"arm": "d", "config": asdict(config), "source": "tiny"}
    continuous = setup(11, config)
    trace, expected_history = "0" * 64, []
    for step in range(4):
        row, trace, _ = ssl_update(*continuous[:2], views, *continuous[2:], config, "d", step, "cpu", trace)
        expected_history.append(row)
    expected_draws = (random.random(), np.random.random(), torch.rand(1))

    interrupted = setup(11, config)
    interrupted_trace, history = "0" * 64, []
    for step in range(2):
        row, interrupted_trace, _ = ssl_update(*interrupted[:2], views, *interrupted[2:], config, "d", step,
                                               "cpu", interrupted_trace)
        history.append(row)
    path = tmp_path / "resume.pt"
    save_ssl_resume(path, fingerprint, *interrupted, 2, history, interrupted_trace, 1.0)
    resumed = setup(999, config)
    # The frozen release must be reloaded from the same source on restoration.
    resumed[0].anchor.load_state_dict(interrupted[0].anchor.state_dict())
    with pytest.raises(ValueError, match="arm, sources, or protocol"):
        load_ssl_resume(path, {**fingerprint, "arm": "c"}, *resumed)
    saved = load_ssl_resume(path, fingerprint, *resumed)
    resumed_trace = saved["trace_sha256"]
    for step in range(saved["step"], 4):
        row, resumed_trace, _ = ssl_update(*resumed[:2], views, *resumed[2:], config, "d", step, "cpu",
                                           resumed_trace)
        history.append(row)
    assert trace == resumed_trace
    assert history == expected_history
    for name, value in continuous[0].state_dict().items():
        torch.testing.assert_close(value, resumed[0].state_dict()[name], rtol=0, atol=0)
    assert random.random() == expected_draws[0]
    assert np.random.random() == expected_draws[1]
    torch.testing.assert_close(torch.rand(1), expected_draws[2], rtol=0, atol=0)
    assert [item.name for item in tmp_path.iterdir()] == ["resume.pt"]


def test_cpu_resume_rejects_saved_cuda_state(tmp_path, monkeypatch):
    config = replace(AdaptationConfig(), effective_batch_size=2, microbatch_size=2, mask_tokens=2)
    state = setup(3, config)
    path = tmp_path / "resume.pt"
    save_ssl_resume(path, {}, *state, 0, [], "0" * 64, 0.0)
    saved = torch.load(path, weights_only=True)
    saved["rng"]["cuda"] = [torch.zeros(8, dtype=torch.uint8)]
    torch.save(saved, path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="without CUDA"):
        load_ssl_resume(path, {}, *setup(4, config))


def test_ssl_completion_verifies_fingerprint_and_artifacts(tmp_path):
    assert ssl_completion(tmp_path, {"arm": "a"}) is None
    for name in SSL_ARTIFACTS:
        (tmp_path / name).write_text(name)
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in SSL_ARTIFACTS}
    (tmp_path / "complete.json").write_text(json.dumps({"fingerprint": {"arm": "a"}, "sha256": hashes}))
    assert ssl_completion(tmp_path, {"arm": "a"})["sha256"] == hashes
    with pytest.raises(ValueError, match="differs from requested inputs"):
        ssl_completion(tmp_path, {"arm": "b"})
    (tmp_path / "encoder.pt").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ssl_completion(tmp_path, {"arm": "a"})


def test_diagnostic_indices_are_fixed_and_source_balanced():
    rows = [{"source": "ptbxl" if i % 3 else "mimic"} for i in range(30)]
    state = torch.get_rng_state()
    first = diagnostic_indices(rows)
    assert torch.equal(first, diagnostic_indices(rows))
    assert torch.equal(torch.get_rng_state(), state)
    assert [rows[i]["source"] for i in first.tolist()] == ["ptbxl"] * 4 + ["mimic"] * 4
    with pytest.raises(ValueError, match="four training records per source"):
        diagnostic_indices(rows[:9])


def test_sampler_and_mask_stream_do_not_depend_on_model_rng():
    data_a, data_b = ShuffledStream(9), ShuffledStream(9)
    mask_a, mask_b = torch.Generator().manual_seed(71), torch.Generator().manual_seed(71)
    for _ in range(5):
        first = data_a.take(7), contiguous_masks(7, 40, 8, mask_a)
        torch.randn(300)
        second = data_b.take(7), contiguous_masks(7, 40, 8, mask_b)
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])


@pytest.mark.skipif(not RELEASED_WEIGHTS.exists(), reason="released encoder unavailable")
def test_released_small_forward_equivalence_and_token_alignment():
    from ecg_experiment.xecg import load_xecg

    encoder = load_xecg(drop_path_prob=0).eval()
    torch.manual_seed(54)
    signal = torch.randn(1, 150, 12)
    with torch.no_grad():
        expected_pool, expected_tokens = encoder(signal)
        pooled, tokens = encode_tokens(encoder, signal)
    torch.testing.assert_close(pooled, expected_pool, rtol=0, atol=0)
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    assert torch.isfinite(tokens).all()
    # The actual released wrapper performs eight flips, restoring position order.
    blocks, norm = encoder.core.model.blocks, encoder.core.model.post_blocks_norm
    encoder.core.model.blocks = nn.ModuleList(nn.Identity() for _ in range(9))
    encoder.core.model.post_blocks_norm = nn.Identity()
    positions = torch.arange(40).float().reshape(1, 40, 1).expand(1, 40, 1024)
    torch.testing.assert_close(encoder.core(positions), positions, rtol=0, atol=0)
    encoder.core.model.blocks, encoder.core.model.post_blocks_norm = blocks, norm
