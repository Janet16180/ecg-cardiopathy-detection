"""CPU checks for Experiment010's lead protocol and resumable runner."""

import json
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import cpc_crosslead as objective
from ecg_experiment.cpc import CPCEncoder, CPCPretrainer
from scripts import run_cpc_crosslead as run
from scripts import run_cpc_experiment as base
from scripts.run_cpc_tokenization import bootstrap


def test_missing_leads_are_zeroed_after_normalization():
    normalized = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1).expand(2, 12, 17) + 2
    for group in (objective.GROUP_A, objective.GROUP_B):
        view = objective.lead_view(normalized, group)
        for lead in range(12):
            torch.testing.assert_close(view[:, lead], normalized[:, lead] if lead in group else torch.zeros_like(view[:, lead]))
    assert set(objective.GROUP_A).isdisjoint(objective.GROUP_B)
    assert set(objective.GROUP_A + objective.GROUP_B) == {0, 1, 6, 7, 8, 9, 10, 11}


def test_crosslead_swaps_targets_only(monkeypatch):
    calls = []

    class Encoder(nn.Module):
        def forward(self, signal):
            value = signal[:, :, 0].sum(dim=1).reshape(-1, 1, 1, 1)
            tokens = value.expand(-1, 2, 4, 256).clone()
            return tokens, tokens + 100

    def recorded_loss(tokens, contexts, heads):
        calls.append((float(tokens[0, 0, 0, 0]), float(contexts[0, 0, 0, 0])))
        return tokens.mean() + contexts.mean()

    monkeypatch.setattr(objective, "cpc_loss", recorded_loss)
    signal = torch.arange(1, 13, dtype=torch.float32).reshape(1, 12, 1).expand(1, 12, 2500)
    results = {}
    for variant in objective.VARIANTS:
        model = objective.CrossLeadPretrainer(variant)
        model.encoder = Encoder()
        calls.clear()
        loss, _ = model(signal)
        results[variant] = (list(calls), float(loss))
    full = 78.0
    a = sum(i + 1 for i in objective.GROUP_A)
    b = sum(i + 1 for i in objective.GROUP_B)
    assert results["native"][0] == [(full, full + 100), (a, a + 100), (b, b + 100)]
    assert results["withinlead"][0] == [(full, full + 100), (a, a + 100), (b, b + 100)]
    assert results["crosslead"][0] == [(full, full + 100), (b, a + 100), (a, b + 100)]
    assert results["native"][1] == pytest.approx(full * 2 + 100)


def test_encoder_is_causal_with_independent_halves():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    encoder = CPCEncoder().eval()
    signal = torch.randn(1, 12, 2500)
    changed = signal.clone()
    changed[:, :, 900:] += torch.randn_like(changed[:, :, 900:]) * 10
    with torch.no_grad():
        tokens, contexts = encoder(signal)
        changed_tokens, changed_contexts = encoder(changed)
    torch.testing.assert_close(tokens[:, 0, :20], changed_tokens[:, 0, :20], atol=0, rtol=0)
    torch.testing.assert_close(contexts[:, 0, :20], changed_contexts[:, 0, :20], atol=0, rtol=0)
    assert not torch.equal(tokens[:, 0, -1], changed_tokens[:, 0, -1])


def test_strict_epoch20_bootstrap_and_source_hashes(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    source = CPCPretrainer()
    epoch = tmp_path / "cpc_ssl"
    epoch.mkdir()
    normalized = epoch.parent / "normalization.json"
    normalized.write_text("{}")
    fingerprint = "completed004"
    model_state = base.cpu_state(source)
    base.atomic_torch(epoch / "epoch_state.pt", {"fingerprint": fingerprint, "epoch": 20, "model": model_state})
    base.atomic_torch(epoch / "encoder.pt", {"fingerprint": fingerprint, "variant": "cpc",
                       "epochs": 20, "encoder": base.cpu_state(source.encoder)})
    original_path = str((tmp_path / "cache.bin").resolve())
    base.atomic_json(epoch / "config.json", {"fingerprint": fingerprint,
                      "inputs": {"cache": {original_path: "abc"}}})
    args = Namespace(bootstrap_dir=epoch, manifest_dir=tmp_path)
    encoder, heads, _ = bootstrap(args, {original_path: "abc"})
    assert len(heads) == 3 and encoder.keys() == source.encoder.state_dict().keys()
    with pytest.raises(ValueError, match="input changed"):
        bootstrap(args, {original_path: "changed"})
    final = torch.load(epoch / "encoder.pt", weights_only=True)
    final["encoder"][next(iter(encoder))] += 1
    base.atomic_torch(epoch / "encoder.pt", final)
    with pytest.raises(ValueError, match="differs"):
        bootstrap(args, {original_path: "abc"})
    (tmp_path / "cache.bin").write_bytes(b"x")
    monkeypatch.setattr(run.base, "make_source_hashes", lambda pool, manifest: {})
    hashes = run.source_hashes(args, object())
    imported = str((run.ROOT / "scripts/run_cpc_tokenization.py").resolve())
    assert hashes[imported] == base.digest_file(imported)
    assert str(normalized.resolve()) in hashes


class TinyPretrainer(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.encoder = nn.Linear(3, 3)
        self.heads = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(3)])
        self.dropout = nn.Dropout(0.3)

    def load_bootstrap(self, encoder, heads):
        self.encoder.load_state_dict(encoder)
        self.heads.load_state_dict(heads)

    def forward(self, signal):
        hidden = self.dropout(self.encoder(signal))
        loss = sum((head(hidden) ** 2).mean() for head in self.heads)
        return loss, {key: float(loss.detach()) for key in (
            "full_cpc", "aux_a", "aux_b", "token_variance", "adjacent_token_cosine",
            "view_a_token_variance", "view_b_token_variance")}


def test_epoch_resume_reproduces_shuffle_and_dropout(tmp_path):
    torch.set_num_threads(1)
    dataset = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10

    def setup():
        base.seed_all(42)
        model = TinyPretrainer("native")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        generator = torch.Generator().manual_seed(42)
        return model, optimizer, generator

    def epoch(model, optimizer, generator):
        for indices in torch.randperm(len(dataset), generator=generator).split(2):
            run.checked_step(model, optimizer, dataset[indices])

    uninterrupted = setup()
    epoch(*uninterrupted)
    epoch(*uninterrupted)
    expected = base.cpu_state(uninterrupted[0])
    first = setup()
    epoch(*first)
    run.save_epoch(tmp_path, "test-fp", 1, *first, [{"epoch": 1}])
    resumed = setup()
    start, history = run.resume_or_new(tmp_path, "test-fp", *resumed)
    assert start == 1 and history == [{"epoch": 1}]
    epoch(*resumed)
    for key, value in expected.items():
        torch.testing.assert_close(value, resumed[0].state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="fingerprint"):
        run.resume_or_new(tmp_path, "changed", *setup())


def test_profile_roundtrip_restores_dropout_rng(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(run, "CrossLeadPretrainer", TinyPretrainer)
    base.seed_all(123)
    model = TinyPretrainer("native")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(42)
    signal = torch.randn(2, 3)
    run.checked_step(model, optimizer, signal)  # Populate AdamW moments first.
    assert run.profile_roundtrip(model, optimizer, signal, "native", generator)


def test_all_ssl_precede_all_six_transfers(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(run, "CrossLeadPretrainer", TinyPretrainer)
    monkeypatch.setattr(run.base, "Pool", lambda directory: Namespace(train_rows=[{}] * 14))
    monkeypatch.setattr(run, "source_hashes", lambda args, pool: {"synthetic": "fixed"})
    base.seed_all(42)
    bootstrap_model = TinyPretrainer("native")
    monkeypatch.setattr(run, "bootstrap", lambda args, hashes: (
        base.cpu_state(bootstrap_model.encoder), base.cpu_state(bootstrap_model.heads), {}))
    monkeypatch.setattr(run, "fixed_normalization", lambda pool, args, hashes, config: (
        np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)))
    monkeypatch.setattr(run, "ssl_fingerprint", lambda args, pool, mean, std, hashes, variant: (
        f"{variant}-fixed", {"variant": variant}))
    batches = [(torch.randn(2, 3), None, None) for _ in range(7)]
    monkeypatch.setattr(run.base, "loader", lambda *args: batches)
    transfers = []

    def fake_transfer(args, pool, mean, std, hashes, variant, budget):
        assert all((args.output_dir / f"{arm}_ssl/encoder.pt").exists() for arm in run.VARIANTS)
        transfers.append((variant, budget))

    monkeypatch.setattr(run.base, "fine_tune", fake_transfer)
    args = Namespace(stage="all", variant="all", labels="all", cache_dir=tmp_path,
                     manifest_dir=tmp_path, bootstrap_dir=tmp_path, output_dir=tmp_path / "out",
                     device="cpu", threads=1, batch_size=2, ssl_batch_size=2, ssl_epochs=10,
                     epochs=40, patience=8, bootstrap=5, profile_updates=5)
    run.run(args)
    assert set(transfers) == {(arm, budget) for arm in run.VARIANTS for budget in ("1", "0.1")}
    for arm in run.VARIANTS:
        history = json.loads((args.output_dir / f"{arm}_ssl/history.json").read_text())
        assert len(history) == 10 and history[-1]["epoch"] == 10
