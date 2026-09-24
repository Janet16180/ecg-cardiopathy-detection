"""Causality and contrastive invariants of the local CPC experiment."""

import numpy as np
import torch

from ecg_experiment.cpc import CPCEncoder, CPCPretrainer, cmsc_loss, temporal_candidate_mask
from ecg_experiment.cpc_pool import Pool, PoolDataset, resume_or_new, save_epoch
from ecg_experiment.reproducibility import seed_everything


def test_future_perturbation_preserves_past_tokens_and_contexts():
    torch.set_num_threads(1)
    seed_everything(42)
    model = CPCEncoder().eval()
    signal = torch.randn(1, 12, 2500)
    changed = signal.clone()
    changed[:, :, 500:1250] += 100 * torch.randn_like(changed[:, :, 500:1250])
    with torch.no_grad():
        tokens, contexts = model(signal)
        changed_tokens, changed_contexts = model(changed)
    # A token at index 20 ends at raw index 320; everything from 500 onward is future.
    torch.testing.assert_close(tokens[:, 0, :21], changed_tokens[:, 0, :21], atol=0, rtol=0)
    torch.testing.assert_close(contexts[:, 0, :21], changed_contexts[:, 0, :21], atol=0, rtol=0)
    torch.testing.assert_close(tokens[:, 1], changed_tokens[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(contexts[:, 1], changed_contexts[:, 1], atol=0, rtol=0)


def test_half_context_reset_and_exact_token_shape():
    torch.set_num_threads(1)
    model = CPCEncoder().eval()
    original = torch.randn(2, 12, 2500)
    changed = original.clone()
    changed[:, :, :1250] = torch.randn_like(changed[:, :, :1250])
    with torch.no_grad():
        token_a, context_a = model(original)
        token_b, context_b = model(changed)
    assert token_a.shape == context_a.shape == (2, 2, 79, 256)
    torch.testing.assert_close(token_a[:, 1], token_b[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(context_a[:, 1], context_b[:, 1], atol=0, rtol=0)


def test_cpc_mask_uses_only_same_half_distant_negatives():
    mask, valid = temporal_candidate_mask(79, 4, "cpu")
    assert valid.tolist()[:4] == [False, False, False, True]
    assert valid[-4:].tolist() == [False] * 4
    target = 10 + 4
    assert mask[10, target]
    for offset in (-3, -2, -1, 1, 2, 3):
        assert not mask[10, target + offset]
    assert mask[10, target - 4] and mask[10, target + 4]


def test_cmsc_ignores_same_patient_off_diagonal():
    contexts = torch.randn(3, 2, 8, 256, requires_grad=True)
    loss = cmsc_loss(contexts, ["a", "a", "b"])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(contexts.grad).all()
    # With only one patient, each diagonal is the sole candidate, so loss is zero.
    identical_patients = cmsc_loss(contexts.detach(), ["a", "a", "a"])
    torch.testing.assert_close(identical_patients, torch.zeros(()), atol=0, rtol=0)


def test_cpc_and_hybrid_have_finite_nonzero_encoder_and_head_gradients():
    torch.set_num_threads(1)
    signal = torch.randn(2, 12, 2500)
    for hybrid in (False, True):
        seed_everything(42)
        model = CPCPretrainer(hybrid=hybrid)
        loss, details = model(signal, ["patient-1", "patient-2"])
        assert torch.isfinite(loss)
        assert np.isfinite(list(details.values())).all()
        loss.backward()
        assert model.encoder.convs[0].conv.weight.grad.abs().sum() > 0
        assert model.encoder.context.weight_ih_l0.grad.abs().sum() > 0
        assert all(head.weight.grad.abs().sum() > 0 for head in model.heads)


def test_pool_global_normalization_uses_only_training_rows(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    signals = np.zeros((3, 12, 2500), dtype=np.float32)
    signals[0] = 1
    signals[1] = 3
    signals[2] = 1000
    np.save(directory / "signals.npy", signals)
    np.save(directory / "ecg_ids.npy", np.array(["1", "2", "3"]))
    (directory / "rows.csv").write_text("ecg_id,patient_id,source,split\n1,a,ptbxl,train\n2,b,ptbxl,train\n3,c,ptbxl,test\n")
    (directory / "complete.json").write_text("{}")
    pool = Pool(directory)
    mean, std = pool.normalization(tmp_path / "out")
    np.testing.assert_allclose(mean, 2)
    np.testing.assert_allclose(std, 1)
    dataset = PoolDataset(pool, [pool.rows[2]], mean, std)
    assert torch.all(dataset[0][0] == 998)


def test_epoch_checkpoint_restores_optimizer_dropout_and_shuffle(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    dataset = TensorDataset(x)

    def create():
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.2),
                                    torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(42)
        return model, optimizer, generator

    def epoch(model, optimizer, generator):
        sequence = []
        for (signal,) in DataLoader(dataset, batch_size=3, shuffle=True, generator=generator):
            sequence.extend(signal[:, 0].tolist())
            optimizer.zero_grad(set_to_none=True)
            loss = model(signal).square().mean()
            loss.backward()
            optimizer.step()
        return sequence

    seed_everything(42)
    baseline, baseline_optimizer, baseline_generator = create()
    baseline_order = [epoch(baseline, baseline_optimizer, baseline_generator) for _ in range(2)]

    seed_everything(42)
    interrupted, interrupted_optimizer, interrupted_generator = create()
    resumed_order = [epoch(interrupted, interrupted_optimizer, interrupted_generator)]
    save_epoch(tmp_path, "matching", 1, interrupted, interrupted_optimizer,
               interrupted_generator, [{"epoch": 1}])
    rebuilt, rebuilt_optimizer, rebuilt_generator = create()
    start, history, _, _, _ = resume_or_new(tmp_path, "matching", rebuilt,
                                             rebuilt_optimizer, rebuilt_generator)
    assert start == 1 and history == [{"epoch": 1}]
    resumed_order.append(epoch(rebuilt, rebuilt_optimizer, rebuilt_generator))
    assert baseline_order == resumed_order
    for expected, actual in zip(baseline.parameters(), rebuilt.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
