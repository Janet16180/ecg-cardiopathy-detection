"""Invariants of the Experiment 064 beat tokens, objective and phase-matched patch kNN."""

from __future__ import annotations

import copy

import numpy as np
import torch

from ecg_experiment import dino_knn064 as dk


def synthetic_record(rate_hz: float = 1.2, seed: int = 0) -> np.ndarray:
    """Ten seconds of 12-lead spikes at 500 Hz, in mV."""
    rng = np.random.default_rng(seed)
    t = np.arange(5000) / 500
    beats = np.arange(0.25, 10, 1 / rate_hz)
    wave = sum(np.exp(-(((t - b) / 0.012) ** 2)) for b in beats)
    gains = rng.uniform(0.5, 1.5, (12, 1))
    return (gains * wave[None] + 0.01 * rng.standard_normal((12, 5000))).astype(np.float32)


def test_preprocess_keeps_millivolts_at_250_hz() -> None:
    record = synthetic_record()
    out = dk.preprocess(record)
    assert out.shape == (12, 2500)
    assert out.dtype == np.float32
    assert 0.2 < np.abs(out).max() < dk.CLIP_MV


def test_score_beats_cover_edges_and_align_on_r() -> None:
    record = synthetic_record()
    beats = dk.score_beats(record)
    peaks = dk.peaks_250(record)
    assert len(beats.windows) == len(peaks)
    assert beats.windows.shape[1:] == (12, dk.WINDOW)
    assert not beats.valid[0].all()
    assert beats.valid[len(peaks) // 2].all()
    middle = beats.windows[len(peaks) // 2]
    assert abs(int(np.abs(middle).sum(axis=0).argmax()) - dk.PRE) <= 2


def test_pool_windows_are_complete() -> None:
    record = synthetic_record()
    windows = dk.pool_windows(dk.preprocess(record), dk.peaks_250(record))
    assert windows.shape[1:] == (12, dk.STORED)
    assert len(windows) >= 9


def test_prematurity_flags_early_beat() -> None:
    peaks = np.array([0, 200, 400, 520, 800, 1000])
    values = dk.prematurity(peaks)
    assert values[0] == 0
    assert values[2] == 0
    assert values[3] > 0.3


def test_token_order_is_lead_major() -> None:
    windows = torch.arange(2 * 12 * dk.WINDOW, dtype=torch.float32).reshape(2, 12, dk.WINDOW)
    patches = dk.to_patches(windows)
    leads, phases = dk.token_grid(torch.device("cpu"))
    token = 5 * dk.PHASES + 7
    assert leads[token] == 5
    assert phases[token] == 7
    assert torch.equal(patches[1, token], windows[1, 5, 70:80])


def test_masks_and_local_views() -> None:
    generator = torch.Generator().manual_seed(0)
    masks = dk.block_masks(2000, generator, torch.device("cpu"))
    assert masks.shape == (2000, dk.TOKENS)
    assert 0.1 < masks.float().mean() < 0.4
    assert masks.any(dim=1).all()
    index = dk.local_tokens(50, generator, torch.device("cpu"))
    assert index.shape == (50, dk.LOCAL_LEADS * dk.LOCAL_PHASES)
    leads = index // dk.PHASES
    assert all(len(set(row.tolist())) == dk.LOCAL_LEADS for row in leads)


def test_loss_is_finite_and_teacher_receives_no_gradient() -> None:
    torch.manual_seed(0)
    student = dk.Network(width=32, depth=2, heads=4)
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    stored = torch.randn(8, 12, dk.STORED) * 0.3
    parts = dk.simdino_loss(student, teacher, stored, torch.Generator().manual_seed(1), dk.new_loss_module())
    assert all(torch.isfinite(value) for value in parts.values())
    parts["total"].backward()
    assert student.backbone.embed.weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters())
    before = [p.clone() for p in teacher.parameters()]
    dk.ema_update(teacher, student, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, teacher.parameters(), strict=True))


def test_schedules() -> None:
    assert dk.ema_momentum(0, 100) == dk.EMA_START
    assert abs(dk.ema_momentum(100, 100) - 1) < 1e-12
    assert dk.learning_rate(0, 100, 1.0, 10) == 0.1
    assert abs(dk.learning_rate(100, 100, 1.0, 10) - 0.01) < 1e-12


def test_invalid_tokens_do_not_change_valid_features() -> None:
    torch.manual_seed(0)
    backbone = dk.BeatViT(width=32, depth=2, heads=4).eval()
    patches = torch.randn(1, dk.TOKENS, dk.PATCH)
    leads, phases = dk.token_grid(torch.device("cpu"))
    valid = torch.ones(1, dk.TOKENS, dtype=torch.bool)
    valid[0, :40] = False
    altered = patches.clone()
    altered[0, :40] = 100.0
    _, a = backbone(patches, leads, phases, valid=valid)
    _, b = backbone(altered, leads, phases, valid=valid)
    assert torch.allclose(a[0, 40:], b[0, 40:], atol=1e-5)


def test_knn_distance_is_zero_for_bank_members() -> None:
    rng = np.random.default_rng(0)
    bank = torch.from_numpy(rng.standard_normal((12, 24, 50, 8))).float()
    bank = torch.nn.functional.normalize(bank, dim=-1)
    queries = bank[:, :, :3].permute(2, 0, 1, 3).contiguous()
    cosine = dk.knn_distance(queries, bank, 1, 1, True)
    euclid = dk.knn_distance(queries, bank, 1, 0, False)
    assert cosine.abs().max() < 1e-5
    assert euclid.abs().max() < 1e-3
    far = dk.knn_distance(-queries, bank, 1, 0, True)
    assert (far > 0).all()


def test_beat_map_takes_maximum_and_fills_minimum() -> None:
    windows = np.zeros((2, 12, dk.WINDOW), np.float32)
    valid = np.ones((2, dk.PHASES), bool)
    beats = dk.Beats(windows, valid, np.array([200, 260]), np.zeros(2))
    scores = np.zeros((2, 12, dk.PHASES), np.float32)
    scores[0, 3, 10] = 5.0
    scores[1, 3, 0] = 7.0
    scores[1, 4, 23] = -1.0
    out = dk.beat_map(scores, beats)
    assert out.shape == (12, 5000)
    assert out[3, 2 * (200 + 100) + 5] == 5.0
    assert out[3, 2 * 260 + 5] == 7.0
    assert out[0, 0] == out.min() == -1.0


def test_signed_log1p_is_monotone_and_bounded() -> None:
    values = np.array([-1e9, -2.0, 0.0, 0.5, 3.0, 1e30])
    out = dk.signed_log1p(values)
    assert (np.diff(out) > 0).all()
    assert np.isfinite(out.astype(np.float16)).all()
