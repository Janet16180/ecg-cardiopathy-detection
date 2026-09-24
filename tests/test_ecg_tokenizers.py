"""Causal beat detection, chunked CPC contexts and beat metadata preparation."""

import json

import numpy as np
import pytest
import torch

from ecg_experiment.cpc import CPCEncoder
from ecg_experiment.ecg_tokenizers import (
    CausalChunkEncoder,
    _sha256,
    beat_metadata,
    causal_beat_boundaries,
    detect_confirmed_beats,
    load_beat_metadata,
)
from scripts.data.prepare_beat_tokens import prepare


def synthetic_half(seed=1):
    rng = np.random.default_rng(seed)
    half = rng.normal(0, 0.005, (12, 1250)).astype(np.float32)
    grid = np.arange(1250)
    for peak in (100, 340, 580, 820, 1060):
        pulse = np.exp(-0.5 * ((grid - peak) / 4) ** 2).astype(np.float32)
        half[1] += pulse
        half[7] -= 0.7 * pulse
    return half


def test_r_like_events_confirm_later_and_preserve_rr():
    peaks, confirmations = detect_confirmed_beats(synthetic_half())
    assert len(peaks) == 5
    np.testing.assert_allclose(peaks, [100, 340, 580, 820, 1060], atol=15)
    np.testing.assert_array_equal(confirmations - peaks, 16)
    np.testing.assert_allclose(np.diff(peaks), 240, atol=1)
    boundaries, forced = causal_beat_boundaries(confirmations)
    assert boundaries[-1]
    assert forced[-1]
    assert np.all(np.diff(np.r_[-1, np.flatnonzero(boundaries)]) <= 24)


def test_future_perturbation_cannot_change_past_confirmed_boundaries():
    original = synthetic_half()
    changed = original.copy()
    changed[:, 850:] = synthetic_half(2)[:, 850:] * 20
    _, first = detect_confirmed_beats(original)
    _, second = detect_confirmed_beats(changed)
    np.testing.assert_array_equal(first[first < 850], second[second < 850])
    a, _ = causal_beat_boundaries(first)
    b, _ = causal_beat_boundaries(second)
    # Token 53 ends at raw sample 848; later raw changes cannot affect it.
    np.testing.assert_array_equal(a[:54], b[:54])


def test_online_fallback_and_half_reset():
    empty, forced = causal_beat_boundaries(np.array([], dtype=np.int16))
    np.testing.assert_array_equal(np.flatnonzero(empty), [23, 47, 71, 78])
    np.testing.assert_array_equal(empty, forced)
    a = np.zeros((12, 2500), dtype=np.float32)
    a[:, :1250] = synthetic_half()
    b = a.copy()
    b[:, 1250:] = synthetic_half(3)
    first = beat_metadata(a)
    changed = beat_metadata(b)
    for left, right in zip(first, changed, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
    assert np.all(first[0][:, -1])


def test_boundaries_reject_confirmations_outside_visible_half():
    with pytest.raises(ValueError, match="outside this half"):
        causal_beat_boundaries(np.array([10, 1249]))
    with pytest.raises(ValueError, match="strictly increasing"):
        causal_beat_boundaries(np.array([40, 40]))


def test_fixed_and_learned_match_at_zero_router_initialization():
    torch.manual_seed(10)
    fixed = CausalChunkEncoder("fixedchunk").eval()
    torch.manual_seed(10)
    learned = CausalChunkEncoder("learnedchunk").eval()
    tokens = torch.randn(1, 2, 79, 256)
    with torch.no_grad():
        fixed_context = fixed.chunk_context(tokens)
        learned_context = learned.chunk_context(tokens)
    torch.testing.assert_close(fixed_context, learned_context, atol=0, rtol=0)
    torch.testing.assert_close(fixed.pooled(fixed_context),
                               learned.pooled(learned_context), atol=0, rtol=0)
    assert learned.diagnostics["chunk_count"] == 5
    assert learned.rate_penalty.item() == 0


def test_fixed_chunks_emit_every_sixteen_tokens_and_at_the_end():
    model = CausalChunkEncoder("fixedchunk").eval()
    with torch.no_grad():
        model.chunk_context(torch.randn(1, 2, 79, 256))
    emitted = torch.nonzero(model.last_gates[0, 0]).flatten().tolist()
    assert emitted == [15, 31, 47, 63, 78]
    assert model.diagnostics["forced_ends"] == 1


def test_beat_chunks_follow_boundaries_with_minimum_and_maximum_lengths():
    model = CausalChunkEncoder("beatchunk").eval()
    beat_boundaries = torch.zeros(1, 2, 79, dtype=torch.bool)
    beat_boundaries[:, :, [1, 9, 11]] = True
    with torch.no_grad():
        model.chunk_context(torch.randn(1, 2, 79, 256), beat_boundaries)
    emitted = torch.nonzero(model.last_gates[0, 0]).flatten().tolist()
    # Step 1 is shorter than four tokens; after 11, only the 24-token cap and the end emit.
    assert emitted == [9, 33, 57, 78]
    with pytest.raises(ValueError, match="Beat boundaries"):
        model.chunk_context(torch.randn(1, 2, 79, 256))


def test_all_chunk_arms_are_causal_on_token_grid():
    tokens = torch.randn(1, 2, 79, 256)
    changed = tokens.clone()
    changed[:, :, 40:] += 20
    beat_boundaries = torch.zeros(1, 2, 79, dtype=torch.bool)
    beat_boundaries[:, :, [9, 21, 37, 50, 62, 78]] = True
    for variant in CausalChunkEncoder.VARIANTS:
        model = CausalChunkEncoder(variant).eval()
        beat = beat_boundaries if variant == "beatchunk" else None
        with torch.no_grad():
            before = model.chunk_context(tokens, beat)
            after = model.chunk_context(changed, beat)
        torch.testing.assert_close(before[:, :, :40], after[:, :, :40], atol=0, rtol=0)


def test_bootstrap_copies_native_gru_without_unused_native_parameters():
    native = CPCEncoder()
    chunk = CausalChunkEncoder("fixedchunk")
    chunk.load_from_cpc_state_dict(native.state_dict())
    assert not any(key.startswith("core.context.") for key in chunk.state_dict())
    for layer, cell in enumerate(chunk.cells):
        torch.testing.assert_close(cell.weight_ih, getattr(native.context, f"weight_ih_l{layer}"))
        torch.testing.assert_close(cell.weight_hh, getattr(native.context, f"weight_hh_l{layer}"))


def test_learned_boundaries_receive_context_gradient():
    model = CausalChunkEncoder("learnedchunk")
    tokens = torch.randn(1, 2, 79, 256)
    contexts = model.chunk_context(tokens)
    (contexts[:, :, 30:].square().mean() + model.rate_penalty).backward()
    assert torch.isfinite(model.gate.weight.grad).all()
    assert float(model.gate.weight.grad.abs().sum()) > 0


def test_classifier_pool_uses_only_emitted_states():
    model = CausalChunkEncoder("fixedchunk")
    contexts = torch.full((1, 2, 79, 256), 100.0)
    gates = torch.zeros(1, 2, 79)
    gates[:, :, 15] = 1
    gates[:, :, 78] = 1
    contexts[:, :, 15] = 2
    contexts[:, :, 78] = 4
    model.last_gates = gates
    pooled = model.pooled(contexts)
    assert pooled.shape == (1, 512)
    torch.testing.assert_close(pooled[:, :256], torch.full((1, 256), 3.0))
    torch.testing.assert_close(pooled[:, 256:], torch.full((1, 256), 4.0))


def test_small_cache_produces_hashed_aligned_metadata(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "beats"
    cache.mkdir()
    signal = np.stack([np.concatenate([synthetic_half(i), synthetic_half(i + 10)], axis=1)
                       for i in (1, 2, 3)]).astype(np.float32)
    ids = np.asarray(["a", "b", "c"])
    np.save(cache / "signals.npy", signal)
    np.save(cache / "ecg_ids.npy", ids)
    (cache / "complete.json").write_text(json.dumps({
        "record_count": 3, "shape": [3, 12, 2500],
        "signals_sha256": _sha256(cache / "signals.npy"),
        "ecg_ids_sha256": _sha256(cache / "ecg_ids.npy")}))
    info = prepare(cache, output, workers=2)
    assert info["record_count"] == 3
    boundaries = load_beat_metadata(output, ids)
    assert boundaries.shape == (3, 2, 79)
    assert np.all(boundaries[:, :, -1])
    assert prepare(cache, output, workers=2) == info
    with pytest.raises(ValueError, match="IDs differ"):
        load_beat_metadata(output, np.asarray(["a", "b", "wrong"]))
