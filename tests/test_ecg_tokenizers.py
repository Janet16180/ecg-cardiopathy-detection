import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ecg_experiment.cpc import CPCEncoder
from ecg_experiment.ecg_tokenizers import (
    CausalChunkEncoder, beat_metadata, causal_beat_boundaries,
    detect_confirmed_beats, load_beat_metadata, _sha256,
)
from scripts.prepare_beat_tokens import prepare


def synthetic_half(seed=1):
    rng = np.random.default_rng(seed)
    half = rng.normal(0, 0.005, (12, 1250)).astype(np.float32)
    grid = np.arange(1250)
    for peak in (100, 340, 580, 820, 1060):
        pulse = np.exp(-0.5 * ((grid - peak) / 4) ** 2).astype(np.float32)
        half[1] += pulse
        half[7] -= 0.7 * pulse
    return half


class BeatDetectorTests(unittest.TestCase):
    def test_r_like_events_confirm_later_and_preserve_rr(self):
        peaks, confirmations = detect_confirmed_beats(synthetic_half())
        self.assertEqual(len(peaks), 5)
        np.testing.assert_allclose(peaks, [100, 340, 580, 820, 1060], atol=15)
        np.testing.assert_array_equal(confirmations - peaks, 16)
        np.testing.assert_allclose(np.diff(peaks), 240, atol=1)
        boundaries, forced = causal_beat_boundaries(confirmations)
        self.assertTrue(boundaries[-1])
        self.assertTrue(forced[-1])
        self.assertTrue(np.all(np.diff(np.r_[-1, np.flatnonzero(boundaries)]) <= 24))

    def test_future_perturbation_cannot_change_past_confirmed_boundaries(self):
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

    def test_online_fallback_and_half_reset(self):
        empty, forced = causal_beat_boundaries(np.array([], dtype=np.int16))
        np.testing.assert_array_equal(np.flatnonzero(empty), [23, 47, 71, 78])
        np.testing.assert_array_equal(empty, forced)
        a = np.zeros((12, 2500), dtype=np.float32)
        a[:, :1250] = synthetic_half()
        b = a.copy()
        b[:, 1250:] = synthetic_half(3)
        first = beat_metadata(a)
        changed = beat_metadata(b)
        for left, right in zip(first, changed):
            np.testing.assert_array_equal(left[0], right[0])
        self.assertTrue(np.all(first[0][:, -1]))


class ChunkEncoderTests(unittest.TestCase):
    def test_fixed_and_learned_match_at_zero_router_initialization(self):
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
        self.assertEqual(learned.diagnostics["chunk_count"], 5)
        self.assertEqual(learned.rate_penalty.item(), 0)

    def test_all_chunk_arms_are_causal_on_token_grid(self):
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

    def test_bootstrap_copies_native_gru_without_unused_native_parameters(self):
        native = CPCEncoder()
        chunk = CausalChunkEncoder("fixedchunk")
        chunk.load_bootstrap(native.state_dict())
        self.assertFalse(any(key.startswith("core.context.") for key in chunk.state_dict()))
        for layer, cell in enumerate(chunk.cells):
            torch.testing.assert_close(cell.weight_ih, getattr(native.context, f"weight_ih_l{layer}"))
            torch.testing.assert_close(cell.weight_hh, getattr(native.context, f"weight_hh_l{layer}"))

    def test_learned_boundaries_receive_context_gradient(self):
        model = CausalChunkEncoder("learnedchunk")
        tokens = torch.randn(1, 2, 79, 256)
        contexts = model.chunk_context(tokens)
        (contexts[:, :, 30:].square().mean() + model.rate_penalty).backward()
        self.assertTrue(torch.isfinite(model.gate.weight.grad).all())
        self.assertGreater(float(model.gate.weight.grad.abs().sum()), 0)

    def test_classifier_pool_uses_only_emitted_states(self):
        model = CausalChunkEncoder("fixedchunk")
        contexts = torch.full((1, 2, 79, 256), 100.0)
        gates = torch.zeros(1, 2, 79)
        gates[:, :, 15] = 1
        gates[:, :, 78] = 1
        contexts[:, :, 15] = 2
        contexts[:, :, 78] = 4
        model.last_gates = gates
        pooled = model.pooled(contexts)
        self.assertEqual(pooled.shape, (1, 512))
        torch.testing.assert_close(pooled[:, :256], torch.full((1, 256), 3.0))
        torch.testing.assert_close(pooled[:, 256:], torch.full((1, 256), 4.0))


class BeatPreparationTests(unittest.TestCase):
    def test_small_cache_produces_hashed_aligned_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, output = root / "cache", root / "beats"
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
            self.assertEqual(info["record_count"], 3)
            boundaries = load_beat_metadata(output, ids)
            self.assertEqual(boundaries.shape, (3, 2, 79))
            self.assertTrue(np.all(boundaries[:, :, -1]))
            self.assertEqual(prepare(cache, output, workers=2), info)
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                load_beat_metadata(output, np.asarray(["a", "b", "wrong"]))


if __name__ == "__main__":
    unittest.main()
