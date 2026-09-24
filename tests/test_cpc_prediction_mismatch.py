"""Temporal integrity, matched controls, frozen checkpoint and probe recovery."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ecg_experiment.cpc import CPCEncoder, CPCPretrainer
from ecg_experiment.cpc_prediction_mismatch import (
    aligned_representations, extract_branches, features_for_arm, pool_branch,
    supervised_indices, validate_bootstrap,
)
from scripts.experiments import run_cpc_prediction_mismatch as runner


def test_horizon_alignment_uses_only_the_past_query_and_retains_error_magnitude():
    positions = torch.arange(12).float()
    tokens = torch.stack((torch.ones_like(positions), positions + 30), dim=-1)[None, None].repeat(1, 2, 1, 1)
    contexts = torch.stack((torch.ones_like(positions), positions), dim=-1)[None, None].repeat(1, 2, 1, 1)
    aligned = aligned_representations(tokens, contexts, nn.Identity())
    assert aligned["residual"].shape == (1, 2, 5, 2)
    torch.testing.assert_close(aligned["prediction"], F.normalize(contexts[:, :, 3:8], dim=-1))
    torch.testing.assert_close(aligned["ordinary"], F.normalize(tokens[:, :, 7:12], dim=-1))
    torch.testing.assert_close(aligned["context"], contexts[:, :, 7:12])
    changed = contexts.clone()
    changed[:, :, 4:] = torch.randn_like(changed[:, :, 4:]) * 100
    changed_aligned = aligned_representations(tokens, changed, nn.Identity())
    torch.testing.assert_close(changed_aligned["prediction"][:, :, 0], aligned["prediction"][:, :, 0])
    torch.testing.assert_close(aligned["residual"], aligned["ordinary"] - aligned["prediction"])
    assert not torch.allclose(aligned["residual"].norm(dim=-1), torch.ones(1, 2, 5))


def test_pooling_and_dimension_matched_controls_use_identical_context_branch():
    values = torch.tensor([[[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]]])
    torch.testing.assert_close(pool_branch(values), torch.tensor([[11., 16.5, 16.5, 22.]]))
    branches = np.random.default_rng(4).normal(size=(3, 3, 512)).astype(np.float32)
    context = features_for_arm(branches, "context")
    ordinary = features_for_arm(branches, "ordinary")
    residual = features_for_arm(branches, "residual")
    assert context.shape == (3, 512) and ordinary.shape == residual.shape == (3, 1024)
    np.testing.assert_array_equal(context, ordinary[:, :512])
    np.testing.assert_array_equal(context, residual[:, :512])
    np.testing.assert_array_equal(ordinary[:, 512:], branches[:, 1])
    np.testing.assert_array_equal(residual[:, 512:], branches[:, 2])


def test_actual_encoder_has_no_future_or_cross_half_context_leakage():
    torch.manual_seed(32)
    encoder = CPCEncoder().eval()
    signal = torch.randn(1, 12, 2500)
    changed = signal.clone()
    # Context token 3 ends at downsampled input sample 48. Later raw input cannot
    # affect it; the second half is independently reset as in the original run.
    changed[:, :, 80:1250] = torch.randn_like(changed[:, :, 80:1250]) * 5
    changed[:, :, 1250:] = torch.randn_like(changed[:, :, 1250:]) * 10
    with torch.no_grad():
        tokens, contexts = encoder(signal)
        changed_tokens, changed_contexts = encoder(changed)
    torch.testing.assert_close(contexts[:, 0, :4], changed_contexts[:, 0, :4], rtol=0, atol=0)
    assert not torch.equal(tokens[:, 0, 7:], changed_tokens[:, 0, 7:])
    half_only = signal.clone()
    half_only[:, :, 1250:] += 300
    with torch.no_grad():
        _, half_contexts = encoder(half_only)
    torch.testing.assert_close(contexts[:, 0], half_contexts[:, 0], rtol=0, atol=0)


def checkpoint_objects():
    model = CPCPretrainer()
    weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    final = {"variant": "cpc", "epochs": 20, "fingerprint": "test",
             "encoder": {key.removeprefix("encoder."): value.clone() for key, value in weights.items() if key.startswith("encoder.")}}
    state = {"epoch": 20, "fingerprint": "test", "model": weights}
    config = {"fingerprint": "test", "inputs": {"settings": {"variant": "cpc", "epochs": 20,
              "seed": 42, "horizons": [4, 8, 12], "cmsc_weight": 0}}}
    return final, state, config


def test_checkpoint_requires_the_trained_heads_and_matching_final_encoder():
    final, state, config = checkpoint_objects()
    model = validate_bootstrap(final, state, config)
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    features = extract_branches(model, torch.randn(2, 12, 2500))
    assert features.shape == (2, 3, 512) and torch.isfinite(features).all()
    bad = {**state, "model": dict(state["model"])}
    del bad["model"]["heads.0.weight"]
    with pytest.raises(ValueError, match="all three"):
        validate_bootstrap(final, bad, config)
    bad["model"]["heads.0.weight"] = torch.zeros(128, 256)
    with pytest.raises(ValueError, match="wrong shape"):
        validate_bootstrap(final, bad, config)
    wrong = copy.deepcopy(config)
    wrong["inputs"]["settings"]["horizons"] = [8, 4, 12]
    with pytest.raises(ValueError, match="completed 20-epoch"):
        validate_bootstrap(final, state, wrong)
    final["encoder"][next(iter(final["encoder"]))].add_(1)
    with pytest.raises(ValueError, match="does not match"):
        validate_bootstrap(final, state, config)


def test_supervised_indices_filter_the_label_subset_and_reject_split_mismatches():
    cached = [{"ecg_id": str(i), "patient_id": f"p{i}", "split": split}
              for i, split in enumerate(("train", "train", "train", "validation", "test"))]
    selected = [{**cached[i], "target": str(i % 2)} for i in (2, 0)]
    np.testing.assert_array_equal(supervised_indices(cached, selected, "train"), [2, 0])
    with pytest.raises(ValueError, match="patient/split"):
        supervised_indices(cached, [{**cached[4], "target": "1"}], "train")
    with pytest.raises(ValueError, match="patient/split"):
        supervised_indices(cached, [{**selected[0], "patient_id": "wrong"}], "train")
    with pytest.raises(ValueError, match="Duplicate or absent"):
        supervised_indices(cached, [selected[0], selected[0]], "train")


def test_probe_scaler_sees_only_labeled_training_and_candidates_resume(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    features = rng.normal(size=(16, 3)).astype(np.float32)
    labels = np.tile([0, 1], 8)
    features[:, 0] = 4 * labels - 2
    features[:, 1:] = 0  # Exact ties across C make the documented tie rule testable.
    feature_rows = [{"ecg_id": str(i), "patient_id": f"p{i}",
                     "split": "train" if i < 8 else "validation" if i < 14 else "test"} for i in range(16)]
    rows = {"labeled_train": [{**feature_rows[i], "target": str(labels[i])} for i in (0, 1, 2, 3)],
            "development": [{**feature_rows[i], "target": str(labels[i])} for i in (8, 9, 10, 11)]}
    features[4:8] = 10000  # Unlabeled rows remain in the feature cache.
    features[12:] = -10000  # Calibration/test data cannot fit the scaler either.
    fingerprint = {"arm": "test", "data": "tiny"}
    fitted, selection = runner.fit_probe(features, feature_rows, rows, tmp_path, fingerprint, c_values=(0.01, 1.0))
    np.testing.assert_allclose(fitted["mean"], features[:4].astype(np.float64).mean(axis=0), rtol=0, atol=0)
    assert selection["C"] == 0.01 and selection["best_development_auroc"] == 1
    def forbid_fit(*args, **kwargs):
        raise AssertionError("Completed candidates should not be refit")
    monkeypatch.setattr(runner.LogisticRegression, "fit", forbid_fit)
    restored, resumed_selection = runner.fit_probe(features, feature_rows, rows, tmp_path, fingerprint,
                                                  resume=True, c_values=(0.01, 1.0))
    assert resumed_selection == selection
    for name in fitted:
        np.testing.assert_array_equal(restored[name], fitted[name])
    with pytest.raises(ValueError, match="identity/checksum"):
        runner.fit_probe(features, feature_rows, rows, tmp_path, {"changed": True}, resume=True, c_values=(0.01, 1.0))


def test_extraction_resumes_verified_chunks_without_repeating_completed_rows(tmp_path, monkeypatch):
    rows = [{"ecg_id": str(i), "patient_id": str(i), "source": "ptbxl", "split": "train"} for i in range(5)]
    class Pool:
        signals = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None], (5, 12, 2500))
        @staticmethod
        def indices(selected):
            return [int(row["ecg_id"]) for row in selected]
    args = SimpleNamespace(output_dir=tmp_path, batch_size=2, threads=1, device="cpu", resume=False)
    seen = []
    def interrupted(_, signal):
        if len(seen):
            raise RuntimeError("simulated interruption")
        seen.extend(signal[:, 0, 0].tolist())
        return signal[:, 0, 0, None, None].expand(-1, 3, 512).contiguous()
    monkeypatch.setattr(runner, "extract_branches", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.extract(args, Pool(), nn.Identity(), np.zeros(12), np.ones(12), rows, {"test": True})
    assert json.loads((tmp_path / "features/progress.json").read_text())["completed_rows"] == 2
    def resumed(_, signal):
        seen.extend(signal[:, 0, 0].tolist())
        return signal[:, 0, 0, None, None].expand(-1, 3, 512).contiguous()
    monkeypatch.setattr(runner, "extract_branches", resumed)
    args.resume = True
    runner.extract(args, Pool(), nn.Identity(), np.zeros(12), np.ones(12), rows, {"test": True})
    assert seen == [0, 1, 2, 3, 4]
    matrix = np.load(tmp_path / "features/features.npy", mmap_mode="r+")
    np.testing.assert_array_equal(matrix[:, 0, 0], np.arange(5))
    matrix[0, 0, 0] = 99
    matrix.flush()
    with pytest.raises(ValueError, match="checksum"):
        runner.extract(args, Pool(), nn.Identity(), np.zeros(12), np.ones(12), rows, {"test": True})
