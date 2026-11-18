"""Tests of candidate S-R (xresnet1d101 from scratch): role guard, inputs, loss masking, schedule, windows."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import resample_poly

from ecg_experiment.mlc01_dataset import array_sha256, microvolts
from ecg_experiment.mlc02_data import prediction_path
from ecg_experiment.mlc02_scratch_network import (
    CANDIDATE_DIR,
    CANDIDATES,
    CLOSED_ROLES,
    CROP,
    MODEL_ID,
    ONE_CYCLE,
    PEAK_LR,
    RECORD_SAMPLES,
    SEEDS,
    VENDORED,
    WINDOW_STARTS,
    build_model,
    check_roles,
    choose_epoch,
    crops,
    epoch_batches,
    fit_standardizer,
    make_optimizer,
    masked_bce,
    read_records,
    read_windows,
    select_roles,
    standardize,
    to_model_rate,
    window_probabilities,
)
from ecg_experiment.mlc02_scratch_runs import masks
from ecg_experiment.mlc02_waveforms import RAIL_MV


def synthetic_cache(directory: Path, block: str, records: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    windows = (np.round(rng.normal(0, 0.5, (records, 12, 5000)) * 1000) / 1000).astype(np.float32)
    np.save(directory / f"{block}.npy", windows)
    role, source = block.split("__")[:2]
    return pd.DataFrame({"record_key": [f"{source}:{i}" for i in range(records)], "source": source,
                         "family": source, "role": role,
                         "sha256_10s": [array_sha256(microvolts(w.T)) for w in windows],
                         "block": block, "row": np.arange(records)})


@pytest.mark.parametrize("role", CLOSED_ROLES)
def test_check_roles_refuses_every_test_role(role):
    with pytest.raises(PermissionError):
        check_roles([role])
    with pytest.raises(PermissionError):
        check_roles(["train", role])


def test_check_roles_accepts_train_and_validation_only():
    assert check_roles(["validation", "train", "train"]) == ("train", "validation")
    with pytest.raises(ValueError, match="unknown"):
        check_roles(["closed"])
    with pytest.raises(ValueError, match="unknown"):
        check_roles([])


def test_read_windows_refuses_test_records_before_opening_any_file(tmp_path):
    records = pd.DataFrame({"record_key": ["sph:1"], "source": ["sph"], "family": ["sph"],
                            "role": ["test_external"], "sha256_10s": ["x"], "block": ["test_external__sph"],
                            "row": [0]})
    with pytest.raises(PermissionError):
        read_windows(records, tmp_path)
    disguised = records.assign(role="train")
    with pytest.raises(PermissionError):
        read_windows(disguised, tmp_path)
    with pytest.raises(PermissionError):
        select_roles(records, ["test_external"])
    assert not list(tmp_path.iterdir())


def test_read_windows_returns_rows_and_checks_hashes(tmp_path):
    records = synthetic_cache(tmp_path, "train__ptbxl", 5)
    expected = np.load(tmp_path / "train__ptbxl.npy")
    order = records.iloc[[3, 0, 4]]
    np.testing.assert_array_equal(read_windows(order, tmp_path), expected[[3, 0, 4]])
    broken = order.assign(sha256_10s=order["sha256_10s"].where(order["row"] != 0, "0" * 64))
    with pytest.raises(ValueError, match="manifest hash"):
        read_windows(broken, tmp_path)


def test_to_model_rate_fills_rail_then_decimates_whole_record():
    rng = np.random.default_rng(1)
    windows = rng.normal(0, 1, (2, 12, 5000)).astype(np.float32)
    windows[1, 3, 100:110] = np.nan
    result = to_model_rate(windows)
    assert result.shape == (2, 12, RECORD_SAMPLES)
    assert result.dtype == np.float32
    filled = np.where(np.isnan(windows[1]), np.float32(RAIL_MV), windows[1]).astype(np.float64)
    np.testing.assert_allclose(result[1], resample_poly(filled, 1, 5, axis=1), rtol=0, atol=1e-5)
    assert np.isfinite(result).all()


def test_standardizer_is_one_scalar_over_all_samples_and_leads():
    rng = np.random.default_rng(2)
    signals = rng.normal(0.3, 2.0, (37, 12, 1000)).astype(np.float32)
    signals[:, 0] += 5
    stats = fit_standardizer(signals, chunk=8)
    assert math.isclose(stats["mean"], float(signals.astype(np.float64).mean()), rel_tol=1e-12)
    assert math.isclose(stats["std"], float(signals.astype(np.float64).std()), rel_tol=1e-12)
    scaled = standardize(signals, stats)
    assert scaled.dtype == np.float32
    assert abs(float(scaled.mean())) < 1e-5
    assert abs(float(scaled.std()) - 1) < 1e-5


def test_masked_bce_gives_zero_gradient_on_undefined_entries():
    torch.manual_seed(0)
    logits = torch.randn(6, 14, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 2, (6, 14)).double()
    labels[0, 3] = labels[2, :5] = labels[5, 13] = float("nan")
    loss = masked_bce(logits, labels)
    loss.backward()
    undefined = torch.isnan(labels)
    assert (logits.grad[undefined] == 0).all()
    assert (logits.grad[~undefined] != 0).all()
    defined = ~undefined
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[defined], labels[defined])
    assert torch.allclose(loss, expected)


def test_masked_bce_equals_plain_bce_when_every_entry_is_defined():
    logits, labels = torch.randn(4, 14), torch.randint(0, 2, (4, 14)).float()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert torch.allclose(masked_bce(logits, labels), expected)


def test_epoch_batches_are_reproducible_full_and_within_the_record():
    rows, starts = epoch_batches(1000, SEEDS[0], 3, batch=128)
    again = epoch_batches(1000, SEEDS[0], 3, batch=128)
    np.testing.assert_array_equal(rows, again[0])
    np.testing.assert_array_equal(starts, again[1])
    assert rows.shape == (7, 128)
    assert len(np.unique(rows)) == 7 * 128
    assert starts.min() >= 0
    assert starts.max() <= RECORD_SAMPLES - CROP - 1
    assert not np.array_equal(rows, epoch_batches(1000, SEEDS[0], 4, batch=128)[0])


def test_crops_cut_the_requested_windows():
    signals = np.arange(3 * 12 * 1000, dtype=np.float32).reshape(3, 12, 1000)
    result = crops(signals, np.array([2, 0]), np.array([10, 749]))
    np.testing.assert_array_equal(result[0], signals[2, :, 10:10 + CROP])
    np.testing.assert_array_equal(result[1], signals[0, :, 749:749 + CROP])
    assert result.flags["C_CONTIGUOUS"]


class WindowMean(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)[:, :1].repeat(1, 14) * torch.arange(1, 15, dtype=x.dtype)


def test_window_probabilities_average_four_consecutive_windows():
    rng = np.random.default_rng(3)
    signals = rng.normal(0, 1, (5, 12, 1000)).astype(np.float32)
    result = window_probabilities(WindowMean(), signals, "cpu", batch=2)
    means = [signals[:, 0, start:start + CROP].astype(np.float64).mean(axis=1) for start in WINDOW_STARTS]
    logits = np.stack(means, axis=1)[:, :, None] * np.arange(1, 15)
    expected = (1 / (1 + np.exp(-logits))).mean(axis=1)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    assert WINDOW_STARTS == (0, 250, 500, 750)


@pytest.mark.skipif(not VENDORED.exists(), reason="needs the vendored benchmark code")
def test_model_matches_the_benchmark_configuration():
    torch.manual_seed(0)
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 1_873_422
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert all(torch.all(m.weight == 1) and torch.all(m.bias == 0) for m in norms)
    head = model[-1]
    assert [type(layer).__name__ for layer in head] == [
        "AdaptiveConcatPool1d", "Flatten", "BatchNorm1d", "Dropout", "Linear", "ReLU", "BatchNorm1d",
        "Dropout", "Linear"]
    assert (head[3].p, head[7].p, head[4].out_features, head[8].out_features) == (0.25, 0.5, 128, 14)
    assert model.eval()(torch.zeros(2, 12, CROP)).shape == (2, 14)


def fastai_one_cycle(steps: int) -> tuple[np.ndarray, np.ndarray]:
    def cos(start: float, end: float, pct: float) -> float:
        return end + (start - end) / 2 * (np.cos(np.pi * pct) + 1)

    first = int(steps * ONE_CYCLE["pct_start"])
    low, final = PEAK_LR / 25, PEAK_LR / (25 * 1e4)
    lrs, moms = [low], [0.95]
    for step in range(1, steps):
        if step <= first:
            lrs.append(cos(low, PEAK_LR, step / first))
            moms.append(cos(0.95, 0.85, step / first))
        else:
            lrs.append(cos(PEAK_LR, final, (step - first) / (steps - first)))
            moms.append(cos(0.85, 0.95, (step - first) / (steps - first)))
    return np.array(lrs), np.array(moms)


def test_one_cycle_follows_the_fastai_v1_schedule():
    steps_per_epoch, epochs = 439, 50
    model = torch.nn.Linear(2, 1)
    optimizer, scheduler = make_optimizer(model, steps_per_epoch, epochs)
    lrs, moms = [], []
    for _ in range(steps_per_epoch * epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        moms.append(optimizer.param_groups[0]["betas"][0])
        optimizer.step()
        scheduler.step()
    expected_lr, expected_mom = fastai_one_cycle(steps_per_epoch * epochs)
    assert np.abs(np.array(lrs) - expected_lr).max() < 1e-3 * PEAK_LR
    assert np.abs(np.array(moms) - expected_mom).max() < 1e-3
    assert optimizer.param_groups[0]["betas"][1] == 0.99
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    assert math.isclose(max(lrs), PEAK_LR, rel_tol=1e-6)


def test_choose_epoch_takes_the_earliest_best():
    history = [{"epoch": 1, "selection_metric": 0.8}, {"epoch": 2, "selection_metric": 0.9},
               {"epoch": 3, "selection_metric": 0.9}, {"epoch": 4, "selection_metric": 0.85}]
    assert choose_epoch(history) == 2


def test_leave_one_family_out_masks_keep_the_family_out_of_fitting_and_selection():
    records = pd.DataFrame({"role": ["train", "validation", "train", "validation", "train"],
                            "family": ["ptbxl", "ptbxl", "georgia", "georgia", "cpsc"]})
    selected = masks(records, "ptbxl")
    np.testing.assert_array_equal(selected["train"], [False, False, True, False, True])
    np.testing.assert_array_equal(selected["validation"], [False, False, False, True, False])
    np.testing.assert_array_equal(selected["held_out"], [True, True, False, False, False])
    main = masks(records, None)
    assert set(main) == {"train", "validation"}


@pytest.mark.skipif(not (CANDIDATES.parent / "features/rows.csv").exists(), reason="needs MLC-02 inputs")
def test_real_records_are_aligned_and_test_roles_are_not_selected():
    records = read_records()
    assert len(records) == 99_167
    allowed = select_roles(records, ["train", "validation"])
    assert allowed["role"].value_counts().to_dict() == {"train": 56_250, "validation": 15_138}
    assert not allowed["block"].str.startswith(CLOSED_ROLES).any()


@pytest.mark.skipif(not (CANDIDATE_DIR / "settings.json").exists(), reason="needs S-R checkpoints")
def test_predict_reproduces_the_saved_validation_predictions_on_cpu():
    from ecg_experiment.mlc02_data import load_predictions
    from ecg_experiment.mlc02_scratch_network import predict

    records = select_roles(read_records(), ["validation"])
    saved = load_predictions(prediction_path(CANDIDATES, MODEL_ID, "validation", SEEDS),
                             records["record_key"].to_numpy())["probability"]
    rows = np.random.default_rng(0).choice(len(records), 48, replace=False)
    probability = predict(read_windows(records.iloc[np.sort(rows)]), CANDIDATE_DIR, "cpu")
    np.testing.assert_allclose(probability, saved[np.sort(rows)], rtol=0, atol=1e-5)
