from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_experiment import mlc02_data as md
from ecg_experiment import mlc02_feature_data as fd
from ecg_experiment import mlc02_finetune as ft
from ecg_experiment import mlc02_finetune_data as fdata
from ecg_experiment import mlc02_finetune_runs as fr

MISSING = Path("/nonexistent/mlc02")
HAVE_CACHE = (fdata.CACHE / fdata.INDEX).exists() and md.MANIFEST.exists()
HAVE_XECG = (ft.DEFAULT_CHECKPOINT_DIR / "model.safetensors").exists()
HAVE_BASE = (ft.CANDIDATES / ft.BASE / "model.json").exists()
HAVE_F = (ft.DIRECTORY / "model.json").exists()
needs_data = pytest.mark.skipif(not (HAVE_CACHE and HAVE_XECG and HAVE_BASE),
                                reason="local MLC-02 data absent")
needs_cache = pytest.mark.skipif(not HAVE_CACHE, reason="local MLC-02 data absent")
needs_xecg = pytest.mark.skipif(not HAVE_XECG, reason="released xECG absent")


# Role guard


@pytest.mark.parametrize("role", fd.CLOSED_ROLES)
def test_closed_roles_are_refused_before_any_file_is_read(role):
    manifest = MISSING / "manifest.csv"
    features = {"role": np.array([role]), "record_key": np.array(["x"])}
    calls = [
        lambda: fdata.cache_rows((role,), MISSING, manifest),
        lambda: fdata.cache_rows(("train", role), MISSING, manifest),
        lambda: fdata.read_windows((role,), np.array(["sph:1"]), MISSING, manifest),
        lambda: fdata.prepare_inputs(role, 1, MISSING, MISSING, manifest),
        lambda: fdata.load_inputs(role, MISSING, manifest),
        lambda: fdata.cached_xecg_features(features, MISSING),
    ]
    for call in calls:
        with pytest.raises(PermissionError, match="closed"):
            call()


@pytest.mark.parametrize("block", ["test_same_source__ptbxl", "test_external__sph", "sph_calibration__sph",
                                   "test_external__sph__supplement", "sph_calibration__sph__supplement"])
def test_blocks_of_closed_roles_are_never_opened(block):
    with pytest.raises(PermissionError, match="closed"):
        fdata.open_blocks(["validation__ptbxl", block], {"validation__ptbxl": 1, block: 1}, MISSING)


def test_unknown_roles_are_refused():
    for roles in ((), ("sph",), ("training",)):
        with pytest.raises(ValueError, match="reads roles"):
            fdata.cache_rows(roles, MISSING, MISSING / "manifest.csv")


def test_block_role_reads_the_role_prefix():
    assert fdata.block_role("validation__cpsc_2018__supplement") == "validation"
    assert fdata.block_role("sph_calibration__sph") == "sph_calibration"
    assert fdata.block_role("test_same_source__ptbxl") == "test_same_source"


@needs_cache
def test_cache_rows_align_with_the_manifest_and_catch_a_changed_hash(tmp_path):
    data, rows, sizes = fdata.cache_rows(("validation",))
    assert len(rows) == 15138
    assert (rows["record_key"].to_numpy() == data["record_key"]).all()
    assert set(rows["role"]) == {"validation"}
    assert all(fdata.block_role(name) == "validation" for name in sizes)
    index = pd.read_csv(fdata.CACHE / fdata.INDEX, dtype=str)
    changed = index["record_key"] == rows["record_key"].iloc[5]
    index.loc[changed, "sha256_10s"] = "0" * 64
    index.to_csv(tmp_path / fdata.INDEX, index=False)
    with pytest.raises(ValueError, match="sha256_10s"):
        fdata.cache_rows(("validation",), tmp_path)


@needs_cache
def test_read_windows_returns_hash_checked_validation_windows():
    data, _, _ = fdata.cache_rows(("validation",))
    windows = fdata.read_windows(("validation",), data["record_key"][[0, 5000, 15137]])
    assert windows.shape == (3, 12, 5000)
    assert windows.dtype == np.float32
    with pytest.raises(ValueError, match="not included records"):
        fdata.read_windows(("validation",), np.array(["ptbxl:1"]))


def test_window_inputs_require_the_cache_layout():
    with pytest.raises(ValueError, match="float32"):
        fdata.window_inputs(np.zeros((2, 12, 5000), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        fdata.window_inputs(np.zeros((2, 5000, 12), dtype=np.float32))
    signal = np.random.default_rng(0).normal(size=(2, 12, 5000)).astype(np.float32)
    signal[0, 3, 10] = np.nan
    inputs = fdata.window_inputs(signal)
    assert inputs.shape == (2, 1000, 12)
    assert inputs.dtype == np.float32
    assert np.isfinite(inputs).all()


# Model and parameter groups


@pytest.fixture(scope="module")
def released():
    if not HAVE_XECG:
        pytest.skip("released xECG absent")
    return ft.FineTuned(ft.xecg_encoder(), np.zeros(1024), np.ones(1024))


@needs_xecg
def test_drop_path_and_dropout_follow_the_released_configuration(released):
    assert ft.DROP_PATH == ft.pretrained_drop_path() == 0.0
    assert released.encoder.core.dropout_rates == [0.0] * ft.BLOCKS
    assert released.encoder.dropout.p == 0.0


def _expected_rate(layer, rate):
    if layer == "head":
        return 10 * rate
    if layer == "embedding":
        return rate * 0.75**9
    return rate * 0.75 ** (8 - int(layer.removeprefix("block_")))


@needs_xecg
def test_parameter_groups_cover_every_parameter_once_with_layerwise_rates(released):
    rate = 3e-5
    groups = ft.parameter_groups(released, rate)
    names = [name for group in groups for name in group["parameter_names"]]
    assert sorted(names) == sorted(name for name, _ in released.named_parameters())
    assert len(names) == len(set(names))
    parameters = dict(released.named_parameters())
    modules = dict(released.named_modules())
    for group in groups:
        layer, kind = group["name"].split("/")
        assert group["lr"] == group["base_lr"]
        assert group["base_lr"] == pytest.approx(_expected_rate(layer, rate))
        assert group["weight_decay"] == (0.01 if kind == "decay" else 0.0)
        for name in group["parameter_names"]:
            owner = type(modules[name.rsplit(".", 1)[0]]).__name__
            bias = name.endswith(("bias", "_bias_"))
            assert (kind == "no_decay") == (bias or owner in ft.NORM_CLASSES or name == "encoder.mask_token")
            if kind == "decay" and parameters[name].ndim == 1:
                assert name.endswith("learnable_skip")
    rates = {group["name"].split("/")[0]: group["base_lr"] for group in groups}
    assert rates["block_8"] == pytest.approx(rate)
    assert rates["block_0"] == pytest.approx(rate * 0.75**8)
    assert len(rates) == 11


def test_every_unexpected_parameter_is_refused():
    with pytest.raises(ValueError, match="no layer-wise rate"):
        ft.layer_name("encoder.core.model.post_blocks_norm.weight")


# Schedule and evaluation points


def test_plan_evaluates_every_half_epoch_over_five_epochs():
    schedule = ft.plan(56250)
    assert (schedule["per_epoch"], schedule["total"], schedule["warmup"]) == (1758, 8790, 440)
    assert schedule["points"] == [879, 1758, 2637, 3516, 4395, 5274, 6153, 7032, 7911, 8790]
    assert ft.plan(10)["points"] == [1, 2, 3, 4, 5]


def test_rate_factor_warms_up_linearly_then_decays_to_zero():
    total, warmup = 8790, 440
    factors = np.array([ft.rate_factor(update, total, warmup) for update in range(1, total + 1)])
    assert factors[0] == pytest.approx(1 / 440)
    assert factors[warmup - 1] == 1.0
    assert np.all(np.diff(factors[:warmup]) > 0)
    assert np.all(np.diff(factors[warmup:]) <= 0)
    assert factors[-1] == pytest.approx(0.0, abs=1e-15)
    assert ft.rate_factor(warmup + (total - warmup) // 2, total, warmup) == pytest.approx(0.5, abs=1e-3)


# Masked loss


class LookupLogits(torch.nn.Module):
    """Logits looked up by the record number written in the input."""

    def __init__(self, records):
        super().__init__()
        self.table = torch.nn.Parameter(torch.linspace(-1, 1, records * 14).view(records, 14))

    def forward(self, signal):
        return self.table[signal[:, 0, 0].long()]


def _lookup_batch():
    records = 7
    inputs = np.repeat(np.arange(records, dtype=np.float32)[:, None, None], 1000, axis=1).repeat(12, axis=2)
    labels = np.random.default_rng(1).integers(0, 2, size=(records, 14)).astype(np.float32)
    labels[np.random.default_rng(2).random(labels.shape) < 0.3] = np.nan
    labels[3] = np.nan
    return inputs, labels


@pytest.mark.parametrize("microbatch", [1, 2, 3, 7])
def test_masked_loss_accumulates_to_the_full_batch_mean_over_defined_entries(microbatch):
    inputs, labels = _lookup_batch()
    model = LookupLogits(len(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss, norm = ft.train_update(model, optimizer, inputs, labels, np.arange(len(inputs)), microbatch, "cpu",
                                 autocast=False)
    defined = ~np.isnan(labels)
    probability = 1 / (1 + np.exp(-model.table.detach().double().numpy()))
    y = np.nan_to_num(labels)
    expected = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))[defined].mean()
    assert loss == pytest.approx(expected, rel=1e-5)
    gradient = model.table.grad.double().numpy()
    assert np.all(gradient[~defined] == 0)
    full = (probability - y) * defined / defined.sum()
    assert norm == pytest.approx(np.sqrt(np.sum(full**2)), rel=1e-4)
    scale = min(1.0, ft.CLIP_NORM / norm)
    np.testing.assert_allclose(gradient[defined], (full * scale)[defined], rtol=1e-4)


def test_an_effective_batch_without_defined_labels_is_refused():
    inputs, labels = _lookup_batch()
    model = LookupLogits(len(inputs))
    with pytest.raises(RuntimeError, match="no defined label"):
        ft.train_update(model, torch.optim.SGD(model.parameters(), lr=0.0), inputs, labels, np.array([3]), 1,
                        "cpu", autocast=False)


# Checkpoints


def _optimizer(model):
    return torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-2, "base_lr": 1e-2}])


def test_resumable_state_round_trips_and_refuses_another_run(tmp_path):
    inputs, labels = _lookup_batch()
    model = LookupLogits(len(inputs))
    optimizer = _optimizer(model)
    generator = torch.Generator().manual_seed(5)
    ft.train_update(model, optimizer, inputs, labels, np.arange(7), 3, "cpu", autocast=False)
    order = torch.randperm(7, generator=generator)
    progress = {"fingerprint": {"rate": 1e-2, "seed": 5}, "update": 1, "order": order,
                "history": [{"point": 1, "selection_metric": 0.5, "eligible": True}],
                "best": {"point": 1, "selection_metric": 0.5}, "start": {"records": 7}, "resumes": []}
    ft.save_state(tmp_path / "resume.pt", model, optimizer, generator, progress, "cpu")
    copy = torch.Generator()
    copy.set_state(generator.get_state())
    expected_next = torch.randperm(7, generator=copy)
    other = LookupLogits(len(inputs))
    other_optimizer = _optimizer(other)
    other_generator = torch.Generator().manual_seed(99)
    saved = ft.restore_state(tmp_path / "resume.pt", other, other_optimizer, other_generator,
                             progress["fingerprint"], "cpu")
    assert torch.equal(other.table, model.table)
    assert torch.equal(saved["order"], order)
    assert saved["history"] == progress["history"]
    assert saved["update"] == 1
    state, other_state = optimizer.state_dict()["state"][0], other_optimizer.state_dict()["state"][0]
    assert all(torch.equal(state[key], other_state[key]) for key in state)
    assert torch.equal(torch.randperm(7, generator=other_generator), expected_next)
    with pytest.raises(ValueError, match="another identity"):
        ft.restore_state(tmp_path / "resume.pt", other, other_optimizer, other_generator, {"rate": 3e-5},
                         "cpu")


# Choice of rate and gate


def _run(value):
    return {"best": {"selection_metric": value}}


def test_rate_choice_takes_the_higher_metric_and_breaks_exact_ties_towards_1e5():
    assert fr.choose_rate({1e-5: _run(0.96), 3e-5: _run(0.97)}) == 3e-5
    assert fr.choose_rate({1e-5: _run(0.97), 3e-5: _run(0.96)}) == 1e-5
    assert fr.choose_rate({1e-5: _run(0.97), 3e-5: _run(0.97)}) == 1e-5
    assert fr.choose_rate({1e-5: _run(0.97)}) == 1e-5


def test_forecast_applies_the_twelve_hour_gate_to_four_runs():
    fast = fr.forecast(0.7, 60.0, 3.0, 10.0, 56250)
    assert (fast["full_plan_runs"], fast["reduced_plan_runs"]) == (4, 3)
    assert fast["full_plan"]
    assert fast["rates"] == [1e-5, 3e-5]
    hours = (4 * fast["seconds_per_run"] + fast["finalize_seconds"]) / 3600
    assert fast["full_plan_hours"] == pytest.approx(hours)
    slow = fr.forecast(1.5, 60.0, 3.0, 10.0, 56250)
    assert not slow["full_plan"]
    assert slow["rates"] == [1e-5]


def test_stride_positions_spread_a_smoke_subset():
    assert fr.stride_positions(10, None).tolist() == list(range(10))
    assert fr.stride_positions(10, 3).tolist() == [0, 3, 6]
    assert fr.stride_positions(5, 9).tolist() == list(range(5))


# Step-0 equivalence and predict, on real validation records


def _spread_validation(count):
    data, _, _ = fdata.cache_rows(("validation",))
    sources = [np.flatnonzero(data["source"] == source)[:count] for source in md.VALIDATION_SOURCES]
    return data, np.concatenate(sources)


@needs_data
def test_step0_reproduces_the_frozen_features_and_hx_predictions_on_a_small_batch():
    data, positions = _spread_validation(2)
    seed = fr.SEEDS[0]
    model, origin = ft.initial_model(seed, "cpu")
    head = np.load(ft.CANDIDATES / ft.BASE / f"weights__seed-{seed}.npz")
    assert np.array_equal(model.head[0].weight.detach().numpy(), head["w1"])
    assert np.array_equal(model.head[3].bias.detach().numpy(), head["b2"])
    assert origin["head_seed"] == seed
    windows = fdata.read_windows(("validation",), data["record_key"][positions])
    probability, pooled = ft.evaluate(model, fdata.window_inputs(windows), "cpu", pooled=True)
    cached = fdata.cached_xecg_features(md.subset(data, positions)).astype(np.float64)
    relative = np.abs(pooled - cached).max(axis=1) / np.abs(cached).max(axis=1)
    assert relative.max() < fr.FEATURE_TOLERANCE
    path = md.prediction_path(ft.CANDIDATES, ft.BASE, "validation", (seed,))
    expected = md.load_predictions(path, data["record_key"])["probability"][positions]
    assert np.abs(probability - expected).max() < fr.STEP0_TOLERANCE


@pytest.mark.skipif(not (HAVE_F and HAVE_CACHE), reason="candidate F not trained yet")
def test_predict_reproduces_the_saved_validation_predictions():
    data, positions = _spread_validation(3)
    windows = fdata.read_windows(("validation",), data["record_key"][positions])
    index = ft.checked_index(ft.DIRECTORY)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seeds in (tuple(index["seeds"]), (index["seeds"][0],)):
        probability = ft.predict(windows, seeds=seeds, device=device)
        path = md.prediction_path(ft.CANDIDATES, ft.CANDIDATE, "validation", seeds)
        expected = md.load_predictions(path, data["record_key"])["probability"][positions]
        assert np.abs(probability - expected).max() < fr.PREDICT_TOLERANCE


def test_sessions_are_recorded_at_start_and_completed_at_the_end(tmp_path):
    path = tmp_path / "sessions.json"
    killed = {"pid": 1, "started_unix": 10.0, "status": "started"}
    fr.record_session(path, killed)
    session = {"pid": 2, "started_unix": 20.0, "status": "started"}
    fr.record_session(path, session)
    fr.record_session(path, {**session, "status": "finished", "wall_seconds": 5.0})
    sessions = fr.read_json(path)
    assert [item["status"] for item in sessions] == ["started", "finished"]
    assert sessions[1]["wall_seconds"] == 5.0


@needs_data
def test_start_check_passes_cached_features_and_refuses_a_changed_record():
    data, _, _ = fdata.cache_rows(("validation",))
    validation = {"labels": data["labels"], "sources": data["source"], "keys": data["record_key"],
                  "positions": np.arange(len(data["record_key"]))}
    cached = fdata.cached_xecg_features(data)
    seed = fr.SEEDS[0]
    path = md.prediction_path(ft.CANDIDATES, ft.BASE, "validation", (seed,))
    expected = md.load_predictions(path, data["record_key"])["probability"]
    check = fr.start_check(seed, validation, cached)
    record = check(expected, cached)
    assert record["feature_max_abs_difference"] == 0
    assert record["probability_max_abs_difference"] == 0
    assert record["step0_selection_metric"] == record["hx_selection_metric"]
    zero = int(np.flatnonzero(np.abs(cached).max(axis=1) == 0)[0]) if record["zero_feature_records"] else 0
    changed = cached.copy()
    changed[zero] += 1e-3
    with pytest.raises(RuntimeError, match="step-0 check failed"):
        check(expected, changed)
    shifted = expected.copy()
    shifted[1, 2] = min(1.0, shifted[1, 2] + 2e-4)
    with pytest.raises(RuntimeError, match="step-0 check failed"):
        check(shifted, cached)


def _profile(directory, microbatch, seconds, reserved, free):
    profile = {"microbatch": microbatch, "seconds_per_update": seconds,
               "peak_bytes": {"reserved": reserved},
               "memory_before": {"free_bytes": free, "reserved_bytes": 0},
               "forecast": {"full_plan_hours": 1.0}}
    fr.write_json_atomic(directory / f"profile__microbatch-{microbatch}.json", profile)


def test_microbatch_choice_takes_the_fastest_profile_that_fits(tmp_path):
    gib = 1024**3
    _profile(tmp_path, 8, 0.7, 5 * gib, 10 * gib)
    _profile(tmp_path, 16, 0.5, 9 * gib, 10 * gib)
    assert fr.choose_microbatch(tmp_path)["microbatch"] == 16
    _profile(tmp_path, 16, 0.5, int(9.5 * gib), 10 * gib)
    chosen = fr.choose_microbatch(tmp_path)
    assert chosen["microbatch"] == 8
    table = fr.read_json(tmp_path / "profile.json")["microbatch_choice"]["profiles"]
    assert {row["microbatch"]: row["fits"] for row in table} == {8: True, 16: False}
