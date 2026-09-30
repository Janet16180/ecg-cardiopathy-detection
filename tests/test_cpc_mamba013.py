"""Independent official-reference and scientific invariance tests for 013."""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from ecg_experiment.cpc_mamba013 import (
    Mamba2Reference,
    Mamba3SISO,
    mamba2_scan,
    mamba3_siso_scan,
    matched_initial_models,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mamba013"


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _oracle():
    source = (FIXTURES / "upstream_siso_step.py.txt").read_bytes()
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert hashlib.sha256(source).hexdigest() == provenance["fixture_sha256"]
    # The unchanged official function only invokes einops.repeat when B/C
    # heads differ. Tests use already expanded heads, requiring no dependency.
    namespace = {"torch": torch, "math": math, "F": F, "Optional": Optional, "Tuple": tuple}
    exec(compile(source, "upstream_siso_step.py.txt", "exec"), namespace)
    return namespace["mamba3_siso_step_ref"]


@pytest.mark.parametrize("length", [1, 7, 79])
def test_mamba3_matches_official_recurrent_output_and_all_input_gradients(length):
    torch.manual_seed(13)
    shape = (2, length, 2)
    raw = [torch.randn(*shape, 8), torch.randn(*shape, 8), torch.randn(*shape, 4),
           -torch.rand(*shape), torch.rand(*shape), torch.randn(*shape), torch.randn(*shape, 2)]
    inputs = [x.requires_grad_() for x in raw]
    q, k, v, adt, dt, trap, angle = inputs
    actual = mamba3_siso_scan(*inputs)
    expected, _ = _oracle()(q, k, v, adt.transpose(1, 2), dt.transpose(1, 2),
                            trap.transpose(1, 2), torch.zeros(2, 8), torch.zeros(2, 8), angle)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    weights = torch.randn_like(actual)
    grads_actual = torch.autograd.grad((actual * weights).sum(), inputs, retain_graph=True)
    grads_expected = torch.autograd.grad((expected * weights).sum(), inputs)
    for actual_grad, expected_grad in zip(grads_actual, grads_expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


def test_mamba2_ssd_matches_direct_recurrence_and_gradients():
    torch.manual_seed(2)
    shape = (2, 9, 2)
    inputs = [torch.randn(*shape, 4, dtype=torch.float64).requires_grad_() for _ in range(3)]
    inputs += [(-torch.rand(*shape, dtype=torch.float64)).requires_grad_(),
               torch.rand(*shape, dtype=torch.float64).requires_grad_()]
    q, k, v, adt, dt = inputs
    state = q.new_zeros(2, 2, 4, 4)
    outputs = []
    for time in range(shape[1]):
        state = adt[:, time].exp()[..., None, None] * state
        state = state + dt[:, time, :, None, None] * k[:, time, :, :, None] * v[:, time, :, None, :]
        outputs.append(torch.einsum("bhn,bhnp->bhp", q[:, time], state))
    expected = torch.stack(outputs, 1)
    actual = mamba2_scan(*inputs)
    torch.testing.assert_close(actual, expected)
    for grad_a, grad_e in zip(torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True),
                              torch.autograd.grad(expected.square().sum(), inputs), strict=True):
        torch.testing.assert_close(grad_a, grad_e)


@pytest.mark.parametrize("mixer", [Mamba2Reference, Mamba3SISO])
def test_full_mixer_causality_batch_independence_and_optimizer_replay(mixer):
    torch.manual_seed(131)
    model = mixer(width=16, state_size=8, head_dim=8)
    data = torch.randn(2, 9, 16)
    expected = model(data)
    changed = data.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 10
    torch.testing.assert_close(model(changed)[:, :5], expected[:, :5])
    torch.testing.assert_close(model(data[:1]), expected[:1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    expected.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    restored = copy.deepcopy(model)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for instance, opt in [(model, optimizer), (restored, restored_optimizer)]:
        opt.zero_grad(set_to_none=True)
        instance(data).square().sum().backward()
        opt.step()
    for actual, reference in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_paired_common_initialization_and_half_independence():
    torch.set_num_threads(1)
    models = matched_initial_models(13042)
    for name in ("mamba2", "mamba3"):
        for attr in ("convs",):
            a = getattr(models[name].encoder, attr).state_dict()
            b = getattr(models["gru"].encoder, attr).state_dict()
            assert a.keys() == b.keys()
            assert all(torch.equal(a[key], b[key]) for key in a)
        assert all(torch.equal(value, models["gru"].heads.state_dict()[key])
                   for key, value in models[name].heads.state_dict().items())
        data = torch.randn(1, 12, 2500)
        _, context = models[name].encoder(data)
        data[:, :, 1250:] = torch.randn_like(data[:, :, 1250:])
        _, changed = models[name].encoder(data)
        torch.testing.assert_close(context[:, 0], changed[:, 0], rtol=0, atol=0)
        assert models[name].encoder.pooled(context).shape == (1, 512)


@pytest.mark.parametrize("arm", ["gru", "mamba2", "mamba3"])
def test_full_cpc_disk_checkpoint_rng_and_optimizer_resume(tmp_path, arm):
    from ecg_experiment.files import write_torch_atomic
    from ecg_experiment.training import checked_step

    torch.manual_seed(13042)
    model = matched_initial_models(13042)[arm].train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    signal = torch.randn(1, 12, 2500)
    loss, _ = model(signal)
    checked_step(loss, model, optimizer, "CPU initial")
    checkpoint = tmp_path / "latest.pt"
    write_torch_atomic(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "rng": torch.get_rng_state(), "completed_updates": 1})
    reference_input = signal + 0.01 * torch.randn_like(signal)
    expected_loss, _ = model(reference_input)
    checked_step(expected_loss, model, optimizer, "CPU reference")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = matched_initial_models(13042)[arm].train()
    restored.load_state_dict(saved["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0.01)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    replay_input = signal + 0.01 * torch.randn_like(signal)
    actual_loss, _ = restored(replay_input)
    checked_step(actual_loss, restored, resumed_optimizer, "CPU resumed")
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    expected_state = optimizer.state_dict()["state"]
    actual_state = resumed_optimizer.state_dict()["state"]
    for index, state in expected_state.items():
        for key, value in state.items():
            torch.testing.assert_close(actual_state[index][key], value, atol=0, rtol=0)


def test_loader_resume_does_not_consume_model_rng():
    import numpy as np
    from torch.utils.data import TensorDataset

    from ecg_experiment.mamba25k013 import _training_loader

    data = TensorDataset(torch.arange(8))
    before = torch.get_rng_state().clone()
    loader = _training_loader(data, np.arange(8))
    loader.num_workers = 0
    loader.pin_memory = False
    assert torch.equal(next(iter(loader))[0], torch.arange(8))
    assert torch.equal(before, torch.get_rng_state())


def test_profile_gate_rejects_failed_or_different_identity(tmp_path, monkeypatch):
    from ecg_experiment import mamba25k013

    monkeypatch.setattr(mamba25k013, "OUTPUT", tmp_path)
    for receipt in ({"identity": {"seed": 1}, "gate_passed": False},
                    {"identity": {"seed": 2}, "gate_passed": True}):
        (tmp_path / "profile.json").write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="Matching passed"):
            mamba25k013._check_profile({"seed": 1})
    (tmp_path / "profile.json").write_text(json.dumps({"identity": {"seed": 1}, "gate_passed": True}))
    mamba25k013._check_profile({"seed": 1})


def test_prespecified_mamba_contrasts_use_paired_patient_draws():
    import numpy as np

    from ecg_experiment.mamba25k013_stats import ARMS, paired_patient_auc

    targets = np.asarray([0, 1, 0, 1, 0, 1])
    patients = np.asarray(["a", "a", "b", "b", "c", "c"])
    predictions = {arm: np.asarray([0.1, 0.9, 0.3, 0.7, 0.2, 0.8]) for arm in ARMS}
    results = paired_patient_auc(targets, patients, predictions, draws=12)
    assert set(results) == {"mamba3_minus_mamba2", "mamba2_minus_gru", "mamba3_minus_gru"}
    assert all(row["ci95"] == [0, 0] and row["valid_draws"] == 12 for row in results.values())


def test_complete_io_profile_covers_every_selected_row(monkeypatch):
    import numpy as np
    from torch.utils.data import TensorDataset

    from ecg_experiment import mamba25k013

    monkeypatch.setattr(mamba25k013, "SUBSET_SIZE", 5)
    data = TensorDataset(torch.ones(8, 12, 10), torch.arange(8), torch.arange(8))
    seen = []
    original = mamba25k013._training_loader

    def loader(dataset, indices):
        seen.extend(indices.tolist())
        result = original(dataset, indices)
        result.num_workers = 0
        return result

    monkeypatch.setattr(mamba25k013, "_training_loader", loader)
    order = np.asarray([7, 3, 1, 6, 0, 3, 7])
    receipt = mamba25k013._measure_selected_io(data, order)
    assert receipt["records"] == 5
    assert seen == order[:5].tolist()
    with pytest.raises(ValueError, match="complete selected-row"):
        mamba25k013._measure_selected_io(data, np.asarray([1, 1, 2, 3, 4]))
    with pytest.raises(ValueError, match="complete selected-row"):
        mamba25k013._measure_selected_io(data, order[:4])
