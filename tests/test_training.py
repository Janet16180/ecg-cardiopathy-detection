"""Optimizer step, learning-rate schedule and profile timing helpers."""

import math

import pytest
import torch

from ecg_experiment.training import checked_step, parameter_count, time_updates, warmup_cosine_lr


@pytest.mark.parametrize("base_lr", [1e-3, 3e-4])
@pytest.mark.parametrize("epochs", [1, 10, 20])
def test_warmup_cosine_matches_the_frozen_expression_bitwise(base_lr, epochs):
    for epoch in range(epochs):
        decay = 0.1 + 0.9 * (1 + math.cos(math.pi * epoch / epochs)) / 2
        frozen = base_lr * min(1.0, (epoch + 1) / 2) * decay
        assert warmup_cosine_lr(base_lr, epoch, epochs) == frozen


def test_warmup_cosine_warms_up_then_decays_to_the_floor():
    rates = [warmup_cosine_lr(1.0, epoch, 1000) for epoch in range(1000)]
    assert rates[0] == pytest.approx(0.5, rel=1e-5)
    assert max(rates) == rates[1]
    assert rates[-1] == pytest.approx(0.1, abs=1e-5)


def test_checked_step_clips_and_updates():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = (model(torch.full((1, 2), 100.0)) ** 2).sum()
    norm = checked_step(loss, model, optimizer, "test")
    assert float(norm) > 1.0
    total = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
    assert float(total) == pytest.approx(1.0, rel=1e-5)
    assert not torch.equal(before, model.weight)


def test_checked_step_without_clipping_keeps_raw_gradients():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss = (model(torch.full((1, 2), 100.0)) ** 2).sum()
    assert checked_step(loss, model, optimizer, "test", clip=False) is None
    assert float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))) > 1.0


def test_checked_step_rejects_nonfinite_loss_and_gradients():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match="Nonfinite SSL loss"):
        checked_step(model(torch.tensor([[float("nan")]])).sum(), model, optimizer, "SSL")
    model.weight.register_hook(lambda grad: grad * float("inf"))
    loss = model(torch.tensor([[1.0]])).sum()
    with pytest.raises(RuntimeError, match="Nonfinite SSL gradients"):
        checked_step(loss, model, optimizer, "SSL")


def test_time_updates_measures_only_after_warmup():
    seen = []

    def step(batch):
        seen.append(batch)
        return batch * 10, batch

    timing = time_updates(range(1, 20), step, warmup=2, measured=3, device="cpu", too_small="small")
    assert seen == [1, 2, 3, 4, 5]
    assert timing.updates == 5
    assert timing.measured_records == 3 + 4 + 5
    assert timing.last == 50
    assert timing.seconds >= 0
    short = time_updates(range(1, 4), step, warmup=2, measured=5, device="cpu", too_small="small")
    assert short.updates == 3
    with pytest.raises(RuntimeError, match="too few"):
        time_updates(range(1, 3), step, warmup=2, measured=1, device="cpu", too_small="too few")


def test_parameter_count():
    assert parameter_count(torch.nn.Linear(3, 2)) == 8
