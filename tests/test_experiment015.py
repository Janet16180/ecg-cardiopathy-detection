"""Meaningful local checks for the matched cached-teacher pilot."""

import numpy as np
import torch

from scripts.experiments.run_jepa_cpc_distillation import fixed_batches, load_state, masked_loss, save_state

torch.set_num_threads(1)


def test_label_mask_teacher_stop_gradient_and_control_graph():
    logits = torch.tensor([0.2, -0.4, 1.1], requires_grad=True)
    target = torch.tensor([1.0, 1.0, 0.0])
    mask = torch.tensor([True, False, True])
    projected = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
    teacher = torch.tensor([[0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    loss, bce, cosine = masked_loss(logits, projected, target, mask, teacher, 0.1)
    loss.backward()
    assert torch.allclose(bce, torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask]))
    assert logits.grad[1] == 0
    assert projected.grad.abs().sum() > 0
    assert teacher.grad is None
    assert cosine > 0
    logits.grad = projected.grad = None
    control, _, _ = masked_loss(logits, projected, target, mask, teacher, 0.0)
    control.backward()
    assert torch.all(projected.grad == 0)
    assert teacher.grad is None


def test_batches_are_deterministic_complete_and_matched():
    exposed = np.zeros(15360, dtype=bool)
    exposed[np.random.default_rng(42).choice(15360, 1518, replace=False)] = True
    a = fixed_batches(15360, exposed, 0)
    b = fixed_batches(15360, exposed, 0)
    assert a == b and a != fixed_batches(15360, exposed, 1)
    assert len(a) == 120 and all(len(batch) == 128 and exposed[batch].any() for batch in a)
    assert sorted(i for batch in a for i in batch) == list(range(15360))
    assert sum(int(exposed[batch].sum()) for batch in a) == 1518


def test_mid_epoch_checkpoint_restores_optimizer_and_rng(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.ones(4, 3)
    model(x).square().mean().backward()
    optimizer.step()
    save_state(tmp_path, "fixed", model, optimizer, 2, 17, [{"epoch": 1}],
               {"auc": 0.6, "epoch": 1, "model": None}, {"updates": 17}, 9.2)
    expected = torch.rand(4)
    reloaded = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.01)
    epoch, batch, history, best, totals, elapsed = load_state(tmp_path, "fixed", reloaded, resumed_optimizer)
    assert (epoch, batch, history, totals, elapsed) == (2, 17, [{"epoch": 1}], {"updates": 17}, 9.2)
    assert best["auc"] == 0.6
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), reloaded.parameters()))
    for key, value in optimizer.state_dict()["state"].items():
        assert all(torch.equal(v, resumed_optimizer.state_dict()["state"][key][name])
                   for name, v in value.items() if torch.is_tensor(v))
    assert torch.equal(expected, torch.rand(4))
