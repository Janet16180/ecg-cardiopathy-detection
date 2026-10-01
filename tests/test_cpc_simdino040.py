"""Scientific invariants and exact replay for the controlled CPC/SimDINO suite."""

from __future__ import annotations

import io
import math

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from ecg_experiment.cpc import HALF_SAMPLES, SIGNAL_SAMPLES, TOKEN_COUNT, WIDTH, cpc_loss
from ecg_experiment.cpc_encoder_variants039 import create_model as patch_create_model
from ecg_experiment.cpc_simdino040 import (
    CODING_EPS,
    CODING_WEIGHT,
    CONTEXTS,
    EMA_END,
    EMA_START,
    OBJECTIVES,
    RECEPTIVE_FIELD,
    architecture_spec,
    create_model,
    distillation_parts,
    masked_views,
    upstream_loss,
)
from ecg_experiment.reproducibility import cpu_state


@pytest.fixture(autouse=True)
def cpu_thread() -> None:
    torch.set_num_threads(1)


@pytest.mark.parametrize("context", CONTEXTS)
def test_causal_support_half_independence_and_unchanged_shapes(context: str) -> None:
    model = create_model("cpc", context, 39042).eval()
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    with torch.inference_mode():
        tokens, contexts = model.encoder(signal)
        assert tokens.shape == contexts.shape == (2, 2, TOKEN_COUNT, WIDTH)
        assert model.encoder.pooled(contexts).shape == (2, 512)
        for half in (0, 1):
            changed = signal.clone()
            changed[:, :, half * HALF_SAMPLES : (half + 1) * HALF_SAMPLES] += 5
            other_tokens, other_contexts = model.encoder(changed)
            torch.testing.assert_close(other_tokens[:, 1 - half], tokens[:, 1 - half], rtol=0, atol=0)
            torch.testing.assert_close(other_contexts[:, 1 - half], contexts[:, 1 - half], rtol=0, atol=0)
        for cutoff in (0, 16, 517, 1248):
            changed = signal.clone()
            for half in (0, 1):
                start, end = half * HALF_SAMPLES + cutoff + 1, (half + 1) * HALF_SAMPLES
                changed[:, :, start:end] += 10 * torch.randn_like(changed[:, :, start:end])
            other_tokens, other_contexts = model.encoder(changed)
            prefix = torch.arange(TOKEN_COUNT) * 16 <= cutoff
            torch.testing.assert_close(other_tokens[:, :, prefix], tokens[:, :, prefix], rtol=0, atol=0)
            torch.testing.assert_close(other_contexts[:, :, prefix], contexts[:, :, prefix], rtol=0, atol=0)
    raw = torch.randn(1, 12, HALF_SAMPLES, requires_grad=True)
    tokens = model.encoder.convs(raw)
    for index in (0, 1, 31, 78):
        (gradient,) = torch.autograd.grad(tokens[0, 0, index], raw, retain_graph=True)
        active = gradient[0].abs().sum(dim=0).nonzero().flatten()
        assert int(active.min()) == max(0, index * 16 - RECEPTIVE_FIELD + 1)
        assert int(active.max()) == index * 16
        assert len(active) == min(RECEPTIVE_FIELD, index * 16 + 1)
    assert 4 * 16 - RECEPTIVE_FIELD + 1 > 0


@pytest.mark.parametrize("context", CONTEXTS)
def test_original_patch_context_heads_and_matched_objective_initialization(context: str) -> None:
    before = torch.get_rng_state().clone()
    baseline = patch_create_model("patch", context, 39042)
    models = [create_model(objective, context, 39042) for objective in OBJECTIVES]
    assert torch.equal(before, torch.get_rng_state())
    for model in models:
        for name, tensor in baseline.state_dict().items():
            assert torch.equal(tensor, model.state_dict()[name]), name
        for name, tensor in models[0].state_dict().items():
            assert torch.equal(tensor, model.state_dict()[name]), name
        for name, tensor in model.encoder.state_dict().items():
            assert torch.equal(tensor, model.teacher.state_dict()[name]), name
        assert all(not p.requires_grad for p in model.teacher.parameters())
        assert model.teacher.training is False
    repeated = create_model("hybrid", context, 39042)
    assert all(
        torch.equal(tensor, repeated.state_dict()[name]) for name, tensor in models[-1].state_dict().items()
    )


def test_frontend_and_cpc_heads_match_across_contexts() -> None:
    gru = create_model("hybrid", "gru", 39042)
    xlstm = create_model("hybrid", "xlstm", 39042)
    for name, tensor in gru.state_dict().items():
        if name.startswith(("encoder.convs.", "heads.")):
            assert torch.equal(tensor, xlstm.state_dict()[name]), name


def test_raw_masks_cover_full_support_all_leads_without_changing_other_samples() -> None:
    signal = torch.randn(3, 12, SIGNAL_SAMPLES) + 3
    views, token_masks, starts = masked_views(signal, torch.Generator().manual_seed(39042))
    assert views.shape == (2, 3, 12, SIGNAL_SAMPLES)
    assert token_masks.shape == (2, 3, 2, TOKEN_COUNT)
    assert torch.all(token_masks.sum(dim=-1) == 8)
    assert torch.all((starts >= 3) & (starts <= 71))
    for view in range(2):
        for record in range(3):
            expected = signal[record].clone()
            for half in (0, 1):
                first = int(starts[view, record, half])
                start = half * HALF_SAMPLES + first * 16 - 48
                end = half * HALF_SAMPLES + (first + 7) * 16
                expected[:, start : end + 1] = 0
                assert end - start + 1 == 161
                for token in range(first, first + 8):
                    support_start = half * HALF_SAMPLES + token * 16 - 48
                    support_end = half * HALF_SAMPLES + token * 16
                    assert torch.count_nonzero(views[view, record, :, support_start : support_end + 1]) == 0
            assert torch.equal(expected, views[view, record])
    assert not torch.equal(starts[0], starts[1])


def test_mask_boundary_starts_are_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    starts = torch.tensor([[[3, 71]], [[71, 3]]])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: starts)
    signal = torch.ones(1, 12, SIGNAL_SAMPLES)
    views, masks, actual_starts = masked_views(signal, torch.Generator())
    assert torch.equal(starts, actual_starts)
    assert masks[0, 0, 0, 3:11].all()
    assert masks[0, 0, 1, 71:79].all()
    assert torch.count_nonzero(views[0, 0, :, :161]) == 0
    assert torch.count_nonzero(views[0, 0, :, 2338:2499]) == 0
    assert views[0, 0, :, 2499].eq(1).all()


def test_normalized_upstream_coding_rate_equivalence_gradients_and_actual_batch() -> None:
    torch.manual_seed(40)
    student = torch.randn(2, 16, 2, TOKEN_COUNT, WIDTH, requires_grad=True)
    teacher = torch.randn(16, 2, TOKEN_COUNT, WIDTH, requires_grad=True)
    masks = torch.zeros(2, 16, 2, TOKEN_COUNT, dtype=torch.bool)
    masks[..., 9:17] = True
    upstream = upstream_loss()
    parts = distillation_parts(student, teacher, masks, upstream)
    reference_student = student.detach().clone().requires_grad_()
    global_features = F.normalize(reference_student.mean(dim=(2, 3)), dim=-1)
    expected = upstream.calc_expansion(global_features)
    assert upstream.eps == CODING_EPS
    torch.testing.assert_close(parts["rate"], expected, rtol=0, atol=0)
    (actual_gradient,) = torch.autograd.grad(parts["rate"], student, retain_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected, reference_student, retain_graph=True)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    for name in ("global", "patch", "rate", "simdino"):
        assert torch.isfinite(parts[name])
    assert parts["simdino"] == parts["global"] + CODING_WEIGHT * parts["rate"] + parts["patch"]
    parts["simdino"].backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None
    gram = global_features @ global_features.transpose(-1, -2)
    scalar = WIDTH / (16 * CODING_EPS)
    rate = -0.5 * torch.linalg.slogdet(torch.eye(16) + scalar * gram).logabsdet.mean()
    rate *= CODING_EPS * math.sqrt(16 / (WIDTH * min(WIDTH, 16)))
    torch.testing.assert_close(parts["rate"], rate, rtol=1e-4, atol=1e-6)
    (sample_gradient,) = torch.autograd.grad(rate, reference_student)
    torch.testing.assert_close(actual_gradient, sample_gradient, rtol=1e-3, atol=1e-7)


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_finite_gradients_and_expected_head_movement(objective: str, context: str) -> None:
    model = create_model(objective, context, 39042)
    before = cpu_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, parts = model(torch.randn(2, 12, SIGNAL_SAMPLES))
    assert torch.isfinite(loss)
    assert parts["cmsc"] == 0
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("teacher.") or (name.startswith("heads.") and objective == "simdino"):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    state = model.state_dict()
    assert any(
        not torch.equal(state[name], tensor) for name, tensor in before.items() if name.startswith("encoder.")
    )
    head_changed = any(
        not torch.equal(state[name], tensor) for name, tensor in before.items() if name.startswith("heads.")
    )
    assert head_changed == (objective != "simdino")
    assert all(
        torch.equal(state[name], tensor) for name, tensor in before.items() if name.startswith("teacher.")
    )
    assert int(model.ema_updates) == 0


def test_pure_cpc_exact_clean_loss_and_inactive_teacher_rng() -> None:
    model = create_model("cpc", "gru", 39042)
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    rng = torch.get_rng_state().clone()
    mask_state = model.mask_rng_state.clone()
    tokens, contexts = model.encoder(signal)
    expected = cpc_loss(tokens, contexts, model.heads)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual, parts = model(signal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(mask_state, model.mask_rng_state)
    assert parts["global"] == parts["rate"] == parts["patch"] == parts["simdino"] == 0
    assert model.after_step(10) is None
    assert int(model.ema_updates) == 0


def test_teacher_eval_delayed_exact_ema_endpoints_and_schedule_checks() -> None:
    model = create_model("hybrid", "gru", 39042).train()
    assert model.teacher.training is False
    before = cpu_state(model.teacher)
    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1)
    assert all(torch.equal(tensor, model.teacher.state_dict()[name]) for name, tensor in before.items())
    momentum = model.after_step(2)
    assert momentum == pytest.approx(EMA_END - (EMA_END - EMA_START) * (1 + math.cos(math.pi / 2)) / 2)
    for name, tensor in before.items():
        expected = tensor * momentum + model.encoder.state_dict()[name] * (1 - momentum)
        torch.testing.assert_close(model.teacher.state_dict()[name], expected, rtol=0, atol=2e-7)
    assert int(model.ema_updates) == 1
    assert model.after_step(2) == pytest.approx(EMA_END)
    with pytest.raises(ValueError, match="exhausted"):
        model.after_step(2)
    with pytest.raises(ValueError, match="unchanged schedule"):
        model.after_step(3)
    model.eval().train()
    assert model.teacher.training is False


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("objective", ["simdino", "hybrid"])
def test_checkpoint_resume_exact_next_loss_update_teacher_and_rng(objective: str, context: str) -> None:
    torch.manual_seed(4040)
    model = create_model(objective, context, 39042)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)

    def update() -> tuple[float, dict[str, float]]:
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model(signal)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        model.after_step(10)
        return float(loss.detach()), parts

    update()
    checkpoint = io.BytesIO()
    torch.save(
        {"model": cpu_state(model), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()},
        checkpoint,
    )
    expected = update()
    expected_state = cpu_state(model)
    expected_rng = torch.get_rng_state().clone()
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    actual = update()
    assert actual == expected
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, tensor in expected_state.items():
        assert torch.equal(tensor, model.state_dict()[name]), name


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_receipt_parameter_counts_and_upstream_binding(objective: str) -> None:
    spec = architecture_spec(objective, "gru")
    assert spec["global_width"] == WIDTH
    assert spec["receptive_field_samples"] == 49
    assert spec["upstream_source"].endswith("loss_utils.py")
    assert spec["student_parameters"] == (
        spec["frontend_parameters"] + spec["context_parameters"] + spec["cpc_head_parameters"]
    )
    assert spec["teacher_parameters"] == spec["frontend_parameters"] + spec["context_parameters"]


@pytest.mark.parametrize(("objective", "context"), [("unknown", "gru"), ("cpc", "unknown")])
def test_unknown_model_names_raise(objective: str, context: str) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        create_model(objective, context, 39042)


@pytest.mark.parametrize("context", CONTEXTS)
def test_auxiliary_dropout_preserves_clean_global_stream_and_advances_local_state(context: str) -> None:
    model = create_model("hybrid", context, 39042)
    signal = torch.randn(2, 12, SIGNAL_SAMPLES)
    views, _, _ = model.draw_views(signal)
    global_state = torch.get_rng_state().clone()
    local_state = model.aux_dropout_rng_state.clone()
    first = model.masked_contexts(views)
    assert torch.equal(global_state, torch.get_rng_state())
    assert not torch.equal(local_state, model.aux_dropout_rng_state)
    model.aux_dropout_rng_state.copy_(local_state)
    replay = model.masked_contexts(views)
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    assert torch.equal(global_state, torch.get_rng_state())
    cpc = create_model("cpc", context, 39042)
    torch.set_rng_state(global_state)
    expected_loss, _ = cpc(signal)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(global_state)
    _, observed_parts = model(signal)
    assert observed_parts["cpc"] == float(expected_loss.detach())
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_coding_rate_rejects_nonfinite_factorization() -> None:
    student = torch.randn(2, 2, 2, TOKEN_COUNT, WIDTH)
    student[0, 0, 0, 0, 0] = torch.nan
    teacher = torch.randn(2, 2, TOKEN_COUNT, WIDTH)
    mask = torch.ones(2, 2, 2, TOKEN_COUNT, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="finite positive definite"):
        distillation_parts(student, teacher, mask, upstream_loss())
