"""Causal local Transformer CPC with controlled SimDINOv2 objectives.

The imported Experiment 039 patch projection spans 33 raw samples. A single
attention block accesses only the current and preceding token, extending support
to 49 samples with unchanged sixteen-sample spacing. The EMA teacher is outside
the student encoder and never supplies CPC targets or prediction-head inputs.
"""

from __future__ import annotations

import copy
import importlib.util
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ecg_experiment import ROOT
from ecg_experiment.cpc import HALF_SAMPLES, HORIZONS, LEADS, TOKEN_COUNT, WIDTH, check_signal_batch, cpc_loss
from ecg_experiment.cpc_encoder_variants039 import PatchFrontend
from ecg_experiment.cpc_encoder_variants039 import create_model as patch_create_model

OBJECTIVES = ("cpc", "simdino", "hybrid")
CONTEXTS = ("gru", "xlstm")
ATTENTION_HEADS = 4
FEEDFORWARD_WIDTH = 512
RECEPTIVE_FIELD = 49
TOKEN_STRIDE = 16
MASK_TOKENS = 8
MASK_SEED_OFFSET = 400
AUXILIARY_SEED_OFFSET = 401
CUDA_RNG_CAPACITY = 8192
CODING_EPS = 0.05
CODING_WEIGHT = 0.1
NORMALIZATION_EPS = 1e-8
EMA_START = 0.99
EMA_END = 0.9999
UPSTREAM_SOURCE = "third_party/bench-xecg/bench_xecg/utils/loss_utils.py"


def upstream_loss() -> nn.Module:
    """Import the unchanged upstream coding-rate implementation.

    Returns
    -------
    nn.Module
        SimDINOv2 loss configured with the prespecified coding-rate epsilon.

    Raises
    ------
    ImportError
        If the vendored source cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location("_ecg_simdino040_loss_utils", ROOT / UPSTREAM_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import the vendored SimDINOv2 loss")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimDINOv2Loss(eps=CODING_EPS, coeff=CODING_WEIGHT)


class LocalTransformerBlock(nn.Module):
    """One pre-normalized causal attention/MLP residual block without dropout."""

    def __init__(self) -> None:
        """Build four-head attention and a width-512 feedforward layer."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, ATTENTION_HEADS, dropout=0, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(WIDTH)
        self.feedforward = nn.Sequential(
            nn.Linear(WIDTH, FEEDFORWARD_WIDTH), nn.GELU(), nn.Linear(FEEDFORWARD_WIDTH, WIDTH)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend to the current and immediately preceding token only.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape [independent halves, time, 256].

        Returns
        -------
        torch.Tensor
            Updated tokens with unchanged shape and causal alignment.
        """
        position = torch.arange(x.shape[1], device=x.device)
        separation = position[:, None] - position[None, :]
        mask = (separation < 0) | (separation > 1)
        normalized = self.attention_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.feedforward(self.feedforward_norm(x))


class TransformerFrontend(PatchFrontend):
    """Preserve the frozen patch tensors and append one local attention block."""

    def __init__(self, patch: PatchFrontend) -> None:
        """Reuse a matched patch projection and construct the extra block.

        Parameters
        ----------
        patch : PatchFrontend
            Frozen-factory initialized patch projection and normalization.
        """
        nn.Module.__init__(self)
        self.projection = patch.projection
        self.norm = patch.norm
        self.block = LocalTransformerBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 79 channel-first tokens with maximum support of 49 samples.

        Parameters
        ----------
        x : torch.Tensor
            Independent raw halves of shape [batch, 12, 1250].

        Returns
        -------
        torch.Tensor
            Local Transformer tokens [batch, 256, 79].
        """
        tokens = super().forward(x).transpose(1, 2)
        return self.block(tokens).transpose(1, 2)


def masked_views(
    signal: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw two independent raw masks covering every selected token's support.

    Parameters
    ----------
    signal : torch.Tensor
        Normalized full records [batch, 12, 2500].
    generator : torch.Generator
        Explicit CPU generator whose state the caller checkpoints.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Views [2, batch, 12, 2500], token masks [2, batch, 2, 79] and starts
        [2, batch, 2]. All twelve leads are zero over [16*a-48, 16*(a+7)].
    """
    check_signal_batch(signal)
    starts = torch.randint(3, 72, (2, len(signal), 2), generator=generator).to(signal.device)
    tokens = torch.arange(TOKEN_COUNT, device=signal.device)
    token_mask = (tokens >= starts[..., None]) & (tokens < starts[..., None] + MASK_TOKENS)
    samples = torch.arange(HALF_SAMPLES, device=signal.device)
    raw_mask = (samples >= TOKEN_STRIDE * starts[..., None] - RECEPTIVE_FIELD + 1) & (
        samples <= TOKEN_STRIDE * (starts[..., None] + MASK_TOKENS - 1)
    )
    raw_mask = raw_mask.flatten(-2)
    views = signal.unsqueeze(0).expand(2, -1, -1, -1).masked_fill(raw_mask.unsqueeze(2), 0)
    return views, token_mask, starts


def distillation_parts(
    student: torch.Tensor, teacher: torch.Tensor, token_mask: torch.Tensor, loss_module: nn.Module
) -> dict[str, torch.Tensor]:
    """Compute normalized global, coding-rate and aligned masked-context losses.

    Parameters
    ----------
    student : torch.Tensor
        Two student context views [2, batch, 2, 79, 256].
    teacher : torch.Tensor
        Clean detached teacher contexts [batch, 2, 79, 256].
    token_mask : torch.Tensor
        Selected token positions [2, batch, 2, 79].
    loss_module : nn.Module
        Imported SimDINOv2 loss exposing ``calc_expansion``.

    Returns
    -------
    dict[str, torch.Tensor]
        Scalar ``global``, negative ``rate``, ``patch`` and combined ``simdino``.

    Raises
    ------
    FloatingPointError
        If coding-rate factorization is nonfinite or not positive definite.
    """
    teacher = teacher.detach()
    student_global = F.normalize(student.mean(dim=(2, 3)), dim=-1, eps=NORMALIZATION_EPS)
    teacher_global = F.normalize(teacher.mean(dim=(1, 2)), dim=-1, eps=NORMALIZATION_EPS)
    global_loss = (1 - (student_global * teacher_global.unsqueeze(0)).sum(dim=-1)).mean()
    normalized_student = F.normalize(student, dim=-1, eps=NORMALIZATION_EPS)
    normalized_teacher = F.normalize(teacher, dim=-1, eps=NORMALIZATION_EPS).unsqueeze(0)
    patch_loss = (1 - (normalized_student * normalized_teacher).sum(dim=-1))[token_mask].mean()
    with torch.autocast(device_type=student.device.type, enabled=False):
        features = student_global.float()
        with torch.no_grad():
            width, batch = features.shape[-1], features.shape[1]
            matrix = torch.eye(width, device=features.device) + width / (batch * CODING_EPS) * (
                features.transpose(-1, -2) @ features
            )
            factor, info = torch.linalg.cholesky_ex(matrix)
            if info.any() or not torch.isfinite(factor).all():
                raise FloatingPointError("Coding-rate factorization must be finite positive definite")
        rate = loss_module.calc_expansion(features)
    return {
        "global": global_loss,
        "rate": rate,
        "patch": patch_loss,
        "simdino": global_loss + CODING_WEIGHT * rate + patch_loss,
    }


class SimDINOPretrainer(nn.Module):
    """One matched student backbone and an inactive-or-EMA clean teacher."""

    def __init__(self, base: nn.Module, objective: str, seed: int) -> None:
        """Attach the shared student, detached teacher and portable mask RNG.

        Parameters
        ----------
        base : nn.Module
            Deterministically initialized ordinary CPC model.
        objective : str
            ``cpc``, ``simdino`` or ``hybrid``.
        seed : int
            Initialization seed; masks use seed plus 400.
        """
        super().__init__()
        self.encoder = base.encoder
        self.heads = base.heads
        self.hybrid = False
        self.objective = objective
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.simdino_loss = upstream_loss()
        generator = torch.Generator(device="cpu").manual_seed(seed + MASK_SEED_OFFSET)
        self.register_buffer("mask_rng_state", generator.get_state())
        auxiliary = torch.Generator(device="cpu").manual_seed(seed + AUXILIARY_SEED_OFFSET)
        self.register_buffer("aux_dropout_rng_state", auxiliary.get_state())
        self.register_buffer("aux_cuda_rng_state", torch.zeros(CUDA_RNG_CAPACITY, dtype=torch.uint8))
        self.register_buffer("aux_cuda_rng_length", torch.zeros((), dtype=torch.int64))
        self.register_buffer(
            "aux_dropout_seed", torch.tensor(seed + AUXILIARY_SEED_OFFSET, dtype=torch.int64)
        )
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.int64))
        self.register_buffer("ema_total_steps", torch.zeros((), dtype=torch.int64))

    def train(self, mode: bool = True) -> SimDINOPretrainer:
        """Change student mode while retaining the teacher in evaluation mode.

        Parameters
        ----------
        mode : bool
            Requested student training mode.

        Returns
        -------
        SimDINOPretrainer
            This model, with a deterministic frozen teacher.
        """
        super().train(mode)
        self.teacher.eval()
        return self

    def draw_views(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw views and persist the next CPU-generator state on any device.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized full-record student input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Masked views, aligned selected-token masks and mask starts.
        """
        generator = torch.Generator(device="cpu")
        generator.set_state(self.mask_rng_state.cpu())
        views = masked_views(signal, generator)
        self.mask_rng_state.copy_(generator.get_state().to(self.mask_rng_state.device))
        return views

    def masked_contexts(self, views: torch.Tensor) -> torch.Tensor:
        """Encode masked views with checkpointed dropout isolated from clean CPC.

        Parameters
        ----------
        views : torch.Tensor
            Masked full records [2, batch, 12, 2500].

        Returns
        -------
        torch.Tensor
            Student contexts [2, batch, 2, 79, 256], preserving global RNG states.

        Raises
        ------
        ValueError
            If the CUDA RNG state exceeds its portable fixed-size buffer.
        """
        device = views.device
        devices = []
        if device.type == "cuda":
            devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.aux_dropout_rng_state.cpu())
            if device.type == "cuda":
                length = int(self.aux_cuda_rng_length)
                if length:
                    state = self.aux_cuda_rng_state[:length].cpu()
                else:
                    state = torch.Generator(device=device).manual_seed(int(self.aux_dropout_seed)).get_state()
                torch.cuda.set_rng_state(state, device)
            contexts = torch.stack([self.encoder(view)[1] for view in views])
            self.aux_dropout_rng_state.copy_(torch.get_rng_state().to(self.aux_dropout_rng_state.device))
            if device.type == "cuda":
                state = torch.cuda.get_rng_state(device)
                if len(state) > CUDA_RNG_CAPACITY:
                    raise ValueError("CUDA RNG state exceeds its portable buffer")
                self.aux_cuda_rng_state[: len(state)].copy_(state.to(self.aux_cuda_rng_state.device))
                self.aux_cuda_rng_length.fill_(len(state))
        return contexts

    def forward(
        self, signal: torch.Tensor, patient_ids: Sequence[str] | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the controlled clean CPC and/or masked distillation objective.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized full records [batch, 12, 2500].
        patient_ids : Sequence[str] or None
            Accepted for CPC interface compatibility; these objectives use no CMSC.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar objective and detached component values. Forward never updates EMA.
        """
        check_signal_batch(signal)
        parts = {name: signal.new_zeros(()) for name in ("cpc", "cmsc", "global", "rate", "patch", "simdino")}
        if self.objective != "simdino":
            tokens, contexts = self.encoder(signal)
            parts["cpc"] = cpc_loss(tokens, contexts, self.heads)
        if self.objective != "cpc":
            with torch.no_grad():
                _, teacher_contexts = self.teacher(signal)
            views, masks, _ = self.draw_views(signal)
            student_contexts = self.masked_contexts(views)
            parts.update(distillation_parts(student_contexts, teacher_contexts, masks, self.simdino_loss))
        loss = parts["cpc"] + parts["simdino"]
        return loss, {name: float(value.detach()) for name, value in parts.items()}

    @torch.no_grad()
    def after_step(self, total_steps: int) -> float | None:
        """Update the teacher only after the runner's successful optimizer step.

        Parameters
        ----------
        total_steps : int
            Frozen number of successful training updates in the cosine schedule.

        Returns
        -------
        float or None
            Applied EMA momentum; None for the inactive pure-CPC teacher.

        Raises
        ------
        ValueError
            If the schedule is invalid, changed on resume or exhausted.
        """
        if self.objective == "cpc":
            return None
        if total_steps < 2 or int(self.ema_total_steps) not in (0, total_steps):
            raise ValueError("EMA requires an unchanged schedule of at least two steps")
        updates = int(self.ema_updates)
        if updates >= total_steps:
            raise ValueError("EMA schedule is exhausted")
        fraction = (updates + 1) / total_steps
        momentum = EMA_END - (EMA_END - EMA_START) * (1 + math.cos(math.pi * fraction)) / 2
        for target, source in zip(self.teacher.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(momentum).add_(source.detach(), alpha=1 - momentum)
        for target, source in zip(self.teacher.buffers(), self.encoder.buffers(), strict=True):
            target.copy_(source)
        self.ema_updates.add_(1)
        self.ema_total_steps.fill_(total_steps)
        return momentum


def create_model(
    objective: str, context: str, seed: int, device: torch.device | str = "cpu"
) -> SimDINOPretrainer:
    """Build the same matched Transformer backbone for all three objectives.

    Parameters
    ----------
    objective : str
        ``cpc``, ``simdino`` or ``hybrid``.
    context : str
        Frozen native ``gru`` or ``xlstm`` context.
    seed : int
        Existing Experiment 039 initialization seed.
    device : torch.device or str
        Destination after isolated deterministic CPU initialization.

    Returns
    -------
    SimDINOPretrainer
        Model retaining encoder/head checkpoint prefixes and an external teacher.

    Raises
    ------
    ValueError
        If either architecture name is unknown.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if context not in CONTEXTS:
        raise ValueError(f"Unknown context: {context}")
    base = patch_create_model("patch", context, seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 40)
        base.encoder.convs = TransformerFrontend(base.encoder.convs)
    return SimDINOPretrainer(base, objective, seed).to(device)


def architecture_spec(objective: str, context: str) -> dict[str, Any]:
    """Describe fixed architecture, objectives and measured parameter groups.

    Parameters
    ----------
    objective : str
        Controlled objective accepted by ``create_model``.
    context : str
        Frozen context accepted by ``create_model``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible configuration and exact parameter counts for receipts.
    """
    model = create_model(objective, context, 39042)
    return {
        "objective": objective,
        "context": context,
        "frontend": "local_transformer",
        "patch_samples": 33,
        "token_stride": TOKEN_STRIDE,
        "tokens_per_half": TOKEN_COUNT,
        "receptive_field_samples": RECEPTIVE_FIELD,
        "width": WIDTH,
        "attention_blocks": 1,
        "attention_heads": ATTENTION_HEADS,
        "attention_key_offsets": [-1, 0],
        "feedforward_width": FEEDFORWARD_WIDTH,
        "attention_dropout": 0,
        "pre_layer_norm": True,
        "half_independence": True,
        "cpc_horizons": list(HORIZONS),
        "clean_cpc": True,
        "teacher_cpc": False,
        "teacher_active": objective != "cpc",
        "student_views": 2,
        "clean_teacher_views": 1,
        "mask_tokens_per_half": MASK_TOKENS,
        "mask_start_inclusive": [3, 71],
        "mask_all_leads": LEADS,
        "mask_raw_interval": "[16*a-48, 16*(a+7)] inclusive",
        "mask_seed_offset": MASK_SEED_OFFSET,
        "coding_eps": CODING_EPS,
        "auxiliary_dropout_seed_offset": AUXILIARY_SEED_OFFSET,
        "normalization_eps": NORMALIZATION_EPS,
        "coding_weight": CODING_WEIGHT,
        "global_width": WIDTH,
        "global_pool": "mean over both halves and every context token",
        "simdino_weight": 1.0,
        "ema_momentum": [EMA_START, EMA_END],
        "ema_schedule": "mu(u)=end-(end-start)*(1+cos(pi*u/total_steps))/2; completed u=1..total_steps",
        "upstream_source": UPSTREAM_SOURCE,
        "parameter_prefixes": {
            "frontend": "encoder.convs.",
            "context": "encoder.context.",
            "cpc_heads": "heads.",
            "teacher": "teacher.",
        },
        "frontend_parameters": sum(p.numel() for p in model.encoder.convs.parameters()),
        "context_parameters": sum(p.numel() for p in model.encoder.context.parameters()),
        "cpc_head_parameters": sum(p.numel() for p in model.heads.parameters()),
        "student_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "active_student_parameters": (
            sum(p.numel() for p in model.encoder.parameters())
            + (sum(p.numel() for p in model.heads.parameters()) if objective != "simdino" else 0)
        ),
        "teacher_parameters": sum(p.numel() for p in model.teacher.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
