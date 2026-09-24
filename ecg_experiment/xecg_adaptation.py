"""Experimental self-supervised continuation of the released xECG encoder.

These ECG objectives are inspired by vision SSL; they do not reproduce the
published vision methods. Encoder architecture and downstream head stay xECG.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

ARMS = ("a", "b", "c", "d")
GRAM_ARMS = ("b", "c", "d")
VISIBLE_ARMS = ("c", "d")
LEADS = 12
RARITY_NEIGHBORS = 3
TERM_NAMES = ("masked", "pooled", "expansion", "gram", "visible_uniform", "visible_weighted")


class ShuffledStream:
    """An infinite permutation stream with an exact step-boundary cursor."""

    def __init__(self, size: int, seed: int = 42) -> None:
        """
        Start the stream at the first permutation.

        Parameters
        ----------
        size : int
            Number of pool records.
        seed : int
            Seed of the private permutation generator.

        Raises
        ------
        ValueError
            If the pool is empty.
        """
        if size < 1:
            raise ValueError("The SSL pool must be nonempty")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(size, generator=self.generator)
        self.position = 0
        self.cycles = 0

    def take(self, count: int) -> torch.Tensor:
        """
        Return the next indices, starting a new permutation at each cycle end.

        Parameters
        ----------
        count : int
            Number of indices.

        Returns
        -------
        torch.Tensor
            Int64 record indices.

        Raises
        ------
        ValueError
            If ``count`` is not positive.
        """
        if count < 1:
            raise ValueError("Batch size must be positive")
        pieces = []
        while count:
            if self.position == self.size:
                self.permutation = torch.randperm(self.size, generator=self.generator)
                self.position = 0
                self.cycles += 1
            n = min(count, self.size - self.position)
            pieces.append(self.permutation[self.position:self.position + n])
            self.position += n
            count -= n
        return torch.cat(pieces)

    def state_dict(self) -> dict[str, Any]:
        """
        Capture the cursor, current permutation and generator state.

        Returns
        -------
        dict[str, Any]
            Resumable stream state.
        """
        return {"size": self.size, "permutation": self.permutation.clone(),
                "position": self.position, "cycles": self.cycles,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore a state captured by ``state_dict``.

        Parameters
        ----------
        state : dict[str, Any]
            Stream state for a pool of the same size.

        Raises
        ------
        ValueError
            If the state is malformed or for another pool size.
        """
        permutation = state["permutation"]
        if (state["size"] != self.size or permutation.shape != (self.size,)
                or permutation.dtype != torch.int64
                or not torch.equal(permutation.sort().values, torch.arange(self.size))
                or not 0 <= state["position"] <= self.size or state["cycles"] < 0):
            raise ValueError("Malformed SSL sampler state")
        self.permutation = permutation.clone()
        self.position = state["position"]
        self.cycles = state["cycles"]
        self.generator.set_state(state["generator"])


def encode_tokens(encoder: nn.Module, signal: torch.Tensor,
                  mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mask patch embeddings explicitly, preserving the raw padding contract.

    Parameters
    ----------
    encoder : nn.Module
        xECG model exposing ``patch_embedding``, ``core``, ``pooling`` and ``mask_token``.
    signal : torch.Tensor
        Signals of shape [batch, time, 12].
    mask : torch.Tensor | None
        Boolean patch mask of shape [batch, patches].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Pooled features, patch tokens and the all-true valid-patch mask.

    Raises
    ------
    ValueError
        If the signal or mask is malformed.
    """
    if signal.ndim != 3 or signal.shape[-1] != LEADS:
        raise ValueError("xECG input must be [batch,time,12]")
    if signal.shape[1] % encoder.patch_size:
        raise ValueError("xECG signal length must be a multiple of patch size")
    patches = encoder.patch_embedding(signal)
    # This cache contains complete, unpadded records. A masked patch remains a
    # real position; a physical zero at its first sample is not padding either.
    valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=signal.device)
    padding = torch.zeros_like(patches, dtype=torch.bool)
    if mask is not None:
        if mask.shape != valid.shape or mask.dtype != torch.bool:
            raise ValueError("Patch mask must be boolean [batch,patches]")
        patches = torch.where(mask[..., None], encoder.mask_token[None, None, :], patches)
    tokens = encoder.core(patches)
    pooled, _ = encoder.pooling(tokens, padding)
    return pooled, tokens, valid


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """
    Interpolate floating teacher state toward the student and copy the rest.

    Parameters
    ----------
    teacher : nn.Module
        EMA model, updated in place.
    student : nn.Module
        Online model with the same state structure.
    momentum : float
        Weight kept on the teacher, in [0, 1].

    Raises
    ------
    ValueError
        If the momentum is out of range or the state structures differ.
    """
    if not 0 <= momentum <= 1:
        raise ValueError("EMA momentum must be between zero and one")
    target, source = teacher.state_dict(), student.state_dict()
    if target.keys() != source.keys():
        raise ValueError("EMA/student state structures differ")
    for name, value in target.items():
        if value.is_floating_point():
            value.lerp_(source[name], 1.0 - momentum)
        else:
            value.copy_(source[name])


def frozen_copy(encoder: nn.Module) -> nn.Module:
    """
    Deep-copy a model as a frozen, eval-mode network.

    Parameters
    ----------
    encoder : nn.Module
        Model to copy.

    Returns
    -------
    nn.Module
        Copy without trainable parameters.
    """
    result = copy.deepcopy(encoder)
    result.requires_grad_(False)
    result.eval()
    return result


def cosine_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """
    One minus float32 cosine similarity against a detached target.

    Parameters
    ----------
    student : torch.Tensor
        Student features.
    teacher : torch.Tensor
        Target features of the same shape.

    Returns
    -------
    torch.Tensor
        Distances over the last dimension.
    """
    return 1.0 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1, eps=1e-8)


def selected_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """
    Average selected tokens per ECG, giving each ECG equal weight irrespective of its token count.

    Parameters
    ----------
    values : torch.Tensor
        Per-token values [batch, tokens].
    selected : torch.Tensor
        Boolean selection with the same shape.

    Returns
    -------
    torch.Tensor
        Scalar mean over records.

    Raises
    ------
    ValueError
        If shapes differ or a record has no selected token.
    """
    if values.shape != selected.shape or not selected.any(dim=1).all():
        raise ValueError("Every record needs at least one selected token")
    return (values.masked_fill(~selected, 0).sum(dim=1) / selected.sum(dim=1)).mean()


@dataclass(frozen=True)
class AdaptationConfig:
    """Fixed optimization, masking and objective-weight settings of the adaptation run."""

    updates: int = 1000
    effective_batch_size: int = 64
    microbatch_size: int = 8
    mask_tokens: int = 8
    learning_rate: float = 1e-5
    final_learning_rate: float = 1e-6
    warmup_updates: int = 50
    weight_decay: float = 0.04
    gradient_clip: float = 1.0
    expansion_epsilon: float = 0.05
    expansion_weight: float = 0.1
    gram_weight: float = 1.0
    visible_weight: float = 0.25
    ema_start: float = 0.99
    ema_end: float = 0.9999
    seed: int = 42

    def __post_init__(self) -> None:
        """
        Validate sizes.

        Raises
        ------
        ValueError
            If a size is not positive or the batch does not split into microbatches.
        """
        if min(self.updates, self.effective_batch_size, self.microbatch_size, self.mask_tokens) < 1:
            raise ValueError("Update and batch/mask sizes must be positive")
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError("Effective batch size must be divisible by microbatch size")

    def as_dict(self) -> dict[str, Any]:
        """
        Return the settings as a plain dictionary.

        Returns
        -------
        dict[str, Any]
            Field names and values in declaration order.
        """
        return asdict(self)


def learning_rate_at(step: int, config: AdaptationConfig) -> float:
    """
    LR for a zero-indexed optimizer update; the final update reaches min LR.

    Parameters
    ----------
    step : int
        Zero-indexed optimizer update.
    config : AdaptationConfig
        Schedule settings.

    Returns
    -------
    float
        Linear warmup, then cosine decay to ``final_learning_rate``.

    Raises
    ------
    ValueError
        If the step is outside the update budget.
    """
    if not 0 <= step < config.updates:
        raise ValueError("Optimizer update outside the fixed budget")
    if step < config.warmup_updates:
        return config.learning_rate * (step + 1) / max(1, config.warmup_updates)
    progress = (step - config.warmup_updates) / max(1, config.updates - config.warmup_updates - 1)
    return (config.final_learning_rate
            + (config.learning_rate - config.final_learning_rate) * (1 + math.cos(math.pi * progress)) / 2)


def ema_momentum_at(step: int, config: AdaptationConfig) -> float:
    """
    Cosine-increase the EMA momentum from ``ema_start`` to ``ema_end``.

    Parameters
    ----------
    step : int
        Zero-indexed optimizer update.
    config : AdaptationConfig
        Schedule settings.

    Returns
    -------
    float
        Momentum for this update.
    """
    progress = step / max(1, config.updates - 1)
    return config.ema_end - (config.ema_end - config.ema_start) * (1 + math.cos(math.pi * progress)) / 2


def contiguous_masks(batch_size: int, tokens: int, masked: int,
                     generator: torch.Generator) -> torch.Tensor:
    """
    Draw two independent contiguous token masks per record.

    Parameters
    ----------
    batch_size : int
        Number of records.
    tokens : int
        Tokens per record.
    masked : int
        Length of each masked span.
    generator : torch.Generator
        Generator for the span starts.

    Returns
    -------
    torch.Tensor
        Boolean masks of shape [batch_size, 2, tokens].

    Raises
    ------
    ValueError
        If the span leaves no visible or no hidden token.
    """
    if not 0 < masked < tokens:
        raise ValueError("Mask must leave both visible and hidden tokens")
    starts = torch.randint(tokens - masked + 1, (batch_size, 2), generator=generator)
    positions = torch.arange(tokens)[None, None, :]
    return (positions >= starts[..., None]) & (positions < starts[..., None] + masked)


def coding_rate(features: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    """
    Upstream expansion, evaluated on the actual pooled microbatch.

    Parameters
    ----------
    features : torch.Tensor
        Pooled features [batch, dim].
    epsilon : float
        Distortion; must be positive.

    Returns
    -------
    torch.Tensor
        Scalar negative scaled coding rate.

    Raises
    ------
    ValueError
        If the input is not a matrix or ``epsilon`` is not positive.
    RuntimeError
        If the determinant is not finite and positive.
    """
    if features.ndim != 2 or epsilon <= 0:
        raise ValueError("Coding rate requires a feature matrix and positive epsilon")
    normalized = F.normalize(features, dim=-1, eps=1e-8)
    m, d = normalized.shape
    identity = torch.eye(m, device=features.device, dtype=features.dtype)
    matrix = identity + d / (m * epsilon) * (normalized @ normalized.T)
    sign, logdet = torch.linalg.slogdet(matrix)
    if not torch.isfinite(logdet) or sign != 1:
        raise RuntimeError("Invalid coding-rate determinant")
    return -0.5 * epsilon * math.sqrt(m / (d * min(d, m))) * logdet


@torch.no_grad()
def rarity_weights(frozen_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weight tokens by within-record rarity against their three most similar distant tokens.

    Parameters
    ----------
    frozen_tokens : torch.Tensor
        Anchor tokens [batch, tokens, dim]; at least six tokens.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rank-based weights in [1, 2] and raw rarity, both [batch, tokens].

    Raises
    ------
    ValueError
        If there are fewer than six tokens.
    """
    normalized = F.normalize(frozen_tokens.float(), dim=-1, eps=1e-8)
    tokens = normalized.shape[1]
    if tokens < 2 * RARITY_NEIGHBORS:
        raise ValueError("Rarity requires at least six tokens for three distant matches")
    similarity = normalized @ normalized.transpose(-1, -2)
    positions = torch.arange(tokens, device=normalized.device)
    excluded = (positions[:, None] - positions[None, :]).abs() <= 1
    similarity = similarity.masked_fill(excluded, -torch.inf)
    rarity = 1.0 - similarity.topk(RARITY_NEIGHBORS, dim=-1).values.mean(dim=-1)
    less = (rarity[:, :, None] > rarity[:, None, :]).sum(dim=-1)
    equal = (rarity[:, :, None] == rarity[:, None, :]).sum(dim=-1)
    rank = less + (equal - 1) / 2
    return 1.0 + rank / (tokens - 1), rarity


def gram_loss(student: torch.Tensor, frozen: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """
    Squared difference of student and anchor token Gram matrices over visible pairs.

    Parameters
    ----------
    student : torch.Tensor
        Student tokens [batch, tokens, dim].
    frozen : torch.Tensor
        Anchor tokens with the same shape.
    visible : torch.Tensor
        Boolean visible-token mask [batch, tokens].

    Returns
    -------
    torch.Tensor
        Scalar mean over records.

    Raises
    ------
    ValueError
        If a record has fewer than two visible tokens.
    """
    if not (visible.sum(dim=1) >= 2).all():
        raise ValueError("Gram retention needs two visible tokens per ECG")
    s = F.normalize(student.float(), dim=-1, eps=1e-8)
    f = F.normalize(frozen.detach().float(), dim=-1, eps=1e-8)
    difference = (s @ s.transpose(-1, -2) - f @ f.transpose(-1, -2)).square()
    pairs = visible[:, :, None] & visible[:, None, :]
    pairs &= ~torch.eye(student.shape[1], dtype=torch.bool, device=student.device)[None]
    return (difference.masked_fill(~pairs, 0).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2))).mean()


def visible_loss(student: torch.Tensor, frozen: torch.Tensor, visible: torch.Tensor,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Cosine retention of visible student tokens to the anchor, optionally weighted.

    Parameters
    ----------
    student : torch.Tensor
        Student tokens [batch, tokens, dim].
    frozen : torch.Tensor
        Anchor tokens with the same shape.
    visible : torch.Tensor
        Boolean visible-token mask [batch, tokens].
    weights : torch.Tensor | None
        Positive per-token weights; uniform when omitted.

    Returns
    -------
    torch.Tensor
        Scalar mean over records.

    Raises
    ------
    ValueError
        If the weights are malformed or select no token in a record.
    """
    distance = cosine_distance(student, frozen)
    if weights is None:
        return selected_mean(distance, visible)
    if weights.shape != visible.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Visible retention weights must be finite and positive")
    selected = weights.detach().masked_fill(~visible, 0)
    if not (selected.sum(dim=1) > 0).all():
        raise ValueError("Visible retention requires selected tokens")
    return ((selected * distance).sum(dim=1) / selected.sum(dim=1)).mean()


def _without_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    """Biases, norms, mask tokens and other vectors are not decayed."""
    return parameter.ndim <= 1 or "bias" in name or "norm" in name.lower() or "mask_token" in name


def ssl_parameter_groups(student: nn.Module, config: AdaptationConfig) -> list[dict[str, Any]]:
    """
    Split trainable parameters into weight-decayed and undecayed optimizer groups.

    Parameters
    ----------
    student : nn.Module
        Model being adapted.
    config : AdaptationConfig
        Supplies the weight decay.

    Returns
    -------
    list[dict[str, Any]]
        Decayed group followed by the undecayed group.
    """
    decay, no_decay = [], []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if _without_weight_decay(name, parameter):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _arm_total(terms: dict[str, torch.Tensor], arm: str, config: AdaptationConfig) -> torch.Tensor:
    """Combine averaged terms: every arm, plus Gram (b-d), plus visible retention (c uniform, d weighted)."""
    total = terms["masked"] + terms["pooled"] + config.expansion_weight * terms["expansion"]
    if arm in GRAM_ARMS:
        total = total + config.gram_weight * terms["gram"]
    if arm in VISIBLE_ARMS:
        visible_term = terms["visible_weighted" if arm == "d" else "visible_uniform"]
        total = total + config.visible_weight * visible_term
    return total


class AdaptationModel(nn.Module):
    """Two masked student views, clean EMA teacher, and fixed release anchor."""

    def __init__(self, student: nn.Module) -> None:
        """
        Wrap the student with frozen EMA and anchor copies.

        Parameters
        ----------
        student : nn.Module
            Released xECG model to adapt.
        """
        super().__init__()
        self.student = student
        self.ema = frozen_copy(student)
        self.anchor = frozen_copy(student)

    def train(self, mode: bool = True) -> AdaptationModel:
        """
        Set the student's mode while the EMA and anchor stay in eval mode.

        Parameters
        ----------
        mode : bool
            Training mode for the student.

        Returns
        -------
        AdaptationModel
            This model.
        """
        super().train(mode)
        self.ema.eval()
        self.anchor.eval()
        return self

    def forward(self, signal: torch.Tensor, masks: torch.Tensor, arm: str,
                config: AdaptationConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the arm's objective over both masked views.

        Parameters
        ----------
        signal : torch.Tensor
            Signals of shape [batch, time, 12].
        masks : torch.Tensor
            Boolean view masks of shape [batch, 2, patches].
        arm : str
            One of ``ARMS``.
        config : AdaptationConfig
            Objective weights.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Scalar loss and every detached view-averaged term.

        Raises
        ------
        ValueError
            If the arm is unknown or the masks are malformed.
        """
        if arm not in ARMS or masks.shape != (signal.shape[0], 2, signal.shape[1] // self.student.patch_size):
            raise ValueError("Unknown adaptation arm or malformed two-view masks")
        with torch.no_grad():
            ema_pool, ema_tokens, _ = encode_tokens(self.ema, signal)
            _, anchor_tokens, _ = encode_tokens(self.anchor, signal)
            weights, _ = rarity_weights(anchor_tokens)
        terms = {name: [] for name in TERM_NAMES}
        for view in range(2):
            mask = masks[:, view]
            pooled, tokens, _ = encode_tokens(self.student, signal, mask)
            terms["masked"].append(selected_mean(cosine_distance(tokens, ema_tokens), mask))
            terms["pooled"].append(cosine_distance(pooled, ema_pool).mean())
            terms["expansion"].append(coding_rate(pooled, config.expansion_epsilon))
            terms["gram"].append(gram_loss(tokens, anchor_tokens, ~mask))
            terms["visible_uniform"].append(visible_loss(tokens, anchor_tokens, ~mask))
            terms["visible_weighted"].append(visible_loss(tokens, anchor_tokens, ~mask, weights))
        averaged = {name: torch.stack(values).mean() for name, values in terms.items()}
        total = _arm_total(averaged, arm, config)
        return total, {name: value.detach() for name, value in averaged.items()}


def _effective_rank(normalized: torch.Tensor) -> torch.Tensor:
    """Exponentiated entropy of the centered singular-value energy spectrum."""
    spectrum = torch.linalg.svdvals(normalized - normalized.mean(dim=0))
    probabilities = spectrum.square() / spectrum.square().sum().clamp_min(1e-12)
    return (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()


@torch.no_grad()
def representation_diagnostics(model: AdaptationModel, signal: torch.Tensor) -> dict[str, Any]:
    """
    Summarize student, EMA and anchor representations in eval mode.

    Parameters
    ----------
    model : AdaptationModel
        Model to inspect; its training mode is restored afterwards.
    signal : torch.Tensor
        Signals of shape [batch, time, 12].

    Returns
    -------
    dict[str, Any]
        Norm, variance, rank, release-similarity and rarity statistics.
    """
    was_training = model.training
    model.eval()
    pooled, tokens, _ = encode_tokens(model.student, signal)
    ema_pool, ema_tokens, _ = encode_tokens(model.ema, signal)
    anchor_pool, anchor_tokens, _ = encode_tokens(model.anchor, signal)
    normalized = F.normalize(pooled, dim=-1, eps=1e-8)
    rank = _effective_rank(normalized)
    similarities = normalized @ F.normalize(anchor_pool, dim=-1, eps=1e-8).T
    off_diagonal = ~torch.eye(len(signal), device=signal.device, dtype=torch.bool)
    other_record_cosine = float(similarities[off_diagonal].mean()) if len(signal) > 1 else None
    weights, rarity = rarity_weights(anchor_tokens)
    result = {"student_token_norm": float(tokens.norm(dim=-1).mean()),
              "ema_token_norm": float(ema_tokens.norm(dim=-1).mean()),
              "anchor_token_norm": float(anchor_tokens.norm(dim=-1).mean()),
              "pooled_variance": float(pooled.var(dim=0, unbiased=False).mean()),
              "pooled_effective_rank": float(rank),
              "student_release_same_record_cosine": float(similarities.diagonal().mean()),
              "student_release_other_record_cosine": other_record_cosine,
              "ema_release_cosine_distance": float(cosine_distance(ema_pool, anchor_pool).mean()),
              "rarity_min": float(rarity.min()), "rarity_max": float(rarity.max()),
              "rarity_weight_min": float(weights.min()), "rarity_weight_max": float(weights.max()),
              "rarity_weight_mean": float(weights.mean()),
              "rarity_weights": weights.cpu().tolist()}
    model.train(was_training)
    return result
