"""Experimental self-supervised continuation of the released xECG encoder.

These ECG objectives are inspired by vision SSL; they do not reproduce the
published vision methods. Encoder architecture and downstream head stay xECG.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("a", "b", "c", "d")


class ShuffledStream:
    """An infinite permutation stream with an exact step-boundary cursor."""

    def __init__(self, size: int, seed: int = 42):
        if size < 1:
            raise ValueError("The SSL pool must be nonempty")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(size, generator=self.generator)
        self.position = 0
        self.cycles = 0

    def take(self, count: int) -> torch.Tensor:
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

    def state_dict(self):
        return {"size": self.size, "permutation": self.permutation.clone(),
                "position": self.position, "cycles": self.cycles,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
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
                  mask: torch.Tensor | None = None):
    """Mask patch embeddings explicitly, preserving the raw padding contract."""
    if signal.ndim != 3 or signal.shape[-1] != 12:
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
        if (mask & ~valid).any():
            raise ValueError("SSL masks may not select padded patches")
        patches = torch.where(mask[..., None], encoder.mask_token[None, None, :], patches)
    tokens = encoder.core(patches)
    pooled, _ = encoder.pooling(tokens, padding)
    return pooled, tokens, valid


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, momentum: float):
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


def frozen_copy(encoder: nn.Module):
    result = copy.deepcopy(encoder)
    result.requires_grad_(False)
    result.eval()
    return result


def cosine_distance(student: torch.Tensor, teacher: torch.Tensor):
    return 1.0 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1, eps=1e-8)


def selected_mean(values: torch.Tensor, selected: torch.Tensor):
    """Give each ECG equal weight irrespective of its number of valid tokens."""
    if values.shape != selected.shape or not selected.any(dim=1).all():
        raise ValueError("Every record needs at least one selected token")
    return (values.masked_fill(~selected, 0).sum(dim=1) / selected.sum(dim=1)).mean()


@dataclass(frozen=True)
class AdaptationConfig:
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

    def __post_init__(self):
        if min(self.updates, self.effective_batch_size, self.microbatch_size, self.mask_tokens) < 1:
            raise ValueError("Update and batch/mask sizes must be positive")
        if self.effective_batch_size % self.microbatch_size:
            raise ValueError("Effective batch size must be divisible by microbatch size")

    def as_dict(self):
        return asdict(self)


def learning_rate_at(step: int, config: AdaptationConfig):
    """LR for a zero-indexed optimizer update; the final update reaches min LR."""
    if not 0 <= step < config.updates:
        raise ValueError("Optimizer update outside the fixed budget")
    if step < config.warmup_updates:
        return config.learning_rate * (step + 1) / max(1, config.warmup_updates)
    progress = (step - config.warmup_updates) / max(1, config.updates - config.warmup_updates - 1)
    return config.final_learning_rate + (config.learning_rate - config.final_learning_rate) * (1 + math.cos(math.pi * progress)) / 2


def ema_momentum_at(step: int, config: AdaptationConfig):
    progress = step / max(1, config.updates - 1)
    return config.ema_end - (config.ema_end - config.ema_start) * (1 + math.cos(math.pi * progress)) / 2


def contiguous_masks(batch_size: int, tokens: int, masked: int,
                     generator: torch.Generator):
    if not 0 < masked < tokens:
        raise ValueError("Mask must leave both visible and hidden tokens")
    starts = torch.randint(tokens - masked + 1, (batch_size, 2), generator=generator)
    positions = torch.arange(tokens)[None, None, :]
    return (positions >= starts[..., None]) & (positions < starts[..., None] + masked)


def coding_rate(features: torch.Tensor, epsilon: float = 0.05):
    """Upstream expansion, evaluated on the actual pooled microbatch."""
    if features.ndim != 2 or epsilon <= 0:
        raise ValueError("Coding rate requires a feature matrix and positive epsilon")
    normalized = F.normalize(features, dim=-1, eps=1e-8)
    m, d = normalized.shape
    matrix = torch.eye(m, device=features.device, dtype=features.dtype) + d / (m * epsilon) * (normalized @ normalized.T)
    sign, logdet = torch.linalg.slogdet(matrix)
    if not torch.isfinite(logdet) or sign != 1:
        raise RuntimeError("Invalid coding-rate determinant")
    return -0.5 * epsilon * math.sqrt(m / (d * min(d, m))) * logdet


@torch.no_grad()
def rarity_weights(frozen_tokens: torch.Tensor):
    normalized = F.normalize(frozen_tokens.float(), dim=-1, eps=1e-8)
    tokens = normalized.shape[1]
    if tokens < 6:
        raise ValueError("Rarity requires at least six tokens for three distant matches")
    similarity = normalized @ normalized.transpose(-1, -2)
    positions = torch.arange(tokens, device=normalized.device)
    excluded = (positions[:, None] - positions[None, :]).abs() <= 1
    similarity = similarity.masked_fill(excluded, -torch.inf)
    rarity = 1.0 - similarity.topk(3, dim=-1).values.mean(dim=-1)
    less = (rarity[:, :, None] > rarity[:, None, :]).sum(dim=-1)
    equal = (rarity[:, :, None] == rarity[:, None, :]).sum(dim=-1)
    rank = less + (equal - 1) / 2
    return 1.0 + rank / (tokens - 1), rarity


def gram_loss(student: torch.Tensor, frozen: torch.Tensor, visible: torch.Tensor):
    if not (visible.sum(dim=1) >= 2).all():
        raise ValueError("Gram retention needs two visible tokens per ECG")
    s = F.normalize(student.float(), dim=-1, eps=1e-8)
    f = F.normalize(frozen.detach().float(), dim=-1, eps=1e-8)
    difference = (s @ s.transpose(-1, -2) - f @ f.transpose(-1, -2)).square()
    pairs = visible[:, :, None] & visible[:, None, :]
    pairs &= ~torch.eye(student.shape[1], dtype=torch.bool, device=student.device)[None]
    return (difference.masked_fill(~pairs, 0).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2))).mean()


def visible_loss(student, frozen, visible, weights=None):
    distance = cosine_distance(student, frozen)
    if weights is None:
        return selected_mean(distance, visible)
    if weights.shape != visible.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Visible retention weights must be finite and positive")
    selected = weights.detach().masked_fill(~visible, 0)
    if not (selected.sum(dim=1) > 0).all():
        raise ValueError("Visible retention requires selected tokens")
    return ((selected * distance).sum(dim=1) / selected.sum(dim=1)).mean()


def ssl_parameter_groups(student: nn.Module, config: AdaptationConfig):
    decay, no_decay = [], []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or "bias" in name or "norm" in name.lower() or "mask_token" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class AdaptationModel(nn.Module):
    """Two masked student views, clean EMA teacher, and fixed release anchor."""

    def __init__(self, student: nn.Module):
        super().__init__()
        self.student = student
        self.ema = frozen_copy(student)
        self.anchor = frozen_copy(student)

    def train(self, mode=True):
        super().train(mode)
        self.ema.eval()
        self.anchor.eval()
        return self

    def forward(self, signal, masks, arm: str, config: AdaptationConfig):
        if arm not in ARMS or masks.shape != (signal.shape[0], 2, signal.shape[1] // self.student.patch_size):
            raise ValueError("Unknown adaptation arm or malformed two-view masks")
        with torch.no_grad():
            ema_pool, ema_tokens, _ = encode_tokens(self.ema, signal)
            anchor_pool, anchor_tokens, _ = encode_tokens(self.anchor, signal)
            weights, _ = rarity_weights(anchor_tokens)
        terms = {name: [] for name in ("masked", "pooled", "expansion", "gram", "visible_uniform", "visible_weighted")}
        for view in range(2):
            mask = masks[:, view]
            pooled, tokens, _ = encode_tokens(self.student, signal, mask)
            terms["masked"].append(selected_mean(cosine_distance(tokens, ema_tokens), mask))
            terms["pooled"].append(cosine_distance(pooled, ema_pool).mean())
            terms["expansion"].append(coding_rate(pooled, config.expansion_epsilon))
            terms["gram"].append(gram_loss(tokens, anchor_tokens, ~mask))
            terms["visible_uniform"].append(visible_loss(tokens, anchor_tokens, ~mask))
            terms["visible_weighted"].append(visible_loss(tokens, anchor_tokens, ~mask, weights))
        terms = {name: torch.stack(values).mean() for name, values in terms.items()}
        total = terms["masked"] + terms["pooled"] + config.expansion_weight * terms["expansion"]
        if arm in ("b", "c", "d"):
            total = total + config.gram_weight * terms["gram"]
        if arm in ("c", "d"):
            total = total + config.visible_weight * terms["visible_weighted" if arm == "d" else "visible_uniform"]
        return total, {name: value.detach() for name, value in terms.items()}


@torch.no_grad()
def representation_diagnostics(model: AdaptationModel, signal: torch.Tensor):
    was_training = model.training
    model.eval()
    pooled, tokens, _ = encode_tokens(model.student, signal)
    ema_pool, ema_tokens, _ = encode_tokens(model.ema, signal)
    anchor_pool, anchor_tokens, _ = encode_tokens(model.anchor, signal)
    normalized = F.normalize(pooled, dim=-1, eps=1e-8)
    spectrum = torch.linalg.svdvals(normalized - normalized.mean(dim=0))
    probabilities = spectrum.square() / spectrum.square().sum().clamp_min(1e-12)
    rank = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
    similarities = normalized @ F.normalize(anchor_pool, dim=-1, eps=1e-8).T
    off_diagonal = ~torch.eye(len(signal), device=signal.device, dtype=torch.bool)
    weights, rarity = rarity_weights(anchor_tokens)
    result = {"student_token_norm": float(tokens.norm(dim=-1).mean()),
              "ema_token_norm": float(ema_tokens.norm(dim=-1).mean()),
              "anchor_token_norm": float(anchor_tokens.norm(dim=-1).mean()),
              "pooled_variance": float(pooled.var(dim=0, unbiased=False).mean()),
              "pooled_effective_rank": float(rank),
              "student_release_same_record_cosine": float(similarities.diagonal().mean()),
              "student_release_other_record_cosine": float(similarities[off_diagonal].mean()) if len(signal) > 1 else None,
              "ema_release_cosine_distance": float(cosine_distance(ema_pool, anchor_pool).mean()),
              "rarity_min": float(rarity.min()), "rarity_max": float(rarity.max()),
              "rarity_weight_min": float(weights.min()), "rarity_weight_max": float(weights.max()),
              "rarity_weight_mean": float(weights.mean()),
              "rarity_weights": weights.cpu().tolist()}
    model.train(was_training)
    return result
