"""Float32 Mamba-3 SISO and Mamba-2 reference CPC models for Experiment 013.

Equations follow state-spaces/mamba revision
e9594ce1c732d97440f0332fdc43170a2294dbfa. The 79-token quadratic dual form
uses ordinary PyTorch operations supported by V100; it is not the official
BF16/TMA kernel. Tests compare it with the upstream recurrent SISO oracle.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .cpc import WIDTH, CPCPretrainer, cpc_loss, prediction_heads
from .cpc_delta_memory import DeltaCPCEncoder

Variant = Literal["mamba2", "mamba3"]
UPSTREAM_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def _decay_matrix(adt: torch.Tensor) -> torch.Tensor:
    """Return causal exp(sum(a[j+1:i+1])) with no upper-triangle overflow."""
    length = adt.shape[1]
    cumulative = adt.transpose(1, 2).cumsum(-1)
    log_decay = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    mask = torch.ones(length, length, dtype=torch.bool, device=adt.device).tril()
    return log_decay.masked_fill(~mask, -torch.inf).exp()


def _rotate(value: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Apply pairwise rotations to the first twice-angle-count state channels."""
    rotated_width = 2 * angle.shape[-1]
    pairs = value[..., :rotated_width].unflatten(-1, (-1, 2))
    first, second = pairs.unbind(-1)
    cosine, sine = angle.cos(), angle.sin()
    rotated = torch.stack((first * cosine - second * sine, first * sine + second * cosine), -1)
    return torch.cat((rotated.flatten(-2), value[..., rotated_width:]), -1)


def mamba3_siso_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact zero-state SISO dual form after normalization/bias.

    Query/key/value are [batch,time,heads,channels], scalar controls are
    [batch,time,heads], and angles are unactivated [batch,time,heads,angles].
    Unlike the fused upstream wrapper, no input is quantized to BF16. The
    trapezoidal previous-input term is retained, including the first/last step.
    """
    phase = (angles.tanh() * math.pi * dt.unsqueeze(-1)).cumsum(1).remainder(2 * math.pi)
    query, key = _rotate(query, phase), _rotate(key, phase)
    weight = trap.sigmoid()
    gamma = dt * weight
    shifted = F.pad((dt * (1 - weight))[:, 1:], (0, 0, 0, 1))
    similarity = torch.einsum("bthn,bshn->bhts", query, key)
    decay = _decay_matrix(adt)
    length = value.shape[1]
    strict_lower = torch.ones(length, length, dtype=torch.bool, device=value.device).tril(-1)
    # Source j contributes gamma[j] at j and dt[j+1]*(1-trap[j+1]) thereafter.
    scale = gamma.transpose(1, 2).unsqueeze(-2)
    scale = scale + strict_lower * shifted.transpose(1, 2).unsqueeze(-2)
    coefficient = decay * scale
    return torch.einsum("bhts,bshp->bthp", similarity * coefficient, value)


def mamba2_scan(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    adt: torch.Tensor, dt: torch.Tensor,
) -> torch.Tensor:
    """Compute Mamba-2 SSD using its discretized current-input injection."""
    similarity = torch.einsum("bthn,bshn->bhts", query, key)
    coefficients = similarity * _decay_matrix(adt) * dt.transpose(1, 2).unsqueeze(-2)
    return torch.einsum("bhts,bshp->bthp", coefficients, value)


class RMSNorm(nn.Module):
    """Trainable RMS normalization with the upstream epsilon."""

    def __init__(self, width: int) -> None:
        """Initialize the channel scale to one."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize independently at every token."""
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * self.weight


def _dt_bias(heads: int) -> nn.Parameter:
    """Initialize the inverse-softplus step size as in upstream Mamba-2/3."""
    dt = torch.exp(torch.rand(heads) * math.log(100) + math.log(0.001))
    return nn.Parameter(dt + torch.log(-torch.expm1(-dt)))


class Mamba3SISO(nn.Module):
    """Mamba-3 SISO: expansion 2, one B/C group, half-state complex rotation."""

    def __init__(self, width: int = WIDTH, state_size: int = 64, head_dim: int = 64) -> None:
        """Construct the documented SISO variant without optional output norm."""
        super().__init__()
        self.inner, self.state_size = 2 * width, state_size
        self.head_dim, self.heads = head_dim, self.inner // head_dim
        self.angle_count = state_size // 4
        if self.inner % head_dim or state_size % 4 or self.angle_count == 0:
            raise ValueError("Expansion width must divide heads; state size must be a positive multiple of 4")
        sizes = [self.inner, self.inner, state_size, state_size, self.heads, self.heads, self.heads]
        self.sizes = [*sizes, self.angle_count]
        self.in_proj = nn.Linear(width, sum(self.sizes), bias=False)
        self.dt_bias = _dt_bias(self.heads)
        self.b_bias = nn.Parameter(torch.ones(self.heads, state_size))
        self.c_bias = nn.Parameter(torch.ones(self.heads, state_size))
        self.b_norm, self.c_norm = RMSNorm(state_size), RMSNorm(state_size)
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the upstream SISO equations with a fresh state per sequence."""
        z, value, key, query, dt, a, trap, angles = self.in_proj(tokens).split(self.sizes, -1)
        dt = F.softplus(dt + self.dt_bias)
        a = -(a.clamp_min(0) + (1 - a.clamp_max(0)).reciprocal()).clamp_min(1e-4)
        key = self.b_norm(key).unsqueeze(-2) + self.b_bias
        query = self.c_norm(query).unsqueeze(-2) + self.c_bias
        value = value.unflatten(-1, (self.heads, self.head_dim))
        angles = angles.unsqueeze(-2).expand(-1, -1, self.heads, -1)
        output = mamba3_siso_scan(query, key, value, a * dt, dt, trap, angles)
        output = output + self.skip[:, None] * value
        return self.out_proj(output.flatten(-2) * F.silu(z))


class Mamba2Reference(nn.Module):
    """Mamba2Simple with identical inner/state/head sizes and PyTorch SSD."""

    def __init__(self, width: int = WIDTH, state_size: int = 64, head_dim: int = 64) -> None:
        """Use upstream defaults: expand 2, causal width 4 and one B/C group."""
        super().__init__()
        self.inner, self.state_size = 2 * width, state_size
        self.head_dim, self.heads = head_dim, self.inner // head_dim
        if self.inner % head_dim:
            raise ValueError("Expanded width must be divisible by head dimension")
        self.conv_width = self.inner + 2 * state_size
        self.in_proj = nn.Linear(width, self.inner + self.conv_width + self.heads, bias=False)
        self.conv = nn.Conv1d(self.conv_width, self.conv_width, 4, groups=self.conv_width)
        self.dt_bias = _dt_bias(self.heads)
        self.a_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.norm = RMSNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution, SSD, post-gate RMS norm and output map."""
        z, xbc, dt = self.in_proj(tokens).split([self.inner, self.conv_width, self.heads], -1)
        xbc = F.silu(self.conv(F.pad(xbc.transpose(1, 2), (3, 0)))).transpose(1, 2)
        value, key, query = xbc.split([self.inner, self.state_size, self.state_size], -1)
        dt = F.softplus(dt + self.dt_bias)
        value = value.unflatten(-1, (self.heads, self.head_dim))
        key = key.unsqueeze(-2).expand(-1, -1, self.heads, -1)
        query = query.unsqueeze(-2).expand_as(key)
        output = mamba2_scan(query, key, value, -self.a_log.exp() * dt, dt)
        output = (output + self.skip[:, None] * value).flatten(-2)
        return self.out_proj(self.norm(output * F.silu(z)))


class MambaResidualBlock(nn.Module):
    """Common pre-RMS residual wrapper for either sequence mixer."""

    def __init__(self, variant: Variant) -> None:
        """Build one of the explicitly named reference operators."""
        super().__init__()
        if variant not in ("mamba2", "mamba3"):
            raise ValueError(f"Unknown Mamba variant: {variant}")
        self.norm = RMSNorm(WIDTH)
        self.mixer = Mamba3SISO() if variant == "mamba3" else Mamba2Reference()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return same-width residual contexts without cross-sequence state."""
        return tokens + self.mixer(self.norm(tokens))


class MambaCPCPretrainer(nn.Module):
    """Causal ECG-CPC using a compact pair of Mamba-2 or Mamba-3 SISO blocks."""

    def __init__(self, variant: Variant) -> None:
        """Share the historical stem, half handling, head shapes and readout."""
        super().__init__()
        self.encoder = DeltaCPCEncoder(complex_ranges=False)
        self.encoder.context = nn.Sequential(MambaResidualBlock(variant), MambaResidualBlock(variant))
        self.heads = prediction_heads()

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the common strictly causal CPC objective."""
        tokens, contexts = self.encoder(signal)
        loss = cpc_loss(tokens, contexts, self.heads)
        return loss, {"cpc": float(loss.detach()), "cmsc": 0.0}


def matched_initial_models(seed: int) -> dict[str, nn.Module]:
    """Create fresh GRU/Mamba-2/Mamba-3 with identical stem and CPC heads."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gru = CPCPretrainer()
        models = {"gru": gru, "mamba2": MambaCPCPretrainer("mamba2"),
                  "mamba3": MambaCPCPretrainer("mamba3")}
    for name in ("mamba2", "mamba3"):
        models[name].encoder.convs.load_state_dict(gru.encoder.convs.state_dict())
        models[name].heads.load_state_dict(gru.heads.state_dict())
    return models
