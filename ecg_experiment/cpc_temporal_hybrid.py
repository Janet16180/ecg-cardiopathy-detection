"""Compact causal temporal-scale hybrid for Experiment 012.

This ECG adaptation uses gated short explicit, medium explicit and long implicit
filters. It is not a reproduction of the genomic StripedHyena 2 model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .cpc import (
    CONV_KERNELS,
    CONV_WIDTHS,
    TOKEN_COUNT,
    WIDTH,
    CausalConvBlock,
    CPCPretrainer,
    check_signal_batch,
    cpc_loss,
    prediction_heads,
    split_halves,
)

LOCAL_SUPPORT = 5
MEDIUM_SUPPORT = 17


def causal_fft_conv(signal: torch.Tensor, lag_filter: torch.Tensor) -> torch.Tensor:
    """Apply a causal channelwise FIR with linear, rather than circular, FFT convolution.

    ``lag_filter[:, 0]`` multiplies the present token. The FFT length covers the
    full linear convolution, so the end of one half cannot wrap to its start.
    """
    if signal.ndim != 3 or lag_filter.ndim != 2 or signal.shape[1] != lag_filter.shape[0]:
        raise ValueError("Expected signal [batch, channels, time] and filter [channels, lag]")
    length = signal.shape[-1]
    support = lag_filter.shape[-1]
    if support < 1:
        raise ValueError("Filter support must be positive")
    fft_length = length + support - 1
    transformed = torch.fft.rfft(signal, n=fft_length)
    filter_fft = torch.fft.rfft(lag_filter, n=fft_length)
    return torch.fft.irfft(transformed * filter_fft.unsqueeze(0), n=fft_length)[..., :length]


def fold_to_local(lag_filter: torch.Tensor, support: int = LOCAL_SUPPORT) -> torch.Tensor:
    """Fold every learned coefficient into causal local support with mean scaling.

    All coefficients stay active in the local arm, giving both arms identical
    trainable parameter counts while changing only effective temporal support.
    """
    if lag_filter.ndim != 2 or not 0 < support <= lag_filter.shape[-1]:
        raise ValueError("Invalid filter or local support")
    lag = torch.arange(lag_filter.shape[-1], device=lag_filter.device) % support
    folded = lag_filter.new_zeros(lag_filter.shape[0], support)
    folded.index_add_(1, lag, lag_filter)
    counts = torch.bincount(lag, minlength=support).to(lag_filter.dtype)
    return folded / counts.unsqueeze(0)


class ImplicitLagFilter(nn.Module):
    """Generate a channelwise long FIR from normalized causal lag positions."""

    def __init__(self, width: int, support: int = TOKEN_COUNT) -> None:
        """Create a small coordinate MLP and fixed lag positions."""
        super().__init__()
        if width < 1 or support < 1:
            raise ValueError("Width and support must be positive")
        self.support = support
        self.network = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, width))
        self.register_buffer("positions", torch.linspace(0, 1, support).unsqueeze(-1))

    def forward(self) -> torch.Tensor:
        """Return stable, decayed filters ordered from present to oldest lag."""
        raw = self.network(self.positions).transpose(0, 1)
        envelope = torch.exp(-3 * self.positions[:, 0]).unsqueeze(0)
        return raw * envelope / self.support**0.5


class TemporalHybridBlock(nn.Module):
    """One residual, input-gated causal filter block."""

    def __init__(self, kind: str, *, local_control: bool, width: int = WIDTH,
                 token_count: int = TOKEN_COUNT) -> None:
        """Choose short, medium, or implicit long filtering without changing width."""
        super().__init__()
        if kind not in {"short", "medium", "long"}:
            raise ValueError("Unknown temporal-filter kind")
        self.kind = kind
        self.local_control = local_control
        self.input_norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        if kind == "long":
            self.implicit = ImplicitLagFilter(width, token_count)
            self.explicit = None
        else:
            support = LOCAL_SUPPORT if kind == "short" else MEDIUM_SUPPORT
            self.explicit = nn.Parameter(torch.randn(width, support) / support**0.5)
            self.implicit = None

    def lag_filter(self) -> torch.Tensor:
        """Return the effective causal filter for this arm."""
        coefficients = self.implicit() if self.implicit is not None else self.explicit
        if coefficients is None:
            raise RuntimeError("No temporal filter")
        if self.local_control and self.kind != "short":
            return fold_to_local(coefficients)
        return coefficients

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix only current and earlier positions and retain the residual path."""
        hidden = self.input_norm(tokens)
        values = self.value(hidden).transpose(1, 2)
        filtered = causal_fft_conv(values, self.lag_filter()).transpose(1, 2)
        gate = torch.sigmoid(self.gate(hidden))
        return tokens + self.output(gate * F.silu(filtered))


class TemporalHybridEncoder(nn.Module):
    """Original causal waveform stem with three gated temporal-scale blocks."""

    def __init__(self, *, local_control: bool) -> None:
        """Build the independent-half CPC encoder."""
        super().__init__()
        self.convs = nn.Sequential(*(
            CausalConvBlock(CONV_WIDTHS[index], CONV_WIDTHS[index + 1], kernel)
            for index, kernel in enumerate(CONV_KERNELS)
        ))
        self.context = nn.Sequential(*(
            TemporalHybridBlock(kind, local_control=local_control)
            for kind in ("short", "medium", "long")
        ))

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return causal tokens and contexts shaped [batch, 2, 79, 256]."""
        check_signal_batch(signal)
        batch = len(signal)
        tokens = self.convs(split_halves(signal)).transpose(1, 2)
        if tokens.shape[1] != TOKEN_COUNT:
            raise ValueError("Unexpected CPC token count")
        contexts = self.context(tokens)
        return (tokens.reshape(batch, 2, TOKEN_COUNT, WIDTH),
                contexts.reshape(batch, 2, TOKEN_COUNT, WIDTH))

    @staticmethod
    def pooled(contexts: torch.Tensor) -> torch.Tensor:
        """Use the unchanged half-averaged mean/max 512-feature CPC readout."""
        halves = torch.cat((contexts.mean(dim=2), contexts.amax(dim=2)), dim=-1)
        return halves.mean(dim=1)


class TemporalHybridPretrainer(nn.Module):
    """Apply the ordinary CPC objective to either temporal-support arm."""

    def __init__(self, *, local_control: bool) -> None:
        """Construct matched encoder and prediction-head layout."""
        super().__init__()
        self.encoder = TemporalHybridEncoder(local_control=local_control)
        self.heads = prediction_heads()

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the standard causal CPC loss and component fields."""
        tokens, contexts = self.encoder(signal)
        loss = cpc_loss(tokens, contexts, self.heads)
        return loss, {"cpc": float(loss.detach()), "cmsc": 0.0}


def matched_initial_models(seed: int) -> dict[str, nn.Module]:
    """Create a GRU reference and both hybrid arms with identical shared weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gru = CPCPretrainer()
        mixed = TemporalHybridPretrainer(local_control=False)
        local = TemporalHybridPretrainer(local_control=True)
    for model in (mixed, local):
        model.encoder.convs.load_state_dict(gru.encoder.convs.state_dict())
        model.heads.load_state_dict(gru.heads.state_dict())
    local.encoder.context.load_state_dict(mixed.encoder.context.state_dict())
    return {"gru": gru, "mixed": mixed, "local": local}
