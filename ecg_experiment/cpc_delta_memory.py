"""Compact KDA/CKDA context mixers for the fixed causal ECG-CPC experiment.

The recurrent scan is a readable PyTorch reference. It is deliberately separate
from the CPC stem and objective so KDA and CKDA differ only in gate ranges.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from .cpc import (
    CONV_KERNELS,
    CONV_WIDTHS,
    WIDTH,
    CausalConvBlock,
    CPCPretrainer,
    check_signal_batch,
    cpc_loss,
    prediction_heads,
    split_halves,
)


def recurrent_delta_memory(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    decay: torch.Tensor,
    write_rate: torch.Tensor,
) -> torch.Tensor:
    """Apply the diagonal-decay, rank-one delta update with fresh zero state.

    Inputs are ``[batch, time, heads, channels]`` except ``write_rate``, whose
    final dimension is one. Key and query vectors must already be normalized.
    State is independent for every batch member, including the two ECG halves.
    """
    batch, length, heads, key_width = keys.shape
    value_width = values.shape[-1]
    state = values.new_zeros(batch, heads, key_width, value_width)
    outputs = []

    for time in range(length):
        decayed = state * decay[:, time, :, :, None]
        key = keys[:, time]
        predicted = torch.einsum("bhk,bhkv->bhv", key, decayed)
        correction = write_rate[:, time] * (values[:, time] - predicted)
        state = decayed + key.unsqueeze(-1) * correction.unsqueeze(-2)
        output = torch.einsum("bhk,bhkv->bhv", queries[:, time], state)
        outputs.append(output)

    return torch.stack(outputs, dim=1)


class DeltaMemoryBlock(nn.Module):
    """One residual KDA or CKDA block with identical trainable dimensions."""

    def __init__(self, *, complex_ranges: bool, width: int = WIDTH, heads: int = 4) -> None:
        """Create projections; ``complex_ranges`` changes only gate activations."""
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by head count")

        self.complex_ranges = complex_ranges
        self.heads = heads
        self.head_width = width // heads
        self.input_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.query_conv = nn.Conv1d(width, width, 3, groups=width)
        self.key_conv = nn.Conv1d(width, width, 3, groups=width)
        self.value_conv = nn.Conv1d(width, width, 3, groups=width)
        self.decay = nn.Linear(width, width)
        self.write_rate = nn.Linear(width, heads)
        self.output_norm = nn.LayerNorm(width)
        self.output_gate = nn.Linear(width, width)
        self.output = nn.Linear(width, width, bias=False)

    @staticmethod
    def _causal_projection(
        hidden: torch.Tensor, projection: nn.Linear, convolution: nn.Conv1d
    ) -> torch.Tensor:
        """Project and apply the short left-padded depthwise convolution."""
        channels_first = projection(hidden).transpose(1, 2)
        return F.silu(convolution(F.pad(channels_first, (2, 0)))).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix only present and previous tokens, preserving ``[batch,time,width]``."""
        batch, length, _ = tokens.shape
        hidden = self.input_norm(tokens)
        shape = (batch, length, self.heads, self.head_width)
        queries = self._causal_projection(hidden, self.query, self.query_conv)
        keys = self._causal_projection(hidden, self.key, self.key_conv)
        values = self._causal_projection(hidden, self.value, self.value_conv)
        queries = F.normalize(queries.reshape(shape), dim=-1)
        keys = F.normalize(keys.reshape(shape), dim=-1)
        values = values.reshape(shape)

        decay_logits = self.decay(hidden).reshape(shape)
        rate_logits = self.write_rate(hidden).unsqueeze(-1)
        if self.complex_ranges:
            decay = torch.tanh(decay_logits)
            write_rate = 2 * torch.sigmoid(rate_logits)
        else:
            decay = torch.sigmoid(decay_logits)
            write_rate = torch.sigmoid(rate_logits)

        mixed = recurrent_delta_memory(queries, keys, values, decay, write_rate)
        mixed = self.output_norm(mixed.reshape(batch, length, -1))
        output_gate = torch.sigmoid(self.output_gate(hidden))
        return tokens + self.output(output_gate * mixed)


class DeltaCPCEncoder(nn.Module):
    """The existing causal CNN stem followed by two KDA or CKDA blocks."""

    def __init__(self, *, complex_ranges: bool) -> None:
        """Create the shared CPC stem and variant-specific context mixer."""
        super().__init__()
        self.convs = nn.Sequential(
            *(
                CausalConvBlock(CONV_WIDTHS[index], CONV_WIDTHS[index + 1], kernel)
                for index, kernel in enumerate(CONV_KERNELS)
            )
        )
        self.context = nn.Sequential(
            DeltaMemoryBlock(complex_ranges=complex_ranges),
            DeltaMemoryBlock(complex_ranges=complex_ranges),
        )

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return causal tokens and contexts for each independent five-second half."""
        check_signal_batch(signal)
        batch = len(signal)
        tokens = self.convs(split_halves(signal)).transpose(1, 2)
        contexts = self.context(tokens)
        return tokens.reshape(batch, 2, -1, WIDTH), contexts.reshape(batch, 2, -1, WIDTH)

    @staticmethod
    def pooled(contexts: torch.Tensor) -> torch.Tensor:
        """Use the unchanged mean/max pooling across independent ECG halves."""
        halves = torch.cat((contexts.mean(dim=2), contexts.amax(dim=2)), dim=-1)
        return halves.mean(dim=1)


class DeltaCPCPretrainer(nn.Module):
    """CPC objective with the KDA or CKDA encoder and standard prediction heads."""

    def __init__(self, *, complex_ranges: bool) -> None:
        """Create the encoder and shared CPC prediction-head layout."""
        super().__init__()
        self.encoder = DeltaCPCEncoder(complex_ranges=complex_ranges)
        self.heads = prediction_heads()

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the unchanged CPC loss and its standard component names."""
        tokens, contexts = self.encoder(signal)
        loss = cpc_loss(tokens, contexts, self.heads)
        return loss, {"cpc": float(loss.detach()), "cmsc": 0.0}


def matched_initial_models(seed: int) -> dict[str, nn.Module]:
    """Initialize GRU, KDA, and CKDA with identical stems and CPC heads.

    KDA and CKDA also share every context parameter, isolating the two gate
    range changes. Construction leaves the caller's CPU RNG state unchanged.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        gru = CPCPretrainer()
        kda = DeltaCPCPretrainer(complex_ranges=False)
        ckda = DeltaCPCPretrainer(complex_ranges=True)

    for model in (kda, ckda):
        model.encoder.convs.load_state_dict(gru.encoder.convs.state_dict())
        model.heads.load_state_dict(gru.heads.state_dict())
    ckda.encoder.context.load_state_dict(kda.encoder.context.state_dict())
    return {"gru": gru, "kda": kda, "ckda": ckda}
