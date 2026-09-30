"""Matched causal waveform front ends for the Experiment 039 CPC study.

All tokens end at sample 16*j within an independently encoded 1250-sample half.
The frozen CNN covers 33 samples. Multiscale branches cover 41, 49 and 57 samples
after the final downsampling stages. Patch tokens cover exactly 33 samples.
Layer normalization acts only on channels, never across temporal positions.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ecg_experiment.cpc import HALF_SAMPLES, LEADS, TOKEN_COUNT, WIDTH, CausalConvBlock, CPCPretrainer
from ecg_experiment.cpc_xlstm_native_gru import create_model as native_create_model

ENCODERS = ("cnn", "multiscale", "patch")
CONTEXTS = ("gru", "xlstm")
TOKEN_STRIDE = 16
PATCH_SAMPLES = 33
MULTISCALE_KERNELS = (3, 5, 7)
FRONTEND_SEED_OFFSET = 39


class CausalMultiscaleResidual(nn.Module):
    """Parallel depthwise temporal filters with channel fusion and a residual."""

    def __init__(self, width: int = 128) -> None:
        """Build filters at the stem's four-sample spacing.

        Parameters
        ----------
        width : int
            Input and output channel count.
        """
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(width, width, kernel, groups=width) for kernel in MULTISCALE_KERNELS
        )
        self.fusion = nn.Conv1d(width * len(MULTISCALE_KERNELS), width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse causal branches without changing temporal spacing.

        Parameters
        ----------
        x : torch.Tensor
            Stem features of shape [batch, channels, time].

        Returns
        -------
        torch.Tensor
            Residual features with the input shape.
        """
        branches = [
            branch(F.pad(x, (kernel - 1, 0)))
            for kernel, branch in zip(MULTISCALE_KERNELS, self.branches, strict=True)
        ]
        residual = self.fusion(torch.cat(branches, dim=1))
        return F.gelu(self.norm((x + residual).transpose(1, 2))).transpose(1, 2)


class MultiscaleFrontend(nn.Sequential):
    """Four causal stride-two stages with multiscale fusion after the stem."""

    def __init__(self) -> None:
        """Build the compact 57-sample maximum-support front end."""
        super().__init__(
            CausalConvBlock(LEADS, 64, 5),
            CausalConvBlock(64, 128, 3),
            CausalMultiscaleResidual(128),
            CausalConvBlock(128, 192, 3),
            CausalConvBlock(192, WIDTH, 3),
        )


class PatchFrontend(nn.Module):
    """Project flattened local 12-lead patches ending every sixteen samples."""

    def __init__(self) -> None:
        """Build a 33-sample patch projection with tokenwise normalization."""
        super().__init__()
        self.projection = nn.Linear(LEADS * PATCH_SAMPLES, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract left-padded causal patches and return channel-first tokens.

        Parameters
        ----------
        x : torch.Tensor
            Independent halves of shape [batch, 12, 1250].

        Returns
        -------
        torch.Tensor
            Tokens of shape [batch, 256, 79].
        """
        patches = F.pad(x, (PATCH_SAMPLES - 1, 0)).unfold(-1, PATCH_SAMPLES, TOKEN_STRIDE)
        patches = patches.transpose(1, 2).flatten(-2)
        return F.gelu(self.norm(self.projection(patches))).transpose(1, 2)


def create_model(encoder: str, context: str, seed: int, device: torch.device | str = "cpu") -> CPCPretrainer:
    """Create ordinary CPC with independently matched front end and context.

    The frozen native factory initializes all shared tensors first. Replacement
    front ends use seed + 39 in an isolated CPU RNG scope, so context and head
    weights match across encoder arms and front ends match across contexts.

    Parameters
    ----------
    encoder : str
        ``cnn``, ``multiscale`` or ``patch`` waveform front end.
    context : str
        ``gru`` for the native baseline or ``xlstm`` for the frozen mLSTM stack.
    seed : int
        Shared initialization seed.
    device : torch.device or str
        Target device after deterministic CPU construction.

    Returns
    -------
    CPCPretrainer
        Unchanged ordinary CPC model and encoder interfaces.

    Raises
    ------
    ValueError
        If either architecture name is unknown.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown CPC encoder: {encoder}")
    if context not in CONTEXTS:
        raise ValueError(f"Unknown CPC context: {context}")
    model = native_create_model(context, seed, "cpu")
    if encoder != "cnn":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + FRONTEND_SEED_OFFSET)
            model.encoder.convs = MultiscaleFrontend() if encoder == "multiscale" else PatchFrontend()
    return model.to(device)


def architecture_spec(encoder: str, context: str) -> dict[str, Any]:
    """Describe the waveform support and measured parameter counts for receipts.

    Parameters
    ----------
    encoder : str
        Waveform front end accepted by ``create_model``.
    context : str
        Context architecture accepted by ``create_model``.

    Returns
    -------
    dict[str, Any]
        JSON-compatible configuration, alignment and exact parameter counts.
    """
    model = create_model(encoder, context, seed=0)
    specs = {
        "cnn": {"kernels": [5, 3, 3, 3], "strides": [2, 2, 2, 2], "receptive_field_samples": 33},
        "multiscale": {
            "kernels": [5, 3, 3, 3],
            "strides": [2, 2, 2, 2],
            "residual_after_stage": 2,
            "residual_kernels": list(MULTISCALE_KERNELS),
            "branch_receptive_fields_samples": [41, 49, 57],
            "receptive_field_samples": 57,
        },
        "patch": {
            "patch_samples": PATCH_SAMPLES,
            "patch_stride": TOKEN_STRIDE,
            "projection_input_width": LEADS * PATCH_SAMPLES,
            "receptive_field_samples": 33,
        },
    }
    return {
        "encoder": encoder,
        "context": context,
        **specs[encoder],
        "width": WIDTH,
        "sample_rate_hz": 250,
        "half_samples": HALF_SAMPLES,
        "tokens_per_half": TOKEN_COUNT,
        "token_stride_samples": TOKEN_STRIDE,
        "token_spacing_ms": 64,
        "token_end_samples": list(range(0, HALF_SAMPLES, TOKEN_STRIDE)),
        "support_interval": "[max(0, 16*j - receptive_field_samples + 1), 16*j]",
        "half_independence": True,
        "normalization": "tokenwise LayerNorm",
        "context_support": "all preceding and current tokens within the same half",
        "frontend_seed_offset": 0 if encoder == "cnn" else FRONTEND_SEED_OFFSET,
        "frontend_parameters": sum(parameter.numel() for parameter in model.encoder.convs.parameters()),
        "context_parameters": sum(parameter.numel() for parameter in model.encoder.context.parameters()),
        "prediction_head_parameters": sum(parameter.numel() for parameter in model.heads.parameters()),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
