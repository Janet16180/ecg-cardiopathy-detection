"""Deterministic fixed-factor linear output interpolation with unchanged encoder/head parameters."""

from __future__ import annotations

import torch
from torch.nn import functional

from ecg_experiment.ssl_synthetic_model import DenseWaveformModel as OriginalDenseWaveformModel


def deterministic_linear_upsample(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Evaluate align-corners-false factor-five interpolation using deterministic convolution.

    Parameters
    ----------
    signal : torch.Tensor
        Batch by channel by latent time.
    kernel : torch.Tensor
        Fixed per-channel triangular weights, shape channels by one by nine.

    Returns
    -------
    torch.Tensor
        Native signal with exactly five times the latent sample count.
    """
    padded = torch.cat((signal[..., :1], signal, signal[..., -1:]), dim=-1)
    output = functional.conv_transpose1d(padded, kernel, stride=5, padding=2, groups=signal.shape[1])
    return output[..., 5:-5]


class DenseWaveformModel(OriginalDenseWaveformModel):
    """Original trainable model with an equivalent deterministic output linear map."""

    def __init__(self, width: int = 32, stride: int = 5, scale_mv: float = 2.0) -> None:
        if stride != 5:
            raise ValueError("The fixed interpolation successor requires stride five")
        super().__init__(width, stride, scale_mv)
        triangle = torch.tensor([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=torch.int64)
        self.register_buffer("interpolation_kernel", triangle[None, None].repeat(12, 1, 1), persistent=False)

    def forward(self, signal_mv: torch.Tensor) -> torch.Tensor:
        """
        Produce the same native interpolation grid without the unsupported CUDA backward kernel.

        Parameters
        ----------
        signal_mv : torch.Tensor
            Batch by 12 leads by native time in physical mV.

        Returns
        -------
        torch.Tensor
            Single-signal native-grid dense outputs with no added trainable parameters.
        """
        encoded = self.encoder(signal_mv / self.scale_mv)
        logits = self.head(encoded)
        if signal_mv.shape[-1] != 5 * logits.shape[-1]:
            raise ValueError("Native input length must be divisible by the fixed stride")
        return deterministic_linear_upsample(logits, self.interpolation_kernel.to(logits.dtype) / 5)
