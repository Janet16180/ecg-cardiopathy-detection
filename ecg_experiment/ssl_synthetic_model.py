"""Small coordinate-preserving waveform model for the original SSL localization pilot."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional


class TemporalBlock(nn.Module):
    """Residual temporal convolutions with fixed dilation and no fitted normalization."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.second = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply two convolutions and an additive residual.

        Parameters
        ----------
        signal : torch.Tensor
            Batch by feature by time input.

        Returns
        -------
        torch.Tensor
            Features on the same temporal grid.
        """
        residual = self.second(functional.gelu(self.first(signal)))
        return functional.gelu(signal + residual)


class WaveformEncoder(nn.Module):
    """Twelve-input temporal encoder with named lead coordinates retained in the head."""

    def __init__(self, width: int = 32, stride: int = 5) -> None:
        super().__init__()
        self.entry = nn.Conv1d(12, width, 11, stride=stride, padding=5)
        self.blocks = nn.Sequential(*(TemporalBlock(width, dilation) for dilation in (1, 2, 4, 8)))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode the waveform without time embeddings or a raw-signal skip.

        Parameters
        ----------
        signal : torch.Tensor
            Batch by 12 leads by native time.

        Returns
        -------
        torch.Tensor
            Batch by 32 features by downsampled time.
        """
        return self.blocks(functional.gelu(self.entry(signal)))


class DenseWaveformModel(nn.Module):
    """Encoder and fresh dense twelve-channel head for reconstruction or localization."""

    def __init__(self, width: int = 32, stride: int = 5, scale_mv: float = 2.0) -> None:
        super().__init__()
        self.encoder = WaveformEncoder(width, stride)
        self.head = nn.Conv1d(width, 12, 1)
        self.scale_mv = scale_mv

    def forward(self, signal_mv: torch.Tensor) -> torch.Tensor:
        """
        Return native-grid outputs from a single physical-unit input waveform.

        Parameters
        ----------
        signal_mv : torch.Tensor
            Batch by 12 leads by time, in mV.

        Returns
        -------
        torch.Tensor
            Unnormalized scores on the input grid; no paired counterpart is consumed.
        """
        encoded = self.encoder(signal_mv / self.scale_mv)
        output = self.head(encoded)
        return functional.interpolate(output, size=signal_mv.shape[-1], mode="linear", align_corners=False)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """
    Hash sorted parameter identities and exact tensor bytes, independent of torch serialization.

    Parameters
    ----------
    state : dict
        State dictionary containing CPU or device tensors.

    Returns
    -------
    str
        SHA-256 of names, dtypes, dimensions and contiguous bytes.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def masked_waveforms(
    signal: torch.Tensor, generator: torch.Generator, block_samples: int = 50, blocks: int = 13
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mask the same independently selected temporal blocks in every lead of a record.

    Parameters
    ----------
    signal : torch.Tensor
        Batch by lead by time physical waveforms.
    generator : torch.Generator
        CPU generator dedicated to masks, independent of model initialization and sampling.
    block_samples : int
        Native samples per block.
    blocks : int
        Number of distinct blocks masked in each record.

    Returns
    -------
    tuple of torch.Tensor
        Zero-masked signal and broadcast boolean mask with the same shape.
    """
    count = signal.shape[-1] // block_samples
    if signal.shape[-1] % block_samples or not 0 < blocks < count:
        raise ValueError("Mask blocks must partition the record with some visible samples remaining")
    selected = torch.rand(signal.shape[0], count, generator=generator).argsort(dim=1)[:, :blocks]
    native = torch.zeros(signal.shape[0], count, dtype=torch.bool)
    native.scatter_(1, selected, True)
    mask = native.repeat_interleave(block_samples, dim=1).unsqueeze(1).expand_as(signal).to(signal.device)
    return signal.masked_fill(mask, 0.0), mask


def ssl_update(
    model: DenseWaveformModel,
    optimizer: torch.optim.Optimizer,
    signal: torch.Tensor,
    generator: torch.Generator,
) -> float:
    """
    Make one masked-only reconstruction update on real physical waveforms.

    Parameters
    ----------
    model : DenseWaveformModel
        Encoder and temporary reconstruction head.
    optimizer : torch.optim.Optimizer
        Fresh SSL optimizer.
    signal : torch.Tensor
        Physical-unit waveform batch.
    generator : torch.Generator
        Independent temporal-mask RNG.

    Returns
    -------
    float
        Finite masked MSE before the update.
    """
    model.train()
    masked, mask = masked_waveforms(signal, generator)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(masked)
    loss = functional.mse_loss(prediction[mask], (signal / model.scale_mv)[mask])
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite SSL loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def localization_update(
    model: DenseWaveformModel,
    optimizer: torch.optim.Optimizer,
    signal: torch.Tensor,
    support: torch.Tensor,
    positive_weight: float,
    *,
    frozen_encoder: bool = False,
) -> float:
    """
    Make one full-map supervised update with an optional frozen encoder.

    Parameters
    ----------
    model : DenseWaveformModel
        Single-signal localizer.
    optimizer : torch.optim.Optimizer
        Fresh localization optimizer, without SSL state.
    signal : torch.Tensor
        Observed edited/zero-intervention waveforms only.
    support : torch.Tensor
        Exact thresholded intervention support.
    positive_weight : float
        One training-only fixed class weight.
    frozen_encoder : bool
        Hold encoder in evaluation mode for diagnostic arm F.

    Returns
    -------
    float
        Finite weighted BCE before the update.
    """
    model.train()
    if frozen_encoder:
        if any(parameter.requires_grad for parameter in model.encoder.parameters()):
            raise ValueError("Frozen arm requires non-trainable encoder parameters")
        model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    logits = model(signal)
    loss = functional.binary_cross_entropy_with_logits(
        logits, support.to(logits.dtype), pos_weight=logits.new_tensor(positive_weight)
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite localization loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def matched_schedule(kinds: np.ndarray, steps: int, seed: int) -> np.ndarray:
    """
    Construct identical 12-positive/2-sham/2-nuisance batches for every arm.

    Parameters
    ----------
    kinds : np.ndarray
        One positive, sham or nuisance string per fixed training example.
    steps : int
        Fixed optimizer update count.
    seed : int
        Dedicated NumPy schedule seed.

    Returns
    -------
    np.ndarray
        Step by 16 matrix of bank positions, cycling independently shuffled pools.
    """
    rng = np.random.default_rng(seed)
    columns = []
    for kind, size in (("positive", 12), ("sham", 2), ("nuisance", 2)):
        pool = np.flatnonzero(kinds == kind)
        if not len(pool):
            raise ValueError(f"Empty training pool: {kind}")
        cycles = [rng.permutation(pool) for _ in range((steps * size + len(pool) - 1) // len(pool))]
        columns.append(np.concatenate(cycles)[: steps * size].reshape(steps, size))
    return np.concatenate(columns, axis=1)


def model_description(model: DenseWaveformModel) -> dict[str, Any]:
    """
    Report architecture accounting without scoring model outputs.

    Parameters
    ----------
    model : DenseWaveformModel
        Actual model used by profile and full execution.

    Returns
    -------
    dict
        Parameter count and exact component initialization hashes.
    """
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "encoder_sha256": state_digest(model.encoder.state_dict()),
        "head_sha256": state_digest(model.head.state_dict()),
        "latent_interval_ms": 20,
        "score_grid_ms": 4,
    }
