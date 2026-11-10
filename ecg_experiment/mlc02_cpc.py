"""Frozen released ECG-CPC encoder with the input path of ``scripts/features/extract_ecg_cpc.py``.

That script produced ``outputs/avance4_modelos/features/ecg_cpc`` (extractor SHA-256 ``048cb459...``) but
scripts cannot be imported, so its loader and preprocessing are repeated here unchanged: four consecutive
2.5 s crops of the 10 s, 500 Hz, mV window, each resampled to 240 Hz with ``resampy``'s default filter, no
normalization, mean over the 300 S4 tokens of each crop, then mean over the four crops (512 dimensions). The
author's ``clinical_ts`` code comes from ``third_party/ecg-fm-benchmarking`` and ``resampy`` from
``third_party/ecg-cpc-deps``.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file

CHECKPOINT_SHA256 = "bc253edc6ac279ce2caec86d3c3a21b740e43735ca03be1dc7c50e630b0ad4cf"
CHECKPOINT = ROOT / "third_party/checkpoints/ecg-cpc/ECG-CPC Checkpoint/last_11597276.ckpt"
REPOSITORY = ROOT / "third_party/ecg-fm-benchmarking"
REPOSITORY_COMMIT = "238409835ef55358a10bbc3459dfa9aaa91ad5e5"
DEPENDENCIES = ROOT / "third_party/ecg-cpc-deps"
SCRIPT = ROOT / "scripts/features/extract_ecg_cpc.py"
SCRIPT_SHA256 = "048cb4593d72c3b9ce8c2d43800386645e0c792a5ca2aace5a2f9c1b4e433feb"
EXPECTED_PREFIXES = ("ts_encoder.encoder.", "ts_encoder.predictor.")
SSL_HEAD_TENSORS = {"ts_encoder.head_ssl.proj.0.weight"}
OMEGA_LENGTH = 601
KERNEL_LENGTH_TOKENS = 1200
FEATURE_DIMENSION = 512
CROPS_PER_RECORD = 4
CROP_SAMPLES = 1250
TOKENS_PER_CROP = 300
VIEW_SHAPE = (12, 600)
BATCH = 16
SOURCE_RATE = 500
TARGET_RATE = 240

_CONFIG_TYPES: dict[tuple[str, str], type] = {}


class MetadataUnpickler(pickle.Unpickler):
    """Replace the training framework's config classes by placeholders; tensors unpickle unchanged."""

    def find_class(self, module: str, name: str) -> Any:
        """
        Resolve a pickled global, replacing framework config classes by placeholders.

        Parameters
        ----------
        module : str
            Module of the pickled global.
        name : str
            Name of the pickled global.

        Returns
        -------
        Any
            The resolved global or a placeholder class.

        Raises
        ------
        ValueError
            If a framework object other than a config class is pickled.
        """
        if not module.startswith("clinical_ts."):
            return super().find_class(module, name)
        if "Config" not in name:
            raise ValueError(f"Unexpected checkpoint object: {module}.{name}")
        return _CONFIG_TYPES.setdefault((module, name), type(name, (), {}))


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """
    Load the verified released checkpoint.

    Parameters
    ----------
    path : Path
        Released Lightning checkpoint.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, int]]
        State dict and the training epoch and global step.

    Raises
    ------
    ValueError
        If the checkpoint differs from the verified release.
    """
    if sha256_file(path) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 differs from verified Figshare release")
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=types.SimpleNamespace(__name__="ecg_cpc_metadata_pickle", Unpickler=MetadataUnpickler),
    )
    return payload["state_dict"], {"epoch": payload["epoch"], "global_step": payload["global_step"]}


def load_exact(module: nn.Module, state: dict[str, torch.Tensor], prefix: str) -> int:
    """
    Strictly load the tensors under a prefix into a module.

    Parameters
    ----------
    module : nn.Module
        Module to load.
    state : dict[str, torch.Tensor]
        Checkpoint state dict.
    prefix : str
        Key prefix of the module's tensors.

    Returns
    -------
    int
        Number of loaded tensors.
    """
    selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    module.load_state_dict(selected, strict=True, assign=True)
    return len(selected)


def torch_cauchy_conj(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Exact conjugate-pair Cauchy sum with native complex PyTorch operations.

    Parameters
    ----------
    v : torch.Tensor
        Complex numerators.
    z : torch.Tensor
        Complex evaluation nodes.
    w : torch.Tensor
        Complex poles, one per conjugate pair.

    Returns
    -------
    torch.Tensor
        Sum over poles of ``v / (z - w) + conj(v) / (z - conj(w))``.
    """
    dims = max(v.ndim, z.ndim, w.ndim)
    v, z, w = (x.reshape((1,) * (dims - x.ndim) + tuple(x.shape)) for x in (v, z, w))
    z, v, w = z.unsqueeze(-2), v.unsqueeze(-1), w.unsqueeze(-1)
    return (v / (z - w) + v.conj() / (z - w.conj())).sum(dim=-2)


def add_paths() -> None:
    """Make the author's ``clinical_ts`` package and the vendored ``resampy`` importable."""
    for path in (str(REPOSITORY / "code"), str(DEPENDENCIES)):
        if path not in sys.path:
            sys.path.insert(0, path)


def patch_s4_backend() -> None:
    """Route the official S4 Cauchy reduction through ``torch_cauchy_conj``, as the released script did."""
    from clinical_ts.ts.s4_modules import s42

    s42.has_cauchy_extension = False
    s42.cauchy_conj = torch_cauchy_conj


class ReleasedCPC(nn.Module):
    """Released ECG-CPC S4 encoder and predictor without the SSL head, frozen."""

    def __init__(self, checkpoint: Path = CHECKPOINT) -> None:
        """
        Build the encoder and strictly load the released backbone tensors.

        Parameters
        ----------
        checkpoint : Path
            Verified released checkpoint.

        Raises
        ------
        ValueError
            If the checkpoint holds unexpected tensors or S4 buffer lengths.
        """
        super().__init__()
        add_paths()
        from clinical_ts.template_modules import ShapeConfig, StaticStats
        from clinical_ts.ts.encoder import RNNEncoder, RNNEncoderConfig
        from clinical_ts.ts.s4 import S4Predictor, S4PredictorConfig

        patch_s4_backend()
        state, training = checkpoint_state(checkpoint)
        omitted = set(state) - {key for key in state if key.startswith(EXPECTED_PREFIXES)}
        if omitted != SSL_HEAD_TENSORS:
            raise ValueError(f"Unexpected non-backbone checkpoint tensors: {omitted}")
        omega_lengths = {value.shape[0] for key, value in state.items() if key.endswith(".omega")}
        if omega_lengths != {OMEGA_LENGTH}:
            raise ValueError(f"Unexpected S4 Fourier buffer lengths: {omega_lengths}")
        shape = ShapeConfig(channels=12, length=2400, sequence_last=True)
        self.encoder = RNNEncoder(
            RNNEncoderConfig(features=[FEATURE_DIMENSION] * 4, kss=[3, 1, 1, 1], strides=[2, 1, 1, 1]),
            shape,
            StaticStats(),
        )
        self.predictor = S4Predictor(
            S4PredictorConfig(causal=True, state_dim=8, model_dim=FEATURE_DIMENSION, backbone="s42"),
            self.encoder.get_output_shape(),
        )
        loaded = load_exact(self.encoder, state, EXPECTED_PREFIXES[0])
        loaded += load_exact(self.predictor, state, EXPECTED_PREFIXES[1])
        self.loading_info = {
            "loaded_backbone_tensors": loaded,
            "omitted_ssl_head_tensors": sorted(omitted),
            "kernel_length_tokens": KERNEL_LENGTH_TOKENS,
            "checkpoint_training": training,
            "cauchy_backend": "native PyTorch exact conjugate-pair Cauchy sum",
        }
        self.eval()
        self.requires_grad_(False)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool the S4 tokens of each 2.5-second view.

        Parameters
        ----------
        views : torch.Tensor
            ``[batch, 12, 600]`` views at 240 Hz.

        Returns
        -------
        torch.Tensor
            ``[batch, 512]`` features.

        Raises
        ------
        ValueError
            If the input or output has an unexpected shape or nonfinite values.
        """
        if views.ndim != 3 or views.shape[1:] != VIEW_SHAPE:
            raise ValueError("Expected 2.5-second views: [batch,12,600] at 240 Hz")
        tokens = self.predictor(**self.encoder(seq=views))["seq"]
        expected = (len(views), TOKENS_PER_CROP, FEATURE_DIMENSION)
        if tokens.shape != expected or not torch.isfinite(tokens).all():
            raise ValueError(f"Invalid ECG-CPC output {tuple(tokens.shape)}")
        return tokens.mean(dim=1)


def crop_starts() -> list[int]:
    """
    First sample of each 2.5 s crop of the 10 s window.

    Returns
    -------
    list[int]
        ``[0, 1250, 2500, 3750]``.
    """
    return list(range(0, CROPS_PER_RECORD * CROP_SAMPLES, CROP_SAMPLES))


def cpc_input(signal: np.ndarray) -> np.ndarray:
    """
    Official no-normalization mV input: crop first, then ``resampy`` 500 to 240 Hz.

    Parameters
    ----------
    signal : np.ndarray
        Float32 ``[12, 5000]`` 500 Hz physical-mV waveform.

    Returns
    -------
    np.ndarray
        Float32 ``[4, 12, 600]`` crops at 240 Hz.

    Raises
    ------
    ValueError
        If the waveform is malformed or nonfinite.
    """
    add_paths()
    import resampy

    if signal.shape != (12, CROPS_PER_RECORD * CROP_SAMPLES) or not np.isfinite(signal).all():
        raise ValueError("Expected finite [12,5000] 500-Hz physical mV waveform")
    views = [
        resampy.resample(signal[:, start : start + CROP_SAMPLES], SOURCE_RATE, TARGET_RATE, axis=-1)
        for start in crop_starts()
    ]
    return np.stack(views).astype(np.float32)


@torch.inference_mode()
def cpc_features(model: ReleasedCPC, inputs: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Pooled ECG-CPC representations in batches of sixteen records (64 views).

    Parameters
    ----------
    model : ReleasedCPC
        Frozen encoder on ``device``.
    inputs : np.ndarray
        Float32 ``[records, 4, 12, 600]`` outputs of ``cpc_input``.
    device : str
        Torch device.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 512]`` features, mean over the four crops.

    Raises
    ------
    ValueError
        If the output is malformed or nonfinite.
    """
    chunks = []
    for start in range(0, len(inputs), BATCH):
        batch = inputs[start : start + BATCH]
        views = np.ascontiguousarray(batch.reshape(-1, *VIEW_SHAPE))
        embedded = model(torch.from_numpy(views).to(device))
        embedded = embedded.reshape(len(batch), CROPS_PER_RECORD, FEATURE_DIMENSION).mean(dim=1)
        chunks.append(embedded.cpu().numpy())
    features = np.concatenate(chunks).astype(np.float32, copy=False)
    if features.shape != (len(inputs), FEATURE_DIMENSION) or not np.isfinite(features).all():
        raise ValueError(f"Invalid ECG-CPC features: {features.shape}")
    return features
