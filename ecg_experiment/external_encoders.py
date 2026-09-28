"""Frozen released ECG-JEPA and xECG encoders with the input paths of the project's cached PTB-XL features.

The ECG-JEPA path copies ``scripts/features/extract_jepa.py`` (which produced
``data/processed/pretrained/ecg-jepa-full-public``) and the xECG path copies
``scripts/data/prepare_xecg.py`` and the extraction of
``scripts/experiments/run_xecg_probe_finetune016.py`` (which produced
``outputs/experiment016_xecg_probe_finetune/features``). Scripts cannot be
imported, so the few lines are repeated here and checked against the caches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from torch import nn

from . import ROOT
from .files import sha256_file
from .provenance import git_head
from .waveforms import LEADS, read_record
from .xecg import DEFAULT_CHECKPOINT_DIR, load_xecg, preprocess_xecg

PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
JEPA_SOURCE = ROOT / "third_party/ECG_JEPA"
JEPA_CHECKPOINT = ROOT / "third_party/checkpoints/ecg-jepa/multiblock_epoch100.pth"
JEPA_CACHE = ROOT / "data/processed/pretrained/ecg-jepa-full-public"
JEPA_LEADS = (0, 1, 6, 7, 8, 9, 10, 11)
JEPA_SAMPLES = 2500
JEPA_BATCH = 4
JEPA_DIMENSION = 768
XECG_CACHE = ROOT / "outputs/experiment016_xecg_probe_finetune/features"
XECG_BATCH = 16
XECG_DROP_PATH = 0.5
XECG_DIMENSION = 1024


def jepa_input(signal: np.ndarray) -> np.ndarray:
    """
    Apply the official ECG-JEPA PTB-XL lead selection and Fourier resampling.

    Parameters
    ----------
    signal : np.ndarray
        Float32 ``[12, 5000]`` 500 Hz waveform in mV, canonical lead order.

    Returns
    -------
    np.ndarray
        Float32 ``[8, 2500]`` waveform with leads I, II, V1-V6.

    Raises
    ------
    ValueError
        If the input shape is wrong or the result is nonfinite.
    """
    if signal.shape != (12, 5000):
        raise ValueError(f"Expected 12 leads x 5,000 samples, got {signal.shape}")
    reduced = resample(signal[list(JEPA_LEADS)], JEPA_SAMPLES, axis=1).astype(np.float32)
    if not np.isfinite(reduced).all():
        raise ValueError("Nonfinite resampled waveform")
    return reduced


def xecg_input(signal: np.ndarray) -> np.ndarray:
    """
    Convert a 500 Hz waveform to the official xECG 100 Hz time-major input from float64 samples.

    The cached PTB-XL views were resampled from WFDB's float64 samples, so the window is widened to
    float64 first. SPH stores float16, so the widening is exact.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` 500 Hz waveform in mV, canonical lead order.

    Returns
    -------
    np.ndarray
        Float32 ``[1000, 12]`` waveform.
    """
    return preprocess_xecg(np.asarray(signal, dtype=np.float64))


def read_ptb_float64(stem: str) -> np.ndarray:
    """
    Read a PTB-XL record in float64 mV and canonical lead order, as the xECG view cache did.

    Parameters
    ----------
    stem : str
        Record path relative to the PTB-XL release, without extension.

    Returns
    -------
    np.ndarray
        Float64 ``[12, 5000]`` signal.

    Raises
    ------
    ValueError
        If the path escapes the release or the record is not ten seconds of twelve-lead 500 Hz mV.
    """
    import wfdb

    base = (PTB_RAW / stem).resolve()
    if not base.is_relative_to(PTB_RAW.resolve()):
        raise ValueError(f"Record escapes raw directory: {stem}")
    signal, fields = wfdb.rdsamp(str(base))
    names = tuple(name.upper() for name in fields["sig_name"])
    expected = tuple(name.upper() for name in LEADS)
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
        raise ValueError(f"Unexpected WFDB leads for {stem}: {fields['sig_name']}")
    if fields["fs"] != 500 or signal.shape != (5000, 12):
        raise ValueError(f"Expected 10 seconds at 500 Hz for {stem}")
    if fields.get("units") != ["mV"] * 12:
        raise ValueError(f"Expected physical mV in all leads for {stem}: {fields.get('units')}")
    return signal[:, [names.index(name) for name in expected]].T


def ptb_jepa_input(stem: str) -> np.ndarray:
    """
    ECG-JEPA input of one raw PTB-XL record.

    Parameters
    ----------
    stem : str
        ``filename_hr`` of the record.

    Returns
    -------
    np.ndarray
        Float32 ``[8, 2500]`` waveform.
    """
    return jepa_input(read_record(PTB_RAW, stem))


def ptb_xecg_input(stem: str) -> np.ndarray:
    """
    Released xECG input of one raw PTB-XL record.

    Parameters
    ----------
    stem : str
        ``filename_hr`` of the record.

    Returns
    -------
    np.ndarray
        Float32 ``[1000, 12]`` waveform.
    """
    return preprocess_xecg(read_ptb_float64(stem))


def load_jepa() -> nn.Module:
    """
    Load the official frozen ECG-JEPA multiblock encoder on the GPU.

    Returns
    -------
    nn.Module
        Encoder in evaluation mode without gradients.
    """
    # The official code imports its sibling modules by top-level name ("models").
    if str(JEPA_SOURCE) not in sys.path:
        sys.path.insert(0, str(JEPA_SOURCE))
    from models import load_encoder as official_load_encoder

    encoder, _ = official_load_encoder(str(JEPA_CHECKPOINT))
    encoder.eval().to("cuda")
    encoder.requires_grad_(False)
    return encoder


def load_xecg_backbone() -> nn.Module:
    """
    Load the released xECG model on the GPU exactly as the Experiment 016 extraction did.

    Returns
    -------
    nn.Module
        Vanilla-backend model in evaluation mode without gradients.
    """
    model = load_xecg(DEFAULT_CHECKPOINT_DIR, device="cuda", backend="vanilla",
                      drop_path_prob=XECG_DROP_PATH).eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def jepa_features(encoder: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """
    Pooled ECG-JEPA representations in batches of four.

    Parameters
    ----------
    encoder : nn.Module
        Output of ``load_jepa``.
    inputs : np.ndarray
        Float32 ``[records, 8, 2500]`` inputs.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 768]`` features.

    Raises
    ------
    ValueError
        If the output is malformed or nonfinite.
    """
    chunks = []
    for start in range(0, len(inputs), JEPA_BATCH):
        tensor = torch.from_numpy(np.ascontiguousarray(inputs[start:start + JEPA_BATCH])).to("cuda")
        chunks.append(encoder.representation(tensor).cpu().numpy())
    features = np.concatenate(chunks).astype(np.float32, copy=False)
    if features.shape != (len(inputs), JEPA_DIMENSION) or not np.isfinite(features).all():
        raise ValueError(f"Invalid ECG-JEPA features: {features.shape}")
    return features


@torch.inference_mode()
def xecg_features(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """
    Pooled xECG representations in microbatches of sixteen.

    Parameters
    ----------
    model : nn.Module
        Output of ``load_xecg_backbone``.
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` inputs.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 1024]`` features.

    Raises
    ------
    ValueError
        If the output is malformed or nonfinite.
    """
    chunks = []
    for start in range(0, len(inputs), XECG_BATCH):
        tensor = torch.from_numpy(np.ascontiguousarray(inputs[start:start + XECG_BATCH])).to("cuda")
        pooled, _ = model(tensor)
        chunks.append(pooled.float().cpu().numpy())
    features = np.concatenate(chunks).astype(np.float32, copy=False)
    if features.shape != (len(inputs), XECG_DIMENSION) or not np.isfinite(features).all():
        raise ValueError(f"Invalid xECG features: {features.shape}")
    return features


def jepa_cache() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Open the cached ECG-JEPA features after checking the checkpoint and source they came from.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, str]]
        Memory-mapped features, integer ECG IDs of their rows, and file hashes.

    Raises
    ------
    ValueError
        If the checkpoint, official source revision or cache shape differs from the cache metadata.
    """
    metadata = json.loads((JEPA_CACHE / "metadata.json").read_text())
    names = ("features.npy", "ecg_ids.npy", "metadata.json")
    hashes = {name: sha256_file(JEPA_CACHE / name) for name in names}
    hashes["checkpoint"] = sha256_file(JEPA_CHECKPOINT)
    if hashes["checkpoint"] != metadata["checkpoint"]["sha256"]:
        raise ValueError("ECG-JEPA checkpoint differs from the cached features")
    if git_head(JEPA_SOURCE) != metadata["source_commit"]:
        raise ValueError("ECG-JEPA official source revision differs from the cached features")
    features = np.load(JEPA_CACHE / "features.npy", mmap_mode="r")
    ids = np.load(JEPA_CACHE / "ecg_ids.npy").astype(np.int64)
    if features.shape != (metadata["record_count"], JEPA_DIMENSION) or len(ids) != len(features):
        raise ValueError("Malformed ECG-JEPA cache")
    return features, ids, hashes


def xecg_cache() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    Open the cached xECG features after checking them against their receipt.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, str]]
        Memory-mapped features, integer ECG IDs of their rows, and file hashes.

    Raises
    ------
    ValueError
        If the files differ from the receipt or the checkpoint from the one the receipt names.
    """
    receipt = json.loads((XECG_CACHE / "receipt.json").read_text())
    names = ("features.npy", "ecg_ids.npy", "receipt.json")
    hashes = {name: sha256_file(XECG_CACHE / name) for name in names}
    if {name: hashes[name] for name in ("features.npy", "ecg_ids.npy")} != receipt["sha256"]:
        raise ValueError("xECG features differ from their receipt")
    checkpoint = {name: sha256_file(DEFAULT_CHECKPOINT_DIR / name)
                  for name in receipt["fingerprint"]["checkpoint_sha256"]}
    if checkpoint != receipt["fingerprint"]["checkpoint_sha256"]:
        raise ValueError("xECG checkpoint differs from the cached features")
    features = np.load(XECG_CACHE / "features.npy", mmap_mode="r")
    ids = np.load(XECG_CACHE / "ecg_ids.npy").astype(np.int64)
    if features.shape != (len(ids), XECG_DIMENSION):
        raise ValueError("Malformed xECG cache")
    return features, ids, {**hashes, **{f"checkpoint_{name}": value for name, value in checkpoint.items()}}


def source_files() -> tuple[Path, ...]:
    """
    Files whose code defines the encoder input paths reproduced here.

    Returns
    -------
    tuple[Path, ...]
        Historical extraction scripts, the official ECG-JEPA model code and the xECG adapter.
    """
    return (ROOT / "scripts/features/extract_jepa.py", ROOT / "scripts/data/prepare_xecg.py",
            ROOT / "scripts/experiments/run_xecg_probe_finetune016.py", ROOT / "ecg_experiment/xecg.py",
            JEPA_SOURCE / "models.py", DEFAULT_CHECKPOINT_DIR / "xECG.py")
