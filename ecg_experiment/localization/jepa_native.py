"""Strict, pinned ECG-JEPA encoding with native lead/time patch anchors."""

from __future__ import annotations

import builtins
import hashlib
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from scipy.signal import resample

from ecg_experiment.files import sha256_json
from ecg_experiment.native_reference_ntt05 import NativeTokens, lead_patch_tokens

SOURCE_REVISION = "d937ad2c2c8a1e22856ce7e4a23a30f84a71217c"
SOURCE_SHA256 = {
    "ecg_jepa.py": "f7becbdf72fd7c6f1f7db092f74c2e68b87f94e5fcb6894fa9772a77518c6fc9",
    "pos_encoding.py": "eab89499f50f1c5baad65e25b27d4edef60ab8c7838b3f1a6374c5e1882fd094",
    "models.py": "f0f0790f38ba7b39f8878e3a5cbaaa5d4a72ff64cc6d5abc1e821915a2230072",
}
CHECKPOINT_SHA256 = "61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e"
CHECKPOINT_BYTES = 341551405
NATIVE_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
INPUT_LEAD_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)
ENCODER_PARAMETERS = {
    "encoder_embed_dim": 768,
    "encoder_depth": 12,
    "encoder_num_heads": 16,
    "predictor_embed_dim": 384,
    "predictor_depth": 6,
    "predictor_num_heads": 12,
    "c": 8,
    "pos_type": "sincos",
    "mask_scale": (0, 0),
    "leads": list(range(8)),
}
DENSE_ENCODER_PARAMETERS = {
    "embed_dim": ENCODER_PARAMETERS["encoder_embed_dim"],
    "depth": ENCODER_PARAMETERS["encoder_depth"],
    "num_heads": ENCODER_PARAMETERS["encoder_num_heads"],
    "c": ENCODER_PARAMETERS["c"],
    "pos_type": ENCODER_PARAMETERS["pos_type"],
    "mask_scale": ENCODER_PARAMETERS["mask_scale"],
    "leads": ENCODER_PARAMETERS["leads"],
    "p": 50,
    "t": 50,
}
PREPROCESSING = (
    "canonical 12-lead physical mV input [12,5000] at 500 Hz; select I,II,V1,V2,V3,V4,V5,V6; "
    "preserve selected input precision through scipy.signal.resample FFT along time to 2500 samples "
    "at 250 Hz, then cast to float32; "
    "no amplitude normalization, filtering, cropping, padding or derived leads"
)
CONTEXT_SCOPE = (
    "Full 10-second recording and all eight input leads: FFT resampling has global temporal support; "
    "12 encoder blocks propagate same-lead/all-time and same-time/all-lead attention. "
    "Lead/time anchors identify input patches, not isolated contextual receptive fields."
)
EMBEDDING_SPACE = "ecg-jepa-native-v1:" + sha256_json(
    {
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "encoder_parameters": ENCODER_PARAMETERS,
        "preprocessing": PREPROCESSING,
        "tokens": "post-encoder-normalization; lead-major; 8*50; 50 samples/200 ms; no special token",
        "mask": "explicit all-false Boolean [50]; no random masking",
    }
)


def _verified_sources(source_dir: Path) -> dict[str, bytes]:
    """Read and verify exact upstream bytes before any source execution."""
    sources = {name: (source_dir / name).read_bytes() for name in SOURCE_SHA256}
    for name, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256(sources[name]).hexdigest()
        if actual != expected:
            raise ValueError(f"ECG-JEPA source SHA-256 mismatch for {name}: {actual}")
    return sources


def _source_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
    *,
    positional_module: ModuleType,
) -> Any:
    """Bind the official absolute positional import within its private namespace."""
    if name == "pos_encoding" and level == 0:
        return positional_module
    return builtins.__import__(name, globals, locals, fromlist, level)


def _execute_source(name: str, path: Path, source: bytes, module_builtins: dict[str, Any]) -> ModuleType:
    """Execute verified bytes without registering global module or path aliases."""
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__["__builtins__"] = module_builtins
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _instantiate_encoder(source_dir: Path, sources: dict[str, bytes]) -> torch.nn.Module:
    """Construct the official encoder directly without the unused teacher or predictor.

    The pinned ``ecg_jepa.__init__`` forwards these arguments into ``MaskTransformer``.
    Its other encoder arguments have identical defaults in both official constructors.
    """
    positional = _execute_source(
        "_ecg_jepa_native_positional",
        source_dir / "pos_encoding.py",
        sources["pos_encoding.py"],
        vars(builtins).copy(),
    )
    private_builtins = dict(vars(builtins), __import__=partial(_source_import, positional_module=positional))
    official = _execute_source(
        "_ecg_jepa_native_official", source_dir / "ecg_jepa.py", sources["ecg_jepa.py"], private_builtins
    )
    return official.MaskTransformer(**DENSE_ENCODER_PARAMETERS)


def load_encoder(source_dir: Path, checkpoint: Path) -> torch.nn.Module:
    """Load the verified official encoder with every checkpoint key required.

    Parameters
    ----------
    source_dir : Path
        Local official source checkout containing the three pinned Python files.
    checkpoint : Path
        Complete official checkpoint with the pinned byte count and SHA-256.

    Returns
    -------
    torch.nn.Module
        Frozen float32 encoder in evaluation mode on CPU. Device transfers are caller-owned.

    Raises
    ------
    ValueError
        Source or checkpoint integrity fails before deserialization or source execution.
    RuntimeError
        The encoder state has a missing, unexpected or incompatible checkpoint key.
    """
    sources = _verified_sources(source_dir)
    with checkpoint.open("rb") as handle:
        handle.seek(0, 2)
        if handle.tell() != CHECKPOINT_BYTES:
            raise ValueError("ECG-JEPA checkpoint byte count mismatch")
        handle.seek(0)
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != CHECKPOINT_SHA256:
            raise ValueError(f"ECG-JEPA checkpoint SHA-256 mismatch: {actual}")
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
    encoder = _instantiate_encoder(source_dir, sources)
    encoder.load_state_dict(state["encoder"], strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder._ecg_jepa_embedding_space = EMBEDDING_SPACE
    return encoder


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """Select measured leads and FFT-resample one canonical physical-mV ECG.

    Parameters
    ----------
    signal : np.ndarray
        Real finite [12,5000] array in I, II, III, aVR, aVL, aVF, V1 through V6 order,
        representing exactly ten seconds at 500 Hz in physical millivolts.

    Returns
    -------
    np.ndarray
        Contiguous float32 [8,2500] array in I, II, V1 through V6 order at 250 Hz.
        Physical amplitudes are preserved; no record or lead normalization is applied.
    """
    signal = np.asarray(signal)
    if signal.shape != (12, 5000) or np.iscomplexobj(signal) or not np.isfinite(signal).all():
        raise ValueError("Expected real finite canonical [12,5000] physical-mV signal at 500 Hz")
    selected = signal[list(INPUT_LEAD_INDICES)]
    result = np.ascontiguousarray(resample(selected, 2500, axis=-1), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Resampled physical-mV signal cannot be represented as finite float32")
    return result


def encode_signal(encoder: torch.nn.Module, signal: np.ndarray, record_frame: str) -> NativeTokens:
    """Return all contextual native patch tokens without masking or extra tokens.

    Parameters
    ----------
    encoder : torch.nn.Module
        Verified frozen encoder returned by ``load_encoder``, optionally moved to a device.
    signal : np.ndarray
        Canonical [12,5000] physical-mV ECG; preprocessing is performed exactly once here.
    record_frame : str
        Explicit record and coordinate-origin identity for the full ten-second input.

    Returns
    -------
    NativeTokens
        Float32 [400,768] post-normalization vectors with exact 200 ms lead-major anchors
        and an explicit declaration of context beyond each anchor.
    """
    if getattr(encoder, "_ecg_jepa_embedding_space", None) != EMBEDDING_SPACE:
        raise ValueError("Encoder lacks the verified ECG-JEPA embedding-space binding")
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Expected frozen ECG-JEPA encoder in evaluation mode")
    if not record_frame:
        raise ValueError("Missing record coordinate-frame identity")
    device = next(encoder.parameters()).device
    prepared = torch.from_numpy(preprocess_signal(signal)).unsqueeze(0).to(device)
    mask = torch.zeros(50, dtype=torch.bool, device=device)
    with torch.inference_mode():
        values, returned_mask = encoder(prepared.reshape(1, 8, 50, 50), mask=mask)
    if values.shape != (1, 400, 768) or values.dtype != torch.float32:
        raise ValueError("Expected float32 [1,400,768] dense ECG-JEPA output without extra tokens")
    if returned_mask.shape != (50,) or returned_mask.dtype != torch.bool or returned_mask.any():
        raise ValueError("ECG-JEPA returned an incompatible or active patch mask")
    return lead_patch_tokens(
        values[0].cpu().numpy().reshape(8, 50, 768),
        NATIVE_LEADS,
        250.0,
        50,
        2500,
        EMBEDDING_SPACE,
        CONTEXT_SCOPE,
        record_frame,
    )
