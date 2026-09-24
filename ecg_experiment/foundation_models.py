"""Published ECG foundation-model checkpoints and their official preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .files import sha256_file, sha256_json
from .waveforms import SAMPLE_RATE


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PER_VIEW = 2500
HF_MODELS = {
    "hubert-small": ("Edoardo-Coppola/hubert-ecg-small", "model.safetensors"),
    "ecg-fm": ("wanglab/ecg-fm", "mimic_iv_ecg_physionet_pretrained.pt"),
}
PREPROCESSING_SOURCES = ("ecg_experiment/foundation_models.py", "ecg_experiment/waveforms.py")


def preprocessing_source_sha256() -> str:
    """
    Identify the code that decodes and preprocesses records for these models.

    Returns
    -------
    str
        SHA-256 over the digests of ``PREPROCESSING_SOURCES``.
    """
    return sha256_json({name: sha256_file(ROOT / name) for name in PREPROCESSING_SOURCES})


def preprocess_hubert(signal: np.ndarray) -> np.ndarray:
    """
    Apply the official HuBERT-ECG preprocessing to one ten-second record.

    The official FIR/min-max pipeline is followed by the official flattened
    decimation of each five-second view.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` physical-mV signal at 500 Hz.

    Returns
    -------
    np.ndarray
        Float32 ``[2, 6000]`` array, one flattened view per row.
    """
    from scipy import signal as scipy_signal
    from hubert_ecg.utils import ecg_preprocessing

    normalized = ecg_preprocessing(signal, original_frequency=SAMPLE_RATE)
    views = []
    for start in (0, SAMPLES_PER_VIEW):
        view = normalized[:, start : start + SAMPLES_PER_VIEW]
        # ECGDataset flattens the 12 leads before decimating by five.
        views.append(scipy_signal.decimate(view.reshape(-1), 5).astype(np.float32))
    return np.stack(views)


def preprocess_ecg_fm(signal: np.ndarray) -> np.ndarray:
    """
    Apply the official ECG-FM preprocessing to one ten-second record.

    Each lead is z-scored over the full 500 Hz record, then split into two
    nonoverlapping five-second views. Constant leads become zeros.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` physical-mV signal at 500 Hz.

    Returns
    -------
    np.ndarray
        ``[2, 12, 2500]`` array of standardized views.
    """
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    standardized = np.divide(signal - mean, std, out=np.zeros_like(signal), where=std > 0)
    return np.stack((standardized[:, :SAMPLES_PER_VIEW], standardized[:, SAMPLES_PER_VIEW:]))


def checkpoint_info(model_name: str) -> tuple[Path | None, dict[str, str]]:
    """
    Download a pinned published checkpoint and describe its provenance.

    Parameters
    ----------
    model_name : str
        Key of ``HF_MODELS``.

    Returns
    -------
    tuple[Path | None, dict[str, str]]
        Local checkpoint path and its repository, revision, file name and
        SHA-256.
    """
    from huggingface_hub import HfApi, hf_hub_download

    repo, filename = HF_MODELS[model_name]
    revision = HfApi().model_info(repo).sha
    local_dir = ROOT / "third_party" / "checkpoints" / model_name
    checkpoint = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                 local_dir=local_dir)
    if model_name == "hubert-small":
        hf_hub_download(repo_id=repo, filename="config.json", revision=revision,
                        local_dir=local_dir)
    return Path(checkpoint), {"repo": repo, "revision": revision, "filename": filename,
                              "sha256": sha256_file(Path(checkpoint))}


def load_model(model_name: str, checkpoint: Path | None,
               device: str) -> tuple[Any, Callable[[np.ndarray], np.ndarray]]:
    """
    Load a published backbone in evaluation mode with its preprocessing.

    Parameters
    ----------
    model_name : str
        ``"hubert-small"`` or ``"ecg-fm"``.
    checkpoint : Path | None
        Checkpoint returned by ``checkpoint_info``; required.
    device : str
        Torch device for the model.

    Returns
    -------
    tuple[Any, Callable[[np.ndarray], np.ndarray]]
        The model and the matching record preprocessing function.
    """
    if model_name == "hubert-small":
        import hubert_ecg  # noqa: F401 - registers the Hugging Face model type
        from transformers import AutoModel

        # Use the downloaded snapshot cache. The official model implementation
        # registers its custom architecture when hubert_ecg is imported.
        assert checkpoint is not None
        model = AutoModel.from_pretrained(str(checkpoint.parent), local_files_only=True)
        preprocess = preprocess_hubert
    else:
        from fairseq_signals.models import build_model_from_checkpoint

        assert checkpoint is not None
        model = build_model_from_checkpoint(str(checkpoint))
        preprocess = preprocess_ecg_fm
    model.eval().to(device)
    return model, preprocess
