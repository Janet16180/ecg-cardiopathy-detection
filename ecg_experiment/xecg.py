"""Adapter for the released xECG model and the official PTB-XL signal path.

The source and weights live in ``third_party/checkpoints/xecg``.  We import the
upstream class directly; this module only supplies backend conversion and I/O.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample
import torch
from torch import nn
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "third_party/checkpoints/xecg"
DEFAULT_XLSTM_DIR = ROOT / "third_party/xecg-deps"
RELEASE_SHA256 = {
    "config.json": "23ba68fc4edc5883f4a89a55f8da3d866ab040e64a46c9c862fde4bb52459b4a",
    "xECG.py": "373fed5a125abea218c74a8dc5b09e6054e85172a91f99e2ed47f8221fbe266f",
    "model.safetensors": "812dec69ac0fbf13f39e50bde4f35f85435a66966d8785baec65c2b5c70e722c",
}
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SOURCE_SAMPLES = 5000
MODEL_SAMPLES = 1000
MODEL_FS = 100
PATCH_SIZE = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_xecg(signal_12x5000: np.ndarray, input_fs: int = 500) -> np.ndarray:
    """Canonical 12-lead PTB-XL mV -> official 100 Hz time-major input.

    Upstream uses ``neurokit2.signal_resample(..., method='FFT')`` on a
    time-major array, which calls ``scipy.signal.resample`` along axis 0.
    Downstream PTB-XL defaults disable ECG cleaning and normalization.
    """
    signal = np.asarray(signal_12x5000)
    if input_fs != 500 or signal.shape != (12, SOURCE_SAMPLES):
        raise ValueError(f"Expected canonical [12, 5000] ECG at 500 Hz, got {signal.shape} at {input_fs} Hz")
    if not np.issubdtype(signal.dtype, np.number) or not np.isfinite(signal).all():
        raise ValueError("ECG samples must be finite numbers")
    output = resample(signal.T, MODEL_SAMPLES, axis=0).astype(np.float32)
    if output.shape != (MODEL_SAMPLES, 12) or not np.isfinite(output).all():
        raise ValueError("Invalid xECG resampling output")
    return np.ascontiguousarray(output)


def _official_class(checkpoint_dir: Path, xlstm_dir: Path):
    source = checkpoint_dir / "xECG.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official xECG source missing: {source}")
    try:
        import xlstm  # noqa: F401
    except ImportError:
        if not xlstm_dir.is_dir():
            raise ImportError("xLSTM 2.0.4 is required; install it in an isolated environment or third_party/xecg-deps") from None
        sys.path.insert(0, str(xlstm_dir))
        import xlstm  # noqa: F401
    import importlib.metadata
    if importlib.metadata.version("xlstm") != "2.0.4":
        raise RuntimeError("The released xECG requires xLSTM 2.0.4")
    spec = importlib.util.spec_from_file_location("_official_xecg_hf", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official xECG source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.xECG


def _cuda_to_vanilla_weights(state: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    """Convert sLSTM storage layout; the recurrent equations are unchanged.

    xLSTM CUDA stores [head, input, gate*output], while vanilla stores
    [head, gate*output, input].  The bias also changes head/gate order.
    Use xLSTM's own conversion methods so the mapping tracks version 2.0.4.
    """
    result = dict(state)
    cells = dict(model.named_modules())
    for key, value in state.items():
        if not key.endswith(("slstm_cell._recurrent_kernel_", "slstm_cell._bias_")):
            continue
        cell = cells[key.rsplit(".", 1)[0]]
        if key.endswith("_recurrent_kernel_"):
            result[key] = cell._recurrent_kernel_ext2int(value)
        else:
            result[key] = cell._bias_ext2int(value)
    return result


def load_xecg(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    *,
    backend: str = "vanilla",
    device: str | torch.device = "cpu",
    drop_path_prob: float = 0.5,
    xlstm_dir: Path = DEFAULT_XLSTM_DIR,
) -> nn.Module:
    """Instantiate the official xECG architecture and strictly load Hub weights.

    ``vanilla`` uses the exact upstream PyTorch sLSTM recurrence and supports
    CPU / older GPUs.  ``cuda`` uses the upstream custom kernel, when available.
    ``drop_path_prob`` only affects training and follows the official PTB-XL
    fine-tuning configuration (0.5); the released weights have 0.0.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    weight_path = checkpoint_dir / "model.safetensors"
    for filename, expected_hash in RELEASE_SHA256.items():
        if sha256(checkpoint_dir / filename) != expected_hash:
            raise ValueError(f"Released xECG {filename} SHA-256 mismatch")
    config = json.loads(config_path.read_text())
    if backend not in {"vanilla", "cuda"}:
        raise ValueError("backend must be 'vanilla' or 'cuda'")
    if config.get("sampling_freq") != MODEL_FS or config.get("patch_size") != PATCH_SIZE or config.get("embedding_size") != 1024:
        raise ValueError("Unexpected released xECG input or embedding configuration")
    if config.get("xlstm_config") != ["s", "s", "m", "m", "s", "s", "m", "m", "s"]:
        raise ValueError("Unexpected released xECG block configuration")
    constructor_config = {**config, "backend": backend, "drop_path_prob": float(drop_path_prob)}
    cls = _official_class(checkpoint_dir, Path(xlstm_dir))
    model = cls(cls_type=config["cls_type"], config=constructor_config)
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    if backend == "vanilla":
        state = _cuda_to_vanilla_weights(state, model)
    model.load_state_dict(state, strict=True)
    return model.to(device)


class XECGBinaryClassifier(nn.Module):
    """Official pooled xECG with a plain linear binary task head."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if backbone.cls_type not in {"avg", "mean"}:
            raise ValueError("Binary classifier requires average pooled xECG features")
        self.backbone = backbone
        self.head = nn.Linear(backbone.embedding_size, 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1:] != (MODEL_SAMPLES, 12):
            raise ValueError(f"Expected [batch, {MODEL_SAMPLES}, 12] xECG signals, got {tuple(signal.shape)}")
        pooled, _ = self.backbone(signal)
        return self.head(pooled).squeeze(-1)
