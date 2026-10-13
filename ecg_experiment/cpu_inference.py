"""CPU inference helpers for pipeline v4 and its explanation: encoders on the CPU, heads and peak memory."""

from __future__ import annotations

import resource
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ann_heads import AttentionHead
from .external_encoders import JEPA_CHECKPOINT, JEPA_SOURCE
from .pipeline_v4 import load_state

ATTENTION_SEEDS = (43043, 43044, 43045)
JEPA_WIDTH = 768


def load_jepa_cpu(checkpoint: Path = JEPA_CHECKPOINT) -> nn.Module:
    """
    Load the released ECG-JEPA encoder on the CPU, as ``external_encoders.load_jepa`` does on the GPU.

    The official loader calls ``torch.load`` without a map location, so the same encoder is built here with
    the checkpoint mapped to the CPU.

    Parameters
    ----------
    checkpoint : Path
        Released multiblock checkpoint.

    Returns
    -------
    nn.Module
        Encoder in evaluation mode without gradients, on the CPU.
    """
    if str(JEPA_SOURCE) not in sys.path:
        sys.path.insert(0, str(JEPA_SOURCE))
    from ecg_jepa import ecg_jepa

    params = {"encoder_embed_dim": 768, "encoder_depth": 12, "encoder_num_heads": 16,
              "predictor_embed_dim": 384, "predictor_depth": 6, "predictor_num_heads": 12, "c": 8,
              "pos_type": "sincos", "mask_scale": (0, 0), "leads": [0, 1, 2, 3, 4, 5, 6, 7]}
    encoder = ecg_jepa(**params).encoder
    state = torch.load(str(checkpoint), map_location="cpu")
    encoder.load_state_dict(state["encoder"])
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def load_attention_heads(path: Path, seeds: tuple[int, ...] = ATTENTION_SEEDS) -> list[nn.Module]:
    """Load pipeline v4's attention networks from their saved weights, on the CPU in evaluation mode."""
    with np.load(path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return [load_state(AttentionHead(JEPA_WIDTH), arrays, seed).eval() for seed in seeds]


@torch.inference_mode()
def attention_scores(heads: list[nn.Module], tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Score tokens with every attention network after the float16 storage round trip of pipeline v4.

    Parameters
    ----------
    heads : list[nn.Module]
        Output of ``load_attention_heads``.
    tokens : np.ndarray
        ``[records, 400, 768]`` ECG-JEPA tokens.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mean logit ``[records]`` and mean per-token contribution ``[records, 400]`` over the networks.
    """
    tensor = torch.from_numpy(np.asarray(tokens, dtype=np.float16).astype(np.float32))
    found = [head(tensor) for head in heads]
    logits = torch.stack([logit for logit, _ in found]).mean(dim=0)
    contributions = torch.stack([parts for _, parts in found]).mean(dim=0)
    return logits.numpy().astype(np.float64), contributions.numpy().astype(np.float64)


def linear_logit(parameters: dict[str, np.ndarray], prefix: str, x: np.ndarray) -> np.ndarray:
    """Return a saved logistic head's linear part, ``(x - mean) / scale . coef + intercept``."""
    values = np.asarray(x, dtype=np.float64)
    standardized = (values - parameters[f"{prefix}_mean"]) / parameters[f"{prefix}_scale"]
    return standardized @ parameters[f"{prefix}_coef"] + parameters[f"{prefix}_intercept"][0]


def peak_rss_mib(children: bool = False) -> float:
    """Return the peak resident set size of this process (or of its finished children) in MiB."""
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return resource.getrusage(who).ru_maxrss / 1024
