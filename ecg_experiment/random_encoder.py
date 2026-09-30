"""Random-weight and fine-tuned xECG encoders and seed-averaged bootstrap contrasts for Experiment 036.

The random encoders are the released xECG architecture built from a fixed seed with nothing loaded, so
they share the pretrained model's input path, forward pass and pooling. The contrasts reuse the draws of
``intervals.paired_auroc_difference``: one AUROC per score and valid draw, then any weighted sum of them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .external_encoders import XECG_DROP_PATH
from .intervals import patient_groups, two_class_resamples
from .xecg import DEFAULT_CHECKPOINT_DIR, DEFAULT_XLSTM_DIR, XECGBinaryClassifier, _official_class, load_xecg

RANDOM_SEEDS = (36001, 36002, 36003)


def build_random_xecg(seed: int, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> nn.Module:
    """
    Build the released xECG architecture on the CPU with its own seeded initialization.

    Parameters
    ----------
    seed : int
        Seed passed to ``torch.manual_seed`` immediately before the model is built.
    checkpoint_dir : Path
        Directory with the released ``config.json`` and ``xECG.py``; the weights are not read.

    Returns
    -------
    nn.Module
        Model in evaluation mode without gradients, vanilla sLSTM backend.
    """
    config = json.loads((checkpoint_dir / "config.json").read_text())
    constructor_config = {**config, "backend": "vanilla", "drop_path_prob": XECG_DROP_PATH}
    cls = _official_class(checkpoint_dir, DEFAULT_XLSTM_DIR)
    torch.manual_seed(seed)
    model = cls(cls_type=config["cls_type"], config=constructor_config).eval()
    model.requires_grad_(False)
    return model


def load_finetuned_xecg(path: Path) -> nn.Module:
    """
    Load the backbone of an Experiment 016 fine-tuned binary classifier on the CPU.

    Parameters
    ----------
    path : Path
        ``model.pt`` holding the classifier's state dict under ``model``.

    Returns
    -------
    nn.Module
        The backbone in evaluation mode without gradients.
    """
    backbone = load_xecg(DEFAULT_CHECKPOINT_DIR, backend="vanilla", device="cpu",
                         drop_path_prob=XECG_DROP_PATH)
    classifier = XECGBinaryClassifier(backbone)
    classifier.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
    classifier.eval().requires_grad_(False)
    return classifier.backbone


def finetuned_head(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Weight and bias of an Experiment 016 fine-tuned classifier's linear head.

    Parameters
    ----------
    path : Path
        ``model.pt`` holding the classifier's state dict under ``model``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float32 weight ``[1, 1024]`` and bias ``[1]``.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    return state["head.weight"].numpy(), state["head.bias"].numpy()


def state_digest(model: nn.Module) -> str:
    """
    SHA-256 over every state-dict entry's name, dtype, shape and bytes, in state-dict order.

    Parameters
    ----------
    model : nn.Module
        Model to fingerprint.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{values.dtype}|{tuple(values.shape)}".encode())
        digest.update(values.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def auroc_draws(units: np.ndarray, y: np.ndarray, scores: dict[str, np.ndarray], draws: int, seed: int
                ) -> tuple[dict[str, float], np.ndarray, int]:
    """
    Observed AUROC of every score and its AUROC in each valid whole-patient draw.

    The draws are those of ``intervals.paired_auroc_difference`` with the same seed.

    Parameters
    ----------
    units : np.ndarray
        Patient (or record) of each ECG.
    y : np.ndarray
        Binary labels.
    scores : dict[str, np.ndarray]
        Scores on the same ECGs, by name.
    draws : int
        Draws taken, including skipped single-class ones.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple[dict[str, float], np.ndarray, int]
        Observed AUROC per name, a ``[valid draws, names]`` AUROC matrix in ``scores`` order, and the
        number of skipped draws.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    matrix = np.array([[roc_auc_score(y[rows], values[rows]) for values in scores.values()]
                       for rows in two_class_resamples(patient_groups(np.asarray(units)), y, draws, rng)])
    observed = {name: float(roc_auc_score(y, values)) for name, values in scores.items()}
    return observed, matrix, draws - len(matrix)


def linear_contrast(observed: dict[str, float], matrix: np.ndarray, weights: dict[str, float]
                    ) -> dict[str, float]:
    """
    Compute a weighted sum of AUROCs, observed and as a percentile interval over the draws.

    Parameters
    ----------
    observed : dict[str, float]
        Observed AUROC per name, in the column order of ``matrix``.
    matrix : np.ndarray
        Output of ``auroc_draws``.
    weights : dict[str, float]
        Weight per name, for example ``{"a": 1, "b": -0.5, "c": -0.5}``.

    Returns
    -------
    dict[str, float]
        ``difference``, ``ci_low`` and ``ci_high`` (2.5 and 97.5 percentiles).
    """
    vector = np.array([weights.get(name, 0.0) for name in observed])
    low, high = np.percentile(matrix @ vector, [2.5, 97.5])
    return {"difference": float(np.array(list(observed.values())) @ vector),
            "ci_low": float(low), "ci_high": float(high)}


def seed_mean_weights(first: str, others: tuple[str, ...], sign: float = 1.0) -> dict[str, float]:
    """
    Weights of ``first`` minus the mean of ``others``, times ``sign``.

    Parameters
    ----------
    first : str
        Name counted with weight ``sign``.
    others : tuple[str, ...]
        Names averaged with total weight ``-sign``.
    sign : float
        Overall sign.

    Returns
    -------
    dict[str, float]
        Weight per name.
    """
    return {first: sign, **{name: -sign / len(others) for name in others}}


def interval_side(contrast: dict[str, float]) -> str:
    """
    Where an interval lies relative to zero.

    Parameters
    ----------
    contrast : dict[str, float]
        With ``ci_low`` and ``ci_high``.

    Returns
    -------
    str
        ``"above 0"``, ``"below 0"`` or ``"includes 0"``.
    """
    side = "includes 0"
    if contrast["ci_low"] > 0:
        side = "above 0"
    elif contrast["ci_high"] < 0:
        side = "below 0"
    return side


def decision(mean_contrast: dict[str, float], seed_contrasts: list[dict[str, float]], margin: float) -> str:
    """
    Apply the Experiment 036 rule to the primary contrasts.

    Parameters
    ----------
    mean_contrast : dict[str, float]
        Pretrained minus the mean of the random seeds.
    seed_contrasts : list[dict[str, float]]
        Pretrained minus each random seed.
    margin : float
        The material difference, 0.02 AUROC.

    Returns
    -------
    str
        ``pretraining_needed``, ``pretraining_not_needed`` or ``inconclusive``.
    """
    verdict = "inconclusive"
    if mean_contrast["ci_low"] >= margin and all(item["ci_low"] > 0 for item in seed_contrasts):
        verdict = "pretraining_needed"
    elif mean_contrast["ci_high"] < margin:
        verdict = "pretraining_not_needed"
    return verdict
