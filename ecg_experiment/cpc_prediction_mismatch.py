"""Frozen horizon-four CPC prediction mismatch and matched local controls.

The residual is a difference of unit vectors, retaining its error magnitude.
This uses the local CNN/GRU checkpoint; it is not a released S4 CPC model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from ecg_experiment.cpc import HORIZONS, CPCEncoder, CPCPretrainer

ARMS = ("context", "ordinary", "residual")
HORIZON = 4
FIRST_QUERY = 3
FIRST_TARGET = FIRST_QUERY + HORIZON
WIDTH = 256
BRANCH_WIDTH = 2 * WIDTH
BOOTSTRAP_EPOCHS = 20
BOOTSTRAP_SEED = 42
BOOTSTRAP_SETTINGS = {"variant": "cpc", "epochs": BOOTSTRAP_EPOCHS, "seed": BOOTSTRAP_SEED,
                      "horizons": list(HORIZONS), "cmsc_weight": 0}


def _check_bootstrap_identity(final: Mapping[str, Any], state: Mapping[str, Any],
                              config: Mapping[str, Any]) -> None:
    """Require the completed 20-epoch ordinary run and agreeing fingerprints."""
    settings = config.get("inputs", {}).get("settings", {})
    if (final.get("variant") != "cpc" or final.get("epochs") != BOOTSTRAP_EPOCHS
            or state.get("epoch") != BOOTSTRAP_EPOCHS
            or any(settings.get(key) != value for key, value in BOOTSTRAP_SETTINGS.items())):
        raise ValueError("Expected the completed 20-epoch Experiment 004 ordinary CPC checkpoint")
    fingerprint = final.get("fingerprint")
    if not fingerprint or not (fingerprint == state.get("fingerprint") == config.get("fingerprint")):
        raise ValueError("CPC final encoder, epoch state and configuration fingerprints disagree")


def _check_bootstrap_weights(final: Mapping[str, Any], weights: Mapping[str, Any]) -> None:
    """Require trained heads, the final encoder, and finite float32 tensors."""
    head_keys = {key for key in weights if key.startswith("heads.")}
    if head_keys != {f"heads.{i}.weight" for i in range(len(HORIZONS))}:
        raise ValueError("Checkpoint must contain all three trained bias-free CPC prediction heads")
    if any(weights[key].shape != (WIDTH, WIDTH) for key in head_keys):
        raise ValueError("CPC prediction head has the wrong shape")
    encoder = {key.removeprefix("encoder."): value for key, value in weights.items()
               if key.startswith("encoder.")}
    if encoder.keys() != final.get("encoder", {}).keys() or any(
            not torch.equal(value, final["encoder"][key]) for key, value in encoder.items()):
        raise ValueError("Completed CPC encoder does not match the final full epoch checkpoint")
    if not weights or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                          or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("CPC weights must be finite float32 tensors")


def validate_bootstrap(final: Mapping[str, Any], state: Mapping[str, Any],
                       config: Mapping[str, Any]) -> CPCPretrainer:
    """
    Require the completed ordinary CPC encoder and its trained future heads.

    Parameters
    ----------
    final : Mapping[str, Any]
        Contents of the run's ``encoder.pt``.
    state : Mapping[str, Any]
        Contents of the run's ``epoch_state.pt``.
    config : Mapping[str, Any]
        The run's ``config.json``.

    Returns
    -------
    CPCPretrainer
        Frozen eval-mode model with strictly loaded weights.

    Raises
    ------
    ValueError
        If the checkpoint is not the completed, consistent ordinary CPC run.
    """
    _check_bootstrap_identity(final, state, config)
    weights = state.get("model", {})
    _check_bootstrap_weights(final, weights)
    model = CPCPretrainer(hybrid=False)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def load_bootstrap(directory: str | Path, device: str | torch.device = "cpu") -> tuple[CPCPretrainer, dict]:
    """
    Load and validate the Experiment 004 ordinary CPC checkpoint.

    Parameters
    ----------
    directory : str | Path
        Run directory holding ``config.json``, ``encoder.pt`` and ``epoch_state.pt``.
    device : str | torch.device
        Device for the returned model.

    Returns
    -------
    tuple[CPCPretrainer, dict]
        Frozen model and the run configuration.
    """
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    final = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
    # This verified local training-state format includes NumPy RNG objects.
    state = torch.load(directory / "epoch_state.pt", map_location="cpu", weights_only=False)
    return validate_bootstrap(final, state, config).to(device), config


def aligned_representations(tokens: torch.Tensor, contexts: torch.Tensor,
                            prediction_head: nn.Module) -> dict[str, torch.Tensor]:
    """
    Align targets with horizon-four predictions: at target t, predict only from h[t-4].

    No targets come from the query warmup.

    Parameters
    ----------
    tokens : torch.Tensor
        CNN tokens of shape [batch, 2, time, width].
    contexts : torch.Tensor
        GRU contexts with the same shape.
    prediction_head : nn.Module
        The trained horizon-four head.

    Returns
    -------
    dict[str, torch.Tensor]
        ``context``, ``ordinary``, ``prediction`` and ``residual`` target-aligned tensors.

    Raises
    ------
    ValueError
        If the shapes are malformed, too short, or the head changes width.
    """
    if tokens.shape != contexts.shape or tokens.ndim != 4 or tokens.shape[1] != 2:
        raise ValueError("Expected matched [batch,2,tokens,width] token and context tensors")
    if tokens.shape[2] <= FIRST_TARGET:
        raise ValueError("Too few tokens after the shared query/target warmup")
    observed = F.normalize(tokens[:, :, FIRST_TARGET:], dim=-1, eps=1e-8)
    queries = contexts[:, :, FIRST_QUERY:-HORIZON]
    predicted = F.normalize(prediction_head(queries), dim=-1, eps=1e-8)
    if observed.shape != predicted.shape:
        raise ValueError("Prediction head changes the target feature width")
    return {"context": contexts[:, :, FIRST_TARGET:], "ordinary": observed,
            "prediction": predicted, "residual": observed - predicted}


def pool_branch(values: torch.Tensor) -> torch.Tensor:
    """
    Mean/max within each independent half, then mean across the two halves.

    Parameters
    ----------
    values : torch.Tensor
        Representations of shape [batch, 2, time, width].

    Returns
    -------
    torch.Tensor
        Pooled features of shape [batch, 2 * width].
    """
    return CPCEncoder.pooled(values)


@torch.inference_mode()
def extract_branches(model: CPCPretrainer, normalized_signal: torch.Tensor) -> torch.Tensor:
    """
    Extract pooled context, ordinary and residual features from a frozen model.

    Parameters
    ----------
    model : CPCPretrainer
        Frozen eval-mode model from ``validate_bootstrap``.
    normalized_signal : torch.Tensor
        Normalized signals of shape [batch, 12, 2500].

    Returns
    -------
    torch.Tensor
        Features of shape [batch, 3, 512] in ``ARMS`` order.

    Raises
    ------
    ValueError
        If the model is trainable or the features are malformed or nonfinite.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature extraction requires an eval-mode frozen CPC model")
    tokens, contexts = model.encoder(normalized_signal)
    aligned = aligned_representations(tokens, contexts, model.heads[0])
    branches = torch.stack([pool_branch(aligned[arm]) for arm in ARMS], dim=1)
    if branches.shape[1:] != (len(ARMS), BRANCH_WIDTH) or not torch.isfinite(branches).all():
        raise ValueError("Nonfinite or malformed frozen CPC features")
    return branches


def features_for_arm(branches: np.ndarray, arm: str) -> np.ndarray:
    """
    Select an arm's classifier features: context alone, or context plus one branch.

    Parameters
    ----------
    branches : np.ndarray
        Cached features of shape [records, 3, 512].
    arm : str
        One of ``ARMS``.

    Returns
    -------
    np.ndarray
        Features of shape [records, 512] or [records, 1024].

    Raises
    ------
    ValueError
        If the cache shape is wrong or the arm is unknown.
    """
    if branches.ndim != 3 or branches.shape[1:] != (len(ARMS), BRANCH_WIDTH):
        raise ValueError("Expected feature branches [records,3,512]")
    if arm not in ARMS:
        raise ValueError(f"Unknown mismatch arm: {arm}")
    if arm == "context":
        features = np.asarray(branches[:, 0])
    else:
        features = np.concatenate((branches[:, 0], branches[:, ARMS.index(arm)]), axis=1)
    return features


def supervised_indices(feature_rows: Sequence[Mapping[str, str]], selected_rows: Sequence[Mapping[str, str]],
                       expected_split: str) -> np.ndarray:
    """
    Select only the requested labeled rows, verifying patient and split IDs.

    Parameters
    ----------
    feature_rows : Sequence[Mapping[str, str]]
        Rows of the feature cache, in cache order.
    selected_rows : Sequence[Mapping[str, str]]
        Requested labeled rows.
    expected_split : str
        Split every requested row must belong to.

    Returns
    -------
    np.ndarray
        Int64 cache positions in request order.

    Raises
    ------
    ValueError
        On duplicate, absent, mismatched or unresolved rows.
    """
    index = {row["ecg_id"]: (i, row) for i, row in enumerate(feature_rows)}
    if len(index) != len(feature_rows):
        raise ValueError("Duplicate feature ECG identities")
    selected = []
    seen = set()
    for row in selected_rows:
        ecg_id = row["ecg_id"]
        if ecg_id in seen or ecg_id not in index:
            raise ValueError("Duplicate or absent requested supervised ECG")
        seen.add(ecg_id)
        position, cached = index[ecg_id]
        if cached["patient_id"] != row["patient_id"] or cached["split"] != expected_split:
            raise ValueError("Supervised feature patient/split mismatch")
        if row.get("target") not in ("0", "1"):
            raise ValueError("Supervised targets must be binary and resolved")
        selected.append(position)
    return np.asarray(selected, dtype=np.int64)
