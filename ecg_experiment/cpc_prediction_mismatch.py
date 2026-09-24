"""Frozen horizon-four CPC prediction mismatch and matched local controls.

The residual is a difference of unit vectors, retaining its error magnitude.
This uses the local CNN/GRU checkpoint; it is not a released S4 CPC model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ecg_experiment.cpc import CPCPretrainer, HORIZONS


ARMS = ("context", "ordinary", "residual")
HORIZON = 4
FIRST_QUERY = 3
FIRST_TARGET = FIRST_QUERY + HORIZON
WIDTH = 256
BRANCH_WIDTH = 2 * WIDTH


def validate_bootstrap(final, state, config):
    """Require the completed ordinary CPC encoder and its trained future heads."""
    settings = config.get("inputs", {}).get("settings", {})
    if (final.get("variant") != "cpc" or final.get("epochs") != 20
            or state.get("epoch") != 20 or settings.get("variant") != "cpc"
            or settings.get("epochs") != 20 or settings.get("seed") != 42
            or settings.get("horizons") != list(HORIZONS) or settings.get("cmsc_weight") != 0):
        raise ValueError("Expected the completed 20-epoch Experiment 004 ordinary CPC checkpoint")
    if not final.get("fingerprint") or not final["fingerprint"] == state.get("fingerprint") == config.get("fingerprint"):
        raise ValueError("CPC final encoder, epoch state and configuration fingerprints disagree")
    weights = state.get("model", {})
    head_keys = {key for key in weights if key.startswith("heads.")}
    if head_keys != {f"heads.{i}.weight" for i in range(3)}:
        raise ValueError("Checkpoint must contain all three trained bias-free CPC prediction heads")
    if any(weights[key].shape != (WIDTH, WIDTH) for key in head_keys):
        raise ValueError("CPC prediction head has the wrong shape")
    encoder = {key.removeprefix("encoder."): value for key, value in weights.items() if key.startswith("encoder.")}
    if encoder.keys() != final.get("encoder", {}).keys() or any(
            not torch.equal(value, final["encoder"][key]) for key, value in encoder.items()):
        raise ValueError("Completed CPC encoder does not match the final full epoch checkpoint")
    if not weights or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                          or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("CPC weights must be finite float32 tensors")
    model = CPCPretrainer(hybrid=False)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def load_bootstrap(directory: Path, device="cpu"):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    final = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
    # This verified local training-state format includes NumPy RNG objects.
    state = torch.load(directory / "epoch_state.pt", map_location="cpu", weights_only=False)
    return validate_bootstrap(final, state, config).to(device), config


def aligned_representations(tokens: torch.Tensor, contexts: torch.Tensor, prediction_head: nn.Module):
    """At target t, predict only from h[t-4]; no targets from query warmup."""
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


def pool_branch(values: torch.Tensor):
    """Mean/max within each independent half, then mean across the two halves."""
    return torch.cat((values.mean(dim=2), values.amax(dim=2)), dim=-1).mean(dim=1)


@torch.inference_mode()
def extract_branches(model: CPCPretrainer, normalized_signal: torch.Tensor):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature extraction requires an eval-mode frozen CPC model")
    tokens, contexts = model.encoder(normalized_signal)
    aligned = aligned_representations(tokens, contexts, model.heads[0])
    branches = torch.stack([pool_branch(aligned[arm]) for arm in ARMS], dim=1)
    if branches.shape[1:] != (3, BRANCH_WIDTH) or not torch.isfinite(branches).all():
        raise ValueError("Nonfinite or malformed frozen CPC features")
    return branches


def features_for_arm(branches: np.ndarray, arm: str):
    if branches.ndim != 3 or branches.shape[1:] != (3, BRANCH_WIDTH):
        raise ValueError("Expected feature branches [records,3,512]")
    if arm == "context":
        return np.asarray(branches[:, 0])
    if arm not in ("ordinary", "residual"):
        raise ValueError(f"Unknown mismatch arm: {arm}")
    return np.concatenate((branches[:, 0], branches[:, ARMS.index(arm)]), axis=1)


def supervised_indices(feature_rows, selected_rows, expected_split):
    """Select only the requested labeled rows, verifying patient and split IDs."""
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
