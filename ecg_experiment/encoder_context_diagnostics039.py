"""Training-only collapse gates and one-shot diagnostics for CPC Experiment 039."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import xlstm_study as base
from ecg_experiment.cpc_encoder_variants039 import architecture_spec, create_model
from ecg_experiment.cpc_pool import PoolDataset
from ecg_experiment.encoder_context_analysis039 import (
    CONTEXTS,
    ENCODERS,
    OUTPUT,
    SEEDS,
    TIERS,
    cell_path,
)
from ecg_experiment.files import sha256_file, write_json_atomic


def feature_health(features: np.ndarray) -> dict[str, float | bool | int]:
    """Measure finite pooled features and the frozen collapse criterion.

    Parameters
    ----------
    features : np.ndarray
        Training-only pooled feature matrix.

    Returns
    -------
    dict[str, float | bool | int]
        Finite status, low-variance fraction and collapse decision.
    """
    finite = bool(np.isfinite(features).all())
    if features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("Unexpected pooled feature shape")
    variance = np.var(features.astype(np.float64), axis=0) if finite else np.full(512, np.nan)
    fraction = float(np.mean(variance < 1e-8)) if finite else 1.0
    return {"training_records": len(features), "finite": finite,
            "fraction_variance_below_1e_8": fraction,
            "collapsed": bool(not finite or fraction >= 0.90),
            "mean_feature_variance": float(variance.mean()) if finite else 0.0}


def collapse_gate(root: Path) -> dict[str, Any]:
    """Check every original training feature matrix without reading development waveforms.

    Parameters
    ----------
    root : Path
        Repository root containing complete audited readouts.

    Returns
    -------
    dict[str, Any]
        Health checks and triggers for one-shot diagnostic work.
    """
    destination = root / OUTPUT / "diagnostic_gate.json"
    checks = {}
    triggers = []
    for tier in TIERS:
        for seed in SEEDS:
            for encoder in ENCODERS:
                path = cell_path(root, tier, encoder, seed) / "features.npz"
                with np.load(path, allow_pickle=False) as saved:
                    for context in CONTEXTS:
                        name = f"{tier}_{encoder}_{context}_{seed}"
                        checks[name] = {**feature_health(saved[f"train_{context}"]),
                                        "feature_archive_sha256": sha256_file(path)}
                        if checks[name]["collapsed"]:
                            triggers.append({"tier": tier, "seed": seed, "encoder": encoder,
                                             "context": context, "reason": "training_feature_collapse"})
    result = {"status": "passed" if not triggers else "diagnostic_required",
              "checks": checks, "triggers": triggers, "development_used": False}
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise ValueError("Completed diagnostic-gate evidence changed")
    write_json_atomic(destination, result, sort_keys=True)
    return result


def _effective_rank(features: np.ndarray) -> float:
    """Entropy effective rank of centered training features, using at most 2,048 rows."""
    x = features[:2048].astype(np.float64)
    x -= x.mean(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(x.T @ x / max(1, len(x) - 1)), 0)
    if eigenvalues.sum() <= 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _training_signal(root: Path) -> torch.Tensor:
    """Load eight known PTB training rows only, with the frozen train-only normalizer."""
    pool = base.pool(root)
    rows, _ = base.training_examples(pool)
    normalization = json.loads(base.normalizer(root).read_text())
    dataset = PoolDataset(pool, [row.cache_row for row in rows[:8]],
                          np.asarray(normalization["mean"], dtype=np.float32),
                          np.asarray(normalization["std"], dtype=np.float32))
    return torch.stack([dataset[index][0] for index in range(len(dataset))])


def diagnose(root: Path, encoder: str, context: str, triggers: list[dict[str, Any]]) -> dict[str, Any]:
    """Perform one training-only diagnostic for an affected encoder/context package.

    Parameters
    ----------
    root : Path
        Repository root.
    encoder : str
        Affected waveform encoder.
    context : str
        Affected context network.
    triggers : list[dict[str, Any]]
        Prespecified failures, with tier and seed; no new development analysis.

    Returns
    -------
    dict[str, Any]
        One-shot evidence and whether it identifies a specific implementation defect.
    """
    directory = root / OUTPUT / "diagnostics" / f"{encoder}_{context}"
    destination = directory / "diagnostic.json"
    if destination.exists():
        return json.loads(destination.read_text())
    started = time.monotonic()
    trigger = triggers[0]
    path = cell_path(root, trigger["tier"], encoder, trigger["seed"])
    checkpoint_path = path / context / "latest.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(encoder, context, trigger["seed"], "cpu")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    signal = _training_signal(root)
    tokens, contexts = model.encoder(signal)
    loss, _ = model(signal)
    loss.backward()
    gradients = {}
    prefixes = (("frontend", "encoder.convs."), ("context", "encoder.context."), ("heads", "heads."))
    for group, prefix in prefixes:
        values = [parameter.grad for name, parameter in model.named_parameters()
                  if name.startswith(prefix) and parameter.grad is not None]
        gradients[group] = {"finite": all(bool(torch.isfinite(value).all()) for value in values),
                            "norm": float(torch.sqrt(sum(value.square().sum() for value in values)))
                            if values else 0.0,
                            "tensors": len(values)}
    with np.load(path / "features.npz", allow_pickle=False) as features:
        training_features = features[f"train_{context}"]
        health = feature_health(training_features)
        effective_rank = _effective_rank(training_features)
    losses = np.asarray(saved["losses"], dtype=np.float64)
    finite = bool(base.finite_tree(saved) and torch.isfinite(loss) and torch.isfinite(tokens).all()
                  and torch.isfinite(contexts).all() and all(value["finite"] for value in gradients.values()))
    live_groups = all(value["norm"] > 0 for value in gradients.values())
    defect = bool(not finite or not live_groups or health["collapsed"])
    conclusion = ("Training-only checks identify a finite-state, dead-gradient or collapse defect; "
                  "a separate committed correction protocol is required before the single retry."
                  if defect else "Finite states, nonzero gradients and training feature variance passed. "
                  "No specific defect was identified; retain the negative result without tuning.")
    result = {"status": "one_diagnostic_complete", "encoder": encoder, "context": context,
              "triggered_cells": triggers, "diagnostic_tier": trigger["tier"],
              "diagnostic_seed": trigger["seed"], "checkpoint_sha256": sha256_file(checkpoint_path),
              "training_feature_health": health, "effective_rank_first_2048_training_rows": effective_rank,
              "gradients": gradients, "finite_state": finite, "all_group_gradients_nonzero": live_groups,
              "diagnostic_loss": float(loss.detach()), "first_100_mean_loss": float(losses[:100].mean()),
              "last_100_mean_loss": float(losses[-100:].mean()),
              "token_variance": float(tokens.detach().var()),
              "context_variance": float(contexts.detach().var()),
              "normalized_training_signal_mean": float(signal.mean()),
              "normalized_training_signal_std": float(signal.std()),
              "architecture": architecture_spec(encoder, context), "specific_defect_identified": defect,
              "conclusion": conclusion, "diagnostic_attempts": 1, "development_used_for_diagnosis": False,
              "elapsed_seconds": time.monotonic() - started}
    write_json_atomic(destination, result, sort_keys=True)
    return result
