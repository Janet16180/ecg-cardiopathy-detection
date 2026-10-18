"""Disjoint latent-subject banks for the prospectively frozen engineering pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_file

ARRAYS = ("signals", "baselines", "support", "prior", "subjects", "kinds", "modes")


def latent_seed(role: str, index: int) -> int:
    """
    Derive one stable latent identity without shared RNG state or patient information.

    Parameters
    ----------
    role : str
        Training, validation or evaluation namespace.
    index : int
        Nonnegative subject position within that namespace.

    Returns
    -------
    int
        Unsigned 64-bit generator seed from the first eight SHA-256 bytes.
    """
    if role not in {"train", "validation", "evaluation"} or index < 0:
        raise ValueError("Unknown synthetic role or negative subject index")
    payload = f"ssl_pilot_v1:synthetic:{role}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def case_specifications(role: str, count: int, positives: int = 375) -> list[dict[str, Any]]:
    """
    Assign prospective case kinds with independent nuisance flags for positive subjects.

    Parameters
    ----------
    role : str
        Synthetic data role, never a real-data partition name for payload loading.
    count : int
        Number of independent latent subjects.
    positives : int
        Positive training subjects among the fixed total training examples.

    Returns
    -------
    list of dict
        Generator-only parameters; never supplied as model features.
    """
    cases = []
    for index in range(count):
        seed = latent_seed(role, index)
        base = {"subject": index, "seed": seed}
        if role == "train":
            kind = "positive" if index < positives else "nuisance" if index % 2 else "sham"
            cases.append(
                dict(
                    base,
                    kind=kind,
                    mode="focal" if kind != "positive" or index % 2 == 0 else "persistent",
                    target=kind == "positive",
                    nuisance=bool(seed & 1) if kind == "positive" else kind == "nuisance",
                )
            )
            continue
        for mode in ("focal", "persistent"):
            cases.append(dict(base, kind="positive", mode=mode, target=True, nuisance=bool(seed & 1)))
        cases.append(dict(base, kind="sham", mode="focal", target=False, nuisance=False))
        cases.append(dict(base, kind="nuisance", mode="focal", target=False, nuisance=True))
    return cases


def build_bank(
    role: str,
    count: int,
    profile: str,
    render: Callable[..., Any],
    qualify: Callable[[Any], list[str]],
) -> dict[str, Any]:
    """
    Generate a fixed bank, retaining counterparts for truth only and rejecting every bad case.

    Parameters
    ----------
    role : str
        Disjoint latent namespace.
    count : int
        Independent subject count.
    profile : str
        A for training/validation, A or B for held-out subject evaluation.
    render : callable
        Frozen qualified paired generator.
    qualify : callable
        Frozen per-pair gate returning every failure, without filtering or retries.

    Returns
    -------
    dict
        Waveform/mask/diagnostic arrays and label-only case parameters.
    """
    if role in {"train", "validation"} and profile != "A":
        raise ValueError("Held-out edit profile cannot enter training or validation")
    specifications = case_specifications(role, count)
    signals, baselines, supports, priors, metadata = [], [], [], [], []
    for spec in specifications:
        pair = render(
            spec["seed"],
            profile=profile,
            mode=spec["mode"],
            sample_rate=250,
            target=spec["target"],
            nuisance=spec["nuisance"],
        )
        failures = qualify(pair)
        if failures:
            raise ValueError(f"Prospective synthetic case failed qualification: {failures}")
        signals.append(pair.edited)
        baselines.append(pair.baseline)
        supports.append(pair.support)
        priors.append(pair.candidate)
        metadata.append({**spec, "parameters": pair.metadata})
    return {
        "signals": np.stack(signals).astype(np.float32),
        "baselines": np.stack(baselines).astype(np.float32),
        "support": np.stack(supports).astype(bool),
        "prior": np.stack(priors).astype(bool),
        "subjects": np.array([spec["subject"] for spec in specifications]),
        "kinds": np.array([spec["kind"] for spec in specifications]),
        "modes": np.array([spec["mode"] for spec in specifications]),
        "metadata": metadata,
    }


def save_bank(bank: dict[str, Any], directory: Path) -> dict[str, str]:
    """
    Save arrays and label-only parameters to a new local directory, refusing overwrite.

    Parameters
    ----------
    bank : dict
        Output of ``build_bank``.
    directory : Path
        New local bank destination.

    Returns
    -------
    dict
        File-name to SHA-256 mapping for subsequent source-bound receipts.
    """
    directory.mkdir(parents=True, exist_ok=False)
    for name in ARRAYS:
        np.save(directory / f"{name}.npy", bank[name], allow_pickle=False)
    (directory / "metadata.json").write_text(json.dumps(bank["metadata"], indent=2) + "\n")
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir())}


def load_bank(directory: Path) -> dict[str, np.ndarray]:
    """
    Open native arrays read-only, excluding paired/parameter channels from model selection.

    Parameters
    ----------
    directory : Path
        Previously bound local bank.

    Returns
    -------
    dict
        Read-only arrays; the caller explicitly passes only signals to the model.
    """
    return {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in ARRAYS}


def positive_class_weight(bank: dict[str, np.ndarray], schedule: np.ndarray) -> float:
    """
    Compute one fixed class weight from actual training schedule proportions only.

    Parameters
    ----------
    bank : dict
        Training arrays, with no validation or engineering-evaluation labels.
    schedule : np.ndarray
        Matched step-by-batch training positions.

    Returns
    -------
    float
        Clipped ratio of negative to positive cells across actual training presentations.
    """
    per_case = bank["support"].sum(axis=(1, 2), dtype=np.int64)
    presentations = np.bincount(schedule.ravel(), minlength=len(per_case))
    positive = int(presentations @ per_case)
    total = int(schedule.size * np.prod(bank["support"].shape[1:]))
    if not 0 < positive < total:
        raise ValueError("Training schedule must contain support and background cells")
    return min(20.0, (total - positive) / positive)
