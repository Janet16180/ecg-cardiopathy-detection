"""Verified cached inputs and predecessor replay for KEY-01."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ROOT
from .condition_keys import BATCH, GROUPS
from .files import sha256_file
from .paths import from_stored, to_stored
from .pr74_screening_metrics import point_metrics

CACHE = Path("outputs/heads01_token_superclass_20261028")
COHORT = Path("outputs/cohort_age_label_priority_20261028/all_age.csv")
SPH = Path("outputs/pr74_fresh_baseline_exact_20261028/features/sph_rows.csv")


def json_read(path: Path) -> Any:
    """
    Read a local JSON artifact.

    Parameters
    ----------
    path : Path
        JSON artifact path.

    Returns
    -------
    Any
        Decoded JSON value.
    """
    return json.loads(path.read_text())


def verify_inputs() -> dict[str, str]:
    """
    Check all directly used predecessor artifacts against their saved bindings.

    Returns
    -------
    dict[str, str]
        Verified repository-relative input hashes, including completion receipts.
    """
    expected, receipts = {}, {}
    for stage in ("extract_training", "extract_evaluation", "qualify", "evaluate"):
        path = CACHE / stage / "completion.json"
        receipt = json_read(ROOT / path)
        if receipt["status"] != "completed":
            raise ValueError(f"Incomplete predecessor: {stage}")
        receipts[to_stored(ROOT / path)] = sha256_file(ROOT / path)
        for key, digest in (receipt["input_sha256"] | receipt["artifacts"]).items():
            if key in expected and expected[key] != digest:
                raise ValueError(f"Conflicting predecessor digest: {key}")
            expected[key] = digest
    needed = [COHORT, SPH, CACHE / "qualify/labels.csv", CACHE / "evaluate/evaluation.json",
              CACHE / "evaluate/sph_predictions.npz"]
    for stage in ("extract_training", "extract_evaluation"):
        needed += [CACHE / stage / name for name in ("rows.csv", "pooled.npy", "tokens.npy")]
    verified = dict(receipts)
    for path in needed:
        key = to_stored(ROOT / path)
        actual = sha256_file(from_stored(key))
        if actual != expected[key]:
            raise ValueError(f"Changed predecessor input: {key}")
        verified[key] = actual
        print(f"Verified {key}", flush=True)
    return verified


def predecessor_replay() -> dict[str, Any]:
    """
    Reproduce every HEADS-01 saved SPH point metric exactly without refitting.

    Returns
    -------
    dict[str, Any]
        Model and metric field counts for the exact arithmetic replay.
    """
    saved = json_read(ROOT / CACHE / "evaluate/evaluation.json")
    scores = np.load(ROOT / CACHE / "evaluate/sph_predictions.npz")
    held, y = scores["evaluation"].astype(bool), scores["target"].astype(int)
    checked = {}
    for model in saved["models"]:
        point = next(x for x in model["evaluations"] if x["pool"] == "sph")
        actual = point_metrics(y[held], scores[model["id"]][held], point["threshold"])
        if actual != point["metrics"]:
            raise ValueError(f"Predecessor point mismatch: {model['id']}")
        checked[model["id"]] = len(actual)
    return {"exact": True, "models": len(checked), "metric_fields": sum(checked.values())}


def positions(ids: pd.Series, requested: pd.Series) -> np.ndarray:
    """
    Align unique record identifiers without silently accepting absent records.

    Parameters
    ----------
    ids : pd.Series
        Cache record identifiers.
    requested : pd.Series
        Desired cohort order.

    Returns
    -------
    np.ndarray
        Cache index positions of the requested records.
    """
    if ids.duplicated().any() or requested.duplicated().any():
        raise ValueError("Duplicate record identifiers")
    found = pd.Index(ids).get_indexer(requested)
    if np.any(found < 0):
        raise ValueError("A required record is missing from the cache")
    return found


def cached_arrays(part: str, expected: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Open cached arrays and validate the frozen shape and storage precision.

    Parameters
    ----------
    part : str
        Training or evaluation cache name.
    expected : int
        Frozen number of rows in the cache.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Memory-mapped pooled and token arrays.
    """
    pooled = np.load(ROOT / CACHE / f"extract_{part}/pooled.npy", mmap_mode="r")
    tokens = np.load(ROOT / CACHE / f"extract_{part}/tokens.npy", mmap_mode="r")
    if pooled.shape != (expected, 1024) or tokens.shape != (expected, 40, 1024):
        raise ValueError("Unexpected predecessor feature shapes")
    if pooled.dtype != np.float32 or tokens.dtype != np.float16:
        raise ValueError("Unexpected predecessor feature dtypes")
    return pooled, tokens


def load_inputs() -> dict[str, Any]:
    """
    Align the frozen 25k training cohort and SPH evaluation rows with token caches.

    Returns
    -------
    dict[str, Any]
        Metadata, flags, pooled vectors, memory-mapped tokens and cache positions.
    """
    train = pd.read_csv(ROOT / COHORT, dtype=str, keep_default_na=False)
    query = pd.read_csv(ROOT / SPH, dtype=str, keep_default_na=False)
    query = query.loc[query["evaluation"].eq("True")].reset_index(drop=True)
    if len(train) != 25000 or len(query) != 10434:
        raise ValueError("Frozen cohort counts changed")
    if not train["split"].eq("train").all() or train["source"].eq("sph").any():
        raise ValueError("Ineligible training source or role")
    if not query["source"].eq("sph").all() or query["fit"].eq("True").any():
        raise ValueError("Ineligible query source or role")
    labels = pd.read_csv(ROOT / CACHE / "qualify/labels.csv")
    flags = labels.set_index("record_id").loc[train["record_id"], list(GROUPS)].to_numpy(dtype=int)
    qflags = query[list(GROUPS)].to_numpy(dtype=int)
    if not np.array_equal(labels.set_index("record_id").loc[train["record_id"], "target"],
                          train["target"].to_numpy(dtype=int)):
        raise ValueError("Training label identities do not match")
    result: dict[str, Any] = {"train": train, "query": query}
    for role, frame, memberships, part, expected in (
        ("train", train, flags, "training", 28568), ("query", query, qflags, "evaluation", 21677)
    ):
        if not np.isin(memberships, (0, 1)).all():
            raise ValueError("Nonbinary memberships")
        if not np.array_equal(memberships.max(axis=1), frame["target"].to_numpy(dtype=int)):
            raise ValueError("Superclass union no longer equals the binary target")
        rows = pd.read_csv(ROOT / CACHE / f"extract_{part}/rows.csv")
        index = positions(rows["record_id"], frame["record_id"])
        pooled, tokens = cached_arrays(part, expected)
        result.update({f"{role}_flags": memberships, f"{role}_positions": index,
                       f"{role}_pooled": np.asarray(pooled[index]), f"{role}_tokens": tokens})
    known_train = set(train.loc[train["patient_identity_known"].eq("True"), "patient_id"])
    if set(query["patient_key"]) & known_train or set(train["record_id"]) & set(query["record_id"]):
        raise ValueError("Train/query identity overlap")
    return result


def pooling_check(tokens: np.ndarray, index: np.ndarray, pooled: np.ndarray) -> dict[str, Any]:
    """
    Qualify finite used rows and cached-token arithmetic mean against saved pooling.

    Parameters
    ----------
    tokens : np.ndarray
        Memory-mapped raw tokens.
    index : np.ndarray
        Used cache positions only.
    pooled : np.ndarray
        Aligned saved pooled features.

    Returns
    -------
    dict[str, Any]
        Maximum relative L2 error and the number of qualified rows.
    """
    worst = 0.0
    for start in range(0, len(index), BATCH):
        part = tokens[index[start:start + BATCH]].astype(np.float32)
        reference = pooled[start:start + BATCH]
        if not np.isfinite(part).all() or not np.isfinite(reference).all():
            raise ValueError("Nonfinite cached features")
        error = np.linalg.norm(part.mean(axis=1) - reference, axis=1)
        error /= np.maximum(np.linalg.norm(reference, axis=1), 1e-8)
        worst = max(worst, float(error.max()))
    if worst > 0.001:
        raise ValueError(f"Unresolved pooling/padding semantics: {worst}")
    return {"rows": len(index), "max_relative_l2_error": worst, "tolerance": 0.001}
