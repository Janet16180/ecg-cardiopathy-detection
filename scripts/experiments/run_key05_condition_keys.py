"""
KEY-05: condition keys on pooled xECG embeddings for the 14 conditions of dataset v5.

``docs/key05-condition-keys-protocol-2026-11-11.md`` is the contract. Steps, in order (outputs under
``outputs/key05_condition_keys_20261111/``, one receipt per step under ``receipts/``):

- ``integrity``: feature rows against the v5 manifest, P0's stored v5 validation metric from its stored
  predictions, P0's weights against those predictions, train and validation label counts against the v5 build;
- ``keys``: K1 to K7 and N1 fitted on train, every construction and P0 scored on validation;
- ``evaluate``: construction table, per-condition and per-source tables, the ladder, cosines with P0;
- ``pairs``, ``sources``, ``references``, ``stability``, ``measurements``: analyses 3 to 7;
- ``review``: review sheets of a construction (analysis 8);
- ``select``: writes the selection record ``docs/key05-selection-record-2026-11-11.md``, to be committed;
- ``test``: the one test read; refuses to run until the selection record is committed.

Run from the worktree: ``PYTHONPATH=<worktree> OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
OPENBLAS_CORETYPE=Haswell .venv/bin/python -m scripts.experiments.run_key05_condition_keys <step>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import key05_analysis as ka
from ecg_experiment import key05_data as kd
from ecg_experiment import key05_keys as kk
from ecg_experiment import key05_review as kr
from ecg_experiment.files import read_json, sha256_file, write_json_atomic, write_text_atomic
from ecg_experiment.mlc02_bootstrap import (
    DRAWS,
    SEED,
    bootstrap_metrics,
    cluster_resamples,
    paired_comparison,
)
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_data import CONDITIONS, load_predictions
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_feature_models import fit_scaler, standardize
from ecg_experiment.mlc02_feature_runs import environment
from ecg_experiment.paths import to_stored

OUT = kd.OUTPUT
CODE = (
    "ecg_experiment/key05_keys.py",
    "ecg_experiment/key05_data.py",
    "ecg_experiment/key05_analysis.py",
    "ecg_experiment/key05_review.py",
    "scripts/experiments/run_key05_condition_keys.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/mlc02_bootstrap.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/mlc03_effect.py",
    "ecg_experiment/mlc03_label_quality.py",
    "ecg_experiment/intervals.py",
)
EDAC = Path("/sys/devices/system/edac/mc/mc0/ue_count")
PREDICT_TOLERANCE = 1e-12
K7_IDS = tuple(f"K7-m{m}" for m in kk.CLUSTER_COUNTS)
MODELS = (*kk.SINGLE_KEYS, *K7_IDS, "N1", "P0")
EQUIVALENCE_MARGIN = -0.010
TEST_MINIMUM = 100
ROLE_NAMES = {"test_same_source": "PTB-XL fold 10", "test_external": "SPH evaluation half"}


def log(message: str) -> None:
    """
    Print a time-stamped progress line.

    Parameters
    ----------
    message : str
        Text.
    """
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def plain(value: object) -> object:
    """
    Convert NumPy values inside a nested value to Python ones, for JSON.

    Parameters
    ----------
    value : object
        Nested value.

    Returns
    -------
    object
        JSON-serializable structure.
    """
    return json.loads(
        json.dumps(value, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item))
    )


def committed() -> dict[str, Any]:
    """
    Commit of the run, which must start from committed code.

    Returns
    -------
    dict[str, Any]
        ``git_state``.

    Raises
    ------
    RuntimeError
        If a tracked file is modified.
    """
    state = git_state()
    if state["modified_tracked"]:
        raise RuntimeError(f"commit the code before running: {state['modified_tracked']}")
    return state


def receipt(
    step: str, started: float, git: dict[str, Any], outputs: list[Path], extra: dict[str, Any]
) -> None:
    """
    Write the receipt of one step.

    Parameters
    ----------
    step : str
        Step name.
    started : float
        ``time.time()`` at the start.
    git : dict[str, Any]
        ``git_state`` at the start.
    outputs : list[Path]
        Files the step wrote.
    extra : dict[str, Any]
        Step-specific content.
    """
    record = {
        "step": step,
        "command": [
            to_stored(sys.argv[0]) if Path(sys.argv[0]).is_absolute() else sys.argv[0],
            *sys.argv[1:],
        ],
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "git": git,
        "code_sha256": code_sha256(CODE),
        "inputs": {
            "v5_manifest": to_stored(kd.MANIFEST),
            "v5_manifest_sha256": sha256_file(kd.MANIFEST),
            "features": to_stored(kd.FEATURES),
            "p0_model_json_sha256": sha256_file(kd.P0_DIRECTORY / "P0" / "model.json"),
            "p0_validation_sha256": sha256_file(kd.P0_VALIDATION),
        },
        "outputs_sha256": {to_stored(path): sha256_file(path) for path in outputs},
        "environment": environment(),
        "edac_uncorrected": EDAC.read_text().strip() if EDAC.exists() else None,
        "test_roles_read": step == "test",
        **extra,
    }
    write_json_atomic(OUT / "receipts" / f"{step}.json", plain(record), allow_nan=True)


def open_data() -> dict[str, Any]:
    """
    Train and validation records, raw and standardized features.

    Returns
    -------
    dict[str, Any]
        ``train`` and ``validation`` (``load_records`` arrays), ``raw_train``, ``raw_validation``, ``scaler``,
        ``x_train`` and ``x_validation``.
    """
    data = kd.load_records(kd.OPEN_ROLES)
    raw = kd.load_features(data)
    train, validation = data["role"] == "train", data["role"] == "validation"
    scaler = fit_scaler(raw[train])
    return {
        "train": kd.subset(data, train),
        "validation": kd.subset(data, validation),
        "raw_train": raw[train],
        "raw_validation": raw[validation],
        "scaler": scaler,
        "x_train": standardize(raw[train], scaler),
        "x_validation": standardize(raw[validation], scaler),
    }


def save_table(frame: pd.DataFrame, name: str, written: list[Path]) -> Path:
    """
    Write a CSV under the output directory and remember it.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.
    name : str
        File name.
    written : list[Path]
        List the path is appended to.

    Returns
    -------
    Path
        The file.
    """
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    written.append(path)
    return path


def integrity() -> None:
    """Check the inputs before any key is built (step ``integrity``)."""
    started, git = time.time(), committed()
    v5_receipt = read_json(kd.V5 / "receipt.json")
    checks: dict[str, Any] = {"features": kd.feature_receipt()}
    checks["manifest_sha256_equal"] = sha256_file(kd.MANIFEST) == v5_receipt["outputs_sha256"]["manifest.csv"]
    data = kd.load_records(kd.OPEN_ROLES)
    index = kd.feature_rows(data)
    checks["feature_rows"] = {
        "records": len(index),
        "aligned": True,
        "by_role": {role: int((data["role"] == role).sum()) for role in kd.OPEN_ROLES},
    }
    checks["support_train_validation"] = kd.check_support(data)
    validation = kd.subset(data, data["role"] == "validation")
    stored = load_predictions(kd.P0_VALIDATION, validation["record_key"])["probability"]
    found = selection_metric(validation["labels"], stored, validation["source"])["selection_metric"]
    checks["p0_stored_selection_metric"] = {
        "found": found,
        "expected": kd.P0_STORED_METRIC,
        "equal": found == kd.P0_STORED_METRIC,
    }
    raw = kd.load_features(data)
    model = kd.p0_model()
    difference = np.abs(kd.p0_probability(model, raw[data["role"] == "validation"]) - stored).max()
    checks["p0_weights_reproduce_predictions"] = {
        "max_abs_difference": float(difference),
        "tolerance": PREDICT_TOLERANCE,
        "within": bool(difference <= PREDICT_TOLERANCE),
    }
    scaler = fit_scaler(raw[data["role"] == "train"])
    stored_scaler = model["artefacts"]["scaler.npz"]
    checks["scaler_equals_p0"] = {
        name: float(np.abs(scaler[name] - stored_scaler[name]).max()) for name in ("mean", "scale")
    }
    failed = [
        name
        for name, value in (
            ("manifest", checks["manifest_sha256_equal"]),
            ("support", checks["support_train_validation"]["equal"]),
            ("p0 metric", checks["p0_stored_selection_metric"]["equal"]),
            ("p0 weights", checks["p0_weights_reproduce_predictions"]["within"]),
        )
        if not value
    ]
    path = OUT / "integrity.json"
    write_json_atomic(path, plain(checks))
    receipt("integrity", started, git, [path], {"failed": failed})
    if failed:
        raise ValueError(f"integrity checks failed: {failed}")
    log(f"integrity passed: {json.dumps(plain(checks))[:400]}")


def build_k7(x: np.ndarray, labels: np.ndarray, fits: list[dict[str, Any]]) -> dict[str, Any]:
    """
    K7 keys for every cluster count and condition.

    Parameters
    ----------
    x : np.ndarray
        Standardized train features.
    labels : np.ndarray
        Train states.
    fits : list[dict[str, Any]]
        K4 covariance and negative mean per condition.

    Returns
    -------
    dict[str, Any]
        Per K7 id: ``keys`` ``[14, m, 1024]``, ``sizes`` ``[14, m]`` and ``inertia`` ``[14]``.
    """
    out = {}
    for clusters, name in zip(kk.CLUSTER_COUNTS, K7_IDS, strict=True):
        parts = [
            kk.key_k7(x, labels[:, c], clusters, fits[c]["covariance"], fits[c]["negative_mean"])
            for c in range(labels.shape[1])
        ]
        out[name] = {
            "keys": np.stack([p["keys"] for p in parts]),
            "sizes": np.stack([p["sizes"] for p in parts]),
            "inertia": np.array([p["inertia"] for p in parts]),
        }
    return out


def score_all(
    keys: dict[str, Any],
    x: np.ndarray,
    sources: np.ndarray,
    means: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """
    Scores of single keys and K7 for some records.

    Parameters
    ----------
    keys : dict[str, Any]
        Construction to keys (``[14, 1024]`` or ``[14, m, 1024]`` for K7).
    x : np.ndarray
        Standardized features.
    sources : np.ndarray
        Source of each record.
    means : dict[str, np.ndarray]
        Source means for K6.
    names : tuple[str, ...]
        Constructions to score.

    Returns
    -------
    dict[str, np.ndarray]
        Construction to ``[records, 14]``.
    """
    out = {}
    for name in names:
        if name.startswith("K7"):
            out[name] = np.column_stack([kk.max_score(x, keys[name][c]) for c in range(len(CONDITIONS))])
        else:
            out[name] = kk.score_single(name, x, keys[name], sources, means)
    return out


def load_keys() -> dict[str, Any]:
    """
    Keys, source means and the standardizer written by ``keys``.

    Returns
    -------
    dict[str, Any]
        ``keys`` (construction to array), ``means`` (source to mean) and ``scaler``.
    """
    with np.load(OUT / "keys.npz", allow_pickle=False) as archive:
        content = {name: archive[name] for name in archive.files}
    keys = {name: content[name] for name in (*kk.SINGLE_KEYS, *K7_IDS)}
    means = {
        name.removeprefix("source_mean__"): value
        for name, value in content.items()
        if name.startswith("source_mean__")
    }
    return {
        "keys": keys,
        "means": means,
        "scaler": {"mean": content["scaler_mean"], "scale": content["scaler_scale"]},
        "content": content,
    }


def load_scores(validation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Read the validation scores written by ``keys``, checked against the record keys.

    Parameters
    ----------
    validation : dict[str, np.ndarray]
        Validation records.

    Returns
    -------
    dict[str, np.ndarray]
        Model to ``[records, 14]``.

    Raises
    ------
    ValueError
        If the stored keys differ.
    """
    with np.load(OUT / "scores_validation.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["record_key"], validation["record_key"]):
            raise ValueError("stored validation scores are not aligned with the records")
        return {name: archive[name] for name in MODELS}


def keys_step() -> None:
    """Fit every construction on train and score validation (step ``keys``)."""
    started, git = time.time(), committed()
    d = open_data()
    train, validation = d["train"], d["validation"]
    log("single keys")
    single = kk.single_keys(d["x_train"], train["labels"], train["source"])
    log("K7")
    k7 = build_k7(d["x_train"], train["labels"], single["k4_fits"])
    keys = {**single["keys"], **{name: part["keys"] for name, part in k7.items()}}
    log("scores")
    scores = score_all(
        keys, d["x_validation"], validation["source"], single["source_means"], (*kk.SINGLE_KEYS, *K7_IDS)
    )
    log("N1")
    scores["N1"] = kk.neighbour_share(d["x_validation"], d["x_train"], train["labels"])
    scores["P0"] = load_predictions(kd.P0_VALIDATION, validation["record_key"])["probability"]
    arrays = {
        **keys,
        **{f"sizes__{name}": part["sizes"] for name, part in k7.items()},
        **{f"inertia__{name}": part["inertia"] for name, part in k7.items()},
        **{f"shrinkage__{name}": values for name, values in single["shrinkage"].items()},
        **{f"source_mean__{name}": value for name, value in single["source_means"].items()},
        "scaler_mean": d["scaler"]["mean"],
        "scaler_scale": d["scaler"]["scale"],
        "k5_sources": np.array([",".join(names) for names in single["k5_sources"]]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "keys.npz", **arrays)
    np.savez(OUT / "scores_validation.npz", record_key=validation["record_key"], **scores)
    summary = {
        "train_records": len(train["record_key"]),
        "validation_records": len(validation["record_key"]),
        "shrinkage": single["shrinkage"],
        "k5_sources": dict(zip(CONDITIONS, single["k5_sources"], strict=True)),
        "k7_sizes": {name: part["sizes"] for name, part in k7.items()},
    }
    write_json_atomic(OUT / "keys_summary.json", plain(summary))
    receipt(
        "keys", started, git, [OUT / "keys.npz", OUT / "scores_validation.npz", OUT / "keys_summary.json"], {}
    )
    log("keys done")


def p0_cosines(keys: dict[str, np.ndarray], coef: np.ndarray) -> pd.DataFrame:
    """
    Cosine between each key and P0's weight vector of the same condition.

    Parameters
    ----------
    keys : dict[str, np.ndarray]
        Construction to keys.
    coef : np.ndarray
        ``[14, 1024]`` P0 coefficients.

    Returns
    -------
    pd.DataFrame
        ``construction``, ``condition``, ``cosine`` (K7: the mean over centres) and ``cosine_max``.
    """
    rows = []
    for name, values in keys.items():
        for column, condition in enumerate(CONDITIONS):
            parts = values[column][None, :] if values.ndim == 2 else values[column]
            found = [kk.cosine(part, coef[column]) for part in parts]
            rows.append(
                {
                    "construction": name,
                    "condition": condition,
                    "cosine": float(np.mean(found)),
                    "cosine_max": float(np.max(found)),
                }
            )
    return pd.DataFrame(rows)


def evaluate() -> None:
    """Write the construction table, per-condition tables and the ladder (step ``evaluate``)."""
    started, git = time.time(), committed()
    data = kd.load_records(kd.OPEN_ROLES)
    validation = kd.subset(data, data["role"] == "validation")
    scores = load_scores(validation)
    log("bootstrap")
    draws = ka.bootstrap_draws(validation["labels"], validation["group"], validation["source"], scores)
    written: list[Path] = []
    save_table(ka.construction_table(draws), "constructions.csv", written)
    save_table(ka.condition_table(draws), "conditions.csv", written)
    save_table(
        ka.per_source_table(validation["labels"], scores, validation["source"]), "per_source.csv", written
    )
    ladders = [
        ka.ladder_table(draws, ("K1", "K2", "K4", "K6", name, "P0")).assign(k7=name) for name in K7_IDS
    ]
    save_table(pd.concat(ladders), "ladder.csv", written)
    keys = load_keys()["keys"]
    coef = kd.p0_model()["artefacts"]["logistic.npz"]["coef"]
    save_table(p0_cosines(keys, coef), "cosine_p0.csv", written)
    arrays = {
        f"{name}__{part}": model[part]
        for name, model in draws["models"].items()
        for part in ("selection_draws", "condition_draws", "pooled_draws", "pooled_condition_draws")
    }
    np.savez(OUT / "draws_validation.npz", **arrays)
    written.append(OUT / "draws_validation.npz")
    write_json_atomic(OUT / "resamples_validation.json", plain(draws["resamples"]))
    written.append(OUT / "resamples_validation.json")
    receipt("evaluate", started, git, written, {"draws": DRAWS, "seed": SEED})
    log("evaluate done")


def projected_scores(
    name: str, keys: np.ndarray, x: np.ndarray, sources: np.ndarray, means: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Scores of a single-key construction after projecting the other 13 keys out of each key.

    Parameters
    ----------
    name : str
        K1 to K6.
    keys : np.ndarray
        ``[14, 1024]``.
    x : np.ndarray
        Standardized features.
    sources : np.ndarray
        Source of each record.
    means : dict[str, np.ndarray]
        Source means (K6).

    Returns
    -------
    np.ndarray
        ``[records, 14]``.
    """
    return kk.score_single(name, x, ka.projected_keys(keys), sources, means)


def pairs() -> None:
    """Score co-occurring and related conditions, before and after projection (step ``pairs``)."""
    started, git = time.time(), committed()
    d = open_data()
    validation = d["validation"]
    stored = load_keys()
    scores = load_scores(validation)
    rows, own = [], []
    for name in kk.SINGLE_KEYS:
        projected = projected_scores(
            name, stored["keys"][name], d["x_validation"], validation["source"], stored["means"]
        )
        rows += [
            {**row, "projected": False}
            for row in ka.pair_rows(validation["labels"], validation["source"], scores[name], name)
        ]
        rows += [
            {**row, "projected": True}
            for row in ka.pair_rows(validation["labels"], validation["source"], projected, name)
        ]
        for label, values in (("raw", scores[name]), ("projected", projected)):
            metrics = ka.condition_metrics(validation["labels"], values, validation["source"])
            own += [
                {
                    "model": name,
                    "keys": label,
                    "condition": c,
                    "within": metrics["within"][i],
                    "pooled": metrics["pooled"][i],
                }
                for i, c in enumerate(CONDITIONS)
            ]
    for name in (*K7_IDS, "N1", "P0"):
        rows += [
            {**row, "projected": False}
            for row in ka.pair_rows(validation["labels"], validation["source"], scores[name], name)
        ]
    written: list[Path] = []
    save_table(pd.DataFrame(rows), "pairs.csv", written)
    save_table(pd.DataFrame(own), "projected_own_auroc.csv", written)
    receipt("pairs", started, git, written, {"minimum": ka.PAIR_MINIMUM})
    log("pairs done")


def source_cosine_rows(keys: dict[str, np.ndarray], directions: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Cosine of every key (K7: every centre) with every source direction.

    Parameters
    ----------
    keys : dict[str, np.ndarray]
        Construction to keys.
    directions : dict[str, Any]
        Output of ``source_directions``.

    Returns
    -------
    list[dict[str, Any]]
        ``construction``, ``condition``, ``centre``, ``source`` and ``cosine``.
    """
    rows = []
    for name, values in keys.items():
        for column, condition in enumerate(CONDITIONS):
            parts = values[column][None, :] if values.ndim == 2 else values[column]
            for centre, part in enumerate(parts):
                for source, direction in zip(directions["classes"], directions["coef"], strict=True):
                    rows.append(
                        {
                            "construction": name,
                            "condition": condition,
                            "centre": centre,
                            "source": source,
                            "cosine": kk.cosine(part, direction),
                        }
                    )
    return rows


def sources() -> None:
    """Compare keys with the directions that separate the sources (step ``sources``)."""
    started, git = time.time(), committed()
    d = open_data()
    train, validation = d["train"], d["validation"]
    log("source classifier")
    directions = ka.source_directions(d["x_train"], train["source"])
    stored = load_keys()
    scores = load_scores(validation)
    rng = np.random.default_rng(ka.STABILITY_SEED)
    random = kk.unit_rows(rng.standard_normal((2000, d["x_train"].shape[1])))
    baseline = np.abs(random @ kk.unit_rows(directions["coef"]).T)
    share = [
        {
            "model": name,
            "condition": c,
            "between_source_share": ka.between_source_share(scores[name][:, i], validation["source"]),
        }
        for name in MODELS
        for i, c in enumerate(CONDITIONS)
    ]
    written: list[Path] = []
    save_table(pd.DataFrame(source_cosine_rows(stored["keys"], directions)), "source_cosines.csv", written)
    save_table(pd.DataFrame(share), "source_variance_share.csv", written)
    np.savez(OUT / "source_directions.npz", coef=directions["coef"], classes=np.array(directions["classes"]))
    written.append(OUT / "source_directions.npz")
    summary = {
        "classes": directions["classes"],
        "train_accuracy": float(directions["model"].score(d["x_train"], train["source"])),
        "validation_accuracy": float(directions["model"].score(d["x_validation"], validation["source"])),
        "random_abs_cosine": {
            "median": float(np.median(baseline)),
            "p95": float(np.percentile(baseline, 95)),
            "p99": float(np.percentile(baseline, 99)),
            "vectors": len(random),
        },
        "iterations": directions["model"].n_iter_,
    }
    write_json_atomic(OUT / "sources.json", plain(summary))
    written.append(OUT / "sources.json")
    receipt("sources", started, git, written, {})
    log("sources done")


def references() -> None:
    """Build keys from few reference records (step ``references``)."""
    started, git = time.time(), committed()
    d = open_data()
    train, validation = d["train"], d["validation"]
    total = kk.ledoit_wolf(d["x_train"] - d["x_train"].mean(axis=0), np.ones(len(d["x_train"])))
    log(f"total covariance shrinkage {total['shrinkage']:.4f}")
    curve = ka.reference_curve(
        d["x_train"],
        train["labels"],
        d["x_validation"],
        validation["labels"],
        validation["source"],
        total["covariance"],
    )
    written: list[Path] = []
    save_table(curve, "references.csv", written)
    receipt(
        "references",
        started,
        git,
        written,
        {
            "counts": ka.REFERENCE_COUNTS,
            "draws": ka.REFERENCE_DRAWS,
            "seed": ka.REFERENCE_SEED,
            "total_covariance_shrinkage": total["shrinkage"],
        },
    )
    log("references done")


def replicate_metrics(
    keys: dict[str, np.ndarray], d: dict[str, Any], means: dict[str, np.ndarray]
) -> list[dict]:
    """
    Score one replicate's single keys on validation.

    Parameters
    ----------
    keys : dict[str, np.ndarray]
        Construction to ``[14, 1024]``.
    d : dict[str, Any]
        Output of ``open_data``.
    means : dict[str, np.ndarray]
        Source means for K6.

    Returns
    -------
    list[dict]
        One row per construction and condition.
    """
    validation = d["validation"]
    rows = []
    for name, values in keys.items():
        score = kk.score_single(name, d["x_validation"], values, validation["source"], means)
        metrics = ka.condition_metrics(validation["labels"], score, validation["source"])
        rows += [
            {
                "construction": name,
                "condition": c,
                "within": metrics["within"][i],
                "pooled": metrics["pooled"][i],
            }
            for i, c in enumerate(CONDITIONS)
        ]
    return rows


def leave_family_out(
    d: dict[str, Any], full: dict[str, np.ndarray], means: dict[str, np.ndarray]
) -> list[dict]:
    """
    Keys built without one source family, scored on that family's validation records against the full keys.

    Parameters
    ----------
    d : dict[str, Any]
        Output of ``open_data``.
    full : dict[str, np.ndarray]
        Single keys built on all train records.
    means : dict[str, np.ndarray]
        Full-train source means; they centre the held-out family's records for K6 (a constant per source,
        which leaves within-source AUROC unchanged).

    Returns
    -------
    list[dict]
        One row per family, construction and condition entering a source of the family.
    """
    train, validation = d["train"], d["validation"]
    rows = []
    for family in kd.FAMILIES:
        log(f"leave out {family}")
        keep = train["family"] != family
        held = kk.single_keys(d["x_train"][keep], train["labels"][keep], train["source"][keep])
        inside = validation["family"] == family
        labels, groups, sources_f = (validation[name][inside] for name in ("labels", "group", "source"))
        for name in kk.SINGLE_KEYS:
            scores = {
                "without": kk.score_single(
                    name, d["x_validation"][inside], held["keys"][name], sources_f, means
                ),
                "with": kk.score_single(name, d["x_validation"][inside], full[name], sources_f, means),
            }
            draws = ka.bootstrap_draws(labels, groups, sources_f, scores)
            without, with_ = draws["models"]["without"], draws["models"]["with"]
            difference = without["condition_draws"] - with_["condition_draws"]
            for column, condition in enumerate(CONDITIONS):
                if not np.isfinite(without["condition"][column]):
                    continue
                low, high = ka.interval(difference[:, column])
                rows.append(
                    {
                        "family": family,
                        "construction": name,
                        "condition": condition,
                        "without": float(without["condition"][column]),
                        "with": float(with_["condition"][column]),
                        "difference": float(without["condition"][column] - with_["condition"][column]),
                        "ci_low": low,
                        "ci_high": high,
                        "train_positives_without": int((train["labels"][keep, column] == 1).sum()),
                    }
                )
    return rows


def stability() -> None:
    """Measure bootstrap and leave-one-family-out stability of the single keys (step ``stability``)."""
    started, git = time.time(), committed()
    d = open_data()
    train = d["train"]
    stored = load_keys()
    weights = ka.bootstrap_weights(train["group"])
    replicates = {
        name: np.empty((len(weights), len(CONDITIONS), d["x_train"].shape[1]), dtype=np.float32)
        for name in kk.SINGLE_KEYS
    }
    metric_rows = []
    for draw, w in enumerate(weights):
        built = kk.single_keys(d["x_train"], train["labels"], train["source"], w)
        for name in kk.SINGLE_KEYS:
            replicates[name][draw] = built["keys"][name]
        metric_rows += [
            {**row, "draw": draw} for row in replicate_metrics(built["keys"], d, built["source_means"])
        ]
        if draw % 20 == 0:
            log(f"bootstrap {draw}")
    cosine_rows = [
        {
            "construction": name,
            "condition": c,
            **ka.key_cosines(replicates[name][:, i].astype(np.float64), stored["keys"][name][i]),
        }
        for name in kk.SINGLE_KEYS
        for i, c in enumerate(CONDITIONS)
    ]
    written: list[Path] = []
    save_table(pd.DataFrame(cosine_rows), "stability_cosines.csv", written)
    save_table(pd.DataFrame(metric_rows), "stability_auroc.csv", written)
    full = {name: stored["keys"][name] for name in kk.SINGLE_KEYS}
    save_table(pd.DataFrame(leave_family_out(d, full, stored["means"])), "leave_family_out.csv", written)
    receipt("stability", started, git, written, {"draws": len(weights), "seed": ka.STABILITY_SEED})
    log("stability done")


def measurement_rows(
    scores: dict[str, np.ndarray], table: pd.DataFrame, sources: np.ndarray
) -> tuple[list, list]:
    """
    Spearman correlations and binned curves of every model against the defining measurements.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Model to validation scores (P0 as the logit of its probability).
    table : pd.DataFrame
        Output of ``measurement_table`` for the validation records.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    tuple[list, list]
        Correlation rows and curve rows.
    """
    correlations, curves = [], []
    for condition, measures in kr.MEASURES.items():
        column = CONDITIONS.index(condition)
        for measure in measures:
            values = table[measure].to_numpy(dtype=float)
            tags = {"condition": condition, "measure": measure, "unreliable": measure in kr.UNRELIABLE}
            for name, score in scores.items():
                correlations += [
                    {"model": name, **tags, **row}
                    for row in ka.spearman_rows(score[:, column], values, sources)
                ]
                curves += [
                    {"model": name, **tags, **row}
                    for row in ka.binned_curve(score[:, column], values, sources)
                ]
    return correlations, curves


def measurements() -> None:
    """Compare keys with the measurements that define their conditions (step ``measurements``)."""
    started, git = time.time(), committed()
    data = kd.load_records(kd.OPEN_ROLES)
    validation = kd.subset(data, data["role"] == "validation")
    scores = load_scores(validation)
    probability = np.clip(scores.pop("P0"), 1e-12, 1 - 1e-12)
    scores["P0"] = np.log(probability / (1 - probability))
    scores.pop("N1")
    table = kr.measurement_table(validation["record_key"])
    correlations, curves = measurement_rows(scores, table, validation["source"])
    written: list[Path] = []
    save_table(pd.DataFrame(correlations), "measurement_spearman.csv", written)
    save_table(pd.DataFrame(curves), "measurement_curves.csv", written)
    counts = {
        measure: int(table[measure].notna().sum())
        for measures in kr.MEASURES.values()
        for measure in measures
    }
    write_json_atomic(
        OUT / "measurements.json", plain({"validation_records": len(table), "measured": counts})
    )
    written.append(OUT / "measurements.json")
    receipt("measurements", started, git, written, {})
    log("measurements done")


def chosen(metric: str) -> dict[str, Any]:
    """
    Find the single key and the K7 with the highest validation macro AUROC under a metric.

    Parameters
    ----------
    metric : str
        ``selection`` (within-source macro averaged over sources) or ``pooled``.

    Returns
    -------
    dict[str, Any]
        ``metric``, ``selected_single``, ``selected_k7`` and the construction table rows.
    """
    table = pd.read_csv(OUT / "constructions.csv")
    single = table[table["model"].isin(kk.SINGLE_KEYS)].sort_values(metric, ascending=False, kind="stable")
    k7 = table[table["model"].isin(K7_IDS)].sort_values(metric, ascending=False, kind="stable")
    return {
        "metric": metric,
        "selected_single": str(single.iloc[0]["model"]),
        "selected_k7": str(k7.iloc[0]["model"]),
        "table": table,
    }


def review_titles(rows: pd.DataFrame) -> list[str]:
    """
    Panel captions of a review sheet.

    Parameters
    ----------
    rows : pd.DataFrame
        Listed records with ``record_key``, ``source``, ``label``, ``rank``, ``score`` and measurements.

    Returns
    -------
    list[str]
        Two lines per record.
    """
    return [
        f"{row['rank']}. {row['record_key']} ({row['source']}) label {int(row['label'])} "
        f"score {row['score']:.4g}\n{kr.caption(row)}"
        for _, row in rows.iterrows()
    ]


def review() -> None:
    """Plot review sheets of the construction selected under ``--metric`` (step ``review``)."""
    from ecg_experiment.mlc03_label_quality import cached_windows, plot_review

    started, git = time.time(), committed()
    construction = chosen(ARGS.metric)["selected_single"]
    data = kd.load_records(kd.OPEN_ROLES)
    validation = kd.subset(data, data["role"] == "validation")
    score = load_scores(validation)[construction]
    table = kr.measurement_table(validation["record_key"])
    listed, written = [], []
    for column, condition in enumerate(CONDITIONS):
        positions = kr.review_positions(validation["labels"][:, column], score[:, column])
        for kind, index in positions.items():
            rows = table.iloc[index].reset_index()
            rows.insert(1, "source", validation["source"][index])
            rows.insert(2, "label", validation["labels"][index, column])
            rows.insert(3, "rank", np.arange(1, len(index) + 1))
            rows.insert(4, "score", score[index, column])
            signals = cached_windows(validation["record_key"][index])
            path = OUT / "review" / f"{condition}__{kind}.png"
            kr.plot_sheet(
                signals,
                review_titles(rows),
                f"KEY-05 {construction} {condition}: {kind.replace('_', ' ')}"
                " (validation, non-clinical reading)",
                path,
            )
            written.append(path)
            for rank, (key, signal) in enumerate(zip(rows["record_key"], signals, strict=True), start=1):
                single = OUT / "review" / condition / kind / f"{rank:02d}__{key.replace(':', '_')}.png"
                plot_review(
                    signal,
                    f"{condition} {kind} {rank}: {key}",
                    [review_titles(rows.iloc[[rank - 1]])[0]],
                    single,
                )
            listed.append(rows.assign(condition=condition, kind=kind))
    save_table(pd.concat(listed), "review/records.csv", written)
    receipt("review", started, git, written, {"construction": construction, "metric": ARGS.metric})
    log("review done")


def selection_markdown(choice: dict[str, Any], git: dict[str, Any]) -> str:
    """
    Text of the selection record.

    Parameters
    ----------
    choice : dict[str, Any]
        Output of ``chosen``.
    git : dict[str, Any]
        Commit of the validation outputs.

    Returns
    -------
    str
        Markdown with one JSON block.
    """
    table = choice["table"]
    lines = [
        "| construction | selection metric (95% interval) | minus P0 | pooled macro AUROC (95% interval) "
        "| minus P0 |",
        "|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.selection:.4f} ({row.selection_ci_low:.4f}, "
            f"{row.selection_ci_high:.4f}) | "
            f"{row.selection_minus_P0:+.4f} | {row.pooled:.4f} ({row.pooled_ci_low:.4f}, "
            f"{row.pooled_ci_high:.4f}) | {row.pooled_minus_P0:+.4f} |"
        )
    block = {
        "selected_single": choice["selected_single"],
        "selected_k7": choice["selected_k7"],
        "metric": choice["metric"],
        "validation_outputs_commit": git["commit"],
        "constructions_csv_sha256": sha256_file(OUT / "constructions.csv"),
    }
    return "\n".join(
        [
            "# KEY-05 selection record",
            "",
            "11 November 2026. Written by `run_key05_condition_keys select` after every validation",
            "analysis of [KEY-05](key05-condition-keys-protocol-2026-11-11.md) and before any test-role",
            "label was read.",
            "",
            f"- Selected single key (K1 to K6), highest validation {metric_name(choice['metric'])}: "
            f"**{choice['selected_single']}**.",
            f"- Selected K7: **{choice['selected_k7']}**.",
            "",
            "Validation, v5 labels, 2,000 patient-cluster resamples (`outputs/key05_condition_keys_20261111/"
            "constructions.csv`):",
            "",
            *lines,
            "",
            "The selection metric is the within-source macro AUROC averaged over the six validation sources;",
            "the pooled macro AUROC takes all validation records together. N1 and P0 are references, not",
            "candidates.",
            "",
            "```json",
            json.dumps(block, indent=2),
            "```",
            "",
        ]
    )


def metric_name(metric: str) -> str:
    """
    Words for a metric id.

    Parameters
    ----------
    metric : str
        ``selection`` or ``pooled``.

    Returns
    -------
    str
        Description.
    """
    return (
        "selection metric (within-source macro AUROC averaged over sources)"
        if metric == "selection"
        else "pooled macro AUROC"
    )


def select() -> None:
    """Write the selection record (step ``select``)."""
    started, git = time.time(), committed()
    choice = chosen(ARGS.metric)
    text = selection_markdown(choice, git)
    write_text_atomic(kd.SELECTION, text)
    path = OUT / "selection.json"
    write_json_atomic(path, plain({k: v for k, v in choice.items() if k != "table"}))
    receipt("select", started, git, [path, kd.SELECTION], {"metric": ARGS.metric})
    log(f"selected {choice['selected_single']} and {choice['selected_k7']}")


def test_scores(
    record: dict[str, Any], role_data: dict[str, np.ndarray], raw: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Scores of the selected keys and P0 on one test role.

    Parameters
    ----------
    record : dict[str, Any]
        Committed selection.
    role_data : dict[str, np.ndarray]
        The role's records.
    raw : np.ndarray
        Their raw embeddings.

    Returns
    -------
    dict[str, np.ndarray]
        ``single``, ``k7`` and ``P0`` scores.
    """
    stored = load_keys()
    x = standardize(raw, stored["scaler"])
    means = dict(stored["means"])
    for source in np.unique(role_data["source"]):
        means.setdefault(str(source), x[role_data["source"] == source].mean(axis=0))
    names = (record["selected_single"], record["selected_k7"])
    scores = score_all(stored["keys"], x, role_data["source"], means, names)
    return {"single": scores[names[0]], "k7": scores[names[1]], "P0": kd.p0_probability(kd.p0_model(), raw)}


def decision(low: float, high: float) -> str:
    """
    Read an interval by the protocol's decision rule.

    Parameters
    ----------
    low, high : float
        95% interval of the single key minus P0.

    Returns
    -------
    str
        ``equivalent``, ``clearly worse`` or ``undecided``.
    """
    if low > EQUIVALENCE_MARGIN:
        return "equivalent"
    if high < EQUIVALENCE_MARGIN:
        return "clearly worse"
    return "undecided"


def role_read(
    role: str, record: dict[str, Any], data: dict[str, np.ndarray], raw: np.ndarray
) -> tuple[dict, list]:
    """
    Make the test read of one role.

    Parameters
    ----------
    role : str
        ``test_same_source`` or ``test_external``.
    record : dict[str, Any]
        Committed selection.
    data : dict[str, np.ndarray]
        Records of both test roles.
    raw : np.ndarray
        Their raw embeddings.

    Returns
    -------
    tuple[dict, list]
        Macro summary and per-condition rows.
    """
    inside = data["role"] == role
    part = kd.subset(data, inside)
    scores = test_scores(record, part, raw[inside])
    resamples = cluster_resamples(part["group"], part["labels"], DRAWS, SEED, minimum=TEST_MINIMUM)
    boot = {
        name: bootstrap_metrics(
            part["labels"],
            ka.rank_unit(values) if name != "P0" else values,
            resamples,
            None,
            calibration=False,
        )
        for name, values in scores.items()
    }
    summary = {
        "role": role,
        "name": ROLE_NAMES[role],
        "records": int(inside.sum()),
        "groups": resamples["groups"],
        "redraws": resamples["redraws"],
        "entering": [c for c, e in zip(CONDITIONS, resamples["entering"], strict=True) if e],
        "macro": {
            name: {"value": b["observed_macro_auroc"], "ci": ka.interval(b["macro_auroc"])}
            for name, b in boot.items()
        },
    }
    rows = []
    for name in ("single", "k7"):
        comparison = paired_comparison(boot[name], boot["P0"])
        summary[f"{name}_minus_P0"] = comparison["macro"]
        for row in comparison["conditions"]:
            column = CONDITIONS.index(row["condition"])
            rows.append(
                {
                    "role": role,
                    "model": record["selected_single"] if name == "single" else record["selected_k7"],
                    **row,
                    "auroc": float(boot[name]["observed_auroc"][column]),
                    "auroc_P0": float(boot["P0"]["observed_auroc"][column]),
                    "positives": int((part["labels"][:, column] == 1).sum()),
                    "negatives": int((part["labels"][:, column] == 0).sum()),
                    "reading": decision(row["ci_low"], row["ci_high"]),
                }
            )
    return summary, rows


def test() -> None:
    """Make the one test read (step ``test``)."""
    started, git = time.time(), committed()
    if (OUT / "test.json").exists():
        raise FileExistsError("the test read has already been made")
    record = kd.selection_committed()
    data = kd.load_records(kd.TEST_ROLES, allow_test=True)
    support = kd.check_support(data)
    if not support["equal"]:
        raise ValueError("test-role label counts differ from the v5 build")
    raw = kd.load_features(data)
    summaries, rows = [], []
    for role in kd.TEST_ROLES:
        log(f"test read {role}")
        summary, part = role_read(role, record, data, raw)
        summaries.append(summary)
        rows += part
    primary = next(s for s in summaries if s["role"] == "test_external")["single_minus_P0"]
    result = {
        "selection": record,
        "support_test_roles": support,
        "roles": summaries,
        "primary": {
            **primary,
            "decision": decision(primary["ci_low"], primary["ci_high"]),
            "margin": EQUIVALENCE_MARGIN,
        },
    }
    written: list[Path] = []
    save_table(pd.DataFrame(rows), "test_conditions.csv", written)
    write_json_atomic(OUT / "test.json", plain(result))
    written.append(OUT / "test.json")
    receipt("test", started, git, written, {"selection_commit": record["commit"], "minimum": TEST_MINIMUM})
    log(f"primary {json.dumps(plain(result['primary']))}")


STEPS = {
    "integrity": integrity,
    "keys": keys_step,
    "evaluate": evaluate,
    "pairs": pairs,
    "sources": sources,
    "references": references,
    "stability": stability,
    "measurements": measurements,
    "review": review,
    "select": select,
    "test": test,
}
ARGS = argparse.Namespace(metric="selection")


def main() -> None:
    """Parse the step and run it."""
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=sorted(STEPS))
    parser.add_argument("--metric", choices=("selection", "pooled"), default="selection")
    ARGS = parser.parse_args()
    STEPS[ARGS.step]()


if __name__ == "__main__":
    main()
