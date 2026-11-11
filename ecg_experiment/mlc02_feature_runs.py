"""
Stage 2 runs of the MLC-02 feature-based candidates, on train and validation records only.

Every run fits on role ``train`` and scores on role ``validation``; the test roles are refused by
``mlc02_feature_data.load_split``. A run needs its code committed and no modified tracked file, and writes a
receipt with the input hashes, the code SHA-256s, the commit and the environment.

Outputs, under ``outputs/mlc02_condition_models_20261110/``:

- ``candidates/<id>/``: fitted artefacts and ``model.json`` (read by ``mlc02_feature_models.predict``), the
  validation prediction file(s) of ``mlc02_data.prediction_path``, ``candidate.json`` (chosen settings, grid,
  validation selection metric), ``validation_by_source.csv``, ``training_log.csv`` and ``receipt.json``.
- ``lofo/<id>/``: leave-one-family-out over train and validation records, at the main run's settings.
- ``source_identification/``: source-family identification from each feature set on validation records.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import platform
import subprocess
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from ecg_experiment import ROOT
from ecg_experiment.files import read_json, sha256_file, write_csv_atomic, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_data import (
    CONDITIONS,
    MANIFEST,
    VALIDATION_SOURCES,
    prediction_path,
    save_predictions,
)
from ecg_experiment.mlc02_evaluation import metrics_table, selection_metric, source_identification
from ecg_experiment.mlc02_feature_data import (
    FAMILIES,
    FEATURE_FILES,
    OPEN_ROLES,
    feature_hashes,
    fit_imputation,
    impute,
    load_split,
)
from ecg_experiment.mlc02_feature_models import (
    BOOSTING_LEAVES,
    BOOSTING_MAX_ITER,
    BOOSTING_RATES,
    BOOSTING_SEED,
    CANDIDATES,
    EPOCHS,
    INPUTS,
    KINDS,
    LOGISTIC_C,
    LOGISTIC_SEED,
    NETWORK_DECAYS,
    NETWORK_RATES,
    SEEDS,
    choose_per_condition,
    fit_boosting,
    fit_logistic,
    fit_scaler,
    logistic_inputs,
    network_inputs,
    network_probability,
    predict,
    save_boosting,
    save_logistic,
    save_network,
    staged_condition_scores,
    standardize,
    train_network,
)
from ecg_experiment.paths import to_stored

OUTPUT = ROOT / "outputs/mlc02_condition_models_20261110"
LOFO = OUTPUT / "lofo"
SOURCE_IDENTIFICATION = OUTPUT / "source_identification"
GPU_LOCK = Path("/tmp/ecg_project_gpu1.lock")
LOFO_CANDIDATES = ("P0", "L-C", "H-All")
MIN_FREE_GPU_BYTES = 2 * 1024**3
CODE = (
    "ecg_experiment/mlc02_feature_data.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/mlc02_feature_runs.py",
    "scripts/experiments/run_mlc02_feature_candidates.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/key_classifier_models.py",
    "ecg_experiment/condition_keys.py",
    "ecg_experiment/files.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/mlc02_cache_build.py",
)
PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "pandas", "threadpoolctl", "joblib")
ENVIRONMENT = (
    "OPENBLAS_CORETYPE",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
)
TRAIN_RULE = ("role", "train", True)
VALIDATION_RULE = ("role", "validation", True)
PREDICT_TOLERANCE = 1e-12
WORKER: dict[str, Any] = {}


def committed_code() -> dict[str, Any]:
    """
    Commit and code hashes of a run, which must start from committed code.

    Returns
    -------
    dict[str, Any]
        ``git_state`` and ``code_sha256``.

    Raises
    ------
    RuntimeError
        If a code file is uncommitted or a tracked file is modified.
    """
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE], cwd=ROOT, text=True)
    state = git_state()
    if status.strip() or state["modified_tracked"]:
        raise RuntimeError(f"commit the code before running: {status or state['modified_tracked']}")
    return {**state, "code_sha256": code_sha256(CODE)}


def blas_info() -> list[dict[str, Any]]:
    """
    BLAS and OpenMP libraries loaded, with kernel and thread settings.

    Returns
    -------
    list[dict[str, Any]]
        One entry per library of ``threadpoolctl.threadpool_info``.
    """
    from threadpoolctl import threadpool_info

    keys = ("internal_api", "architecture", "num_threads", "version", "threading_layer")
    return [{key: info.get(key) for key in keys} for info in threadpool_info()]


def environment() -> dict[str, Any]:
    """
    Interpreter, package versions, thread and kernel settings.

    Returns
    -------
    dict[str, Any]
        ``python``, ``machine``, ``packages``, ``variables`` (unset ones as None) and ``blas``.
    """
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "packages": {name: version(name) for name in PACKAGES},
        "variables": {name: os.environ.get(name) for name in ENVIRONMENT},
        "blas": blas_info(),
    }


def receipt(command: list[str], started: float, git: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """
    Receipt of one run.

    Parameters
    ----------
    command : list[str]
        Command line.
    started : float
        ``time.time()`` at the start.
    git : dict[str, Any]
        Output of ``committed_code`` taken at the start.
    extra : dict[str, Any]
        Run-specific fields.

    Returns
    -------
    dict[str, Any]
        Command, times, git state, input hashes, environment and the extra fields.
    """
    return {
        "command": command,
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "git": git,
        "inputs": {
            "features": feature_hashes(),
            "manifest": to_stored(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
        },
        "environment": environment(),
        "test_roles_scored": False,
        **extra,
    }


def refuse_coretype() -> None:
    """
    Refuse a forced OpenBLAS kernel: logistic fits use the machine's default, as KEY-04's reproduction did.

    Raises
    ------
    RuntimeError
        If ``OPENBLAS_CORETYPE`` is set.
    """
    if os.environ.get("OPENBLAS_CORETYPE"):
        raise RuntimeError("logistic fits run with the default OpenBLAS kernel; unset OPENBLAS_CORETYPE")


def rule_mask(data: dict[str, np.ndarray], rule: tuple[str, str, bool]) -> np.ndarray:
    """
    Select records by a rule ``(field, value, equal)``.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Manifest arrays.
    rule : tuple[str, str, bool]
        Field, value, and whether records equal to it (True) or different from it (False) are kept.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    field, value, equal = rule
    return (data[field] == value) == equal


def run_pool(
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    task: Callable[[Any], Any],
    tasks: list[Any],
    workers: int,
) -> list[Any]:
    """
    Run tasks in spawned worker processes, results in task order.

    Parameters
    ----------
    initializer : Callable[..., None]
        Worker initializer.
    initargs : tuple[Any, ...]
        Its arguments.
    task : Callable[[Any], Any]
        Function applied to each task.
    tasks : list[Any]
        Tasks.
    workers : int
        Processes.

    Returns
    -------
    list[Any]
        One result per task.
    """
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=initializer, initargs=initargs
    ) as pool:
        return list(pool.map(task, tasks))


def init_logistic_worker(
    name: str,
    scaler: dict[str, np.ndarray],
    threads: int,
    fit_rule: tuple[str, str, bool],
    score_rule: tuple[str, str, bool],
) -> None:
    """
    Load one feature set into a worker: unit rows of the fitting and the scored records.

    Parameters
    ----------
    name : str
        Feature set.
    scaler : dict[str, np.ndarray]
        Standardizer fitted on the fitting records by the parent.
    threads : int
        BLAS and OpenMP threads of this worker.
    fit_rule, score_rule : tuple[str, str, bool]
        ``rule_mask`` rules of the fitting and the scored records.
    """
    from threadpoolctl import threadpool_limits

    WORKER["limits"] = threadpool_limits(threads)
    data, features = load_split(OPEN_ROLES, (name,))
    fit, score = rule_mask(data, fit_rule), rule_mask(data, score_rule)
    WORKER.update(
        z_fit=logistic_inputs(features[name][fit], scaler),
        y_fit=data["labels"][fit],
        z_score=logistic_inputs(features[name][score], scaler),
    )


def logistic_task(task: tuple[int, float]) -> dict[str, Any]:
    """
    Fit one condition at one C in a worker and score the worker's scored records.

    Parameters
    ----------
    task : tuple[int, float]
        Condition column and C.

    Returns
    -------
    dict[str, Any]
        ``fit_logistic`` output with ``column``, ``c``, ``probability``, ``seconds`` and ``blas``.
    """
    column, c = task
    started = time.perf_counter()
    fit = fit_logistic(WORKER["z_fit"], WORKER["y_fit"][:, column], c)
    probability = expit(WORKER["z_score"] @ fit["coef"] + fit["intercept"])
    return {
        **fit,
        "column": column,
        "c": c,
        "probability": probability,
        "seconds": time.perf_counter() - started,
        "blas": blas_info(),
    }


def validation_rows(labels: np.ndarray, probability: np.ndarray, sources: np.ndarray) -> list[dict[str, Any]]:
    """
    Per-source, per-condition validation metrics, and each source's macro AUROC.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Validation source of each record.

    Returns
    -------
    list[dict[str, Any]]
        One row per source and condition (``metrics_table`` fields), then one ``macro`` row per source.
    """
    rows = []
    for source in VALIDATION_SOURCES:
        mask = sources == source
        table = metrics_table(labels[mask], probability[mask])
        keep = (
            "positives",
            "negatives",
            "undefined",
            "prevalence",
            "enters_macro",
            "imprecise",
            "auroc",
            "average_precision",
            "brier",
        )
        rows += [
            {"source": source, "condition": row["condition"], **{key: row[key] for key in keep}}
            for row in table["conditions"]
        ]
        rows.append(
            {
                "source": source,
                "condition": "macro",
                "auroc": table["macro_auroc"],
                "enters_macro": len(table["entering"]),
            }
        )
    return rows


def write_validation(
    candidate: str,
    probability: np.ndarray,
    seeds: tuple[int, ...],
    validation: dict[str, np.ndarray],
    suffix: str = "",
) -> dict[str, Any]:
    """
    Save a validation prediction file and its per-source table.

    Parameters
    ----------
    candidate : str
        Candidate id.
    probability : np.ndarray
        Validation probabilities from ``predict``.
    seeds : tuple[int, ...]
        Seeds whose mean probability this is.
    validation : dict[str, np.ndarray]
        Validation manifest arrays.
    suffix : str
        Suffix of the per-source table name (for one seed of a network).

    Returns
    -------
    dict[str, Any]
        ``selection_metric`` output with ``prediction_file`` and its SHA-256.
    """
    path = prediction_path(CANDIDATES, candidate, "validation", seeds)
    save_predictions(path, validation["record_key"], probability, candidate, seeds)
    rows = validation_rows(validation["labels"], probability, validation["source"])
    write_csv_atomic(CANDIDATES / candidate / f"validation_by_source{suffix}.csv", rows)
    selection = selection_metric(validation["labels"], probability, validation["source"])
    return {**selection, "prediction_file": to_stored(path), "prediction_sha256": sha256_file(path)}


def check_reproduced(name: str, probability: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    """
    Check that ``predict`` reproduces the probabilities computed during fitting.

    Parameters
    ----------
    name : str
        What is compared, for the message.
    probability, expected : np.ndarray
        ``predict`` output and in-fit probabilities.
    tolerance : float
        Largest allowed absolute difference.

    Returns
    -------
    float
        The largest absolute difference.

    Raises
    ------
    ValueError
        If it exceeds the tolerance.
    """
    difference = float(np.max(np.abs(probability - expected)))
    if difference > tolerance:
        raise ValueError(f"{name}: predict differs from the fit by {difference}")
    return difference


def load_open(
    names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Train and validation records with features, and the two role masks.

    Parameters
    ----------
    names : tuple[str, ...]
        Feature sets.

    Returns
    -------
    tuple
        Manifest arrays, features, training mask and validation mask.
    """
    data, features = load_split(OPEN_ROLES, names)
    return data, features, data["role"] == "train", data["role"] == "validation"


def part(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Manifest arrays of masked records.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Manifest arrays.
    mask : np.ndarray
        Boolean mask.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays restricted to the mask.
    """
    return {name: values[mask] for name, values in data.items()}


def grid_edges(values: list[Any], grid: tuple[Any, ...]) -> list[int]:
    """
    Positions whose chosen value is the smallest or largest of its grid.

    Parameters
    ----------
    values : list[Any]
        Chosen values.
    grid : tuple[Any, ...]
        Grid.

    Returns
    -------
    list[int]
        Positions at an edge.
    """
    return [i for i, value in enumerate(values) if value in (min(grid), max(grid))]


def logistic_record(
    results: list[dict[str, Any]], choice: dict[str, Any], chosen: list[float]
) -> dict[str, Any]:
    """
    Grid and choice of a logistic candidate for ``candidate.json``.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Worker results.
    choice : dict[str, Any]
        Output of ``choose_per_condition``.
    chosen : list[float]
        Chosen C per condition.

    Returns
    -------
    dict[str, Any]
        Per-condition grid scores, iterations, edges and ties.
    """
    iterations = {(r["column"], r["c"]): r["iterations"] for r in results}
    grid = {
        name: {
            str(c): {
                "validation_metric": float(choice["scores"][i, column]),
                "iterations": iterations[(column, c)],
            }
            for i, c in enumerate(LOGISTIC_C)
        }
        for column, name in enumerate(CONDITIONS)
    }
    return {
        "chosen_c": dict(zip(CONDITIONS, chosen, strict=True)),
        "grid": grid,
        "at_grid_edge": [CONDITIONS[i] for i in grid_edges(chosen, LOGISTIC_C)],
        "ties": choice["ties"],
        "fit_seconds_total": float(sum(r["seconds"] for r in results)),
        "worker_blas": sorted(
            {(b["internal_api"], b["architecture"], b["num_threads"]) for r in results for b in r["blas"]},
            key=str,
        ),
    }


def run_logistic(candidate: str, workers: int, threads: int, command: list[str]) -> dict[str, Any]:
    """
    Fit P0, L-J or L-C: every condition at every C on train, C chosen per condition on validation.

    Parameters
    ----------
    candidate : str
        P0, L-J or L-C.
    workers : int
        Worker processes.
    threads : int
        BLAS threads per worker.
    command : list[str]
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        ``candidate.json`` content.
    """
    started, git = time.time(), committed_code()
    refuse_coretype()
    name = INPUTS[candidate][0]
    data, features, train, valid = load_open((name,))
    validation = part(data, valid)
    scaler = fit_scaler(features[name][train])
    tasks = [(column, c) for c in sorted(LOGISTIC_C, reverse=True) for column in range(len(CONDITIONS))]
    results = run_pool(
        init_logistic_worker,
        (name, scaler, threads, TRAIN_RULE, VALIDATION_RULE),
        logistic_task,
        tasks,
        workers,
    )
    found = {(r["column"], r["c"]): r for r in results}
    columns = range(len(CONDITIONS))
    probabilities = [np.column_stack([found[(col, c)]["probability"] for col in columns]) for c in LOGISTIC_C]
    choice = choose_per_condition(validation["labels"], validation["source"], probabilities)
    chosen = [LOGISTIC_C[i] for i in choice["chosen"]]
    fits = {
        "coef": np.stack([found[(col, chosen[col])]["coef"] for col in columns]),
        "intercept": np.array([found[(col, chosen[col])]["intercept"] for col in columns]),
        "c": np.array(chosen),
        "iterations": np.array([found[(col, chosen[col])]["iterations"] for col in columns]),
    }
    folder = CANDIDATES / candidate
    record = logistic_record(results, choice, chosen)
    save_logistic(folder, candidate, scaler, fits, {"c": record["chosen_c"]})
    write_npz_atomic(
        folder / "grid.npz",
        c=np.array(LOGISTIC_C),
        scores=choice["scores"],
        coef=np.stack([[found[(col, c)]["coef"] for col in columns] for c in LOGISTIC_C]),
        intercept=np.array([[found[(col, c)]["intercept"] for col in columns] for c in LOGISTIC_C]),
    )
    write_csv_atomic(
        folder / "training_log.csv",
        [{key: r[key] for key in ("column", "c", "iterations", "records", "seconds")} for r in results],
    )
    probability = predict(candidate, {name: features[name][valid]}, CANDIDATES)
    difference = check_reproduced(candidate, probability, choice["probability"], PREDICT_TOLERANCE)
    record |= {
        "candidate": candidate,
        "kind": KINDS[candidate],
        "inputs": [name],
        "seeds": [LOGISTIC_SEED],
        "train_records": int(train.sum()),
        "validation_records": int(valid.sum()),
        "predict_max_abs_difference": difference,
        "validation": write_validation(candidate, probability, (LOGISTIC_SEED,), validation),
    }
    write_json_atomic(folder / "candidate.json", record)
    write_json_atomic(
        folder / "receipt.json",
        receipt(command, started, git, {"workers": workers, "threads_per_worker": threads}),
    )
    return record


def init_boosting_worker(threads: int) -> None:
    """
    Load the measurements of train and validation into a worker.

    Parameters
    ----------
    threads : int
        OpenMP threads of this worker.
    """
    from threadpoolctl import threadpool_limits

    WORKER["limits"] = threadpool_limits(threads)
    data, features = load_split(OPEN_ROLES, ("measurements",))
    train, valid = data["role"] == "train", data["role"] == "validation"
    WORKER.update(
        x_fit=features["measurements"][train],
        y_fit=data["labels"][train],
        x_score=features["measurements"][valid],
        y_score=data["labels"][valid],
        sources=data["source"][valid],
    )


def boosting_task(task: tuple[int, float, int, int]) -> dict[str, Any]:
    """
    Fit one condition at one grid setting up to an iteration cap and score every iteration on validation.

    Parameters
    ----------
    task : tuple[int, float, int, int]
        Condition column, learning rate, leaves and the iteration cap (``BOOSTING_MAX_ITER``).

    Returns
    -------
    dict[str, Any]
        ``column``, ``rate``, ``leaves``, ``curve`` (metric per iteration), ``iterations`` (best count),
        ``metric``, ``probability`` (validation, at the best count) and ``seconds``.
    """
    column, rate, leaves, cap = task
    started = time.perf_counter()
    model = fit_boosting(WORKER["x_fit"], WORKER["y_fit"][:, column], rate, leaves, cap)
    curve, staged = staged_condition_scores(
        model, WORKER["x_score"], WORKER["y_score"][:, column], WORKER["sources"]
    )
    best = int(np.argmax(curve))
    return {
        "column": column,
        "rate": rate,
        "leaves": leaves,
        "curve": curve,
        "iterations": best + 1,
        "metric": float(curve[best]),
        "probability": staged[:, best].copy(),
        "fitted": int(model.n_iter_),
        "seconds": time.perf_counter() - started,
        "blas": blas_info(),
    }


def refit_boosting_task(task: tuple[int, float, int, int]) -> dict[str, Any]:
    """
    Refit one condition at its chosen setting and number of iterations.

    Parameters
    ----------
    task : tuple[int, float, int, int]
        Condition column, learning rate, leaves and iterations.

    Returns
    -------
    dict[str, Any]
        ``column``, ``model`` and its validation ``probability``.
    """
    column, rate, leaves, iterations = task
    model = fit_boosting(WORKER["x_fit"], WORKER["y_fit"][:, column], rate, leaves, iterations)
    return {"column": column, "model": model, "probability": model.predict_proba(WORKER["x_score"])[:, 1]}


def boosting_choice(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Best setting per condition: highest validation metric at its best iteration count, earliest on ties.

    Parameters
    ----------
    results : list[dict[str, Any]]
        ``boosting_task`` outputs.

    Returns
    -------
    list[dict[str, Any]]
        One chosen result per condition, in condition order.
    """
    grid = list(itertools.product(BOOSTING_RATES, BOOSTING_LEAVES))
    found = {(r["column"], r["rate"], r["leaves"]): r for r in results}
    chosen = []
    for column in range(len(CONDITIONS)):
        options = [found[(column, rate, leaves)] for rate, leaves in grid]
        chosen.append(options[int(np.argmax([option["metric"] for option in options]))])
    return chosen


def boosting_record(results: list[dict[str, Any]], chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Grid and choice of G-M for ``candidate.json``.

    Parameters
    ----------
    results : list[dict[str, Any]]
        ``boosting_task`` outputs.
    chosen : list[dict[str, Any]]
        Output of ``boosting_choice``.

    Returns
    -------
    dict[str, Any]
        Chosen settings, the grid's best metric and iteration count, and iteration caps reached.
    """
    grid = {
        name: {
            f"rate={r['rate']},leaves={r['leaves']}": {
                "validation_metric": r["metric"],
                "iterations": r["iterations"],
            }
            for r in results
            if r["column"] == column
        }
        for column, name in enumerate(CONDITIONS)
    }
    settings = {
        name: {"learning_rate": r["rate"], "max_leaf_nodes": r["leaves"], "iterations": r["iterations"]}
        for name, r in zip(CONDITIONS, chosen, strict=True)
    }
    return {
        "chosen": settings,
        "grid": grid,
        "max_iter": BOOSTING_MAX_ITER,
        "at_iteration_cap": [
            r for r in grid if any(v["iterations"] == BOOSTING_MAX_ITER for v in grid[r].values())
        ],
        "fit_seconds_total": float(sum(r["seconds"] for r in results)),
    }


def run_boosting(workers: int, threads: int, command: list[str]) -> dict[str, Any]:
    """
    Fit G-M: every condition at every grid setting on train, settings and iterations chosen on validation.

    Parameters
    ----------
    workers : int
        Worker processes.
    threads : int
        OpenMP threads per worker.
    command : list[str]
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        ``candidate.json`` content.
    """
    started, git = time.time(), committed_code()
    data, features, train, valid = load_open(("measurements",))
    validation = part(data, valid)
    tasks = [
        (column, rate, leaves, BOOSTING_MAX_ITER)
        for rate, leaves in itertools.product(BOOSTING_RATES, BOOSTING_LEAVES)
        for column in range(len(CONDITIONS))
    ]
    results = run_pool(init_boosting_worker, (threads,), boosting_task, tasks[::-1], workers)
    chosen = boosting_choice(results)
    refits = run_pool(
        init_boosting_worker,
        (threads,),
        refit_boosting_task,
        [(r["column"], r["rate"], r["leaves"], r["iterations"]) for r in chosen],
        workers,
    )
    staged = np.column_stack([r["probability"] for r in chosen])
    refit_difference = check_reproduced(
        "G-M refit", np.column_stack([r["probability"] for r in refits]), staged, PREDICT_TOLERANCE
    )
    folder = CANDIDATES / "G-M"
    record = boosting_record(results, chosen)
    save_boosting(folder, [r["model"] for r in refits], record["chosen"])
    write_npz_atomic(
        folder / "curves.npz",
        **{f"{CONDITIONS[r['column']]}__rate-{r['rate']}__leaves-{r['leaves']}": r["curve"] for r in results},
    )
    write_csv_atomic(
        folder / "training_log.csv",
        [
            {key: r[key] for key in ("column", "rate", "leaves", "fitted", "iterations", "metric", "seconds")}
            for r in results
        ],
    )
    probability = predict("G-M", {"measurements": features["measurements"][valid]}, CANDIDATES)
    difference = check_reproduced("G-M", probability, staged, PREDICT_TOLERANCE)
    record |= {
        "candidate": "G-M",
        "kind": "boosting",
        "inputs": ["measurements"],
        "seeds": [BOOSTING_SEED],
        "train_records": int(train.sum()),
        "validation_records": int(valid.sum()),
        "refit_max_abs_difference": refit_difference,
        "predict_max_abs_difference": difference,
        "worker_blas": sorted(
            {(b["internal_api"], b["architecture"], b["num_threads"]) for r in results for b in r["blas"]},
            key=str,
        ),
        "validation": write_validation("G-M", probability, (BOOSTING_SEED,), validation),
    }
    write_json_atomic(folder / "candidate.json", record)
    write_json_atomic(
        folder / "receipt.json",
        receipt(command, started, git, {"workers": workers, "threads_per_worker": threads}),
    )
    return record


def selection_value(labels: np.ndarray, sources: np.ndarray, probability: np.ndarray) -> float:
    """
    Compute the validation selection metric of a probability matrix.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    sources : np.ndarray
        Validation sources.
    probability : np.ndarray
        Aligned probabilities.

    Returns
    -------
    float
        Mean over validation sources of the within-source macro AUROC.
    """
    return selection_metric(labels, probability, sources)["selection_metric"]


def network_arrays(
    candidate: str, features: dict[str, np.ndarray], fit: np.ndarray, score: np.ndarray
) -> dict[str, Any]:
    """
    Standardized float32 inputs of a network, with the transforms fitted on the fitting records.

    Parameters
    ----------
    candidate : str
        H-X, H-J, H-C or H-All.
    features : dict[str, np.ndarray]
        Raw features.
    fit, score : np.ndarray
        Masks of the fitting and the scored records.

    Returns
    -------
    dict[str, Any]
        ``medians`` (or None), ``scaler``, ``z_fit`` and ``z_score``.
    """
    medians = fit_imputation(features["measurements"][fit]) if "measurements" in INPUTS[candidate] else None
    raw_fit = network_inputs(candidate, {name: values[fit] for name, values in features.items()}, medians)
    scaler = fit_scaler(raw_fit)
    raw_score = network_inputs(candidate, {name: values[score] for name, values in features.items()}, medians)
    return {
        "medians": medians,
        "scaler": scaler,
        "z_fit": standardize(raw_fit, scaler).astype(np.float32),
        "z_score": standardize(raw_score, scaler).astype(np.float32),
    }


def gpu_state(device: str) -> dict[str, Any]:
    """
    Device, free memory and numerical flags before a network run.

    Parameters
    ----------
    device : str
        ``cuda`` or ``cpu``.

    Returns
    -------
    dict[str, Any]
        Device name, free and total bytes, and torch flags.

    Raises
    ------
    RuntimeError
        If deterministic algorithms are off, the cuBLAS workspace is not fixed, or the GPU has too little
        free memory.
    """
    import torch

    torch.use_deterministic_algorithms(True)
    flags = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_threads": torch.get_num_threads(),
    }
    if device != "cuda":
        return {"device": device, **flags}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic cuBLAS")
    free, total = torch.cuda.mem_get_info()
    if free < MIN_FREE_GPU_BYTES:
        raise RuntimeError(f"only {free} bytes free on the GPU")
    return {
        "device": device,
        "name": torch.cuda.get_device_name(),
        "free_bytes": free,
        "total_bytes": total,
        **flags,
    }


def locked_training(device: str, **arguments: Any) -> dict[str, Any]:
    """
    ``train_network`` while holding the GPU 1 lock, with peak memory.

    Parameters
    ----------
    device : str
        Torch device.
    **arguments : Any
        Arguments of ``train_network`` other than ``device``.

    Returns
    -------
    dict[str, Any]
        ``train_network`` output with ``seconds`` and ``peak_allocated_bytes``.
    """
    import torch

    started = time.perf_counter()
    with gpu_lock(device, path=GPU_LOCK):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = train_network(device=device, **arguments)
        peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return {**result, "seconds": time.perf_counter() - started, "peak_allocated_bytes": peak}


def profile_network(
    candidate: str,
    arrays: dict[str, Any],
    y: np.ndarray,
    score: Callable[[np.ndarray], float],
    device: str,
    epochs: int,
) -> dict[str, Any]:
    """
    Time a few epochs of the first grid setting and forecast the grid and the seeds.

    Parameters
    ----------
    candidate : str
        Network candidate.
    arrays : dict[str, Any]
        Output of ``network_arrays``.
    y : np.ndarray
        Training states.
    score : Callable[[np.ndarray], float]
        Validation selection metric.
    device : str
        Torch device.
    epochs : int
        Profiled epochs.

    Returns
    -------
    dict[str, Any]
        Seconds per epoch (the first one excluded when possible), peak memory and the forecast.
    """
    result = locked_training(
        device,
        x=arrays["z_fit"],
        y=y,
        rate=NETWORK_RATES[0],
        decay=NETWORK_DECAYS[0],
        seed=SEEDS[0],
        epochs=epochs,
        x_val=arrays["z_score"],
        score=score,
    )
    seconds = [row["seconds"] for row in result["history"]]
    per_epoch = float(np.mean(seconds[1:] if len(seconds) > 1 else seconds))
    runs = len(NETWORK_RATES) * len(NETWORK_DECAYS) + len(SEEDS)
    return {
        "candidate": candidate,
        "epochs": epochs,
        "epoch_seconds": seconds,
        "seconds_per_epoch": per_epoch,
        "peak_allocated_bytes": result["peak_allocated_bytes"],
        "runs": runs,
        "forecast_seconds": per_epoch * EPOCHS * runs,
        "inputs": int(arrays["z_fit"].shape[1]),
    }


def history_rows(run: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Training log rows of one network run.

    Parameters
    ----------
    run : str
        Run label.
    result : dict[str, Any]
        ``train_network`` output.

    Returns
    -------
    list[dict[str, Any]]
        One row per epoch.
    """
    return [{"run": run, **row} for row in result["history"]]


def network_grid(
    arrays: dict[str, Any], y: np.ndarray, score: Callable[[np.ndarray], float], device: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Grid search with the first seed; the setting with the highest best-epoch metric wins, earliest on ties.

    Parameters
    ----------
    arrays : dict[str, Any]
        Output of ``network_arrays``.
    y : np.ndarray
        Training states.
    score : Callable[[np.ndarray], float]
        Validation selection metric.
    device : str
        Torch device.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Every grid run (with ``rate`` and ``decay``) and the chosen one.
    """
    runs = []
    for rate, decay in itertools.product(NETWORK_RATES, NETWORK_DECAYS):
        result = locked_training(
            device,
            x=arrays["z_fit"],
            y=y,
            rate=rate,
            decay=decay,
            seed=SEEDS[0],
            epochs=EPOCHS,
            x_val=arrays["z_score"],
            score=score,
        )
        runs.append({**result, "rate": rate, "decay": decay})
    return runs, runs[int(np.argmax([run["metric"] for run in runs]))]


def network_record(
    grid: list[dict[str, Any]], chosen: dict[str, Any], seeds: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """
    Grid, choice and seed runs of a network candidate for ``candidate.json``.

    Parameters
    ----------
    grid : list[dict[str, Any]]
        Grid runs.
    chosen : dict[str, Any]
        Chosen grid run.
    seeds : dict[int, dict[str, Any]]
        Seed to its run at the chosen setting.

    Returns
    -------
    dict[str, Any]
        Grid table, chosen setting, best epoch per seed, edges and the first seed's rerun check.
    """
    table = [
        {
            "learning_rate": run["rate"],
            "weight_decay": run["decay"],
            "best_epoch": run["epoch"],
            "validation_metric": run["metric"],
            "last_epoch_metric": run["history"][-1]["validation_metric"],
            "seconds": run["seconds"],
            "peak_allocated_bytes": run["peak_allocated_bytes"],
        }
        for run in grid
    ]
    rerun = seeds[SEEDS[0]]["weights"]
    same = all(np.array_equal(rerun[key], chosen["weights"][key]) for key in rerun)
    epochs = {str(seed): run["epoch"] for seed, run in seeds.items()}
    return {
        "grid": table,
        "chosen": {"learning_rate": chosen["rate"], "weight_decay": chosen["decay"]},
        "epoch_by_seed": epochs,
        "at_epoch_edge": [seed for seed, epoch in epochs.items() if epoch in (1, EPOCHS)],
        "first_seed_rerun_identical": same,
        "seed_runs": {
            str(seed): {
                "best_epoch": run["epoch"],
                "gpu_validation_metric": run["metric"],
                "seconds": run["seconds"],
            }
            for seed, run in seeds.items()
        },
    }


def network_validation(
    candidate: str,
    features: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    seeds: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """
    Save the validation predictions of every seed and of their mean, all from ``predict``.

    Parameters
    ----------
    candidate : str
        Network candidate.
    features : dict[str, np.ndarray]
        Raw validation features.
    validation : dict[str, np.ndarray]
        Validation manifest arrays.
    seeds : dict[int, dict[str, Any]]
        Seed to its run, holding the best epoch's GPU probabilities.

    Returns
    -------
    dict[str, Any]
        ``validation`` (the mean of the seeds) and ``by_seed`` (validation and the largest difference from the
        GPU probabilities).
    """
    by_seed = {}
    for seed, run in seeds.items():
        probability = predict(candidate, features, CANDIDATES, seeds=(seed,))
        by_seed[str(seed)] = {
            "validation": write_validation(candidate, probability, (seed,), validation, f"__seed-{seed}"),
            "gpu_max_abs_difference": float(np.max(np.abs(probability - run["probability"]))),
        }
    probability = predict(candidate, features, CANDIDATES)
    return {"validation": write_validation(candidate, probability, SEEDS, validation), "by_seed": by_seed}


def run_network(
    candidate: str, device: str, threads: int, profile_epochs: int | None, command: list[str]
) -> dict[str, Any]:
    """
    Fit a network candidate: grid with the first seed, then three seeds at the chosen setting.

    A profile (``profile_epochs``) writes ``profile.json`` only; the full run needs it.

    Parameters
    ----------
    candidate : str
        H-X, H-J, H-C or H-All.
    device : str
        ``cuda`` (GPU 1 through ``CUDA_VISIBLE_DEVICES``) or ``cpu``.
    threads : int
        CPU threads.
    profile_epochs : int | None
        Epochs of a profile run; None runs the candidate.
    command : list[str]
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        The profile, or ``candidate.json`` content.
    """
    import torch
    from threadpoolctl import threadpool_limits

    started, git = time.time(), committed_code()
    torch.set_num_threads(threads)
    folder = CANDIDATES / candidate
    with threadpool_limits(threads):
        state = gpu_state(device)
        data, features, train, valid = load_open(INPUTS[candidate])
        validation = part(data, valid)
        arrays = network_arrays(candidate, features, train, valid)
        score = partial(selection_value, validation["labels"], validation["source"])
        if profile_epochs:
            profile = profile_network(candidate, arrays, data["labels"][train], score, device, profile_epochs)
            write_json_atomic(folder / "profile.json", {**profile, "gpu": state, "git": git})
            return profile
        profile = read_json(folder / "profile.json")
        grid, chosen = network_grid(arrays, data["labels"][train], score, device)
        seeds = {
            seed: locked_training(
                device,
                x=arrays["z_fit"],
                y=data["labels"][train],
                rate=chosen["rate"],
                decay=chosen["decay"],
                seed=seed,
                epochs=EPOCHS,
                x_val=arrays["z_score"],
                score=score,
            )
            for seed in SEEDS
        }
        record = network_record(grid, chosen, seeds)
        save_network(
            folder,
            candidate,
            arrays["scaler"],
            arrays["medians"],
            {seed: run["weights"] for seed, run in seeds.items()},
            {**record["chosen"], "epoch_by_seed": record["epoch_by_seed"]},
        )
        write_csv_atomic(
            folder / "training_log.csv",
            [
                row
                for run in grid
                for row in history_rows(f"grid rate={run['rate']} decay={run['decay']}", run)
            ]
            + [row for seed, run in seeds.items() for row in history_rows(f"seed {seed}", run)],
        )
        valid_features = {name: values[valid] for name, values in features.items()}
        record |= {
            "candidate": candidate,
            "kind": "network",
            "inputs": list(INPUTS[candidate]),
            "input_width": int(arrays["z_fit"].shape[1]),
            "seeds": list(SEEDS),
            "train_records": int(train.sum()),
            "validation_records": int(valid.sum()),
            **network_validation(candidate, valid_features, validation, seeds),
        }
        write_json_atomic(folder / "candidate.json", record)
        write_json_atomic(
            folder / "receipt.json",
            receipt(command, started, git, {"gpu": state, "profile": profile, "threads": threads}),
        )
    return record


def source_rows(
    candidate: str, held: str, data: dict[str, np.ndarray], probability: np.ndarray
) -> list[dict[str, Any]]:
    """
    Within-source metrics of one held-out family.

    Parameters
    ----------
    candidate : str
        Candidate id.
    held : str
        Held-out family.
    data : dict[str, np.ndarray]
        Manifest arrays of the held-out records.
    probability : np.ndarray
        Their probabilities.

    Returns
    -------
    list[dict[str, Any]]
        Per source: one row per condition and one ``macro`` row.
    """
    rows = []
    for source in [s for s in VALIDATION_SOURCES if s in set(data["source"])]:
        mask = data["source"] == source
        table = metrics_table(data["labels"][mask], probability[mask])
        rows += [
            {
                "candidate": candidate,
                "held_out": held,
                "source": source,
                "condition": row["condition"],
                "positives": row["positives"],
                "negatives": row["negatives"],
                "enters_macro": row["enters_macro"],
                "auroc": row["auroc"],
            }
            for row in table["conditions"]
        ]
        rows.append(
            {
                "candidate": candidate,
                "held_out": held,
                "source": source,
                "condition": "macro",
                "records": int(mask.sum()),
                "enters_macro": len(table["entering"]),
                "auroc": table["macro_auroc"],
            }
        )
    return rows


def lofo_logistic(
    candidate: str,
    held: str,
    data: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    model: dict[str, Any],
    workers: int,
    threads: int,
) -> np.ndarray:
    """
    Logistic fold of leave-one-family-out at the main run's C per condition.

    Parameters
    ----------
    candidate : str
        P0 or L-C.
    held : str
        Held-out family.
    data : dict[str, np.ndarray]
        Train and validation manifest arrays.
    features : dict[str, np.ndarray]
        Their features.
    model : dict[str, Any]
        Main run's ``model.json``.
    workers, threads : int
        Worker processes and BLAS threads per worker.

    Returns
    -------
    np.ndarray
        Probabilities of the held-out records.
    """
    name = INPUTS[candidate][0]
    scaler = fit_scaler(features[name][data["family"] != held])
    tasks = [
        (column, float(model["settings"]["c"][condition])) for column, condition in enumerate(CONDITIONS)
    ]
    results = run_pool(
        init_logistic_worker,
        (name, scaler, threads, ("family", held, False), ("family", held, True)),
        logistic_task,
        tasks,
        workers,
    )
    return np.column_stack([r["probability"] for r in results])


def lofo_network(
    candidate: str,
    held: str,
    data: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    model: dict[str, Any],
    device: str,
) -> np.ndarray:
    """
    Network fold of leave-one-family-out: first seed, chosen setting, the main run's best epoch count.

    Parameters
    ----------
    candidate : str
        H-All.
    held : str
        Held-out family.
    data : dict[str, np.ndarray]
        Train and validation manifest arrays.
    features : dict[str, np.ndarray]
        Their features.
    model : dict[str, Any]
        Main run's ``model.json``.
    device : str
        Torch device.

    Returns
    -------
    np.ndarray
        Probabilities of the held-out records.
    """
    fit = data["family"] != held
    arrays = network_arrays(candidate, features, fit, ~fit)
    settings = model["settings"]
    result = locked_training(
        device,
        x=arrays["z_fit"],
        y=data["labels"][fit],
        rate=settings["learning_rate"],
        decay=settings["weight_decay"],
        seed=SEEDS[0],
        epochs=int(settings["epoch_by_seed"][str(SEEDS[0])]),
    )
    return network_probability(arrays["z_score"], result["weights"])


def run_lofo(candidate: str, workers: int, threads: int, device: str, command: list[str]) -> dict[str, Any]:
    """
    Leave-one-family-out over train and validation records for P0, L-C or H-All, one seed, no re-tuning.

    Parameters
    ----------
    candidate : str
        P0, L-C or H-All.
    workers, threads : int
        Worker processes and threads per worker (threads of the run for H-All).
    device : str
        Torch device for H-All.
    command : list[str]
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        Per held-out family and source, the macro AUROC.
    """
    from threadpoolctl import threadpool_limits

    if candidate not in LOFO_CANDIDATES:
        raise ValueError(f"leave-one-family-out runs for {LOFO_CANDIDATES} only")
    started, git = time.time(), committed_code()
    model = read_json(CANDIDATES / candidate / "model.json")
    data, features = load_split(OPEN_ROLES, INPUTS[candidate])
    folder, rows, state = LOFO / candidate, [], None
    for held in FAMILIES:
        if KINDS[candidate] == "logistic":
            refuse_coretype()
            probability = lofo_logistic(candidate, held, data, features, model, workers, threads)
        else:
            with threadpool_limits(threads):
                state = gpu_state(device)
                probability = lofo_network(candidate, held, data, features, model, device)
        mask = data["family"] == held
        write_npz_atomic(
            folder / f"held_out__{held}.npz",
            format=np.array("mlc02-lofo-v1"),
            record_key=data["record_key"][mask],
            probability=probability,
            condition=np.asarray(CONDITIONS),
            model_id=np.array(candidate),
        )
        rows += source_rows(candidate, held, part(data, mask), probability)
    write_csv_atomic(folder / "lofo.csv", rows)
    summary = {
        "candidate": candidate,
        "families": list(FAMILIES),
        "settings": model["settings"],
        "seed": SEEDS[0] if KINDS[candidate] == "network" else LOGISTIC_SEED,
        "macro_by_held_out_source": {
            f"{r['held_out']}/{r['source']}": r["auroc"] for r in rows if r["condition"] == "macro"
        },
    }
    write_json_atomic(folder / "lofo.json", summary)
    write_json_atomic(
        folder / "receipt.json",
        receipt(command, started, git, {"gpu": state, "workers": workers, "threads": threads}),
    )
    return summary


def identification_sets(
    features: dict[str, np.ndarray], train: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Build the validation feature sets of the source-identification audit.

    Measurements are imputed with the training median plus a missing flag, as in H-All.

    Parameters
    ----------
    features : dict[str, np.ndarray]
        Raw train and validation features.
    train, valid : np.ndarray
        Role masks.

    Returns
    -------
    dict[str, np.ndarray]
        ``xecg``, ``jepa``, ``cpc``, ``measurements`` and ``all`` (the three encoders and the measurements).
    """
    medians = fit_imputation(features["measurements"][train])
    sets = {name: features[name][valid] for name in ("xecg", "jepa", "cpc")}
    sets["measurements"] = impute(features["measurements"][valid], medians)
    sets["all"] = np.hstack([sets["xecg"], sets["jepa"], sets["cpc"], sets["measurements"]])
    return sets


def run_source_identification(threads: int, command: list[str]) -> dict[str, Any]:
    """
    Source-family identification from each feature set on validation records (five folds).

    Parameters
    ----------
    threads : int
        BLAS threads.
    command : list[str]
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        ``source_identification`` output per feature set, with seconds and convergence warnings.
    """
    from threadpoolctl import threadpool_limits

    started, git = time.time(), committed_code()
    data, features, train, valid = load_open(tuple(FEATURE_FILES))
    results = {}
    with threadpool_limits(threads):
        for name, x in identification_sets(features, train, valid).items():
            began = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = source_identification(x, data["family"][valid], data["group"][valid])
            results[name] = {
                **result,
                "features": int(x.shape[1]),
                "seconds": time.perf_counter() - began,
                "warnings": sorted({str(w.message)[:200] for w in caught}),
            }
    write_json_atomic(SOURCE_IDENTIFICATION / "source_identification.json", results)
    write_csv_atomic(
        SOURCE_IDENTIFICATION / "source_identification.csv",
        [
            {
                "features": name,
                **{
                    key: r[key]
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_ovr_auroc",
                        "log_loss",
                        "majority_accuracy",
                        "chance_balanced_accuracy",
                        "records",
                        "seconds",
                    )
                },
            }
            for name, r in results.items()
        ],
    )
    write_json_atomic(
        SOURCE_IDENTIFICATION / "receipt.json", receipt(command, started, git, {"threads": threads})
    )
    return results
