"""
Run KEY-06, the structure of the xECG embedding space, on the train and validation records only.

Steps in order: ``integrity``, ``variance`` (step 1), ``residual`` (step 2), ``prepare`` and ``cluster`` and
``assemble`` (step 3), ``describe`` and ``review`` (step 4), ``cross-encoder`` (step 5), ``outliers`` (step 6)
and ``statements`` (step 7). ``detail`` draws twelve-lead sheets of one cluster and ``note`` records the
analyst's reading of a sheet. ``--smoke`` runs on a fixed sample of 6,000 train and 1,500 validation records,
with 3 subsamples and 50 bootstrap draws, and writes under ``/var/tmp/janetrivera/key06/smoke``; its numbers
are not results.

Examples
--------
OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_key06_embedding_structure variance --workers 12
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_key06_embedding_structure cluster --workers 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import key06_embedding_structure as key06
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.mlc02_bootstrap import labels_digest
from ecg_experiment.mlc02_data import CONDITIONS, VALIDATION_SOURCES, load_predictions
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_waveforms import CACHE
from ecg_experiment.mlc03_signal_review import draw_sheet
from ecg_experiment.mlc03_signal_runs import write_receipt
from ecg_experiment.paths import to_stored

SMOKE_RECORDS = {"train": 6000, "validation": 1500}
SMOKE_SUBSAMPLES = 3
SMOKE_DRAWS = 50
CODE = [
    "ecg_experiment/key06_embedding_structure.py",
    "scripts/experiments/run_key06_embedding_structure.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc03_signal_runs.py",
    "ecg_experiment/mlc03_signal_review.py",
    "ecg_experiment/measurement_extractor042.py",
    "ecg_experiment/mlc_labels.py",
]
FEATURE_FILES = ("xecg.npy", "cpc.npy", "measurements.npy", "rows.csv", "measurements_status.csv")


def paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Output and scratch directories of a run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    tuple[Path, Path]
        ``outputs/key06_embedding_structure_20261111`` and ``/var/tmp/janetrivera/key06``, or their smoke
        versions under the scratch directory.
    """
    if args.smoke:
        return key06.WORK / "smoke" / "outputs", key06.WORK / "smoke" / "work"
    return key06.OUTPUT, key06.WORK


def settings(args: argparse.Namespace) -> dict[str, int]:
    """
    Subsamples and bootstrap draws of a run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict[str, int]
        ``subsamples`` and ``draws``.
    """
    if args.smoke:
        return {"subsamples": SMOKE_SUBSAMPLES, "draws": SMOKE_DRAWS}
    return {"subsamples": key06.SUBSAMPLES, "draws": key06.DRAWS}


def stored(path: Path) -> str:
    """
    Repository-relative path when inside the repository, else absolute.

    Parameters
    ----------
    path : Path
        Any path.

    Returns
    -------
    str
        Text for a receipt.
    """
    absolute = Path(os.path.abspath(path))
    return to_stored(absolute) if absolute.is_relative_to(ROOT) else str(absolute)


def digests(files: list[Path]) -> dict[str, str]:
    """
    SHA-256 of input files.

    Parameters
    ----------
    files : list[Path]
        Files.

    Returns
    -------
    dict[str, str]
        Stored path to digest.
    """
    return {stored(path): sha256_file(path) for path in files}


def receipt(
    args: argparse.Namespace, step: str, started: float, content: dict[str, Any], inputs: list[Path]
) -> None:
    """
    Write a step receipt: command, seeds, settings, inputs, code, git state, threads and memory errors.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    step : str
        Step name.
    started : float
        ``time.time()`` at the start.
    content : dict[str, Any]
        Step-specific content.
    inputs : list[Path]
        Input files to hash.
    """
    output, _ = paths(args)
    import sklearn

    write_receipt(
        output / "receipts" / f"{step}.receipt.json",
        {
            "step": step,
            "command": sys.argv,
            "smoke": bool(args.smoke),
            "workers": args.workers,
            "seed": key06.SEED,
            **settings(args),
            "wall_seconds": time.time() - started,
            "inputs": digests(inputs),
            "code": {name: sha256_file(ROOT / name) for name in CODE},
            "sklearn": sklearn.__version__,
            **content,
        },
    )


def smoke_sample(records: pd.DataFrame) -> pd.DataFrame:
    """
    Draw a fixed random sample of records for a smoke run, in manifest order.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.

    Returns
    -------
    pd.DataFrame
        ``SMOKE_RECORDS`` records per role.
    """
    rng = np.random.default_rng(key06.SEED)
    keep = np.zeros(len(records), dtype=bool)
    for role, count in SMOKE_RECORDS.items():
        rows = np.flatnonzero(records["role"].to_numpy() == role)
        keep[rng.choice(rows, size=count, replace=False)] = True
    return records[keep].reset_index(drop=True)


def bundle(args: argparse.Namespace) -> dict[str, Any]:
    """
    Load records, labels, flags, measurements and standardized xECG embeddings of the run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict[str, Any]
        ``records``, ``train`` (mask), ``labels``, ``flags``, ``measurements``, ``names``, ``z_train`` and
        ``z_validation``.
    """
    records = key06.load_records()
    if args.smoke:
        records = smoke_sample(records)
    train = (records["role"] == "train").to_numpy()
    xecg = key06.feature_rows("xecg", records["position"].to_numpy())
    z_train, z_validation = key06.standardize(xecg[train], xecg[~train])
    return {
        "records": records,
        "train": train,
        "labels": key06.label_matrix(records),
        "flags": key06.load_flags(records),
        "names": key06.measurement_names(),
        "measurements": key06.feature_rows("measurements", records["position"].to_numpy()),
        "z_train": z_train,
        "z_validation": z_validation,
    }


def split(data: dict[str, Any], train: bool) -> dict[str, Any]:
    """
    Factor inputs of the train or validation records.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    train : bool
        True for train.

    Returns
    -------
    dict[str, Any]
        Inputs of ``factor_blocks``.
    """
    mask = data["train"] if train else ~data["train"]
    records = data["records"][mask]
    return {
        "source": records["source"].to_numpy(),
        "labels": data["labels"][mask],
        "measurements": data["measurements"][mask],
        "flags": data["flags"][mask].reset_index(drop=True),
        "age": records["age"].to_numpy(),
        "sex": records["sex"].to_numpy(),
    }


def blocks_of(data: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Factor design matrices of the run.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Output of ``factor_blocks``.
    """
    return key06.factor_blocks(split(data, True), split(data, False), data["names"])


def embeddings(data: dict[str, Any], space: str, work: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and validation embeddings of one space.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    space : str
        ``raw`` or ``residual`` (written by the ``residual`` step and checked against its receipt).
    work : Path
        Scratch directory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 train and validation rows.
    """
    if space == "raw":
        return data["z_train"], data["z_validation"]
    hashes = json.loads((work / "residual_hashes.json").read_text())
    arrays = []
    for name in ("train", "validation"):
        path = work / f"residual_{name}.npy"
        if sha256_file(path) != hashes[path.name]:
            raise ValueError(f"{path} changed since the residual step")
        arrays.append(np.load(path).astype(np.float64))
    return arrays[0], arrays[1]


def parallel(function: Any, tasks: list[tuple], workers: int, work: Path) -> list[Any]:
    """
    Run tasks with joblib processes, large arrays shared through memory maps.

    Parameters
    ----------
    function : Any
        Task function.
    tasks : list[tuple]
        Positional arguments of each call.
    workers : int
        Processes.
    work : Path
        Scratch directory for the memory maps.

    Returns
    -------
    list[Any]
        Results in task order.
    """
    from joblib import Parallel, delayed

    folder = work / "joblib"
    folder.mkdir(parents=True, exist_ok=True)
    return Parallel(n_jobs=workers, temp_folder=str(folder), max_nbytes="1M", verbose=0)(
        delayed(function)(*task) for task in tasks
    )


def flatten(result: dict[str, Any], kind: str) -> dict[str, Any]:
    """
    One table row of a ``binary_probe`` result.

    Parameters
    ----------
    result : dict[str, Any]
        Output of ``binary_probe``.
    kind : str
        Factor kind.

    Returns
    -------
    dict[str, Any]
        The result with ``by_source`` spread into ``auroc_<source>`` columns.
    """
    row = {key: value for key, value in result.items() if key != "by_source"}
    row.update({f"auroc_{source}": value for source, value in result["by_source"].items()})
    return {"kind": kind, **row}


def integrity(args: argparse.Namespace) -> None:
    """
    Check the inputs and reproduce P0's validation selection metric before any analysis.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Raises
    ------
    ValueError
        If a feature file differs from its receipt or the P0 metric or label digest is not reproduced.
    """
    started = time.time()
    output, _ = paths(args)
    feature_receipt = json.loads((key06.FEATURES / "receipt.json").read_text())
    files = {name: sha256_file(key06.FEATURES / name) for name in FEATURE_FILES}
    mismatched = [name for name, digest in files.items() if digest != feature_receipt["files_sha256"][name]]
    if mismatched:
        raise ValueError(f"feature files differ from their receipt: {mismatched}")
    records = key06.load_records()
    validation = records[records["role"] == "validation"].reset_index(drop=True)
    labels = key06.label_matrix(validation)
    prediction = load_predictions(key06.P0_VALIDATION, validation["record_key"].to_numpy())
    metric = selection_metric(labels, prediction["probability"], validation["source"].to_numpy())
    record = json.loads(key06.SELECTION_RECORD.read_text())
    stored_metric = next(c["selection_metric"] for c in record["candidates"] if c["id"] == "P0")
    checks = {
        "p0_selection_metric": metric["selection_metric"],
        "p0_selection_metric_stored": stored_metric,
        "p0_equal": metric["selection_metric"] == stored_metric,
        "validation_labels_digest_equal": labels_digest(labels) == record["validation"]["labels_sha256"],
        "feature_files_equal_receipt": True,
    }
    statements, names = key06.statement_matrix(records)
    train = (records["role"] == "train").to_numpy()
    key06.load_flags(records)
    content = {
        "checks": checks,
        "records": records.groupby(["role", "source"]).size().unstack().to_dict(),
        "statements_outside": len(names),
        "statements_at_least_30_train": int(
            (np.asarray(statements[train].sum(axis=0)).ravel() >= key06.STATEMENT_MIN_TEST).sum()
        ),
        "feature_sha256": files,
    }
    write_json_atomic(output / "integrity" / "integrity.json", content)
    receipt(
        args,
        "integrity",
        started,
        content,
        [key06.MANIFEST, key06.FLAGS_TABLE, key06.P0_VALIDATION, key06.SELECTION_RECORD],
    )
    if not (checks["p0_equal"] and checks["validation_labels_digest_equal"]):
        raise ValueError(f"integrity check failed: {checks}")
    print(json.dumps(checks, indent=1))


def reverse_tasks(data: dict[str, Any], work: Path) -> list[tuple]:
    """
    Binary probe tasks of the reverse direction: the 14 labels, the reliable flags and sex.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    work : Path
        Scratch directory for the feature arrays.

    Returns
    -------
    list[tuple]
        ``(kind, binary_probe arguments)`` per task.
    """
    train, records = data["train"], data["records"]
    x_train, x_validation = (
        array_file(work, "xecg_train", data["z_train"]),
        array_file(work, "xecg_validation", data["z_validation"]),
    )
    strata = records["source"].to_numpy()[~train]
    groups = records["group"].to_numpy()[~train]
    targets = [("label", name, data["labels"][:, index]) for index, name in enumerate(CONDITIONS)]
    for flag in key06.RELIABLE_FLAGS:
        values = data["flags"][flag].to_numpy(dtype=float)
        if values[train].sum() >= 20:
            targets.append(("flag", flag, values))
    sex = np.where(records["sex"] == "M", 1.0, np.where(records["sex"] == "F", 0.0, np.nan))
    targets.append(("sex", "male", sex))
    return [
        (kind, (name, x_train, values[train], x_validation, values[~train], strata, groups))
        for kind, name, values in targets
    ]


def array_file(work: Path, name: str, values: np.ndarray) -> str:
    """
    Write a float64 array to scratch for the probe workers to memory-map.

    Parameters
    ----------
    work : Path
        Scratch directory.
    name : str
        File stem.
    values : np.ndarray
        Array.

    Returns
    -------
    str
        Path of the ``.npy`` file.
    """
    path = work / "arrays" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64))
    return str(path)


def run_probe(kind: str, arguments: tuple) -> dict[str, Any]:
    """
    Run one binary probe and flatten its result.

    Parameters
    ----------
    kind : str
        Factor kind.
    arguments : tuple
        Arguments of ``binary_probe``; ``.npy`` paths stand for the feature arrays and are memory-mapped.

    Returns
    -------
    dict[str, Any]
        Output of ``flatten``.
    """
    arguments = tuple(
        np.load(value, mmap_mode="r") if isinstance(value, str) and value.endswith(".npy") else value
        for value in arguments
    )
    with threadpool_limits(1):
        return flatten(key06.binary_probe(*arguments), kind)


def variance(args: argparse.Namespace) -> None:
    """
    Step 1: variance accounting in both directions.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, work = paths(args)
    directory = output / "variance"
    directory.mkdir(parents=True, exist_ok=True)
    data = bundle(args)
    blocks = blocks_of(data)
    pca = key06.pca_fit(data["z_train"])
    table, components = key06.variance_accounting(blocks, data["z_train"], data["z_validation"], pca, 20)
    table.to_csv(directory / "variance_accounting.csv", index=False)
    components.to_csv(directory / "components.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(1, len(pca["ratio"]) + 1),
            "share": pca["ratio"],
            "cumulative": np.cumsum(pca["ratio"]),
        }
    ).to_csv(directory / "pca_variance.csv", index=False)
    print(table.to_string(), flush=True)
    train, records = data["train"], data["records"]
    source = key06.source_probe(
        data["z_train"],
        records["source"].to_numpy()[train],
        data["z_validation"],
        records["source"].to_numpy()[~train],
    )
    write_json_atomic(directory / "source_prediction.json", source)
    binary = pd.DataFrame(parallel(run_probe, reverse_tasks(data, work), args.workers, work))
    binary.to_csv(directory / "factor_prediction_binary.csv", index=False)
    names = data["names"]
    others = [index for index, name in enumerate(names) if name not in key06.RATE_COLUMNS]
    scores = key06.normal_scores(
        data["measurements"][train][:, others], data["measurements"][~train][:, others]
    )
    targets_train = np.column_stack(
        [data["measurements"][train][:, names.index("hr_bpm")], records["age"].to_numpy()[train], scores[0]]
    )
    targets_validation = np.column_stack(
        [data["measurements"][~train][:, names.index("hr_bpm")], records["age"].to_numpy()[~train], scores[1]]
    )
    continuous = key06.continuous_probes(
        ["hr_bpm", "age", *[names[index] for index in others]],
        data["z_train"],
        targets_train,
        data["z_validation"],
        targets_validation,
        records["source"].to_numpy()[~train],
    )
    continuous.to_csv(directory / "factor_prediction_continuous.csv", index=False)
    receipt(
        args,
        "variance",
        started,
        {
            "kept_components": pca["kept"],
            "source_prediction": source,
            "block_columns": {g: blocks[g][0].shape[1] for g in blocks},
        },
        [key06.MANIFEST, key06.FLAGS_TABLE, key06.FEATURES / "xecg.npy", key06.FEATURES / "measurements.npy"],
    )


def residual(args: argparse.Namespace) -> None:
    """
    Step 2: the residual space, written to scratch with its hashes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, work = paths(args)
    directory = output / "residual"
    directory.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    data = bundle(args)
    blocks = blocks_of(data)
    res_train, res_validation, coefficients = key06.residual_space(
        blocks, data["z_train"], data["z_validation"]
    )
    hashes = {}
    for name, values in (("train", res_train), ("validation", res_validation)):
        path = work / f"residual_{name}.npy"
        np.save(path, values.astype(np.float32))
        hashes[path.name] = sha256_file(path)
    write_json_atomic(work / "residual_hashes.json", hashes)
    write_npz_atomic(
        directory / "coefficients.npz",
        coefficients=coefficients,
        groups=np.array(key06.RESIDUAL_GROUPS),
        columns=np.array([blocks[group][0].shape[1] for group in key06.RESIDUAL_GROUPS]),
    )
    train, records = data["train"], data["records"]
    sources = records["source"].to_numpy()[~train]
    explained = key06.variance_r2(data["z_validation"], data["z_validation"] - res_validation)
    summary = {
        "variance_removed_validation": float(explained),
        "residual_variance_share_validation": float(1.0 - explained),
        "source_mean_norm_raw": {
            s: float(np.linalg.norm(data["z_validation"][sources == s].mean(axis=0)))
            for s in VALIDATION_SOURCES
        },
        "source_mean_norm_residual": {
            s: float(np.linalg.norm(res_validation[sources == s].mean(axis=0))) for s in VALIDATION_SOURCES
        },
    }
    scaled = key06.standardize(res_train, res_validation)
    summary["source_prediction_residual"] = key06.source_probe(
        scaled[0], records["source"].to_numpy()[train], scaled[1], sources
    )
    write_json_atomic(directory / "residual_summary.json", summary)
    print(json.dumps(summary, indent=1))
    receipt(
        args,
        "residual",
        started,
        {"residual_sha256": hashes, **summary},
        [key06.MANIFEST, key06.FLAGS_TABLE, key06.FEATURES / "xecg.npy", key06.FEATURES / "measurements.npy"],
    )


def subset_mask(data: dict[str, Any], subset: str) -> np.ndarray:
    """
    Select the records of one subset.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    subset : str
        ``all`` or ``label_negative`` (no positive among the defined labels).

    Returns
    -------
    np.ndarray
        Boolean mask over all records.
    """
    if subset == "all":
        return np.ones(len(data["records"]), dtype=bool)
    return ~(data["labels"] == 1.0).any(axis=1)


def prepare(args: argparse.Namespace) -> None:
    """
    Step 3, part 1: principal components of every subset and space, and the shared subsamples.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, work = paths(args)
    data = bundle(args)
    train, records = data["train"], data["records"]
    summary = {}
    for subset in key06.SUBSETS:
        mask = subset_mask(data, subset)
        indices = key06.subsample_indices(int((mask & train).sum()), settings(args)["subsamples"])
        for space in key06.SPACES:
            combo = f"{subset}__{space}"
            directory = work / "clusters" / combo
            directory.mkdir(parents=True, exist_ok=True)
            space_train, space_validation = embeddings(data, space, work)
            pca = key06.pca_fit(space_train[mask[train]])
            np.save(directory / "pcs_train.npy", key06.pca_project(space_train[mask[train]], pca))
            np.save(directory / "pcs_validation.npy", key06.pca_project(space_validation[mask[~train]], pca))
            np.save(directory / "subsamples.npy", indices)
            pd.DataFrame({"record_key": records["record_key"][mask & train]}).to_csv(
                directory / "keys_train.csv", index=False
            )
            pd.DataFrame({"record_key": records["record_key"][mask & ~train]}).to_csv(
                directory / "keys_validation.csv", index=False
            )
            summary[combo] = {
                "train": int((mask & train).sum()),
                "validation": int((mask & ~train).sum()),
                "kept_components": pca["kept"],
                "kept_share": float(np.cumsum(pca["ratio"])[pca["kept"] - 1]),
                "subsamples_sha256": sha256_file(directory / "subsamples.npy"),
            }
            (output / "clusters" / combo).mkdir(parents=True, exist_ok=True)
            write_json_atomic(output / "clusters" / combo / "pca.json", summary[combo])
    print(json.dumps(summary, indent=1))
    receipt(
        args,
        "prepare",
        started,
        {"combos": summary, "subsample_seed": key06.SEED + 7},
        [key06.MANIFEST, key06.FEATURES / "xecg.npy"],
    )


def cluster_tasks(work: Path, subsamples: int) -> list[dict[str, Any]]:
    """
    Every clustering fit of the study, the slowest first.

    Parameters
    ----------
    work : Path
        Scratch directory.
    subsamples : int
        Subsamples per method.

    Returns
    -------
    list[dict[str, Any]]
        ``directory``, ``method``, ``replicate`` and ``seed`` per fit.
    """
    tasks = [
        {
            "directory": str(work / "clusters" / f"{subset}__{space}"),
            "method": method,
            "replicate": replicate,
            "seed": key06.cluster_seed(method, replicate),
        }
        for subset in key06.SUBSETS
        for space in key06.SPACES
        for method in key06.METHODS
        for replicate in range(-1, subsamples)
    ]
    cost = {"hdbscan": 0, "kmeans_80": 1, "kmeans_40": 2, "kmeans_20": 3, "kmeans_10": 4}
    return sorted(
        tasks,
        key=lambda task: (cost[task["method"]], "label_negative" in task["directory"], task["replicate"]),
    )


def run_cluster_task(task: dict[str, Any]) -> tuple[str, float]:
    """
    Fit one clustering task unless its file exists.

    Parameters
    ----------
    task : dict[str, Any]
        Output of ``cluster_tasks``.

    Returns
    -------
    tuple[str, float]
        File name and seconds (0 when skipped).
    """
    directory = Path(task["directory"])
    path = directory / f"{task['method']}__rep{task['replicate']:+03d}.npz"
    if path.exists():
        return path.name, 0.0
    points = np.load(directory / "pcs_train.npy")
    if task["replicate"] >= 0:
        points = points[np.load(directory / "subsamples.npy")[task["replicate"]]]
    fitted, seconds = key06.timed(key06.fit_clusters, points, task["method"], task["seed"])
    centres = fitted["centres"] if fitted["centres"] is not None else np.zeros((0, points.shape[1]))
    write_npz_atomic(path, labels=fitted["labels"], centres=centres, seed=task["seed"], seconds=seconds)
    return path.name, seconds


def cluster(args: argparse.Namespace) -> None:
    """
    Step 3, part 2: every full and subsample fit, in parallel and resumable.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    _, work = paths(args)
    tasks = cluster_tasks(work, settings(args)["subsamples"])
    log = work / "cluster.log"
    total = 0.0
    with Pool(args.workers) as pool:
        for done, (name, seconds) in enumerate(pool.imap_unordered(run_cluster_task, tasks), start=1):
            total += seconds
            with open(log, "a") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {done}/{len(tasks)} {name} {seconds:.1f}s\n")
    receipt(
        args,
        "cluster",
        started,
        {
            "tasks": len(tasks),
            "fit_seconds": total,
            "seeds": {
                method: [key06.cluster_seed(method, r) for r in range(-1, settings(args)["subsamples"])]
                for method in key06.METHODS
            },
        },
        [],
    )


def assemble_one(work: Path, output: Path, combo: str, method: str, subsamples: int) -> dict[str, Any]:
    """
    Renumber one full solution, measure its stability and assign the validation records.

    Parameters
    ----------
    work, output : Path
        Scratch and output directories.
    combo : str
        ``<subset>__<space>``.
    method : str
        One of ``METHODS``.
    subsamples : int
        Subsamples fitted.

    Returns
    -------
    dict[str, Any]
        Clusters, noise share and median stabilities.
    """
    directory = work / "clusters" / combo
    full = np.load(directory / f"{method}__rep-01.npz")
    labels, order = key06.order_by_size(full["labels"])
    centres = full["centres"][order] if method != "hdbscan" else None
    points_train, points_validation = (
        np.load(directory / "pcs_train.npy"),
        np.load(directory / "pcs_validation.npy"),
    )
    indices = np.load(directory / "subsamples.npy")
    jaccard = np.stack(
        [
            key06.jaccard_stability(
                labels, indices[r], np.load(directory / f"{method}__rep{r:+03d}.npz")["labels"]
            )
            for r in range(subsamples)
        ]
    )
    assigned = key06.assign_validation(points_train, labels, centres, points_validation)
    clusters = int(labels.max()) + 1
    means = np.stack(
        [points_train[labels == c].mean(axis=0) for c in range(clusters)]
        or [np.full(points_train.shape[1], np.nan)]
    )
    keys = {
        name: pd.read_csv(directory / f"keys_{name}.csv")["record_key"] for name in ("train", "validation")
    }
    frames = []
    for role, points, values in (
        ("train", points_train, labels),
        ("validation", points_validation, assigned),
    ):
        distance = np.where(
            values >= 0, np.linalg.norm(points - means[np.maximum(values, 0)], axis=1), np.nan
        )
        frames.append(
            pd.DataFrame({"record_key": keys[role], "role": role, "cluster": values, "distance": distance})
        )
    target = output / "clusters" / combo / method
    target.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(target / "assignment.parquet", index=False)
    stability = pd.DataFrame(
        {
            "cluster": np.arange(clusters),
            "size_train": np.bincount(labels[labels >= 0], minlength=clusters),
            "size_validation": np.bincount(assigned[assigned >= 0], minlength=clusters),
            "median_jaccard": np.median(jaccard, axis=0),
            "min_jaccard": jaccard.min(axis=0),
            "max_jaccard": jaccard.max(axis=0),
            **{f"jaccard_{r:02d}": jaccard[r] for r in range(subsamples)},
        }
    )
    stability.to_csv(target / "stability.csv", index=False)
    return {
        "clusters": clusters,
        "noise_train": float((labels < 0).mean()),
        "noise_validation": float((assigned < 0).mean()),
        "stable": int((stability["median_jaccard"] >= key06.STABLE_JACCARD).sum()),
    }


def assemble(args: argparse.Namespace) -> None:
    """
    Step 3, part 3: stability and validation assignment of every solution.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, work = paths(args)
    summary = {}
    for subset in key06.SUBSETS:
        for space in key06.SPACES:
            for method in key06.METHODS:
                combo = f"{subset}__{space}"
                summary[f"{combo}/{method}"] = assemble_one(
                    work, output, combo, method, settings(args)["subsamples"]
                )
                print(f"{combo}/{method}", summary[f"{combo}/{method}"], flush=True)
    write_json_atomic(output / "clusters" / "summary.json", summary)
    receipt(args, "assemble", started, {"solutions": summary}, [])


def solutions() -> list[tuple[str, str, str]]:
    """
    Every (subset, space, method) of the study, in a fixed order.

    Returns
    -------
    list[tuple[str, str, str]]
        Subset, space and method.
    """
    return [
        (subset, space, method)
        for subset in key06.SUBSETS
        for space in key06.SPACES
        for method in key06.METHODS
    ]


def membership(records: pd.DataFrame, assignment: pd.DataFrame, role: str) -> np.ndarray:
    """
    Cluster of every record of one role, -2 for records outside the solution's subset.

    Parameters
    ----------
    records : pd.DataFrame
        Records of the run.
    assignment : pd.DataFrame
        ``assignment.parquet`` of one solution.
    role : str
        ``train`` or ``validation``.

    Returns
    -------
    np.ndarray
        One value per record of that role, in records order.
    """
    keys = records.loc[records["role"] == role, "record_key"]
    lookup = assignment[assignment["role"] == role].set_index("record_key")["cluster"]
    return keys.map(lookup).fillna(-2).astype(int).to_numpy()


def all_scores(data: dict[str, Any]) -> np.ndarray:
    """
    Map the 130 measurements to normal scores by train quantiles, aligned with the records.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.

    Returns
    -------
    np.ndarray
        ``[records, 130]`` with NaN for missing.
    """
    train = data["train"]
    scores = np.full(data["measurements"].shape, np.nan)
    scores[train], scores[~train] = key06.normal_scores(
        data["measurements"][train], data["measurements"][~train]
    )
    return scores


def describe_role(
    data: dict[str, Any],
    scores: np.ndarray,
    statements: Any,
    tested: np.ndarray,
    statement_names: list[str],
    member: np.ndarray,
    role: str,
    clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Profile, enrichment and measurement differences of one solution on the records of one role.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    scores : np.ndarray
        Output of ``all_scores``.
    statements : Any
        Statement matrix of all records.
    tested : np.ndarray
        Statement columns tested in this subset.
    statement_names : list[str]
        Names of all statement columns.
    member : np.ndarray
        Output of ``membership`` for the role (-2 outside the subset).
    role : str
        ``train`` or ``validation``.
    clusters : int
        Clusters of the solution.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Cluster profiles, enrichment rows and measurement differences.
    """
    rows = np.flatnonzero((data["records"]["role"] == role).to_numpy())[member != -2]
    values = member[member != -2]
    records = data["records"].iloc[rows].reset_index(drop=True)
    labels = data["labels"][rows]
    strata = key06.stratum_codes(records["source"].to_numpy())
    profile = key06.cluster_profile(
        values, records, labels, data["flags"].iloc[rows], data["measurements"][rows], data["names"], clusters
    )
    label_rows = key06.enrichment_rows(
        key06.enrichment_tables(values, labels == 1.0, strata, np.isfinite(labels)), list(CONDITIONS), "label"
    )
    statement_rows = key06.enrichment_rows(
        key06.enrichment_tables(values, statements[rows][:, tested], strata),
        [statement_names[index] for index in tested],
        "statement",
    )
    differences = key06.within_source_differences(values, scores[rows], strata, clusters)
    difference_rows = pd.DataFrame(
        {
            "cluster": np.repeat(np.arange(clusters), len(data["names"])),
            "measurement": np.tile(data["names"], clusters),
            "smd": differences.ravel(),
        }
    )
    return profile, pd.concat([label_rows, statement_rows], ignore_index=True), difference_rows


def describe_pass(
    args: argparse.Namespace, data: dict[str, Any], statements: Any, statement_names: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Profiles, enrichment and differences of every solution on train and on validation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    data : dict[str, Any]
        Output of ``bundle``.
    statements : Any
        Statement matrix.
    statement_names : list[str]
        Statement names.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``profiles``, ``enrichment`` (train rows with ``q_value``) and ``differences``.
    """
    output, _ = paths(args)
    scores = all_scores(data)
    train = data["train"]
    collected: dict[str, list[pd.DataFrame]] = {"profiles": [], "enrichment": [], "differences": []}
    for subset, space, method in solutions():
        solution = f"{subset}/{space}/{method}"
        assignment = pd.read_parquet(
            output / "clusters" / f"{subset}__{space}" / method / "assignment.parquet"
        )
        clusters = int(assignment.loc[assignment["role"] == "train", "cluster"].max()) + 1
        in_subset = subset_mask(data, subset)
        tested = np.flatnonzero(
            np.asarray(statements[in_subset & train].sum(axis=0)).ravel() >= key06.STATEMENT_MIN_TEST
        )
        for role in key06.OPEN_ROLES:
            frames = describe_role(
                data,
                scores,
                statements,
                tested,
                statement_names,
                membership(data["records"], assignment, role),
                role,
                clusters,
            )
            for name, frame in zip(collected, frames, strict=True):
                collected[name].append(frame.assign(solution=solution, role=role))
        print(f"described {solution}", flush=True)
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in collected.items()}
    enrichment = tables["enrichment"]
    enrichment["q_value"] = np.nan
    for kind in ("label", "statement"):
        rows = (enrichment["role"] == "train") & (enrichment["kind"] == kind)
        enrichment.loc[rows, "q_value"] = key06.benjamini_hochberg(enrichment.loc[rows, "p_value"].to_numpy())
    return tables


def top_text(
    rows: pd.DataFrame, validation: pd.DataFrame, descriptions: dict[str, str], count: int = 5
) -> str:
    """
    List the most enriched significant statements of one cluster, with their validation odds ratios.

    Parameters
    ----------
    rows : pd.DataFrame
        Train statement rows of the cluster.
    validation : pd.DataFrame
        Validation statement rows of the cluster.
    descriptions : dict[str, str]
        Statement descriptions.
    count : int
        Statements listed.

    Returns
    -------
    str
        ``name (description) OR train / validation [members with it/members]`` items joined by ``; ``.
    """
    significant = rows[(rows["q_value"] < key06.BH_LEVEL) & (rows["log_odds_ratio"] > 0)]
    significant = significant.sort_values(["log_odds_ratio", "feature"], ascending=[False, True]).head(count)
    lookup = validation.set_index("feature")["odds_ratio"]
    return "; ".join(
        f"{row.feature} ({descriptions.get(row.feature, '')[:40]}) OR {row.odds_ratio:.1f}/"
        f"{lookup.get(row.feature, np.nan):.1f} [{int(row.a)}/{int(row.members)}]"
        for row in significant.itertuples()
    )


def replication_feature(
    data: dict[str, Any], statements: Any, names: list[str], kind: str, feature: str, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Values and defined mask of one label or statement on some records.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    statements : Any
        Statement matrix.
    names : list[str]
        Statement names.
    kind : str
        ``label`` or ``statement``.
    feature : str
        Condition or statement name.
    rows : np.ndarray
        Record rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean values and defined mask.
    """
    if kind == "label":
        values = data["labels"][rows, list(CONDITIONS).index(feature)]
        return values == 1.0, np.isfinite(values)
    column = np.asarray(statements[rows][:, names.index(feature)].todense()).ravel() > 0
    return column, np.ones(len(rows), dtype=bool)


def rules_for(
    args: argparse.Namespace,
    data: dict[str, Any],
    statements: Any,
    names: list[str],
    tables: dict[str, pd.DataFrame],
    descriptions: dict[str, str],
) -> pd.DataFrame:
    """
    Apply the five rules, with the strongest enrichment and its replication, to every cluster.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    data : dict[str, Any]
        Output of ``bundle``.
    statements : Any
        Statement matrix.
    names : list[str]
        Statement names.
    tables : dict[str, pd.DataFrame]
        Output of ``describe_pass``.
    descriptions : dict[str, str]
        Statement descriptions.

    Returns
    -------
    pd.DataFrame
        One row per cluster with the train profile, validation size and every rule.
    """
    output, _ = paths(args)
    records, train = data["records"], data["train"]
    enrichment, profiles, differences = tables["enrichment"], tables["profiles"], tables["differences"]
    weights = {}
    rows = []
    for subset, space, method in solutions():
        solution = f"{subset}/{space}/{method}"
        combo = f"{subset}__{space}"
        stability = pd.read_csv(output / "clusters" / combo / method / "stability.csv").set_index("cluster")
        assignment = pd.read_parquet(output / "clusters" / combo / method / "assignment.parquet")
        validation_member = membership(records, assignment, "validation")
        validation_rows = np.flatnonzero(~train)[validation_member != -2]
        validation_member = validation_member[validation_member != -2]
        strata = key06.stratum_codes(records["source"].to_numpy()[validation_rows])
        if subset not in weights:
            weights[subset] = key06.bootstrap_weights(
                records["group"].to_numpy()[validation_rows],
                settings(args)["draws"],
                key06.SEED + 11 + key06.SUBSETS.index(subset),
            ).astype(float)
        here = enrichment[enrichment["solution"] == solution]
        profile = profiles[(profiles["solution"] == solution) & (profiles["role"] == "train")].set_index(
            "cluster"
        )
        validation_profile = profiles[
            (profiles["solution"] == solution) & (profiles["role"] == "validation")
        ].set_index("cluster")
        smd = differences[(differences["solution"] == solution) & (differences["role"] == "train")]
        for cluster in stability.index:
            row = {
                "solution": solution,
                "subset": subset,
                "space": space,
                "method": method,
                "cluster": cluster,
                **profile.loc[cluster].drop(["solution", "role"]).to_dict(),
                "size_validation": int(validation_profile.loc[cluster, "size"]),
                "validation_top_label_share": validation_profile.loc[cluster].get("top_label_share", np.nan),
                "median_jaccard": float(stability.loc[cluster, "median_jaccard"]),
            }
            row["rule_stable"] = row["median_jaccard"] >= key06.STABLE_JACCARD
            train_rows = here[(here["role"] == "train") & (here["cluster"] == cluster)]
            strongest = key06.strongest_enrichment(train_rows)
            row.update(
                strongest_columns(
                    data,
                    statements,
                    names,
                    strongest,
                    validation_member == cluster,
                    validation_rows,
                    strata,
                    weights[subset],
                )
            )
            validation_rows_cluster = here[
                (here["role"] == "validation") & (here["cluster"] == cluster) & (here["kind"] == "statement")
            ]
            row["top_statements"] = top_text(
                train_rows[train_rows["kind"] == "statement"], validation_rows_cluster, descriptions
            )
            part = smd[smd["cluster"] == cluster].dropna(subset=["smd"])
            top = part.loc[part["smd"].abs().sort_values(ascending=False, kind="stable").index].head(5)
            row["top_measurements"] = "; ".join(f"{r.measurement} {r.smd:+.2f}" for r in top.itertuples())
            row["candidate"], row["failed_rules"] = key06.verdict(pd.Series(row))
            rows.append(row)
        print(f"rules {solution}", flush=True)
    return pd.DataFrame(rows)


def strongest_columns(
    data: dict[str, Any],
    statements: Any,
    names: list[str],
    strongest: pd.Series | None,
    member: np.ndarray,
    rows: np.ndarray,
    strata: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """
    Columns of the strongest train enrichment and of its validation replication.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    statements : Any
        Statement matrix.
    names : list[str]
        Statement names.
    strongest : pd.Series | None
        Output of ``strongest_enrichment``.
    member : np.ndarray
        Validation membership of the cluster (subset records).
    rows : np.ndarray
        Record rows of the subset's validation records.
    strata : np.ndarray
        Their stratum codes.
    weights : np.ndarray
        Bootstrap weights of those records.

    Returns
    -------
    dict[str, Any]
        ``strongest_kind``, ``strongest_feature``, train and validation log odds ratios, interval and
        ``rule_replicated``.
    """
    if strongest is None:
        return {
            "strongest_kind": "",
            "strongest_feature": "",
            "strongest_log_odds_train": np.nan,
            "strongest_q": np.nan,
            "strongest_log_odds_validation": np.nan,
            "strongest_ci_low": np.nan,
            "strongest_ci_high": np.nan,
            "rule_replicated": False,
        }
    values, defined = replication_feature(
        data, statements, names, strongest["kind"], strongest["feature"], rows
    )
    check = key06.replication(weights, member, values, strata, defined, float(strongest["log_odds_ratio"]))
    return {
        "strongest_kind": strongest["kind"],
        "strongest_feature": strongest["feature"],
        "strongest_log_odds_train": float(strongest["log_odds_ratio"]),
        "strongest_q": float(strongest["q_value"]),
        "strongest_members_with": int(strongest["a"]),
        "strongest_log_odds_validation": check["log_odds_validation"],
        "strongest_ci_low": check["ci_low"],
        "strongest_ci_high": check["ci_high"],
        "rule_replicated": check["replicated"],
    }


def describe(args: argparse.Namespace) -> None:
    """
    Step 4: descriptions, enrichment, the five rules and the candidate list.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, _ = paths(args)
    directory = output / "describe"
    directory.mkdir(parents=True, exist_ok=True)
    data = bundle(args)
    statements, names = key06.statement_matrix(data["records"])
    descriptions = key06.statement_descriptions(names)
    pd.DataFrame(
        {
            "statement": names,
            "description": [descriptions[name] for name in names],
            "mapping": [key06.statement_mapping_note(name) for name in names],
        }
    ).to_csv(directory / "statements.csv", index=False)
    tables = describe_pass(args, data, statements, names)
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    clusters = rules_for(args, data, statements, names, tables, descriptions)
    clusters.to_csv(directory / "clusters.csv", index=False)
    clusters[clusters["candidate"]].to_csv(directory / "candidates.csv", index=False)
    counts = {
        "clusters": len(clusters),
        "candidates": int(clusters["candidate"].sum()),
        **{
            column: int(clusters[column].sum())
            for column in (
                "rule_stable",
                "rule_source",
                "rule_replicated",
                "rule_artefact",
                "rule_artefact_min5",
                "rule_labelled",
            )
        },
    }
    print(json.dumps(counts, indent=1))
    receipt(
        args,
        "describe",
        started,
        {
            "counts": counts,
            "replication_seeds": {
                subset: key06.SEED + 11 + index for index, subset in enumerate(key06.SUBSETS)
            },
        },
        [key06.MANIFEST, key06.FLAGS_TABLE, key06.FEATURES / "measurements.npy"],
    )


def flags_step(args: argparse.Namespace) -> None:
    """
    Write the train flag rates of every cluster to ``describe/flag_rates.csv``.

    They are the inputs of the artefact rule, kept so the report can name every failing flag.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, _ = paths(args)
    records = key06.load_records()
    if args.smoke:
        records = smoke_sample(records)
    flags = key06.load_flags(records)
    data = {
        "records": records,
        "train": (records["role"] == "train").to_numpy(),
        "labels": key06.label_matrix(records),
    }
    frames = []
    for subset, space, method in solutions():
        combo = output / "clusters" / f"{subset}__{space}"
        assignment = pd.read_parquet(combo / method / "assignment.parquet")
        member = membership(records, assignment, "train")
        rows = np.flatnonzero(data["train"] & subset_mask(data, subset))
        values = member[member != -2]
        subset_flags = flags.iloc[rows].reset_index(drop=True)
        for cluster in range(int(values.max()) + 1):
            frames.append(
                key06.flag_rates(values == cluster, subset_flags).assign(
                    solution=f"{subset}/{space}/{method}", cluster=cluster
                )
            )
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(output / "describe" / "flag_rates.csv", index=False)
    print(f"{table[['solution', 'cluster']].drop_duplicates().shape[0]} clusters")
    receipt(args, "flags", started, {"rows": len(table)}, [key06.MANIFEST, key06.FLAGS_TABLE])


def captions_for(data: dict[str, Any], rows: np.ndarray) -> list[str]:
    """
    Sheet captions of some records.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``bundle``.
    rows : np.ndarray
        Record rows.

    Returns
    -------
    list[str]
        One ``key06.caption`` per record.
    """
    names = data["names"]
    columns = [names.index("hr_bpm"), names.index("qrs_dur_ms")]
    return [
        key06.caption(
            data["records"].iloc[row],
            dict(zip(("hr_bpm", "qrs_dur_ms"), data["measurements"][row, columns], strict=True)),
            data["flags"].iloc[row],
        )
        for row in rows
    ]


def review_solution(args: argparse.Namespace, data: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """
    Review sheets of every cluster of one solution: the 24 nearest its centre and 24 random members.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    data : dict[str, Any]
        Output of ``bundle``.
    index : int
        Position of the solution in ``solutions``.

    Returns
    -------
    list[dict[str, Any]]
        One row per plotted record: solution, cluster, sheet, kind, order and record key.
    """
    output, _ = paths(args)
    subset, space, method = solutions()[index]
    combo = f"{subset}__{space}"
    assignment = pd.read_parquet(output / "clusters" / combo / method / "assignment.parquet")
    train = assignment[assignment["role"] == "train"].reset_index(drop=True)
    position = pd.Series(np.arange(len(data["records"])), index=data["records"]["record_key"])
    rows = []
    for cluster in range(int(train["cluster"].max()) + 1):
        members = train[train["cluster"] == cluster]
        nearest = members.sort_values(["distance", "record_key"]).head(key06.REVIEW_NEAREST)["record_key"]
        rng = np.random.default_rng(key06.SEED + 100000 * (index + 1) + cluster)
        drawn = members["record_key"].to_numpy()[
            rng.choice(len(members), size=min(key06.REVIEW_RANDOM, len(members)), replace=False)
        ]
        for kind, keys in (("nearest", nearest.to_numpy()), ("random", drawn)):
            record_rows = position[keys].to_numpy()
            sheet = output / "review" / combo / method / f"c{cluster:02d}_{kind}.png"
            title = (
                f"KEY-06 {subset}/{space}/{method} cluster {cluster}: {len(members)} train members; {kind} "
                f"{len(keys)} (median beats of these records, then lead II, 10 s)"
            )
            key06.draw_cluster_sheet(
                key06.read_windows(data["records"].iloc[record_rows]),
                captions_for(data, record_rows),
                sheet,
                title,
            )
            rows.extend(
                {
                    "solution": f"{subset}/{space}/{method}",
                    "cluster": cluster,
                    "sheet": stored(sheet),
                    "kind": kind,
                    "order": order,
                    "record_key": key,
                }
                for order, key in enumerate(keys)
            )
    return rows


def review_worker(arguments: tuple[argparse.Namespace, int]) -> list[dict[str, Any]]:
    """
    Draw one solution's sheets in a worker process.

    Parameters
    ----------
    arguments : tuple[argparse.Namespace, int]
        Parsed arguments and the solution's position.

    Returns
    -------
    list[dict[str, Any]]
        Output of ``review_solution``.
    """
    args, index = arguments
    with threadpool_limits(1):
        return review_solution(args, WORKER_DATA, index)


WORKER_DATA: dict[str, Any] = {}


def review(args: argparse.Namespace) -> None:
    """
    Step 4, sheets: every cluster of every solution, the 24 nearest its centre and 24 random members.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, _ = paths(args)
    WORKER_DATA.update(bundle(args))
    order = sorted(
        range(len(solutions())),
        key=lambda i: -int(solutions()[i][2].split("_")[-1]) if solutions()[i][2] != "hdbscan" else 0,
    )
    rows = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(review_worker, [(args, index) for index in order]):
            rows.extend(part)
            print(f"sheets {part[0]['solution'] if part else '-'}", flush=True)
    sample = pd.DataFrame(rows).sort_values(["solution", "cluster", "kind", "order"])
    sample.to_csv(output / "review" / "sample.csv", index=False)
    receipt(
        args,
        "review",
        started,
        {
            "sheets": int(sample["sheet"].nunique()),
            "records": len(sample),
            "random_seed_rule": "SEED + 100000 * (solution position + 1) + cluster",
        },
        [key06.MANIFEST, key06.FLAGS_TABLE, CACHE / "index_all.csv"],
    )


def detail(args: argparse.Namespace) -> None:
    """
    Twelve-lead sheets (six records each) of the records nearest one cluster's centre.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``solution`` (``subset/space/method``), ``cluster`` and ``count``.
    """
    output, _ = paths(args)
    subset, space, method = args.solution.split("/")
    data = bundle(args)
    assignment = pd.read_parquet(output / "clusters" / f"{subset}__{space}" / method / "assignment.parquet")
    members = assignment[(assignment["role"] == "train") & (assignment["cluster"] == args.cluster)]
    keys = members.sort_values(["distance", "record_key"]).head(args.count)["record_key"].to_numpy()
    position = pd.Series(np.arange(len(data["records"])), index=data["records"]["record_key"])
    rows = position[keys].to_numpy()
    windows = key06.read_windows(data["records"].iloc[rows])
    records = [
        {"window": window, "title": text}
        for window, text in zip(windows, captions_for(data, rows), strict=True)
    ]
    for sheet in range(0, len(records), 6):
        path = (
            output
            / "review"
            / f"{subset}__{space}"
            / method
            / f"c{args.cluster:02d}_detail_{sheet // 6 + 1}.png"
        )
        draw_sheet(
            records[sheet : sheet + 6],
            path,
            f"KEY-06 {args.solution} cluster {args.cluster}: twelve leads of "
            f"the records nearest the centre, {sheet + 1}-{min(sheet + 6, len(records))}",
        )
        print(stored(path))


def cross_encoder(args: argparse.Namespace) -> None:
    """
    Step 5: membership of candidates (and of stable multi-source clusters) predicted from ECG-CPC.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, work = paths(args)
    data = bundle(args)
    records, train = data["records"], data["train"]
    cpc = key06.feature_rows("cpc", records["position"].to_numpy())
    cpc_train, cpc_validation = key06.standardize(cpc[train], cpc[~train])
    files = {
        "cpc": (array_file(work, "cpc_train", cpc_train), array_file(work, "cpc_validation", cpc_validation)),
        "xecg": (
            array_file(work, "xecg_train", data["z_train"]),
            array_file(work, "xecg_validation", data["z_validation"]),
        ),
    }
    clusters = pd.read_csv(output / "describe" / "clusters.csv")
    chosen = clusters[clusters["candidate"] | (clusters["rule_stable"] & clusters["rule_source"])]
    tasks, labels = [], []
    for row in chosen.itertuples():
        assignment = pd.read_parquet(
            output / "clusters" / f"{row.subset}__{row.space}" / row.method / "assignment.parquet"
        )
        member = {role: membership(records, assignment, role) for role in key06.OPEN_ROLES}
        y_train = np.where(member["train"] == -2, np.nan, (member["train"] == row.cluster).astype(float))
        y_validation = np.where(
            member["validation"] == -2, np.nan, (member["validation"] == row.cluster).astype(float)
        )
        strata = records["source"].to_numpy()[~train]
        groups = records["group"].to_numpy()[~train]
        for encoder, (x_train, x_validation) in files.items():
            tasks.append(
                (
                    "membership",
                    (
                        f"{row.solution}/c{row.cluster:02d}",
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                        strata,
                        groups,
                        settings(args)["draws"],
                        key06.SEED + 21,
                    ),
                )
            )
            labels.append(
                {
                    "solution": row.solution,
                    "cluster": row.cluster,
                    "encoder": encoder,
                    "candidate": row.candidate,
                }
            )
    results = parallel(run_probe, tasks, args.workers, work)
    table = pd.DataFrame([{**label, **result} for label, result in zip(labels, results, strict=True)])
    (output / "cross_encoder").mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "cross_encoder" / "cross_encoder.csv", index=False)
    receipt(
        args,
        "cross_encoder",
        started,
        {"clusters": len(chosen), "probes": len(table), "bootstrap_seed": key06.SEED + 21},
        [key06.FEATURES / "cpc.npy", key06.FEATURES / "xecg.npy"],
    )


def outlier_space(
    args: argparse.Namespace, data: dict[str, Any], space: str, statements: Any, names: list[str]
) -> dict[str, Any]:
    """
    Outliers of one space: scores, description, enrichment and twelve-lead sheets of a fixed random 48.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    data : dict[str, Any]
        Output of ``bundle``.
    space : str
        ``raw`` or ``residual``.
    statements : Any
        Statement matrix.
    names : list[str]
        Statement names.

    Returns
    -------
    dict[str, Any]
        Counts for the receipt.
    """
    output, work = paths(args)
    directory = output / "outliers" / space
    directory.mkdir(parents=True, exist_ok=True)
    records, train = data["records"], data["train"]
    rows = np.flatnonzero(train)
    points = np.load(work / "clusters" / f"all__{space}" / "pcs_train.npy")
    distance = key06.knn_distance(points, jobs=args.workers)
    count = int(round(key06.OUTLIER_SHARE * len(points)))
    chosen = np.argsort(-distance, kind="stable")[:count]
    member = np.full(len(points), -1)
    member[chosen] = 0
    subset = records.iloc[rows].reset_index(drop=True)
    pd.DataFrame(
        {
            "record_key": subset["record_key"],
            "source": subset["source"],
            "distance": distance,
            "outlier": member == 0,
        }
    ).to_parquet(directory / "scores.parquet", index=False)
    profile = key06.cluster_profile(
        member,
        subset,
        data["labels"][rows],
        data["flags"].iloc[rows],
        data["measurements"][rows],
        data["names"],
        1,
    )
    profile.to_csv(directory / "profile.csv", index=False)
    key06.flag_rates(member == 0, data["flags"].iloc[rows].reset_index(drop=True)).to_csv(
        directory / "flags.csv", index=False
    )
    strata = key06.stratum_codes(subset["source"].to_numpy())
    tested = np.flatnonzero(np.asarray(statements[rows].sum(axis=0)).ravel() >= key06.STATEMENT_MIN_TEST)
    enrichment = pd.concat(
        [
            key06.enrichment_rows(
                key06.enrichment_tables(
                    member, data["labels"][rows] == 1.0, strata, np.isfinite(data["labels"][rows])
                ),
                list(CONDITIONS),
                "label",
            ),
            key06.enrichment_rows(
                key06.enrichment_tables(member, statements[rows][:, tested], strata),
                [names[index] for index in tested],
                "statement",
            ),
        ],
        ignore_index=True,
    )
    enrichment = enrichment[enrichment["cluster"] == 0].copy()
    for kind in ("label", "statement"):
        part = enrichment["kind"] == kind
        enrichment.loc[part, "q_value"] = key06.benjamini_hochberg(enrichment.loc[part, "p_value"].to_numpy())
    enrichment.to_csv(directory / "enrichment.csv", index=False)
    rng = np.random.default_rng(key06.SEED + 60 + key06.SPACES.index(space))
    plotted = np.sort(rng.choice(chosen, size=min(key06.OUTLIER_PLOTS, count), replace=False))
    windows = key06.read_windows(subset.iloc[plotted])
    texts = captions_for(data, rows[plotted])
    sheets = [
        {"window": window, "title": f"{text} | kNN {distance[index]:.1f}"}
        for window, text, index in zip(windows, texts, plotted, strict=True)
    ]
    for start in range(0, len(sheets), 6):
        draw_sheet(
            sheets[start : start + 6],
            directory / f"sheet_{start // 6 + 1}.png",
            f"KEY-06 outliers, {space} space: fixed random {len(sheets)} of the {count} train records "
            f"farthest from their 20 nearest neighbours ({start + 1}-{min(start + 6, len(sheets))})",
        )
    pd.DataFrame(
        {"record_key": subset["record_key"].to_numpy()[plotted], "distance": distance[plotted]}
    ).to_csv(directory / "plotted.csv", index=False)
    return {
        "outliers": count,
        "threshold": float(distance[chosen[-1]]),
        "plotted": len(plotted),
        "plot_seed": key06.SEED + 60 + key06.SPACES.index(space),
    }


def outliers(args: argparse.Namespace) -> None:
    """
    Step 6: the 1% of train records farthest from their 20 nearest neighbours, in each space.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    data = bundle(args)
    statements, names = key06.statement_matrix(data["records"])
    summary = {space: outlier_space(args, data, space, statements, names) for space in key06.SPACES}
    print(json.dumps(summary, indent=1))
    receipt(args, "outliers", started, {"spaces": summary}, [key06.MANIFEST, key06.FLAGS_TABLE])


def statement_tasks(
    args: argparse.Namespace,
    data: dict[str, Any],
    statements: Any,
    presence: pd.DataFrame,
    space: str,
    train_x: str,
    validation_x: str,
) -> list[tuple]:
    """
    Probe tasks of step 7 in one space: listed statements and, for reference, the 14 labels.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    data : dict[str, Any]
        Output of ``bundle``.
    statements : Any
        Statement matrix.
    presence : pd.DataFrame
        Output of ``statement_presence`` restricted to the statements to probe.
    space : str
        Space name, for the task names.
    train_x, validation_x : str
        ``.npy`` files of the space's embeddings.

    Returns
    -------
    list[tuple]
        ``(kind, binary_probe arguments)`` per task.
    """
    records, train = data["records"], data["train"]
    sources = records["source"].to_numpy()
    strata, groups = sources[~train], records["group"].to_numpy()[~train]
    tasks = []
    for row in presence.itertuples():
        present = np.isin(sources, row.present_sources.split(";"))
        column = np.asarray(statements[:, row.Index].todense()).ravel().astype(float)
        values = np.where(present, column, np.nan)
        tasks.append(
            (
                "statement",
                (
                    f"{space}|{row.statement}",
                    train_x,
                    values[train],
                    validation_x,
                    values[~train],
                    strata,
                    groups,
                    settings(args)["draws"],
                    key06.SEED + 31,
                ),
            )
        )
    for index, name in enumerate(CONDITIONS):
        values = data["labels"][:, index]
        tasks.append(
            (
                "label",
                (
                    f"{space}|{name}",
                    train_x,
                    values[train],
                    validation_x,
                    values[~train],
                    strata,
                    groups,
                    settings(args)["draws"],
                    key06.SEED + 31,
                ),
            )
        )
    return tasks


def statements_step(args: argparse.Namespace) -> None:
    """
    Step 7: how well the embedding separates each statement outside the 14 conditions.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    """
    started = time.time()
    output, work = paths(args)
    directory = output / "statements"
    directory.mkdir(parents=True, exist_ok=True)
    data = bundle(args)
    statements, names = key06.statement_matrix(data["records"])
    presence = key06.statement_presence(statements, names, data["records"])
    descriptions = key06.statement_descriptions(names)
    presence["description"] = presence["statement"].map(descriptions)
    presence["mapping"] = presence["statement"].map(key06.statement_mapping_note)
    presence.to_csv(directory / "presence.csv", index=False)
    probed = presence[presence["train_positives"] >= key06.STATEMENT_MIN_PROBE]
    tasks = []
    for space in key06.SPACES:
        train_x, validation_x = embeddings(data, space, work)
        train_file = array_file(work, f"{space}_train", train_x)
        validation_file = array_file(work, f"{space}_validation", validation_x)
        tasks.extend(statement_tasks(args, data, statements, probed, space, train_file, validation_file))
    results = pd.DataFrame(parallel(run_probe, tasks, args.workers, work))
    results[["space", "feature"]] = results["name"].str.split("|", n=1, expand=True)
    table = results.merge(
        presence, how="left", left_on="feature", right_on="statement", suffixes=("", "_presence")
    )
    table.to_csv(directory / "statements.csv", index=False)
    receipt(
        args,
        "statements",
        started,
        {
            "probed_statements": len(probed),
            "listed_by_rule": int(probed["probe_rule"].sum()),
            "bootstrap_seed": key06.SEED + 31,
        },
        [key06.MANIFEST, key06.FEATURES / "xecg.npy"],
    )


def note(args: argparse.Namespace) -> None:
    """
    Record the analyst's reading of one review sheet in ``review/notes.csv``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``sheet`` and the reading fields.
    """
    output, _ = paths(args)
    path = output / "review" / "notes.csv"
    fields = ("rate", "rhythm", "qrs", "axis", "voltage", "st_t", "noise", "summary")
    row = {
        "sheet": args.sheet,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        **{name: getattr(args, name) or "" for name in fields},
    }
    if not (output / args.sheet).exists():
        raise FileNotFoundError(output / args.sheet)
    table = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=list(row))
    table = pd.concat([table[table["sheet"] != args.sheet], pd.DataFrame([row])], ignore_index=True)
    table.sort_values("sheet").to_csv(path, index=False)
    print(f"{len(table)} notes")


STEPS = {
    "integrity": integrity,
    "variance": variance,
    "residual": residual,
    "prepare": prepare,
    "cluster": cluster,
    "assemble": assemble,
    "describe": describe,
    "flags": flags_step,
    "review": review,
    "detail": detail,
    "cross-encoder": cross_encoder,
    "outliers": outliers,
    "statements": statements_step,
    "note": note,
}


def parse(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; ``sys.argv`` when None.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=sorted(STEPS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--solution")
    parser.add_argument("--cluster", type=int)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--sheet")
    for name in ("rate", "rhythm", "qrs", "axis", "voltage", "st_t", "noise", "summary"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args(argv)
    if args.workers > 16:
        parser.error("at most 16 workers while another agent shares the machine")
    return args


def main(argv: list[str] | None = None) -> None:
    """
    Run one step.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.
    """
    args = parse(argv)
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
