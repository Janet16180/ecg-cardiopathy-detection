"""
MLC-03 Part D: P0 and H-All trained on dataset v5, against the v4-trained models, on v5 validation only.

Steps (outputs under ``outputs/mlc03_data_quality_20261111/effect/``; scratch in
``/var/tmp/janetrivera/mlc03/effect``):

- ``integrity``: the stored v4 validation predictions reproduce the MLC-02 selection metrics of P0 (0.9643)
  and H-All (0.9706) exactly, and P0 refitted on v4 train at its chosen C reproduces its stored validation
  probabilities;
- ``p0``: P0's grid on v5 train, C per condition chosen on v5 validation;
- ``profile``: three epochs of H-All on the GPU, which gate the full run;
- ``h-all``: a rerun of H-All's first seed on v4 (recipe check), then three seeds on v5;
- ``evaluate``: selection metric and per-condition within-source AUROC of the v4- and v5-trained models on v5
  validation, paired patient-cluster bootstrap intervals of the differences, and the v4-trained models on v4
  validation for reference.

Run from the worktree with ``OPENBLAS_CORETYPE`` unset (MLC-02's default kernel):
``PYTHONPATH=<worktree> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m
scripts.experiments.run_mlc03_effect <step>``; the GPU steps also need ``CUDA_VISIBLE_DEVICES=0
CUBLAS_WORKSPACE_CONFIG=:4096:8``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment import mlc03_effect as me
from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.gpu import GPU_LOCK_PATH, gpu_lock
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_data import CONDITIONS, load_predictions, prediction_path, save_predictions
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_feature_data import OPEN_ROLES, feature_hashes, load_split
from ecg_experiment.mlc02_feature_models import (
    EPOCHS,
    INPUTS,
    LOGISTIC_C,
    LOGISTIC_SEED,
    SEEDS,
    choose_per_condition,
    fit_scaler,
    load_candidate,
    logistic_inputs,
    predict,
    save_logistic,
    save_network,
    train_network,
)
from ecg_experiment.mlc02_feature_runs import (
    environment,
    gpu_state,
    network_arrays,
    refuse_coretype,
    run_pool,
    selection_value,
)
from ecg_experiment.paths import to_stored

SCRATCH = Path("/var/tmp/janetrivera/mlc03/effect")
MODELS = me.EFFECT / "candidates"
CODE = (
    "ecg_experiment/mlc03_effect.py",
    "scripts/experiments/run_mlc03_effect.py",
    "ecg_experiment/mlc03_dataset_v5.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/mlc02_feature_runs.py",
    "ecg_experiment/mlc02_feature_data.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/mlc02_bootstrap.py",
    "ecg_experiment/key_classifier_models.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/gpu.py",
)
EDAC = Path("/sys/devices/system/edac/mc/mc0/ue_count")
PREDICT_TOLERANCE = 1e-12
REFIT_TOLERANCE = 1e-10


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
    Convert NumPy scalars and arrays inside a nested value to Python ones, for JSON.

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


def receipt(step: str, started: float, git: dict[str, Any], extra: dict[str, Any]) -> None:
    """
    Write the receipt of one step under ``effect/receipts``.

    Parameters
    ----------
    step : str
        Step name.
    started : float
        ``time.time()`` at the start.
    git : dict[str, Any]
        ``git_state`` at the start.
    extra : dict[str, Any]
        Step-specific content.
    """
    record = {
        "step": step,
        "command": [sys.executable, *sys.argv],
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "git": git,
        "code_sha256": code_sha256(CODE),
        "inputs": {
            "v5_manifest": to_stored(mv.V5 / "manifest.csv"),
            "v5_manifest_sha256": sha256_file(mv.V5 / "manifest.csv")
            if (mv.V5 / "manifest.csv").exists()
            else None,
            "features": feature_hashes(),
            "stored_predictions_sha256": {name: sha256_file(path) for name, path in me.STORED.items()},
        },
        "environment": environment(),
        "edac_uncorrected": EDAC.read_text().strip() if EDAC.exists() else None,
        "test_roles_scored": False,
        **extra,
    }
    write_json_atomic(me.EFFECT / "receipts" / f"{step}.json", plain(record), allow_nan=True)


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


def write_split(folder: Path, z_fit: np.ndarray, y_fit: np.ndarray, z_score: np.ndarray) -> None:
    """
    Save one split's logistic inputs for the workers.

    Parameters
    ----------
    folder : Path
        Scratch folder.
    z_fit, y_fit, z_score : np.ndarray
        Unit training rows, training states and unit scored rows.
    """
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "z_fit.npy", z_fit)
    np.save(folder / "y_fit.npy", y_fit)
    np.save(folder / "z_score.npy", z_score)


def fit_grid(folder: Path, tasks: list[tuple[int, float]], workers: int, threads: int) -> dict[tuple, dict]:
    """
    Run logistic fits in spawned workers.

    Parameters
    ----------
    folder : Path
        Output of ``write_split``.
    tasks : list[tuple[int, float]]
        Condition column and C.
    workers, threads : int
        Processes and BLAS threads per process.

    Returns
    -------
    dict[tuple, dict]
        ``(column, c)`` to ``me.fit_task`` output.
    """
    results = run_pool(me.init_fit_worker, (str(folder), threads), me.fit_task, tasks, workers)
    return {(r["column"], r["c"]): r for r in results}


def integrity(workers: int, threads: int) -> dict[str, Any]:
    """
    Reproduce the stored v4 results before any v5 fit (step ``integrity``).

    Parameters
    ----------
    workers, threads : int
        Processes and BLAS threads per process.

    Returns
    -------
    dict[str, Any]
        The checks.

    Raises
    ------
    ValueError
        If a stored metric is not reproduced exactly or the refit differs beyond ``REFIT_TOLERANCE``.
    """
    started, git = time.time(), committed()
    refuse_coretype()
    data, features = load_split(OPEN_ROLES, ("xecg",))
    train, valid = data["role"] == "train", data["role"] == "validation"
    validation = {name: values[valid] for name, values in data.items()}
    checks: dict[str, Any] = {}
    for model, expected in me.STORED_METRIC.items():
        probability = me.stored_on(model, validation)
        found = selection_metric(validation["labels"], probability, validation["source"])["selection_metric"]
        checks[f"{model}_stored_selection_metric"] = {
            "found": found,
            "expected": expected,
            "equal": found == expected,
        }
    chosen = read_json(me.CANDIDATES / "P0" / "candidate.json")["chosen_c"]
    scaler = fit_scaler(features["xecg"][train])
    folder = SCRATCH / "v4"
    write_split(
        folder,
        logistic_inputs(features["xecg"][train], scaler),
        data["labels"][train],
        logistic_inputs(features["xecg"][valid], scaler),
    )
    tasks = [(column, float(chosen[name])) for column, name in enumerate(CONDITIONS)]
    found = fit_grid(folder, tasks, workers, threads)
    refit = np.column_stack([found[task]["probability"] for task in tasks])
    stored = me.stored_on("P0", validation)
    difference = float(np.max(np.abs(refit - stored)))
    checks["P0_refit_v4"] = {
        "max_abs_difference": difference,
        "tolerance": REFIT_TOLERANCE,
        "selection_metric": selection_metric(validation["labels"], refit, validation["source"])[
            "selection_metric"
        ],
        "iterations": [found[task]["iterations"] for task in tasks],
    }
    write_json_atomic(me.EFFECT / "integrity.json", plain(checks), allow_nan=True)
    receipt("integrity", started, git, {"workers": workers, "threads_per_worker": threads, "checks": checks})
    failed = [name for name, check in checks.items() if name.endswith("metric") and not check["equal"]]
    if failed or difference > REFIT_TOLERANCE:
        raise ValueError(f"integrity failed: {failed}, refit difference {difference}")
    log(json.dumps(plain(checks)))
    return checks


def fit_p0(workers: int, threads: int) -> dict[str, Any]:
    """
    Fit P0 on v5 train, C per condition chosen on v5 validation (step ``p0``).

    Parameters
    ----------
    workers, threads : int
        Processes and BLAS threads per process.

    Returns
    -------
    dict[str, Any]
        ``candidate.json`` content.
    """
    started, git = time.time(), committed()
    refuse_coretype()
    data, features, loaded = me.load_v5(("xecg",))
    train, valid = data["role"] == "train", data["role"] == "validation"
    validation = {name: values[valid] for name, values in data.items()}
    scaler = fit_scaler(features["xecg"][train])
    folder = SCRATCH / "v5"
    write_split(
        folder,
        logistic_inputs(features["xecg"][train], scaler),
        data["labels"][train],
        logistic_inputs(features["xecg"][valid], scaler),
    )
    tasks = [(column, c) for c in sorted(LOGISTIC_C, reverse=True) for column in range(len(CONDITIONS))]
    found = fit_grid(folder, tasks, workers, threads)
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
    target = MODELS / "P0"
    save_logistic(target, "P0", scaler, fits, {"c": dict(zip(CONDITIONS, chosen, strict=True))})
    probability = predict("P0", {"xecg": features["xecg"][valid]}, MODELS)
    difference = float(np.max(np.abs(probability - choice["probability"])))
    if difference > PREDICT_TOLERANCE:
        raise ValueError(f"predict differs from the fit by {difference}")
    path = prediction_path(MODELS, "P0", "validation", (LOGISTIC_SEED,))
    save_predictions(path, validation["record_key"], probability, "P0", (LOGISTIC_SEED,))
    v4_choice = read_json(me.CANDIDATES / "P0" / "candidate.json")["chosen_c"]
    record = {
        "candidate": "P0",
        "trained_on": "v5",
        "train_records": int(train.sum()),
        "validation_records": int(valid.sum()),
        "loaded": loaded,
        "chosen_c": dict(zip(CONDITIONS, chosen, strict=True)),
        "chosen_c_v4": v4_choice,
        "c_changed": [name for name, c in zip(CONDITIONS, chosen, strict=True) if c != v4_choice[name]],
        "grid_scores": {
            str(c): dict(zip(CONDITIONS, choice["scores"][i], strict=True)) for i, c in enumerate(LOGISTIC_C)
        },
        "ties": choice["ties"],
        "iterations": {f"{CONDITIONS[col]}@{c}": found[(col, c)]["iterations"] for col, c in found},
        "predict_max_abs_difference": difference,
        "selection_metric": selection_metric(validation["labels"], probability, validation["source"])[
            "selection_metric"
        ],
        "prediction_file": to_stored(path),
        "prediction_sha256": sha256_file(path),
    }
    write_json_atomic(target / "candidate.json", plain(record), allow_nan=True)
    receipt("p0", started, git, {"workers": workers, "threads_per_worker": threads, "fits": len(found)})
    log(f"P0 on v5: selection metric {record['selection_metric']:.4f}, C changed for {record['c_changed']}")
    return record


def locked_run(device: str, **arguments: Any) -> dict[str, Any]:
    """
    ``train_network`` while holding the shared GPU 0 lock, with peak memory.

    Parameters
    ----------
    device : str
        ``cuda`` or ``cpu``.
    **arguments : Any
        Other arguments of ``train_network``.

    Returns
    -------
    dict[str, Any]
        ``train_network`` output with ``seconds`` and ``peak_allocated_bytes``.
    """
    import torch

    started = time.perf_counter()
    with gpu_lock(device, path=GPU_LOCK_PATH):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = train_network(device=device, **arguments)
        peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return {**result, "seconds": time.perf_counter() - started, "peak_allocated_bytes": peak}


def network_setup(v5: bool, threads: int, device: str) -> dict[str, Any]:
    """
    H-All inputs of v4 or v5, standardized as MLC-02 does, and the validation scorer.

    Parameters
    ----------
    v5 : bool
        Use dataset v5 (else v4).
    threads : int
        Torch and BLAS threads.
    device : str
        ``cuda`` or ``cpu``.

    Returns
    -------
    dict[str, Any]
        ``data``, ``features``, ``train``, ``valid``, ``arrays`` (``network_arrays``), ``score``, ``gpu`` and
        ``loaded`` (the v5 load check, None for v4).
    """
    import torch

    torch.set_num_threads(threads)
    state = gpu_state(device)
    if v5:
        data, features, loaded = me.load_v5(INPUTS["H-All"])
    else:
        (data, features), loaded = load_split(OPEN_ROLES, INPUTS["H-All"]), None
    train, valid = data["role"] == "train", data["role"] == "validation"
    arrays = network_arrays("H-All", features, train, valid)
    score = partial(selection_value, data["labels"][valid], data["source"][valid])
    return {
        "data": data,
        "features": features,
        "train": train,
        "valid": valid,
        "arrays": arrays,
        "score": score,
        "gpu": state,
        "loaded": loaded,
    }


def profile(threads: int, device: str, epochs: int = 3) -> dict[str, Any]:
    """
    Time a few epochs of H-All on v5 (step ``profile``).

    Parameters
    ----------
    threads : int
        Torch and BLAS threads.
    device : str
        ``cuda`` or ``cpu``.
    epochs : int
        Profiled epochs.

    Returns
    -------
    dict[str, Any]
        Seconds per epoch, peak memory and the forecast of the full step.
    """
    from threadpoolctl import threadpool_limits

    started, git = time.time(), committed()
    with threadpool_limits(threads):
        setup = network_setup(True, threads, device)
        result = locked_run(
            device,
            x=setup["arrays"]["z_fit"],
            y=setup["data"]["labels"][setup["train"]],
            rate=me.H_ALL_RATE,
            decay=me.H_ALL_DECAY,
            seed=SEEDS[0],
            epochs=epochs,
            x_val=setup["arrays"]["z_score"],
            score=setup["score"],
        )
    seconds = [row["seconds"] for row in result["history"]]
    per_epoch = float(np.mean(seconds[1:]))
    record = {
        "epochs": epochs,
        "epoch_seconds": seconds,
        "seconds_per_epoch": per_epoch,
        "peak_allocated_bytes": result["peak_allocated_bytes"],
        "runs": 1 + len(SEEDS),
        "forecast_seconds": per_epoch * EPOCHS * (1 + len(SEEDS)),
        "gpu": setup["gpu"],
        "git": git,
        "started_unix": started,
    }
    write_json_atomic(MODELS / "H-All" / "profile.json", plain(record), allow_nan=True)
    log(json.dumps(plain({k: v for k, v in record.items() if k != "gpu"})))
    return record


def fit_h_all(threads: int, device: str) -> dict[str, Any]:
    """
    Rerun H-All's first seed on v4, then fit its three seeds on v5 (step ``h-all``).

    Parameters
    ----------
    threads : int
        Torch and BLAS threads.
    device : str
        ``cuda`` or ``cpu``.

    Returns
    -------
    dict[str, Any]
        ``candidate.json`` content.
    """
    from threadpoolctl import threadpool_limits

    started, git = time.time(), committed()
    planned = read_json(MODELS / "H-All" / "profile.json")
    with threadpool_limits(threads):
        old = network_setup(False, threads, device)
        rerun = locked_run(
            device,
            x=old["arrays"]["z_fit"],
            y=old["data"]["labels"][old["train"]],
            rate=me.H_ALL_RATE,
            decay=me.H_ALL_DECAY,
            seed=SEEDS[0],
            epochs=EPOCHS,
            x_val=old["arrays"]["z_score"],
            score=old["score"],
        )
        stored = load_candidate("H-All", me.CANDIDATES)["artefacts"][f"weights__seed-{SEEDS[0]}.npz"]
        epoch_v4 = read_json(me.CANDIDATES / "H-All" / "candidate.json")["epoch_by_seed"][str(SEEDS[0])]
        recipe = {
            "seed": SEEDS[0],
            "epoch": rerun["epoch"],
            "stored_epoch": epoch_v4,
            "weights_identical": all(np.array_equal(rerun["weights"][k], stored[k]) for k in stored),
            "weights_max_abs_difference": max(
                float(np.max(np.abs(rerun["weights"][k] - stored[k]))) for k in stored
            ),
            "gpu_metric": rerun["metric"],
        }
        log(f"H-All recipe check on v4: {recipe}")
        del old, rerun
        new = network_setup(True, threads, device)
        runs = {
            seed: locked_run(
                device,
                x=new["arrays"]["z_fit"],
                y=new["data"]["labels"][new["train"]],
                rate=me.H_ALL_RATE,
                decay=me.H_ALL_DECAY,
                seed=seed,
                epochs=EPOCHS,
                x_val=new["arrays"]["z_score"],
                score=new["score"],
            )
            for seed in SEEDS
        }
    target = MODELS / "H-All"
    epochs = {str(seed): run["epoch"] for seed, run in runs.items()}
    save_network(
        target,
        "H-All",
        new["arrays"]["scaler"],
        new["arrays"]["medians"],
        {seed: run["weights"] for seed, run in runs.items()},
        {"learning_rate": me.H_ALL_RATE, "weight_decay": me.H_ALL_DECAY, "epoch_by_seed": epochs},
    )
    valid = new["valid"]
    validation = {name: values[valid] for name, values in new["data"].items()}
    features = {name: values[valid] for name, values in new["features"].items()}
    by_seed = {}
    for seed, run in runs.items():
        probability = predict("H-All", features, MODELS, seeds=(seed,))
        path = prediction_path(MODELS, "H-All", "validation", (seed,))
        save_predictions(path, validation["record_key"], probability, "H-All", (seed,))
        by_seed[str(seed)] = {
            "epoch": run["epoch"],
            "gpu_metric": run["metric"],
            "selection_metric": selection_metric(validation["labels"], probability, validation["source"])[
                "selection_metric"
            ],
            "gpu_max_abs_difference": float(np.max(np.abs(probability - run["probability"]))),
            "seconds": run["seconds"],
            "peak_allocated_bytes": run["peak_allocated_bytes"],
        }
    probability = predict("H-All", features, MODELS)
    path = prediction_path(MODELS, "H-All", "validation", SEEDS)
    save_predictions(path, validation["record_key"], probability, "H-All", SEEDS)
    record = {
        "candidate": "H-All",
        "trained_on": "v5",
        "train_records": int(new["train"].sum()),
        "validation_records": int(valid.sum()),
        "loaded": new["loaded"],
        "settings": {"learning_rate": me.H_ALL_RATE, "weight_decay": me.H_ALL_DECAY, "epochs": EPOCHS},
        "epoch_by_seed": epochs,
        "epoch_by_seed_v4": read_json(me.CANDIDATES / "H-All" / "candidate.json")["epoch_by_seed"],
        "by_seed": by_seed,
        "selection_metric": selection_metric(validation["labels"], probability, validation["source"])[
            "selection_metric"
        ],
        "recipe_check_v4": recipe,
        "prediction_file": to_stored(path),
        "prediction_sha256": sha256_file(path),
        "input_width": int(new["arrays"]["z_fit"].shape[1]),
    }
    write_json_atomic(target / "candidate.json", plain(record), allow_nan=True)
    receipt(
        "h-all", started, git, {"gpu": new["gpu"], "profile": planned, "threads": threads, "device": device}
    )
    log(f"H-All on v5: selection metric {record['selection_metric']:.4f}, epochs {epochs}")
    return record


def evaluate(threads: int) -> dict[str, Any]:
    """
    Compare v4- and v5-trained models on v5 validation, with the v4 reference (step ``evaluate``).

    Parameters
    ----------
    threads : int
        BLAS threads.

    Returns
    -------
    dict[str, Any]
        ``effect.json`` content.
    """
    from threadpoolctl import threadpool_limits

    started, git = time.time(), committed()
    data4, _ = load_split(OPEN_ROLES, ())
    valid4 = data4["role"] == "validation"
    validation4 = {name: values[valid4] for name, values in data4.items()}
    included, labels5 = me.v5_states(validation4["record_key"])
    labels5, keys5 = labels5[included], validation4["record_key"][included]
    sources5, groups5 = validation4["source"][included], validation4["group"][included]
    result: dict[str, Any] = {
        "records": {"v4_validation": int(valid4.sum()), "v5_validation": int(included.sum())}
    }
    tables, rows = [], []
    with threadpool_limits(threads):
        for model in ("P0", "H-All"):
            seeds = (LOGISTIC_SEED,) if model == "P0" else SEEDS
            trained5 = load_predictions(prediction_path(MODELS, model, "validation", seeds), keys5)[
                "probability"
            ]
            trained4 = me.stored_on(model, validation4)
            reference = selection_metric(validation4["labels"], trained4, validation4["source"])
            comparison = me.compare(labels5, groups5, sources5, trained5, trained4[included], ("v5", "v4"))
            comparison["reference_v4_on_v4"] = {
                "selection_metric": reference["selection_metric"],
                "macro_auroc_by_source": reference["macro_auroc_by_source"],
                "per_condition": reference["per_condition"],
            }
            result[model] = comparison
            for labels, probability, sources, trained, scored in (
                (validation4["labels"], trained4, validation4["source"], "v4", "v4"),
                (labels5, trained4[included], sources5, "v4", "v5"),
                (labels5, trained5, sources5, "v5", "v5"),
            ):
                table = me.per_source_auroc(labels, probability, sources)
                tables.append(table.assign(model=model, trained_on=trained, scored_on=scored))
            selection = comparison["selection"]
            rows += [
                {
                    "model": model,
                    "trained_on": "v4",
                    "scored_on": "v4",
                    "selection_metric": reference["selection_metric"],
                },
                {
                    "model": model,
                    "trained_on": "v4",
                    "scored_on": "v5",
                    "selection_metric": selection["v4"]["value"],
                    "ci_low": selection["v4"]["ci"][0],
                    "ci_high": selection["v4"]["ci"][1],
                },
                {
                    "model": model,
                    "trained_on": "v5",
                    "scored_on": "v5",
                    "selection_metric": selection["v5"]["value"],
                    "ci_low": selection["v5"]["ci"][0],
                    "ci_high": selection["v5"]["ci"][1],
                },
                {
                    "model": model,
                    "trained_on": "v5 minus v4",
                    "scored_on": "v5",
                    "selection_metric": selection["difference"]["value"],
                    "ci_low": selection["difference"]["ci"][0],
                    "ci_high": selection["difference"]["ci"][1],
                },
            ]
            log(f"{model}: {json.dumps(plain(selection))}")
    pd.DataFrame(rows).to_csv(me.EFFECT / "selection_metric.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(me.EFFECT / "auroc_by_source.csv", index=False)
    pd.DataFrame([{"model": m, **row} for m in ("P0", "H-All") for row in result[m]["conditions"]]).to_csv(
        me.EFFECT / "difference_by_condition.csv", index=False
    )
    pd.DataFrame([{"model": m, **row} for m in ("P0", "H-All") for row in result[m]["by_source"]]).to_csv(
        me.EFFECT / "difference_by_source.csv", index=False
    )
    write_json_atomic(me.EFFECT / "effect.json", plain(result), allow_nan=True)
    receipt("evaluate", started, git, {"threads": threads, "draws": me.DRAWS, "seed": me.SEED})
    return result


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; None reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=("integrity", "p0", "profile", "h-all", "evaluate"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.workers * args.threads > 16:
        raise ValueError("at most 16 threads while another agent runs")
    if args.step == "integrity":
        integrity(args.workers, args.threads)
    elif args.step == "p0":
        fit_p0(args.workers, args.threads)
    elif args.step == "profile":
        profile(args.threads, args.device)
    elif args.step == "h-all":
        fit_h_all(args.threads, args.device)
    else:
        evaluate(args.threads)


if __name__ == "__main__":
    main()
