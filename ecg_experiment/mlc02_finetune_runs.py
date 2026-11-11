"""
Runs of candidate F of MLC-02, on train and validation records only.

Steps, each a command of ``scripts/experiments/run_mlc02_finetune.py``:

1. ``prepare``: the xECG inputs of the train and validation roles on local disk, every window's first-10-s
   hash checked against the manifest (``mlc02_finetune_data.prepare_inputs``).
2. ``profile``: on the GPU, the frozen-encoder forward pass and the step-0 check on every validation record,
   about 100 training updates, the peak memory, and the forecast in GPU hours against the protocol's 12-hour
   gate. The gate decides whether both top rates run or 1e-5 alone.
3. ``train``: the grid with seed 20261110 (each rate's run keeps its best evaluation point), the choice of the
   rate, then seeds 20261111 and 20261112 at that rate. The grid run of the chosen rate is seed 20261110's
   run. Every run is resumable from its last evaluation point and holds the GPU lock while it runs.
4. ``finalize``: the kept checkpoints of the three seeds, ``model.json``, the validation prediction files of
   every seed and of their mean (from ``predict``), ``candidate.json`` and ``receipt.json``.

No step reads a test role: the loaders of ``mlc02_finetune_data`` refuse them before opening any file.
"""

from __future__ import annotations

import gc
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_data import (
    CONDITIONS,
    MANIFEST,
    VALIDATION_SOURCES,
    load_predictions,
    prediction_path,
    save_predictions,
    subset,
)
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_feature_data import feature_hashes
from ecg_experiment.mlc02_feature_models import (
    CANDIDATES,
    SEEDS,
    load_candidate,
    network_probability,
    standardize,
)
from ecg_experiment.mlc02_feature_runs import environment, validation_rows, write_rows
from ecg_experiment.mlc02_features import GPU_LOCKS, torch_flags
from ecg_experiment.mlc02_finetune import (
    BASE,
    BATCH,
    CANDIDATE,
    CLIP_NORM,
    DIRECTORY,
    DROP_PATH,
    EPOCHS,
    EVAL_BATCH,
    HEAD_FACTOR,
    LAYER_DECAY,
    TOP_RATES,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    evaluate,
    initial_model,
    make_optimizer,
    plan,
    predict,
    predict_inputs,
    rate_factor,
    save_state,
    save_weights,
    set_rates,
    train_run,
    train_update,
    ue_count,
    write_model,
)
from ecg_experiment.mlc02_finetune_data import (
    CACHE,
    INPUTS,
    WORK,
    cached_xecg_features,
    load_inputs,
    read_windows,
    window_inputs,
)
from ecg_experiment.paths import to_stored
from ecg_experiment.xecg import RELEASE_SHA256

TRAINING_CODE = (
    "ecg_experiment/mlc02_finetune.py",
    "ecg_experiment/mlc02_finetune_data.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/mlc02_feature_data.py",
    "ecg_experiment/mlc02_features.py",
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/external_encoders.py",
    "ecg_experiment/xecg.py",
    "ecg_experiment/cuda_runtime20261028.py",
    "ecg_experiment/files.py",
    "ecg_experiment/mlc01_dataset.py",
)
CODE = (
    *TRAINING_CODE,
    "ecg_experiment/mlc02_finetune_runs.py",
    "scripts/experiments/run_mlc02_finetune.py",
    "ecg_experiment/mlc02_feature_runs.py",
    "ecg_experiment/mlc02_cache_build.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/paths.py",
    "third_party/checkpoints/xecg/xECG.py",
)
GIB = 1024**3
HEADROOM = GIB
PROFILE_FREE = 6 * GIB
SMOKE_FREE = 3 * GIB
PROFILE_UPDATES = 100
PROFILE_SKIPPED = 5
DRIFT_RECORDS = 256
GATE_HOURS = 12.0
FEATURE_TOLERANCE = 1e-4
STEP0_TOLERANCE = 1e-4
PREDICT_TOLERANCE = 1e-4
PREDICT_CHECK_RECORDS = 512
LOGS = Path("/var/tmp/janetrivera/mlc02/logs/finetune")
RECIPE = {
    "fixed": "10 November 2026, before the first training run",
    "encoder": "released xECG, vanilla xLSTM backend, Stage 1 input path (100 Hz, time-major)",
    "encoder_drop_path": DROP_PATH,
    "encoder_dropout": 0.0,
    "head": "H-X network: 1,024 -> 512 ReLU -> dropout 0.3 -> 14 logits, on pooled features standardized "
            "with H-X's training standardizer (fixed)",
    "head_initialization": "H-X weights of the run's seed (H-X saved one head per seed)",
    "loss": "binary cross-entropy averaged over the defined entries of each effective batch; "
            "undefined labels masked",
    "class_weights": None,
    "augmentation": None,
    "optimizer": "AdamW, betas (0.9, 0.999), eps 1e-8, foreach off",
    "weight_decay": WEIGHT_DECAY,
    "weight_decay_excludes": "biases, LayerNorm and MultiHeadLayerNorm weights, the unused mask token",
    "top_rates": list(TOP_RATES),
    "layer_decay": LAYER_DECAY,
    "rates": "top block r; block i r * 0.75**(8 - i); patch embedding r * 0.75**9; head 10 r",
    "head_rate_factor": HEAD_FACTOR,
    "effective_batch": BATCH,
    "epochs": EPOCHS,
    "warmup_fraction": WARMUP_FRACTION,
    "schedule": "linear warm-up over the first 5% of updates (ceiling), cosine decay to zero at the last "
                "update",
    "gradient_clipping": CLIP_NORM,
    "precision": "BF16 autocast for training forwards, FP32 master weights; FP32 evaluation and predict",
    "evaluation": "validation after every half epoch (10 points); step 0 is scored but not eligible",
    "evaluation_batch": EVAL_BATCH,
    "choice_of_point": "first point with the highest validation selection metric, per seed",
    "choice_of_rate": "higher best-point selection metric of seed 20261110; an exact tie goes to 1e-5",
    "seeds": list(SEEDS),
    "seed_role": "record order and dropout; the head starts from H-X's weights of the same seed",
    "candidate_prediction": "mean probability of the three seeds at their kept points",
}


def log(message: str) -> None:
    """
    Print a timestamped progress line.

    Parameters
    ----------
    message : str
        Text.
    """
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def committed(require: bool = True) -> dict[str, Any]:
    """
    Commit and code hashes of a run, which must start from committed code.

    Parameters
    ----------
    require : bool
        Raise on uncommitted code (off only for smoke tests, whose outputs are not results).

    Returns
    -------
    dict[str, Any]
        ``commit``, ``modified_tracked``, ``uncommitted_code`` and ``code_sha256``.

    Raises
    ------
    RuntimeError
        If ``require`` and a code file is uncommitted or a tracked file is modified.
    """
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *CODE], cwd=ROOT, text=True)
    state = git_state()
    if require and (status.strip() or state["modified_tracked"]):
        raise RuntimeError(f"commit the code before running: {status or state['modified_tracked']}")
    return {**state, "uncommitted_code": status.splitlines(), "code_sha256": code_sha256(CODE)}


def gpu_lock_path(gpu: int) -> Path:
    """
    Project lock of a physical GPU, after checking that it is the only visible one.

    Parameters
    ----------
    gpu : int
        Physical GPU index.

    Returns
    -------
    Path
        Lock file.

    Raises
    ------
    RuntimeError
        If ``CUDA_VISIBLE_DEVICES`` does not name that GPU alone.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={gpu} for GPU {gpu}")
    return GPU_LOCKS[gpu]


def nvidia_smi() -> list[str]:
    """
    Memory and utilization of every GPU, as reported by ``nvidia-smi``.

    Returns
    -------
    list[str]
        One CSV line per GPU: index, memory used, memory total, utilization.
    """
    query = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"]
    return subprocess.check_output(query, text=True).strip().splitlines()


def free_memory(required: int) -> dict[str, Any]:
    """
    Free memory of the visible GPU, required to cover a job's peak plus headroom.

    Memory this process holds in its allocator cache, released first, counts as available.

    Parameters
    ----------
    required : int
        Bytes needed.

    Returns
    -------
    dict[str, Any]
        ``free_bytes``, ``reserved_bytes`` (this process), ``total_bytes``, ``required_bytes`` and the
        ``nvidia_smi`` lines.

    Raises
    ------
    RuntimeError
        If less is available.
    """
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    if free + reserved < required:
        raise RuntimeError(f"{free + reserved} bytes available on the GPU, {required} required")
    return {"free_bytes": int(free), "reserved_bytes": int(reserved), "total_bytes": int(total),
            "required_bytes": int(required), "nvidia_smi": nvidia_smi(),
            "device": torch.cuda.get_device_name()}


def stride_positions(records: int, limit: int | None) -> np.ndarray:
    """
    Evenly spaced records for a smoke test, or all of them.

    Parameters
    ----------
    records : int
        Records available.
    limit : int | None
        Records wanted; None keeps all.

    Returns
    -------
    np.ndarray
        Positions in order.
    """
    if limit is None or limit >= records:
        return np.arange(records)
    return np.arange(0, records, records // limit)[:limit]


def open_data(limits: dict[str, int | None]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Training and validation inputs in memory, with labels and sources.

    Parameters
    ----------
    limits : dict[str, int | None]
        ``train`` and ``validation`` record limits (None for all; smoke tests only set them).

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        ``train`` (``inputs``, float32 ``labels``), ``validation`` (``inputs``, ``labels``, ``sources``,
        ``data``, ``keys`` and ``positions`` in the full role) and the two preparation records.
    """
    train_data, train_inputs, train_record = load_inputs("train")
    validation_data, validation_inputs, validation_record = load_inputs("validation")
    kept = stride_positions(len(train_inputs), limits["train"])
    if len(kept) < len(train_inputs):
        train_inputs, train_data = train_inputs[kept], subset(train_data, kept)
    positions = stride_positions(len(validation_inputs), limits["validation"])
    full_keys = validation_data["record_key"]
    if len(positions) < len(validation_inputs):
        validation_inputs, validation_data = validation_inputs[positions], subset(validation_data, positions)
    train = {"inputs": train_inputs, "labels": train_data["labels"].astype(np.float32)}
    validation = {"inputs": validation_inputs, "labels": validation_data["labels"],
                  "sources": validation_data["source"], "data": validation_data, "keys": full_keys,
                  "positions": positions}
    return train, validation, {"train": train_record, "validation": validation_record}


def start_check(seed: int, validation: dict[str, Any], cached: np.ndarray) -> Callable[..., dict[str, Any]]:
    """
    Step-0 check of a run: the frozen-encoder features and H-X's validation predictions of the head seed.

    Parameters
    ----------
    seed : int
        H-X head seed.
    validation : dict[str, Any]
        Output of ``open_data``.
    cached : np.ndarray
        Stage 1 xECG features of the same validation records.

    Returns
    -------
    Callable[..., dict[str, Any]]
        Called with the step-0 probabilities and pooled features; returns the record and raises when the
        features differ from Stage 1 by more than 1e-4 relative per record (a record whose cached features are
        all zero must match exactly) or a probability differs from H-X's by more than 1e-4.
    """
    base = load_candidate(BASE)
    scaler, weights = base["artefacts"]["scaler.npz"], base["artefacts"][f"weights__seed-{seed}.npz"]
    path = prediction_path(CANDIDATES, BASE, "validation", (seed,))
    expected = load_predictions(path, validation["keys"])["probability"][validation["positions"]]
    cached = np.asarray(cached, dtype=np.float64)
    reference = selection_metric(validation["labels"], expected, validation["sources"])["selection_metric"]

    def check(probability: np.ndarray, pooled: np.ndarray) -> dict[str, Any]:
        difference = np.abs(np.asarray(pooled, dtype=np.float64) - cached)
        largest, scale = difference.max(axis=1), np.abs(cached).max(axis=1)
        relative = np.divide(largest, scale, out=np.where(largest > 0, np.inf, 0.0), where=scale > 0)
        new_head = network_probability(standardize(pooled, scaler), weights)
        old_head = network_probability(standardize(cached, scaler), weights)
        record = {
            "records": len(pooled),
            "head_seed": seed,
            "feature_max_abs_difference": float(difference.max()),
            "feature_max_relative_difference": float(relative.max()),
            "feature_records_above_tolerance": int((relative > FEATURE_TOLERANCE).sum()),
            "feature_bit_identical_records": int((largest == 0).sum()),
            "zero_feature_records": int((scale == 0).sum()),
            "probability_max_abs_difference": float(np.abs(probability - expected).max()),
            "float64_head_new_features_max_abs_difference": float(np.abs(new_head - expected).max()),
            "float64_head_cached_features_max_abs_difference": float(np.abs(old_head - expected).max()),
            "hx_selection_metric": float(reference),
            "step0_selection_metric": float(selection_metric(
                validation["labels"], probability, validation["sources"])["selection_metric"]),
            "feature_tolerance_relative": FEATURE_TOLERANCE,
            "probability_tolerance": STEP0_TOLERANCE,
            "prediction_file": to_stored(path),
        }
        failed = record["probability_max_abs_difference"] > STEP0_TOLERANCE
        if record["feature_records_above_tolerance"] or failed:
            raise RuntimeError(f"step-0 check failed: {record}")
        return record

    return check


def run_prepare(workers: int, command: list[str]) -> dict[str, Any]:
    """
    Prepare the xECG inputs of the train and validation roles.

    Parameters
    ----------
    workers : int
        Processes, at most 8.
    command : list[str]
        Command line.

    Returns
    -------
    dict[str, Any]
        The two preparation records.

    Raises
    ------
    ValueError
        If more than 8 workers are asked for.
    """
    from ecg_experiment.mlc02_finetune_data import prepare_inputs

    if workers > 8:
        raise ValueError("at most 8 loader workers")
    git = committed(require=False)
    records = {role: prepare_inputs(role, workers) for role in ("validation", "train")}
    write_json_atomic(INPUTS / "prepare_receipt.json", {"command": command, "git": git, "records": records,
                                                        "unix_time": time.time()})
    return records


def bf16_drift(model: Any, inputs: np.ndarray) -> float:
    """
    Largest probability difference between BF16-autocast and FP32 evaluation.

    Parameters
    ----------
    model : FineTuned
        Model on the GPU.
    inputs : np.ndarray
        A few validation inputs.

    Returns
    -------
    float
        Maximum absolute difference.
    """
    exact, _ = evaluate(model, inputs, "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mixed, _ = evaluate(model, inputs, "cuda")
    return float(np.max(np.abs(exact - mixed)))


def time_updates(model: Any, train: dict[str, Any], rate: float, microbatch: int,
                 updates: int) -> dict[str, Any]:
    """
    Time the first training updates of the seed-20261110 run at one rate.

    Parameters
    ----------
    model : FineTuned
        Model at step 0 on the GPU.
    train : dict[str, Any]
        Training inputs and labels.
    rate : float
        Top rate.
    microbatch : int
        Records per forward pass.
    updates : int
        Updates to time.

    Returns
    -------
    dict[str, Any]
        Per-update seconds, losses, gradient norms, the optimizer and whether the encoder's lowest and highest
        layers changed.
    """
    schedule = plan(len(train["inputs"]))
    torch.manual_seed(SEEDS[0])
    optimizer = make_optimizer(model, rate)
    order = torch.randperm(len(train["inputs"]), generator=torch.Generator().manual_seed(SEEDS[0])).numpy()
    low = model.encoder.patch_embedding.conv.weight.detach().clone()
    high = model.encoder.core.model.blocks[-1].ffn.proj_down.weight.detach().clone()
    seconds, losses, norms = [], [], []
    for update in range(updates):
        set_rates(optimizer, rate_factor(update + 1, schedule["total"], schedule["warmup"]))
        started = time.perf_counter()
        loss, norm = train_update(model, optimizer, train["inputs"], train["labels"],
                                  order[(update * BATCH + np.arange(BATCH)) % len(order)], microbatch, "cuda")
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        losses.append(loss)
        norms.append(norm)
    changed = {"patch_embedding": not torch.equal(low, model.encoder.patch_embedding.conv.weight.detach()),
               "top_block": not torch.equal(
                   high, model.encoder.core.model.blocks[-1].ffn.proj_down.weight.detach())}
    return {"seconds": seconds, "losses": losses, "grad_norms": norms, "optimizer": optimizer,
            "changed": changed}


def time_checkpoint(model: Any, optimizer: Any, directory: Path) -> dict[str, float]:
    """
    Time one resumable checkpoint and one weight file, then delete them.

    Parameters
    ----------
    model : FineTuned
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer with its state.
    directory : Path
        Local scratch directory.

    Returns
    -------
    dict[str, float]
        Seconds and bytes of both files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress = {"fingerprint": {}, "update": 0, "order": None, "history": [], "best": None, "start": {},
                "resumes": []}
    save_state(directory / "resume.pt", model, optimizer, torch.Generator(), progress, "cuda")
    state_seconds = time.perf_counter() - started
    started = time.perf_counter()
    save_weights(model, directory / "weights.safetensors")
    weight_seconds = time.perf_counter() - started
    sizes = {name: (directory / name).stat().st_size for name in ("resume.pt", "weights.safetensors")}
    shutil.rmtree(directory)
    return {"state_seconds": state_seconds, "weights_seconds": weight_seconds,
            "state_bytes": sizes["resume.pt"], "weights_bytes": sizes["weights.safetensors"]}


def forecast(
    per_update: float, evaluation: float, checkpoint: float, load: float, records: int
) -> dict[str, Any]:
    """
    GPU hours of the full plan (two rates with seed 20261110, two more seeds) and of 1e-5 alone.

    Parameters
    ----------
    per_update : float
        Seconds per training update.
    evaluation : float
        Seconds per validation evaluation.
    checkpoint : float
        Seconds per evaluation point spent on checkpoints (state and, at most, weights).
    load : float
        Seconds to build a model.
    records : int
        Training records.

    Returns
    -------
    dict[str, Any]
        Seconds per run, the two plans in hours, the gate and the rates to run.
    """
    schedule = plan(records)
    points = len(schedule["points"])
    run = schedule["total"] * per_update + (points + 1) * evaluation + points * checkpoint + load
    final = len(SEEDS) * (evaluation + load)
    full = (len(TOP_RATES) + len(SEEDS) - 1) * run + final
    reduced = len(SEEDS) * run + final
    passed = full / 3600 <= GATE_HOURS
    return {"updates_per_run": schedule["total"], "evaluations_per_run": points + 1, "seconds_per_run": run,
            "finalize_seconds": final, "full_plan_runs": len(TOP_RATES) + len(SEEDS) - 1,
            "full_plan_hours": full / 3600, "reduced_plan_runs": len(SEEDS),
            "reduced_plan_hours": reduced / 3600,
            "gate_hours": GATE_HOURS, "full_plan": passed,
            "rates": list(TOP_RATES) if passed else [TOP_RATES[0]]}


def run_profile(gpu: int, microbatch: int, updates: int, command: list[str], limits: dict[str, int | None],
                output: Path = DIRECTORY, work: Path = WORK) -> dict[str, Any]:
    """
    Real GPU profile: frozen forward and step-0 check on validation, timed updates, memory and the forecast.

    Parameters
    ----------
    gpu : int
        Physical GPU.
    microbatch : int
        Records per forward pass.
    updates : int
        Training updates to time.
    command : list[str]
        Command line.
    limits : dict[str, int | None]
        Record limits (smoke tests only).
    output : Path
        Candidate directory.
    work : Path
        Local work directory.

    Returns
    -------
    dict[str, Any]
        ``profile.json`` content.
    """
    smoke = any(value is not None for value in limits.values())
    git = committed(require=not smoke)
    lock = gpu_lock_path(gpu)
    train, validation, records = open_data(limits)
    cached = cached_xecg_features(validation["data"])
    started = time.time()
    with gpu_lock("cuda", path=lock):
        memory = free_memory(SMOKE_FREE if smoke else PROFILE_FREE)
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        clock = time.perf_counter()
        model, origin = initial_model(SEEDS[0], "cuda")
        load_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        probability, pooled = evaluate(model, validation["inputs"], "cuda", pooled=True)
        evaluation_seconds = time.perf_counter() - clock
        start = start_check(SEEDS[0], validation, cached)(probability, pooled)
        log(f"step-0 check passed: {start}")
        drift = bf16_drift(model, validation["inputs"][:DRIFT_RECORDS])
        timed = time_updates(model, train, TOP_RATES[0], microbatch, updates)
        checkpoint = time_checkpoint(model, timed.pop("optimizer"), work / "profile_checkpoint")
        peak = {"allocated": int(torch.cuda.max_memory_allocated()),
                "reserved": int(torch.cuda.max_memory_reserved())}
        after = free_memory(0)
        del model
        torch.cuda.empty_cache()
    per_update = float(np.mean(timed["seconds"][PROFILE_SKIPPED:]))
    saving = checkpoint["state_seconds"] + checkpoint["weights_seconds"]
    plan_forecast = forecast(per_update, evaluation_seconds, saving, load_seconds, len(train["inputs"]))
    profile = {
        "candidate": CANDIDATE, "smoke": smoke, "limits": limits, "command": command, "git": git,
        "training_code_sha256": code_sha256(TRAINING_CODE), "origin": origin, "microbatch": microbatch,
        "effective_batch": BATCH, "train_records": len(train["inputs"]),
        "validation_records": len(validation["inputs"]), "memory_before": memory, "memory_after": after,
        "peak_bytes": peak, "model_load_seconds": load_seconds,
        "validation_evaluation_seconds": evaluation_seconds,
        "step0": start, "bf16_max_abs_probability_drift": drift, "bf16_drift_records": DRIFT_RECORDS,
        "profiled_rate": TOP_RATES[0], "updates": updates, "update_seconds": timed["seconds"],
        "seconds_per_update": per_update, "skipped_updates": PROFILE_SKIPPED, "losses": timed["losses"],
        "grad_norms": timed["grad_norms"], "encoder_changed": timed["changed"], "checkpoint": checkpoint,
        "forecast": plan_forecast, "ue_count": ue_count(), "torch_flags": torch_flags(),
        "wall_seconds": time.time() - started, "inputs": records, "test_roles_scored": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / f"profile__microbatch-{microbatch}.json", profile)
    write_json_atomic(output / "profile.json", profile)
    return profile


def run_name(rate: float, seed: int) -> str:
    """
    Directory name of one run.

    Parameters
    ----------
    rate : float
        Top rate.
    seed : int
        Seed.

    Returns
    -------
    str
        ``rate-<r>__seed-<s>``.
    """
    return f"rate-{rate:g}__seed-{seed}"


def record_session(path: Path, session: dict[str, Any]) -> None:
    """
    Add or complete one process session in a run's ``sessions.json``.

    A session is written when it starts and completed when it ends, so that a killed process leaves its
    start behind.

    Parameters
    ----------
    path : Path
        ``sessions.json``.
    session : dict[str, Any]
        Session record with ``pid`` and ``started_unix``; an existing entry with both is updated.
    """
    sessions = read_json(path) if path.exists() else []
    same = [item for item in sessions
            if (item["pid"], item["started_unix"]) == (session["pid"], session["started_unix"])]
    if same:
        same[0].update(session)
    else:
        sessions.append(session)
    write_json_atomic(path, sessions)


def one_run(rate: float, seed: int, context: dict[str, Any]) -> dict[str, Any]:
    """
    Run, resume or reuse one fine-tuning run under the GPU lock.

    Parameters
    ----------
    rate : float
        Top rate.
    seed : int
        Seed.
    context : dict[str, Any]
        ``train``, ``validation``, ``records``, ``cached``, ``profile``, ``lock``, ``git``, ``work``,
        ``output`` and ``smoke``.

    Returns
    -------
    dict[str, Any]
        ``run.json`` content.
    """
    name = run_name(rate, seed)
    published = context["output"] / "runs" / name
    if (published / "run.json").exists() and read_json(published / "run.json").get("complete"):
        log(f"{name} is complete; reusing it")
        return read_json(published / "run.json")
    profile = context["profile"]
    fingerprint = {"rate": rate, "seed": seed, "microbatch": profile["microbatch"],
                   "train_records": len(context["train"]["inputs"]),
                   "validation_records": len(context["validation"]["inputs"]),
                   "train_inputs_sha256": context["records"]["train"]["inputs_sha256"],
                   "validation_inputs_sha256": context["records"]["validation"]["inputs_sha256"],
                   "training_code_sha256": code_sha256(TRAINING_CODE), "smoke": context["smoke"]}
    setting = {"microbatch": profile["microbatch"], "device": "cuda", "autocast": True,
               "directory": context["work"] / "runs" / name, "published": published}
    log(f"waiting for the GPU lock for {name}")
    with gpu_lock("cuda", path=context["lock"]):
        memory = free_memory(profile["peak_bytes"]["reserved"] + HEADROOM)
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        session = {"pid": os.getpid(), "started_unix": started, "status": "started", "git": context["git"],
                   "memory_before": memory, "ue_count_before": ue_count()}
        record_session(published / "sessions.json", session)
        log(f"start {name}: {memory['nvidia_smi']}")
        check = start_check(seed, context["validation"], context["cached"])
        result = train_run(rate, seed, context["train"], context["validation"], setting, fingerprint, check,
                           log)
        session.update({"status": "finished", "wall_seconds": time.time() - started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "ue_count_after": ue_count()})
        gc.collect()
        torch.cuda.empty_cache()
    record_session(published / "sessions.json", session)
    return result


def choose_rate(runs: dict[float, dict[str, Any]]) -> float:
    """
    Top rate with the highest best-point selection metric of seed 20261110; an exact tie goes to 1e-5.

    Parameters
    ----------
    runs : dict[float, dict[str, Any]]
        Rate to its seed-20261110 ``run.json``.

    Returns
    -------
    float
        The chosen rate.
    """
    chosen = None
    for rate in TOP_RATES:
        if rate in runs and (chosen is None or runs[rate]["best"]["selection_metric"]
                             > runs[chosen]["best"]["selection_metric"]):
            chosen = rate
    return chosen


def run_train(gpu: int, command: list[str], limits: dict[str, int | None], output: Path = DIRECTORY,
              work: Path = WORK) -> dict[str, Any]:
    """
    Grid with seed 20261110 at the rates the gate allows, the choice of rate, then the two other seeds.

    Parameters
    ----------
    gpu : int
        Physical GPU.
    command : list[str]
        Command line.
    limits : dict[str, int | None]
        Record limits (smoke tests only).
    output : Path
        Candidate directory holding ``profile.json``.
    work : Path
        Local work directory.

    Returns
    -------
    dict[str, Any]
        ``runs.json`` content: the rates run, the chosen rate and each run's best point.

    Raises
    ------
    RuntimeError
        If the training code changed since the profile.
    """
    smoke = any(value is not None for value in limits.values())
    git = committed(require=not smoke)
    profile = read_json(output / "profile.json")
    if profile["training_code_sha256"] != code_sha256(TRAINING_CODE) or profile["limits"] != limits:
        raise RuntimeError("the training code or the records changed since the profile; profile again")
    train, validation, records = open_data(limits)
    context = {"train": train, "validation": validation, "records": records,
               "cached": cached_xecg_features(validation["data"]), "profile": profile,
               "lock": gpu_lock_path(gpu), "git": git, "work": work, "output": output, "smoke": smoke}
    rates = profile["forecast"]["rates"]
    grid = {rate: one_run(rate, SEEDS[0], context) for rate in rates}
    chosen = choose_rate(grid)
    seeds = {SEEDS[0]: grid[chosen], **{seed: one_run(chosen, seed, context) for seed in SEEDS[1:]}}
    summary = {"command": command, "rates": rates, "full_plan": profile["forecast"]["full_plan"],
               "chosen_rate": chosen,
               "grid": {run_name(rate, SEEDS[0]): run["best"] for rate, run in grid.items()},
               "seeds": {str(seed): {"run": run_name(chosen, seed), **run["best"]}
                         for seed, run in seeds.items()},
               "first_seed_from_grid": True, "unix_time": time.time()}
    write_json_atomic(output / "runs.json", summary)
    return summary


def kept_probability(work: Path, rate: float, seed: int, point: int) -> np.ndarray:
    """
    Read the validation probabilities of a run at its kept point, as scored during training.

    Parameters
    ----------
    work : Path
        Local work directory.
    rate, seed : float, int
        Run.
    point : int
        Evaluation point.

    Returns
    -------
    np.ndarray
        Float64 ``[records, 14]``.
    """
    with np.load(work / "runs" / run_name(rate, seed) / f"validation__point-{point:02d}.npz") as archive:
        return archive["probability"]


def save_validation(output: Path, probability: np.ndarray, seeds: tuple[int, ...], validation: dict[str, Any],
                    suffix: str, smoke: bool) -> dict[str, Any]:
    """
    Write a validation prediction file and its per-source table.

    Parameters
    ----------
    output : Path
        Candidate directory.
    probability : np.ndarray
        Validation probabilities from ``predict``.
    seeds : tuple[int, ...]
        Seeds whose mean probability this is.
    validation : dict[str, Any]
        Output of ``open_data``.
    suffix : str
        Suffix of the per-source table (for one seed).
    smoke : bool
        Smoke tests hold a subset of the records and skip the prediction-file convention's key check.

    Returns
    -------
    dict[str, Any]
        ``selection_metric`` output with the prediction file and its SHA-256.
    """
    if output.name != CANDIDATE:
        raise ValueError(f"the candidate directory must be named {CANDIDATE}")
    data = validation["data"]
    path = prediction_path(output.parent, CANDIDATE, "validation", seeds)
    save_predictions(path, data["record_key"], probability, CANDIDATE, seeds)
    if not smoke:
        load_predictions(path, validation["keys"])
    write_rows(output / f"validation_by_source{suffix}.csv", validation_rows(data["labels"], probability,
                                                                             data["source"]))
    selection = selection_metric(data["labels"], probability, data["source"])
    return {**selection, "prediction_file": str(path), "prediction_sha256": sha256_file(path)}


def compared_candidates() -> dict[str, Any]:
    """
    Read the validation tables of H-X and P0, for the comparison in ``candidate.json``.

    Returns
    -------
    dict[str, Any]
        Per candidate: selection metric, per-source macro AUROC and per-condition AUROC.
    """
    keys = ("selection_metric", "macro_auroc_by_source", "per_condition")
    return {name: {key: read_json(CANDIDATES / name / "candidate.json")["validation"][key] for key in keys}
            for name in (BASE, "P0")}


def check_inputs_equal(validation: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """
    Recompute the validation inputs from the cached windows with ``predict``'s path and compare.

    Parameters
    ----------
    validation : dict[str, Any]
        Output of ``open_data``.
    smoke : bool
        Smoke test (subset of records).

    Returns
    -------
    dict[str, Any]
        ``windows`` (float32 ``[records, 12, 5000]``), ``inputs`` and whether they equal the prepared inputs.

    Raises
    ------
    ValueError
        If the recomputed inputs differ from the prepared ones.
    """
    windows = read_windows(("validation",), validation["data"]["record_key"])
    inputs = window_inputs(windows)
    if not np.array_equal(inputs, validation["inputs"]):
        raise ValueError("predict's input path differs from the prepared validation inputs")
    return {"windows": windows, "inputs": inputs, "identical": True, "smoke": smoke}


def seed_predictions(output: Path, work: Path, summary: dict[str, Any], validation: dict[str, Any],
                     inputs: dict[str, Any], smoke: bool) -> tuple[dict[str, Any], np.ndarray]:
    """
    Per-seed validation predictions from ``predict_inputs``, checked against the in-training probabilities.

    Parameters
    ----------
    output : Path
        Candidate directory with ``model.json``.
    work : Path
        Local work directory.
    summary : dict[str, Any]
        ``runs.json`` content.
    validation : dict[str, Any]
        Output of ``open_data``.
    inputs : dict[str, Any]
        Output of ``check_inputs_equal``.
    smoke : bool
        Smoke test.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        Per seed: validation record and the largest difference from training; and the per-seed probabilities.

    Raises
    ------
    ValueError
        If ``predict`` differs from the training probabilities by more than 1e-4.
    """
    by_seed, probabilities = {}, []
    for seed in SEEDS:
        kept = summary["seeds"][str(seed)]
        probability = predict_inputs(inputs["inputs"], output, seeds=(seed,), device="cuda")
        trained = kept_probability(work, summary["chosen_rate"], seed, kept["point"])
        difference = float(np.max(np.abs(probability - trained)))
        if difference > PREDICT_TOLERANCE:
            raise ValueError(f"predict differs from training for seed {seed} by {difference}")
        by_seed[str(seed)] = {"validation": save_validation(output, probability, (seed,), validation,
                                                            f"__seed-{seed}", smoke),
                              "max_abs_difference_from_training": difference, **kept}
        probabilities.append(probability)
    return by_seed, np.stack(probabilities)


def copy_weights(output: Path, work: Path, summary: dict[str, Any]) -> dict[str, str]:
    """
    Copy each seed's kept weights from its run into the candidate directory.

    Parameters
    ----------
    output : Path
        Candidate directory.
    work : Path
        Local work directory.
    summary : dict[str, Any]
        ``runs.json`` content.

    Returns
    -------
    dict[str, str]
        File name to SHA-256.

    Raises
    ------
    ValueError
        If a copy differs from the weights recorded at the kept point.
    """
    digests = {}
    for seed in SEEDS:
        source = work / "runs" / run_name(summary["chosen_rate"], seed) / "best.safetensors"
        name = f"weights__seed-{seed}.safetensors"
        shutil.copyfile(source, output / name)
        digests[name] = sha256_file(output / name)
        if digests[name] != summary["seeds"][str(seed)]["weights_sha256"]:
            raise ValueError(f"{name} differs from the weights kept during training")
    return digests


def run_finalize(gpu: int, command: list[str], limits: dict[str, int | None], output: Path = DIRECTORY,
                 work: Path = WORK) -> dict[str, Any]:
    """
    Kept checkpoints, ``model.json``, validation prediction files, ``candidate.json`` and the receipt.

    Parameters
    ----------
    gpu : int
        Physical GPU.
    command : list[str]
        Command line.
    limits : dict[str, int | None]
        Record limits (smoke tests only).
    output : Path
        Candidate directory.
    work : Path
        Local work directory.

    Returns
    -------
    dict[str, Any]
        ``candidate.json`` content.
    """
    smoke = any(value is not None for value in limits.values())
    git = committed(require=not smoke)
    started = time.time()
    profile, summary = read_json(output / "profile.json"), read_json(output / "runs.json")
    train, validation, records = open_data(limits)
    inputs = check_inputs_equal(validation, smoke)
    weights = copy_weights(output, work, summary)
    settings = {"top_rate": summary["chosen_rate"], "microbatch": profile["microbatch"],
                "kept_points": summary["seeds"], "recipe": RECIPE}
    index = write_model(output, SEEDS, settings)
    with gpu_lock("cuda", path=gpu_lock_path(gpu)):
        memory = free_memory(2 * GIB)
        torch.set_num_threads(4)
        by_seed, probabilities = seed_predictions(output, work, summary, validation, inputs, smoke)
        mean = np.mean(probabilities, axis=0)
        count = min(PREDICT_CHECK_RECORDS, len(mean))
        direct = predict(inputs["windows"][:count], output, device="cuda")
    direct_difference = float(np.max(np.abs(direct - mean[:count])))
    if direct_difference > PREDICT_TOLERANCE:
        raise ValueError(f"predict on waveforms differs from the mean of the seeds by {direct_difference}")
    candidate = {
        "candidate": CANDIDATE, "base": BASE, "recipe": RECIPE, "smoke": smoke, "limits": limits,
        "gate": profile["forecast"], "microbatch": profile["microbatch"],
        "rates_run": summary["rates"], "chosen": {"top_rate": summary["chosen_rate"]},
        "grid": summary["grid"],
        "seed_runs": by_seed, "first_seed_from_grid": summary["first_seed_from_grid"],
        "seeds": list(SEEDS), "train_records": len(train["inputs"]), "validation_records": len(mean),
        "validation": save_validation(output, mean, SEEDS, validation, "", smoke),
        "predict_on_waveforms": {"records": count, "max_abs_difference_from_mean": direct_difference},
        "comparison": compared_candidates(), "test_roles_scored": False,
    }
    write_json_atomic(output / "candidate.json", candidate, allow_nan=smoke)
    write_receipt(output, work, command, git, started, records, index, weights, memory, summary)
    return candidate


def run_sessions(output: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """
    Sessions and timings of every run used.

    Parameters
    ----------
    output : Path
        Candidate directory.
    summary : dict[str, Any]
        ``runs.json`` content.

    Returns
    -------
    dict[str, Any]
        Per run: its sessions (a killed one has no end), the wall seconds of finished sessions, the training
        and evaluation seconds of every evaluation point, the peak memory and the commits.
    """
    names = sorted({*summary["grid"], *(entry["run"] for entry in summary["seeds"].values())})
    result = {}
    for name in names:
        sessions = read_json(output / "runs" / name / "sessions.json")
        history = read_json(output / "runs" / name / "run.json")["history"]
        finished = [item for item in sessions if item["status"] == "finished"]
        result[name] = {
            "sessions": sessions, "killed_sessions": len(sessions) - len(finished),
            "finished_wall_seconds": sum(item["wall_seconds"] for item in finished),
            "train_seconds": sum(row.get("train_seconds", 0.0) for row in history),
            "evaluation_seconds": sum(row["evaluation_seconds"] for row in history),
            "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in finished),
            "commits": sorted({item["git"]["commit"] for item in sessions})}
    return result


def write_receipt(output: Path, work: Path, command: list[str], git: dict[str, Any], started: float,
                  records: dict[str, Any], index: dict[str, Any], weights: dict[str, str],
                  memory: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """
    Receipt of candidate F.

    Parameters
    ----------
    output : Path
        Candidate directory.
    work : Path
        Local work directory.
    command : list[str]
        Finalize command line.
    git : dict[str, Any]
        Commit and code hashes at finalize.
    started : float
        Finalize start time.
    records : dict[str, Any]
        Preparation records of the inputs.
    index : dict[str, Any]
        ``model.json`` content.
    weights : dict[str, str]
        Checkpoint SHA-256.
    memory : dict[str, Any]
        GPU memory at finalize.
    summary : dict[str, Any]
        ``runs.json`` content.

    Returns
    -------
    dict[str, Any]
        The receipt.
    """
    predictions = sorted(output.glob("validation__seeds-*.npz"))
    base_files = ("model.json", *read_json(CANDIDATES / BASE / "model.json")["files"])
    receipt = {
        "command": command, "finalize_started_unix": started, "finalize_wall_seconds": time.time() - started,
        "git": git, "runs": run_sessions(output, summary),
        "inputs": {"manifest": to_stored(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                   "waveform_cache": str(CACHE), "prepared_inputs": records,
                   "stage1_features": feature_hashes(),
                   "base_candidate": {name: sha256_file(CANDIDATES / BASE / name) for name in base_files},
                   "xecg_release_sha256": RELEASE_SHA256, "work_directory": str(work)},
        "outputs": {"model.json": sha256_file(output / "model.json"), **weights,
                    **{path.name: sha256_file(path) for path in predictions},
                    "profile.json": sha256_file(output / "profile.json"),
                    "runs.json": sha256_file(output / "runs.json")},
        "model_index": index,
        "environment": {**environment(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "safetensors": importlib.metadata.version("safetensors"),
                        "xlstm": importlib.metadata.version("xlstm"), "torch_flags": torch_flags(),
                        "cuda_home": os.environ.get("CUDA_HOME"), "pythonpath": os.environ.get("PYTHONPATH"),
                        "executable": sys.executable},
        "gpu_at_finalize": memory, "ue_count": ue_count(),
        "validation_sources": list(VALIDATION_SOURCES), "conditions": list(CONDITIONS),
        "test_roles_scored": False, "test_role_waveforms_loaded": False,
    }
    write_json_atomic(output / "receipt.json", receipt)
    return receipt
