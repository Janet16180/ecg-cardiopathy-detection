"""Pooled encoder features and the 130 measurements of every MLC-01 record, read from the MLC-02 cache.

Each extractor gets the input path of the cached features it is checked against:

- xECG: the float32 window widened to float64, FFT-resampled to 100 Hz, time-major, vanilla xLSTM backend,
  pooled 1,024 (``external_encoders.xecg_input`` and ``xecg_features``, microbatches of 16).
- ECG-JEPA: leads I, II, V1-V6 of the float32 window, FFT-resampled to 250 Hz, pooled 768
  (``external_encoders.jepa_input`` and ``jepa_features``, batches of 4).
- ECG-CPC: four 2.5 s float32 crops, ``resampy`` to 240 Hz, pooled 512 (``mlc02_cpc``, batches of 16 records).
- Measurements: ``measurement_extractor042.extract_array`` on the window as float64 ``(5000, 12)`` in C order,
  with its ``status``, ``implausible`` and ``flat_leads`` fields; a measurement it does not return is NaN.

Samples WFDB reads as missing (the format-16 value -32768, the negative amplifier rail at gain 1000) are given
that stored value, -32.768 mV, before any extractor; no other sample is changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecg_experiment import ROOT
from ecg_experiment.cpu_screening041 import RAW_FEATURES
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.measurement_extractor042 import extract_array
from ecg_experiment.mlc02_waveforms import CACHE, RAIL_MV, index_sha256, load_cache, windows

FEATURES = Path("/var/tmp/janetrivera/mlc02/features")
DIMENSIONS = {"xecg": 1024, "jepa": 768, "cpc": 512, "measurements": len(RAW_FEATURES)}
ENCODERS = ("xecg", "jepa", "cpc")
GPU_LOCKS = {0: Path("/tmp/ecg_project_gpu.lock"), 1: Path("/tmp/ecg_project_gpu1.lock")}
SHARD = 2048
PROFILE_PER_SOURCE = 72
STATUS_COLUMNS = ["status", "implausible", "flat_leads"]
JEPA_DEPENDENCIES = ROOT / "third_party/jepa-deps"
CHECKPOINTS = {
    "xecg": [
        "third_party/checkpoints/xecg/config.json",
        "third_party/checkpoints/xecg/xECG.py",
        "third_party/checkpoints/xecg/model.safetensors",
    ],
    "jepa": ["third_party/checkpoints/ecg-jepa/multiblock_epoch100.pth"],
    "cpc": ["third_party/checkpoints/ecg-cpc/ECG-CPC Checkpoint/last_11597276.ckpt"],
    "measurements": [],
}
CODE = {
    "common": [
        "ecg_experiment/mlc02_features.py",
        "ecg_experiment/mlc02_waveforms.py",
        "ecg_experiment/files.py",
        "ecg_experiment/gpu.py",
        "scripts/experiments/run_mlc02_inputs.py",
    ],
    "xecg": [
        "ecg_experiment/external_encoders.py",
        "ecg_experiment/xecg.py",
        "ecg_experiment/cuda_runtime20261028.py",
    ],
    "jepa": [
        "ecg_experiment/external_encoders.py",
        "third_party/ECG_JEPA/models.py",
        "third_party/ECG_JEPA/ecg_jepa.py",
    ],
    "cpc": ["ecg_experiment/mlc02_cpc.py", "scripts/features/extract_ecg_cpc.py"],
    "measurements": ["ecg_experiment/measurement_extractor042.py", "ecg_experiment/cpu_screening041.py"],
}
ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "OPENBLAS_CORETYPE",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "PYTHONPATH",
    "CUBLAS_WORKSPACE_CONFIG",
)
CACHE_STATE: dict[str, object] = {}


def model_input(window: np.ndarray) -> np.ndarray:
    """
    Set the missing samples of a cached window to the stored negative-rail value.

    Parameters
    ----------
    window : np.ndarray
        Float32 ``[12, 5000]`` cached window.

    Returns
    -------
    np.ndarray
        Finite float32 ``[12, 5000]`` copy.
    """
    return np.where(np.isnan(window), np.float32(RAIL_MV), window).astype(np.float32)


def encoder_inputs(name: str, batch: np.ndarray) -> np.ndarray:
    """
    Model inputs of a batch of cached windows.

    Parameters
    ----------
    name : str
        One of ``ENCODERS``.
    batch : np.ndarray
        Float32 ``[records, 12, 5000]`` windows.

    Returns
    -------
    np.ndarray
        ``[records, 1000, 12]`` (xECG), ``[records, 8, 2500]`` (ECG-JEPA) or ``[records, 4, 12, 600]``
        (ECG-CPC).
    """
    from ecg_experiment.external_encoders import jepa_input, xecg_input
    from ecg_experiment.mlc02_cpc import cpc_input

    function = {"xecg": xecg_input, "jepa": jepa_input, "cpc": cpc_input}[name]
    return np.stack([function(model_input(window)) for window in batch])


def load_model(name: str) -> torch.nn.Module:
    """
    Load one frozen released encoder on the visible GPU.

    Parameters
    ----------
    name : str
        One of ``ENCODERS``.

    Returns
    -------
    torch.nn.Module
        Encoder in evaluation mode without gradients.

    Raises
    ------
    ValueError
        If the model is not frozen.
    """
    if name == "xecg":
        from ecg_experiment.cuda_runtime20261028 import configure_cuda_runtime
        from ecg_experiment.external_encoders import load_xecg_backbone

        configure_cuda_runtime()
        model = load_xecg_backbone()
    elif name == "jepa":
        from ecg_experiment.external_encoders import load_jepa

        if str(JEPA_DEPENDENCIES) not in sys.path:
            sys.path.insert(0, str(JEPA_DEPENDENCIES))
        model = load_jepa()
    else:
        from ecg_experiment.mlc02_cpc import ReleasedCPC

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecg-cpc-mpl")
        os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/ecg-cpc-keops")
        model = ReleasedCPC().to("cuda")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"{name} is not frozen")
    return model


def encode(name: str, model: torch.nn.Module, inputs: np.ndarray) -> np.ndarray:
    """
    Pooled features of prepared inputs with each encoder's own batch size.

    Parameters
    ----------
    name : str
        One of ``ENCODERS``.
    model : torch.nn.Module
        Output of ``load_model``.
    inputs : np.ndarray
        Output of ``encoder_inputs``.

    Returns
    -------
    np.ndarray
        Float32 ``[records, DIMENSIONS[name]]``.
    """
    from ecg_experiment.external_encoders import jepa_features, xecg_features
    from ecg_experiment.mlc02_cpc import cpc_features

    function = {"xecg": xecg_features, "jepa": jepa_features, "cpc": cpc_features}[name]
    return function(model, inputs)


def measure(batch: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the 130 measurements and the extractor status of a batch of cached windows.

    Parameters
    ----------
    batch : np.ndarray
        Float32 ``[records, 12, 5000]`` windows.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Float64 ``[records, 130]`` in ``RAW_FEATURES`` order (NaN where the extractor returned no value) and
        ``status``, ``implausible`` and ``flat_leads`` per record.
    """
    rows = [extract_array(np.ascontiguousarray(model_input(window).T, dtype=np.float64)) for window in batch]
    values = np.array([[row.get(name, np.nan) for name in RAW_FEATURES] for row in rows], dtype=np.float64)
    status = pd.DataFrame(
        [{name: row[name] for name in STATUS_COLUMNS} for row in rows], columns=STATUS_COLUMNS
    )
    return values, status


def shard_bounds(records: int, size: int) -> list[tuple[int, int]]:
    """
    Consecutive shards covering every record once.

    Parameters
    ----------
    records : int
        Number of records.
    size : int
        Records per shard.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, stop)`` per shard.
    """
    return [(start, min(start + size, records)) for start in range(0, records, size)]


def profile_positions(index: pd.DataFrame, per_source: int = PROFILE_PER_SOURCE) -> np.ndarray:
    """
    Select the first records of every source in cache order, for a profile.

    Parameters
    ----------
    index : pd.DataFrame
        Cache index.
    per_source : int
        Records per source.

    Returns
    -------
    np.ndarray
        Sorted global positions.
    """
    return np.sort(index.groupby("source", sort=False).head(per_source)["position"].to_numpy())


def plan_identity(name: str, index: pd.DataFrame, positions: np.ndarray, shard: int) -> dict[str, object]:
    """
    Everything that determines an extraction's outputs, for resuming.

    Parameters
    ----------
    name : str
        Extractor name.
    index : pd.DataFrame
        Cache index.
    positions : np.ndarray
        Global positions to extract, in output order.
    shard : int
        Records per shard.

    Returns
    -------
    dict[str, object]
        Extractor, positions digest, shard size, cache index identity and code and checkpoint SHA-256.
    """
    code = CODE["common"] + CODE[name]
    return {
        "extractor": name,
        "records": len(positions),
        "shard": shard,
        "positions_sha256": sha256_bytes(np.asarray(positions, dtype=np.int64).tobytes()),
        "index_identity_sha256": index_sha256(index),
        "code_sha256": {path: sha256_file(ROOT / path) for path in code},
        "checkpoint_sha256": {path: sha256_file(ROOT / path) for path in CHECKPOINTS[name]},
    }


def sha256_bytes(data: bytes) -> str:
    """
    SHA-256 of a byte string.

    Parameters
    ----------
    data : bytes
        Bytes to hash.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(data).hexdigest()


def prepare(directory: Path, identity: dict[str, object]) -> None:
    """
    Create an extraction directory or require that an existing one has the same plan.

    Parameters
    ----------
    directory : Path
        Extraction directory.
    identity : dict[str, object]
        Output of ``plan_identity``.

    Raises
    ------
    ValueError
        If the directory holds an extraction with another plan.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"{directory} holds an extraction with another plan")
        return
    write_json_atomic(path, identity)


def save_shard(
    directory: Path, number: int, values: np.ndarray, status: pd.DataFrame | None, details: dict[str, object]
) -> None:
    """
    Write one shard atomically; its JSON is written last and marks it complete.

    Parameters
    ----------
    directory : Path
        Extraction directory.
    number : int
        Shard number.
    values : np.ndarray
        Shard features.
    status : pd.DataFrame | None
        Measurement status rows, or None for an encoder.
    details : dict[str, object]
        Timing and bounds, written as ``shard_<number>.json``.
    """
    stem = directory / f"shard_{number:05d}"
    temporary = stem.with_suffix(".tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, stem.with_suffix(".npy"))
    if status is not None:
        status.to_csv(stem.with_suffix(".tmp.csv"), index=False)
        os.replace(stem.with_suffix(".tmp.csv"), stem.with_suffix(".csv"))
    write_json_atomic(stem.with_suffix(".json"), details)


def pending_shards(directory: Path, bounds: list[tuple[int, int]]) -> list[int]:
    """
    Shards without a completion JSON.

    Parameters
    ----------
    directory : Path
        Extraction directory.
    bounds : list[tuple[int, int]]
        Output of ``shard_bounds``.

    Returns
    -------
    list[int]
        Shard numbers still to run.
    """
    return [number for number in range(len(bounds)) if not (directory / f"shard_{number:05d}.json").exists()]


def torch_flags() -> dict[str, object]:
    """
    Torch numerical settings that can change GPU results.

    Returns
    -------
    dict[str, object]
        TF32, cuDNN and determinism flags.
    """
    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def run_encoder(
    name: str,
    directory: Path,
    gpu: int,
    positions: np.ndarray | None,
    shard: int = SHARD,
    required_free: int = 2 * 1024**3,
    cache: Path = CACHE,
) -> dict[str, object]:
    """
    Extract one encoder's pooled features shard by shard under the GPU's project lock.

    Parameters
    ----------
    name : str
        One of ``ENCODERS``.
    directory : Path
        Extraction directory.
    gpu : int
        Physical GPU index; ``CUDA_VISIBLE_DEVICES`` must name it alone.
    positions : np.ndarray | None
        Global positions to extract (a profile), or None for every record.
    shard : int
        Records per shard.
    required_free : int
        Free GPU bytes required before loading the model.
    cache : Path
        Waveform cache directory.

    Returns
    -------
    dict[str, object]
        Run record written to ``run_<start time>.json``.

    Raises
    ------
    RuntimeError
        If the visible GPU is not ``gpu`` or has less free memory than required.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={gpu} for GPU {gpu}")
    index, blocks = load_cache(cache)
    positions = index["position"].to_numpy() if positions is None else np.asarray(positions)
    identity = plan_identity(name, index, positions, shard)
    prepare(directory, identity)
    bounds = shard_bounds(len(positions), shard)
    pending = pending_shards(directory, bounds)
    started = time.time()
    with gpu_lock("cuda", blocking=False, path=GPU_LOCKS[gpu]):
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        if free < required_free:
            raise RuntimeError(f"GPU {gpu}: {free} bytes free, {required_free} required")
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = load_model(name)
        load_seconds = time.perf_counter() - load_started
        for number in pending:
            start, stop = bounds[number]
            clock = time.perf_counter()
            batch = windows(index, blocks, positions[start:stop])
            read = time.perf_counter() - clock
            inputs = encoder_inputs(name, batch)
            prepared = time.perf_counter() - clock
            values = encode(name, model, inputs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - clock
            save_shard(
                directory,
                number,
                values,
                None,
                {
                    "start": start,
                    "stop": stop,
                    "seconds": seconds,
                    "read_seconds": read,
                    "input_seconds": prepared - read,
                },
            )
            print(
                f"{name} shard {number + 1}/{len(bounds)} {stop - start} records {seconds:.1f} s", flush=True
            )
        torch.cuda.synchronize()
        peak = {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }
        del model
        torch.cuda.empty_cache()
    record = {
        "extractor": name,
        "gpu": gpu,
        "gpu_name": torch.cuda.get_device_name(),
        "free_bytes_before": int(free),
        "total_bytes": int(total),
        "peak_gpu_bytes": peak,
        "model_load_seconds": load_seconds,
        "shards_run": pending,
        "wall_seconds": time.time() - started,
        "torch_flags": torch_flags(),
        "environment": {key: os.environ.get(key, "") for key in ENVIRONMENT},
    }
    write_json_atomic(directory / f"run_{int(started)}.json", record)
    return record


def init_measurement_worker(cache: str) -> None:
    """
    Open the cache once per measurement process.

    Parameters
    ----------
    cache : str
        Waveform cache directory.
    """
    CACHE_STATE["index"], CACHE_STATE["blocks"] = load_cache(Path(cache))


def measurement_shard(task: dict[str, object]) -> int:
    """
    Measure and save one shard in a worker process.

    Parameters
    ----------
    task : dict[str, object]
        ``directory``, ``number``, ``start``, ``stop`` and ``positions``.

    Returns
    -------
    int
        The shard number.
    """
    clock = time.perf_counter()
    batch = windows(CACHE_STATE["index"], CACHE_STATE["blocks"], np.asarray(task["positions"]))
    read = time.perf_counter() - clock
    values, status = measure(batch)
    save_shard(
        Path(task["directory"]),
        task["number"],
        values,
        status,
        {
            "start": task["start"],
            "stop": task["stop"],
            "seconds": time.perf_counter() - clock,
            "read_seconds": read,
        },
    )
    return int(task["number"])


def run_measurements(
    directory: Path, workers: int, positions: np.ndarray | None, shard: int = 512, cache: Path = CACHE
) -> dict[str, object]:
    """
    Extract the 130 measurements shard by shard in a process pool.

    Parameters
    ----------
    directory : Path
        Extraction directory.
    workers : int
        Processes.
    positions : np.ndarray | None
        Global positions to extract (a profile), or None for every record.
    shard : int
        Records per shard.
    cache : Path
        Waveform cache directory.

    Returns
    -------
    dict[str, object]
        Run record written to ``run_<start time>.json``.
    """
    index, _ = load_cache(cache)
    positions = index["position"].to_numpy() if positions is None else np.asarray(positions)
    identity = plan_identity("measurements", index, positions, shard)
    prepare(directory, identity)
    bounds = shard_bounds(len(positions), shard)
    tasks = [
        {
            "directory": str(directory),
            "number": number,
            "start": bounds[number][0],
            "stop": bounds[number][1],
            "positions": positions[bounds[number][0] : bounds[number][1]].tolist(),
        }
        for number in pending_shards(directory, bounds)
    ]
    started = time.time()
    with Pool(workers, initializer=init_measurement_worker, initargs=(str(cache),)) as pool:
        for count, number in enumerate(pool.imap_unordered(measurement_shard, tasks)):
            print(
                f"measurements shard {number} ({count + 1}/{len(tasks)}) {time.time() - started:.1f} s",
                flush=True,
            )
    record = {
        "extractor": "measurements",
        "workers": workers,
        "shards_run": [task["number"] for task in tasks],
        "wall_seconds": time.time() - started,
        "environment": {key: os.environ.get(key, "") for key in ENVIRONMENT},
    }
    write_json_atomic(directory / f"run_{int(started)}.json", record)
    return record


def assemble(name: str, directory: Path) -> tuple[np.ndarray, pd.DataFrame | None, dict[str, object]]:
    """
    Concatenate the shards of a finished extraction in output order.

    Parameters
    ----------
    name : str
        Extractor name.
    directory : Path
        Extraction directory.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame | None, dict[str, object]]
        Features ``[records, DIMENSIONS[name]]``, measurement status (or None) and the summed shard timing.

    Raises
    ------
    ValueError
        If a shard is missing or has the wrong shape.
    """
    plan = json.loads((directory / "plan.json").read_text())
    bounds = shard_bounds(plan["records"], plan["shard"])
    if pending_shards(directory, bounds):
        raise ValueError(f"{directory} has unfinished shards")
    values, status, seconds = [], [], 0.0
    for number, (start, stop) in enumerate(bounds):
        stem = directory / f"shard_{number:05d}"
        part = np.load(stem.with_suffix(".npy"))
        if part.shape != (stop - start, DIMENSIONS[name]):
            raise ValueError(f"Shard {stem} has shape {part.shape}")
        values.append(part)
        seconds += json.loads(stem.with_suffix(".json").read_text())["seconds"]
        if name == "measurements":
            status.append(
                pd.read_csv(
                    stem.with_suffix(".csv"), dtype={"status": str, "implausible": str}, keep_default_na=False
                )
            )
    table = pd.concat(status, ignore_index=True) if status else None
    return np.concatenate(values), table, {"shard_seconds": seconds, "shards": len(bounds)}
