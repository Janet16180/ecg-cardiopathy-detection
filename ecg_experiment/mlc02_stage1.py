"""Stage 1 of MLC-02: extractor profiles and forecasts, full runs, publication with a receipt, integrity."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.cpu_screening041 import RAW_FEATURES
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc02_cache_build import git_state, other_manifests
from ecg_experiment.mlc02_features import (
    DIMENSIONS,
    ENCODERS,
    FEATURES,
    assemble,
    profile_positions,
    run_encoder,
    run_measurements,
)
from ecg_experiment.mlc02_integrity import CACHES, check_cache, key04_reproduction
from ecg_experiment.mlc02_waveforms import CACHE, MANIFEST, read_identity
from ecg_experiment.paths import to_stored

OUTPUT = ROOT / "outputs/mlc02_condition_models_20261110"
PUBLISHED = OUTPUT / "features"
PACKAGES = ("numpy", "scipy", "pandas", "torch", "wfdb", "h5py", "resampy", "numba", "xlstm", "scikit-learn")
MARGIN = 1024**3


def package_versions() -> dict[str, str]:
    """
    Versions of the packages that shape the features, where installed.

    Returns
    -------
    dict[str, str]
        Version per package; ``absent`` when not importable by metadata.
    """
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def profile(name: str, gpu: int | None, workers: int) -> dict[str, object]:
    """
    Run one extractor on the profile records and write its forecast for every record.

    Parameters
    ----------
    name : str
        Extractor name.
    gpu : int | None
        Physical GPU for an encoder.
    workers : int
        Processes for the measurements.

    Returns
    -------
    dict[str, object]
        Profile record written to ``profile/<name>/profile.json``.
    """
    index = pd.read_csv(CACHE / "index.csv", dtype={"record_key": str})
    positions = profile_positions(index)
    directory = FEATURES / "profile" / name
    started = time.perf_counter()
    if name == "measurements":
        run = run_measurements(directory, workers, positions, shard=64)
    else:
        run = run_encoder(name, directory, gpu, positions, shard=len(positions))
    wall = time.perf_counter() - started
    _, _, timing = assemble(name, directory)
    per_record = timing["shard_seconds"] / len(positions)
    parallel = workers if name == "measurements" else 1
    forecast = per_record * len(index) / parallel + run.get("model_load_seconds", 0.0)
    record = {
        "extractor": name,
        "records": len(positions),
        "wall_seconds": wall,
        "per_record_seconds": per_record,
        "forecast_seconds_all_records": forecast,
        "forecast_parallel_processes": parallel,
        "all_records": len(index),
        "run": run,
    }
    write_json_atomic(directory / "profile.json", record)
    return record


def extract(name: str, gpu: int | None, workers: int) -> dict[str, object]:
    """
    Run one extractor on every record after its profile, with the memory gate of the profile.

    Parameters
    ----------
    name : str
        Extractor name.
    gpu : int | None
        Physical GPU for an encoder.
    workers : int
        Processes for the measurements.

    Returns
    -------
    dict[str, object]
        Run record of the extraction.

    Raises
    ------
    FileNotFoundError
        If the extractor has no profile.
    """
    profiled = json.loads((FEATURES / "profile" / name / "profile.json").read_text())
    if name == "measurements":
        return run_measurements(FEATURES / name, workers, None)
    required = profiled["run"]["peak_gpu_bytes"]["reserved"] + MARGIN
    return run_encoder(name, FEATURES / name, gpu, None, required_free=required)


def runs(directory: Path) -> list[dict[str, object]]:
    """
    Run records of an extraction directory in time order.

    Parameters
    ----------
    directory : Path
        Extraction directory.

    Returns
    -------
    list[dict[str, object]]
        Contents of every ``run_*.json``.
    """
    return [json.loads(path.read_text()) for path in sorted(directory.glob("run_*.json"))]


def missing_counts(index: pd.DataFrame, values: np.ndarray, status: pd.DataFrame) -> list[dict[str, object]]:
    """
    Measurement status and missing values per source and role.

    Parameters
    ----------
    index : pd.DataFrame
        Cache index.
    values : np.ndarray
        ``[records, 130]`` measurements in index order.
    status : pd.DataFrame
        Extractor status in index order.

    Returns
    -------
    list[dict[str, object]]
        Per source and role: records, records per status, records with any missing measurement, records with
        all 130 missing, and missing values.
    """
    frame = index[["source", "role"]].assign(
        status=status["status"].to_numpy(),
        any_missing=np.isnan(values).any(axis=1),
        all_missing=np.isnan(values).all(axis=1),
        missing_values=np.isnan(values).sum(axis=1),
    )
    rows = []
    for (source, role), part in frame.groupby(["source", "role"], sort=False):
        counts = part["status"].value_counts()
        rows.append(
            {
                "source": source,
                "role": role,
                "records": len(part),
                **{
                    f"status_{key}": int(counts.get(key, 0))
                    for key in ("ok", "derivacion_plana", "sin_latidos")
                },
                "records_any_missing": int(part["any_missing"].sum()),
                "records_all_missing": int(part["all_missing"].sum()),
                "missing_values": int(part["missing_values"].sum()),
            }
        )
    return rows


def publish(command: list[str]) -> dict[str, object]:
    """
    Assemble every extractor, write the arrays locally and copy them with a receipt to the outputs.

    Parameters
    ----------
    command : list[str]
        Command line, for the receipt.

    Returns
    -------
    dict[str, object]
        The receipt.

    Raises
    ------
    ValueError
        If a later manifest differs in identity, or an array is malformed.
    """
    index = pd.read_csv(CACHE / "index.csv", dtype={"record_key": str})
    identity = read_identity(MANIFEST)
    if not identity["record_key"].reset_index(drop=True).equals(index["record_key"]):
        raise ValueError("Cache index differs from the manifest order")
    manifests = {
        "manifest": to_stored(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "other_manifests": other_manifests(),
    }
    local = FEATURES / "final"
    local.mkdir(parents=True, exist_ok=True)
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    extractors = {}
    for name in (*ENCODERS, "measurements"):
        values, status, timing = assemble(name, FEATURES / name)
        if values.shape != (len(index), DIMENSIONS[name]):
            raise ValueError(f"{name}: shape {values.shape}")
        if name != "measurements" and not np.isfinite(values).all():
            raise ValueError(f"{name}: nonfinite features")
        np.save(local / f"{name}.npy", values)
        if status is not None:
            status.insert(0, "record_key", index["record_key"].to_numpy())
            status.to_csv(local / "measurements_status.csv", index=False)
            extractors["measurements_missing"] = missing_counts(index, values, status)
        profiled = json.loads((FEATURES / "profile" / name / "profile.json").read_text())
        extractors[name] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "plan": json.loads((FEATURES / name / "plan.json").read_text()),
            "runs": runs(FEATURES / name),
            "timing": timing,
            "profile": profiled,
        }
    rows = index[
        [
            "position",
            "record_key",
            "source",
            "role",
            "block",
            "row",
            "sha256_10s",
            "sha256_f32",
            "missing_samples_10s",
            "flat_leads_10s",
        ]
    ]
    rows.to_csv(local / "rows.csv", index=False)
    files = [*(f"{name}.npy" for name in (*ENCODERS, "measurements")), "measurements_status.csv", "rows.csv"]
    for file in files:
        shutil.copyfile(local / file, PUBLISHED / file)
    receipt = {
        "stage": "mlc02_stage1_features",
        "command": command,
        "records": len(index),
        **manifests,
        "waveform_cache": {
            "directory": str(CACHE),
            "receipt_sha256": sha256_file(CACHE / "receipt.json"),
            "index_sha256": sha256_file(CACHE / "index.csv"),
            "receipt": json.loads((CACHE / "receipt.json").read_text()),
        },
        "files_sha256": {file: sha256_file(PUBLISHED / file) for file in files},
        "local_copy_sha256_equal": all(
            sha256_file(local / file) == sha256_file(PUBLISHED / file) for file in files
        ),
        "measurement_columns": list(RAW_FEATURES),
        "extractors": extractors,
        "git": git_state(),
        "python": sys.version,
        "packages": package_versions(),
        "rules": {
            "missing_samples": "WFDB missing samples (format-16 value -32768, gain 1000) set to -32.768 mV "
            "before every extractor",
            "row_order": "rows.csv: included MLC-01 manifest records in manifest order",
        },
    }
    write_json_atomic(PUBLISHED / "receipt.json", receipt)
    return receipt


def integrity(features: Path, output: Path, positions: np.ndarray | None) -> pd.DataFrame:
    """
    Compare new features with every earlier cache and write the table.

    Parameters
    ----------
    features : Path
        Directory with ``<extractor>.npy`` in index order (published), or the profile root.
    output : Path
        Directory for ``integrity.csv``, ``integrity.json`` and ``integrity_worst.csv``.
    positions : np.ndarray | None
        Profile positions when ``features`` is the profile root, else None.

    Returns
    -------
    pd.DataFrame
        One row per cache and extractor.
    """
    index = pd.read_csv(CACHE / "index.csv", dtype={"record_key": str})
    if positions is not None:
        index = index.iloc[positions].reset_index(drop=True)
    loaded = {}
    for name in (*ENCODERS, "measurements"):
        if positions is None:
            loaded[name] = np.load(features / f"{name}.npy", mmap_mode="r")
        elif (features / name / "plan.json").exists():
            loaded[name] = assemble(name, features / name)[0]
    rows, worst = [], []
    for spec in CACHES:
        if spec["extractor"] in loaded:
            row, bad = check_cache(spec, index, np.asarray(loaded[spec["extractor"]]))
            rows.append(row)
            worst.append(bad)
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "integrity.csv", index=False)
    pd.concat(worst, ignore_index=True).to_csv(output / "integrity_worst.csv", index=False)
    write_json_atomic(output / "integrity.json", {"rows": rows, "git": git_state()})
    return table


def attribution(gpu: int, workers: int, output: Path) -> dict[str, object]:
    """
    Recompute PTB-XL measurements and the first xECG shard from raw float64 samples and write the record.

    Parameters
    ----------
    gpu : int
        Physical GPU for xECG; ``CUDA_VISIBLE_DEVICES`` must name it alone.
    workers : int
        Processes for the measurements.
    output : Path
        Directory for ``attribution.json``.

    Returns
    -------
    dict[str, object]
        Outputs of ``measurement_attribution`` and ``xecg_attribution`` with timing.

    Raises
    ------
    RuntimeError
        If the visible GPU is not ``gpu``.
    """
    import os

    import torch

    from ecg_experiment.gpu import gpu_lock
    from ecg_experiment.mlc02_attribution import measurement_attribution, xecg_attribution
    from ecg_experiment.mlc02_features import GPU_LOCKS, SHARD, load_model

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={gpu} for GPU {gpu}")
    started = time.perf_counter()
    index = pd.read_csv(CACHE / "index.csv", dtype={"record_key": str, "raw_path": str})
    record = {
        "measurements": measurement_attribution(
            index, assemble("measurements", FEATURES / "measurements")[0], workers
        )
    }
    with gpu_lock("cuda", blocking=False, path=GPU_LOCKS[gpu]):
        torch.set_num_threads(2)
        model = load_model("xecg")
        record["xecg"] = xecg_attribution(
            index, assemble("xecg", FEATURES / "xecg")[0], np.arange(SHARD), model
        )
        del model
        torch.cuda.empty_cache()
    record |= {"wall_seconds": time.perf_counter() - started, "git": git_state()}
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "attribution.json", record)
    return record


def key04(output: Path) -> dict[str, object]:
    """
    Reproduce KEY-04's SPH macro AUROC and write the record.

    Parameters
    ----------
    output : Path
        Directory for ``key04_reproduction_<OpenBLAS kernel>.json``.

    Returns
    -------
    dict[str, object]
        Output of ``key04_reproduction`` with timing.
    """
    started = time.perf_counter()
    record = key04_reproduction() | {"wall_seconds": time.perf_counter() - started, "git": git_state()}
    output.mkdir(parents=True, exist_ok=True)
    name = "_".join(record["openblas_architecture"]) or "unknown"
    write_json_atomic(output / f"key04_reproduction_{name}.json", record)
    return record
