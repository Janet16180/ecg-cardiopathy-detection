"""
Run MLC-04, the one final test of the 14-condition classifier on the Challenge test groups.

Every step takes ``--role validation`` (the dry run on the Challenge calibration groups) or
``--role challenge_test`` (refused by every loader until ``docs/mlc04-freeze-record-2026-11-11.md`` is
committed with a clean tracked tree):

- ``scan``: MLC-01 scan, labels, exclusions and duplicates of the role (``manifest_v4.csv``).
- ``cache``: the first 10 s of every eligible record, as the MLC-02 cache stores them.
- ``measure``: Part A measures and flags, method B and the whole-record windows.
- ``features --extractor NAME [--gpu N]``: one MLC-02 extractor; ``publish`` concatenates the four.
- ``rules``: method A, the analysis table and rules C1-C12 (``manifest_v5.csv``, ``rule_flow.csv``).
- ``thresholds``: the frozen thresholds of P0-v5 (no role, written once).
- ``check``: the dry-run reproductions (validation only).
- ``score``: the frozen models' outputs and every metric. On the test groups it writes a marker before the
  first score and refuses a second run; ``--rerun-after-crash`` continues only an unfinished read whose
  partial outputs were deleted.

Run from the worktree, CPU steps with at most 8 processes:
``PYTHONPATH=<worktree> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m scripts.experiments.run_mlc04_final_test <step> --role <role>``.
Each step runs with the OpenBLAS kernel of the code it reproduces (``KERNELS``): the MLC-03 steps
(``measure``, ``rules``) with the machine's default kernel, the MLC-02 extractors and the model scores
(``features``, ``check``, ``score``) with ``OPENBLAS_CORETYPE=Haswell``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment import mlc04_data as md
from ecg_experiment import mlc04_models as mm
from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.mlc01_build import manifest_table
from ecg_experiment.mlc01_dataset import SPLIT_TABLE, read_split_table
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_data import CHALLENGE_SOURCES, CONDITIONS
from ecg_experiment.mlc02_feature_data import PUBLISHED
from ecg_experiment.mlc02_feature_models import CANDIDATES
from ecg_experiment.mlc02_features import CHECKPOINTS
from ecg_experiment.mlc04_checks import run_checks
from ecg_experiment.mlc04_evaluation import evaluate, json_ready
from ecg_experiment.paths import to_stored

CODE = (
    "ecg_experiment/mlc04_data.py",
    "ecg_experiment/mlc04_models.py",
    "ecg_experiment/mlc04_evaluation.py",
    "ecg_experiment/mlc04_checks.py",
    "scripts/experiments/run_mlc04_final_test.py",
    "ecg_experiment/mlc_labels.py",
    "ecg_experiment/mlc01_dataset.py",
    "ecg_experiment/mlc01_build.py",
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc02_features.py",
    "ecg_experiment/mlc02_cpc.py",
    "ecg_experiment/external_encoders.py",
    "ecg_experiment/measurement_extractor042.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/mlc02_bootstrap.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/mlc03_signal_quality.py",
    "ecg_experiment/mlc03_signal_flags.py",
    "ecg_experiment/mlc03_measurements.py",
    "ecg_experiment/mlc03_label_quality.py",
    "ecg_experiment/mlc03_dataset_v5.py",
    "ecg_experiment/key05_keys.py",
    "ecg_experiment/key05_analysis.py",
)
ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_CORETYPE",
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
)
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "wfdb", "h5py", "pyarrow", "torch")
EDAC = Path("/sys/devices/system/edac/mc/mc0/ue_count")
THRESHOLDS = md.OUTPUT / "freeze" / "thresholds.json"
MAX_WORKERS = 8
KERNELS = {"measure": None, "rules": None, "features": "Haswell", "check": "Haswell", "score": "Haswell"}


def log(message: str) -> None:
    """
    Print a time-stamped progress line.

    Parameters
    ----------
    message : str
        Text.
    """
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def receipt(role: str, step: str, started: float, extra: dict[str, Any]) -> None:
    """
    Write the receipt of one step under ``OUTPUT/receipts``.

    Parameters
    ----------
    role : str
        Role, or ``freeze`` for a step without a role.
    step : str
        Step name.
    started : float
        Unix start time.
    extra : dict[str, Any]
        Step-specific content.
    """
    content = {
        "experiment": "MLC-04",
        "role": role,
        "step": step,
        "command": [sys.executable, *sys.argv],
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "git": git_state(),
        "code_sha256": code_sha256(CODE),
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
        "packages": {name: version(name) for name in PACKAGES},
        "edac_uncorrected": EDAC.read_text().strip() if EDAC.exists() else None,
        **extra,
    }
    path = md.OUTPUT / "receipts" / f"{role}__{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, json_ready(content))


def hashes(paths: list[Path]) -> dict[str, str]:
    """
    SHA-256 of output files, keyed by repository-relative path.

    Parameters
    ----------
    paths : list[Path]
        Files.

    Returns
    -------
    dict[str, str]
        Digest per path.
    """
    return {to_stored(path): sha256_file(path) for path in paths}


def scan_step(role: str, workers: int) -> None:
    """
    Scan a role and write its v4 manifest rows (step ``scan``).

    Parameters
    ----------
    role : str
        Role.
    workers : int
        Processes.
    """
    started, target = time.time(), md.OUTPUT / role
    rows, _ = md.role_rows(role)
    if (target / "manifest_v4.csv").exists():
        raise FileExistsError(f"{target / 'manifest_v4.csv'} exists")
    target.mkdir(parents=True, exist_ok=True)
    scan = md.run_scan(role, workers)
    scan.to_parquet(target / "scan.parquet", index=False)
    frame = md.withhold_screening_labels(md.assemble(role, rows, scan), role)
    manifest_table(frame).to_csv(target / "manifest_v4.csv")
    overlap = cross_role_hashes(md.read_text(target / "manifest_v4.csv"))
    log(f"scanned {len(scan)} records of {role}; included {int(frame['included'].sum())}")
    receipt(
        role,
        "scan",
        started,
        {
            "records": len(scan),
            "included": int(frame["included"].sum()),
            "first_10_s_hashes_shared_with_v4_train_or_validation": overlap,
            "minimum_positive_table_sha256": sha256_file(md.MINIMUM_TABLE),
            "outputs_sha256": hashes([target / "scan.parquet", target / "manifest_v4.csv"]),
        },
    )


def cross_role_hashes(manifest: pd.DataFrame) -> dict[str, int]:
    """
    Count a role's eligible records whose first-10-s hash equals that of a v4 train or validation record.

    Parameters
    ----------
    manifest : pd.DataFrame
        The role's v4 manifest as text.

    Returns
    -------
    dict[str, int]
        Count per source among the role's eligible records, other than the record itself.
    """
    v4 = pd.read_csv(mv.V4 / "manifest.csv", usecols=["record_key", "role", "sha256_10s"], dtype=str)
    v4 = v4[v4["role"].isin(["train", "validation"]) & ~v4["record_key"].isin(manifest["record_key"])]
    eligible = manifest[manifest["exclusion_reason"] == ""]
    shared = eligible["sha256_10s"].isin(set(v4["sha256_10s"]))
    return {source: int(shared[eligible["source"] == source].sum()) for source in CHALLENGE_SOURCES}


def cache_step(role: str, workers: int) -> None:
    """
    Cache the first 10 s of a role's eligible records (step ``cache``).

    Parameters
    ----------
    role : str
        Role.
    workers : int
        Processes.
    """
    started = time.time()
    md.check_role(role)
    manifest = md.read_text(md.OUTPUT / role / "manifest_v4.csv")
    index = md.build_cache(role, manifest, md.WORK / role / "cache", workers)
    log(f"cached {len(index)} records of {role}")
    receipt(
        role,
        "cache",
        started,
        {
            "records": len(index),
            "missing_samples": int(index["missing_samples_10s"].sum()),
            "records_with_flat_leads": int((index["flat_leads_10s"] > 0).sum()),
            "cache": str(md.WORK / role / "cache"),
            "blocks_sha256": {
                path.name: sha256_file(path) for path in sorted((md.WORK / role / "cache").glob("*.npy"))
            },
            "index_sha256": sha256_file(md.WORK / role / "cache" / "index.csv"),
        },
    )


def measure_step(role: str, workers: int) -> None:
    """
    Part A measures and flags, method B and the whole-record windows of a role (step ``measure``).

    Parameters
    ----------
    role : str
        Role.
    workers : int
        Processes.
    """
    started, target = time.time(), md.OUTPUT / role
    md.check_role(role)
    cache = md.WORK / role / "cache"
    manifest = md.read_text(target / "manifest_v4.csv")
    measures, flags = md.run_part_a(role, cache, manifest, workers)
    measures.to_parquet(target / "part_a_measures.parquet", index=False)
    flags.to_parquet(target / "part_a_flags.parquet", index=False)
    log(f"Part A of {len(measures)} records")
    records = mv.load_records((md.MANIFEST_ROLES[role],), path=target / "manifest_v4.csv")
    method_b = md.run_method_b(role, cache, records["record_key"].tolist(), workers)
    method_b.to_parquet(target / "method_b.parquet", index=False)
    windows = md.run_windows(role, records, workers)
    windows.to_csv(target / "windows.csv", index=False)
    log(f"method B of {len(method_b)} records, windows of {len(windows)}")
    names = ["part_a_measures.parquet", "part_a_flags.parquet", "method_b.parquet", "windows.csv"]
    receipt(
        role,
        "measure",
        started,
        {
            "part_a_records": len(measures),
            "method_b_records": len(method_b),
            "long_records": len(windows),
            "outputs_sha256": hashes([target / name for name in names]),
        },
    )


def features_step(role: str, extractor: str, gpu: int | None, workers: int) -> None:
    """
    Run one extractor on a role's cache (step ``features``).

    Parameters
    ----------
    role : str
        Role.
    extractor : str
        One of ``mm.FEATURE_SETS``.
    gpu : int | None
        Physical GPU of an encoder.
    workers : int
        Processes of the measurements.
    """
    started = time.time()
    record = mm.extract(role, extractor, gpu, workers)
    receipt(role, f"features_{extractor}", started, {"run": record})


def publish_step(role: str) -> None:
    """
    Publish a role's features in cache order (step ``publish``).

    Parameters
    ----------
    role : str
        Role.
    """
    started = time.time()
    target = md.OUTPUT / role / "features"
    mm.publish_features(role, target)
    receipt(role, "publish", started, {"outputs_sha256": hashes(sorted(target.iterdir()))})


def role_tables(role: str) -> dict[str, Any]:
    """
    Load a role's records, method A, method B, analysis table, flags and windows.

    Parameters
    ----------
    role : str
        Role.

    Returns
    -------
    dict[str, Any]
        ``records``, ``a``, ``b``, ``table``, ``flags`` and ``windows``.
    """
    target = md.OUTPUT / md.check_role(role)
    records = mv.load_records((md.MANIFEST_ROLES[role],), path=target / "manifest_v4.csv")
    keys = records["record_key"].to_numpy()
    rows = pd.read_csv(target / "features" / "rows.csv", dtype={"record_key": str})
    values = np.load(target / "features" / "measurements.npy", mmap_mode="r")[
        md.feature_positions(rows, keys)
    ]
    a = md.method_a(keys, values, mm.load_measurement_status(role, keys))
    b = pd.read_parquet(target / "method_b.parquet")
    if b["record_key"].tolist() != records["record_key"].tolist():
        raise ValueError("method B rows differ from the included records")
    return {
        "records": records,
        "a": a,
        "b": b,
        "table": mv.analysis(records, a, b),
        "flags": pd.read_parquet(target / "part_a_flags.parquet"),
        "windows": pd.read_csv(target / "windows.csv", dtype={"record_key": str}),
    }


def rules_step(role: str) -> None:
    """
    Apply rules C1-C12 to a role (step ``rules``).

    Parameters
    ----------
    role : str
        Role.
    """
    started, target = time.time(), md.OUTPUT / role
    parts = role_tables(role)
    v4 = md.read_text(target / "manifest_v4.csv")
    v5, flow = md.apply_v5(v4, parts["table"], parts["flags"], parts["windows"])
    v5.to_csv(target / "manifest_v5.csv", index=False)
    flow.to_csv(target / "rule_flow.csv", index=False)
    parts["table"].drop(columns=["codes"]).to_parquet(target / "analysis.parquet", index=False)
    log(
        f"v5 of {role}: {int((v5['included'] == 'True').sum())} of {int((v4['included'] == 'True').sum())} "
        "records kept"
    )
    names = ["manifest_v5.csv", "rule_flow.csv", "analysis.parquet"]
    receipt(
        role,
        "rules",
        started,
        {
            "v4_included": int((v4["included"] == "True").sum()),
            "v5_included": int((v5["included"] == "True").sum()),
            "outputs_sha256": hashes([target / name for name in names]),
        },
    )


def thresholds_step() -> None:
    """Fit and write the frozen thresholds of P0-v5 once (step ``thresholds``)."""
    started = time.time()
    if THRESHOLDS.exists():
        raise FileExistsError(f"{THRESHOLDS} exists; the thresholds are frozen")
    THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(THRESHOLDS, json_ready(mm.fit_thresholds()))
    receipt("freeze", "thresholds", started, {"outputs_sha256": hashes([THRESHOLDS])})


def check_step() -> None:
    """Run the dry-run reproductions on the validation role (step ``check``)."""
    started = time.time()
    result = run_checks(role_tables("validation"), THRESHOLDS)
    path = md.OUTPUT / "validation" / "check.json"
    write_json_atomic(path, json_ready(result))
    log(f"check: failed {result['failed']}")
    receipt("validation", "check", started, {"failed": result["failed"], "outputs_sha256": hashes([path])})


def role_labels(path: Path) -> dict[str, np.ndarray]:
    """
    Included records of a role manifest with their space-B labels.

    Parameters
    ----------
    path : Path
        ``manifest_v4.csv`` or ``manifest_v5.csv`` of a role.

    Returns
    -------
    dict[str, np.ndarray]
        ``record_key``, ``source`` and ``labels`` (``[records, 14]``, NaN undefined), in manifest order.
    """
    columns = [f"b_{name}" for name in CONDITIONS]
    frame = pd.read_csv(
        path,
        usecols=["record_key", "source", "included", *columns],
        dtype={"record_key": str, "source": str, "included": str},
    )
    frame = frame[frame["included"] == "True"].reset_index(drop=True)
    return {
        "record_key": frame["record_key"].to_numpy(dtype=str),
        "source": frame["source"].to_numpy(dtype=str),
        "labels": frame[columns].to_numpy(dtype=np.float64),
    }


def outputs_for(role: str, v4: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Score every frozen model on a role's v4-included records.

    Parameters
    ----------
    role : str
        Role.
    v4 : dict[str, np.ndarray]
        Output of ``role_labels`` on the role's v4 manifest.

    Returns
    -------
    dict[str, np.ndarray]
        Model id to ``[records, 14]``, aligned with ``v4``.
    """
    features = mm.load_features(role, v4["record_key"])
    return mm.predict_all(mm.load_models(), features, v4["source"])


def score_step(role: str, rerun_after_crash: bool) -> None:
    """
    Score the frozen models on a role and compute every metric (step ``score``).

    Parameters
    ----------
    role : str
        Role.
    rerun_after_crash : bool
        See ``open_marker``.
    """
    started, target = time.time(), md.OUTPUT / md.check_role(role) / "score"
    v4 = role_labels(md.OUTPUT / role / "manifest_v4.csv")
    v5 = role_labels(md.OUTPUT / role / "manifest_v5.csv")
    thresholds = read_json(THRESHOLDS)
    if role == "challenge_test":
        md.open_marker(target, rerun_after_crash)
    target.mkdir(parents=True, exist_ok=True)
    outputs = outputs_for(role, v4)
    np.savez(target / "outputs.npz", record_key=v4["record_key"], **outputs)
    position = pd.Series(np.arange(len(v4["record_key"])), index=v4["record_key"])
    kept = position.loc[v5["record_key"]].to_numpy()
    result, tables = evaluate(
        v5,
        v4,
        {name: values[kept] for name, values in outputs.items()},
        outputs,
        np.array([thresholds["thresholds"][name] for name in CONDITIONS]),
    )
    result["thresholds_sha256"] = sha256_file(THRESHOLDS)
    write_json_atomic(target / "evaluation.json", json_ready(result))
    for name, frame in tables.items():
        frame.to_csv(target / f"{name}.csv", index=False)
    written = sorted(path for path in target.iterdir() if path.name != "marker.json")
    if role == "challenge_test":
        marker = read_json(target / "marker.json") | {"completed_unix": time.time()}
        write_json_atomic(target / "marker.json", marker)
    log(f"{role}: metric {result['primary']['value']:.4f} {result['primary']['ci']}")
    receipt(role, "score", started, {"outputs_sha256": hashes(written)})


def artefact_files() -> list[Path]:
    """
    List every frozen file the test read depends on: models, key, thresholds, features and inputs.

    Returns
    -------
    list[Path]
        Files, models first.
    """
    files = [
        path
        for name in ("P0", "H-All")
        for root in (mm.EFFECT_CANDIDATES, CANDIDATES)
        for path in sorted((root / name).iterdir())
        if not path.name.startswith(("validation__", "lofo")) and path.is_file()
    ]
    files += [mm.KEYS, THRESHOLDS, *mm.STORED_VALIDATION.values(), mm.KEY_SCORES]
    files += [
        PUBLISHED / name
        for name in (
            "xecg.npy",
            "jepa.npy",
            "cpc.npy",
            "measurements.npy",
            "rows.csv",
            "measurements_status.csv",
            "receipt.json",
        )
    ]
    validation = md.OUTPUT / "validation" / "features"
    files += sorted(validation.iterdir())
    files += [md.ROOT / name for names in CHECKPOINTS.values() for name in names]
    files += [SPLIT_TABLE, mv.V4 / "manifest.csv", md.MINIMUM_TABLE, mv.V5 / "manifest.csv"]
    return files


def freeze_draft_step() -> None:
    """Write the freeze record's JSON block from the committed code and the dry run (``freeze-draft``)."""
    started, git = time.time(), git_state()
    if git["modified_tracked"]:
        raise RuntimeError(f"commit the code first: {git['modified_tracked']}")
    check = read_json(md.OUTPUT / "validation" / "check.json")
    validation = read_json(md.OUTPUT / "validation" / "score" / "evaluation.json")
    block = {
        "experiment": "MLC-04",
        "code_commit": git["commit"],
        "code_sha256": code_sha256(CODE),
        "artefacts_sha256": hashes(artefact_files()),
        "dry_run": {
            "failed": check["failed"],
            "metric": check["metric"],
            "validation_primary_rule": validation["primary"],
        },
        "test_groups": {
            "split_records": int(sum(md.split_identity(read_split_table(), "test")[0].shape[:1]))
        },
    }
    path = md.OUTPUT / "freeze" / "freeze.json"
    write_json_atomic(path, json_ready(block))
    receipt("freeze", "freeze_draft", started, {"outputs_sha256": hashes([path])})


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; None reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "step",
        choices=(
            "scan",
            "cache",
            "measure",
            "features",
            "publish",
            "rules",
            "thresholds",
            "check",
            "score",
            "freeze-draft",
        ),
    )
    parser.add_argument("--role", choices=md.ROLES)
    parser.add_argument("--extractor", choices=mm.FEATURE_SETS)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--rerun-after-crash", action="store_true")
    args = parser.parse_args(argv)
    if args.workers > MAX_WORKERS:
        parser.error(f"at most {MAX_WORKERS} workers")
    if args.step in KERNELS and os.environ.get("OPENBLAS_CORETYPE") != KERNELS[args.step]:
        parser.error(
            f"{args.step} runs with OPENBLAS_CORETYPE={KERNELS[args.step]}, as the code it reproduces"
        )
    if args.step not in ("thresholds", "check", "freeze-draft") and args.role is None:
        parser.error(f"{args.step} needs --role")
    steps = {
        "scan": lambda: scan_step(args.role, args.workers),
        "cache": lambda: cache_step(args.role, args.workers),
        "measure": lambda: measure_step(args.role, args.workers),
        "features": lambda: features_step(args.role, args.extractor, args.gpu, args.workers),
        "publish": lambda: publish_step(args.role),
        "rules": lambda: rules_step(args.role),
        "thresholds": thresholds_step,
        "check": check_step,
        "freeze-draft": freeze_draft_step,
        "score": lambda: score_step(args.role, args.rerun_after_crash),
    }
    steps[args.step]()


if __name__ == "__main__":
    main()
