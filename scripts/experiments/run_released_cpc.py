#!/usr/bin/env python3
"""Sequential frozen released ECG-CPC extraction and the existing label probes."""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from ecg_experiment.files import write_json_atomic
from ecg_experiment.receipts import artifacts_exist
from scripts.experiments.probe_pretrained import ARTIFACTS as PROBE_ARTIFACTS

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment004_cpc_40k"
PROBES = (("1", "released_full"), ("0.1", "released_10pct"))
RELEASED_FIELDS = ("checkpoint_source", "license", "reported_pretraining", "comparison_limit",
                   "cauchy_backend")


def subprocess_environment() -> dict[str, str]:
    """
    Single-threaded environment with private caches for the extraction subprocesses.

    Returns
    -------
    dict[str, str]
        Environment variables.
    """
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MPLCONFIGDIR="/tmp/ecg-cpc-mpl", KEOPS_CACHE_FOLDER="/tmp/ecg-cpc-keops",
                CUDA_PATH="/usr/local/cuda")


def run_probe(fraction: str, name: str, features: Path, state_path: Path, env: dict[str, str]) -> Path:
    """
    Train one label-budget probe unless it already completed.

    An incomplete earlier probe directory is renamed aside, not overwritten.

    Parameters
    ----------
    fraction : str
        Label fraction, ``"1"`` or ``"0.1"``.
    name : str
        Output subdirectory.
    features : Path
        Released-model feature directory.
    state_path : Path
        Status file updated before the probe starts.
    env : dict[str, str]
        Subprocess environment.

    Returns
    -------
    Path
        The probe directory.
    """
    directory = OUTPUT / name / "ecg-cpc_released_linear_seed42"
    if directory.exists() and not artifacts_exist(directory, PROBE_ARTIFACTS):
        suffix = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        directory.rename(directory.with_name(directory.name + "_interrupted_" + suffix))
    if not directory.exists():
        write_json_atomic(state_path, {"stage": "linear_probe", "label_fraction": fraction,
                                       "pid": os.getpid()})
        subprocess.run([sys.executable, "-m", "scripts.experiments.probe_pretrained", "--embeddings-dir",
                        str(features), "--name", "ecg-cpc_released", "--manifest-dir",
                        str(ROOT / "data/processed/ptbxl" / f"seed42_fraction{fraction}"),
                        "--output-dir", str(OUTPUT / name)], cwd=ROOT, env=env, check=True)
    return directory


def annotate_probe(directory: Path, provenance: dict[str, object]) -> None:
    """
    Record the released model's provenance and pretraining exposure in a probe's config.

    Parameters
    ----------
    directory : Path
        Probe directory.
    provenance : dict[str, object]
        Released feature metadata.
    """
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["pretraining_exposure"] = provenance["pretraining_overlap"]
    config["released_model"] = {key: provenance[key] for key in RELEASED_FIELDS}
    config["released_model"]["checkpoint_sha256"] = provenance["identity"]["checkpoint_sha256"]
    write_json_atomic(config_path, config)


def run_all(state_path: Path) -> None:
    """
    Extract released features, then fit and annotate both label probes.

    Parameters
    ----------
    state_path : Path
        Status file updated at each stage.
    """
    env = subprocess_environment()
    write_json_atomic(state_path, {"stage": "extraction", "pid": os.getpid()})
    subprocess.run([sys.executable, "-m", "scripts.features.extract_ecg_cpc"], cwd=ROOT, env=env, check=True)
    features = OUTPUT / "released_features"
    provenance = json.loads((features / "metadata.json").read_text())
    for fraction, name in PROBES:
        annotate_probe(run_probe(fraction, name, features, state_path, env), provenance)
    write_json_atomic(state_path, {"stage": "complete", "pid": os.getpid(), "label_fractions": ["1", "0.1"]})


def main() -> None:
    """Run the released-model pipeline under its runner lock, recording failures before re-raising."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / "released_status.json"
    with (OUTPUT / "released_runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run_all(state_path)
        except Exception as exc:
            # Top-level status record for the coordinator; the error still propagates.
            write_json_atomic(state_path, {"stage": "failed", "pid": os.getpid(), "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
