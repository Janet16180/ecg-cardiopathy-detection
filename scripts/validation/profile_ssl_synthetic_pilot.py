"""Measure the actual fixed CUDA pilot path without retaining any trained weights or scores."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

import torch

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_preparation import stream_bytes, write_new_json
from ecg_experiment.ssl_synthetic_profile import (
    MASK_SEED,
    PHASES,
    SAMPLING_SEED,
    SCHEDULE_SEED,
    configure_cuda,
    exact_cuda_replay,
    load_profile_inputs,
    profile_phase,
    resource_forecast,
)

OUTPUT = ROOT / "outputs/ssl_synthetic_localization/pilot_v1/profile"
SOURCE_FILES = (
    "ecg_experiment/ssl_synthetic_profile.py",
    "scripts/validation/profile_ssl_synthetic_pilot.py",
    "docs/ssl-synthetic-localization-profile.md",
    "ecg_experiment/ssl_synthetic_model.py",
    "ecg_experiment/ssl_synthetic_data.py",
    "ecg_experiment/ssl_synthetic_metrics.py",
    "ecg_experiment/ssl_synthetic_preparation.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/ssl_synthetic_readiness.py",
    "docs/ssl-synthetic-localization-pilot.md",
    "docs/ssl-synthetic-localization-pilot-generator-v2.md",
    "configs/ssl_synthetic_pilot_v1.json",
    "pyproject.toml",
    "uv.lock",
)


def validate_environment() -> None:
    """Require prospective single-thread and deterministic CUDA environment settings."""
    required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in required):
        raise ValueError("Set numerical thread limits to 1")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("Enable CUDA and set CUBLAS_WORKSPACE_CONFIG=:4096:8")


def run_profile(report: dict[str, Any], inputs: dict[str, Any]) -> None:
    """
    Fill a mutable receipt while holding the nonblocking shared GPU lock.

    Parameters
    ----------
    report : dict
        Mutable receipt preserving every completed phase if a later phase fails.
    inputs : dict
        Verified actual cached inputs and configuration.
    """
    with gpu_lock("cuda", blocking=False):
        report["device"] = configure_cuda()
        report["active_phase"] = "exact_cuda_replay"
        report["replay"] = exact_cuda_replay(inputs["real"], inputs["config"])
        if not report["replay"]["replay_gate_passed"]:
            raise ValueError("Exact two-update CUDA replay failed")
        for phase in PHASES:
            report["active_phase"] = phase
            report["phases"][phase] = profile_phase(phase, inputs)
        initializations = [phase["initialization"] for phase in report["phases"].values()]
        original = report["replay"]["runs"][0]["initialization"]
        report["initialization_gate_passed"] = all(
            value["encoder_sha256"] == original["encoder_sha256"]
            and value["head_sha256"] == original["head_sha256"]
            for value in initializations
        )
        report["frozen_encoder_gate_passed"] = report["phases"]["frozen_localization"][
            "frozen_encoder_unchanged"
        ]
        if not report["initialization_gate_passed"] or not report["frozen_encoder_gate_passed"]:
            raise ValueError("Matched initialization or frozen encoder guard failed")
        report["active_phase"] = "complete"


def main() -> int:
    """
    Emit one new passing or failed actual resource-profile receipt.

    Returns
    -------
    int
        Zero for all profile gates; two for a recorded known prerequisite/device/gate failure.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    validate_environment()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("GPU profile output directory must be new")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": "ssl_synthetic_pilot_profile_v1",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "profile_gate_passed": False,
        "phases": {},
        "replay_gate_passed": False,
        "initialization_gate_passed": False,
        "frozen_encoder_gate_passed": False,
        "training_started": False,
        "performance_scored": False,
        "training_admitted": False,
        "profile_weights_discarded": True,
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
        "seeds": {
            "model": 20261018,
            "ssl_sampling": SAMPLING_SEED,
            "ssl_masking": MASK_SEED,
            "localization_schedule": SCHEDULE_SEED,
        },
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "summary": to_stored(OUTPUT / "summary.json"),
    }
    try:
        inputs = load_profile_inputs()
        for name in (
            "input_sha256",
            "prerequisite_source_sha256",
            "prior_cpu_charges",
            "positive_weight",
            "localization_schedule_sha256",
        ):
            report[name] = inputs[name]
        report["batch_size"] = inputs["config"]["batch_size"]
        report["optimizer"] = {
            "name": "AdamW",
            "learning_rate": inputs["config"]["learning_rate"],
            "weight_decay": inputs["config"]["weight_decay"],
        }
        run_profile(report, inputs)
        report["replay_gate_passed"] = report["replay"]["replay_gate_passed"]
        prior_cpu = sum(charge["seconds"] for charge in inputs["prior_cpu_charges"].values())
        report["resource_forecast"] = resource_forecast(
            report["phases"],
            time.perf_counter() - wall,
            time.process_time() - cpu,
            prior_cpu,
            stream_bytes(),
            inputs["config"],
        )
        report["profile_gate_passed"] = report["resource_forecast"]["resource_gate_passed"]
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        report.update(failure_type=type(error).__name__, failure_detail=str(error), profile_gate_passed=False)
    peaks = list(report["phases"].values())
    if "replay" in report:
        peaks.append(report["replay"])
    report["peak_allocated_bytes"] = max((phase["peak_allocated_bytes"] for phase in peaks), default=0)
    report["peak_reserved_bytes"] = max((phase["peak_reserved_bytes"] for phase in peaks), default=0)
    if torch.cuda.is_initialized():
        report["peak_allocated_bytes"] = max(
            report["peak_allocated_bytes"], torch.cuda.max_memory_allocated()
        )
        report["peak_reserved_bytes"] = max(report["peak_reserved_bytes"], torch.cuda.max_memory_reserved())
    report.update(
        finished_at_utc=datetime.now(UTC).isoformat(),
        wall_seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
    )
    write_new_json(OUTPUT / "summary.json", report)
    print(
        {
            "summary": report["summary"],
            "profile_gate_passed": report["profile_gate_passed"],
            "failure_type": report.get("failure_type"),
        }
    )
    return 0 if report["profile_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
