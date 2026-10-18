"""Run the prospective deterministic-interpolation successor's actual CUDA resource profile."""

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
    load_profile_inputs,
)
from ecg_experiment.ssl_synthetic_profile_v2 import (
    FAILED_PROFILE_SHA256,
    cpu_parity,
    forecast,
    load_failed_profile,
    make_model,
    measure_phase,
    replay,
)

OUTPUT = ROOT / "outputs/ssl_synthetic_localization/pilot_v1/profile_v2"
FAILED_PATH = "outputs/ssl_synthetic_localization/pilot_v1/profile/summary.json"
SOURCE_FILES = (
    "ecg_experiment/ssl_synthetic_profile_v2.py",
    "scripts/validation/profile_ssl_synthetic_pilot_v2.py",
    "tests/test_ssl_synthetic_profile_v2.py",
    "docs/ssl-synthetic-localization-profile-v2.md",
    "ecg_experiment/ssl_synthetic_model_v2.py",
    "tests/test_ssl_synthetic_model_v2.py",
    "docs/ssl-synthetic-localization-pilot-deterministic-output.md",
    "ecg_experiment/ssl_synthetic_profile.py",
    "ecg_experiment/ssl_synthetic_model.py",
    "ecg_experiment/ssl_synthetic_data.py",
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
    """Require one numerical CPU thread and the prospectively frozen CUDA settings."""
    required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in required):
        raise ValueError("Set numerical thread limits to 1")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("Enable CUDA and set CUBLAS_WORKSPACE_CONFIG=:4096:8")


def run_cuda(report: dict[str, Any], inputs: dict[str, Any]) -> None:
    """
    Record all disposable parity/replay/phases under the original nonblocking GPU lock.

    Parameters
    ----------
    report : dict
        Mutable failure-retaining receipt.
    inputs : dict
        Verified original actual-data inputs and schedule.
    """
    with gpu_lock("cuda", blocking=False):
        report["device"] = configure_cuda()
        report["active_phase"] = "cpu_counterfactual_parity"
        report["cpu_parity"] = cpu_parity(make_model, inputs["real"], inputs["config"])
        if not report["cpu_parity"]["parity_gate_passed"]:
            raise ValueError("Original/successor CPU forward or gradient parity failed")
        report["active_phase"] = "exact_cuda_replay"
        report["replay"] = replay(make_model, inputs["real"], inputs["config"])
        if not report["replay"]["replay_gate_passed"]:
            raise ValueError("Exact successor two-update CUDA replay failed")
        for phase in PHASES:
            report["active_phase"] = phase
            report["phases"][phase] = measure_phase(make_model, phase, inputs)
        original = report["replay"]["runs"][0]["initialization"]
        report["initialization_gate_passed"] = all(
            value["initialization"]["encoder_sha256"] == original["encoder_sha256"]
            and value["initialization"]["head_sha256"] == original["head_sha256"]
            for value in report["phases"].values()
        )
        report["frozen_encoder_gate_passed"] = report["phases"]["frozen_localization"][
            "frozen_encoder_unchanged"
        ]
        if not report["initialization_gate_passed"] or not report["frozen_encoder_gate_passed"]:
            raise ValueError("Matched initialization or frozen encoder guard failed")
        report["active_phase"] = "complete"


def main() -> int:
    """
    Create one immutable successor profile receipt, charging the retained failed attempt.

    Returns
    -------
    int
        Zero for complete parity/replay/resource gates, two for a recorded known failure.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    validate_environment()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("GPU profile-v2 output directory must be new")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": "ssl_synthetic_pilot_profile_v2",
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
        failed, inputs = load_failed_profile(), load_profile_inputs()
        for key in (
            "input_sha256",
            "prerequisite_source_sha256",
            "prior_cpu_charges",
            "positive_weight",
            "localization_schedule_sha256",
        ):
            report[key] = inputs[key]
        report["input_sha256"][FAILED_PATH] = FAILED_PROFILE_SHA256
        report["prior_cpu_charges"][FAILED_PATH] = {"seconds": failed["cpu_seconds"], "basis": "cpu_seconds"}
        report["batch_size"] = inputs["config"]["batch_size"]
        report["optimizer"] = {
            "name": "AdamW",
            "learning_rate": inputs["config"]["learning_rate"],
            "weight_decay": inputs["config"]["weight_decay"],
        }
        run_cuda(report, inputs)
        report["replay_gate_passed"] = report["replay"]["replay_gate_passed"]
        prior = sum(
            value["seconds"] for key, value in inputs["prior_cpu_charges"].items() if key != FAILED_PATH
        )
        report["resource_forecast"] = forecast(
            report["phases"],
            time.perf_counter() - wall,
            time.process_time() - cpu,
            prior,
            failed,
            stream_bytes(),
            inputs["config"],
        )
        report["profile_gate_passed"] = report["resource_forecast"]["resource_gate_passed"]
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        report.update(failure_type=type(error).__name__, failure_detail=str(error), profile_gate_passed=False)
    peaks = list(report["phases"].values()) + ([report["replay"]] if "replay" in report else [])
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        report[name] = max((phase[name] for phase in peaks), default=0)
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
