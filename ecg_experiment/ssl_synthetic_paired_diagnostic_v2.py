"""Forward saved-map orchestration with exact receipt accounting and immutable v1 arithmetic."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_paired_diagnostic as frozen
from ecg_experiment.files import sha256_file

CONFIG_PATH = ROOT / "configs/ssl_synthetic_paired_diagnostic_v2.json"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_paired_diagnostic_v2.py",
    "scripts/reports/diagnose_ssl_synthetic_paired_background_v2.py",
    "tests/test_ssl_synthetic_paired_diagnostic_v2.py",
    "configs/ssl_synthetic_paired_diagnostic_v2.json",
    "docs/ssl-synthetic-localization-paired-diagnostic-v2.md",
)


def retained_attempt(config: dict) -> tuple[dict, dict]:
    """Read only SHA-anchored stopped/parent metadata; exact costs are the numerical authority."""
    failure_path = ROOT / config["retained_failure_receipt"]
    if sha256_file(failure_path) != config["retained_failure_sha256"]:
        raise ValueError("Retained stopped attempt changed")
    failure = frozen._json(failure_path)
    if failure["progress"] != {"completed": False, "control_rows": 0, "positive_rows": 0}:
        raise ValueError("Retained attempt no longer precedes diagnostic statistics")
    if any(
        failure[name]
        for name in (
            "waveform_payload_reads",
            "checkpoint_payload_reads",
            "model_inferences",
            "optimizer_updates",
            "gpu_used",
        )
    ):
        raise ValueError("Retained attempt scope changed")
    audit_path = frozen.STUDY / "metrics_audit/summary.json"
    if sha256_file(audit_path) != config["parent_metrics_audit_sha256"]:
        raise ValueError("Exact parent accounting authority changed")
    parent = frozen._json(audit_path)
    return failure, parent


def guard_forward(context: dict, wall: float, cpu: float) -> None:
    """Guard shifted total-attempt clocks and the combined diagnostic family output budget."""
    config = context["config"]
    elapsed_wall, elapsed_cpu = time.perf_counter() - wall, time.process_time() - cpu
    reserve = config["failure_reporting_reserve_seconds"]
    if max(elapsed_wall, elapsed_cpu) + reserve >= config["cpu_seconds_run_ceiling"]:
        raise TimeoutError("Combined main-attempt allowance exhausted")
    if context["prior_cpu"] + elapsed_cpu + reserve >= config["global_cpu_ceiling"]:
        raise TimeoutError("Cumulative CPU allowance exhausted")
    if context["prior_stage_cpu"] + elapsed_cpu + reserve >= config["parent_stage_cpu_ceiling"]:
        raise TimeoutError("Paired-study CPU allowance exhausted")
    family_bytes = sum(
        frozen._stream_bytes(path) for path in frozen.STUDY.glob("diagnostic*") if path.is_dir()
    )
    if (
        family_bytes + 1024**2 >= config["new_output_bytes_ceiling"]
        or frozen._stream_bytes() + 1024**2 >= config["cache_bytes_ceiling"]
    ):
        raise ValueError("Combined diagnostic or whole-stream storage allowance exhausted")


def admit_forward(config: dict, failure: dict, guard: Callable[[], None]) -> dict:
    """Reuse complete frozen admission and add exact failed-source/new-adapter bindings."""
    context = frozen.admit_diagnostic(guard)
    for key, value in context["config"].items():
        if key != "study_id" and value != config[key]:
            raise ValueError("A frozen scientific or resource setting changed")
    context["config"] = config
    frozen._merge(context["sources"], failure["source_sha256"])
    frozen._merge(context["sources"], {name: sha256_file(ROOT / name) for name in SOURCE_NAMES})
    frozen._merge(context["inputs"], failure["input_sha256"])
    frozen._merge(context["inputs"], {config["retained_failure_receipt"]: config["retained_failure_sha256"]})
    bindings = dict(context["sources"])
    frozen._merge(bindings, context["inputs"])
    context["bindings"] = bindings
    context["scope_exceptions"] = frozen.verify_bindings(
        bindings, context["authorities"], config["allowed_array_payloads"], guard
    )
    context["reproduction"]["evaluation_case_flags_checked"] = frozen._reproduce_parent_flags(context)
    return context


def diagnose_paired_background_v2() -> dict:
    """Run the unchanged fixed diagnostic once with exact costs and a new guarded destination."""
    actual_wall, actual_cpu = time.perf_counter(), time.process_time()
    config = frozen._json(CONFIG_PATH)
    destination = ROOT / config["output_dir"]
    if destination.exists():
        raise FileExistsError("Forward diagnostic destination must be new")
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = {}
    progress = {"completed": False, "control_rows": 0, "positive_rows": 0}
    destination.mkdir(parents=True, exist_ok=False)
    failure = {"cpu_seconds": 0.0, "wall_seconds": 0.0}
    try:
        names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        if any(os.environ.get(name) != "1" for name in names) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Diagnostic requires one numerical thread and hidden CUDA")
        failure, parent = retained_attempt(config)
        wall, cpu = actual_wall - failure["wall_seconds"], actual_cpu - failure["cpu_seconds"]
        context = {
            "config": config,
            "prior_cpu": parent["cumulative_cpu_seconds"],
            "prior_stage_cpu": parent["stage_cpu_seconds_including_both_audits"],
        }
        guard_forward(context, wall, cpu)
        context = admit_forward(config, failure, lambda: guard_forward(context, wall, cpu))
        sources, inputs = context["sources"], context["inputs"]
        frozen._write(
            destination / "admission.json",
            {
                "source_sha256": sources,
                "input_sha256": inputs,
                "scope_exceptions": context["scope_exceptions"],
                "reproduction": context["reproduction"],
            },
        )
        controls = frozen.control_rows(context)
        progress["control_rows"] = len(controls)
        guard_forward(context, wall, cpu)
        control_summary = frozen.summarize_controls(controls, config)
        guard_forward(context, wall, cpu)
        positives = frozen.positive_response_rows(context, wall, cpu)
        progress["positive_rows"] = len(positives)
        positive_summary = frozen.summarize_positive(positives, context, wall, cpu)
        losses = frozen.loss_blocks(context["training"], context["logs"])
        if len(controls) != 384 or len(positives) != 512:
            raise ValueError("Fixed diagnostic denominators changed")
        frozen._write(
            destination / "local_case_statistics.json", {"control_rows": controls, "positive_rows": positives}
        )
        guard_forward(context, wall, cpu)
        frozen.verify_bindings(
            context["bindings"],
            context["authorities"],
            config["allowed_array_payloads"],
            lambda: guard_forward(context, wall, cpu),
        )
        outputs = {path.name: sha256_file(path) for path in destination.iterdir() if path.is_file()}
        guard_forward(context, wall, cpu)
        progress["completed"] = True
        run_cpu, run_wall = time.process_time() - actual_cpu, time.perf_counter() - actual_wall
        result = {
            "diagnostic_completed": True,
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": outputs,
            "scope_exceptions": context["scope_exceptions"],
            "reproduction": context["reproduction"],
            "progress": progress,
            **control_summary,
            **positive_summary,
            "training_loss": losses,
            "transition_coordinate": {
                "within_arm": "first=sham, second=nuisance",
                "between_arm": "first=R, second=S",
            },
            "wall_seconds": run_wall,
            "cpu_seconds": run_cpu,
            "retained_failure_wall_seconds": failure["wall_seconds"],
            "retained_failure_cpu_seconds": failure["cpu_seconds"],
            "charged_main_wall_seconds": failure["wall_seconds"] + run_wall,
            "charged_main_cpu_seconds": failure["cpu_seconds"] + run_cpu,
            "prior_cumulative_cpu_seconds": context["prior_cpu"] + failure["cpu_seconds"],
            "cumulative_cpu_seconds": context["prior_cpu"] + failure["cpu_seconds"] + run_cpu,
            "paired_stage_cpu_seconds": context["prior_stage_cpu"] + failure["cpu_seconds"] + run_cpu,
            "actual_stream_bytes_before_summary": frozen._stream_bytes(),
            "new_output_bytes_before_summary": sum(
                frozen._stream_bytes(p) for p in frozen.STUDY.glob("diagnostic*") if p.is_dir()
            ),
            "final_receipt_storage_reserve_bytes": 1024**2,
            "waveform_payload_reads": 0,
            "checkpoint_payload_reads": 0,
            "model_inferences": 0,
            "optimizer_updates": 0,
            "gpu_used": False,
            "new_advancement_rule": None,
            "parent_full_advancement_passed": False,
            "clinical_claim": False,
            "numerical_threads": {name: os.environ[name] for name in names},
        }
        guard_forward(context, wall, cpu)
        frozen._write(destination / "summary.json", result)
        return result
    except Exception as error:
        run_cpu, run_wall = time.process_time() - actual_cpu, time.perf_counter() - actual_wall
        frozen._write(
            destination / "failure.json",
            {
                "source_sha256": sources,
                "input_sha256": inputs,
                "progress": progress,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "wall_seconds": run_wall,
                "cpu_seconds": run_cpu,
                "charged_main_wall_seconds": failure["wall_seconds"] + run_wall,
                "charged_main_cpu_seconds": failure["cpu_seconds"] + run_cpu,
                "waveform_payload_reads": 0,
                "checkpoint_payload_reads": 0,
                "model_inferences": 0,
                "optimizer_updates": 0,
                "gpu_used": False,
            },
        )
        raise RuntimeError("Forward diagnostic stopped; inspect its local failure receipt") from None
