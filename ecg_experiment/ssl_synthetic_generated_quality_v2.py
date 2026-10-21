"""Prospective budget successor reusing frozen generated-quality operations unchanged."""

from __future__ import annotations

import json
import resource
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as original
from ecg_experiment import ssl_synthetic_pretrained_generator as shared

CONFIG = "configs/ssl_synthetic_generated_quality_v2.json"
SOURCES = (
    CONFIG,
    "docs/ssl-synthetic-localization-generated-quality-v2.md",
    "docs/ssl-synthetic-localization-generated-quality-v1-results.md",
    "ecg_experiment/ssl_synthetic_generated_quality_v2.py",
    "scripts/experiments/run_ssl_synthetic_generated_quality_v2.py",
    "tests/test_ssl_synthetic_generated_quality_v2.py",
)


def verify_config(config: dict[str, Any]) -> None:
    """Allow only the prospectively amended path, budget and failure-accounting declarations."""
    expected = json.loads((ROOT / original.CONFIG).read_text())
    expected.update(
        study_id="pulse2pulse_generated_quality_v2",
        output_root="outputs/ssl_synthetic_localization/generated_quality_v2",
    )
    expected["resources"].update(wall_seconds=1800, gpu_lock_wall_seconds=1800)
    numerical = {
        key: value
        for key, value in config.items()
        if key not in {"failed_attempt", "accounting_prior_ledgers"}
    }
    if numerical != expected:
        raise ValueError("Only the prospective v2 resource/path amendment is allowed")


def snapshot(destination: Path) -> dict[str, str]:
    """Reuse frozen source snapshots and add all authored v2 declarations without global changes."""
    sources = original.snapshot(destination)
    for name in SOURCES:
        path = destination / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        sources[name] = shared.file_sha256(path)
    return sources


def admit_amendment(
    output_root: Path, config: dict[str, Any], bindings: dict[str, str]
) -> list[dict[str, Any]]:
    """Bind the retained failed attempt and charge each prior stage's own terminal cost once."""
    failed = config["failed_attempt"]
    for filename, digest in failed["sha256"].items():
        name = f"{failed['root']}/{filename}"
        if shared.file_sha256(output_root / name) != digest:
            raise ValueError("Frozen failed-v1 receipt identity mismatch")
        bindings[name] = digest
    failure = json.loads((output_root / failed["root"] / "failure.json").read_text())
    if failure["forward_counts"] != {
        "cpu_predecessor_replay": 4,
        "gpu_profile": 1,
        "gpu_generation": 0,
        "gpu_replay": 0,
    }:
        raise ValueError("Failed-v1 forward closure differs from the prospective amendment")
    if failure["completed_seeds"]:
        raise ValueError("Failed v1 unexpectedly contains completed bank seeds")
    entries = []
    if len({row["path"] for row in config["accounting_prior_ledgers"]}) != 4:
        raise ValueError("Four distinct prior-stage ledgers are required")
    for row in config["accounting_prior_ledgers"]:
        path = output_root / row["path"]
        if shared.file_sha256(path) != row["sha256"]:
            raise ValueError("Frozen prior terminal ledger identity mismatch")
        bindings[row["path"]] = row["sha256"]
        ledger = json.loads(path.read_text())
        entries.append(
            {**row, "cpu_seconds": ledger["cpu_seconds"], "gpu_wall_seconds": ledger["gpu_wall_seconds"]}
        )
    return entries


def verify_amendment_inputs(output_root: Path, bindings: dict[str, str]) -> None:
    """Reject any change in failed-attempt or terminal-accounting receipts during execution."""
    for name, digest in bindings.items():
        if shared.file_sha256(output_root / name) != digest:
            raise ValueError("Frozen accounting or failed-v1 receipt changed during execution")


def finish_ledger(
    destination: Path,
    result: dict[str, Any],
    receipt_name: str,
    clock: dict[str, float],
    prior_entries: list[dict[str, Any]],
) -> None:
    """Add v2 terminal costs once to distinct main, audit and retained-v1-failure costs."""
    digest = shared.file_sha256(destination / receipt_name)
    terminal = quality.elapsed(clock)
    current_gpu = result.get("gpu_lock_wall_seconds", 0.0)
    shared.write_new_json(
        destination / "prospective_ledger.json",
        {
            "scope": "new_current_machine_studies_only; old factorial unknown",
            "stage": "generated_quality_v2",
            "receipt": receipt_name,
            "receipt_sha256": digest,
            **terminal,
            "gpu_wall_seconds": current_gpu,
            "prior_stage_entries": prior_entries,
            "cumulative_cpu_seconds": sum(row["cpu_seconds"] for row in prior_entries)
            + terminal["cpu_seconds"],
            "cumulative_gpu_wall_seconds": sum(row["gpu_wall_seconds"] for row in prior_entries)
            + current_gpu,
            "prior_accounting_complete": len(prior_entries) == 4,
            "historical_costs_verified": False,
            "measurement_endpoint": "after main receipt write/hash, before ledger serialization",
            "independent_audit_charged_separately": True,
            "reserved_independent_audit": {"cpu_seconds": 180, "wall_seconds": 180, "bytes": 16 * 1024**2},
        },
    )


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute one committed and reviewed study, retaining any failed attempt.

    Returns
    -------
    dict[str, Any]
        Output-bound engineering results without clinical claims.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = json.loads((ROOT / CONFIG).read_text())
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    parent = output_root / config["predecessor_root"]
    resources = config["resources"]
    counts = {"cpu_predecessor_replay": 0, "gpu_profile": 0, "gpu_generation": 0, "gpu_replay": 0}
    attempts = dict.fromkeys(counts, 0)
    gpu_cost = 0.0
    sources, inputs = {}, {}
    rows = []
    amendment_inputs, prior_entries = {}, []
    old_handlers = {}

    def guard() -> None:
        spent = quality.elapsed(clock)
        if spent["cpu_seconds"] + 2 >= resources["cpu_seconds"]:
            raise TimeoutError("Scientific CPU limit reached")
        if spent["wall_seconds"] + 2 >= resources["wall_seconds"]:
            raise TimeoutError("Scientific wall limit reached")
        if (
            shared.directory_bytes(destination) + resources["final_receipt_reserve_bytes"]
            >= resources["output_bytes"]
        ):
            raise ValueError("Saved-output storage reserve exhausted")

    try:
        verify_config(config)
        original.set_runtime()
        original.arm_timers(resources, old_handlers)
        sources = snapshot(destination)
        prior_entries = admit_amendment(output_root, config, amendment_inputs)
        shared.write_new_json(
            destination / "amendment_admission.json",
            {"amendment_input_sha256": amendment_inputs, "prior_stage_entries": prior_entries},
        )
        model, main, inputs, predecessor = original.admit_predecessor(parent, destination, counts, attempts)
        initial_state = shared.tensor_state_sha256(model)
        noises = np.stack(
            [
                torch.empty((1, 8, 5000), dtype=torch.float32)
                .uniform_(
                    -1, 1, generator=torch.Generator(device="cpu").manual_seed(config["first_seed"] + i)
                )
                .numpy()
                for i in range(config["n_samples"])
            ]
        )
        guard()
        banks, profile, gpu_metadata, gpu_cost = original.generate_bank(
            model, noises, config, counts, attempts, clock, destination, rows, guard
        )
        gates = original.bank_gates(
            banks["raw"],
            banks["variants"],
            banks["support"],
            banks["strict_support"],
            rows,
            banks["replay_raw"],
        )
        gates["predecessor_exact"] = predecessor["all_exact"]
        final_state = shared.tensor_state_sha256(model)
        gates["model_state_unchanged"] = final_state == initial_state
        gates["model_source_identity"] = initial_state == main["model_state_sha256"]
        original.save_outputs(destination, banks, rows)
        guard()
        original.verify_bindings(destination, parent, sources, inputs)
        verify_amendment_inputs(output_root, amendment_inputs)
        outputs = {
            p.relative_to(destination).as_posix(): shared.file_sha256(p)
            for p in destination.rglob("*")
            if p.is_file()
        }
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "operational_support_qualified": all(gates.values()),
            "gates": gates,
            "primary": quality.primary_interval(rows),
            "diversity": original.diversity_summary(banks["variants"]),
            "source_sha256": sources,
            "input_sha256": inputs,
            "amendment_input_sha256": amendment_inputs,
            "output_sha256": outputs,
            "model_state_sha256": initial_state,
            "final_state_sha256": final_state,
            "completed_seeds": [r["seed"] for r in rows],
            "forward_counts": counts,
            "forward_attempts": attempts,
            "profile": profile,
            "gpu_metadata": gpu_metadata,
            "gpu_lock_wall_seconds": gpu_cost,
            **quality.elapsed(clock),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": shared.directory_bytes(destination),
            "new_training_updates": 0,
            "real_waveform_reads": 0,
            "external_api_calls": 0,
            "physical_units_verified": False,
            "clinical_fidelity_validated": False,
            "disease_control_validated": False,
            "real_disease_localization_validated": False,
            "support_kind": "introduced numerical change; fixed-position engineering fixtures",
            "accounting_scope": "current_machine_additive_with_failed_v1; historical costs unverified",
        }
        gates["resources"] = (
            result["cpu_seconds"] < resources["cpu_seconds"]
            and result["wall_seconds"] < resources["wall_seconds"]
            and gpu_cost < resources["gpu_lock_wall_seconds"]
            and gpu_metadata["peak_allocated_bytes"] < resources["maximum_gpu_allocated_bytes"]
        )
        result["operational_support_qualified"] = all(gates.values())
        guard()
        shared.write_new_json(destination / "summary.json", result)
        finish_ledger(destination, result, "summary.json", clock, prior_entries)
        return result
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "forward_counts": counts,
            "forward_attempts": attempts,
            "completed_seeds": [r["seed"] for r in rows],
            "source_sha256": sources,
            "input_sha256": inputs,
            "amendment_input_sha256": amendment_inputs,
            "gpu_lock_wall_seconds": clock.get("gpu_cost", gpu_cost),
            **quality.elapsed(clock),
            "historical_costs_verified": False,
        }
        if rows and not (destination / "partial_cases.jsonl").exists():
            quality.write_rows(destination / "partial_cases.jsonl", rows)
        shared.write_new_json(destination / "failure.json", failure)
        finish_ledger(destination, failure, "failure.json", clock, prior_entries)
        raise
    finally:
        for timer, sig in [(signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)]:
            if sig in old_handlers:
                signal.setitimer(timer, 0)
                signal.signal(sig, old_handlers[sig])
