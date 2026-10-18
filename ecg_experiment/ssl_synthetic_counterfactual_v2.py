"""Forward provenance adapter for unchanged saved-map counterfactual diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_counterfactual as original_helpers
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_evaluation import _merge_hashes, verify_inference_hashes
from ecg_experiment.ssl_synthetic_preparation import BASE, write_new_json

DESTINATION = BASE / "pilot_v1/counterfactual_audit_v2"
FAILED = BASE / "pilot_v1/counterfactual_audit_v1/failure.json"
ORIGINAL_AUDIT = BASE / "pilot_v1/final_audit_v1/summary.json"
ORIGINAL_AUDIT_SHA256 = "45f54b7e773653c0f76f48898a6bf3fe38f4cb42df280bf8605aaad0b24d40ad"
FAILED_SHA256 = "8a842ebb36acee66fa22c82026444f1d61afe702301d328c8c1f55068cea06f4"
FORWARD_SOURCES = (
    "ecg_experiment/ssl_synthetic_counterfactual_v2.py",
    "scripts/validation/audit_ssl_synthetic_counterfactual_v2.py",
    "tests/test_ssl_synthetic_counterfactual_v2.py",
    "docs/ssl-synthetic-localization-counterfactual-v2.md",
)


def require_forward_binding(closure: dict[str, Any], failed: dict[str, Any]) -> None:
    """Require the explicit passed-original-audit link and the retained zero-diagnostic failure.

    Parameters
    ----------
    closure : dict
        Independently executed additive reproduction binding, already source/hash verified.
    failed : dict
        Retained original counterfactual admission failure with zero completed diagnostics.
    """
    if closure["input_sha256"].get(to_stored(ORIGINAL_AUDIT)) != ORIGINAL_AUDIT_SHA256:
        raise ValueError("Forward closure does not bind the exact passed original independent audit")
    if failed["audit_completed"] or failed["progress"]["verified_positive_pairs"] != 0:
        raise ValueError("Forward schema adapter cannot replace an already derived diagnostic")
    if failed["progress"]["completed_profiles"]:
        raise ValueError("Original schema-failure history changed")


def _admission(closure_path: Path) -> tuple[dict, dict, dict, dict]:
    original, sources, inputs = original_helpers._admission(closure_path)
    closure = json.loads(closure_path.read_text())
    failed = json.loads(FAILED.read_text())
    require_forward_binding(closure, failed)
    verify_inference_hashes(failed["source_sha256"])
    _merge_hashes(sources, failed["source_sha256"])
    if sha256_file(FAILED) != FAILED_SHA256:
        raise ValueError("Pinned failed counterfactual receipt changed")
    _merge_hashes(sources, {name: sha256_file(ROOT / name) for name in FORWARD_SOURCES})
    _merge_hashes(inputs, {to_stored(FAILED): FAILED_SHA256})
    costs = {
        "original_pilot": original["total_preparation_training_evaluation_cpu_seconds"],
        to_stored(FAILED): failed["cpu_seconds"],
        to_stored(closure_path): closure["cpu_seconds"],
    }
    audit_prefix = to_stored(ORIGINAL_AUDIT.parent) + "/"
    for name in closure["input_sha256"]:
        if name.startswith(audit_prefix) and name.endswith(".json"):
            receipt = json.loads((ROOT / name).read_text())
            if "cpu_seconds" in receipt:
                costs[name] = receipt["cpu_seconds"]
    verify_inference_hashes(sources | inputs)
    return original, sources, inputs, costs


def _runtime_guard() -> dict[str, str | None]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Saved-map diagnostic requires CUDA hidden")
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    threads = {name: os.environ.get(name) for name in names}
    if any(value != "1" for value in threads.values()):
        raise ValueError("Saved-map diagnostic requires one numerical thread")
    return threads


def _derive(
    original: dict[str, Any], config: dict[str, Any], cpu_deadline: float, progress: dict[str, Any]
) -> dict[str, Any]:
    banks, maps, pairs = original_helpers._saved_inputs()
    progress["verified_positive_pairs"] = sum(map(len, pairs.values()))
    summaries = {}
    for profile in ("B", "A"):
        if time.process_time() >= cpu_deadline:
            raise TimeoutError("Cumulative protocol CPU ceiling exhausted")
        prior_rows = json.loads(
            (original_helpers.ORIGINAL / f"{profile}_local_case_metrics.json").read_text()
        )["rows"]
        rows, controls = original_helpers._profile_rows(
            banks[profile], maps[profile], pairs[profile], original["calibration"]["thresholds"], prior_rows
        )
        write_new_json(
            DESTINATION / f"{profile}_local_case_responses.json", {"rows": rows, "controls": controls}
        )
        summaries[profile] = {
            "subjects": 256,
            "positive_cases": len(pairs[profile]),
            "arm_positive_case_comparisons": len(rows),
            "arm_control_comparisons": len(controls),
            "original_edited_metric_values_reproduced": len(rows) * 3,
            "region_cell_counts": {
                name: dict(
                    zip(
                        ("minimum", "median", "maximum"),
                        np.quantile([pair[name] for pair in pairs[profile]], [0, 0.5, 1]).tolist(),
                        strict=True,
                    )
                )
                for name in ("T_cells", "U_cells", "O_cells")
            },
            **original_helpers.summarize_responses(rows, config),
            "nuisance_controls": original_helpers._summarize_controls(controls, config),
        }
        progress["completed_profiles"].append(profile)
    return summaries


def audit_counterfactual(closure_path: Path) -> dict[str, Any]:
    """Run the forward admission once, reusing every frozen matching/metric/bootstrap helper.

    Parameters
    ----------
    closure_path : Path
        Passed explicit original-reproduction binding closure; no replacement model evaluation.

    Returns
    -------
    dict
        Descriptive existing-map audit with complete original failure/provenance/cost history.
    """
    if DESTINATION.exists():
        raise FileExistsError("Forward diagnostic destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in FORWARD_SOURCES}
    inputs = {}
    progress = {"verified_positive_pairs": 0, "completed_profiles": []}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    try:
        threads = _runtime_guard()
        original, sources, inputs, costs = _admission(closure_path)
        config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_text())
        prior_cpu = sum(costs.values())
        cpu_deadline = cpu + config["cpu_seconds_ceiling"] - prior_cpu
        write_new_json(
            DESTINATION / "admission.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "cpu_history": costs,
            },
        )
        if time.process_time() >= cpu_deadline:
            raise TimeoutError("Cumulative protocol CPU ceiling exhausted")
        summaries = _derive(original, config, cpu_deadline, progress)
        verify_inference_hashes(sources | inputs)
        output_hashes = {path.name: sha256_file(path) for path in DESTINATION.iterdir() if path.is_file()}
        size = sum(path.stat().st_size for path in DESTINATION.iterdir() if path.is_file())
        if size > 99 * 1024**2 or time.process_time() >= cpu_deadline:
            raise ValueError("Frozen output/CPU ceiling exhausted")
        summary = {
            "audit_completed": True,
            "version": "forward_provenance_v2",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": output_hashes,
            "progress": progress,
            "profiles": summaries,
            "cpu_history": costs,
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "cumulative_cpu_seconds": prior_cpu + time.process_time() - cpu,
            "new_output_bytes_before_summary": size,
            "new_output_ceiling_bytes": 100 * 1024**2,
            "original_advancement_passed": False,
            "new_advancement_rule": None,
            "new_model_inferences": 0,
            "model_checkpoints_deserialized": 0,
            "gpu_used": False,
            "numerical_thread_environment": threads,
            "real_waveform_payloads_opened": 0,
            "interpretation": "Spent-development descriptive diagnostic; probabilities are uncalibrated",
        }
        write_new_json(DESTINATION / "summary.json", summary)
        return summary
    except Exception as error:
        write_new_json(
            DESTINATION / "failure.json",
            {
                "audit_completed": False,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "progress": progress,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
            },
        )
        raise
