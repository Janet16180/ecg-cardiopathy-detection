"""Verify the native reference adapter by replaying frozen predecessor vectors only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.native_reference_ntt05 import (
    ReferenceInstance,
    SignalRegion,
    match_vector,
    reference_bank,
)
from ecg_experiment.paths import to_stored

PREDECESSOR_SHA = "79cfa5130ccc8d275e398ddb15b056a1affe1beb812fa32c52e12a8c2211e789"
SPACE = "ntt04_frozen_physical_full_median_beat_v1"
FINDINGS = ("NORM", "CRBBB", "CLBBB")
SOURCES = (
    "ecg_experiment/native_reference_ntt05.py",
    "scripts/reports/native_reference_ntt05.py",
    "tests/test_native_reference_ntt05.py",
    "docs/ntt-05-native-signal-adapter.md",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
)


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON without nonfinite substitutions."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_sources() -> dict[str, str]:
    """Require all executable source bytes to match the current committed tree."""
    hashes = {}
    for name in SOURCES:
        committed = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, check=True, capture_output=True)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"Uncommitted engineering source: {name}")
        hashes[name] = digest
    return hashes


def invariant_receipt(path: Path, sources: dict[str, str]) -> tuple[dict, dict]:
    """Bind the executed invariant suite to the exact frozen source bytes."""
    value = json.loads(path.read_text())
    if value["source_sha256"] != sources:
        raise ValueError("Invariant suite belongs to different source bytes")
    counts = value["tests"]
    if counts["passed"] < 26 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise ValueError("Required invariant suite did not pass")
    xml = ROOT / value["junit_path"]
    if sha256_file(xml) != value["junit_sha256"]:
        raise ValueError("Invariant test report bytes differ")
    return counts, {to_stored(path): sha256_file(path), to_stored(xml): value["junit_sha256"]}


def require_budget(started: float) -> None:
    """Enforce the five-minute engineering ceiling at stage boundaries."""
    if time.perf_counter() - started > 300:
        raise TimeoutError("Native engineering verification exceeded 300 seconds")


def predecessor_inputs() -> tuple[dict, dict, list, dict, dict]:
    """Verify saved byte identities and PR67 roles before accessing any feature or finding."""
    base = ROOT / "outputs/ntt04_same_wave"
    path = base / "result.json"
    if sha256_file(path) != PREDECESSOR_SHA:
        raise ValueError("Frozen predecessor result differs")
    result = json.loads(path.read_text())
    hashes = {to_stored(path): PREDECESSOR_SHA}
    for name in (
        "reservation_manifest_local.json",
        "selection_local.json",
        "features_local.json",
        "predictions_local.json",
    ):
        path = base / name
        expected = result["artifact_sha256"][to_stored(path)]
        if sha256_file(path) != expected:
            raise ValueError("Frozen predecessor input differs")
        hashes[to_stored(path)] = expected
    records = json.loads((base / "reservation_manifest_local.json").read_text())["records"]
    permitted = {int(row["ecg_id"]): int(row["patient_id"]) for row in records if row["role"] == "train"}
    selected = json.loads((base / "selection_local.json").read_text())
    rows = selected["targets"] + [r for group in selected["support_draws"] for r in group["supports"]]
    if any(permitted.get(int(row["ecg_id"])) != int(row["patient_id"]) for row in rows):
        raise ValueError("Saved selected identity outside the training role")
    features = json.loads((base / "features_local.json").read_text())
    predictions = json.loads((base / "predictions_local.json").read_text())
    return selected, features, predictions, hashes, result


def predecessor_banks(selected: dict, features: dict) -> dict:
    """Construct the three frozen physical-vector reference banks."""
    banks = {}
    for seed in sorted({group["seed"] for group in selected["support_draws"]}):
        instances = []
        for group in selected["support_draws"]:
            if group["seed"] != seed:
                continue
            for row in group["supports"]:
                vector = np.asarray(features[str(row["ecg_id"])]["raw_full_context"])
                instances.append(
                    ReferenceInstance(
                        str(row["patient_id"]),
                        group["finding"],
                        vector,
                        "existing source record label, not region truth",
                        SPACE,
                    )
                )
        banks[seed] = reference_bank(instances, FINDINGS)
    return banks


def replay(selected: dict, features: dict, predictions: list, original: dict) -> dict:
    """Reconstruct the exact physical full-context choices without extracting new features."""
    banks = predecessor_banks(selected, features)
    saved = {(row["seed"], row["ecg_id"]): row for row in predictions}
    targets = selected["targets"]
    if len(saved) != 180 or len(targets) != 60 or len(banks) != 3:
        raise ValueError("Predecessor denominator differs")
    choices, correct, maximum_score_error = [], 0, 0.0
    for seed, bank in banks.items():
        for target in targets:
            row = saved[(seed, target["ecg_id"])]
            if row["patient_id"] != target["patient_id"] or row["finding"] != target["finding"]:
                raise ValueError("Saved prediction identity/finding mismatch")
            vector = np.asarray(features[str(target["ecg_id"])]["raw_full_context"])
            match = match_vector(
                vector,
                SignalRegion(0, 0.95, ("V1", "V6")),
                bank,
                str(target["patient_id"]),
                -1,
                0,
                "full median beat; region coordinates are replay metadata only",
                SPACE,
                f"ntt04-median:{target['ecg_id']}",
            )
            choice = FINDINGS.index(match.finding)
            choices.append(choice == row["chosen"]["raw_full_context"])
            correct += int(match.finding == target["finding"])
            error = np.max(np.abs(match.scores - np.asarray(row["scores"]["raw_full_context"])))
            maximum_score_error = max(maximum_score_error, float(error))
    accuracy = correct / len(choices)
    exact = bool(all(choices) and correct == 167 and maximum_score_error < 1e-12)
    if abs(accuracy - original["summary"]["balanced_accuracy"]["raw_full_context"]) > 1e-12:
        raise ValueError("Predecessor balanced accuracy mismatch")
    if not exact:
        raise ValueError("Native prototype path does not reproduce predecessor choices")
    return {
        "patients": 60,
        "support_patients": 27,
        "patient_draw_predictions": len(choices),
        "support_draws": 3,
        "choices_exact": True,
        "correct_patient_draw_predictions": correct,
        "predecessor_balanced_accuracy": accuracy,
        "maximum_score_absolute_error": maximum_score_error,
        "embedding_space_enforced": SPACE,
        "new_clinical_result": False,
    }


def main() -> None:
    """Run one immutable engineering verification and bind its final status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--invariants", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "clinical_scoring": False})
    started = time.perf_counter()
    try:
        sources = frozen_sources()
        counts, invariant_hashes = invariant_receipt(args.invariants, sources)
        require_budget(started)
        selected, features, predictions, inputs, original = predecessor_inputs()
        inputs.update(invariant_hashes)
        require_budget(started)
        with threadpool_limits(limits=2):
            checks = replay(selected, features, predictions, original)
        require_budget(started)
        result = {
            "study": "NTT-05",
            "status": "completed_engineering_replay_only",
            "source_sha256": sources,
            "input_sha256": inputs,
            "predecessor_reproduction": checks,
            "engineering_replay_gate_passed": True,
            "invariant_suite": counts,
            "engineering_invariant_gate_passed": True,
            "engineering_combined_gate_passed": True,
            "learned_native_model_executed": False,
            "clinical_localization_evaluated": False,
            "waveform_reads": 0,
            "weight_training": False,
            "gpu_used": False,
            "downloads": False,
            "seconds": time.perf_counter() - started,
            "numpy_version": version("numpy"),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        write_json(output / "result.json", result)
        write_json(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
        print(
            json.dumps(
                {
                    "engineering_replay_passed": True,
                    "new_clinical_score": False,
                    "predictions_exact": checks["patient_draw_predictions"],
                }
            )
        )
    except Exception as error:
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "clinical_scoring": False,
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
