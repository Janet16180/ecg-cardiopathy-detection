"""Profile and cache the frozen CLUSTER-03 native representations on permitted local ECGs."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache import (
    CHECKPOINT,
    SOURCE_DIR,
    QualityExclusionError,
    frozen_sources,
    pack_features,
    read_waveform,
    record_features,
    runtime_fingerprint,
    verified_cohort,
    verify_hashes,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.localization.jepa_native import (
    CHECKPOINT_SHA256,
    EMBEDDING_SPACE,
    encode_signal,
    load_encoder,
)
from ecg_experiment.paths import to_stored


def budget(deadline: float) -> None:
    """Stop between records when the declared stage wall deadline is exceeded."""
    if time.perf_counter() > deadline:
        raise TimeoutError("Prospective native-cache wall deadline exceeded")


def peak_rss() -> int:
    """Return the process high-water resident memory in bytes on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def base_receipt(sources: dict[str, str], inputs: dict[str, str], started: float) -> dict[str, Any]:
    """Bind the model, runtime, committed source and local input provenance."""
    return {
        "study": "CLUSTER-03",
        "status": "completed",
        "source_sha256": sources,
        "input_sha256": inputs,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "embedding_space": EMBEDDING_SPACE,
        "runtime": runtime_fingerprint(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss(),
        "clinical_score": False,
        "gpu_used": False,
        "weight_training": False,
    }


def profile(output: Path, sources: dict[str, str]) -> dict[str, Any]:
    """Measure fixed synthetic inference and the first four permitted real records."""
    started = time.perf_counter()
    deadline = started + 180
    identities, _, _, inputs = verified_cohort()
    rows = identities.head(4).to_dict("records")
    tick = time.perf_counter()
    encoder = load_encoder(SOURCE_DIR, CHECKPOINT)
    load_seconds = time.perf_counter() - tick
    budget(deadline)
    signal = np.random.default_rng(20261019).normal(0, 0.1, (12, 5000)).astype(np.float32)
    tick = time.perf_counter()
    tokens = encode_signal(encoder, signal, "cluster03:synthetic-profile:original-10s")
    synthetic_seconds = time.perf_counter() - tick
    valid = tokens.values.shape == (400, 768) and bool(np.isfinite(tokens.values).all())
    frozen = not encoder.training and all(not parameter.requires_grad for parameter in encoder.parameters())
    elapsed, waveforms = [], {}
    for row in rows:
        budget(deadline)
        tick = time.perf_counter()
        signal = read_waveform(row, waveforms)
        record_features(encoder, signal, int(row["ecg_id"]))
        elapsed.append(time.perf_counter() - tick)
        budget(deadline)
    projected = load_seconds + 1.5 * max(elapsed) * len(identities)
    gates = {
        "finite_dense_shape": valid,
        "all_parameters_frozen": frozen,
        "four_fixed_records_passed": len(elapsed) == 4,
        "real_record_budget": max(elapsed) <= 3,
        "rss_budget": peak_rss() <= 6 * 1024**3,
        "profile_budget": time.perf_counter() - started <= 180,
        "projected_extraction_budget": projected <= 2400,
    }
    return {
        **base_receipt(sources, inputs, started),
        "stage": "native_cache_cpu_profile",
        "profile_gate_passed": all(gates.values()),
        "gates": gates,
        "profile_ecg_ids": [int(row["ecg_id"]) for row in rows],
        "candidate_patients": len(identities),
        "load_seconds": load_seconds,
        "synthetic_seconds": synthetic_seconds,
        "synthetic_forward_count": 1,
        "real_profile_forward_count": 4,
        "real_record_seconds": elapsed,
        "projected_extraction_seconds": projected,
        "waveform_sha256": waveforms,
    }


def validate_profile(
    profile_path: Path, sources: dict[str, str], inputs: dict[str, str], ecg_ids: list[int]
) -> dict[str, Any]:
    """Admit extraction only from the exact completed passing profile and runtime."""
    saved = json.loads(profile_path.read_text())
    status_path = profile_path.with_name("status.json")
    status = json.loads(status_path.read_text())
    if status.get("status") != "completed" or status.get("result_sha256") != sha256_file(profile_path):
        raise ValueError("Profile final status does not bind these result bytes")
    expected = {
        "stage": "native_cache_cpu_profile",
        "status": "completed",
        "source_sha256": sources,
        "input_sha256": inputs,
        "runtime": runtime_fingerprint(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "profile_ecg_ids": ecg_ids,
        "candidate_patients": 1993,
        "profile_gate_passed": True,
    }
    if any(saved.get(name) != value for name, value in expected.items()):
        raise ValueError("Profile source, input, model, runtime or cohort differs")
    required = {
        "finite_dense_shape",
        "all_parameters_frozen",
        "four_fixed_records_passed",
        "real_record_budget",
        "rss_budget",
        "profile_budget",
        "projected_extraction_budget",
    }
    if set(saved["gates"]) != required or not all(value is True for value in saved["gates"].values()):
        raise ValueError("Profile gate is incomplete")
    times = saved["real_record_seconds"]
    if len(times) != 4 or not np.isfinite(times).all() or min(times) < 0 or max(times) > 3:
        raise ValueError("Profile real-record measurements violate prospective limits")
    projected = saved["load_seconds"] + 1.5 * max(times) * 1993
    if (
        saved["seconds"] > 180
        or saved["peak_rss_bytes"] > 6 * 1024**3
        or projected > 2400
        or saved["projected_extraction_seconds"] != projected
    ):
        raise ValueError("Profile budget measurements violate prospective limits")
    verify_hashes(saved["waveform_sha256"], {})
    return saved


def extract(output: Path, sources: dict[str, str], profile_path: Path) -> dict[str, Any]:
    """Extract every admitted candidate once, preserving only prospective quality exclusions."""
    started = time.perf_counter()
    deadline = started + 2400
    identities, _, _, inputs = verified_cohort()
    saved = validate_profile(profile_path, sources, inputs, identities.ecg_id.head(4).astype(int).tolist())
    inputs[to_stored(profile_path)] = sha256_file(profile_path)
    inputs[to_stored(profile_path.with_name("status.json"))] = sha256_file(
        profile_path.with_name("status.json")
    )
    rows = identities.to_dict("records")
    write_json_atomic(output / "identity_rows_local.json", rows)
    tick = time.perf_counter()
    encoder = load_encoder(SOURCE_DIR, CHECKPOINT)
    load_seconds = time.perf_counter() - tick
    accepted, excluded, features, elapsed = [], [], [], []
    waveforms = dict(saved["waveform_sha256"])
    for index, row in enumerate(rows):
        budget(deadline)
        tick = time.perf_counter()
        try:
            signal = read_waveform(row, waveforms)
            feature = record_features(encoder, signal, int(row["ecg_id"]))
        except QualityExclusionError as error:
            excluded.append({**row, "reason": str(error), "type": type(error).__name__})
        else:
            accepted.append(row)
            features.append(feature)
        elapsed.append(time.perf_counter() - tick)
        write_json_atomic(
            output / "status.json",
            {
                "status": "extracting",
                "completed": index + 1,
                "total": len(rows),
                "accepted": len(accepted),
                "excluded": len(excluded),
            },
        )
        if (index + 1) % 100 == 0:
            print(f"Native cache: {index + 1}/{len(rows)}", flush=True)
        budget(deadline)
    write_json_atomic(output / "accepted_local.json", accepted)
    write_json_atomic(output / "exclusions_local.json", excluded)
    write_npz_atomic(output / "native_features_local.npz", **pack_features(accepted, features))
    budget(deadline)
    return {
        **base_receipt(sources, inputs, started),
        "stage": "native_cache_extraction",
        "candidate_patients": len(rows),
        "accepted_patients": len(accepted),
        "excluded_patients": len(excluded),
        "retention_fraction": len(accepted) / len(rows),
        "retention_gate_passed": len(accepted) >= 0.95 * len(rows),
        "load_seconds": load_seconds,
        "record_seconds": elapsed,
        "extraction_forward_count": len(accepted),
        "profile_forwards_reused": False,
        "waveform_sha256": waveforms,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in sorted(output.iterdir())
            if path.name not in ("status.json", "result.json", "failure.json")
        },
    }


def main() -> None:
    """Execute a committed profile or admitted extraction with typed local failure receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "extract"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.stage == "extract" and args.profile is None:
        parser.error("extraction requires a completed --profile result.json")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    write_json_atomic(output / "status.json", {"status": "running", "stage": args.stage})
    started = time.perf_counter()
    sources: dict[str, str] = {}
    try:
        sources = frozen_sources()
        with threadpool_limits(limits=2):
            result = (
                profile(output, sources)
                if args.stage == "profile"
                else extract(
                    output, sources, args.profile if args.profile.is_absolute() else ROOT / args.profile
                )
            )
        write_json_atomic(output / "result.json", result)
        gate = "profile_gate_passed" if args.stage == "profile" else "retention_gate_passed"
        if not result[gate]:
            raise RuntimeError(f"Prospective {gate} failed")
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
        print(
            json.dumps({"stage": args.stage, "status": "completed", "seconds": result["seconds"]}), flush=True
        )
    except Exception as error:
        write_json_atomic(
            output / "failure.json",
            {
                "stage": args.stage,
                "type": type(error).__name__,
                "message": str(error),
                "source_sha256": sources,
                "runtime": runtime_fingerprint(),
                "seconds": time.perf_counter() - started,
                "peak_rss_bytes": peak_rss(),
            },
        )
        write_json_atomic(output / "status.json", {"status": "failed", "stage": args.stage})
        raise


if __name__ == "__main__":
    main()
