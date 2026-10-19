"""Audit the forward-only v2 generated bank after the retained v1 resource stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import noise_for_seed, require
from ecg_experiment.ssl_synthetic_quality_saved_audit import (
    coherence_values,
    diversity_values,
    intervention_values,
    morphology_values,
    primary_values,
    reconstruct,
)

RELATIVE = "outputs/ssl_synthetic_localization/generated_quality_v2"
PARENT = "outputs/ssl_synthetic_localization/pretrained_generator_standalone_v1"
CORE_NAMES = (
    "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
    "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
    "ecg_experiment/intervals.py",
)
CORE_HASHES = dict(
    zip(
        CORE_NAMES,
        (
            "14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6",
            "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
        ),
        strict=True,
    )
)


def compare(expected: Any, actual: Any, label: str) -> int:
    """Compare complete nested numeric metadata with a fixed tolerance.

    Parameters
    ----------
    expected, actual : Any
        Recomputed and saved values.
    label : str
        Diagnostic path.

    Returns
    -------
    int
        Count of checked scalar values.
    """
    if isinstance(expected, dict):
        require(set(expected) == set(actual), f"{label} fields")
        return sum(compare(value, actual[key], f"{label}/{key}") for key, value in expected.items())
    if isinstance(expected, list):
        require(len(expected) == len(actual), f"{label} length")
        return sum(compare(left, right, label) for left, right in zip(expected, actual, strict=True))
    if isinstance(expected, float):
        require(bool(np.isfinite(actual)) and abs(expected - actual) <= 1e-12, f"{label} numeric")
    else:
        require(expected == actual, f"{label} value")
    return 1


def file_digest(path: Path, guard: Any) -> str:
    """Hash an allowlisted file with deadline checks.

    Parameters
    ----------
    path : Path
        Authorized file.
    guard : callable
        Audit resource guard.

    Returns
    -------
    str
        File digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            guard()
            part = stream.read(1024**2)
            if not part:
                break
            digest.update(part)
    return digest.hexdigest()


def bound_path(root: Path, name: str) -> Path:
    """Confine a binding to its declared directory.

    Parameters
    ----------
    root : Path
        Permitted directory.
    name : str
        Relative filename.

    Returns
    -------
    Path
        Bound local path.
    """
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "relative path only")
    path = root / relative
    require(not path.is_symlink(), "bound file symlink prohibited")
    return path


def bindings(
    root: Path, source_root: Path, study: Path, parent: Path, main: dict[str, Any], guard: Any
) -> int:
    """Verify authored snapshots, predecessor inputs and saved outputs.

    Parameters
    ----------
    root, source_root, study, parent : Path
        Authorized output host, source and artifact locations.
    main : dict
        Main summary.
    guard : callable
        Audit deadline guard.

    Returns
    -------
    int
        Verified binding count.
    """
    count = 0
    for name, expected in main["source_sha256"].items():
        for path in (bound_path(source_root, name), bound_path(study / "source_snapshot", name)):
            require(file_digest(path, guard) == expected, f"source binding {name}")
            count += 1
    for root, catalog in ((parent, main["input_sha256"]), (study, main["output_sha256"])):
        for name, expected in catalog.items():
            require(file_digest(bound_path(root, name), guard) == expected, f"artifact binding {name}")
            count += 1
    amendment = main["amendment_input_sha256"]
    for name in ("failure.json", "profile.json", "prospective_ledger.json"):
        require(
            f"outputs/ssl_synthetic_localization/generated_quality_v1/{name}" in amendment,
            "failed v1 receipt binding",
        )
    for name, expected in amendment.items():
        require(file_digest(bound_path(root, name), guard) == expected, "amendment receipt binding")
        count += 1
    return count


def audit_arrays(study: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct every fixed seed and intervention and independently score the bank.

    Parameters
    ----------
    study : Path
        Saved numerical bank directory.
    main : dict
        Bound main summary.
    guard : callable
        Audit deadline guard.

    Returns
    -------
    dict[str, Any]
        Recomputed primary, scalar check count and saved-array integrity.
    """
    arrays = {
        name: np.load(study / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("noise", "raw", "variants", "strict_support", "support", "replay_raw")
    }
    for name, shape, dtype in (
        ("noise", (128, 1, 8, 5000), np.float32),
        ("raw", (128, 8, 5000), np.float32),
        ("variants", (128, 4, 12, 5000), np.float32),
        ("strict_support", (128, 2, 12, 5000), bool),
        ("support", (128, 2, 12, 5000), bool),
        ("replay_raw", (8, 5000), np.float32),
    ):
        require(arrays[name].shape == shape and arrays[name].dtype == dtype, f"{name} geometry")
    rows = [json.loads(line) for line in (study / "cases.jsonl").read_text().splitlines()]
    require(len(rows) == 128, "128 case rows")
    require(main["completed_seeds"] == list(range(2026103000, 2026103128)), "complete fixed seeds")
    checks, coherence_rows, hashes = 0, [], set()
    for index, row in enumerate(rows):
        guard()
        seed = 2026103000 + index
        require(arrays["noise"][index].tobytes() == noise_for_seed(seed).tobytes(), "CPU noise bytes")
        raw = np.asarray(arrays["raw"][index])
        expected, strict, support, parameters = reconstruct(raw, index)
        require(expected.tobytes() == arrays["variants"][index].tobytes(), "all variant bytes")
        require(strict.tobytes() == arrays["strict_support"][index].tobytes(), "strict mask bytes")
        require(support.tobytes() == arrays["support"][index].tobytes(), "threshold mask bytes")
        independent = expected[0, [0, 1, 6, 7, 8, 9, 10, 11]]
        require(bool(np.all(np.ptp(raw, axis=1) > 0)), "nonflat native")
        require(bool(np.all(np.ptp(expected[0], axis=1) > 0)), "nonflat canonical")
        coherent = coherence_values(independent, index)
        expected_row = {
            "index": index,
            "seed": seed,
            "standardization": parameters,
            "coherence": coherent,
            "morphology": morphology_values(independent, raw),
            "interventions": intervention_values(expected, strict, support),
        }
        checks += compare(expected_row, row, f"case{index}")
        coherence_rows.append(coherent)
        hashes.add(hashlib.sha256(raw.tobytes()).hexdigest())
    require(len(hashes) == 128, "distinct raw banks")
    require(arrays["raw"][0].tobytes() == arrays["replay_raw"].tobytes(), "saved GPU replay bytes")
    primary = primary_values(coherence_rows)
    checks += compare(primary, main["primary"], "primary")
    checks += compare(diversity_values(arrays["variants"]), main["diversity"], "diversity")
    return {
        "primary": primary,
        "scalar_checks": checks,
        "distinct_raw_hashes": len(hashes),
        "complete_variant_reconstruction": True,
        "exact_masks": True,
        "saved_gpu_replay_exact": True,
    }


def verify_resources(root: Path, study: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Verify declared stage budgets, profile forecasts and ledger arithmetic.

    Parameters
    ----------
    root, study : Path
        Output host and main output directory.
    main : dict
        Bound main summary.
    guard : callable
        Deadline guard.

    Returns
    -------
    dict[str, Any]
        Verified terminal costs and bytes.
    """
    profile = json.loads((study / "profile.json").read_text())
    compare(profile, main["profile"], "profile")
    for kind, limit in (("cpu", 1800), ("wall", 1800)):
        forecast = profile[f"spent_{kind}_seconds"] + 260 * profile[f"{kind}_seconds"] + 30
        require(abs(forecast - profile[f"forecast_{kind}_seconds"]) <= 1e-12, "profile forecast")
        require(forecast < limit and 0 < main[f"{kind}_seconds"] < limit, "main time cap")
    require(
        main["forward_counts"]
        == {"cpu_predecessor_replay": 4, "gpu_profile": 1, "gpu_generation": 128, "gpu_replay": 1},
        "forward counts",
    )
    require(main["forward_attempts"] == main["forward_counts"], "no incomplete forwards")
    require(profile["forecast_saved_bytes"] < 512 * 1024**2, "profile output forecast")
    require(profile["peak_allocated_bytes"] < 2 * 1024**3, "profile memory forecast")
    metadata = main["gpu_metadata"]
    require(metadata["free_bytes_at_admission"] >= 2 * 1024**3, "GPU admission memory")
    require(metadata["peak_allocated_bytes"] < 2 * 1024**3, "GPU peak allocation")
    require(metadata["deterministic"] is True, "deterministic GPU path")
    require(metadata["matmul_tf32"] is False and metadata["cudnn_tf32"] is False, "float32 CUDA math")
    require(metadata["torch_threads"] == metadata["interop_threads"] == 1, "single torch threads")
    require(0 < main["gpu_lock_wall_seconds"] < 1800, "charged lock wall")
    ledger = json.loads((study / "prospective_ledger.json").read_text())
    require(ledger["cpu_seconds"] >= main["cpu_seconds"], "terminal CPU accounting")
    require(ledger["wall_seconds"] >= main["wall_seconds"], "terminal wall accounting")
    require(ledger["cpu_seconds"] < 1800 and ledger["wall_seconds"] < 1800, "terminal main cap")
    require(ledger["gpu_wall_seconds"] == main["gpu_lock_wall_seconds"], "GPU ledger")
    require(ledger["receipt_sha256"] == file_digest(study / "summary.json", guard), "main ledger binding")
    entries = ledger["prior_stage_entries"]
    expected_paths = {
        f"{PARENT}/prospective_ledger.json",
        f"{PARENT}/independent_audit_v1/prospective_ledger.json",
        "outputs/ssl_synthetic_localization/generated_quality_v1/prospective_ledger.json",
        "outputs/ssl_synthetic_localization/generated_quality_v1/independent_failure_audit_v1/prospective_ledger.json",
    }
    require({entry["path"] for entry in entries} == expected_paths, "exact prior stage accounting scope")
    admission = json.loads((study / "amendment_admission.json").read_text())
    compare(entries, admission["prior_stage_entries"], "admitted prior ledger entries")
    compare(main["amendment_input_sha256"], admission["amendment_input_sha256"], "admitted amendment bytes")
    require(
        len(entries) == len({entry["path"] for entry in entries}) == 4, "four distinct prior stage self-costs"
    )
    prior_cpu, prior_gpu = 0.0, 0.0
    for entry in entries:
        path = bound_path(root, entry["path"])
        require(file_digest(path, guard) == entry["sha256"], "prior ledger byte identity")
        prior = json.loads(path.read_text())
        require(prior["cpu_seconds"] == entry["cpu_seconds"], "prior self CPU cost")
        require(prior["gpu_wall_seconds"] == entry["gpu_wall_seconds"], "prior self GPU cost")
        prior_cpu += prior["cpu_seconds"]
        prior_gpu += prior["gpu_wall_seconds"]
    require(ledger["prior_accounting_complete"] is True, "prior ledger completeness")
    require(ledger["cumulative_cpu_seconds"] == prior_cpu + ledger["cpu_seconds"], "additive CPU ledger")
    require(
        ledger["cumulative_gpu_wall_seconds"] == prior_gpu + ledger["gpu_wall_seconds"], "additive GPU ledger"
    )
    actual = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file() and "independent_audit_v1" not in path.parts
    )
    receipt_bytes = sum((study / name).stat().st_size for name in ("summary.json", "prospective_ledger.json"))
    require(actual - receipt_bytes == main["output_bytes_before_receipt"], "main byte accounting")
    require(actual < 512 * 1024**2, "main byte cap")
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": ledger["gpu_wall_seconds"],
        "main_bytes": actual,
        "main_ledger_sha256": file_digest(study / "prospective_ledger.json", guard),
        "prior_cpu_seconds": prior_cpu,
        "prior_gpu_wall_seconds": prior_gpu,
        "current_machine_cpu_through_main": ledger["cumulative_cpu_seconds"],
        "current_machine_gpu_wall_seconds": ledger["cumulative_gpu_wall_seconds"],
    }


def verify_figures(study: Path) -> None:
    """Verify fixed rendered figure artifact signatures and dimensions.

    Parameters
    ----------
    study : Path
        Saved study artifacts.
    """
    for name, size in (
        ("fixed_first_eight.png", (2600, 2080)),
        ("known_support.png", (2860, 2600)),
        ("quality_diagnostics.png", (1950, 520)),
    ):
        with (study / name).open("rb") as stream:
            header = stream.read(24)
        require(header[:8] == b"\x89PNG\r\n\x1a\n", "required PNG signature")
        require(struct.unpack(">II", header[16:24]) == size, "required figure geometry")


def run(root: Path, source_root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Execute one admitted copied checker under separate audit budgets.

    Parameters
    ----------
    root, source_root : Path
        Output host and frozen source worktree.
    main_sha, script_sha : str
        Explicit coordinator-approved bindings.

    Returns
    -------
    dict[str, Any]
        Independent saved-output audit summary.
    """
    clock = (time.process_time(), time.perf_counter())
    study, parent = root / RELATIVE, root / PARENT
    destination = study / "independent_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "execute exact copied checker")
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit already attempted",
    )
    costs: dict[str, Any] = {}

    def guard() -> None:
        require(time.process_time() - clock[0] < 178, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 178, "audit wall deadline")
        require(
            sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) < 15 * 1024**2,
            "audit receipt byte reserve",
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"audit timer {signum}")

    for timer, sig in ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)):
        require(signal.getitimer(timer) == (0, 0), "exclusive audit timer")
        signal.signal(sig, interrupted)
        signal.setitimer(timer, 178)
    receipt_name = "summary.json"
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single numerical thread")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(file_digest(Path(__file__), guard) == script_sha, "admitted checker SHA")
        for name, expected in CORE_HASHES.items():
            require(file_digest(source_root / name, guard) == expected, "independent core SHA")
        require(file_digest(study / "summary.json", guard) == main_sha, "admitted main SHA")
        main = json.loads((study / "summary.json").read_text())
        verified = bindings(root, source_root, study, parent, main, guard)
        costs = verify_resources(root, study, main, guard)
        verify_figures(study)
        recomputed = audit_arrays(study, main, guard)
        parent_main = json.loads((parent / "summary.json").read_text())
        parent_audit = json.loads((parent / "independent_audit_v1/summary.json").read_text())
        require(
            parent_audit["all_passed"] is True and parent_audit["mismatch_count"] == 0,
            "passed independent predecessor audit",
        )
        require(
            main["model_state_sha256"] == main["final_state_sha256"] == parent_main["model_state_sha256"],
            "source-bound unchanged model identity",
        )
        closure = json.loads((study / "predecessor_closure.json").read_text())
        require(
            closure["all_exact"] is True and closure["cpu_predecessor_forwards"] == 4,
            "source-bound predecessor replay",
        )
        require(
            all(all(case.values()) for case in closure["cases"]) and len(closure["cases"]) == 4,
            "all source-bound predecessor output representations",
        )
        gate_names = (
            "complete_128_bank",
            "finite_bank",
            "nonflat_native_and_standardized",
            "distinct_raw_outputs",
            "exact_gpu_replay",
            "limb_equations",
            "sham_exact",
            "support_nonempty_and_subset",
            "support_masks_exact",
            "no_changes_outside_hull",
            "retained_energy_at_least_99pct",
            "predecessor_exact",
            "model_state_unchanged",
            "model_source_identity",
            "resources",
        )
        compare(dict.fromkeys(gate_names, True), main["gates"], "all operational gates")
        require(
            all(main["gates"].values()) and main["operational_support_qualified"] is True, "operational gates"
        )
        for key in ("new_training_updates", "real_waveform_reads", "external_api_calls"):
            require(main[key] == 0, f"zero {key}")
        for key in (
            "physical_units_verified",
            "clinical_fidelity_validated",
            "disease_control_validated",
            "real_disease_localization_validated",
        ):
            require(main[key] is False, f"unverified {key}")
        require(bindings(root, source_root, study, parent, main, guard) == verified, "bindings unchanged")
        require(file_digest(study / "summary.json", guard) == main_sha, "main unchanged")
        require(
            file_digest(study / "prospective_ledger.json", guard) == costs["main_ledger_sha256"],
            "main ledger unchanged",
        )
        require(file_digest(Path(__file__), guard) == script_sha, "checker unchanged")
        for name, expected in CORE_HASHES.items():
            require(file_digest(source_root / name, guard) == expected, "core unchanged")
        guard()
        total_bytes = sum(p.stat().st_size for p in study.rglob("*") if p.is_file())
        require(total_bytes + 1024**2 < 528 * 1024**2, "main plus audit byte reserve")
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            **recomputed,
            "resources": costs,
            "bindings_verified": verified,
            "study_bytes_before_audit_receipt": total_bytes,
            "main_sha256": main_sha,
            "script_sha256": script_sha,
            "core_sha256": CORE_HASHES,
            "independent_model_forwards": 0,
            "predecessor_runtime_replay_source_bound": True,
            "architecture_execution_source_bound": True,
            "raw_real_ecg_reads": 0,
            "gpu_used": False,
            "clinical_claim": False,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cpu_seconds": time.process_time() - clock[0],
            "wall_seconds": time.perf_counter() - clock[1],
        }
        with (destination / receipt_name).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        receipt_name = "failure.json"
        with (destination / receipt_name).open("x") as stream:
            json.dump(
                {
                    "all_passed": False,
                    "failure_type": type(error).__name__,
                    "error": str(error),
                    "cpu_seconds": time.process_time() - clock[0],
                    "wall_seconds": time.perf_counter() - clock[1],
                    "main_sha256": main_sha,
                    "script_sha256": script_sha,
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        raise
    finally:
        digest = hashlib.sha256((destination / receipt_name).read_bytes()).hexdigest()
        cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
        with (destination / "prospective_ledger.json").open("x") as stream:
            json.dump(
                {
                    "receipt": receipt_name,
                    "sha256": digest,
                    "cpu_seconds": cpu,
                    "wall_seconds": wall,
                    "gpu_wall_seconds": 0.0,
                    "main_cpu_seconds": costs.get("main_cpu_seconds"),
                    "stage_main_plus_audit_cpu_seconds": costs["main_cpu_seconds"] + cpu if costs else None,
                    "stage_gpu_wall_seconds": costs.get("main_gpu_wall_seconds"),
                    "current_machine_cumulative_cpu_seconds": costs["current_machine_cpu_through_main"] + cpu
                    if costs
                    else None,
                    "current_machine_cumulative_gpu_wall_seconds": costs.get(
                        "current_machine_gpu_wall_seconds"
                    ),
                    "historical_costs_verified": False,
                    "measurement_endpoint": "after audit receipt write/hash, before ledger serialization",
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse the explicitly admitted copied-script invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
