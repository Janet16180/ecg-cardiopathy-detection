"""Execute the prospectively bounded public-generator quality study on CUDA."""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_pretrained_generator as shared
from ecg_experiment.gpu import gpu_lock

CONFIG = "configs/ssl_synthetic_generated_quality_v1.json"
ADMISSION = "configs/ssl_synthetic_generated_quality_admission_v1.json"
SOURCES = (
    CONFIG,
    ADMISSION,
    "docs/ssl-synthetic-localization-generated-quality-admission-v1.md",
    "docs/ssl-synthetic-localization-generated-quality-v1.md",
    "ecg_experiment/ssl_synthetic_generated_quality.py",
    "ecg_experiment/ssl_synthetic_generated_quality_run.py",
    "scripts/experiments/run_ssl_synthetic_generated_quality.py",
    "tests/test_ssl_synthetic_generated_quality.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/gpu.py",
    *shared.SOURCE_NAMES,
)


def prepare_case(raw: np.ndarray, index: int, seed: int) -> tuple[np.ndarray, ...]:
    """Run the entire fixed CPU transformation and scoring path for one generated sample.

    Parameters
    ----------
    raw : np.ndarray
        Native eight-lead float32 signal.
    index : int
        Fixed position in the bank.
    seed : int
        Noise seed for the retained row.

    Returns
    -------
    tuple[np.ndarray, ...]
        Four variants, strict masks, thresholded masks and measurement metadata.
    """
    standardized, parameters = quality.standardize(raw)
    variants, strict, support = quality.intervention_bank(standardized, index)
    row = {
        "index": index,
        "seed": seed,
        "standardization": parameters,
        "coherence": quality.coherence(standardized, index),
        "morphology": quality.morphology(standardized, raw),
        "interventions": quality.intervention_metrics(variants, strict, support),
    }
    return variants, strict, support, row


def gpu_forward(model: torch.nn.Module, noise: np.ndarray) -> np.ndarray:
    """Execute the single admitted batch-one CUDA path.

    Parameters
    ----------
    model : torch.nn.Module
        Reviewed, strictly loaded generator in evaluation mode on CUDA0.
    noise : np.ndarray
        Retained CPU float32 input of shape (1,8,5000).

    Returns
    -------
    np.ndarray
        Native float32 CPU output with shape (8,5000).
    """
    with torch.inference_mode():
        output = model(torch.from_numpy(noise).to("cuda:0"))
    torch.cuda.synchronize(0)
    raw = output[0].detach().cpu().numpy().copy()
    if raw.shape != (8, 5000) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Native GPU output geometry or finiteness failed")
    return raw


def predecessor_closure(
    parent: Path,
    summary: dict[str, Any],
    model: torch.nn.Module,
    counts: dict[str, int],
    attempts: dict[str, int],
) -> dict[str, Any]:
    """Exactly replay the available immediate predecessor's four CPU inputs.

    Parameters
    ----------
    parent : Path
        Verified local standalone probe directory.
    summary : dict[str, Any]
        Passed predecessor summary.
    model : torch.nn.Module
        Reviewed CPU generator.

    Returns
    -------
    dict[str, Any]
        Exact replay checks for all four seeds and three representations.
    """
    noises = np.load(parent / "noise.npy", allow_pickle=False)
    checks = []
    for index, files in enumerate(summary["case_files"]):
        attempts["cpu_predecessor_replay"] += 1
        actual = shared.generate(model, noises[index])
        counts["cpu_predecessor_replay"] += 1
        checks.append(
            {
                key: bool(np.array_equal(value, np.load(parent / files[key], allow_pickle=False)))
                for key, value in zip(["raw", "author_export", "canonical12"], actual, strict=True)
            }
        )
    if not all(all(row.values()) for row in checks):
        raise ValueError("Immediate predecessor CPU replay failed")
    return {"cpu_predecessor_forwards": 4, "all_exact": True, "cases": checks}


def snapshot(destination: Path) -> dict[str, str]:
    """Retain immutable copies of every source named by the executable manifest.

    Parameters
    ----------
    destination : Path
        New study directory.

    Returns
    -------
    dict[str, str]
        Repository-relative source digests.
    """
    digests = {}
    for name in dict.fromkeys(SOURCES):
        path = destination / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        digests[name] = shared.file_sha256(path)
    return digests


def set_runtime() -> None:
    """Require the frozen numerical resource settings and deterministic CUDA operations."""
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(key) != "1":
            raise ValueError("All numerical thread environment settings must be one")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Deterministic CUDA requires the prospective CUBLAS workspace setting")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def timeout_handler(signum: int, frame: Any) -> None:
    """Interrupt before the frozen wall or CPU budget is exhausted."""
    raise TimeoutError(f"Quality experiment deadline reached ({signum})")


def bank_gates(
    raw: np.ndarray,
    variants: np.ndarray,
    masks: np.ndarray,
    strict: np.ndarray,
    rows: list[dict[str, Any]],
    replay: np.ndarray,
) -> dict[str, bool]:
    """Collect the prospective operational checks without grading physiology.

    Parameters
    ----------
    raw, variants, masks, strict : np.ndarray
        Complete numerical banks and exact intervention supports.
    rows : list[dict[str, Any]]
        Fixed-seed measurement records.
    replay : np.ndarray
        First-seed CUDA replay output.

    Returns
    -------
    dict[str, bool]
        Every bank and intervention operational gate.
    """
    limb = all(
        np.allclose(case, quality.derive_leads(case[[0, 1, 6, 7, 8, 9, 10, 11]]), atol=2e-6, rtol=1e-6)
        for subject in variants
        for case in subject
    )
    stats = [row["interventions"][mode] for row in rows for mode in ["focal", "repeated"]]
    return {
        "complete_128_bank": raw.shape == (128, 8, 5000) and len(rows) == 128,
        "finite_bank": bool(np.isfinite(raw).all() and np.isfinite(variants).all()),
        "nonflat_native_and_standardized": bool(
            np.all(np.ptp(raw, axis=2) > 0) and np.all(np.ptp(variants[:, 0], axis=2) > 0)
        ),
        "distinct_raw_outputs": len({shared.array_sha256(case) for case in raw}) == len(raw),
        "exact_gpu_replay": bool(np.array_equal(raw[0], replay)),
        "limb_equations": limb,
        "sham_exact": all(
            np.array_equal(variants[i, 0], quality.derive_leads(quality.standardize(case)[0]))
            for i, case in enumerate(raw)
        ),
        "support_nonempty_and_subset": bool(
            masks.reshape(128, 2, -1).any(axis=2).all()
            and strict.reshape(128, 2, -1).any(axis=2).all()
            and np.all(~masks | strict)
        ),
        "support_masks_exact": all(r["strict_mask_exact"] and r["threshold_mask_exact"] for r in stats),
        "no_changes_outside_hull": all(r["outside_intended_hull_cells"] == 0 for r in stats),
        "retained_energy_at_least_99pct": all(r["retained_energy_fraction"] >= 0.99 for r in stats),
    }


def diversity_summary(variants: np.ndarray) -> dict[str, Any]:
    """Describe all distinct-seed standardized RMS distances without choosing a cutoff.

    Parameters
    ----------
    variants : np.ndarray
        Canonical four-variant bank.

    Returns
    -------
    dict[str, Any]
        Pair count and fixed percentiles of independent-lead pairwise distances.
    """
    independent = variants[:, 0, [0, 1, 6, 7, 8, 9, 10, 11]].astype(np.float64)
    distances = [
        float(np.sqrt(np.mean((independent[i] - independent[j]) ** 2)))
        for i in range(len(independent))
        for j in range(i)
    ]
    return {
        "pairs": len(distances),
        "percentiles_0_25_50_75_100": np.percentile(distances, [0, 25, 50, 75, 100]).tolist(),
        "near_duplicate_cutoff": None,
    }


def save_outputs(destination: Path, banks: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> None:
    """Persist complete banks and create the fixed local figures.

    Parameters
    ----------
    destination : Path
        Fresh study output directory.
    banks : dict[str, np.ndarray]
        Numerical banks keyed by filename stem.
    rows : list[dict[str, Any]]
        Per-seed measurements.
    """
    for name, values in banks.items():
        path = destination / f"{name}.npy"
        if path.exists():
            if not np.array_equal(np.load(path, allow_pickle=False), values):
                raise ValueError("Incrementally saved bank differs from completed bank")
        else:
            quality.save_array(path, values)
    quality.write_rows(destination / "cases.jsonl", rows)
    quality.make_figures(destination, banks["raw"], banks["variants"], banks["support"], rows)


def run_study() -> dict[str, Any]:
    """Execute one committed and reviewed study, retaining any failed attempt.

    Returns
    -------
    dict[str, Any]
        Output-bound engineering results without clinical claims.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = json.loads((ROOT / CONFIG).read_text())
    destination = ROOT / config["output_root"]
    destination.mkdir(exist_ok=False)
    parent = ROOT / config["predecessor_root"]
    resources = config["resources"]
    counts = {"cpu_predecessor_replay": 0, "gpu_profile": 0, "gpu_generation": 0, "gpu_replay": 0}
    attempts = dict.fromkeys(counts, 0)
    gpu_cost = 0.0
    sources, inputs = {}, {}
    rows = []
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
        set_runtime()
        arm_timers(resources, old_handlers)
        sources = snapshot(destination)
        model, main, inputs, predecessor = admit_predecessor(parent, destination, counts, attempts)
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
        banks, profile, gpu_metadata, gpu_cost = generate_bank(
            model, noises, config, counts, attempts, clock, destination, rows, guard
        )
        gates = bank_gates(
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
        save_outputs(destination, banks, rows)
        guard()
        verify_bindings(destination, parent, sources, inputs)
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
            "diversity": diversity_summary(banks["variants"]),
            "source_sha256": sources,
            "input_sha256": inputs,
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
            "accounting_scope": "current_machine_standalone_only; historical costs unverified",
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
        finish_ledger(destination, result, "summary.json", clock, parent)
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
            "gpu_lock_wall_seconds": clock.get("gpu_cost", gpu_cost),
            **quality.elapsed(clock),
            "historical_costs_verified": False,
        }
        if rows and not (destination / "partial_cases.jsonl").exists():
            quality.write_rows(destination / "partial_cases.jsonl", rows)
        shared.write_new_json(destination / "failure.json", failure)
        finish_ledger(destination, failure, "failure.json", clock, parent)
        raise
    finally:
        for timer, sig in [(signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)]:
            if sig in old_handlers:
                signal.setitimer(timer, 0)
                signal.signal(sig, old_handlers[sig])


def generate_bank(
    model: torch.nn.Module,
    noises: np.ndarray,
    config: dict[str, Any],
    counts: dict[str, int],
    attempts: dict[str, int],
    clock: dict[str, float],
    destination: Path,
    rows: list[dict[str, Any]],
    guard: Any,
) -> tuple[dict[str, np.ndarray], ...]:
    """Profile and execute the exact CUDA bank under the shared project lock.

    Parameters
    ----------
    model : torch.nn.Module
        Reviewed CPU generator.
    noises : np.ndarray
        Fixed input bank.
    config, counts, clock : dict
        Prospective configuration and mutable accounting.
    destination : Path
        Fresh local study directory.
    rows : list[dict[str, Any]]
        Completed measurement rows, retained on failure.
    guard : callable
        Live resource check.

    Returns
    -------
    tuple
        Complete banks, profile, GPU metadata and charged lock wall.
    """
    resources = config["resources"]
    quality.save_array(destination / "noise.npy", noises)
    with measured_gpu_lock(clock):
        free, total = torch.cuda.mem_get_info(0)
        if free < resources["minimum_gpu_free_bytes"]:
            raise ValueError("GPU0 admission has less than the frozen free-memory requirement")
        torch.cuda.reset_peak_memory_stats(0)
        model.to("cuda:0")
        torch.cuda.synchronize(0)
        profile_start = {"cpu": time.process_time(), "wall": time.perf_counter()}
        attempts["gpu_profile"] += 1
        raw_profile = gpu_forward(model, noises[0])
        counts["gpu_profile"] += 1
        prepare_case(raw_profile, 0, config["first_seed"])
        profile = quality.elapsed(profile_start)
        spent = quality.elapsed(clock)
        profile["spent_cpu_seconds"] = spent["cpu_seconds"]
        profile["spent_wall_seconds"] = spent["wall_seconds"]
        for label in ["cpu", "wall"]:
            profile[f"forecast_{label}_seconds"] = (
                spent[f"{label}_seconds"]
                + resources["profile_safety_factor"] * 130 * profile[f"{label}_seconds"]
                + resources["report_reserve_seconds"]
            )
        profile["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        profile["forecast_saved_bytes"] = (
            noises.nbytes + 128 * (8 * 5000 * 4 + 4 * 12 * 5000 * 4 + 4 * 12 * 5000) + 32 * 1024**2
        )
        shared.write_new_json(destination / "profile.json", profile)
        if (
            profile["forecast_wall_seconds"] >= resources["wall_seconds"]
            or profile["forecast_cpu_seconds"] >= resources["cpu_seconds"]
            or profile["forecast_saved_bytes"] >= resources["output_bytes"]
            or profile["peak_allocated_bytes"] >= resources["maximum_gpu_allocated_bytes"]
        ):
            raise ValueError("Actual-path profile fails the frozen conservative resource forecasts")
        banks = create_banks(destination, noises)
        for index, noise in enumerate(noises):
            guard()
            attempts["gpu_generation"] += 1
            raw = gpu_forward(model, noise)
            counts["gpu_generation"] += 1
            banks["raw"][index] = raw
            banks["raw"].flush()
            variants, strict, support, row = prepare_case(raw, index, config["first_seed"] + index)
            for key, value in [
                ("variants", variants),
                ("strict_support", strict),
                ("support", support),
            ]:
                banks[key][index] = value
                banks[key].flush()
            rows.append(row)
        guard()
        attempts["gpu_replay"] += 1
        replay = gpu_forward(model, noises[0])
        counts["gpu_replay"] += 1
        quality.save_array(destination / "replay_raw.npy", replay)
        metadata = {
            "name": torch.cuda.get_device_name(0),
            "device_index": 0,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0],
            "free_bytes_at_admission": free,
            "total_bytes": total,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }
        model.cpu()
        torch.cuda.synchronize(0)
    banks["replay_raw"] = replay
    return banks, profile, metadata, clock["gpu_cost"]


@contextmanager
def measured_gpu_lock(clock: dict[str, float]) -> Iterator[None]:
    """Charge actual lock occupancy even when GPU admission or a forward fails.

    Parameters
    ----------
    clock : dict[str, float]
        Mutable stage accounting.

    Yields
    ------
    None
        Exclusive access through the project's shared lock.
    """
    with gpu_lock("cuda", blocking=False):
        start = time.perf_counter()
        try:
            yield
        finally:
            clock["gpu_cost"] = time.perf_counter() - start


def create_banks(destination: Path, noises: np.ndarray) -> dict[str, np.ndarray]:
    """Allocate fixed banks once and preserve each completed sample on disk.

    Parameters
    ----------
    destination : Path
        Fresh local study directory.
    noises : np.ndarray
        Frozen CPU input bank.

    Returns
    -------
    dict[str, np.ndarray]
        Noise and four write-through numerical memory maps.
    """
    if not np.array_equal(np.load(destination / "noise.npy", allow_pickle=False), noises):
        raise ValueError("Retained fixed noise differs from the profiled bank")
    banks = {"noise": noises}
    for name, shape, dtype in [
        ("raw", (128, 8, 5000), np.float32),
        ("variants", (128, 4, 12, 5000), np.float32),
        ("strict_support", (128, 2, 12, 5000), np.bool_),
        ("support", (128, 2, 12, 5000), np.bool_),
    ]:
        path = destination / f"{name}.npy"
        if path.exists():
            raise FileExistsError(path)
        banks[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return banks


def finish_ledger(
    destination: Path, result: dict[str, Any], receipt_name: str, clock: dict[str, float], parent: Path
) -> None:
    """Charge this stage once without relabeling historical reported costs as measured.

    Parameters
    ----------
    destination : Path
        Current study directory.
    result : dict[str, Any]
        Main success or failure receipt.
    receipt_name : str
        Saved receipt filename.
    clock : dict[str, float]
        Stage start times.
    parent : Path
        Available immediate predecessor.
    """
    digest = shared.file_sha256(destination / receipt_name)
    prior_path = parent / "prospective_ledger.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    terminal = quality.elapsed(clock)
    shared.write_new_json(
        destination / "prospective_ledger.json",
        {
            "scope": "new_current_machine_studies_only; old factorial unknown",
            "stage": "generated_quality_v1",
            "receipt": receipt_name,
            "receipt_sha256": digest,
            **terminal,
            "gpu_wall_seconds": result.get("gpu_lock_wall_seconds", 0.0),
            "prior_main_cpu_seconds": prior.get("cpu_seconds"),
            "prior_main_ledger_sha256": shared.file_sha256(parent / "prospective_ledger.json")
            if prior
            else None,
            "measurement_endpoint": "after main summary write and hash, before ledger serialization",
            "independent_audit_charged_separately": True,
            "reserved_independent_audit": {"cpu_seconds": 180, "wall_seconds": 180, "bytes": 16 * 1024**2},
        },
    )


def arm_timers(resources: dict[str, Any], old_handlers: dict[int, Any]) -> None:
    """Arm CPU and wall timers before scientific admission.

    Parameters
    ----------
    resources : dict[str, Any]
        Frozen resource bounds.
    old_handlers : dict[int, Any]
        Original handlers restored by the caller.
    """
    for timer, sig, limit in [
        (signal.ITIMER_REAL, signal.SIGALRM, resources["wall_seconds"]),
        (signal.ITIMER_PROF, signal.SIGPROF, resources["cpu_seconds"]),
    ]:
        if signal.getitimer(timer) != (0.0, 0.0):
            raise ValueError("An existing process timer prevents deadline admission")
        old_handlers[sig] = signal.signal(sig, timeout_handler)
        signal.setitimer(timer, limit - 2)


def admit_predecessor(
    parent: Path, destination: Path, counts: dict[str, int], attempts: dict[str, int]
) -> tuple[Any, ...]:
    """Verify every predecessor binding before four exact CPU replays.

    Parameters
    ----------
    parent, destination : Path
        Parent and fresh successor directories.
    counts, attempts : dict[str, int]
        Completed and attempted forward accounting.

    Returns
    -------
    tuple
        Loaded model, parent summary, verified input bindings and replay closure.
    """
    frozen = json.loads((ROOT / ADMISSION).read_text())["predecessor_sha256"]
    for name, digest in frozen.items():
        if shared.file_sha256(parent / name) != digest:
            raise ValueError("Frozen immediate predecessor identity mismatch")
    main = json.loads((parent / "summary.json").read_text())
    if not main["technical_generation_feasible"]:
        raise ValueError("Technical generator predecessor did not pass")
    inputs = {
        "summary.json": shared.file_sha256(parent / "summary.json"),
        "prospective_ledger.json": shared.file_sha256(parent / "prospective_ledger.json"),
    }
    inputs.update(frozen)
    for name, digest in main["input_sha256"].items():
        if shared.file_sha256(parent / name) != digest:
            raise ValueError("Predecessor public-asset binding changed")
        inputs[name] = digest
    model = shared.load_generator(
        parent / "pytorch_model.bin",
        shared.architecture_class(
            (parent / "mit_pulse2pulse.py").read_text(), (parent / "hf_modeling_deepfake.py").read_text()
        ),
    )
    for name, digest in main["output_sha256"].items():
        if shared.file_sha256(parent / name) != digest:
            raise ValueError("Predecessor saved output or source binding changed")
        inputs[name] = digest
    audit = json.loads((parent / "independent_audit_v1/summary.json").read_text())
    if not audit["all_passed"]:
        raise ValueError("Immediate predecessor independent audit failed")
    inputs["independent_audit_v1/summary.json"] = shared.file_sha256(
        parent / "independent_audit_v1/summary.json"
    )
    predecessor = predecessor_closure(parent, main, model, counts, attempts)
    shared.write_new_json(destination / "predecessor_closure.json", predecessor)
    return model, main, inputs, predecessor


def verify_bindings(destination: Path, parent: Path, sources: dict[str, str], inputs: dict[str, str]) -> None:
    """Recheck unchanged source copies and all consumed predecessor artifacts.

    Parameters
    ----------
    destination, parent : Path
        Current and predecessor output directories.
    sources, inputs : dict[str, str]
        Source and input bindings recorded before generation.
    """
    for name, digest in sources.items():
        if (
            shared.file_sha256(ROOT / name) != digest
            or shared.file_sha256(destination / "source_snapshot" / name) != digest
        ):
            raise ValueError("Committed scientific source changed during execution")
    for name, digest in inputs.items():
        if shared.file_sha256(parent / name) != digest:
            raise ValueError("Predecessor evidence changed during execution")
