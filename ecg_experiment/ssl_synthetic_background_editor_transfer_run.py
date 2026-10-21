"""Execute the admitted CUDA-only, no-training background/editor transfer study."""

import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ssl_synthetic_background_editor_transfer as core
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as gpu_runtime
from ecg_experiment import ssl_synthetic_ludb_localizer as predecessor
from ecg_experiment import ssl_synthetic_ludb_localizer_run as runtime
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment import ssl_synthetic_pretrained_generator as identity
from ecg_experiment import ssl_synthetic_randomized_localization_run as inference
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored

CONFIG = "configs/ssl_synthetic_background_editor_transfer_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-background-editor-transfer-v1.md"
SOURCES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *runtime.SOURCES,
            CONFIG,
            PROTOCOL,
            "ecg_experiment/ssl_synthetic_background_editor_transfer.py",
            "ecg_experiment/ssl_synthetic_background_editor_transfer_run.py",
            "scripts/experiments/ssl_synthetic_background_editor_transfer.py",
            "tests/test_ssl_synthetic_background_editor_transfer.py",
        )
    )
)
REAL_ROOT = "outputs/ssl_synthetic_localization/ludb_localizer_v1"
GENERATED_ROOT = "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"
REQUIRED_INPUTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *runtime.REQUIRED_INPUTS,
            *[
                f"{REAL_ROOT}/{name}"
                for name in [
                    "summary.json",
                    "prospective_ledger.json",
                    "inputs.npy",
                    "record_ids.json",
                    "input_metadata.json",
                    "model_rows.jsonl",
                    "bootstrap.npy",
                    "independent_audit_v4/summary.json",
                    "independent_audit_v4/prospective_ledger.json",
                ]
            ],
            *[
                f"{GENERATED_ROOT}/{name}"
                for name in ["fresh_raw.npy", "fresh_strict_support.npy", "fresh_cases.jsonl"]
            ],
        )
    )
)


def preflight(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify complete executable declarations without reading ECG arrays or performing inference.

    Parameters
    ----------
    name : str
        Committed repository-relative identity admission.

    Returns
    -------
    tuple
        Frozen configuration and complete identity admission.
    """
    config, admission = shared.read_json(from_stored(CONFIG)), shared.read_json(from_stored(name))
    if not set(SOURCES) <= set(admission["source_sha256"]) or not set(REQUIRED_INPUTS) <= set(
        admission["input_sha256"]
    ):
        raise ValueError("Transfer executable/input roster incomplete")
    if (
        admission["source_sha256"][CONFIG] != admission["frozen_config_sha256"]
        or admission["source_sha256"][PROTOCOL] != admission["frozen_protocol_sha256"]
        or admission["prior_ledger"]["path"] != config["prior_ledger"]
        or admission["prior_ledger"]["current_machine_stages"] != 34
        or admission["prior_ledger"]["current_machine_cumulative_cpu_seconds"] != 759.933143906
        or admission["prior_ledger"]["current_machine_cumulative_gpu_wall_seconds"] != 473.0495132012293
    ):
        raise ValueError("Transfer scientific or actual stage34 identity differs")
    if config["sources"] != list(core.SOURCES) or config["variants"] != list(core.VARIANTS):
        raise ValueError("Prospective tensor source/variant order differs")
    for path, digest in admission["source_sha256"].items():
        if sha256_file(from_stored(path)) != digest:
            raise ValueError("Frozen transfer source changed: " + path)
    return config, admission


def verify_inputs(admission: dict[str, Any], guard: Any) -> None:
    """Rehash only explicitly admitted saved inputs and original output-manifest anchors.

    Parameters
    ----------
    admission : dict
        Flat immutable input identities.
    guard : Any
        Live measured resource guard.

    Returns
    -------
    None
        Raises on altered or unanchored predecessor artifacts.
    """
    for path, digest in admission["input_sha256"].items():
        guard()
        if sha256_file(from_stored(path)) != digest:
            raise ValueError("Admitted transfer input changed: " + path)
    for root, expected, auditdir in [
        (
            REAL_ROOT,
            "e25e7be0b12774c485b56b5bff1664b5718d7efc25b18135eb67f45fd9ffd4fc",
            "independent_audit_v4",
        ),
        (
            GENERATED_ROOT,
            "8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111",
            "independent_audit_v1",
        ),
    ]:
        parent = shared.read_json(from_stored(root) / "summary.json")
        if admission["input_sha256"][root + "/summary.json"] != expected:
            raise ValueError("Original verified real-input parent identity differs")
        audit = shared.read_json(from_stored(root) / auditdir / "summary.json")
        if not audit["all_passed"] or audit["mismatch_count"] != 0:
            raise ValueError("A required original independent audit did not pass")
        for path, digest in admission["input_sha256"].items():
            if not path.startswith(root + "/"):
                continue
            relative = path[len(root) + 1 :]
            if relative in parent["output_sha256"] and parent["output_sha256"][relative] != digest:
                raise ValueError("Input differs from its original parent output manifest")
            if relative.endswith(".npy") and relative not in parent["output_sha256"]:
                raise ValueError("Numerical input lacks its original parent manifest anchor")


def close_predecessors(destination: Path, config: dict[str, Any], guard: Any) -> None:
    """Reproduce all saved synthetic endpoints and the full real-landmark predecessor exactly.

    Parameters
    ----------
    destination : Path
        New exclusive output directory.
    config : dict
        Frozen predecessor source roots.
    guard : Any
        Live stage resource guard.

    Returns
    -------
    None
        Writes exact closure receipts without new inference.
    """
    oldconfig = shared.read_json(from_stored(config["localizer_config"]))
    runtime.close_saved_studies(destination, oldconfig, guard)
    rows = runtime.read_rows(from_stored(config["localizer_root"]) / "model_rows.jsonl")
    statistics, draws = predecessor.aggregate(rows, oldconfig)
    parent = shared.read_json(from_stored(config["localizer_root"]) / "summary.json")
    if statistics != parent["statistics"] or not np.array_equal(
        draws, np.load(from_stored(config["localizer_root"]) / "bootstrap.npy", allow_pickle=False)
    ):
        raise ValueError("Exact real-landmark statistics/all2000draws closure differs")
    shared.write_new(
        destination / "localizer_predecessor_closure.json",
        {"all_exact": True, "statistics": statistics, "rows": len(rows)},
    )


def new_array(path: Path, shape: tuple[int, ...], dtype: Any) -> np.memmap:
    """Exclusively allocate a bounded standard NumPy memmap.

    Parameters
    ----------
    path : Path
        New output file, never overwritten.
    shape : tuple
        Prospectively bounded complete tensor geometry.
    dtype : Any
        Float32 maps/inputs or Boolean final-difference masks.

    Returns
    -------
    np.memmap
        Writable standard NPY mapping.
    """
    with path.open("xb"):
        pass
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def prepare_inputs(destination: Path, config: dict[str, Any], guard: Any) -> None:
    """Construct matched inputs without raw records or a second real-input standardization.

    Parameters
    ----------
    destination : Path
        Exclusive study output directory.
    config : dict
        Frozen source roots and cohort contract.
    guard : Any
        Live guard during construction and flush.

    Returns
    -------
    None
        Saves all observations, exact masks, identities and measured edit geometry.
    """
    root = from_stored(config["localizer_root"])
    generated = from_stored(config["generated_root"])
    identifiers = json.loads((root / "record_ids.json").read_text())
    expected = [n for n in range(1, 201) if n % 5]
    if identifiers != expected:
        raise ValueError("Original real input ID order differs")
    real = np.load(root / "inputs.npy", mmap_mode="r", allow_pickle=False)
    raw = np.load(generated / "fresh_raw.npy", mmap_mode="r", allow_pickle=False)
    old = np.load(generated / "fresh_observed.npy", mmap_mode="r", allow_pickle=False)
    oldsupport = np.load(generated / "fresh_support.npy", mmap_mode="r", allow_pickle=False)
    oldstrict = np.load(generated / "fresh_strict_support.npy", mmap_mode="r", allow_pickle=False)
    inputs = new_array(destination / "inputs.npy", (2, 64, 6, 12, 5000), np.float32)
    strict = new_array(destination / "strict_support.npy", (2, 64, 4, 12, 5000), bool)
    support = new_array(destination / "support.npy", (2, 64, 4, 12, 5000), bool)
    metadata = []
    for sourcepos, source in enumerate(core.SOURCES):
        for slot in range(64):
            guard()
            if source == "generated":
                native, transform = quality.standardize(raw[slot])
            else:
                native, transform = np.array(real[slot, list(core.NATIVE)], dtype=np.float32), None
            observed, exact, target, row = core.make_case(native, 2000 + slot)
            if source == "generated" and not (
                np.array_equal(observed[:4], old[slot])
                and np.array_equal(target[:2], oldsupport[slot, 2:])
                and np.array_equal(exact[:2], oldstrict[slot])
            ):
                raise ValueError("Original generated variants or final targets differ")
            if source == "real" and not np.array_equal(observed[0], real[slot]):
                raise ValueError("Real canonical sham changed or was normalized twice")
            inputs[sourcepos, slot], strict[sourcepos, slot], support[sourcepos, slot] = (
                observed,
                exact,
                target,
            )
            metadata.append(
                {
                    **row,
                    "source": source,
                    "slot": slot,
                    "record_id": identifiers[slot] if source == "real" else None,
                    "standardization": transform,
                    "canonical_sham_exact": True,
                    "original_generated_four_exact": source == "generated",
                }
            )
    inputs.flush()
    strict.flush()
    support.flush()
    quality.write_rows(destination / "geometry.jsonl", metadata)
    shared.write_new(
        destination / "record_ids.json", {"real": identifiers[:64], "generated": list(range(2000, 2064))}
    )
    del inputs, strict, support, real, raw, old, oldsupport, oldstrict


def score_maps(
    destination: Path, map_directory: Path, slots: list[int], config: dict[str, Any], guard: Any
) -> list[dict[str, Any]]:
    """Score saved float32 maps outside the shared GPU lock, including fixed donor masks.

    Parameters
    ----------
    destination : Path
        Complete input/target bank location.
    map_directory : Path
        Either full or profile model probability directory.
    slots : list
        Actual geometry slots retained in those arrays.
    config : dict
        Frozen study settings.
    guard : Any
        Live CPU/wall/RSS/storage guard.

    Returns
    -------
    list
        Every fixed per-model/source/slot metric row.
    """
    inputs = np.load(destination / "inputs.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(destination / "support.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    for fit in core.FITS:
        for arm in core.ARMS:
            maps = np.load(map_directory / f"fit_{fit}_{arm}.npy", mmap_mode="r", allow_pickle=False)
            for sourcepos, source in enumerate(core.SOURCES):
                for position, slot in enumerate(slots):
                    guard()
                    rows.append(
                        core.score_record(
                            maps[sourcepos, position],
                            inputs[sourcepos, slot],
                            support[sourcepos, slot],
                            source,
                            slot,
                            fit,
                            arm,
                            support[sourcepos, (slot + 1) % 64],
                        )
                    )
            del maps
    del inputs, support
    return rows


def infer_bank(
    destination: Path, inputs: np.ndarray, models: dict[str, Any], slots: list[int], guard: Any
) -> dict[str, str]:
    """Persist all twelve full-output model banks using identical batch-eight write/flush/hash paths.

    Parameters
    ----------
    destination : Path
        New probability directory.
    inputs : np.ndarray
        All admitted canonical observed variants.
    models : dict
        Every fixed strictly loaded CUDA state.
    slots : list
        Ordered source slots, either first8 or all64.
    guard : Any
        Live actual-path resource guard.

    Returns
    -------
    dict
        Exact saved file identities for each model bank.
    """
    destination.mkdir()
    hashes = {}
    for name, model in models.items():
        maps = new_array(destination / (name + ".npy"), (2, len(slots), 6, 12, 5000), np.float32)
        for source in range(2):
            observed = np.asarray(inputs[source, slots]).reshape(-1, 12, 5000)
            flat = maps[source].reshape(-1, 12, 5000)
            for start in range(0, len(observed), 8):
                guard()
                predicted = inference.infer(model, observed[start : start + 8], "cuda")
                runtime.require_maps(predicted, 8)
                flat[start : start + 8] = predicted
                maps.flush()
            del flat, observed
        hashes[name] = sha256_file(destination / (name + ".npy"))
        del maps
    return hashes


def figures(destination: Path, statistics: dict[str, Any] | None, profile: bool = False) -> None:
    """Show fixed examples or complete all-arm cell means, without selecting favorable examples.

    Parameters
    ----------
    destination : Path
        Study root containing saved maps and inputs.
    statistics : dict or None
        Complete output-derived statistics, or None for the fixed profile figure.
    profile : bool
        Select the fixed profile path rather than the aggregate result.

    Returns
    -------
    None
        Saves bounded static figures locally.
    """
    import matplotlib.pyplot as plt

    if profile:
        inputs = np.load(destination / "inputs.npy", mmap_mode="r", allow_pickle=False)
        maps = np.load(destination / "profile/probabilities/fit_1_B.npy", mmap_mode="r", allow_pickle=False)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for source in range(2):
            for column, variant in enumerate([2, 5]):
                axes[source, column].plot(inputs[source, 0, variant, 1], color="black", linewidth=0.5)
                twin = axes[source, column].twinx()
                twin.plot(maps[source, 0, variant, 1], color="teal", linewidth=0.6)
                twin.set_ylim(0, 1)
                axes[source, column].set(
                    title=f"{core.SOURCES[source]} slot0 {core.VARIANTS[variant]} fit1 B", xlabel="Sample"
                )
        fig.savefig(destination / "profile/fixed_examples.png", dpi=100)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 4, figsize=(21, 5), constrained_layout=True)
    for axis, metric in zip(axes, ["edited_ap", "response_ap", "amplitude_gain", "donor_gain"], strict=True):
        for source in core.SOURCES:
            for family in ["hann", "biphasic"]:
                axis.plot(
                    core.ARMS,
                    [
                        statistics["endpoints"][f"{source}__{arm}__{family}__{metric}"]["mean"]
                        for arm in core.ARMS
                    ],
                    "o-",
                    label=f"{source}/{family}",
                )
        axis.axhline(0, color="grey", linewidth=0.5)
        axis.set(ylabel=metric, xlabel="All fixed arms")
    axes[0].legend(fontsize=8)
    fig.suptitle("Introduced numerical changes; no disease-region ground truth")
    fig.savefig(destination / "all_source_family_arm_results.png", dpi=120)
    plt.close(fig)


def resource_guard(config: dict[str, Any], clock: dict[str, float]) -> None:
    """Keep whole-stage, cumulative lock, memory and conservative storage caps live.

    Parameters
    ----------
    config : dict
        Frozen hard resource limits.
    clock : dict
        Whole-stage clocks plus completed and currently active GPU-lock intervals.

    Returns
    -------
    None
        Raises before consuming the terminal receipt reserve.
    """
    limits = config["resources"]
    measured = runtime.elapsed(clock)
    if (
        measured["cpu_seconds"] + 2 >= limits["cpu_seconds"]
        or measured["wall_seconds"] + 2 >= limits["wall_seconds"]
    ):
        raise TimeoutError("Transfer CPU/wall reserve reached")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
        raise MemoryError("Transfer RSS limit")
    if clock.get("output_bound", 0) >= limits["output_bytes"]:
        raise ValueError("Transfer conservative storage bound exceeds cap")
    active = time.perf_counter() - clock["active_gpu_start"] if "active_gpu_start" in clock else 0
    if clock.get("gpu_cost", 0) + active + 2 >= limits["gpu_lock_seconds"]:
        raise TimeoutError("Transfer cumulative GPU-lock reserve reached")
    if active and torch.cuda.max_memory_allocated(0) >= limits["cuda_allocated_bytes"]:
        raise MemoryError("Transfer CUDA memory limit")


def cold_models(destination: Path, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Load all fixed states and retain their original CUDA replay evidence under the lock.

    Parameters
    ----------
    destination : Path
        New study output directory.
    config : dict
        Frozen checkpoint and replay specification.
    guard : Any
        Live resource guard called within the active shared GPU lock.

    Returns
    -------
    dict
        All twelve unchanged CUDA-resident models.
    """
    models = {}
    torch.cuda.reset_peak_memory_stats(0)
    check = torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")
    if check[0, 0].item() != 16:
        raise ValueError("Actual small CUDA operation failed")
    del check
    replay_inputs = np.load(
        from_stored(config["model_root"]) / "fresh_observed.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    for fit in core.FITS:
        for arm in core.ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            model, digest = runtime.load_model(config, fit, arm)
            models[name] = model.to("cuda")
            predicted = inference.infer(model, replay_inputs[0], "cuda")
            reference = np.load(
                from_stored(config["model_root"]) / f"fit_{fit}" / arm / "fresh_probabilities.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            replay = runtime.retain_replay(
                destination / "profile",
                name,
                predicted,
                reference[0],
                digest,
                config["profile"]["parent_replay_atol"],
            )
            if not replay["all_passed"]:
                raise ValueError("Frozen parent CUDA map replay failed")
            del reference
    del replay_inputs
    return models


def gpu_inference(
    destination: Path,
    config: dict[str, Any],
    clock: dict[str, float],
    guard: Any,
    profile: bool,
    models: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure a nonblocking shared-lock phase and retain every model's actual-path evidence.

    Parameters
    ----------
    destination : Path
        Complete study input/output directory.
    config : dict
        Frozen model, profile and resource identities.
    clock : dict
        Whole-stage and additive GPU-lock accounting.
    guard : Any
        Live stage resource guard.
    profile : bool
        Whether to run cold loading, old replay and the first eight slots.
    models : dict or None
        CPU-resident immutable models after the profile, or None before cold loading.

    Returns
    -------
    tuple
        CPU-resident unchanged models and phase resource/map/state evidence.
    """
    from ecg_experiment.gpu import gpu_lock

    models = {} if models is None else models
    started = runtime.new_clock()
    phase = {}
    states = {}
    with gpu_lock("cuda", blocking=False):
        clock["active_gpu_start"] = time.perf_counter()
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("Actual CUDA access unavailable; no CPU inference fallback")
            free, total = torch.cuda.mem_get_info(0)
            if free < config["resources"]["minimum_free_cuda_bytes"]:
                raise MemoryError("Insufficient free CUDA memory")
            if profile:
                models = cold_models(destination, config, guard)
            else:
                for model in models.values():
                    model.to("cuda")
            inputs = np.load(destination / "inputs.npy", mmap_mode="r", allow_pickle=False)
            repeat_clock = runtime.new_clock()
            slots = config["profile"]["slots"] if profile else list(range(64))
            directory = destination / "profile/probabilities" if profile else destination / "probabilities"
            hashes = infer_bank(directory, inputs, models, slots, guard)
            phase["recurring_inference"] = runtime.elapsed(repeat_clock)
            if not profile:
                for name in models:
                    guard()
                    first = np.load(
                        destination / "profile/probabilities" / (name + ".npy"),
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    full = np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                    if not np.array_equal(first, full[:, :8]):
                        raise ValueError("Profile/full first-eight batch paths differ")
                    del first, full
            for name, model in models.items():
                states[name] = identity.tensor_state_sha256(model)
                fit, arm = int(name.split("_")[1]), name.split("_")[2]
                expected = shared.read_json(from_stored(config["model_root"]) / "summary.json")[
                    "final_state_sha256"
                ][str(fit)][arm]
                if states[name] != expected:
                    raise ValueError("Frozen model state changed during inference")
                model.cpu()
            torch.cuda.synchronize()
            phase.update(
                map_sha256=hashes,
                tensor_state_sha256=states,
                free_cuda_bytes_at_admission=free,
                total_cuda_bytes=total,
                device=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                torch_cuda=torch.version.cuda,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            )
            del inputs
        finally:
            occupied = time.perf_counter() - clock.pop("active_gpu_start")
            clock["gpu_cost"] = clock.get("gpu_cost", 0) + occupied
            phase["gpu_lock_seconds"] = occupied
            phase["actual_phase"] = runtime.elapsed(started)
            shared.write_new(
                destination / ("profile/gpu_phase.json" if profile else "full_gpu_phase.json"), phase
            )
    return models, phase


def profile_gate(
    destination: Path, config: dict[str, Any], clock: dict[str, float], phase: dict[str, Any], guard: Any
) -> None:
    """Include identical CPU scoring and figure persistence before forecasting the full panel.

    Parameters
    ----------
    destination : Path
        New study root with completed actual CUDA profile maps.
    config : dict
        Prospectively frozen forecast multipliers, reserves and caps.
    clock : dict
        Actual whole-stage cost including cold load, replay and all predecessor closures.
    phase : dict
        Actual-profile recurring inference, flush/hash and lock measurements.
    guard : Any
        Live stage resource guard.

    Returns
    -------
    None
        Retains profile rows/forecast and raises before full map allocation if a gate fails.
    """
    started = runtime.new_clock()
    rows = score_maps(
        destination, destination / "profile/probabilities", config["profile"]["slots"], config, guard
    )
    quality.write_rows(destination / "profile/record_rows.jsonl", rows)
    scoring = runtime.elapsed(started)
    figures(destination, None, True)
    actual = runtime.elapsed(clock)
    factor = config["profile"]["factor"] * config["profile"]["recurrence_multiplier"]
    infer = phase["recurring_inference"]
    forecast = {
        key: actual[key] + factor * (infer[key] + scoring[key]) + config["profile"][reserve]
        for key, reserve in [("cpu_seconds", "cpu_reserve_seconds"), ("wall_seconds", "wall_reserve_seconds")]
    }
    forecast["gpu_lock_seconds"] = (
        clock["gpu_cost"] + factor * infer["wall_seconds"] + config["profile"]["gpu_reserve_seconds"]
    )
    forecast["output_bytes"] = clock["output_bound"]
    gates = {key: forecast[key] < config["resources"][key] for key in forecast}
    gates["cuda_allocated_bytes"] = (
        phase["peak_allocated_bytes"] < config["resources"]["cuda_allocated_bytes"]
    )
    shared.write_new(
        destination / "profile/summary.json",
        {
            "actual_at_forecast": actual,
            "recurring_inference": infer,
            "recurring_scoring": scoring,
            "forecast": forecast,
            "forecast_gates": gates,
            "profile_rows": len(rows),
            "full_batch8_forwards": 1152,
            "profile_batch8_forwards": 144,
            "parent_batch4_replays": 12,
        },
    )
    if not all(gates.values()):
        raise ValueError("Full actual-path profile forecast failed before full inference")


def run(root: Path, name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Execute the fixed study with immutable inputs, actual CUDA gates and retained failures.

    Parameters
    ----------
    root : Path
        Source checkout whose outputs alias the baseline-owned local output tree.
    name : str
        Committed executable identity admission.
    clock : dict
        CPU/wall clocks captured before scientific imports.

    Returns
    -------
    dict
        Provisional output-derived result; independent verification remains required.
    """
    config, admission = preflight(name)
    destination = shared.confined(root, config["output_root"])
    destination.mkdir(parents=True, exist_ok=False)
    result = {"study_id": config["study_id"], "status": "started"}
    timers = {}
    verified = False
    limits = config["resources"]
    compatible = {
        **config,
        "resources": {
            **limits,
            "peak_rss_bytes": limits["rss_bytes"],
            "gpu_lock_wall_seconds": limits["gpu_lock_seconds"],
            "maximum_gpu_allocated_bytes": limits["cuda_allocated_bytes"],
        },
    }

    def guard() -> None:
        """Enforce the frozen live resource bounds."""
        resource_guard(config, clock)

    def interrupted(signum: int, frame: Any) -> None:
        """Interrupt before the terminal receipt reserve is exhausted."""
        raise TimeoutError(f"Transfer process timer:{signum}")

    try:
        shared.setup_timers(limits, clock, interrupted, timers)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("Prospective CUDA0 visibility required")
        gpu_runtime.set_runtime()
        result["runtime"] = {
            "environment": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }
        sources = runtime.bridge_runtime.peak_runtime.snapshot(destination, admission, name)
        result["source_sha256"] = sources
        shared.verify_prior(root, admission)
        verified = True
        verify_inputs(admission, guard)
        close_predecessors(destination, config, guard)
        if shutil.disk_usage(destination).free < limits["minimum_free_output_bytes"]:
            raise OSError("Insufficient free output disk")
        forecast = (
            2 * 64 * 6 * 12 * 5000 * 4 * 12
            + 2 * 8 * 6 * 12 * 5000 * 4 * 12
            + 2 * 64 * 6 * 12 * 5000 * 4
            + 2 * 2 * 64 * 4 * 12 * 5000
            + 12 * 4 * 12 * 5000 * 4
            + 128 * 1024**2
        )
        clock["output_bound"] = shared.directory_bytes(destination) + forecast
        guard()
        prepare_inputs(destination, config, guard)
        (destination / "profile").mkdir()
        models, phase = gpu_inference(destination, config, clock, guard, True)
        profile_gate(destination, config, clock, phase, guard)
        models, fullphase = gpu_inference(destination, config, clock, guard, False, models)
        shared.write_new(
            destination / "model_states.json",
            {
                "profile": phase["tensor_state_sha256"],
                "full": fullphase["tensor_state_sha256"],
                "all_unchanged": phase["tensor_state_sha256"] == fullphase["tensor_state_sha256"],
            },
        )
        del models
        rows = score_maps(destination, destination / "probabilities", list(range(64)), config, guard)
        quality.write_rows(destination / "record_rows.jsonl", rows)
        statistics, draws = core.summarize(rows, config)
        runtime.save_array(destination / "bootstrap.npy", draws)
        figures(destination, statistics)
        verify_inputs(admission, guard)
        for path, digest in sources.items():
            guard()
            if sha256_file(from_stored(path)) != digest:
                raise ValueError("Executable source changed during transfer study")
        result.update(
            status="PASS",
            statistics=statistics,
            technical_passed_before_independent_audit=True,
            independent_audit_required=True,
            clinical_claim=False,
            new_training_updates=0,
            new_generator_forwards=0,
            raw_record_reads=0,
            profile=phase,
            full_gpu=fullphase,
            operation_counts={
                "parent_batch4_replays": 12,
                "profile_batch8_forwards": 144,
                "full_batch8_forwards": 1152,
                "metric_rows": len(rows),
            },
            output_sha256={
                p.relative_to(destination).as_posix(): sha256_file(p)
                for p in destination.rglob("*")
                if p.is_file() and not p.is_symlink()
            },
        )
    except Exception as error:
        result.update(
            status="FAIL", error_type=type(error).__name__, error=str(error), scientific_result_valid=False
        )
    finally:
        runtime.finish(destination, compatible, admission, result, clock, verified, timers)
    return result
