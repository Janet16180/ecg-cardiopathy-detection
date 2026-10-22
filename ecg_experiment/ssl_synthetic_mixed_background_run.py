"""Execute the prospectively admitted matched generated/mixed B-localizer study."""

import gc
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ssl_synthetic_background_editor_transfer as transfer
from ecg_experiment import ssl_synthetic_background_editor_transfer_run as execution
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as gpu_runtime
from ecg_experiment import ssl_synthetic_ludb_localizer_run as clocks
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment import ssl_synthetic_mixed_background as core
from ecg_experiment import ssl_synthetic_mixed_background_cohort as cohort
from ecg_experiment import ssl_synthetic_mixed_background_data as data
from ecg_experiment import ssl_synthetic_mixed_background_training as training
from ecg_experiment import ssl_synthetic_pretrained_generator as identities
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import from_stored

CONFIG = "configs/ssl_synthetic_mixed_background_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-mixed-background-v1.md"
SOURCES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *execution.SOURCES,
            *cohort.SOURCES,
            CONFIG,
            PROTOCOL,
            "ecg_experiment/ssl_synthetic_mixed_background.py",
            "ecg_experiment/ssl_synthetic_mixed_background_data.py",
            "ecg_experiment/ssl_synthetic_mixed_background_training.py",
            "ecg_experiment/ssl_synthetic_mixed_background_run.py",
            "scripts/experiments/ssl_synthetic_mixed_background.py",
            "tests/test_ssl_synthetic_mixed_background.py",
        )
    )
)
GEN = "outputs/ssl_synthetic_localization/randomized_localization_v1"
MODEL = "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"
EVAL = "outputs/ssl_synthetic_localization/background_editor_transfer_v1"
FIG = "outputs/ssl_synthetic_localization/background_editor_transfer_figures_v1"
COHORT = "outputs/ssl_synthetic_localization/mixed_background_cohort_v1"
REQUIRED_INPUTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *[
                f"{GEN}/{n}"
                for n in [
                    "observed.npy",
                    "support.npy",
                    "summary.json",
                    "independent_audit_v1/summary.json",
                    "independent_audit_v1/prospective_ledger.json",
                ]
            ],
            *[
                f"{MODEL}/{n}"
                for n in [
                    "summary.json",
                    "prospective_ledger.json",
                    "independent_audit_v1/summary.json",
                    "independent_audit_v1/prospective_ledger.json",
                ]
            ],
            *[
                f"{MODEL}/fit_{fit}/{n}"
                for fit in transfer.FITS
                for n in [
                    "initial_state.pt",
                    "training_schedule.npy",
                    "B/final_state.pt",
                    "B/optimizer.pt",
                    "B/losses.npy",
                    "B/loss_components.npy",
                ]
            ],
            *[
                f"{EVAL}/{n}"
                for n in [
                    "summary.json",
                    "prospective_ledger.json",
                    "independent_audit_v1/summary.json",
                    "independent_audit_v1/prospective_ledger.json",
                    "inputs.npy",
                    "support.npy",
                    "strict_support.npy",
                    "geometry.jsonl",
                    "record_rows.jsonl",
                    "bootstrap.npy",
                    *[f"probabilities/fit_{f}_B.npy" for f in transfer.FITS],
                ]
            ],
            f"{FIG}/summary.json",
            f"{FIG}/prospective_ledger.json",
            f"{COHORT}/selection.json",
            f"{COHORT}/summary.json",
            "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
            "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt",
        )
    )
)


def preflight(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the complete frozen executable contract before arrays, outputs or CUDA operations.

    Parameters
    ----------
    name : str
        Committed repository-relative input/source identity admission.

    Returns
    -------
    tuple
        Frozen scientific configuration and complete explicit identity admission.
    """
    config, admission = shared.read_json(from_stored(CONFIG)), shared.read_json(from_stored(name))
    if not set(SOURCES) <= set(admission["source_sha256"]) or not set(REQUIRED_INPUTS) <= set(
        admission["input_sha256"]
    ):
        raise ValueError("Mixed-background source/input identity roster incomplete")
    if (
        admission["source_sha256"][CONFIG] != admission["frozen_config_sha256"]
        or admission["source_sha256"][PROTOCOL] != admission["frozen_protocol_sha256"]
        or admission["prior_ledger"]["path"] != config["prior_ledger"]
        or admission["prior_ledger"]["sha256"]
        != "73758dc5c2f8b20543a63549d61fc6ce669286cd988c2dd135c6453e1a024d4f"
        or admission["prior_ledger"]["current_machine_stages"] != 37
    ):
        raise ValueError("Frozen protocol/config or actual37 accounting base differs")
    if (
        config["fits"] != [1, 2, 3]
        or config["training_arms"] != list(core.TRAINING_ARMS)
        or config["objective"] != "B"
        or config["updates_per_arm_fit"] != 400
        or config["positive_weight"] != 20
        or config["batch_size"] != 8
        or config["bootstrap"]["draws"] != 2000
        or config["bootstrap"]["seed"] != 2026106701
        or config["bootstrap"]["fit_resampling"] is not False
        or config["primary"]["gates"] != 14
        or config["initialization_seeds"] != [2026103801, 2026103802, 2026103803]
        or config["schedule_seeds"] != [2026103901, 2026103902, 2026103903]
    ):
        raise ValueError("Matched fixed-fit/update/seed/decision contract differs")
    for path, digest in admission["source_sha256"].items():
        if sha256_file(from_stored(path)) != digest:
            raise ValueError("Mixed-background frozen source changed: " + path)
    return config, admission


def verify_inputs(admission: dict[str, Any], guard: Any) -> None:
    """Bind every saved input to both its current admission and immutable original output manifest.

    Parameters
    ----------
    admission : dict
        Explicit complete source/input identities and actual cumulative accounting base.
    guard : Any
        Live measured stage resource guard.

    Returns
    -------
    None
        Raises on any altered original numerical input or failed required audit.
    """
    for path, digest in admission["input_sha256"].items():
        guard()
        if sha256_file(from_stored(path)) != digest:
            raise ValueError("Mixed study input changed: " + path)
    verify_parent_manifests(admission)


def verify_parent_manifests(admission: dict[str, Any]) -> None:
    """Require every admitted numerical parent input to retain its original provenance.

    Parameters
    ----------
    admission : dict
        Flat declared byte identities anchored to the original audited parent summaries.

    Returns
    -------
    None
        Raises on an original output-manifest or independent-audit mismatch.
    """
    anchors = {
        GEN: "fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f",
        MODEL: "8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111",
        EVAL: "c3ea28105ef99659b0f8854fd32430169ecb48f456b02b8b709fec306da93ba7",
        FIG: "c4d2b4cd77bbdabbebfd8b7dfea261a1bb5cd595abb6aefa4af3509006744923",
    }
    for root, digest in anchors.items():
        if admission["input_sha256"][root + "/summary.json"] != digest:
            raise ValueError("Original parent summary anchor differs")
        parent = shared.read_json(from_stored(root) / "summary.json")
        for path, expected in admission["input_sha256"].items():
            if (
                path.startswith(root + "/")
                and path[len(root) + 1 :] in parent.get("output_sha256", {})
                and parent["output_sha256"][path[len(root) + 1 :]] != expected
            ):
                raise ValueError("Original numerical output manifest differs")
        if root != FIG:
            audit = shared.read_json(from_stored(root) / "independent_audit_v1/summary.json")
            if not audit["all_passed"] or audit["mismatch_count"] != 0:
                raise ValueError("Required original independent audit failed")
    parent = shared.read_json(from_stored(MODEL) / "summary.json")
    for filename in ["observed.npy", "support.npy"]:
        if parent["input_sha256"]["learning"][filename] != admission["input_sha256"][GEN + "/" + filename]:
            raise ValueError("Generated TRAIN bank differs from original B fitting provenance")


def score_models(
    config: dict[str, Any], directory: Path, suffix: str, arm: str, slots: list[int], guard: Any
) -> list[dict[str, Any]]:
    """Reuse complete frozen transfer equations with the original nonself donor masks.

    Parameters
    ----------
    config : dict
        Fixed saved evaluation root.
    directory : Path
        Parent B or new mixed full/profile probability directory.
    suffix : str
        Fixed B or mixed filename identity.
    arm : str
        Generated or mixed training identity, separately from objective B.
    slots : list
        Complete64 evaluation slots or the first8 resource-profile slots.
    guard : Any
        Live measured stage resource guard.

    Returns
    -------
    list
        Every fixed fitted-state/evaluation-source/slot metric row.
    """
    root = from_stored(config["evaluation_root"])
    inputs = np.load(root / "inputs.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(root / "support.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    for fit in transfer.FITS:
        maps = np.load(directory / f"fit_{fit}_{suffix}.npy", mmap_mode="r", allow_pickle=False)
        for sourcepos, source in enumerate(transfer.SOURCES):
            for position, slot in enumerate(slots):
                guard()
                row = transfer.score_record(
                    maps[sourcepos, position],
                    inputs[sourcepos, slot],
                    support[sourcepos, slot],
                    source,
                    slot,
                    fit,
                    "B",
                    support[sourcepos, (slot + 1) % 64],
                )
                rows.append({**row, "training_arm": arm})
        del maps
    return rows


def close_predecessor(destination: Path, config: dict[str, Any], guard: Any) -> list[dict[str, Any]]:
    """Close all parent statistics/draws and independently recompute every referenced B map endpoint.

    Parameters
    ----------
    destination : Path
        Exclusive new study output root.
    config : dict
        Bound main35 root and unchanged transfer scientific settings.
    guard : Any
        Live measured CPU/wall/RSS guard.

    Returns
    -------
    list
        All384 exact generated-only B reference rows, with the new training-arm label.
    """
    root = from_stored(config["evaluation_root"])
    oldrows = clocks.read_rows(root / "record_rows.jsonl")
    statistics, draws = transfer.summarize(
        oldrows, shared.read_json(from_stored(config["evaluation_config"]))
    )
    if statistics != shared.read_json(root / "summary.json")["statistics"] or not np.array_equal(
        draws, np.load(root / "bootstrap.npy", allow_pickle=False)
    ):
        raise ValueError("Complete prior transfer statistics/six-gate/all2000draw closure differs")
    rows = score_models(config, root / "probabilities", "B", "generated", list(range(64)), guard)
    expected = {(r["source"], r["slot"], r["fit_id"]): r for r in oldrows if r["arm"] == "B"}
    for row in rows:
        original = {key: value for key, value in row.items() if key != "training_arm"}
        if original != expected[row["source"], row["slot"], row["fit_id"]]:
            raise ValueError("Complete parent B map-derived endpoint closure differs")
    shared.write_new(
        destination / "predecessor_closure.json",
        {"all_exact": True, "B_map_rows": 384, "statistics": statistics, "all2000draws_exact": True},
    )
    return rows


@contextmanager
def measured_gpu(destination: Path, name: str, clock: dict[str, float]) -> Iterator[dict[str, Any]]:
    """Charge every nonblocking shared-lock interval through its release, including failed phases.

    Parameters
    ----------
    destination : Path
        Existing study output root.
    name : str
        Fixed phase identity.
    clock : dict
        Additive whole-stage CPU/wall/GPU-lock accounting.

    Yields
    ------
    dict
        Phase metadata completed by the caller and preserved on failure.
    """
    phase = {}
    started = clocks.new_clock()
    try:
        with gpu_lock("cuda", blocking=False):
            clock["active_gpu_start"] = time.perf_counter()
            if not torch.cuda.is_available():
                raise RuntimeError("Actual CUDA required; no CPU fallback")
            yield phase
            torch.cuda.synchronize(0)
    finally:
        occupied = time.perf_counter() - clock.pop("active_gpu_start") if "active_gpu_start" in clock else 0.0
        clock["gpu_cost"] = clock.get("gpu_cost", 0) + occupied
        phase.update(**clocks.elapsed(started), gpu_lock_seconds=occupied)
        shared.write_new(destination / (name + "_gpu_phase.json"), phase)


def cuda_admission(config: dict[str, Any], small_operation: bool) -> dict[str, Any]:
    """Check actual free device memory inside the shared lock and perform the bounded initial GPU operation.

    Parameters
    ----------
    config : dict
        Frozen actual CUDA memory limits.
    small_operation : bool
        Run the actual16x16 operation only for initial cold qualification.

    Returns
    -------
    dict
        Actual hardware/runtime memory identity at phase admission.
    """
    free, total = torch.cuda.mem_get_info(0)
    if free < config["resources"]["minimum_free_cuda_bytes"]:
        raise MemoryError("Insufficient actual free CUDA memory")
    if small_operation:
        torch.cuda.reset_peak_memory_stats(0)
        matrix = torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")
        if matrix[0, 0].item() != 16:
            raise ValueError("Actual small CUDA operation failed")
        del matrix
    return {
        "name": torch.cuda.get_device_name(0),
        "free_bytes": free,
        "total_bytes": total,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def profile_gpu(
    destination: Path,
    config: dict[str, Any],
    plans: list[dict],
    banks: dict,
    counts: dict,
    clock: dict,
    guard: Any,
) -> dict:
    """Measure both actual training banks and the frozen-B inference/flush/hash path before full fitting.

    Parameters
    ----------
    destination : Path
        New study output root.
    config : dict
        Frozen actual-path profile and model declarations.
    plans : list
        Three untouched original initial states/schedules.
    banks : dict
        Prepared generated/mixed training inputs and fixed20 weights.
    counts : dict
        Actual completed/attempted operation counters.
    clock : dict
        Additive whole-stage resource clocks.
    guard : Any
        Live stage/GPU resource guard.

    Returns
    -------
    dict
        Every training timing, frozen B replay and inference-persistence phase measurement.
    """
    profile = destination / "profile"
    profile.mkdir()
    with measured_gpu(destination, "resource_profile", clock) as phase:
        phase["cuda"] = cuda_admission(config, True)
        phase["training"] = {
            arm: training.profile_one(plans[0], arm, banks[arm], destination, counts, guard)
            for arm in core.TRAINING_ARMS
        }
        models = {}
        for fit in transfer.FITS:
            model, _ = clocks.load_model(config, fit, "B")
            models[f"fit_{fit}_B"] = model.to("cuda")
        inputs = np.load(
            from_stored(config["evaluation_root"]) / "inputs.npy", mmap_mode="r", allow_pickle=False
        )
        started = clocks.new_clock()
        phase["profile_map_sha256"] = execution.infer_bank(
            profile / "frozen_B_probabilities", inputs, models, config["profile"]["inference_slots"], guard
        )
        phase["recurring_inference"] = clocks.elapsed(started)
        counts["frozen_B_profile_batches"] = 36
        proofs = {}
        for name, model in models.items():
            actual = np.load(
                profile / "frozen_B_probabilities" / (name + ".npy"), mmap_mode="r", allow_pickle=False
            )
            reference = np.load(
                from_stored(config["evaluation_root"]) / "probabilities" / (name + ".npy"),
                mmap_mode="r",
                allow_pickle=False,
            )
            error = max(
                float(
                    np.max(
                        np.abs(
                            actual[source, slot].astype(np.float64)
                            - reference[source, slot].astype(np.float64)
                        )
                    )
                )
                for source in range(2)
                for slot in range(8)
            )
            proof = {
                "maximum_absolute_difference": error,
                "atol": config["profile"]["parent_replay_atol"],
                "passed": error <= config["profile"]["parent_replay_atol"],
            }
            shared.write_new(profile / (name + "_replay.json"), proof)
            proofs[name] = proof
            if not proof["passed"]:
                raise ValueError("Frozen B resource-profile parent map replay differs")
            model.cpu()
            del actual, reference
        phase["frozen_B_replays"] = proofs
        phase["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        del models, inputs
    return phase


def profile_gate(
    destination: Path, config: dict, phase: dict, baseline_rows: list, clock: dict, guard: Any
) -> None:
    """Time known B CPU scoring/persistence and forecast all2400updates plus324mixed inference batches.

    Parameters
    ----------
    destination : Path
        New study output root.
    config : dict
        Frozen multipliers, reserves and hard caps.
    phase : dict
        Measured actual training and inference-persistence timings.
    baseline_rows : list
        Exactly closed full original B rows.
    clock : dict
        Whole-stage clocks including all earlier input/cohort/closure work.
    guard : Any
        Live measured resource guard.

    Returns
    -------
    None
        Saves the profile forecast and refuses full fitting if any gate fails.
    """
    started = clocks.new_clock()
    rows = score_models(
        config,
        destination / "profile/frozen_B_probabilities",
        "B",
        "generated",
        config["profile"]["inference_slots"],
        guard,
    )
    known = {(r["source"], r["slot"], r["fit_id"]): r for r in baseline_rows}
    if any(row != known[row["source"], row["slot"], row["fit_id"]] for row in rows):
        raise ValueError("Known frozen-B profile scoring differs")
    quality.write_rows(destination / "profile/record_rows.jsonl", rows)
    scoring = clocks.elapsed(started)
    actual = clocks.elapsed(clock)
    forecast = {}
    for key, reserve in [("cpu_seconds", "cpu_reserve_seconds"), ("wall_seconds", "wall_reserve_seconds")]:
        recurring = max(
            t[key] for bank in phase["training"].values() for t in bank["original_step_timings"][1:]
        )
        inference = (phase["recurring_inference"][key] + scoring[key]) / 36
        forecast[key] = actual[key] + 2 * 2400 * recurring + 2 * 324 * inference + config["profile"][reserve]
    update = max(
        t["wall_seconds"] for bank in phase["training"].values() for t in bank["original_step_timings"][1:]
    )
    forecast["gpu_lock_seconds"] = (
        clock["gpu_cost"]
        + 2 * 2400 * update
        + 2 * 324 * phase["recurring_inference"]["wall_seconds"] / 36
        + config["profile"]["gpu_reserve_seconds"]
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
            "recurring_CPU_scoring": scoring,
            "forecast": forecast,
            "forecast_gates": gates,
            "profile_training_updates": 10,
            "frozen_B_profile_batches": 36,
            "future_scientific_updates": 2400,
            "future_mixed_inference_batches": 324,
        },
    )
    if not all(gates.values()):
        raise ValueError("Full actual training/inference/scoring resource forecast failed")


def fit_and_infer(
    destination: Path, config: dict, plans: list, banks: dict, counts: dict, clock: dict, guard: Any
) -> dict:
    """Close all three generated fits first, then train and persist every fixed mixed state/map.

    Parameters
    ----------
    destination : Path
        New study root.
    config : dict
        Frozen budgets and reference bank declarations.
    plans : list
        All original paired initialization/schedule specifications.
    banks : dict
        Actual generated and mixed training banks, released after fitting.
    counts : dict
        Actual attempted/completed operation counters.
    clock : dict
        Whole-stage additive resource clocks.
    guard : Any
        Live resource guard during every update/inference/persistence path.

    Returns
    -------
    dict
        Every exact baseline closure and unchanged final mixed tensor/map identity.
    """
    with measured_gpu(destination, "full_training_inference", clock) as phase:
        phase["cuda"] = cuda_admission(config, False)
        models = {}
        states = {}
        training_rows = []
        for arm in core.TRAINING_ARMS:
            for plan in plans:
                model, row = training.train_one(plan, arm, banks[arm], config, counts, guard)
                training_rows.append(row)
                if arm == "mixed":
                    name = f"fit_{plan['fit_id']}_mixed"
                    models[name] = model
                    states[name] = row["final_tensor_state_sha256"]
                else:
                    del model
        banks.clear()
        gc.collect()
        for model in models.values():
            model.to("cuda")
        inputs = np.load(
            from_stored(config["evaluation_root"]) / "inputs.npy", mmap_mode="r", allow_pickle=False
        )
        subset = destination / "mixed_subset_probabilities"
        subset_hash = execution.infer_bank(
            subset, inputs, models, config["profile"]["inference_slots"], guard
        )
        counts["mixed_subset_batches"] = 36
        full = destination / "probabilities"
        hashes = execution.infer_bank(full, inputs, models, list(range(64)), guard)
        counts["mixed_full_batches"] = 288
        evidence = {}
        for name, model in models.items():
            guard()
            first = np.load(subset / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            complete = np.load(full / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            exact = bool(np.array_equal(first, complete[:, :8]))
            digest = identities.tensor_state_sha256(model)
            row = {
                "subset_full_exact": exact,
                "state_unchanged": digest == states[name],
                "final_tensor_state_sha256": digest,
                "subset_sha256": subset_hash[name],
                "full_sha256": hashes[name],
            }
            shared.write_new(destination / (name + "_inference_integrity.json"), row)
            evidence[name] = row
            if not exact or digest != states[name]:
                raise ValueError("Mixed subset/full map or state integrity differs")
            model.cpu()
            del first, complete
        phase.update(
            training_results=training_rows,
            mixed_inference=evidence,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
        del inputs, models
    return phase


def figures(destination: Path, statistics: dict) -> None:
    """Show every fixed training/source/editor cell and both real control contrasts.

    Parameters
    ----------
    destination : Path
        New local study root.
    statistics : dict
        Complete output-derived points/intervals with all fourteen individual guards.

    Returns
    -------
    None
        Saves aggregate figures without selecting a fit or clinical outcome.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    names = []
    values = []
    errors = []
    for source in transfer.SOURCES:
        for family in ["hann", "biphasic"]:
            row = statistics["endpoints"][f"mixed_minus_generated__{source}__{family}__pooled__edited_ap"]
            names.append(f"{source}/{family}")
            values.append(row["mean"])
            errors.append([row["mean"] - row["ci_low"], row["ci_high"] - row["mean"]])
    axes[0].scatter(np.arange(4), values)
    axes[0].vlines(
        np.arange(4),
        [v - e[0] for v, e in zip(values, errors, strict=True)],
        [v + e[1] for v, e in zip(values, errors, strict=True)],
    )
    axes[0].set(xticks=np.arange(4), xticklabels=names, ylabel="Mixed minus generated-only direct AP")
    axes[0].axhline(0, color="grey")
    for index, control in enumerate(["sham", "nuisance"]):
        row = statistics["endpoints"][f"mixed_minus_generated__real__{control}__top600_mean"]
        axes[1].scatter(index, row["mean"])
        axes[1].vlines(index, row["ci_low"], row["ci_high"])
    axes[1].axhline(0.02, color="tab:red", linestyle="--")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Real sham", "Real nuisance"],
        ylabel="Mixed minus generated-only top600 probability",
    )
    figure.suptitle(
        "Introduced numerical-change transfer; conditional fixed-fit intervals, not disease accuracy"
    )
    figure.savefig(destination / "all_source_family_and_control_contrasts.png", dpi=120)
    plt.close(figure)


def run(root: Path, name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Execute only admitted matched learning, preserving failures and every actual resource cost.

    Parameters
    ----------
    root : Path
        Isolated source checkout hosting baseline-owned data/output aliases.
    name : str
        Frozen executable identity admission.
    clock : dict
        Actual CPU/wall clocks captured before scientific imports.

    Returns
    -------
    dict
        Complete provisional result requiring the separate independent audit.
    """
    config, admission = preflight(name)
    destination = shared.confined(root, config["output_root"])
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        "study_id": config["study_id"],
        "status": "started",
        "final_input_reverification_completed": False,
    }
    timers = {}
    verified = False
    counts = dict.fromkeys(
        [
            "profile_update_attempts",
            "profile_updates",
            "generated_update_attempts",
            "generated_updates",
            "mixed_update_attempts",
            "mixed_updates",
            "frozen_B_profile_batches",
            "mixed_subset_batches",
            "mixed_full_batches",
        ],
        0,
    )
    limits = config["resources"]
    compatible = {
        **config,
        "resources": {
            **limits,
            "peak_rss_bytes": limits["rss_bytes"],
            "gpu_lock_wall_seconds": limits["gpu_lock_seconds"],
            "maximum_gpu_allocated_bytes": limits["cuda_allocated_bytes"],
            "output_bytes": limits["output_bytes"] - 1024**2 + 16384,
        },
    }

    def guard() -> None:
        """Keep actual whole-stage/GPU/memory and prospective output bounds live."""
        execution.resource_guard(config, clock)

    def interrupted(signum: int, frame: Any) -> None:
        """Retain the attempt before the terminal bookkeeping reserve is exhausted."""
        raise TimeoutError(f"Mixed-background process timer:{signum}")

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
            "torch": torch.__version__,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }
        sources = clocks.bridge_runtime.peak_runtime.snapshot(destination, admission, name)
        result["source_sha256"] = sources
        shared.verify_prior(root, admission)
        verified = True
        verify_inputs(admission, guard)
        baseline = close_predecessor(destination, config, guard)
        records = data.selected_records(config)
        data.verify_raw(config, records, guard)
        if shutil.disk_usage(destination).free < limits["minimum_free_output_bytes"]:
            raise OSError("Insufficient free output disk")
        planned = (
            80 * 4 * 12 * 5000 * 4
            + 80 * 4 * 12 * 5000
            + 80 * 2 * 12 * 5000
            + 3 * 2 * 64 * 6 * 12 * 5000 * 4
            + 2 * 3 * 2 * 8 * 6 * 12 * 5000 * 4
            + 32 * 1024**2
            + config["profile"]["output_reserve_bytes"]
        )
        clock["output_bound"] = shared.directory_bytes(destination) + planned
        guard()
        banks, qualification = data.prepare_banks(destination, config, records, guard)
        plans = training.prepare_plans(config, destination)
        phase = profile_gpu(destination, config, plans, banks, counts, clock, guard)
        profile_gate(destination, config, phase, baseline, clock, guard)
        fitted = fit_and_infer(destination, config, plans, banks, counts, clock, guard)
        rows = baseline + score_models(
            config, destination / "probabilities", "mixed", "mixed", list(range(64)), guard
        )
        quality.write_rows(destination / "record_rows.jsonl", rows)
        statistics, draws = core.summarize(rows, config)
        clocks.save_array(destination / "bootstrap.npy", draws)
        figures(destination, statistics)
        verify_inputs(admission, guard)
        data.verify_raw(config, records, guard)
        for path, digest in sources.items():
            guard()
            if sha256_file(from_stored(path)) != digest:
                raise ValueError("Mixed executable source changed during execution")
        result.update(
            status="PASS",
            statistics=statistics,
            training_bank_qualification=qualification,
            training_and_inference=fitted,
            technical_passed_before_independent_audit=True,
            independent_audit_required=True,
            clinical_claim=False,
            new_generator_forwards=0,
            raw_record_decodes=40,
            annotation_reads=0,
            clinical_labels_interpreted=0,
            final_input_reverification_completed=True,
            baseline_probability_references={
                str(f): f"{config['evaluation_root']}/probabilities/fit_{f}_B.npy" for f in transfer.FITS
            },
            reserved_terminal_output_bytes=1024**2,
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
        result["operation_counts"] = counts
        result["raw_record_decode_attempts"] = len(
            list((destination / "training_record_receipts").glob("*_started.json"))
        )
        clocks.finish(destination, compatible, admission, result, clock, verified, timers)
    return result
