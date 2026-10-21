"""Execute one prospectively admitted frozen-localizer diagnostic without fitting or generator calls."""

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ssl_synthetic_generated_quality_run as gpu_runtime
from ecg_experiment import ssl_synthetic_ludb_coherence_run_v2 as bridge_runtime
from ecg_experiment import ssl_synthetic_ludb_localizer as core
from ecg_experiment import ssl_synthetic_ludb_peak_timing as peak
from ecg_experiment import ssl_synthetic_ludb_qualification as structural
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment import ssl_synthetic_paired_suppression as paired
from ecg_experiment import ssl_synthetic_paired_suppression_run as paired_runtime
from ecg_experiment import ssl_synthetic_pretrained_generator as state_identity
from ecg_experiment import ssl_synthetic_randomized_localization as architecture
from ecg_experiment import ssl_synthetic_randomized_localization_run as inference
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored

CONFIG = "configs/ssl_synthetic_ludb_localizer_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-localizer-v1.md"
SOURCES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *bridge_runtime.SOURCES,
            *paired_runtime.SOURCES,
            CONFIG,
            PROTOCOL,
            "ecg_experiment/ssl_synthetic_ludb_localizer.py",
            "ecg_experiment/ssl_synthetic_ludb_localizer_run.py",
            "scripts/experiments/run_ssl_synthetic_ludb_localizer.py",
            "tests/test_ssl_synthetic_ludb_localizer.py",
        )
    )
)


def checkpoint_path(config: dict[str, Any], fit: int, arm: str) -> Path:
    """Name the immutable paired-study model file without choosing among fits or arms.

    Parameters
    ----------
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    fit : int
        Prospectively fixed fit ID from one through three.
    arm : str
        One of the four fixed B, P, K and PK objective identities.

    Returns
    -------
    Path
        Declared transformed values, identities or complete descriptive results.
    """
    return from_stored(config["model_root"]) / f"fit_{fit}" / arm / "final_state.pt"


def load_model(config: dict[str, Any], fit: int, arm: str) -> tuple[torch.nn.Module, str]:
    """Strictly load the six CPU weight tensors into the unchanged observed-only architecture.

    Parameters
    ----------
    config : dict
        Frozen study roots and the complete model roster.
    fit : int
        One of the three prospectively fixed fit IDs.
    arm : str
        One of B, P, K and PK.

    Returns
    -------
    tuple[torch.nn.Module, str]
        Evaluation-only CPU model and its separate named-tensor state digest.
    """
    if fit not in core.FITS or arm not in core.ARMS:
        raise ValueError("No model selection or substitute arm is permitted")
    state = torch.load(checkpoint_path(config, fit, arm), map_location="cpu", weights_only=True)
    model = architecture.build_model()
    model.load_state_dict(state, strict=True)
    model.eval()
    digest = state_identity.tensor_state_sha256(model)
    expected = shared.read_json(from_stored(config["model_root"]) / "summary.json")["final_state_sha256"]
    if digest != expected[str(fit)][arm]:
        raise ValueError("Frozen model named-tensor digest differs from its actual parent")
    return model, digest


PROFILE_CONFIG = "configs/ssl_synthetic_ludb_localizer_profile_v1.json"
PROFILE_PROTOCOL = "docs/ssl-synthetic-localization-ludb-localizer-profile-v1.md"
SOURCES = (*SOURCES, PROFILE_CONFIG, PROFILE_PROTOCOL)
MODEL_ROOT = "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"
TIMING_ROOT = "outputs/ssl_synthetic_localization/ludb_peak_timing_v1"
COHERENCE_ROOT = "outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2"
REQUIRED_INPUTS: tuple[str, ...] = (
    "outputs/ssl_synthetic_localization/localizer_landmark_inventory_v1/summary.json",
    *[
        f"{MODEL_ROOT}/fit_{fit}/{arm}/{name}"
        for fit in core.FITS
        for arm in core.ARMS
        for name in ["final_state.pt", "fresh_probabilities.npy", "fresh_metrics.json"]
    ],
    *[
        f"{MODEL_ROOT}/{name}"
        for name in [
            "summary.json",
            "prospective_ledger.json",
            "independent_audit_v1/summary.json",
            "independent_audit_v1/prospective_ledger.json",
            "fresh_observed.npy",
            "fresh_support.npy",
        ]
    ],
    "outputs/ssl_synthetic_localization/randomized_localization_v1/spatial_prior.npy",
    *[
        f"{TIMING_ROOT}/{name}"
        for name in [
            "summary.json",
            "prospective_ledger.json",
            "independent_audit_v2/summary.json",
            "independent_audit_v2/prospective_ledger.json",
            "record_rows.jsonl",
            "bootstrap_gains.npy",
            "release_file_identities.json",
        ]
    ],
    *[
        f"{COHERENCE_ROOT}/{name}"
        for name in [
            "summary.json",
            "prospective_ledger.json",
            "independent_audit_v1/summary.json",
            "independent_audit_v1/prospective_ledger.json",
            "generated_rows.jsonl",
            "ludb_rows.jsonl",
            "bootstrap_cohort_means_and_contrasts.npy",
        ]
    ],
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read an admitted saved numerical row table without reopening original annotations.

    Parameters
    ----------
    path : Path
        Admitted or newly created local artifact path.

    Returns
    -------
    list[dict[str, Any]]
        Declared transformed values, identities or complete descriptive results.
    """
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_array(path: Path, value: np.ndarray) -> None:
    """Persist a complete exclusively created array without unsafe pickle support.

    Parameters
    ----------
    path : Path
        Admitted or newly created local artifact path.
    value : np.ndarray
        Complete numerical array to persist without pickle support.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def preflight(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require complete current scientific source and input declarations before output or GPU work.

    Parameters
    ----------
    name : str
        Committed repository-relative identity admission filename.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Declared transformed values, identities or complete descriptive results.
    """
    config, admission = shared.read_json(from_stored(CONFIG)), shared.read_json(from_stored(name))
    sources = admission["source_sha256"]
    inventory = admission["inventory"]
    if admission["input_sha256"].get(inventory["path"]) != inventory["sha256"]:
        raise ValueError("Setup inventory identity must also be explicitly input-admitted")
    if not set(SOURCES).issubset(sources) or not set(REQUIRED_INPUTS).issubset(admission["input_sha256"]):
        raise ValueError("Localizer source or predecessor-input roster is incomplete")
    if (
        sources[CONFIG] != admission["frozen_config_sha256"]
        or sources[PROTOCOL] != admission["frozen_protocol_sha256"]
        or admission["prior_ledger"] != config["prior_ledger"]
    ):
        raise ValueError("Localizer scientific or actual stage30 accounting identity differs")
    for path, expected in sources.items():
        if sha256_file(from_stored(path)) != expected:
            raise ValueError("Localizer source changed: " + path)
    assets = admission["assets"]
    if (
        assets["logical_root"] != config["raw_alias"]
        or assets["actual_root"] != config["actual_raw_target"]
        or not os.path.samefile(from_stored(assets["logical_root"]), from_stored(assets["actual_root"]))
    ):
        raise ValueError("Localizer read-only-use alias differs")
    return config, admission


def verify_inputs(admission: dict[str, Any], guard: Any) -> None:
    """Hash every explicitly admitted source, predecessor and400 identical qualified real signal assets.

    Parameters
    ----------
    admission : dict[str, Any]
        Committed byte identities for executable sources, local inputs and prior accounting.
    guard : Any
        Live CPU, wall, RSS, GPU and retained-byte guard.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    for path, expected in admission["input_sha256"].items():
        guard()
        if sha256_file(from_stored(path)) != expected:
            raise ValueError("Admitted localizer input changed: " + path)
    assets = admission["assets"]
    expected_names = {
        f"{identifier}.{extension}" for identifier in range(1, 201) for extension in ["dat", "hea"]
    }
    qualified = shared.read_json(from_stored(TIMING_ROOT) / "release_file_identities.json")
    if set(assets["sha256"]) != expected_names:
        raise ValueError("Require all400 qualified signal/header assets")
    for name, digest in assets["sha256"].items():
        guard()
        first, second = from_stored(assets["logical_root"]) / name, from_stored(assets["actual_root"]) / name
        if not os.path.samefile(first, second) or sha256_file(second) != digest or qualified[name] != digest:
            raise ValueError("Localizer release or alias identity changed")


def verify_parent_manifests(config: dict[str, Any], admission: dict[str, Any]) -> None:
    """Anchor every admitted predecessor payload to its original independently audited output manifest.

    Parameters
    ----------
    config : dict
        Frozen paths for the original three completed parents.
    admission : dict
        Exact input identities, including all checkpoint, landmark and probability payloads.

    Returns
    -------
    None
        Raise before input preparation if original annotation or checkpoint provenance differs.
    """
    fixed = [
        (
            config["model_root"],
            "8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111",
            "independent_audit_v1",
            "main_sha256",
        ),
        (
            config["landmark_root"],
            "2e48b4e06512f126bb0eddf12306b54c462da1e300fa2624c0effed1c1bc8c29",
            "independent_audit_v2",
            "main_summary_sha256",
        ),
        (
            config["coherence_root"],
            "b34fa682898d0cf381518a43dcc12c73afe3577fc6ccf2b959faf19b02254e0b",
            "independent_audit_v1",
            "main_summary_sha256",
        ),
    ]
    receipts = {
        "summary.json",
        "prospective_ledger.json",
        "independent_audit_v1/summary.json",
        "independent_audit_v1/prospective_ledger.json",
        "independent_audit_v2/summary.json",
        "independent_audit_v2/prospective_ledger.json",
    }
    for root, expected, audit_directory, summary_key in fixed:
        if admission["input_sha256"][root + "/summary.json"] != expected:
            raise ValueError("Original audited parent summary identity differs")
        parent = shared.read_json(from_stored(root) / "summary.json")
        audit = shared.read_json(from_stored(root) / audit_directory / "summary.json")
        if audit[summary_key] != expected or not audit["all_passed"] or audit["mismatch_count"] != 0:
            raise ValueError("Original parent independent audit binding differs")
        for name, digest in admission["input_sha256"].items():
            if name.startswith(root + "/"):
                relative = name[len(root) + 1 :]
                if relative not in receipts and parent["output_sha256"].get(relative) != digest:
                    raise ValueError("Admitted numerical input differs from its original parent manifest")
    model = shared.read_json(from_stored(config["model_root"]) / "summary.json")
    prior = "outputs/ssl_synthetic_localization/randomized_localization_v1/spatial_prior.npy"
    if admission["input_sha256"][prior] != model["input_sha256"]["learning"]["spatial_prior.npy"]:
        raise ValueError("Original TRAIN-only spatial-prior identity differs")


def close_saved_studies(destination: Path, config: dict[str, Any], guard: Any) -> list[dict[str, Any]]:
    """Exactly reproduce all synthetic AP rows, complete old verdict and both public-study draw arrays.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    guard : Any
        Live CPU, wall, RSS, GPU and retained-byte guard.

    Returns
    -------
    list[dict[str, Any]]
        Declared transformed values, identities or complete descriptive results.
    """
    replication = paired_runtime.replication
    root = from_stored(config["model_root"])
    parent_audit = shared.read_json(root / "independent_audit_v1/summary.json")
    if not parent_audit["all_passed"] or parent_audit["mismatch_count"] != 0:
        raise ValueError("Original paired-study independent technical audit did not pass")
    observed = np.load(root / "fresh_observed.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(root / "fresh_support.npy", mmap_mode="r", allow_pickle=False)
    prior = np.load(
        from_stored("outputs/ssl_synthetic_localization/randomized_localization_v1/spatial_prior.npy"),
        allow_pickle=False,
    )
    fits = []
    for fit in core.FITS:
        row = {"fit_id": fit}
        for arm in core.ARMS:
            guard()
            directory = root / f"fit_{fit}" / arm
            maps = np.load(directory / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False)
            actual = {
                "cases": replication.score_panel(
                    maps, observed, support, np.arange(2000, 2064), np.arange(2026105000, 2026105064), prior
                ),
                "controls": replication.control_scores(maps),
                "hotspots": replication.hotspot_scores(
                    maps, support, np.arange(2000, 2064), np.arange(2026105000, 2026105064)
                ),
                "top600_control": paired.control_top600(maps),
            }
            if actual != shared.read_json(directory / "fresh_metrics.json"):
                raise ValueError("Full saved synthetic AP/case/control reconstruction differs")
            row[arm] = actual
        fits.append(row)
    aggregate = paired.aggregate(fits)
    if aggregate != shared.read_json(root / "summary.json")["aggregate"]:
        raise ValueError("Exact old full aggregate/fifteen-gate closure failed")
    shared.write_new(
        destination / "synthetic_predecessor_closure.json",
        {
            "all_exact": True,
            "model_panels": 12,
            "backgrounds_per_panel": 64,
            "aggregate": aggregate,
            "waveform_map_derived_AP_rows": 1536,
        },
    )
    timing = from_stored(config["landmark_root"])
    rows = read_rows(timing / "record_rows.jsonl")
    statistics = peak.aggregate(rows)
    draws = np.asarray(statistics.pop("bootstrap_gains"))
    if statistics != shared.read_json(timing / "summary.json")["statistics"] or not np.array_equal(
        draws, np.load(timing / "bootstrap_gains.npy", allow_pickle=False)
    ):
        raise ValueError("Exact landmark predecessor primary/control/bootstrap closure failed")
    audit = shared.read_json(timing / "independent_audit_v2/summary.json")
    if not (audit["all_passed"] and audit["mismatch_count"] == 0 and audit["scientific_passed"]):
        raise ValueError("Independent landmark prerequisite did not pass")
    shared.write_new(
        destination / "timing_predecessor_closure.json", {"all_exact": True, "statistics": statistics}
    )
    coherence = from_stored(config["coherence_root"])
    statistics, draws = bridge_runtime.original.core.summarize(
        read_rows(coherence / "generated_rows.jsonl"), read_rows(coherence / "ludb_rows.jsonl")
    )
    if statistics != shared.read_json(coherence / "summary.json")["statistics"] or not np.array_equal(
        draws, np.load(coherence / "bootstrap_cohort_means_and_contrasts.npy", allow_pickle=False)
    ):
        raise ValueError("Exact coherence predecessor statistics/bootstrap closure failed")
    audit = shared.read_json(coherence / "independent_audit_v1/summary.json")
    if not (audit["all_passed"] and audit["mismatch_count"] == 0):
        raise ValueError("Independent descriptive predecessor did not pass")
    shared.write_new(
        destination / "coherence_predecessor_closure.json", {"all_exact": True, "statistics": statistics}
    )
    return rows


def prepare_records(
    identifiers: list[int], destination: Path, config: dict[str, Any], guard: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Read structural-only headers through the alias and preserve every frozen input transformation.

    Parameters
    ----------
    identifiers : list[int]
        Prospectively fixed numeric record IDs in their declared order.
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    guard : Any
        Live CPU, wall, RSS, GPU and retained-byte guard.

    Returns
    -------
    tuple[np.ndarray, list[dict[str, Any]]]
        Declared transformed values, identities or complete descriptive results.
    """
    arrays, metadata = [], []
    for identifier in identifiers:
        guard()
        physical, signal = structural.read_signal(
            from_stored(config["raw_alias"]) / str(identifier), destination / "headers" / str(identifier)
        )
        values, transform = core.prepare_input(physical)
        arrays.append(values)
        metadata.append({"record_id": identifier, "transformation": transform, "signal_metadata": signal})
    return np.stack(arrays), metadata


def profile_figure(destination: Path, inputs: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    """Show fixed record1 leadII for every model without computing a diagnostic score.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    inputs : np.ndarray
        Exact standardized float32 observed-only model inputs.
    predictions : dict[str, np.ndarray]
        All twelve profile probability banks without diagnostic endpoint scores.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(16, 8), constrained_layout=True)
    for index, (name, maps) in enumerate(predictions.items()):
        axis = axes.flat[index]
        axis.plot(np.arange(5000) / 500, inputs[0, 1], color="black", linewidth=0.5)
        probability = axis.twinx()
        probability.plot(np.arange(5000) / 500, maps[0, 1], color="tab:red", linewidth=0.7)
        probability.set_ylim(0, 1)
        axis.set(title=name, xlabel="Seconds")
    figure.suptitle("Fixed unscored profile record1, leadII; synthetic-trained outputs")
    figure.savefig(destination / "profile_fixed_maps.png", dpi=120)
    plt.close(figure)


def elapsed(clock: dict[str, float]) -> dict[str, float]:
    """Measure process CPU and monotonic wall cost, including synchronous persistence.

    Parameters
    ----------
    clock : dict[str, float]
        Actual whole-stage CPU, wall and GPU-lock accounting; phase timing never resets it.

    Returns
    -------
    dict[str, float]
        Declared transformed values, identities or complete descriptive results.
    """
    return {
        "cpu_seconds": time.process_time() - clock["cpu"],
        "wall_seconds": time.perf_counter() - clock["wall"],
    }


def new_clock() -> dict[str, float]:
    """Start a bounded measured phase without resetting the actual whole-stage accounting.

    Returns
    -------
    dict[str, float]
        Declared transformed values, identities or complete descriptive results.
    """
    return {"cpu": time.process_time(), "wall": time.perf_counter()}


def retain_replay(
    destination: Path,
    name: str,
    predicted: np.ndarray,
    reference: np.ndarray,
    identity: str,
    tolerance: float,
) -> dict[str, Any]:
    """Preserve each successful or failed replay maximum before enforcing its gate.

    Parameters
    ----------
    destination : Path
        Exclusively created profile directory retaining failures.
    name : str
        Fixed model identity.
    predicted, reference : np.ndarray
        New CUDA maps and original saved four-variant maps.
    identity : str
        Original model named-tensor digest.
    tolerance : float
        Prospectively frozen absolute replay tolerance with zero relative tolerance.

    Returns
    -------
    dict
        Complete replay evidence, including a Boolean gate enforced by the caller after persistence.
    """
    shape_valid = predicted.shape == reference.shape == (4, 12, 5000)
    finite = bool(np.isfinite(predicted).all() and np.isfinite(reference).all())
    difference = (
        float(np.max(np.abs(predicted.astype(np.float64) - reference.astype(np.float64))))
        if shape_valid and finite
        else None
    )
    passed = shape_valid and predicted.dtype == np.float32 and finite and difference <= tolerance
    row = {
        "model": name,
        "shape": list(predicted.shape),
        "dtype": str(predicted.dtype),
        "finite": finite,
        "maximum_absolute_difference": difference,
        "atol": tolerance,
        "rtol": 0,
        "initial_tensor_state_sha256": identity,
        "all_passed": bool(passed),
    }
    if shape_valid:
        save_array(destination / (name + "_synthetic_replay.npy"), predicted)
    shared.write_new(destination / (name + "_synthetic_replay.json"), row)
    return row


def new_probabilities(path: Path, count: int) -> np.memmap:
    """Allocate profile and full banks using the same exclusive memmap persistence path.

    Parameters
    ----------
    path : Path
        New local array filename.
    count : int
        Eight profile or160 full records.

    Returns
    -------
    np.memmap
        Float32 twelve-output write-through probability bank.
    """
    if path.exists():
        raise FileExistsError(path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(count, 12, 5000))


def write_batch(maps: np.memmap, start: int, predicted: np.ndarray) -> None:
    """Persist and flush one identical batch primitive in profile and complete inference.

    Parameters
    ----------
    maps : np.memmap
        Exclusively allocated profile or full probability bank.
    start : int
        Fixed first record offset for the batch.
    predicted : np.ndarray
        Finite float32 batch-eight output.

    Returns
    -------
    None
        Persist the exact maps and synchronously flush the modified pages.
    """
    maps[start : start + len(predicted)] = predicted
    maps.flush()


def profile(
    destination: Path, config: dict[str, Any], clock: dict[str, float], guard: Any
) -> tuple[dict[str, torch.nn.Module], dict[str, str], dict[str, np.ndarray]]:
    """Measure fixed initialization/replay and actual repeated batch-eight paths before full inference.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    clock : dict[str, float]
        Actual whole-stage CPU, wall and GPU-lock accounting; phase timing never resets it.
    guard : Any
        Live CPU, wall, RSS, GPU and retained-byte guard.

    Returns
    -------
    tuple[dict[str, torch.nn.Module], dict[str, str], dict[str, np.ndarray]]
        Declared transformed values, identities or complete descriptive results.
    """
    fixed_clock = new_clock()
    if not torch.cuda.is_available():
        raise RuntimeError("Actual PyTorch CUDA access failed inside the acquired project lock")
    if float((torch.ones(1, device="cuda") * 2).cpu()[0]) != 2:
        raise ValueError("Small actual CUDA operation failed")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    if free < config["resources"]["minimum_free_gpu_bytes"]:
        raise MemoryError("CUDA0 lacks the prospectively required free memory")
    torch.cuda.reset_peak_memory_stats(0)
    models, identities, replay = {}, {}, []
    root = from_stored(config["model_root"])
    synthetic = np.load(root / "fresh_observed.npy", mmap_mode="r", allow_pickle=False)[0]
    for fit in core.FITS:
        for arm in core.ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            model, identity = load_model(config, fit, arm)
            models[name], identities[name] = model.to("cuda"), identity
            predicted = inference.infer(model, synthetic, "cuda")
            reference = np.load(
                root / f"fit_{fit}" / arm / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False
            )[0]
            replay_row = retain_replay(
                destination, name, predicted, reference, identity, config["profile"]["old_map_atol"]
            )
            replay.append(replay_row)
            shared.write_new(
                destination / (name + "_fixed_progress.json"),
                {
                    "completed_synthetic_replays": len(replay),
                    "checkpoint_loads": len(models),
                    "elapsed_fixed_phase": elapsed(fixed_clock),
                },
            )
            if not replay_row["all_passed"]:
                raise ValueError("Fixed synthetic CUDA checkpoint/map replay exceeds frozen tolerance")
    fixed = elapsed(fixed_clock)
    repeated_clock = new_clock()
    inputs, metadata = prepare_records(config["profile"]["record_ids"], destination, config, guard)
    save_array(destination / "inputs.npy", inputs)
    shared.write_new(destination / "input_metadata.json", metadata)
    predictions = {}
    hashes = {}
    for name, model in models.items():
        guard()
        predicted = inference.infer(model, inputs, "cuda")
        require_maps(predicted, 8)
        retained_map = new_probabilities(destination / (name + ".npy"), 8)
        write_batch(retained_map, 0, predicted)
        hashes[name] = sha256_file(destination / (name + ".npy"))
        del retained_map
        shared.write_new(
            destination / (name + "_repeated_progress.json"),
            {
                "completed_batch8_forwards": len(hashes),
                "elapsed_repeated_phase": elapsed(repeated_clock),
                "map_sha256": hashes[name],
            },
        )
        predictions[name] = predicted
    repeated = elapsed(repeated_clock)
    final_fixed_clock = new_clock()
    profile_figure(destination, inputs, predictions)
    figure_digest = sha256_file(destination / "profile_fixed_maps.png")
    fixed = {key: value + elapsed(final_fixed_clock)[key] for key, value in fixed.items()}
    amendment = shared.read_json(from_stored(PROFILE_CONFIG))
    actual = elapsed(clock)
    setup = {key: max(0.0, actual[key] - fixed[key] - repeated[key]) for key in fixed}
    forecast = {
        key: setup[key]
        + amendment["safety_factor"] * fixed[key]
        + amendment["safety_factor"] * amendment["repeated_batch_multiplier"] * repeated[key]
        + amendment["cpu_wall_overhead_reserve_seconds"]
        for key in fixed
    }
    forecast["gpu_lock_wall_seconds"] = (
        2 * fixed["wall_seconds"] + 40 * repeated["wall_seconds"] + amendment["gpu_overhead_reserve_seconds"]
    )
    forecast["output_bytes"] = (
        shared.directory_bytes(destination.parent)
        + 460800000
        + 38400000
        + 51200000
        + 64 * 1024**2
        + 8 * 1024**2
        + config["profile"]["output_reserve_bytes"]
    )
    limits = config["resources"]
    gates = {
        key: forecast[key] < limits[key]
        for key in ["cpu_seconds", "wall_seconds", "gpu_lock_wall_seconds", "output_bytes"]
    }
    gates["cuda_allocated_memory"] = (
        torch.cuda.max_memory_allocated(0) < limits["maximum_gpu_allocated_bytes"]
    )
    result = {
        "peak_allocated_bytes_at_profile": torch.cuda.max_memory_allocated(0),
        "fixed_phase": fixed,
        "repeated_phase": repeated,
        "setup_outside_phases": setup,
        "actual_at_forecast": actual,
        "forecast": forecast,
        "forecast_gates": gates,
        "profile_scores_computed": 0,
        "record_ids": config["profile"]["record_ids"],
        "synthetic_replays": replay,
        "profile_map_sha256": hashes,
        "figure_sha256": figure_digest,
        "operations": {
            "checkpoint_loads": 12,
            "synthetic_forwards": 12,
            "real_batch8_forwards": 12,
            "real_record_reads": 8,
            "profile_figures": 1,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "free_bytes_at_admission": free,
            "total_bytes": total,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    shared.write_new(destination / "summary.json", result)
    if not all(gates.values()):
        raise ValueError("Actual-path profile forecast failed; no full real inference or diagnostic score")
    clock["output_bound"] = forecast["output_bytes"]
    return models, identities, predictions


def require_maps(predicted: np.ndarray, count: int) -> None:
    """Reject incomplete or invalid maps across all twelve outputs, including unscored derived leads.

    Parameters
    ----------
    predicted : np.ndarray
        New finite float32 probability maps from the fixed inference path.
    count : int
        Expected fixed inference batch size.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    if (
        predicted.shape != (count, 12, 5000)
        or predicted.dtype != np.float32
        or not np.isfinite(predicted).all()
        or np.any(predicted < 0)
        or np.any(predicted > 1)
    ):
        raise ValueError("All saved twelve-output maps must be finite float32 probabilities")


def full_inference(
    destination: Path,
    config: dict[str, Any],
    models: dict[str, torch.nn.Module],
    identities: dict[str, str],
    profile_maps: dict[str, np.ndarray],
    guard: Any,
) -> None:
    """Persist all160 fixed records and all240 full batch-eight model forwards without endpoint selection.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    models : dict[str, torch.nn.Module]
        All twelve strictly loaded, unchanged evaluation-only models.
    identities : dict[str, str]
        Original named-tensor state digests for every fixed model.
    profile_maps : dict[str, np.ndarray]
        Retained batch-eight profile maps for exact full-first-batch comparison.
    guard : Any
        Live CPU, wall, RSS, GPU and retained-byte guard.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    identifiers = [identifier for identifier in range(1, 201) if identifier % 5]
    inputs, metadata = prepare_records(identifiers, destination, config, guard)
    save_array(destination / "inputs.npy", inputs)
    shared.write_new(destination / "input_metadata.json", metadata)
    shared.write_new(destination / "record_ids.json", identifiers)
    directory = destination / "probabilities"
    directory.mkdir()
    states = {}
    for name, model in models.items():
        guard()
        path = directory / (name + ".npy")
        maps = new_probabilities(path, 160)
        for start in range(0, 160, 8):
            guard()
            predicted = inference.infer(model, inputs[start : start + 8], "cuda")
            require_maps(predicted, 8)
            if start == 0 and not np.array_equal(predicted, profile_maps[name]):
                raise ValueError("Profile/full first batch differs at fixed batch-eight geometry")
            write_batch(maps, start, predicted)
        final = state_identity.tensor_state_sha256(model)
        if final != identities[name]:
            raise ValueError("Frozen model tensor state changed during diagnostic inference")
        states[name] = {
            "initial_tensor_state_sha256": identities[name],
            "final_tensor_state_sha256": final,
            "unchanged": True,
            "full_batches": 20,
            "profile_first_batch_exact": True,
            "probabilities_sha256": sha256_file(path),
        }
        del maps
    shared.write_new(destination / "model_states.json", states)


def score_outputs(
    destination: Path, structure: list[dict[str, Any]], config: dict[str, Any], guard: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute only the committed native-output endpoints after all full inference paths have completed.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    structure : list[dict[str, Any]]
        Saved role and annotation rows; original annotation streams are not reopened.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    guard : Any
        Live CPU, wall, RSS, GPU and retained-byte guard.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        Declared transformed values, identities or complete descriptive results.
    """
    indexed = {row["record_id"]: row for row in structure}
    identifiers = [identifier for identifier in range(1, 201) if identifier % 5]
    inputs = np.load(destination / "inputs.npy", mmap_mode="r", allow_pickle=False)
    path = destination / "energy.npy"
    energy = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(160, 8, 5000))
    for position, _identifier in enumerate(identifiers):
        guard()
        energy[position] = np.stack(
            [structural.activity_proxy(inputs[position, index]) for index in core.NATIVE_INDICES]
        )
    energy.flush()
    rows, retained = [], shared.directory_bytes(destination)
    with (destination / "model_rows.jsonl").open("x") as stream:
        for fit in core.FITS:
            for arm in core.ARMS:
                maps = np.load(
                    destination / "probabilities" / f"fit_{fit}_{arm}.npy", mmap_mode="r", allow_pickle=False
                )
                for position, identifier in enumerate(identifiers):
                    guard()
                    row = core.score_record(maps[position], energy[position], indexed[identifier], fit, arm)
                    line = json.dumps(row, allow_nan=False) + "\n"
                    retained += len(line.encode())
                    if retained + 16 * 1024**2 >= config["resources"]["output_bytes"]:
                        raise ValueError("Localizer metric row/figure receipt reserve exhausted")
                    stream.write(line)
                    stream.flush()
                    rows.append(row)
                del maps
    statistics, bootstrap = core.aggregate(rows, config)
    save_array(destination / "bootstrap.npy", bootstrap)
    return statistics, rows


def figures(
    destination: Path,
    rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    structure: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Show all fixed fits/arms and the prospectively selected eight input/landmark examples.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    rows : list[dict[str, Any]]
        Complete saved numerical rows with every required record, model and lead.
    statistics : dict[str, Any]
        Complete output-derived descriptive summaries; no scientific-success threshold.
    structure : list[dict[str, Any]]
        Saved role and annotation rows; original annotation streams are not reopened.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    names = [f"{fit}{arm}" for fit in core.FITS for arm in core.ARMS]
    for metric, label in [("N_mean_midrank", "Native N rank"), ("mean_shift_N_midrank", "Six-control mean")]:
        axes[0].plot(
            names,
            [
                np.mean([row[metric] for row in rows if f"{row['fit_id']}{row['arm']}" == name])
                for name in names
            ],
            "o-",
            label=label,
        )
    axes[0].set(ylim=(0, 1), ylabel="Mean empirical N rank")
    axes[0].legend()
    arms = list(core.ARMS)
    axes[1].bar(
        arms,
        [
            statistics["per_arm"][arm]["energy_spearman"]["mean"]
            if statistics["per_arm"][arm]["energy_spearman"]["mean"] is not None
            else np.nan
            for arm in arms
        ],
    )
    for position, arm in enumerate(arms):
        count = statistics["per_arm"][arm]["energy_spearman"]["defined_count"]
        axes[1].text(position, -0.95, f"n={count}" if count else "undefined, n=0", ha="center")
    axes[1].set(ylim=(-1, 1), ylabel="Defined-only energy Spearman; denominator shown")
    axes[2].bar(arms, [statistics["per_arm"][arm]["energy_jaccard"] for arm in arms])
    axes[2].set(ylim=(0, 1), ylabel="Strict-q90 map/energy Jaccard")
    figure.savefig(destination / "all_model_landmark_and_activity.png", dpi=120)
    plt.close(figure)
    identifiers = shared.read_json(destination / "record_ids.json")
    inputs = np.load(destination / "inputs.npy", mmap_mode="r", allow_pickle=False)
    indexed = {row["record_id"]: row for row in structure}
    figure, axes = plt.subplots(8, 4, figsize=(18, 20), constrained_layout=True)
    for arm_position, arm in enumerate(core.ARMS):
        maps = np.load(destination / "probabilities" / f"fit_1_{arm}.npy", mmap_mode="r", allow_pickle=False)
        for row_position, (identifier, lead) in enumerate(
            [(identifier, lead) for identifier in [1, 2, 3, 4] for lead in ["ii", "v2"]]
        ):
            position, native = identifiers.index(identifier), structural.LEADS.index(lead)
            axis = axes[row_position, arm_position]
            axis.plot(np.arange(5000) / 500, inputs[position, native], color="black", linewidth=0.5)
            for sample in indexed[identifier]["leads"][native]["N_samples"]:
                axis.axvline(sample / 500, color="tab:blue", alpha=0.25, linewidth=0.7)
            twin = axis.twinx()
            twin.plot(np.arange(5000) / 500, maps[position, native], color="tab:red", linewidth=0.6)
            twin.set_ylim(0, 1)
            axis.set(title=f"Record{identifier} {lead.upper()}, fit1 {arm}", xlabel="Seconds")
        del maps
    figure.suptitle(
        "Fixed examples: standardized input, released N markers and frozen synthetic-trained maps"
    )
    figure.savefig(destination / "fixed_input_map_N_overlays.png", dpi=120)
    plt.close(figure)


def finish(
    destination: Path,
    config: dict[str, Any],
    admission: dict[str, Any],
    result: dict[str, Any],
    clock: dict[str, float],
    verified: bool,
    timers: dict[int, Any],
) -> None:
    """Retain failures and charge actual CPU/GPU once after guarded receipt hashing and byte enumeration.

    Parameters
    ----------
    destination : Path
        Fresh local output directory that preserves failures and refuses overwrites.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    admission : dict[str, Any]
        Committed byte identities for executable sources, local inputs and prior accounting.
    result : dict[str, Any]
        Success or retained-failure receipt awaiting independent technical verification.
    clock : dict[str, float]
        Actual whole-stage CPU, wall and GPU-lock accounting; phase timing never resets it.
    verified : bool
        Whether the actual prior ledger has been verified for additive accounting.
    timers : dict[int, Any]
        Active process timer handlers kept live through receipt hashing and byte enumeration.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    import signal

    receipt, size, digest = "summary.json", None, None
    limits = config["resources"]
    try:
        result.update(
            **elapsed(clock),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            gpu_lock_wall_seconds=clock.get("gpu_cost", 0.0),
        )
        shared.write_new(destination / receipt, result)
        digest = sha256_file(destination / receipt)
        size = shared.directory_bytes(destination)
    except Exception as error:
        receipt = "finalization_failure.json"
        result = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_result_valid": False,
            "complete_byte_enumeration": False,
        }
        shared.write_new(destination / receipt, result)
        digest = sha256_file(destination / receipt)
    finally:
        for timer, (signum, previous_handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous_handler)
    measured = elapsed(clock)
    rss, gpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, clock.get("gpu_cost", 0.0)
    gates = {
        "cpu": measured["cpu_seconds"] < limits["cpu_seconds"],
        "wall": measured["wall_seconds"] < limits["wall_seconds"],
        "gpu": gpu < limits["gpu_lock_wall_seconds"],
        "rss": rss < limits["peak_rss_bytes"],
        "output": size is not None and size + 16384 < limits["output_bytes"],
    }
    prior = admission["prior_ledger"]
    shared.write_new(
        destination / "prospective_ledger.json",
        {
            "status": "PASS" if result["status"] == "PASS" and all(gates.values()) else "FAIL",
            "technical_passed": result["status"] == "PASS" and all(gates.values()),
            "stage": config["study_id"],
            "receipt": receipt,
            "receipt_sha256": digest,
            **measured,
            "gpu_wall_seconds": gpu,
            "peak_rss_bytes": rss,
            "resource_gates": gates,
            "prior_accounting_verified": verified,
            "prior_ledger_sha256": prior["sha256"] if verified else None,
            "current_machine_stages": prior["current_machine_stages"] + 1 if verified else None,
            "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"]
            + measured["cpu_seconds"]
            if verified
            else None,
            "current_machine_cumulative_gpu_wall_seconds": prior[
                "current_machine_cumulative_gpu_wall_seconds"
            ]
            + gpu
            if verified
            else None,
            "output_bytes_before_terminal_ledger": size,
            "historical_costs_verified": False,
            "measurement_endpoint": "after guarded receipt hash/bytes before ledger serialization",
            "terminal_bookkeeping_reserve_seconds": 2,
            "terminal_bookkeeping_reserve_bytes": 16384,
        },
    )
    if not all(gates.values()) or result["status"] != "PASS":
        raise RuntimeError("Retained localizer diagnostic technical failure")


def resource_guard(config: dict[str, Any], clock: dict[str, float]) -> None:
    """Enforce live stage/GPU/RSS caps and the conservative admitted storage bound per batch.

    Parameters
    ----------
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.
    clock : dict[str, float]
        Actual whole-stage CPU, wall and GPU-lock accounting; phase timing never resets it.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    limits, cost = config["resources"], elapsed(clock)
    if cost["cpu_seconds"] + 2 >= limits["cpu_seconds"] or cost["wall_seconds"] + 2 >= limits["wall_seconds"]:
        raise TimeoutError("Localizer stage CPU/wall reserve reached")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["peak_rss_bytes"]:
        raise MemoryError("Localizer stage RSS ceiling")
    if clock.get("output_bound", 0) >= limits["output_bytes"]:
        raise ValueError("Localizer prospective retained-file byte bound exceeded")
    if "gpu_lock_start" in clock and "gpu_cost" not in clock:
        if time.perf_counter() - clock["gpu_lock_start"] + 2 >= limits["gpu_lock_wall_seconds"]:
            raise TimeoutError("Localizer GPU-lock reserve reached")
        if torch.cuda.max_memory_allocated(0) >= limits["maximum_gpu_allocated_bytes"]:
            raise MemoryError("Localizer CUDA allocated-memory ceiling")


def run(root: Path, admission_name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Execute the fixed diagnostic after source, predecessor and actual-path qualification.

    Parameters
    ----------
    root : Path
        Source checkout hosting local outputs and read-only-use raw aliases.
    admission_name : str
        Committed repository-relative identity admission filename.
    clock : dict[str, float]
        Actual whole-stage CPU, wall and GPU-lock accounting; phase timing never resets it.

    Returns
    -------
    dict[str, Any]
        Declared transformed values, identities or complete descriptive results.
    """
    config, admission = preflight(admission_name)
    destination = shared.confined(root, config["output_root"])
    destination.mkdir(parents=True, exist_ok=False)
    limits, timers, verified = config["resources"], {}, False
    result: dict[str, Any] = {"study_id": config["study_id"], "status": "started"}

    def guard() -> None:
        resource_guard(config, clock)

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Localizer process timer:{signum}")

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("Prospective CUDA0 visibility is required")
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
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }
        shared.setup_timers(limits, clock, interrupted, timers)
        sources = bridge_runtime.peak_runtime.snapshot(destination, admission, admission_name)
        result["source_sha256"] = sources
        shared.verify_prior(root, admission)
        verified = True
        verify_inputs(admission, guard)
        verify_parent_manifests(config, admission)
        structure = close_saved_studies(destination, config, guard)
        profile_directory = destination / "profile"
        profile_directory.mkdir()
        with gpu_runtime.measured_gpu_lock(clock):
            clock["gpu_lock_start"] = time.perf_counter()
            models, identities, profile_maps = profile(profile_directory, config, clock, guard)
            full_inference(destination, config, models, identities, profile_maps, guard)
            torch.cuda.synchronize()
            gpu_metadata = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            }
            for model in models.values():
                model.cpu()
        statistics, rows = score_outputs(destination, structure, config, guard)
        figures(destination, rows, statistics, structure, config)
        guard()
        verify_inputs(admission, guard)
        for name, digest in sources.items():
            if sha256_file(from_stored(name)) != digest:
                raise ValueError("Localizer executable source changed during execution")
        result.update(
            status="PASS",
            statistics=statistics,
            gpu_metadata=gpu_metadata,
            technical_passed_before_independent_audit=True,
            independent_audit_required=True,
            no_training=True,
            generator_forwards=0,
            optimizer_updates=0,
            original_annotation_stream_reads=0,
            real_record_reads=168,
            parser_records_scored=0,
            clinical_claim=False,
            tensor_state_sha256=identities,
            operation_counts={
                "synthetic_replay_forwards": 12,
                "profile_batch8_forwards": 12,
                "full_batch8_forwards": 240,
                "model_record_rows": len(rows),
            },
            output_sha256={
                path.relative_to(destination).as_posix(): sha256_file(path)
                for path in destination.rglob("*")
                if path.is_file() and not path.is_symlink()
            },
        )
    except Exception as error:
        result.update(
            status="FAIL", error_type=type(error).__name__, error=str(error), scientific_result_valid=False
        )
    finally:
        finish(destination, config, admission, result, clock, verified, timers)
    return result
