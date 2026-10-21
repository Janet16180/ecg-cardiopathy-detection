"""Diagnose saved generated-background hotspot maps without any model execution."""

from __future__ import annotations

import csv
import json
import os
import resource
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_randomized_localization as learning
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.paths import from_stored, to_stored
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

MODES = ("sham", "nuisance", "focal", "repeated")
ARMS = ("supervised", "permuted")
CONFIG = "configs/ssl_synthetic_control_diagnosis_v1.json"
SOURCES = (
    CONFIG,
    "docs/ssl-synthetic-localization-control-diagnosis-v1.md",
    "ecg_experiment/ssl_synthetic_control_diagnosis.py",
    "scripts/reports/diagnose_ssl_synthetic_controls.py",
    "tests/test_ssl_synthetic_control_diagnosis.py",
    "ecg_experiment/ssl_synthetic_randomized_localization.py",
    "ecg_experiment/ssl_synthetic_generated_quality.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/synthetic_audit_io.py",
    "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
    "ecg_experiment/ssl_synthetic_pretrained_generator.py",
)


def ranked_mask(values: np.ndarray, budget: int = 600) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a fixed ranked area with ascending flattened indices breaking exact ties.

    Parameters
    ----------
    values : np.ndarray
        Finite numerical ranking scores in canonical C-order lead-time layout.
    budget : int
        Exact number of selected cells.

    Returns
    -------
    tuple
        Boolean selected cells and cutoff/tie counts.
    """
    if not np.isfinite(values).all() or not 0 < budget <= values.size:
        raise ValueError("Rank scores must be finite and the fixed budget must fit")
    flat = values.ravel(order="C")
    selected = np.argsort(-flat, kind="stable")[:budget]
    result = np.zeros(flat.shape, dtype=bool)
    result[selected] = True
    cutoff = flat[selected[-1]]
    above = int(np.count_nonzero(flat > cutoff))
    return result.reshape(values.shape), {
        "score": float(cutoff),
        "strictly_above_count": above,
        "equal_cutoff_count": int(np.count_nonzero(flat == cutoff)),
        "selected_tied_count": int(budget - above),
    }


def activity_mask(observed: np.ndarray) -> np.ndarray:
    """Apply the prospective strict upper-decile derivative-energy activity rule.

    Parameters
    ----------
    observed : np.ndarray
        Canonical twelve-lead float32 observed waveform in robust-scale units.

    Returns
    -------
    np.ndarray
        Boolean activity by lead and sample, without physiological labels.
    """
    if observed.shape != (12, 5000) or observed.dtype != np.float32 or not np.isfinite(observed).all():
        raise ValueError("Activity requires finite canonical float32 (12,5000) values")
    derivative = np.diff(observed.astype(np.float64), axis=1, prepend=observed[:, :1])
    energy = np.stack([np.convolve(row**2, np.ones(20) / 20, mode="same") for row in derivative])
    return energy > np.quantile(energy, 0.9, axis=1, method="linear")[:, None]


def variant_values(hotspot: np.ndarray, active: np.ndarray, prior: np.ndarray, shift: int) -> dict[str, Any]:
    """Measure fixed-area activity, lead allocation and saved spatial-prior alignment.

    Parameters
    ----------
    hotspot, active, prior : np.ndarray
        Canonical boolean hotspot, derivative activity and fixed ranked prior.
    shift : int
        Common prospective temporal circular shift.

    Returns
    -------
    dict
        Descriptive overlap fractions, occupancy and per-lead allocations.
    """
    activity = float(np.count_nonzero(hotspot & active) / 600)
    shifted = float(np.count_nonzero(hotspot & np.roll(active, shift, axis=1)) / 600)
    prior_precision = float(np.count_nonzero(hotspot & prior) / 600)
    shifted_prior = float(np.count_nonzero(hotspot & np.roll(prior, shift, axis=1)) / 600)
    lead_counts = hotspot.sum(axis=1)
    return {
        "activity_precision": activity,
        "shifted_activity_precision": shifted,
        "activity_enrichment": activity - shifted,
        "activity_occupancy": float(active.mean()),
        "activity_occupancy_per_lead": active.mean(axis=1).tolist(),
        "lead_counts": lead_counts.tolist(),
        "lead_fractions": (lead_counts / 600).tolist(),
        "prior_precision": prior_precision,
        "shifted_prior_precision": shifted_prior,
        "prior_enrichment": prior_precision - shifted_prior,
    }


def diagnose_case(
    probabilities: dict[str, np.ndarray],
    observed: np.ndarray,
    support: np.ndarray,
    prior: np.ndarray,
    index: int,
) -> dict[str, Any]:
    """Measure one evaluation seed using a common activity shift across both arms.

    Parameters
    ----------
    probabilities : dict
        Each arm's four saved evaluation maps for this seed.
    observed, support : np.ndarray
        Four evaluation variants and their separate introduced-change masks.
    prior : np.ndarray
        Fixed top-600 training spatial-prior mask.
    index : int
        Global evaluation background index, from 80 through 127.

    Returns
    -------
    dict
        Complete per-arm, per-variant and positive-mode diagnostic rows.
    """
    if index not in range(80, 128) or observed.shape != (4, 12, 5000) or support.shape != observed.shape:
        raise ValueError("Diagnosis may use the fixed 48 evaluation backgrounds only")
    shift = int(np.random.default_rng(2026103600 + index).integers(500, 4501))
    activity = [activity_mask(values) for values in observed]
    result = {"index": index, "seed": 2026103000 + index, "common_shift": shift, "arms": {}}
    for arm in ARMS:
        values = probabilities[arm]
        if (
            values.dtype != np.float32
            or values.shape != (4, 12, 5000)
            or not ((values >= 0) & (values <= 1)).all()
        ):
            raise ValueError("Saved maps must have finite float32 probability geometry")
        hotspots, variants = [], {}
        for position, mode in enumerate(MODES):
            hotspot, cutoff = ranked_mask(values[position])
            hotspots.append(hotspot)
            variants[mode] = {**variant_values(hotspot, activity[position], prior, shift), "cutoff": cutoff}
        positives = {}
        for position, mode in [(2, "focal"), (3, "repeated")]:
            target = support[position]
            if not target.any():
                raise ValueError("Evaluation positive mask is empty")
            precision = float(np.count_nonzero(hotspots[position] & target) / 600)
            nuisance_precision = float(np.count_nonzero(hotspots[1] & target) / 600)
            positives[mode] = {
                "target_precision": precision,
                "target_recall": float(np.count_nonzero(hotspots[position] & target) / target.sum()),
                "nuisance_target_precision": nuisance_precision,
                "precision_response": precision - nuisance_precision,
                "nuisance_hotspot_persistence": float(
                    np.count_nonzero(hotspots[position] & hotspots[1]) / 600
                ),
            }
        result["arms"][arm] = {
            "variants": variants,
            "positives": positives,
            "control_persistence": float(np.count_nonzero(hotspots[0] & hotspots[1]) / 600),
            "primary_activity_enrichment": float(
                (variants["sham"]["activity_enrichment"] + variants["nuisance"]["activity_enrichment"]) / 2
            ),
        }
    return result


def primary_interval(values: np.ndarray) -> dict[str, Any]:
    """Bootstrap the 48 fixed whole-seed values using the frozen paired draws."""
    if values.shape != (48,) or not np.isfinite(values).all():
        raise ValueError("The primary comparison requires all 48 finite seed values")
    groups = patient_groups(np.arange(48))
    rng = np.random.default_rng(2026103701)
    draws = [float(values[patient_resample(groups, rng)].mean()) for _ in range(2000)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 2026103701,
        "skipped_draws": 0,
        "unit": "generator_seed",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the three primary intervals and descriptive secondary seed means."""
    values = {
        arm: np.array([row["arms"][arm]["primary_activity_enrichment"] for row in rows]) for arm in ARMS
    }
    primary = {arm: primary_interval(value) for arm, value in values.items()}
    primary["supervised_minus_permuted"] = primary_interval(values["supervised"] - values["permuted"])
    means = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows]
        variants = {}
        for mode in MODES:
            variant_rows = [row["variants"][mode] for row in arm_rows]
            keys = [key for key in variant_rows[0] if key != "cutoff"]
            variants[mode] = {
                key: np.mean([row[key] for row in variant_rows], axis=0).tolist() for key in keys
            }
        positives = {
            mode: {
                key: float(np.mean([row["positives"][mode][key] for row in arm_rows]))
                for key in arm_rows[0]["positives"][mode]
            }
            for mode in ["focal", "repeated"]
        }
        means[arm] = {
            "variants": variants,
            "positives": positives,
            "control_persistence": float(np.mean([row["control_persistence"] for row in arm_rows])),
        }
    return {"primary": primary, "secondary_means": means}


def flatten_case(row: dict[str, Any], arm: str, mode: str) -> dict[str, Any]:
    """Flatten one arm/variant row without dropping cutoff, lead or positive-case fields."""
    arm_row = row["arms"][arm]
    result = {key: row[key] for key in ["index", "seed", "common_shift"]}
    result.update(arm=arm, variant=mode)
    for name, value in arm_row["variants"][mode].items():
        if isinstance(value, list):
            result.update({f"{name}_{index}": entry for index, entry in enumerate(value)})
        elif isinstance(value, dict):
            result.update({f"{name}_{key}": entry for key, entry in value.items()})
        else:
            result[name] = value
    for key in [
        "target_precision",
        "target_recall",
        "nuisance_target_precision",
        "precision_response",
        "nuisance_hotspot_persistence",
    ]:
        result[key] = arm_row["positives"].get(mode, {}).get(key)
    result["control_persistence"] = arm_row["control_persistence"]
    result["primary_activity_enrichment"] = arm_row["primary_activity_enrichment"]
    return result


def write_cases(destination: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Save complete per-seed JSON and a deterministic 384-row CSV view."""
    with (destination / "cases.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    flattened = [flatten_case(row, arm, mode) for row in rows for arm in ARMS for mode in MODES]
    columns = list(flattened[0])
    with (destination / "cases.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flattened)
    return columns


def make_figures(
    destination: Path,
    maps: dict[str, np.ndarray],
    observed: np.ndarray,
    support: np.ndarray,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> list[str]:
    """Render the fixed first-eight control gallery and aggregate diagnostic figure.

    Parameters
    ----------
    destination : Path
        Fresh local diagnostic directory.
    maps : dict
        Both arms' saved evaluation maps.
    observed, support : np.ndarray
        Evaluation waveforms and introduced-change masks, with no training rows.
    rows : list
        Per-evaluation-seed diagnostic values.
    aggregate : dict
        Primary intervals and descriptive secondary means.

    Returns
    -------
    list
        Two fixed PNG artifact filenames.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(8, 4, figsize=(20, 20))
    for index in range(8):
        for column, arm, position in [
            (0, "supervised", 0),
            (1, "supervised", 1),
            (2, "permuted", 0),
            (3, "permuted", 1),
        ]:
            hotspot, _ = ranked_mask(maps[arm][index, position])
            active = activity_mask(observed[index, position])
            image = np.zeros((12, 5000, 3), dtype=np.float32)
            image[active] = [0.7, 0.7, 0.7]
            image[hotspot] = [0.8, 0.15, 0.1]
            image[active & hotspot] = [0.15, 0.6, 0.1]
            axes[index, column].imshow(image, aspect="auto", interpolation="nearest", extent=(0, 10, 12, 0))
        axes[index, 0].set_ylabel(f"Seed {2026103080 + index}")
    titles = ["Supervised sham", "Supervised nuisance", "Permuted sham", "Permuted nuisance"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("Seconds")
    fig.suptitle("First eight fixed evaluation seeds: gray activity, red ranked hotspots, green overlap")
    fig.tight_layout()
    fig.savefig(destination / "fixed_control_hotspots.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for arm in ARMS:
        values = [row["arms"][arm]["primary_activity_enrichment"] for row in rows]
        axes[0].plot(np.arange(80, 128), values, marker=".", linewidth=0.5, label=arm)
        lead = np.mean(
            [
                aggregate["secondary_means"][arm]["variants"][mode]["lead_fractions"]
                for mode in ["sham", "nuisance"]
            ],
            axis=0,
        )
        axes[1].plot(np.arange(12), lead, marker=".", label=arm)
        axes[2].hist(
            [row["arms"][arm]["control_persistence"] for row in rows],
            bins=np.linspace(0, 1, 21),
            histtype="step",
            label=arm,
        )
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set(xlabel="Global evaluation index", ylabel="Control activity alignment difference")
    axes[1].set_xticks(np.arange(12), learning.quality.LEADS, rotation=90)
    axes[1].set(ylabel="Control top-600 allocation fraction")
    axes[2].set(xlabel="Sham/nuisance hotspot intersection / 600", ylabel="Seed count")
    for axis in axes:
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "diagnostic_summary.png", dpi=120)
    plt.close(fig)
    return ["fixed_control_hotspots.png", "diagnostic_summary.png"]


def write_new_json(path: Path, value: Any) -> None:
    """Write a new finite JSON receipt without replacing an existing artifact."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(destination: Path, guard: Any) -> dict[str, str]:
    """Save every authored dependency before opening the admitted saved-map payloads."""
    sources = {}
    for name in SOURCES:
        guard()
        path = destination / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(from_stored(name).read_bytes())
        sources[to_stored(from_stored(name))] = file_sha256(path, guard)
    return sources


def load_inputs(parent: Path, config: dict[str, Any], inputs: dict[str, str], guard: Any) -> dict[str, Any]:
    """Hash the explicit allowlist and expose only evaluation slices for diagnostic scoring."""
    for name, expected in config["predecessor_sha256"].items():
        if file_sha256(relative_file(parent, name), guard) != expected:
            raise ValueError("Frozen saved-map predecessor identity mismatch")
        inputs[name] = expected
    main = read_json(parent / "summary.json")
    audit = read_json(parent / "independent_audit_v1/summary.json")
    if not main["all_passed"] or not audit["all_passed"] or audit["mismatch_count"]:
        raise ValueError("The saved learning predecessor did not pass independently")
    helper = "ecg_experiment/ssl_synthetic_randomized_localization.py"
    if file_sha256(from_stored(helper), guard) != main["source_sha256"][helper]:
        raise ValueError("Frozen predecessor metric helper bytes changed")
    observed = np.load(parent / "observed.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(parent / "support.npy", mmap_mode="r", allow_pickle=False)
    if (
        observed.shape != (128, 4, 12, 5000)
        or observed.dtype != np.float32
        or support.shape != observed.shape
        or support.dtype != np.bool_
    ):
        raise ValueError("Admitted observed/support geometry changed")
    maps = {
        arm: np.load(parent / arm / "evaluation_probabilities.npy", mmap_mode="r", allow_pickle=False)
        for arm in ARMS
    }
    for values in maps.values():
        if values.shape != (48, 4, 12, 5000) or values.dtype != np.float32:
            raise ValueError("Admitted saved evaluation map geometry changed")
    prior = np.load(parent / "spatial_prior.npy", allow_pickle=False)
    if prior.shape != (12, 5000) or prior.dtype != np.float64 or not np.isfinite(prior).all():
        raise ValueError("Admitted training-derived spatial-prior geometry changed")
    return {"main": main, "maps": maps, "observed": observed[80:], "support": support[80:], "prior": prior}


def predecessor_closure(
    parent: Path, arrays: dict[str, Any], destination: Path, guard: Any
) -> dict[str, Any]:
    """Exactly reproduce saved AP, controls, contrasts and operation counts before new diagnostics."""
    metrics = {}
    for arm in ARMS:
        guard()
        rows = learning.score_maps(arrays["maps"][arm], arrays["observed"], arrays["support"])
        learning.add_spatial_prior(rows, arrays["support"], arrays["prior"])
        expected = {"cases": rows, "controls": learning.control_scores(arrays["maps"][arm])}
        compare_values(
            expected,
            read_json(parent / arm / "evaluation_metrics.json"),
            arm + " exact prior metrics",
            tolerance=0,
        )
        metrics[arm] = rows
    summary = learning.summarize_metrics(metrics["supervised"], metrics["permuted"])
    for key, value in summary.items():
        compare_values(value, arrays["main"][key], "exact predecessor " + key, tolerance=0)
    counts = {
        "profile_update_attempts": 10,
        "profile_updates": 10,
        "profile_inference_attempts": 2,
        "profile_inferences": 2,
    }
    for arm in ARMS:
        counts.update(
            {
                arm + "_updates": 400,
                arm + "_update_attempts": 400,
                arm + "_inferences": 24,
                arm + "_inference_attempts": 24,
            }
        )
    compare_values(counts, arrays["main"]["counts"], "exact prior operation counts", tolerance=0)
    closure = {"all_exact": True, "metric_case_rows": 96, "counts": counts, **summary}
    write_new_json(destination / "predecessor_closure.json", closure)
    return closure


def configure() -> None:
    """Require hidden CUDA and one numerical/PyTorch thread before any numerical payload access."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Saved-map diagnosis requires CUDA_VISIBLE_DEVICES to be empty")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(name) != "1":
            raise ValueError("Every numerical thread environment setting must equal one")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def elapsed(clock: dict[str, float]) -> dict[str, float]:
    """Measure process CPU and monotonic wall cost from stage admission."""
    return {
        "cpu_seconds": time.process_time() - clock["cpu"],
        "wall_seconds": time.perf_counter() - clock["wall"],
    }


def directory_bytes(path: Path) -> int:
    """Count actual saved file bytes without opening numerical payloads."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def arm_timers(config: dict[str, Any], handlers: dict[int, Any]) -> None:
    """Install exclusive process CPU/wall deadlines with the frozen failure reserve."""

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Saved-map diagnosis deadline reached ({signum})")

    resources = config["resources"]
    for timer in [signal.ITIMER_REAL, signal.ITIMER_PROF]:
        if signal.getitimer(timer) != (0.0, 0.0):
            raise ValueError("Existing process timer prevents exclusive diagnostic admission")
    for timer, sig, limit in [
        (signal.ITIMER_REAL, signal.SIGALRM, resources["wall_seconds"]),
        (signal.ITIMER_PROF, signal.SIGPROF, resources["cpu_seconds"]),
    ]:
        handlers[sig] = signal.signal(sig, interrupted)
        signal.setitimer(timer, limit - resources["failure_reserve_seconds"])


def restore_timers(handlers: dict[int, Any]) -> None:
    """Restore only signal handlers and timers installed by this attempt."""
    for timer, sig in [(signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)]:
        if sig in handlers:
            signal.setitimer(timer, 0)
            signal.signal(sig, handlers[sig])


def verify_bindings(
    destination: Path, parent: Path, sources: dict[str, str], inputs: dict[str, str], guard: Any
) -> None:
    """Rehash all declared current/snapshot source files and allowlisted parent inputs."""
    for name, expected in sources.items():
        if (
            file_sha256(from_stored(name), guard) != expected
            or file_sha256(destination / "source_snapshot" / name, guard) != expected
        ):
            raise ValueError("Saved-map diagnosis source or source snapshot changed")
    for name, expected in inputs.items():
        if file_sha256(relative_file(parent, name), guard) != expected:
            raise ValueError("Allowlisted saved-map input changed")


def finish_ledger(
    destination: Path, parent: Path, config: dict[str, Any], name: str, clock: dict[str, float]
) -> None:
    """Charge the terminal success/failure cost once after the last audited cumulative cost."""
    digest = file_sha256(destination / name)
    prior_path = parent / "independent_audit_v1/prospective_ledger.json"
    expected = config["predecessor_sha256"]["independent_audit_v1/prospective_ledger.json"]
    prior_digest = file_sha256(prior_path) if prior_path.is_file() else None
    prior = read_json(prior_path) if prior_digest == expected else {}
    terminal = elapsed(clock)
    prior_cpu = prior.get("current_machine_cumulative_cpu_seconds")
    prior_gpu = prior.get("current_machine_cumulative_gpu_wall_seconds")
    write_new_json(
        destination / "prospective_ledger.json",
        {
            "stage": "generated_control_diagnosis_v1",
            "receipt": name,
            "receipt_sha256": digest,
            **terminal,
            "gpu_wall_seconds": 0.0,
            "prior_ledger_sha256": prior_digest,
            "prior_accounting_verified": prior_digest == expected,
            "prior_current_machine_cpu_seconds": prior_cpu,
            "prior_current_machine_gpu_wall_seconds": prior_gpu,
            "current_machine_cumulative_cpu_seconds": prior_cpu + terminal["cpu_seconds"]
            if prior_cpu is not None
            else None,
            "current_machine_cumulative_gpu_wall_seconds": prior_gpu,
            "historical_costs_verified": False,
            "cpu_budget_passed": terminal["cpu_seconds"] < config["resources"]["cpu_seconds"],
            "wall_budget_passed": terminal["wall_seconds"] < config["resources"]["wall_seconds"],
            "output_bytes_before_terminal_ledger": directory_bytes(destination),
            "measurement_endpoint": "after receipt write/hash, before terminal ledger serialization",
            "reserved_independent_audit": {"cpu_seconds": 180, "wall_seconds": 240, "bytes": 32 * 1024**2},
        },
    )


def run_diagnosis(output_root: Path) -> dict[str, Any]:
    """Execute the frozen saved-map diagnostic once with no model or optimizer operations.

    Parameters
    ----------
    output_root : Path
        Local output-host repository root containing the audited predecessor.

    Returns
    -------
    dict
        Executed descriptive metrics, closure evidence, identity hashes and costs.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = read_json(from_stored(CONFIG))
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    parent = output_root / config["predecessor_root"]
    resources = config["resources"]
    sources, inputs, rows, handlers = {}, {}, [], {}

    def guard() -> None:
        spent = elapsed(clock)
        if (
            spent["cpu_seconds"] + 2 >= resources["cpu_seconds"]
            or spent["wall_seconds"] + 2 >= resources["wall_seconds"]
        ):
            raise TimeoutError("Saved-map diagnostic CPU/wall deadline reached")
        if directory_bytes(destination) + resources["receipt_reserve_bytes"] >= resources["output_bytes"]:
            raise ValueError("Saved-map diagnostic output receipt reserve exhausted")

    try:
        configure()
        arm_timers(config, handlers)
        sources = snapshot(destination, guard)
        arrays = load_inputs(parent, config, inputs, guard)
        prior_ledger = read_json(parent / "independent_audit_v1/prospective_ledger.json")
        if (
            prior_ledger["current_machine_cumulative_cpu_seconds"]
            != config["predecessor_cumulative_cpu_seconds"]
            or prior_ledger["current_machine_cumulative_gpu_wall_seconds"]
            != config["predecessor_cumulative_gpu_wall_seconds"]
        ):
            raise ValueError("Frozen cumulative predecessor costs changed")
        verify_bindings(destination, parent, sources, inputs, guard)
        closure = predecessor_closure(parent, arrays, destination, guard)
        prior, prior_cutoff = ranked_mask(arrays["prior"])
        for offset, index in enumerate(config["evaluation_indices"]):
            guard()
            rows.append(
                diagnose_case(
                    {arm: values[offset] for arm, values in arrays["maps"].items()},
                    arrays["observed"][offset],
                    arrays["support"][offset],
                    prior,
                    index,
                )
            )
        aggregate = summarize(rows)
        columns = write_cases(destination, rows)
        figures = make_figures(
            destination, arrays["maps"], arrays["observed"], arrays["support"], rows, aggregate
        )
        guard()
        verify_bindings(destination, parent, sources, inputs, guard)
        result = {
            "study_id": config["study_id"],
            "output_relative": to_stored(ROOT / config["output_root"]),
            "predecessor_relative": to_stored(ROOT / config["predecessor_root"]),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "integrity_completed": True,
            "descriptive_only": True,
            "predecessor_closure": closure,
            **aggregate,
            "prior_cutoff": prior_cutoff,
            "case_rows": len(rows),
            "csv_rows": len(rows) * 8,
            "csv_columns": columns,
            "figures": figures,
            "evaluation_indices": config["evaluation_indices"],
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": {
                path.relative_to(destination).as_posix(): file_sha256(path, guard)
                for path in destination.rglob("*")
                if path.is_file()
            },
            **elapsed(clock),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": directory_bytes(destination),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "thread_environment": {
                name: os.environ[name]
                for name in [
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "new_generator_forwards": 0,
            "new_model_forwards": 0,
            "new_optimizer_updates": 0,
            "real_waveform_reads": 0,
            "external_api_calls": 0,
            "gpu_used": False,
            "clinical_claim": False,
            "activity_kind": "strict upper-decile smoothed derivative-energy proxy",
            "support_kind": "introduced numerical changes on generated backgrounds",
            "shift_interpretation": "one fixed alignment reference; not a permutation significance test",
        }
        guard()
        write_new_json(destination / "summary.json", result)
        guard()
        finish_ledger(destination, parent, config, "summary.json", clock)
        return result
    except Exception as error:
        if rows and not (destination / "partial_cases.jsonl").exists():
            with (destination / "partial_cases.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_case_rows": len(rows),
            "source_sha256": sources,
            "input_sha256": inputs,
            **elapsed(clock),
            "gpu_wall_seconds": 0.0,
            "new_generator_forwards": 0,
            "new_model_forwards": 0,
            "new_optimizer_updates": 0,
            "real_waveform_reads": 0,
            "external_api_calls": 0,
            "historical_costs_verified": False,
        }
        write_new_json(destination / "failure.json", failure)
        finish_ledger(destination, parent, config, "failure.json", clock)
        raise
    finally:
        restore_timers(handlers)
