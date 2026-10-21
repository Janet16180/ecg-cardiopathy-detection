"""Independently audit saved-control hotspot diagnosis without any model execution."""

from __future__ import annotations

import argparse
import csv
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

from ecg_experiment.ssl_synthetic_control_diagnosis_audit import (
    MODES,
    seed_values,
    summarize_rows,
    top_selection,
)
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_randomized_saved_audit import (
    control_rows,
    score_rows,
)
from ecg_experiment.ssl_synthetic_randomized_saved_audit import (
    summarize_rows as summarize_predecessor,
)
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

RELATIVE = "outputs/ssl_synthetic_localization/generated_control_diagnosis_v1"
PARENT = "outputs/ssl_synthetic_localization/randomized_localization_v1"
CONFIG_NAME = "configs/ssl_synthetic_control_diagnosis_v1.json"
ARMS = ("supervised", "permuted")
INPUT_NAMES = {
    "observed.npy",
    "support.npy",
    "spatial_prior.npy",
    "supervised/evaluation_probabilities.npy",
    "permuted/evaluation_probabilities.npy",
    "supervised/evaluation_metrics.json",
    "permuted/evaluation_metrics.json",
    "summary.json",
    "prospective_ledger.json",
    "independent_audit_v1/summary.json",
    "independent_audit_v1/prospective_ledger.json",
}
PRIOR_LEDGER_SHA = "383ab5707ecfeb28773d0617c7aaa8b8875cc50d77fa17bd82c1db299cf8f1a5"
CORE_HASHES = dict(
    zip(
        (
            "ecg_experiment/ssl_synthetic_control_diagnosis_audit.py",
            "ecg_experiment/synthetic_audit_io.py",
            "ecg_experiment/ssl_synthetic_randomized_saved_audit.py",
            "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
            "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
            "ecg_experiment/intervals.py",
        ),
        (
            "219c4cd8a8ee2ee298b9dac49ce7ea11bc1a736d2b1500640999a7a6e455ce4b",
            "f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a",
            "8a2c6a019691181a7b805b50da8215a7a16406fa23f8643a6b1075f6ecb53aad",
            "14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6",
            "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
        ),
        strict=True,
    )
)


def bindings(
    source_root: Path, study: Path, parent: Path, main: dict[str, Any], config: dict[str, Any], guard: Any
) -> int:
    """Bind only admitted predecessor files, authored sources and new output artifacts.

    Parameters
    ----------
    source_root, study, parent : Path
        Frozen source tree and authorized artifact roots.
    main, config : dict
        Admitted main summary and its frozen prospective config.
    guard : callable
        Deadline/byte check.

    Returns
    -------
    int
        Count of verified bindings.
    """
    require(
        set(main["input_sha256"]) == set(config["predecessor_sha256"]) == INPUT_NAMES,
        "exact eleven-file predecessor allowlist",
    )
    require(main["input_sha256"] == config["predecessor_sha256"], "frozen predecessor declarations")
    count = 0
    for name, expected in main["source_sha256"].items():
        value = Path(name)
        require(
            value.parts[0] in {"configs", "docs", "ecg_experiment", "scripts", "tests"}
            and value.suffix in {".py", ".json", ".md"},
            "authored text source only",
        )
        for path in (relative_file(source_root, name), relative_file(study / "source_snapshot", name)):
            require(file_sha256(path, guard) == expected, f"source binding {name}")
            count += 1
    for name, expected in main["input_sha256"].items():
        require(file_sha256(relative_file(parent, name), guard) == expected, f"admitted input binding {name}")
        count += 1
    for name, expected in main["output_sha256"].items():
        require(file_sha256(relative_file(study, name), guard) == expected, f"new output binding {name}")
        count += 1
    return count


def arrays_from_parent(parent: Path) -> dict[str, Any]:
    """Expose evaluation-only signal/mask slices and declared maps/prior.

    Parameters
    ----------
    parent : Path
        Read-only saved learning study directory.

    Returns
    -------
    dict[str, Any]
        Only the approved evaluation numeric views and fixed TRAIN aggregate.
    """
    observed = np.load(parent / "observed.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(parent / "support.npy", mmap_mode="r", allow_pickle=False)
    require(
        observed.shape == (128, 4, 12, 5000) and observed.dtype == np.float32, "observed full-file geometry"
    )
    require(support.shape == observed.shape and support.dtype == bool, "support full-file geometry")
    result = {
        "observed": observed[80:128],
        "support": support[80:128],
        "prior": np.load(parent / "spatial_prior.npy", allow_pickle=False),
        "maps": {
            arm: np.load(parent / arm / "evaluation_probabilities.npy", mmap_mode="r", allow_pickle=False)
            for arm in ARMS
        },
    }
    require(
        result["prior"].shape == (12, 5000)
        and result["prior"].dtype == np.float64
        and bool(np.isfinite(result["prior"]).all()),
        "prior aggregate geometry/finite",
    )
    require(bool(np.isfinite(result["observed"]).all()), "evaluation-only observed finite")
    require(not bool(result["support"][:, :2].any()), "evaluation control targets zero")
    for values in result["maps"].values():
        require(values.shape == (48, 4, 12, 5000) and values.dtype == np.float32, "evaluation-map geometry")
        require(
            bool(np.isfinite(values).all()) and bool(((values >= 0) & (values <= 1)).all()),
            "evaluation-map finite probability range",
        )
    return result


def predecessor_closure(
    parent: Path, study: Path, main: dict[str, Any], arrays: dict[str, Any], guard: Any
) -> int:
    """Independently reproduce all old case/control/contrast metrics without main scorers.

    Parameters
    ----------
    parent, study : Path
        Old read-only study and new diagnostic output directory.
    main : dict
        New admitted diagnostic summary.
    arrays : dict
        Evaluation-only approved views.
    guard : callable
        Audit deadline check.

    Returns
    -------
    int
        Independently checked predecessor scalar count.
    """
    previous = read_json(parent / "summary.json")
    audit = read_json(parent / "independent_audit_v1/summary.json")
    require(
        previous["all_passed"] is True and audit["all_passed"] is True and audit["mismatch_count"] == 0,
        "passed independently audited learning predecessor",
    )
    rows, checks = {}, 0
    for arm in ARMS:
        guard()
        values = score_rows(arrays["maps"][arm], arrays["observed"], arrays["support"], arrays["prior"])
        checks += compare_values(
            {"cases": values, "controls": control_rows(arrays["maps"][arm])},
            read_json(parent / arm / "evaluation_metrics.json"),
            f"old {arm} metrics",
        )
        rows[arm] = values
    summary = summarize_predecessor(rows["supervised"], rows["permuted"])
    for key, expected in summary.items():
        checks += compare_values(expected, previous[key], f"old {key}")
    counts = {
        "profile_update_attempts": 10,
        "profile_updates": 10,
        "profile_inference_attempts": 2,
        "profile_inferences": 2,
    }
    for arm in ARMS:
        counts.update(
            {
                arm + suffix: number
                for suffix, number in (
                    ("_updates", 400),
                    ("_update_attempts", 400),
                    ("_inferences", 24),
                    ("_inference_attempts", 24),
                )
            }
        )
    checks += compare_values(counts, previous["counts"], "old operation counters")
    expected = {"all_exact": True, "metric_case_rows": 96, "counts": counts, **summary}
    checks += compare_values(
        expected, read_json(study / "predecessor_closure.json"), "production closure file"
    )
    checks += compare_values(expected, main["predecessor_closure"], "production closure embedding")
    return checks


def flatten_independent(row: dict[str, Any], arm: str, mode: str) -> dict[str, Any]:
    """Build an explicit independent scalar CSV row in its declared column order.

    Parameters
    ----------
    row : dict
        Independently reconstructed seed row.
    arm, mode : str
        Declared model and variant.

    Returns
    -------
    dict[str, Any]
        Flat scalar view, with empty positive fields on controls.
    """
    variant = row["arms"][arm]["variants"][mode]
    result = {
        "index": row["index"],
        "seed": row["seed"],
        "common_shift": row["common_shift"],
        "arm": arm,
        "variant": mode,
    }
    for key in (
        "activity_precision",
        "shifted_activity_precision",
        "activity_enrichment",
        "activity_occupancy",
    ):
        result[key] = variant[key]
    for key in ("activity_occupancy_per_lead", "lead_counts", "lead_fractions"):
        for lead in range(12):
            result[f"{key}_{lead}"] = variant[key][lead]
    for key in ("prior_precision", "shifted_prior_precision", "prior_enrichment"):
        result[key] = variant[key]
    for key in ("score", "strictly_above_count", "equal_cutoff_count", "selected_tied_count"):
        result["cutoff_" + key] = variant["cutoff"][key]
    positive = row["arms"][arm]["positives"].get(mode, {})
    for key in (
        "target_precision",
        "target_recall",
        "nuisance_target_precision",
        "precision_response",
        "nuisance_hotspot_persistence",
    ):
        result[key] = positive.get(key)
    result["control_persistence"] = row["arms"][arm]["control_persistence"]
    result["primary_activity_enrichment"] = row["arms"][arm]["primary_activity_enrichment"]
    return result


def check_csv(path: Path, rows: list[dict[str, Any]], main: dict[str, Any]) -> int:
    """Verify all 384 CSV rows against independent numeric/empty-field expectations.

    Parameters
    ----------
    path : Path
        Declared new CSV output.
    rows : list
        Independently reconstructed JSON rows.
    main : dict
        New main metadata with declared columns/count.

    Returns
    -------
    int
        Scalar CSV checks.
    """
    expected = [flatten_independent(row, arm, mode) for row in rows for arm in ARMS for mode in MODES]
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        actual = list(reader)
        require(reader.fieldnames == list(expected[0]) == main["csv_columns"], "explicit CSV columns/order")
    require(len(actual) == len(expected) == main["csv_rows"] == 384, "all 384 CSV rows")
    checks = 0
    for position, (target, saved) in enumerate(zip(expected, actual, strict=True)):
        for key, value in target.items():
            field = saved[key]
            if value is None:
                require(field == "", "empty positive fields on controls")
            elif isinstance(value, str):
                require(field == value, "CSV categorical order")
            elif isinstance(value, int):
                require(int(field) == value, "CSV integer identity")
            else:
                compare_values(float(value), float(field), f"CSV{position}/{key}")
            checks += 1
    return checks


def verify_metrics(study: Path, main: dict[str, Any], arrays: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Recompute every diagnostic row, mean and primary interval from saved arrays.

    Parameters
    ----------
    study : Path
        New diagnosis output directory.
    main : dict
        Admitted main summary.
    arrays : dict
        Evaluation-only saved numeric inputs.
    guard : callable
        Audit deadline check.

    Returns
    -------
    dict[str, Any]
        Independent diagnostic aggregates and check counts.
    """
    saved = [json.loads(line) for line in (study / "cases.jsonl").read_text().splitlines()]
    require(len(saved) == main["case_rows"] == 48, "complete 48-seed diagnostic rows")
    rows, checks = [], 0
    for offset, index in enumerate(range(80, 128)):
        guard()
        row = seed_values(
            {arm: arrays["maps"][arm][offset] for arm in ARMS},
            arrays["observed"][offset],
            arrays["support"][offset],
            arrays["prior"],
            index,
        )
        checks += compare_values(row, saved[offset], f"diagnostic seed{index}")
        rows.append(row)
    aggregate = summarize_rows(rows)
    for key, expected in aggregate.items():
        checks += compare_values(expected, main[key], f"diagnostic {key}")
    checks += compare_values(top_selection(arrays["prior"])[1], main["prior_cutoff"], "fixed prior cutoff")
    csv_checks = check_csv(study / "cases.csv", rows, main)
    return {
        **aggregate,
        "prior_cutoff": top_selection(arrays["prior"])[1],
        "diagnostic_scalar_checks": checks,
        "csv_scalar_checks": csv_checks,
    }


def verify_resources(study: Path, parent: Path, main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Verify terminal costs, strict stage ceilings and one-time additive accounting.

    Parameters
    ----------
    study, parent : Path
        New diagnostic and old read-only learning roots.
    main : dict
        Admitted diagnostic main receipt.
    guard : callable
        Audit deadline check.

    Returns
    -------
    dict[str, Any]
        Verified terminal main/prior costs and exact byte totals.
    """
    require(0 < main["cpu_seconds"] < 180 and 0 < main["wall_seconds"] < 240, "main CPU/wall caps")
    ledger = read_json(study / "prospective_ledger.json")
    require(
        ledger["receipt"] == "summary.json"
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json", guard),
        "main ledger receipt binding",
    )
    require(main["cpu_seconds"] <= ledger["cpu_seconds"] < 180, "terminal main CPU cap")
    require(main["wall_seconds"] <= ledger["wall_seconds"] < 240, "terminal main wall cap")
    require(ledger["gpu_wall_seconds"] == 0, "zero diagnostic GPU wall")
    require(
        ledger["cpu_budget_passed"] is True and ledger["wall_budget_passed"] is True,
        "terminal main budget flags",
    )
    require(
        ledger["prior_accounting_verified"] is True and ledger["prior_ledger_sha256"] == PRIOR_LEDGER_SHA,
        "verified literal predecessor terminal ledger",
    )
    previous = parent / "independent_audit_v1/prospective_ledger.json"
    require(file_sha256(previous, guard) == PRIOR_LEDGER_SHA, "predecessor cumulative identity")
    prior = read_json(previous)
    old_cpu = prior["current_machine_cumulative_cpu_seconds"]
    old_gpu = prior["current_machine_cumulative_gpu_wall_seconds"]
    require(
        old_cpu == 32.489216225 and old_gpu == 38.433311437256634, "frozen current-machine predecessor totals"
    )
    require(
        ledger["prior_current_machine_cpu_seconds"] == old_cpu
        and ledger["prior_current_machine_gpu_wall_seconds"] == old_gpu,
        "prior ledger scalar identity",
    )
    require(
        ledger["current_machine_cumulative_cpu_seconds"] == old_cpu + ledger["cpu_seconds"],
        "main CPU added once",
    )
    require(ledger["current_machine_cumulative_gpu_wall_seconds"] == old_gpu, "cumulative GPU unchanged")
    compare_values(
        {"cpu_seconds": 180, "wall_seconds": 240, "bytes": 32 * 1024**2},
        ledger["reserved_independent_audit"],
        "separate prospective audit allowance",
    )
    actual = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file() and "independent_audit_v1" not in path.parts
    )
    receipts = sum((study / name).stat().st_size for name in ("summary.json", "prospective_ledger.json"))
    require(actual - receipts == main["output_bytes_before_receipt"], "main byte accounting before receipt")
    require(
        actual - (study / "prospective_ledger.json").stat().st_size
        == ledger["output_bytes_before_terminal_ledger"],
        "main bytes before terminal ledger",
    )
    require(actual < 32 * 1024**2, "main separate byte cap")
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": 0.0,
        "main_bytes": actual,
        "main_ledger_sha256": file_sha256(study / "prospective_ledger.json", guard),
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_seconds": old_gpu,
    }


def verify_boundaries(main: dict[str, Any], study: Path) -> None:
    """Verify descriptive scope, zero new model work and fixed figure artifacts.

    Parameters
    ----------
    main : dict
        Admitted main receipt.
    study : Path
        New sibling output directory.
    """
    require(
        main["integrity_completed"] is True and main["descriptive_only"] is True,
        "descriptive integrity completion",
    )
    require("scientific_passed" not in main and "scientific_gates" not in main, "no new scientific verdict")
    require(main["evaluation_indices"] == list(range(80, 128)), "all 48 spent evaluation indices")
    require(main["torch_threads"] == main["torch_interop_threads"] == 1, "single main PyTorch threads")
    require(main["cuda_visible_devices"] == "" and main["gpu_used"] is False, "CUDA hidden and GPU unused")
    require(all(value == "1" for value in main["thread_environment"].values()), "single numerical threads")
    for name in (
        "new_generator_forwards",
        "new_model_forwards",
        "new_optimizer_updates",
        "real_waveform_reads",
        "external_api_calls",
    ):
        require(main[name] == 0, f"zero {name}")
    require(main["clinical_claim"] is False, "no clinical claim")
    figures = {"fixed_control_hotspots.png": (2200, 2200), "diagnostic_summary.png": (1920, 480)}
    require(main["figures"] == list(figures), "fixed two-figure catalog")
    for name, shape in figures.items():
        with (study / name).open("rb") as stream:
            header = stream.read(24)
        require(header[:8] == b"\x89PNG\r\n\x1a\n", "fixed PNG signature")
        require(struct.unpack(">II", header[16:24]) == shape, "fixed figure dimensions")


def terminal_ledger(
    destination: Path, receipt: str, clock: tuple[float, float], costs: dict[str, Any]
) -> None:
    """Measure own terminal audit cost after its receipt write/hash and add it once.

    Parameters
    ----------
    destination : Path
        Fresh audit directory.
    receipt : str
        Saved success/failure receipt filename.
    clock : tuple
        CPU and monotonic start values.
    costs : dict
        Verified main cumulative costs, absent if admission failed.
    """
    digest = file_sha256(destination / receipt)
    cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
    byte_count = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    with (destination / "prospective_ledger.json").open("x") as stream:
        json.dump(
            {
                "receipt": receipt,
                "sha256": digest,
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "gpu_wall_seconds": 0.0,
                "main_cpu_seconds": costs.get("main_cpu_seconds"),
                "stage_main_plus_audit_cpu_seconds": costs["main_cpu_seconds"] + cpu if costs else None,
                "current_machine_cumulative_cpu_seconds": costs["current_machine_cpu_through_main"] + cpu
                if costs
                else None,
                "current_machine_cumulative_gpu_wall_seconds": costs.get("current_machine_gpu_wall_seconds"),
                "historical_costs_verified": False,
                "audit_budget_passed": cpu < 180 and wall < 240,
                "audit_bytes_before_terminal_ledger": byte_count,
                "audit_output_budget_passed": byte_count + 1024**2 < 32 * 1024**2,
                "measurement_endpoint": "after audit receipt write/hash, before ledger serialization",
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def run(root: Path, source_root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Execute one source-reviewed copied checker within the frozen separate audit allowance.

    Parameters
    ----------
    root, source_root : Path
        Output host and committed source worktree.
    main_sha, script_sha : str
        Explicit coordinator-admitted SHA-256 bindings.

    Returns
    -------
    dict[str, Any]
        Independent descriptive arithmetic/integrity result.
    """
    clock = time.process_time(), time.perf_counter()
    study, parent = root / RELATIVE, root / PARENT
    destination = study / "independent_audit_v1"
    require(
        Path(__file__).absolute() == destination / "audit_script.py", "execute exact copied audit location"
    )
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit previously attempted",
    )
    receipt = "summary.json"
    costs: dict[str, Any] = {}

    def guard() -> None:
        require(time.process_time() - clock[0] < 178, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 238, "audit wall deadline")
        actual = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        require(actual < 31 * 1024**2, "audit one-MiB receipt reserve")

    def interrupt(signum: int, frame: Any) -> None:
        raise TimeoutError(f"diagnostic audit timer {signum}")

    for timer in (signal.ITIMER_REAL, signal.ITIMER_PROF):
        require(signal.getitimer(timer) == (0, 0), "exclusive audit deadline timer")
    for timer, sig, seconds in (
        (signal.ITIMER_REAL, signal.SIGALRM, 238),
        (signal.ITIMER_PROF, signal.SIGPROF, 178),
    ):
        signal.signal(sig, interrupt)
        signal.setitimer(timer, seconds)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "audit CUDA hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single audit numerical thread")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(file_sha256(Path(__file__), guard) == script_sha, "admitted checker bytes")
        require(len(CORE_HASHES) == 6, "complete independent source catalog")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "independent source bytes")
        require(file_sha256(study / "summary.json", guard) == main_sha, "admitted main bytes")
        main = read_json(study / "summary.json")
        config = read_json(study / "source_snapshot" / CONFIG_NAME)
        verified = bindings(source_root, study, parent, main, config, guard)
        costs = verify_resources(study, parent, main, guard)
        verify_boundaries(main, study)
        arrays = arrays_from_parent(parent)
        closure_checks = predecessor_closure(parent, study, main, arrays, guard)
        diagnostic = verify_metrics(study, main, arrays, guard)
        require(
            bindings(source_root, study, parent, main, config, guard) == verified,
            "bindings unchanged after audit",
        )
        require(file_sha256(study / "summary.json", guard) == main_sha, "main unchanged after audit")
        require(
            file_sha256(study / "prospective_ledger.json", guard) == costs["main_ledger_sha256"],
            "main ledger unchanged",
        )
        require(file_sha256(Path(__file__), guard) == script_sha, "checker unchanged")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "independent core unchanged")
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "descriptive_only": True,
            **diagnostic,
            "predecessor_scalar_checks": closure_checks,
            "bindings_verified": verified,
            "resources": costs,
            "main_sha256": main_sha,
            "script_sha256": script_sha,
            "core_sha256": CORE_HASHES,
            "numeric_observed_support_indices": list(range(80, 128)),
            "whole_file_hashing_of_allowlisted_arrays": True,
            "numeric_training_rows_read": 0,
            "checkpoint_reads": 0,
            "native_generator_array_reads": 0,
            "raw_real_ecg_reads": 0,
            "new_model_forwards": 0,
            "new_generator_forwards": 0,
            "new_optimizer_updates": 0,
            "gpu_used": False,
            "clinical_claim": False,
            "fixed_figure_renderings_source_bound": True,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cpu_seconds": time.process_time() - clock[0],
            "wall_seconds": time.perf_counter() - clock[1],
        }
        with (destination / receipt).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        receipt = "failure.json"
        with (destination / receipt).open("x") as stream:
            json.dump(
                {
                    "all_passed": False,
                    "failure_type": type(error).__name__,
                    "error": str(error),
                    "main_sha256": main_sha,
                    "script_sha256": script_sha,
                    "cpu_seconds": time.process_time() - clock[0],
                    "wall_seconds": time.perf_counter() - clock[1],
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        raise
    finally:
        terminal_ledger(destination, receipt, clock, costs)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse explicit copied-script and admitted main identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
