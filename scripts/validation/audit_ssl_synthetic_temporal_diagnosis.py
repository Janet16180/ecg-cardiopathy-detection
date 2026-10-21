"""Independently audit saved-map lead/time decomposition without neural or signal access."""

from __future__ import annotations

import argparse
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

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_quality_saved_audit import average_precision
from ecg_experiment.ssl_synthetic_suppression_saved_audit import suppression_summary
from ecg_experiment.ssl_synthetic_temporal_saved_audit import ARMS, diagnostic_summary, endpoint_rows
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

RELATIVE = "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v1"
CONFIG = "configs/ssl_synthetic_temporal_diagnosis_v1.json"
ADMISSION = "configs/ssl_synthetic_temporal_diagnosis_admission_v1.json"
CORE_HASHES = dict(
    zip(
        (
            "ecg_experiment/ssl_synthetic_temporal_saved_audit.py",
            "ecg_experiment/ssl_synthetic_suppression_saved_audit.py",
            "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
            "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
            "ecg_experiment/ssl_synthetic_control_diagnosis_audit.py",
            "ecg_experiment/synthetic_audit_io.py",
            "ecg_experiment/intervals.py",
            "docs/ssl-synthetic-localization-temporal-diagnosis-independent-audit-v1.md",
            "tests/test_ssl_synthetic_temporal_saved_audit.py",
        ),
        (
            "f74e46cf6d9e1a2a172e3ea218d2e0b2a85f134d0f69b547a48839234d3054fd",
            "55aa6f1cec9edc12ccdc61216376cebd70c6dab8ac7fd382da396ac14ef27ee7",
            "14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6",
            "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "219c4cd8a8ee2ee298b9dac49ce7ea11bc1a736d2b1500640999a7a6e455ce4b",
            "f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a",
            "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
            "f59bdb53da526348dbd76f41ed8a0784e801f1f08df78777af344178a7d3a162",
            "2b85c368dd1b3c742ae08382d271780b50969000a6d50cf3093e7b5a6a7721af",
        ),
        strict=True,
    )
)
INPUT_NAMES = (
    "summary.json",
    "prospective_ledger.json",
    "independent_audit_v1/summary.json",
    "independent_audit_v1/prospective_ledger.json",
    "fresh_support.npy",
    *(
        f"fit_{fit}/{arm}/{name}"
        for fit in (1, 2, 3)
        for arm in ARMS
        for name in ("fresh_probabilities.npy", "fresh_metrics.json")
    ),
)


def arrays_from_parent(parent: Path) -> tuple[np.ndarray, dict[int, Any], dict[int, Any]]:
    """Open only the admitted support, maps and metric records, without signal/state access."""
    support = np.load(parent / "fresh_support.npy", mmap_mode="r", allow_pickle=False)
    require(support.shape == (64, 4, 12, 5000) and support.dtype == bool, "saved support geometry")
    maps, metrics = {}, {}
    for fit in (1, 2, 3):
        maps[fit], metrics[fit] = {}, {}
        for arm in ARMS:
            directory = parent / f"fit_{fit}" / arm
            maps[fit][arm] = np.load(directory / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False)
            require(
                maps[fit][arm].shape == (64, 4, 12, 5000) and maps[fit][arm].dtype == np.float32,
                "saved native map geometry",
            )
            metrics[fit][arm] = read_json(directory / "fresh_metrics.json")
    return support, maps, metrics


def bindings(
    source_root: Path,
    study: Path,
    parent: Path,
    main: dict[str, Any],
    config: dict[str, Any],
    admission: dict[str, Any],
    guard: Any,
) -> int:
    """Bind exactly29 parent files plus immutable source snapshots and new outputs."""
    require(
        set(config["allowed_parent_files"]) == set(INPUT_NAMES) and len(config["allowed_parent_files"]) == 29,
        "exact twenty-nine-file scientific allowlist",
    )
    allowed = admission["parent"]["sha256"]
    require(
        set(allowed) == set(INPUT_NAMES) and main["input_sha256"] == allowed,
        "exact identity-admitted input allowlist",
    )
    count = 0
    for name, expected in allowed.items():
        require(file_sha256(relative_file(parent, name), guard) == expected, "allowed parent binding")
        count += 1
    for name, expected in main["source_sha256"].items():
        for directory in (source_root, study / "source_snapshot"):
            require(file_sha256(relative_file(directory, name), guard) == expected, "source/snapshot binding")
            count += 1
    for name, expected in main["output_sha256"].items():
        require(
            file_sha256(relative_file(study, name), guard) == expected, "immutable diagnostic output binding"
        )
        count += 1
    return count


def parent_closure(
    parent: Path,
    study: Path,
    main: dict[str, Any],
    support: np.ndarray,
    maps: dict[int, Any],
    metrics: dict[int, Any],
    guard: Any,
) -> dict[str, Any]:
    """Independently close every old full-grid AP and all frozen four-arm aggregates first."""
    checks = 0
    for fit in (1, 2, 3):
        for arm in ARMS:
            rows = metrics[fit][arm]["cases"]
            require(len(rows) == 64, "complete parent metric rows")
            for offset, index in enumerate(range(2000, 2064)):
                guard()
                require(
                    rows[offset]["index"] == index and rows[offset]["seed"] == 2026105000 + offset,
                    "old case identities",
                )
                for mode, position in (("focal", 2), ("repeated", 3)):
                    score = average_precision(support[offset, position], maps[fit][arm][offset, position])
                    compare_values(
                        score, rows[offset][mode]["edited_ap"], "independent full-grid predecessor AP"
                    )
                    checks += 1
    aggregate = suppression_summary([{"fit_id": fit, **metrics[fit]} for fit in (1, 2, 3)])
    parent_main = read_json(parent / "summary.json")
    aggregate_checks = compare_values(
        aggregate, parent_main["aggregate"], "independent full parent aggregate"
    )
    closure = {
        "all_exact": True,
        "full_grid_edited_AP_checks": checks,
        "aggregate": aggregate,
        "parent_scientific_passed": aggregate["scientific_passed"],
    }
    compare_values(closure, read_json(study / "predecessor_closure.json"), "saved full predecessor closure")
    compare_values(closure, main["predecessor_closure"], "embedded full predecessor closure")
    require(checks == 1536, "all full-grid positive AP closures")
    require(all(parent_main["integrity_gates"].values()), "parent technical integrity passed")
    audit = read_json(parent / "independent_audit_v1/summary.json")
    require(audit["all_passed"] is True and audit["mismatch_count"] == 0, "parent technical audit passed")
    return {
        "full_grid_edited_AP_checks": checks,
        "aggregate_scalar_checks": aggregate_checks,
        "parent_scientific_passed": aggregate["scientific_passed"],
    }


def metric_evidence(
    study: Path, main: dict[str, Any], support: np.ndarray, maps: dict[int, Any], guard: Any
) -> dict[str, Any]:
    """Reconstruct every true-lead-conditional and lead-presence endpoint with independent AP."""
    rows = []
    for fit in (1, 2, 3):
        for arm in ARMS:
            guard()
            rows.extend(endpoint_rows(maps[fit][arm], support, fit, arm))
    saved = [json.loads(line) for line in (study / "cases.jsonl").read_text().splitlines()]
    checks = compare_values(rows, saved, "complete endpoint case identities and arithmetic")
    summary = diagnostic_summary(rows)
    for key, value in summary.items():
        checks += compare_values(value, main[key], "complete descriptive summaries")
    require(main["case_rows"] == len(rows) == 1536, "all fixed mode rows")
    require("scientific_passed" not in main and "all_passed" not in main, "no new scientific success rule")
    return {"metric_scalar_checks": checks, "descriptive": summary, "case_rows": len(rows)}


def verified_accounting(
    study: Path, parent: Path, main: dict[str, Any], admission: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Capture receipt/prior-bound costs before any later numerical or binding failure."""
    path = study / "prospective_ledger.json"
    ledger = read_json(path)
    require(
        ledger["receipt"] == "summary.json"
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json", guard),
        "main terminal receipt binding",
    )
    prior_path = parent / "independent_audit_v1/prospective_ledger.json"
    require(
        file_sha256(prior_path, guard)
        == ledger["prior_ledger_sha256"]
        == admission["prior_ledger"]["sha256"]
        == "705cc964022f4f1d78d08d7d1eff75a665f02c150e311dacf2e1c4503ede5496",
        "audited suppression ledger identity",
    )
    prior = read_json(prior_path)
    for field, literal in (("cpu_seconds", 390.154950634), ("gpu_wall_seconds", 447.98139213304967)):
        previous = prior[f"current_machine_cumulative_{field}"]
        require(
            previous
            == literal
            == admission["prior_ledger"][f"current_machine_cumulative_{field}"]
            == ledger[f"prior_current_machine_{field}"],
            "literal audited prior costs",
        )
        require(np.isfinite(ledger[field]) and ledger[field] >= 0, "finite measured self costs")
        require(
            ledger[f"current_machine_cumulative_{field}"] == previous + ledger[field],
            "once-only cumulative addition",
        )
    require(
        ledger["gpu_wall_seconds"] == 0 and ledger["cpu_seconds"] >= main["cpu_seconds"],
        "CPU-only self-cost agreement",
    )
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": 0.0,
        "main_ledger_sha256": file_sha256(path, guard),
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def main_resources(
    study: Path, parent: Path, main: dict[str, Any], admission: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Check terminal caps and all output bytes rather than trusting summary snapshots."""
    costs = verified_accounting(study, parent, main, admission, guard)
    ledger_path = study / "prospective_ledger.json"
    ledger = read_json(ledger_path)
    require(
        ledger["prior_accounting_verified"] is True
        and ledger["cpu_budget_passed"] is True
        and ledger["wall_budget_passed"] is True,
        "main terminal accounting/budget flags",
    )
    require(0 <= main["cpu_seconds"] <= costs["main_cpu_seconds"] < 180, "actual terminal main CPU")
    require(0 <= main["wall_seconds"] <= costs["main_wall_seconds"] < 240, "actual terminal main wall")
    require(
        main["gpu_used"] is False and main["CUDA_VISIBLE_DEVICES"] == "", "CPU-only hidden-CUDA execution"
    )
    require(main["torch_threads"] == main["interop_threads"] == 1, "one production intraop/interop thread")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(main["runtime_environment"][key] == "1", "one production numerical thread")
    compare_values(
        dict.fromkeys(
            (
                "generator_forwards",
                "localizer_forwards",
                "optimizer_updates",
                "real_waveform_reads",
                "downloads",
                "external_api_calls",
            ),
            0,
        ),
        main["operations"],
        "zero prohibited operations",
    )
    files = [path for path in study.rglob("*") if path.is_file() and "independent_audit_v1" not in path.parts]
    size = sum(path.stat().st_size for path in files)
    require(size < 16 * 1024**2, "actual main storage ceiling")
    require(
        size - ledger_path.stat().st_size == ledger["output_bytes_before_terminal_ledger"],
        "actual main bytes before terminal ledger",
    )
    require(
        size - ledger_path.stat().st_size - (study / "summary.json").stat().st_size
        == main["output_bytes_before_receipt"],
        "actual main bytes before receipt",
    )
    catalog = {
        path.relative_to(study).as_posix()
        for path in files
        if path.parent != study or path.name not in {"summary.json", "prospective_ledger.json"}
    }
    require(catalog == set(main["output_sha256"]), "complete main file catalog")
    return {**costs, "main_bytes": size, "prior_costs_added_once": True}


def figure(study: Path, names: list[str]) -> None:
    """Verify the single fixed all-arm/all-fit PNG without rendering another figure."""
    require(names == ["lead_and_time_diagnostics.png"], "single fixed diagnostic figure")
    with (study / names[0]).open("rb") as stream:
        header = stream.read(24)
    require(
        header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", header[16:24]) == (1440, 1080),
        "fixed PNG signature/geometry",
    )


def write_ledger(destination: Path, receipt: str, clock: tuple[float, float], costs: dict[str, Any]) -> None:
    """Charge own success/failure cost once after receipt hashing and byte enumeration."""
    digest = file_sha256(destination / receipt)
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
    ledger = {
        "receipt": receipt,
        "sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "main_ledger_sha256": costs.get("main_ledger_sha256"),
        "main_cpu_seconds": costs.get("main_cpu_seconds"),
        "stage_main_plus_audit_cpu_seconds": costs["main_cpu_seconds"] + cpu if costs else None,
        "current_machine_cumulative_cpu_seconds": costs["current_machine_cpu_through_main"] + cpu
        if costs
        else None,
        "current_machine_cumulative_gpu_wall_seconds": costs.get("current_machine_gpu_wall_through_main"),
        "audit_cpu_budget_passed": cpu < 180,
        "audit_wall_budget_passed": wall < 240,
        "audit_bytes_before_terminal_ledger": size,
        "historical_costs_verified": False,
        "measurement_endpoint": "after receipt hash/bytes, before terminal ledger assembly/serialization",
    }
    with (destination / "prospective_ledger.json").open("x") as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(
        sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 16 * 1024**2,
        "terminal audit byte cap",
    )
    require(cpu < 180 and wall < 240, "terminal audit time caps")


def run(root: Path, source_root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Run one copied admitted checker with exact allowed-map access and no neural work."""
    clock = time.process_time(), time.perf_counter()
    study = root / RELATIVE
    destination = study / "independent_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "execute exact copied checker")
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit already attempted",
    )
    costs: dict[str, Any] = {}
    receipt = "summary.json"

    def guard() -> None:
        require(time.process_time() - clock[0] < 178, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 238, "audit wall deadline")
        require(
            sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 15 * 1024**2,
            "audit storage reserve",
        )

    def interrupt(signum: int, frame: Any) -> None:
        raise TimeoutError(f"audit timer {signum}")

    for timer, signum, duration in (
        (signal.ITIMER_REAL, signal.SIGALRM, 238),
        (signal.ITIMER_PROF, signal.SIGPROF, 178),
    ):
        require(signal.getitimer(timer) == (0, 0), "exclusive independent timer")
        signal.signal(signum, interrupt)
        signal.setitimer(timer, duration)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "one independent numerical thread")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(file_sha256(Path(__file__), guard) == script_sha, "admitted checker bytes")
        require(len(CORE_HASHES) >= 7, "frozen independent source catalog")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "frozen independent arithmetic bytes")
        require(file_sha256(study / "summary.json", guard) == main_sha, "admitted main receipt bytes")
        require(
            file_sha256(source_root / CONFIG, guard)
            == "c1bd46492710f8113b7f1f0d224a20b16bc007006006a28b751b7c4069349935",
            "frozen diagnosis science config",
        )
        require(
            file_sha256(source_root / ADMISSION, guard)
            == "3034db4a76d1245622ccf477f47d58c498274e0fe71d91440c7c92eb38383945",
            "frozen diagnosis identity admission",
        )
        main, config, admission = (
            read_json(study / "summary.json"),
            read_json(source_root / CONFIG),
            read_json(source_root / ADMISSION),
        )
        require(admission["parent"]["root"] == config["parent_root"], "exact allowed predecessor root")
        parent = relative_file(root, config["parent_root"])
        costs = verified_accounting(study, parent, main, admission, guard)
        count = bindings(source_root, study, parent, main, config, admission, guard)
        costs = main_resources(study, parent, main, admission, guard)
        support, maps, metrics = arrays_from_parent(parent)
        predecessor = parent_closure(parent, study, main, support, maps, metrics, guard)
        results = metric_evidence(study, main, support, maps, guard)
        compare_values(
            dict.fromkeys(
                (
                    "parent_full_grid_AP_exact",
                    "parent_full_aggregate_exact",
                    "fixed_complete_rows",
                    "sources_and_inputs_unchanged",
                ),
                True,
            ),
            main["integrity_gates"],
            "all technical integrity predicates",
        )
        require(
            main["technical_passed"] is True and main["clinical_claim"] is False,
            "technical-only claim boundary",
        )
        require(
            main["oracle_conditioning"] == "active-lead oracle at scoring only; not deployable performance",
            "explicit scoring oracle",
        )
        require(
            main["derived_lead_dependence"]
            == "equal canonical-lead weights do not imply physical independence",
            "explicit derived-lead dependence",
        )
        figure(study, main["figures"])
        require(
            bindings(source_root, study, parent, main, config, admission, guard) == count,
            "all consumed bindings unchanged",
        )
        require(file_sha256(study / "summary.json", guard) == main_sha, "main receipt unchanged")
        require(
            file_sha256(study / "prospective_ledger.json", guard) == costs["main_ledger_sha256"],
            "main terminal ledger unchanged",
        )
        require(file_sha256(Path(__file__), guard) == script_sha, "checker unchanged")
        for name, expected in CORE_HASHES.items():
            require(file_sha256(source_root / name, guard) == expected, "independent source unchanged")
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "main_sha256": main_sha,
            "script_sha256": script_sha,
            "core_sha256": CORE_HASHES,
            "bindings_verified": count,
            "predecessor": predecessor,
            **results,
            "resources": costs,
            "independent_model_forwards": 0,
            "independent_optimizer_updates": 0,
            "gpu_used": False,
            "raw_real_ecg_reads": 0,
            "checkpoint_or_signal_arrays_opened": 0,
            "clinical_claim": False,
            "oracle_conditioning": True,
            "derived_leads_not_independent": True,
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
                allow_nan=False,
            )
            stream.write("\n")
        raise
    finally:
        write_ledger(destination, receipt, clock, costs)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse explicit roots and coordinator-admitted source/receipt identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
