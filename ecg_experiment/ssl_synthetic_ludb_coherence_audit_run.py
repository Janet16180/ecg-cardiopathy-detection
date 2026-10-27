"""Run the independently frozen conditional activity-coherence audit without neural work."""

import json
import os
import resource
import signal
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_coherence_audit as core
from ecg_experiment.ssl_synthetic_ludb_audit import summary as qualification_summary
from ecg_experiment.ssl_synthetic_ludb_audit import waveform
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_ludb_coherence_bridge_v1.json"
STUDY = "outputs/ssl_synthetic_localization/ludb_coherence_bridge_v1"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_coherence_bridge.py"
RUNNER = "ecg_experiment/ssl_synthetic_ludb_coherence_audit_run.py"
OWN_HASHES = dict(zip(
    (
        'ecg_experiment/ssl_synthetic_ludb_coherence_audit.py',
        'ecg_experiment/ssl_synthetic_ludb_audit.py',
        'ecg_experiment/intervals.py',
        'ecg_experiment/paths.py',
        'ecg_experiment/__init__.py',
        'ecg_experiment/synthetic_audit_io.py',
        'ecg_experiment/ssl_synthetic_pretrained_saved_audit.py',
        'tests/test_ssl_synthetic_ludb_coherence_audit.py',
        'docs/ssl-synthetic-localization-ludb-coherence-independent-audit-v1.md',
    ),
    (
        'dfef0225e41bb5469b3228b2dac2f9c23bb5b558e451bb26cd01ca415fba2d71',
        '03b6646553b74d2e9a31f5c6f62cd904594962422aeb131b0d5b980cb8d41c46',
        'b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af',
        '8eee1da7579b23b06edbf5d6d76e4cd9a7ced4e8b17d2dd757f20bff7a1e80ee',
        'cca26ce1124d553e877ed33383ffaba3af13f89bd6628e7a1b9ca2cc5c1c6a35',
        'f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a',
        '650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09',
        '702c5de7018dbd14081e4c3e26696e35c6d8b405a56a0d4d4a826c1ea8e4d6a3',
        '1c6e9b2040318fc1fbe0a9ff4f200bed84a98eadd45f0c8bf388e67fa261a037',
    ),
    strict=True,
))


def write_new(path: Path, value: Any) -> None:
    """Retain a strict exclusive audit receipt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def byte_count(directory: Path) -> int:
    """Enumerate actual regular audit artifacts without following raw-data links."""
    return sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file() and not path.is_symlink()
    )


def predecessor(root: Path, study: Path, admission: dict[str, Any]) -> int:
    """Require the complete audited qualification PASS and close its saved rows/bootstrap exactly."""
    spec = admission["qualification"]
    parent = relative_file(root, spec["directory"])
    for name, key in (
        ("summary.json", "summary_sha256"),
        ("prospective_ledger.json", "ledger_sha256"),
        ("independent_audit_v1/summary.json", "audit_summary_sha256"),
        ("independent_audit_v1/prospective_ledger.json", "audit_ledger_sha256"),
    ):
        require(file_sha256(parent / name) == spec[key], "qualified predecessor receipt binding")
    main = read_json(parent / "summary.json")
    audit = read_json(parent / "independent_audit_v1/summary.json")
    require(
        main["technical_passed_before_independent_audit"]
        and main["scientific_passed_before_independent_audit"]
        and audit["all_passed"]
        and audit["scientific_passed"]
        and audit["mismatch_count"] == 0,
        "complete QRS qualification PASS",
    )
    for name in ("prospective_ledger.json", "independent_audit_v1/prospective_ledger.json"):
        require(
            all(read_json(parent / name)["resource_gates"].values()), "qualified predecessor resource PASS"
        )
    for name in ("record_rows.jsonl", "bootstrap_gains.npy"):
        require(
            file_sha256(parent / name) == main["output_sha256"][name], "retained qualified numerical binding"
        )
    rows = sorted(
        [json.loads(line) for line in (parent / "record_rows.jsonl").read_text().splitlines()],
        key=lambda row: row["record_id"],
    )
    expected = qualification_summary(rows)
    bootstrap = np.asarray(expected.pop("bootstrap_gains"))
    checks = compare_values(
        expected, main["statistics"], "exact independently reconstructed qualification", 0
    )
    require(
        np.array_equal(bootstrap, np.load(parent / "bootstrap_gains.npy", allow_pickle=False)),
        "exact qualification two-thousand-draw closure",
    )
    closure = {
        "all_exact": True,
        "subject_rows": 200,
        "measurement_subjects": 160,
        "bootstrap_draws": 2000,
        "statistics": expected,
        "receipt_sha256": {
            name: spec[key]
            for name, key in (
                ("summary.json", "summary_sha256"),
                ("prospective_ledger.json", "ledger_sha256"),
                ("independent_audit_v1/summary.json", "audit_summary_sha256"),
                ("independent_audit_v1/prospective_ledger.json", "audit_ledger_sha256"),
            )
        },
    }
    return checks + compare_values(
        closure,
        read_json(study / "qualified_predecessor_closure.json"),
        "exact saved qualification closure",
        0,
    )


def source_bindings(source: Path, study: Path, main: dict[str, Any], admission: dict[str, Any]) -> int:
    """Check all immutable main sources, snapshots and complete output byte identities."""
    count = 0
    for name, digest in main["source_sha256"].items():
        require(file_sha256(source / name) == digest, "main source identity")
        snapshot = study / "source_snapshot" / name
        if name not in admission["source_sha256"]:
            snapshot = study / "admission_snapshot.json"
        require(file_sha256(snapshot) == digest, "main source snapshot identity")
        count += 2
    catalog = {
        path.relative_to(study).as_posix()
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "independent_audit_v1" not in path.parts
        and path.name not in {"summary.json", "prospective_ledger.json"}
    }
    require(catalog == set(main["output_sha256"]), "complete bridge output catalog")
    for name, digest in main["output_sha256"].items():
        require(file_sha256(relative_file(study, name)) == digest, "bridge output identity")
        count += 1
    return count


def assets(root: Path, config: dict[str, Any], admission: dict[str, Any]) -> tuple[Path, Path]:
    """Admit exactly four hundred alias-backed real files and the four original generated artifacts."""
    expected = {f"{number}.{extension}" for number in range(1, 201) for extension in ("dat", "hea")}
    require(set(admission["asset_sha256"]) == expected, "exact four-hundred-file real signal allowlist")
    actual = relative_file(root, admission["actual_data_root"])
    logical = root / admission["logical_data_root"]
    for name, digest in admission["asset_sha256"].items():
        require(
            os.path.samefile(logical / name, actual / name), "read-only raw alias points to admitted asset"
        )
        require(file_sha256(actual / name) == digest, "admitted real signal identity")
    generated = relative_file(root, config["generated"]["directory"])
    require(
        set(config["generated"]["allowed_files"])
        == {"raw.npy", "cases.jsonl", "summary.json", "independent_audit_v1/summary.json"},
        "exact old-bank allowlist",
    )
    for name, digest in config["generated"]["allowed_files"].items():
        require(file_sha256(generated / name) == digest, "original generated bank identity")
    return actual, generated


def input_signal(raw: np.ndarray, data: Path, index: int, cohort: str) -> tuple[np.ndarray, Any]:
    """Read the admitted cohort independently without indexing the smaller generated bank for real records."""
    if cohort == "generated":
        return raw[index], None
    require(cohort == "ludb", "prospective cohort identity")
    physical, metadata = waveform(data / str(index + 1))
    return physical[[0, 1, 6, 7, 8, 9, 10, 11]].astype(np.float32), metadata


def numerical(
    data: Path, generated_root: Path, study: Path, main: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Reconstruct every original/real row, activity cell and independent bootstrap without main endpoints."""
    raw = np.load(generated_root / "raw.npy", mmap_mode="r", allow_pickle=False)
    require(raw.shape == (128, 8, 5000) and raw.dtype == np.float32, "original native bank geometry")
    original = [json.loads(line) for line in (generated_root / "cases.jsonl").read_text().splitlines()]
    require(
        len(original) == 128
        and read_json(generated_root / "independent_audit_v1/summary.json")["all_passed"],
        "complete independently audited original generated bank",
    )
    cohorts, checks = {}, 0
    for cohort, count, filename in (
        ("generated", 128, "generated_rows.jsonl"),
        ("ludb", 200, "ludb_rows.jsonl"),
    ):
        saved = [json.loads(line) for line in (study / filename).read_text().splitlines()]
        masks = np.load(study / f"{cohort}_activity_masks.npy", mmap_mode="r", allow_pickle=False)
        require(
            masks.shape == (count, 8, 5000) and masks.dtype == bool and len(saved) == count,
            "complete fixed activity mask geometry and rows",
        )
        rows = []
        for index in range(count):
            guard()
            values, metadata = input_signal(raw, data, index, cohort)
            row, expected_masks = core.prepare(values, index, cohort)
            if metadata is not None:
                row["signal_metadata"] = metadata
            checks += compare_values(row, saved[index], "independent signal-derived bridge row")
            require(np.array_equal(expected_masks, masks[index]), "every saved activity-mask cell")
            if cohort == "generated":
                require(
                    original[index]["index"] == index and original[index]["seed"] == row["generator_seed"],
                    "original generated seed and row order",
                )
                for key in ("standardization", "coherence"):
                    checks += compare_values(
                        row[key], original[index][key], "exact original generator closure", 0
                    )
            rows.append(row)
        cohorts[cohort] = rows
    old_primary = core.old_primary(cohorts["generated"])
    checks += compare_values(
        old_primary, read_json(generated_root / "summary.json")["primary"], "exact old seed interval", 0
    )
    old_closure = {
        "all_exact": True,
        "records": 128,
        "transformations": 128,
        "coherence_rows": 128,
        "primary": old_primary,
    }
    checks += compare_values(old_closure, main["generated_predecessor_closure"], "embedded old closure", 0)
    checks += compare_values(
        old_closure, read_json(study / "generated_predecessor_closure.json"), "saved old closure", 0
    )
    aggregate, bootstrap = core.summarize(cohorts["generated"], cohorts["ludb"])
    checks += compare_values(aggregate, main["statistics"], "all descriptive bridge outputs")
    saved_draws = np.load(study / "bootstrap_cohort_means_and_contrasts.npy", allow_pickle=False)
    require(saved_draws.shape == (2000, 3) and saved_draws.dtype == np.float64, "bootstrap native geometry")
    require(np.allclose(saved_draws, bootstrap, rtol=0, atol=1e-12), "all real-first bootstrap cohort draws")
    return {
        "row_scalar_checks": checks,
        "rows": 328,
        "activity_cells": 13120000,
        "bootstrap_draws": 2000,
        "statistics": aggregate,
        "original_128_exact": True,
    }


def accounting(root: Path, study: Path, main: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    """Verify immediate qualified-audit cost lineage and actual terminal main ceilings."""
    prior = admission["prior_ledger"]
    spec = admission["qualification"]
    require(
        prior["path"] == spec["directory"] + "/independent_audit_v1/prospective_ledger.json"
        and prior["sha256"] == spec["audit_ledger_sha256"],
        "immediate qualified-audit predecessor",
    )
    path = relative_file(root, prior["path"])
    require(file_sha256(path) == prior["sha256"], "actual prior ledger identity")
    previous = read_json(path)
    ledger = read_json(study / "prospective_ledger.json")
    require(
        ledger["receipt_sha256"] == file_sha256(study / "summary.json")
        and ledger["prior_ledger_sha256"] == prior["sha256"],
        "main receipt and prior bindings",
    )
    require(
        ledger["current_machine_stages"] == previous["current_machine_stages"] + 1,
        "immediate next actual stage",
    )
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + suffix
        require(
            previous[key] == prior[key] and ledger[key] == previous[key] + ledger[suffix],
            "exactly-once current-machine bridge cost",
        )
    require(
        ledger["gpu_wall_seconds"] == 0 and all(ledger["resource_gates"].values()),
        "main CPU-only resource PASS",
    )
    require(
        main["cpu_seconds"] <= ledger["cpu_seconds"] < 180
        and main["wall_seconds"] <= ledger["wall_seconds"] < 300,
        "main terminal CPU/active-wall caps",
    )
    require(main["peak_rss_bytes"] < 2 * 1024**3, "main RSS ceiling")
    size = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file() and not path.is_symlink() and "independent_audit_v1" not in path.parts
    )
    require(
        size < 128 * 1024**2
        and size - (study / "prospective_ledger.json").stat().st_size
        == ledger["output_bytes_before_terminal_ledger"],
        "actual final main storage",
    )
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "prior_ledger_sha256": file_sha256(study / "prospective_ledger.json"),
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
        "stages_through_main": ledger["current_machine_stages"],
        "main_bytes": size,
    }


def figure_checks(study: Path) -> None:
    """Verify fixed figure identities/layouts without claiming a fresh pixel reconstruction."""
    for name, shape in (
        ("all_record_coherence_scatter.png", (1440, 720)),
        ("coherence_ECDFs.png", (1800, 600)),
        ("coherence_histograms.png", (1800, 600)),
        ("fixed_activity_rasters.png", (1920, 1200)),
    ):
        with (study / name).open("rb") as stream:
            header = stream.read(24)
        require(
            header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", header[16:24]) == shape,
            "fixed PNG signature and geometry",
        )


def terminal(destination: Path, clock: dict[str, float], costs: dict[str, Any], receipt: str) -> None:
    """Charge self once after actual receipt hashing and byte enumeration."""
    digest, size = file_sha256(destination / receipt), byte_count(destination)
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "cpu": cpu < 180,
        "wall": wall < 300,
        "rss": rss < 2 * 1024**3,
        "output": size + 16384 < 64 * 1024**2,
    }
    write_new(
        destination / "prospective_ledger.json",
        {
            "receipt": receipt,
            "sha256": digest,
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "gpu_wall_seconds": 0.0,
            "peak_rss_bytes": rss,
            "resource_gates": gates,
            "prior_ledger_sha256": costs.get("prior_ledger_sha256"),
            "current_machine_stages": costs["stages_through_main"] + 1 if costs else None,
            "current_machine_cumulative_cpu_seconds": costs["cpu_through_main"] + cpu if costs else None,
            "current_machine_cumulative_gpu_wall_seconds": costs.get("gpu_through_main"),
            "bytes_before_terminal_ledger": size,
            "historical_costs_verified": False,
            "measurement_endpoint": "after receipt hash/byte enumeration before terminal serialization",
            "terminal_bookkeeping_reserve_seconds": 2,
            "terminal_bookkeeping_reserve_bytes": 16384,
        },
    )
    require(all(gates.values()), "independent bridge terminal resource gates")


def run(
    root: Path,
    source: Path,
    admission_name: str,
    main_sha: str,
    script_sha: str,
    runner_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Execute a single independently admitted descriptive audit, retaining all failure evidence."""
    study = root / STUDY
    destination = study / "independent_audit_v1"
    destination.mkdir(exist_ok=False)
    costs: dict[str, Any] = {}

    def guard() -> None:
        require(time.process_time() - clock["cpu"] < 178, "audit CPU reserve")
        require(time.perf_counter() - clock["wall"] < 298, "audit wall reserve")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "audit RSS ceiling")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent descriptive audit timer:{signum}")

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGPROF, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, 298 - (time.perf_counter() - clock["wall"])))
    signal.setitimer(signal.ITIMER_PROF, max(0.01, 178 - (time.process_time() - clock["cpu"])))
    try:
        require(file_sha256(Path(__file__)) == runner_sha, "actual imported independent runner identity")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hidden CUDA")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(name) == "1", "single numerical thread")
        own = {**OWN_HASHES, CHECKER: script_sha, RUNNER: runner_sha}
        for name, digest in own.items():
            require(file_sha256(source / name) == digest, "independent source admission")
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        require(file_sha256(study / "summary.json") == main_sha, "explicit main summary admission")
        main, admission = read_json(study / "summary.json"), read_json(source / admission_name)
        require(
            file_sha256(source / admission_name) == main["source_sha256"][admission_name],
            "identity-only main admission",
        )
        costs = accounting(root, study, main, admission)
        predecessor_checks = predecessor(root, study, admission)
        config = read_json(source / CONFIG)
        data, generated = assets(root, config, admission)
        source_checks = source_bindings(source, study, main, admission)
        values = numerical(data, generated, study, main, guard)
        figure_checks(study)
        source_checks += source_bindings(source, study, main, admission)
        assets(root, config, admission)
        require(
            main["status"] == "PASS"
            and main["technical_passed_before_independent_audit"]
            and all(main["technical_gates"].values()),
            "main technical flags only",
        )
        require(
            "scientific_passed" not in main and "scientific_passed" not in main["statistics"],
            "no descriptive scientific-success rule",
        )
        require(
            main["new_annotation_stream_reads"]
            == main["generator_forwards"]
            == main["optimizer_updates"]
            == 0,
            "zero prohibited bridge operations",
        )
        require(
            main["raw_inputs_copied"] == 0
            and main["waveform_record_reads"] == 200
            and main["coherence_rows"] == 328,
            "complete fixed bridge operation counts",
        )
        environment = dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"
        )
        environment["CUDA_VISIBLE_DEVICES"] = ""
        compare_values(environment, main["runtime"]["environment"], "main one-thread hidden-CUDA environment")
        require(
            main["gpu_used"] is False
            and main["clinical_quality_claim"] is False
            and main["physiological_fidelity_claim"] is False,
            "no GPU or clinical-quality claim",
        )
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "numerical": values,
            "predecessor_scalar_checks": predecessor_checks,
            "source_binding_checks": source_checks,
            "source_sha256": own,
            "main_summary_sha256": main_sha,
            "accounting": costs,
            "clinical_quality_claim": False,
            "gpu_used": False,
            "new_annotation_stream_reads": 0,
        }
        write_new(destination / "summary.json", result)
        terminal(destination, clock, costs, "summary.json")
        return result
    except Exception as error:
        write_new(
            destination / "failure.json",
            {
                "all_passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "costs_captured": costs,
            },
        )
        if not (destination / "prospective_ledger.json").exists():
            terminal(destination, clock, costs, "failure.json")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
