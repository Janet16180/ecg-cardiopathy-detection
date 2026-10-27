"""Reuse unchanged independent coherence mathematics after the actual released-N recovery gate."""

import json
import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_coherence_audit_run as inherited
from ecg_experiment.ssl_synthetic_ludb_landmark_audit import aggregate
from ecg_experiment.ssl_synthetic_ludb_landmark_audit_recovery import metadata_preflight, next_stage
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_ludb_coherence_bridge_v2.json"
STUDY = "outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2"
RUNNER = "ecg_experiment/ssl_synthetic_ludb_coherence_audit_v2.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_coherence_v2.py"
RECOVERY_CONFIG_SHA = "cdbc897b915761f14d4cc4b65016c0c15d8f885b2011edaca9077a3ae51947c9"
OWN_HASHES = dict(zip(
    (
        'docs/ssl-synthetic-localization-ludb-coherence-independent-audit-v2.md',
        'tests/test_ssl_synthetic_ludb_coherence_audit_v2.py',
        'ecg_experiment/ssl_synthetic_ludb_coherence_audit_run.py',
        'ecg_experiment/ssl_synthetic_ludb_landmark_audit.py',
        'ecg_experiment/ssl_synthetic_ludb_landmark_audit_run.py',
        'ecg_experiment/ssl_synthetic_ludb_landmark_audit_recovery.py',
        'configs/ssl_synthetic_ludb_landmark_audit_recovery_v2.json',
        'configs/ssl_synthetic_ludb_coherence_bridge_v1.json',
        'configs/ssl_synthetic_ludb_coherence_bridge_v2.json',
        'docs/ssl-synthetic-localization-ludb-coherence-bridge-v1.md',
        'docs/ssl-synthetic-localization-ludb-coherence-bridge-v2.md',
        'docs/ssl-synthetic-localization-ludb-coherence-bridge-audit-recovery-v1.md',
    ),
    (
        'fef47594b2f01aff7f70fdad404d99b4b7fab9a4d0e5a9bd2c61efc5f193688b',
        '1a40d5f2f7d4f92d9258f9fc476bceb8e641f56627454d7a4a56a0e3bd43778e',
        'ad61d77f9bfb8769c73a85ef24370b8583686350b277995972d69587553c1515',
        '06b60712e6247d61b4db14594782f6e5ae178a453b8b91349033e4f2ca9355d8',
        'a16c9dc1d3513698b13a14255a2c60105009bdf12c56010e025bb24302e429aa',
        '4c96dda7fe1ed4893dd7c32e0cb0b04f132346f82dc0865a4f1c60cf2eb8813b',
        'cdbc897b915761f14d4cc4b65016c0c15d8f885b2011edaca9077a3ae51947c9',
        'a9b8f8a937302d1c6b4d34c1e34adad6607ceb454bc79cd6f1c8fa44b622f003',
        'ffe865730e4ba769e5ebdf7a9aae914a2e3d006dffb327eb898b7d53d6ef5a31',
        '368055277f75dd3355a8b492ef29de8e4848b99cc6c2cccae14fb475f2f604c2',
        'c323286ed916520c09a90c5de07ca9db75fd4a805e8f5a75eec1789e7625b39c',
        '7842319b756a162e9dbc7f103e48ea689796af58a052d5e38908c7563cce2342',
    ),
    strict=True,
))


def prerequisite(root: Path, source: Path, admission: dict[str, Any], config: dict[str, Any]) -> int:
    """Verify actual26/failed27/setup/recovery28 without reading any new bridge signal or descriptor."""
    spec, gate = admission["qualification"], config["conditional_admission"]
    audit_name = gate["qualification_audit_directory"]
    require(
        audit_name == "independent_audit_v2"
        and gate["qualification_audit_stage"] == 28
        and gate["qualification_main_stage"] == 26,
        "actual landmark recovery prerequisite",
    )
    directory = relative_file(root, spec["directory"])
    receipts = {
        "summary.json": spec["summary_sha256"],
        "prospective_ledger.json": spec["ledger_sha256"],
        audit_name + "/summary.json": spec["audit_summary_sha256"],
        audit_name + "/prospective_ledger.json": spec["audit_ledger_sha256"],
    }
    for name, digest in receipts.items():
        require(file_sha256(directory / name) == digest, "actual released-N prerequisite receipt")
    main, audit = read_json(directory / "summary.json"), read_json(directory / audit_name / "summary.json")
    require(
        main["scientific_passed_before_independent_audit"]
        and main["technical_passed_before_independent_audit"]
        and audit["all_passed"]
        and audit["scientific_passed"]
        and audit["mismatch_count"] == 0,
        "complete landmark engineering/technical recovery PASS",
    )
    setup, reconciled = metadata_preflight(root, source, RECOVERY_CONFIG_SHA)
    binding = setup["prior_reconciliation"]
    require(
        all(spec["failed_audit_reconciliation"][key] == binding[key] for key in ("path", "sha256")),
        "same retained failed27 setup identity",
    )
    ledger = next_stage(
        root,
        directory / audit_name / "prospective_ledger.json",
        spec["audit_ledger_sha256"],
        reconciled,
        binding["sha256"],
    )
    prior = admission["prior_ledger"]
    require(
        prior["path"] == spec["directory"] + "/" + audit_name + "/prospective_ledger.json"
        and prior["sha256"] == spec["audit_ledger_sha256"],
        "immediate actual recovery28 ledger",
    )
    for key in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        require(prior[key] == ledger[key], "exact latest audited costs")
    manifest = directory / "release_file_identities.json"
    require(file_sha256(manifest) == main["output_sha256"][manifest.name], "qualified release identities")
    qualified = read_json(manifest)
    expected = {f"{n}.{extension}" for n in range(1, 201) for extension in ("dat", "hea")}
    require(
        set(admission["asset_sha256"]) == expected
        and all(qualified[name] == digest for name, digest in admission["asset_sha256"].items()),
        "same400 already qualified signal/header identities",
    )
    return len(receipts) + len(expected)


def closure(root: Path, study: Path, admission: dict[str, Any], config: dict[str, Any]) -> int:
    """Reproduce exact point statistics and bootstrap using the frozen independent landmark aggregate."""
    spec, name = admission["qualification"], config["conditional_admission"]["qualification_audit_directory"]
    parent = root / spec["directory"]
    main = read_json(parent / "summary.json")
    for filename in ("record_rows.jsonl", "bootstrap_gains.npy"):
        require(
            file_sha256(parent / filename) == main["output_sha256"][filename], "qualified numerical identity"
        )
    rows = sorted(
        [json.loads(line) for line in (parent / "record_rows.jsonl").read_text().splitlines()],
        key=lambda row: row["record_id"],
    )
    statistics = aggregate(rows)
    draws = np.asarray(statistics.pop("bootstrap_gains"))
    checks = compare_values(statistics, main["statistics"], "exact independent landmark closure", 0)
    require(
        np.array_equal(draws, np.load(parent / "bootstrap_gains.npy", allow_pickle=False)),
        "exact all2000 landmark draws",
    )
    expected = {
        "all_exact": True,
        "subject_rows": 200,
        "measurement_subjects": 160,
        "bootstrap_draws": 2000,
        "statistics": statistics,
        "receipt_sha256": {
            "summary.json": spec["summary_sha256"],
            "prospective_ledger.json": spec["ledger_sha256"],
            name + "/summary.json": spec["audit_summary_sha256"],
            name + "/prospective_ledger.json": spec["audit_ledger_sha256"],
        },
    }
    return checks + compare_values(
        expected,
        read_json(study / "qualified_predecessor_closure.json"),
        "saved actual landmark prerequisite closure",
        0,
    )


def accounting(root: Path, study: Path, main: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    """Verify bridge29 appends actual recovery28 and retains every intervening failed self-cost."""
    prior = admission["prior_ledger"]
    previous = read_json(relative_file(root, prior["path"]))
    ledger_path = study / "prospective_ledger.json"
    ledger = next_stage(root, ledger_path, file_sha256(ledger_path), previous, prior["sha256"])
    require(
        ledger["current_machine_stages"] == 29
        and ledger["receipt_sha256"] == file_sha256(study / "summary.json"),
        "actual descriptive bridge29 receipt",
    )
    require(
        main["cpu_seconds"] <= ledger["cpu_seconds"] < 180
        and main["wall_seconds"] <= ledger["wall_seconds"] < 300
        and ledger["peak_rss_bytes"] < 2 * 1024**3,
        "actual bridge29 terminal resource ceilings",
    )
    retained = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file() and not path.is_symlink() and "independent_audit_v1" not in path.parts
    )
    require(
        retained - ledger_path.stat().st_size == ledger["output_bytes_before_terminal_ledger"]
        and retained < 128 * 1024**2,
        "actual bridge29 byte arithmetic",
    )
    return {
        "prior_ledger_sha256": file_sha256(ledger_path),
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
        "stages_through_main": 29,
        "main_bytes": retained,
    }


def metadata_setup(root: Path, source: Path, admission_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check prerequisite/source schemas without reading a bridge signal, mask or bootstrap array."""
    admission, config = read_json(source / admission_name), read_json(source / CONFIG)
    inherited_config = read_json(source / "configs/ssl_synthetic_ludb_coherence_bridge_v1.json")
    for key in config["inherited_scientific_method"]["unchanged_sections"]:
        require(config[key] == inherited_config[key], "unchanged inherited descriptor method")
    prerequisite(root, source, admission, config)
    for bindings in (inherited.OWN_HASHES, OWN_HASHES, admission["source_sha256"]):
        for name, digest in bindings.items():
            require(file_sha256(source / name) == digest, "complete conditional executable source binding")
    return admission, config


def run(
    root: Path,
    source: Path,
    admission_name: str,
    main_sha: str,
    entry_sha: str,
    runner_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Run one source-admitted descriptive reconstruction; all neural and clinical operations remain zero."""
    study = root / STUDY
    destination = study / "independent_audit_v1"
    destination.mkdir(exist_ok=False)
    costs: dict[str, Any] = {}
    timers: dict[int, Any] = {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 178 and time.perf_counter() - clock["wall"] < 298,
            "descriptive audit receipt reserve",
        )
        require(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "descriptive audit RSS"
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Conditional independent bridge timer {signum}")

    try:
        for timer, signum, limit, key in (
            (signal.ITIMER_REAL, signal.SIGALRM, 298, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 178, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            elapsed = (time.perf_counter() if key == "wall" else time.process_time()) - clock[key]
            signal.setitimer(timer, max(0.01, limit - elapsed))
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(
                os.environ.get(key) == "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            ),
            "hidden CUDA and single-thread descriptor audit",
        )
        require(
            file_sha256(Path(__file__)) == runner_sha and file_sha256(source / CHECKER) == entry_sha,
            "literal conditional independent executables",
        )
        admission, config = metadata_setup(root, source, admission_name)
        own = {**inherited.OWN_HASHES, **OWN_HASHES, RUNNER: runner_sha, CHECKER: entry_sha}
        for name, digest in own.items():
            require(file_sha256(source / name) == digest, "frozen independent descriptor source")
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        require(
            file_sha256(study / "summary.json") == main_sha
            and file_sha256(source / admission_name)
            == read_json(study / "summary.json")["source_sha256"][admission_name],
            "actual bridge29 receipt and identity admission",
        )
        main = read_json(study / "summary.json")
        costs = accounting(root, study, main, admission)
        predecessor_checks = closure(root, study, admission, config)
        data, generated = inherited.assets(root, config, admission)
        source_checks = inherited.source_bindings(source, study, main, admission)
        values = inherited.numerical(data, generated, study, main, guard)
        inherited.figure_checks(study)
        source_checks += inherited.source_bindings(source, study, main, admission)
        inherited.assets(root, config, admission)
        require(
            main["status"] == "PASS"
            and main["technical_passed_before_independent_audit"]
            and all(main["technical_gates"].values()),
            "descriptive bridge technical conjunction",
        )
        require(
            "scientific_passed" not in main and "scientific_passed" not in main["statistics"],
            "no descriptive scientific-success flag",
        )
        require(
            main["new_annotation_stream_reads"]
            == main["generator_forwards"]
            == main["optimizer_updates"]
            == 0
            and main["raw_inputs_copied"] == 0
            and main["waveform_record_reads"] == 200
            and main["coherence_rows"] == 328
            and main["gpu_used"] is False
            and main["clinical_quality_claim"] is False
            and main["physiological_fidelity_claim"] is False,
            "fixed operations and no clinical or physiology claim",
        )
        guard()
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
    except Exception as error:
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        inherited.write_new(destination / "summary.json", result)
        inherited.terminal(destination, clock, costs, "summary.json")
    except Exception as error:
        if not (destination / "prospective_ledger.json").exists():
            for timer in timers:
                signal.setitimer(timer, 0)
            inherited.write_new(
                destination / "finalization_failure.json",
                {
                    "all_passed": False,
                    "mismatch_count": 1,
                    "error_type": type(error).__name__,
                    "measurement_scope": "retained bounded finalization failure",
                },
            )
            inherited.terminal(destination, clock, costs, "finalization_failure.json")
        raise
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
    require(result["all_passed"], "retained conditional descriptive audit failure")
    return result
