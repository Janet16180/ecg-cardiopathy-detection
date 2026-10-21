"""Recover only legacy accounting/path schemas while reusing the frozen landmark mathematics."""

import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_ludb_landmark_audit_run as frozen
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_ludb_landmark_audit_recovery_v2.json"
RUNNER = "ecg_experiment/ssl_synthetic_ludb_landmark_audit_recovery.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_landmark_recovery.py"
OWN_HASHES = dict(zip(
    (
        'scripts/validation/audit_ssl_synthetic_ludb_landmarks.py',
        'ecg_experiment/ssl_synthetic_ludb_landmark_audit_run.py',
        'configs/ssl_synthetic_ludb_landmark_audit_recovery_v2.json',
        'docs/ssl-synthetic-localization-ludb-landmark-independent-audit-recovery-v2.md',
        'tests/test_ssl_synthetic_ludb_landmark_audit_recovery.py',
    ),
    (
        '114745fef4e82e7fb7d0206f9c518a9bb9a110035d1eac1d59f4276c6f2c67d3',
        'a16c9dc1d3513698b13a14255a2c60105009bdf12c56010e025bb24302e429aa',
        'cdbc897b915761f14d4cc4b65016c0c15d8f885b2011edaca9077a3ae51947c9',
        'fa126b5cff0cbc22f441756e53649de3e51a436effebcc2ff0043a6bbb365cb2',
        '02033d0bc7235d792e0a4d1731daf29397aaf937f412758f93fdca6820732dd7',
    ),
    strict=True,
))


def main_files(study: Path) -> dict[str, Path]:
    """Separate both retained independent attempts from the immutable main manifest."""
    return {
        path.relative_to(study).as_posix(): path
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(study).parts[0] not in {"independent_audit_v1", "independent_audit_v2"}
    }


def next_stage(
    root: Path, path: Path, digest: str, previous: dict[str, Any], previous_sha: str
) -> dict[str, Any]:
    """Accept pinned legacy receipt path/gate schemas without modifying any historical file."""
    require(file_sha256(path) == digest, "immutable historical ledger")
    ledger = read_json(path)
    name = ledger.get("receipt", "summary.json")
    receipt = path.parent / Path(name).name
    require(
        Path(name).name in {"summary.json", "failure.json"}
        and (name == Path(name).name or relative_file(root, name) == receipt),
        "known receipt path schema",
    )
    require(
        file_sha256(receipt) == ledger.get("receipt_sha256", ledger.get("sha256")), "historical bound receipt"
    )
    require(
        ledger["prior_ledger_sha256"] == previous_sha
        and ledger["current_machine_stages"] == previous["current_machine_stages"] + 1,
        "sequential actual historical stage",
    )
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + suffix
        require(
            ledger[suffix] >= 0 and ledger[key] == previous[key] + ledger[suffix],
            "exactly-once historical measured cost",
        )
    gates = ledger["resource_gates"] if "resource_gates" in ledger else read_json(receipt)["resource_gates"]
    require(all(gates.values()) and ledger["gpu_wall_seconds"] == 0, "bound historical resource evidence")
    return ledger


def legacy_history(root: Path, source: Path, admission: dict[str, Any]) -> dict[str, Any]:
    """Derive only the absent baseline count in memory, retaining all original schema/bytes."""
    prefix = root / "outputs/ssl_synthetic_localization"
    digest = "a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf"
    path = prefix / "generated_temporal_diagnosis_v2/independent_audit_v1/prospective_ledger.json"
    require(file_sha256(path) == digest, "literal legacy independently audited baseline")
    original = read_json(path)
    require(
        original.get("current_machine_stages", 18) == 18
        and original["current_machine_cumulative_cpu_seconds"] == 429.49696094999996
        and original["current_machine_cumulative_gpu_wall_seconds"] == 447.98139213304967,
        "declared baseline18 with original literal cumulative costs",
    )
    previous = {**original, "current_machine_stages": 18}
    for directory, current in frozen.HISTORY:
        previous = next_stage(root, prefix / directory / "prospective_ledger.json", current, previous, digest)
        digest = current
    path = prefix / "ludb_qualification_reconciliation_v1/reconciled_prior.json"
    current = "8ff30f48c3d92057be2c641dd3a168a2d009a9f23fc4206bd5ce24612fff2a24"
    require(file_sha256(path) == current, "original failed-main reconciliation identity")
    record = read_json(path)
    for key in (
        "verified_acquisition_ledger",
        "retained_failed_main_ledger",
        "retained_failed_main_receipt",
        "failed_main_admission",
    ):
        spec = record[key]
        directory = source if key == "failed_main_admission" else root
        require(
            file_sha256(relative_file(directory, spec["path"])) == spec["sha256"],
            "reconciled immutable input",
        )
    failed = read_json(relative_file(root, record["retained_failed_main_ledger"]["path"]))
    require(
        failed["prior_accounting_verified"] is False
        and failed["prior_ledger_sha256"] is None
        and failed["current_machine_cumulative_cpu_seconds"] is None
        and failed["current_machine_cumulative_gpu_wall_seconds"] is None,
        "original failed23 null fields retained",
    )
    require(
        record["current_machine_stages"] == 23
        and record["current_machine_cumulative_cpu_seconds"]
        == previous["current_machine_cumulative_cpu_seconds"] + failed["cpu_seconds"]
        and record["current_machine_cumulative_gpu_wall_seconds"]
        == previous["current_machine_cumulative_gpu_wall_seconds"]
        and all(failed["resource_gates"].values()),
        "actual failed23 self-cost once",
    )
    previous, digest = record, current
    for directory, current in (
        ("ludb_qualification_v2", "9d97c0578fba75ba459fdab5796dd4fc81d735c9147c874d16c2d5f466995a0f"),
        ("ludb_annotation_diagnosis_v1", "cb86ee511b4740ab7b84a589fc81448556df3d5561907b4bd07e2f4748a020ba"),
    ):
        previous = next_stage(root, prefix / directory / "prospective_ledger.json", current, previous, digest)
        digest = current
    require(
        admission["prior_ledger"]["sha256"] == digest
        and previous["current_machine_cumulative_cpu_seconds"] == 435.78864790999995,
        "main uses actual diagnosis25",
    )
    return previous


def metadata_preflight(root: Path, source: Path, config_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify full metadata/catalog schemas without reading raw files or any saved score array."""
    require(file_sha256(source / CONFIG) == config_sha, "explicit reconciliation config identity")
    config = read_json(source / CONFIG)
    for key in (
        "main_summary",
        "main_ledger",
        "failed_audit_ledger",
        "failed_audit_summary",
        "main_admission",
    ):
        spec = config[key]
        directory = source if key == "main_admission" else root
        require(
            file_sha256(relative_file(directory, spec["path"])) == spec["sha256"], "immutable recovery input"
        )
    admission = read_json(source / config["main_admission"]["path"])
    prior = legacy_history(root, source, admission)
    main_spec = config["main_ledger"]
    main = next_stage(
        root,
        relative_file(root, main_spec["path"]),
        main_spec["sha256"],
        prior,
        admission["prior_ledger"]["sha256"],
    )
    require(
        main["current_machine_stages"] == 26
        and main["cpu_seconds"] < 600
        and main["wall_seconds"] < 1200
        and main["peak_rss_bytes"] < 2 * 1024**3,
        "actual bounded main26",
    )
    failed = read_json(root / config["failed_audit_ledger"]["path"])
    failure = read_json(root / config["failed_audit_summary"]["path"])
    require(
        failed["prior_ledger_sha256"] is None
        and failed["current_machine_stages"] is None
        and failed["current_machine_cumulative_cpu_seconds"] is None
        and failed["current_machine_cumulative_gpu_wall_seconds"] is None,
        "original failed27 null accounting retained",
    )
    require(
        failure["error_type"] == "KeyError"
        and failure["error"] == "'current_machine_stages'"
        and failure["all_passed"] is False
        and failed["sha256"] == config["failed_audit_summary"]["sha256"]
        and failed["gpu_wall_seconds"] == 0
        and all(failed["resource_gates"].values()),
        "actual pre-data failure27",
    )
    reconciled = config["reconciliation"]
    require(
        reconciled["current_machine_stages"] == 27
        and reconciled["current_machine_cumulative_cpu_seconds"]
        == main["current_machine_cumulative_cpu_seconds"] + failed["cpu_seconds"]
        and reconciled["current_machine_cumulative_gpu_wall_seconds"]
        == main["current_machine_cumulative_gpu_wall_seconds"]
        and reconciled["failed_audit_cpu_seconds_counted_once"] == failed["cpu_seconds"],
        "measured failed27 added exactly once",
    )
    study = root / frozen.STUDY
    result = read_json(root / config["main_summary"]["path"])
    for name, digest in result["source_sha256"].items():
        require(file_sha256(source / name) == digest, "all66 immutable executed main sources")
    files = main_files(study)
    require(
        set(files) - {"summary.json", "prospective_ledger.json"} == set(result["output_sha256"]),
        "immutable main manifest excludes both independent attempts",
    )
    require(
        sum(path.stat().st_size for path in files.values())
        - (study / "prospective_ledger.json").stat().st_size
        == main["output_bytes_before_terminal_ledger"],
        "actual main byte arithmetic excludes both attempts",
    )
    for name, digest in {**frozen.OWN_HASHES, **OWN_HASHES}.items():
        require(file_sha256(source / name) == digest, "all original and successor source bindings")
    spec = config["prior_reconciliation"]
    require(file_sha256(root / spec["path"]) == spec["sha256"], "actual setup reconciliation identity")
    actual = read_json(root / spec["path"])
    require(
        actual["kind"] == "setup_metadata_reconciliation_not_experiment"
        and actual["new_cost_stage_created"] is False
        and actual["original_failed_audit_nulls_preserved"] is True,
        "setup reconciliation preserves original failed27",
    )
    for key in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        require(actual[key] == reconciled[key], "actual reconciled prior27 costs")
    for key, source_key in (
        ("verified_main_accounting_ledger", "main_ledger"),
        ("retained_main_summary", "main_summary"),
        ("retained_failed_audit_ledger", "failed_audit_ledger"),
        ("retained_failed_audit_receipt", "failed_audit_summary"),
    ):
        require(actual[key] == config[source_key], "actual setup input binding")
    require(
        actual["failed_stage_cpu_seconds_counted_once"] == failed["cpu_seconds"],
        "actual failed cost counted once",
    )
    return config, actual


def identities(root: Path, source: Path, main: dict[str, Any], admission: dict[str, Any], guard: Any) -> int:
    """Verify original artifacts/official assets while filtering only the two named audit subdirectories."""
    study = root / frozen.STUDY
    for name, digest in main["source_sha256"].items():
        require(file_sha256(source / name, guard) == digest, "immutable executed main source")
        target = study / "source_snapshot" / name
        if name not in admission["source_sha256"]:
            target = study / "admission_snapshot.json"
        require(file_sha256(target, guard) == digest, "immutable executed source snapshot")
    files = main_files(study)
    require(
        set(files) - {"summary.json", "prospective_ledger.json"} == set(main["output_sha256"]),
        "unchanged original main artifact catalog",
    )
    for name, digest in main["output_sha256"].items():
        require(file_sha256(files[name], guard) == digest, "unchanged original main artifact hash")
    manifest = root / "outputs/ssl_synthetic_localization/ludb_acquisition_v4/SHA256SUMS.txt"
    require(
        file_sha256(manifest, guard) == "cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f",
        "original official checksum manifest",
    )
    hashes = {
        name.removeprefix("*").removeprefix("./"): digest
        for digest, name in (line.split(maxsplit=1) for line in manifest.read_text().splitlines())
    }
    expected = {
        f"{number}.{extension}" for number in range(1, 201) for extension in ("dat", "hea", *frozen.LEADS)
    }
    require(set(admission["asset_sha256"]) == expected, "all2800 original asset identities")
    data = frozen.raw_directory(root, admission)
    for name, digest in admission["asset_sha256"].items():
        guard()
        require(
            file_sha256(data / name, guard) == digest == hashes["data/" + name], "official raw asset identity"
        )
    require(
        file_sha256(manifest.parent / "release.zip", guard)
        == "d03192c7361ab5deeaba3f0d46e274b9ecb0b74c9caf8964151e20f1b5a7df06",
        "unchanged released archive",
    )
    return len(expected) + len(main["output_sha256"]) + 2 * len(main["source_sha256"]) + 2


def run(
    root: Path,
    source: Path,
    main_sha: str,
    entry_sha: str,
    runner_sha: str,
    config_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Execute one new recovery attempt using the unchanged scientific reconstruction functions."""
    destination = root / frozen.STUDY / "independent_audit_v2"
    destination.mkdir(exist_ok=False)
    prior: dict[str, Any] = {}
    prior_sha = None
    timers: dict[int, Any] = {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 598 and time.perf_counter() - clock["wall"] < 1198,
            "recovery receipt reserve",
        )
        require(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "recovery RSS ceiling"
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent audit recovery timer {signum}")

    try:
        for timer, signum, limit, key in (
            (signal.ITIMER_REAL, signal.SIGALRM, 1198, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 598, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            elapsed = (time.perf_counter() if key == "wall" else time.process_time()) - clock[key]
            signal.setitimer(timer, max(0.01, limit - elapsed))
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(
                os.environ.get(name) == "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            ),
            "hidden CUDA and one numerical thread",
        )
        require(
            file_sha256(Path(__file__)) == runner_sha and file_sha256(source / CHECKER) == entry_sha,
            "literal new executable source identities",
        )
        config, prior = metadata_preflight(root, source, config_sha)
        prior_sha = config["prior_reconciliation"]["sha256"]
        own = {**frozen.OWN_HASHES, **OWN_HASHES, CONFIG: config_sha, RUNNER: runner_sha, CHECKER: entry_sha}
        for name, digest in own.items():
            require(file_sha256(source / name) == digest, "all old/new independent source identities")
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        require(main_sha == config["main_summary"]["sha256"], "same immutable main26 summary")
        main = read_json(root / config["main_summary"]["path"])
        admission = read_json(source / config["main_admission"]["path"])
        checks = identities(root, source, main, admission, guard)
        old_checks = frozen.predecessor(root, root / frozen.STUDY, main)
        values = frozen.numerical(root, root / frozen.STUDY, admission, main, guard)
        frozen.figure_checks(root / frozen.STUDY)
        checks += identities(root, source, main, admission, guard)
        require(
            main["technical_passed_before_independent_audit"]
            and main["independent_audit_required"]
            and main["clinical_claim"] is False
            and main["measurement_AP_values"] == main["pathology_masks"] == main["parser_metric_values"] == 0,
            "unchanged landmark-only main flags",
        )
        closure = {
            "all_exact": True,
            "lead_streams": 12,
            "total_events": 571,
            "N_events": 107,
            "N_without_immediate_onset": 35,
            "before_measurement_records": True,
        }
        compare_values(closure, main["diagnostic_closure"], "exact95 prerequisite", 0)
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "numerical": values,
            "scientific_passed": values["statistics"]["scientific_passed"],
            "identity_checks": checks,
            "predecessor_checks": old_checks,
            "main_summary_sha256": main_sha,
            "reconciliation": config["prior_reconciliation"],
            "reconciled_prior": prior,
            "source_sha256": own,
            "clinical_validation": False,
            "gpu_used": False,
            "annotation_verification_scope": "independent MIT byte events versus saved main WFDB rows",
        }
    except Exception as error:
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    frozen.finalize(destination, result, clock, prior, prior_sha, timers)
    return result
