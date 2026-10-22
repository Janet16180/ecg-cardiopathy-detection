"""Narrow recovery of the immutable mixed audit's single-record functional caller."""

import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_mixed_background_audit_run as frozen
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = frozen.CONFIG
STUDY = frozen.STUDY
RUNNER = "ecg_experiment/ssl_synthetic_mixed_background_audit_run_v2.py"
CHECKER = "scripts/validation/ssl_synthetic_mixed_background_audit_v2.py"
PROTOCOL = "docs/ssl-synthetic-localization-mixed-background-audit-recovery-v2.md"
TESTS = "tests/test_ssl_synthetic_mixed_background_audit_runtime_v2.py"
RECOVERY = "configs/ssl_synthetic_mixed_background_audit_recovery_v2.json"
AUDIT_DIRS = {"independent_audit_v1", "independent_audit_v2"}
PRIOR37_SHA, PRIOR37_RECEIPT = frozen.PRIOR37_SHA, frozen.PRIOR37_RECEIPT
core, inputs, previous = frozen.core, frozen.inputs, frozen.previous
predecessor = frozen.predecessor
profile_training = frozen.profile_training
profile_forecast = frozen.profile_forecast
profile_maps = frozen.profile_maps
operation_checks = frozen.operation_checks
binding_path = frozen.binding_path
OLD_MAIN = "configs/ssl_synthetic_mixed_background_admission_v1.json"
OLD_AUDIT = "configs/ssl_synthetic_mixed_background_audit_admission_v1.json"


def source_union(main: dict[str, Any], audit: dict[str, Any]) -> dict[str, str]:
    """Bind the recovery overlay while preserving every executed original-source obligation.

    Parameters
    ----------
    main, audit : dict
        Original executed main and separately admitted recovery source declarations.

    Returns
    -------
    dict
        Complete no-override source roster including both historical admissions.
    """
    require(
        {RUNNER, CHECKER, PROTOCOL, TESTS, RECOVERY, OLD_MAIN, OLD_AUDIT}.issubset(audit["source_sha256"]),
        "explicit complete recovery and historical-admission source roster",
    )
    return frozen.source_union(main, audit)


def record_replay(state: dict[str, Any], record: np.ndarray) -> np.ndarray:
    """Adapt exactly one declared model record to the unchanged batched functional helper.

    Parameters
    ----------
    state : dict
        Plain six-tensor float32 CPU model state.
    record : np.ndarray
        One float32 canonical12×5000 model input.

    Returns
    -------
    np.ndarray
        The singleton batch's complete float32 canonical probability map.
    """
    require(
        record.shape == (12, 5000) and record.dtype == np.float32, "single declared model-record geometry"
    )
    output = frozen.functional_replay(state, record[None])
    require(output.shape == (1, 12, 5000) and output.dtype == np.float32, "single batched functional output")
    return output[0]


def latest_accounting(root: Path, source: Path, study: Path, main: dict) -> dict[str, Any]:
    """Verify preserved failed39 and append its real cost once after the actual main38.

    Parameters
    ----------
    root, source, study : Path
        Explicit artifact, source and immutable main output roots.
    main : dict
        Original executed main receipt.

    Returns
    -------
    dict
        Actual39 as the verified immediate prior for recovery40, without reconciliation.
    """
    recovery = read_json(source / RECOVERY)
    costs = main_accounting(root, study, main)
    require(
        file_sha256(study / "summary.json") == recovery["main_summary_sha256"]
        and file_sha256(study / "prospective_ledger.json") == recovery["main_ledger_sha256"],
        "unchanged actual main38 identities",
    )
    for name in ("main_admission", "failed_audit_admission"):
        declaration = recovery[name]
        require(
            file_sha256(source / declaration["path"]) == declaration["sha256"],
            "original executed main and failed-audit admission identities",
        )
    original = read_json(source / recovery["main_admission"]["path"])
    failed_admission = read_json(source / recovery["failed_audit_admission"]["path"])
    failed = study / recovery["failed_directory"]
    require(
        file_sha256(failed / "summary.json") == recovery["failed_summary_sha256"]
        and file_sha256(failed / "prospective_ledger.json") == recovery["failed_ledger_sha256"],
        "original failed39 receipt and ledger unchanged",
    )
    receipt = read_json(failed / "summary.json")
    ledger = read_json(failed / "prospective_ledger.json")
    require(
        receipt
        == {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": "ValueError",
            "error": "Require float32 batched actual native-rate model inputs",
        },
        "exact preserved functional-call failure",
    )
    require(
        ledger["sha256"] == recovery["failed_summary_sha256"]
        and not ledger["technical_passed"]
        and ledger["prior_accounting_verified"]
        and all(ledger["resource_gates"].values())
        and ledger["prior_ledger_sha256"] == costs["prior_ledger_sha256"],
        "actual failed39 directly follows verified actual main38",
    )
    require(
        ledger["cpu_seconds"] == recovery["failed_cpu_seconds"] and ledger["gpu_wall_seconds"] == 0,
        "actual once-only failed CPU and zero GPU",
    )
    require(
        ledger["current_machine_stages"] == costs["stages_through_main"] + 1 == recovery["prior_stage"]
        and ledger["current_machine_cumulative_cpu_seconds"]
        == costs["cpu_through_main"] + ledger["cpu_seconds"]
        == recovery["prior_cpu_seconds"]
        and ledger["current_machine_cumulative_gpu_wall_seconds"]
        == costs["gpu_through_main"]
        == recovery["prior_gpu_wall_seconds"],
        "exact actual38 to failed39 additive lineage",
    )
    sources = {
        **frozen.source_union(original, failed_admission),
        recovery["main_admission"]["path"]: recovery["main_admission"]["sha256"],
        recovery["failed_audit_admission"]["path"]: recovery["failed_audit_admission"]["sha256"],
    }
    for name, digest in sources.items():
        require(
            file_sha256(failed / "source_snapshot" / name) == digest,
            "complete retained failed39 source snapshot",
        )
    retained = sum(
        p.stat().st_size
        for p in failed.rglob("*")
        if p.is_file() and not p.is_symlink() and p != failed / "prospective_ledger.json"
    )
    require(retained == ledger["output_bytes_before_terminal_ledger"], "preserved failed39 retained bytes")
    return {
        "prior_ledger_sha256": recovery["failed_ledger_sha256"],
        "stages_through_main": ledger["current_machine_stages"],
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def main_accounting(root: Path, study: Path, main: dict[str, Any]) -> dict[str, Any]:
    """Verify actual37 through actual main with exact once-only cumulative resource additions.

    Parameters
    ----------
    root, study : Path
        Baseline artifact root and executed main directory.
    main : dict
        Actual immutable main summary.

    Returns
    -------
    dict
        Verified immediate-main accounting anchor for the single independent attempt.
    """
    prior_root = root / "outputs/ssl_synthetic_localization/background_editor_transfer_figures_v1"
    require(file_sha256(prior_root / "prospective_ledger.json") == PRIOR37_SHA, "actual37 ledger identity")
    require(file_sha256(prior_root / "summary.json") == PRIOR37_RECEIPT, "actual37 receipt identity")
    prior = read_json(prior_root / "prospective_ledger.json")
    ledger = read_json(study / "prospective_ledger.json")
    require(
        prior["current_machine_stages"] == 37
        and prior["technical_passed"]
        and all(prior["resource_gates"].values()),
        "actual37 completed resource gates",
    )
    require(
        prior["current_machine_cumulative_cpu_seconds"] == 1076.7763161960002
        and prior["current_machine_cumulative_gpu_wall_seconds"] == 524.1993697541766,
        "actual37 exact cumulative costs",
    )
    require(
        ledger["prior_ledger_sha256"] == PRIOR37_SHA
        and ledger["prior_accounting_verified"]
        and ledger["current_machine_stages"] == 38
        and ledger["technical_passed"]
        and all(ledger["resource_gates"].values()),
        "actual main38 verified predecessor and resources",
    )
    require(
        ledger["receipt_sha256"] == file_sha256(study / "summary.json")
        and main["status"] == "PASS"
        and main["technical_passed_before_independent_audit"],
        "actual completed main receipt",
    )
    require(
        ledger["cpu_seconds"] < 1800
        and ledger["wall_seconds"] < 2400
        and ledger["gpu_wall_seconds"] < 900
        and ledger["peak_rss_bytes"] < 2 * 1024**3
        and ledger["output_bytes_before_terminal_ledger"] + 1024**2 < 1024**3,
        "independently checked actual main hard resource ceilings",
    )
    for cost in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + cost
        require(ledger[key] == prior[key] + ledger[cost], "exact additive main resource cost")
    retained = sum(
        p.stat().st_size
        for p in study.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and not AUDIT_DIRS.intersection(p.relative_to(study).parts)
        and p.name != "prospective_ledger.json"
    )
    require(retained == ledger["output_bytes_before_terminal_ledger"], "exact retained main bytes")
    return {
        "prior_ledger_sha256": file_sha256(study / "prospective_ledger.json"),
        "stages_through_main": ledger["current_machine_stages"],
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def identities(root: Path, source: Path, study: Path, main: dict, admission: dict, guard: Any) -> int:
    """Rehash complete original source/input and saved-main catalogs with the audit subtree excluded.

    Parameters
    ----------
    root, source, study : Path
        Explicit artifact, source and executed main roots.
    main, admission : dict
        Actual main summary and unchanged original admission.
    guard : callable
        Live whole-audit resource guard.

    Returns
    -------
    int
        Complete exact byte-identity check count.
    """
    count = 0
    for name, digest in main["source_sha256"].items():
        require(file_sha256(relative_file(source, name), guard) == digest, "current main source bytes")
        count += 1
    for name, digest in admission["input_sha256"].items():
        require(file_sha256(binding_path(root, source, name), guard) == digest, "admitted input bytes")
        count += 1
    for name, digest in main["output_sha256"].items():
        require(
            file_sha256(relative_file(study, name), guard) == digest, "complete saved main artifact bytes"
        )
        count += 1
    actual = {
        p.relative_to(study).as_posix()
        for p in study.rglob("*")
        if p.is_file() and not p.is_symlink() and not AUDIT_DIRS.intersection(p.relative_to(study).parts)
    }
    require(
        actual == set(main["output_sha256"]) | {"summary.json", "prospective_ledger.json"},
        "complete immutable main artifact membership",
    )
    return count


def metadata_setup(root: Path, source: Path, main_name: str, audit_name: str) -> tuple[dict, dict, dict]:
    """Run complete source and actual-container preflight before the charged numerical audit.

    Parameters
    ----------
    root, source : Path
        Artifact and source checkout roots.
    main_name, audit_name : str
        Original executed and separate independent identity declarations.

    Returns
    -------
    tuple
        Original main admission, separate audit declaration and frozen config.
    """
    main, audit = read_json(source / main_name), read_json(source / audit_name)
    recovery = read_json(source / RECOVERY)
    require(main_name == recovery["main_admission"]["path"] == OLD_MAIN, "original main admission retained")
    require(
        recovery["cpu_seconds"] == 600
        and recovery["wall_seconds"] == 900
        and recovery["functional_cpu_replays"] == 24
        and recovery["functional_atol"] == 1e-5
        and recovery["score_atol"] == 1e-12,
        "unchanged recovery budgets/replay/tolerances",
    )
    union = source_union(main, audit)
    for name, digest in union.items():
        require(file_sha256(relative_file(source, name)) == digest, "complete immutable runtime source")
    config = read_json(source / CONFIG)
    require(
        config["bootstrap"]["seed"] == 2026106701 and config["bootstrap"]["draws"] == 2000,
        "fixed protocol/bootstrap",
    )
    study = root / STUDY
    summary = read_json(study / "summary.json")
    require(file_sha256(study / "summary.json") == audit["main_summary_sha256"], "actual main identity")
    require(
        file_sha256(study / "prospective_ledger.json") == audit["main_ledger_sha256"],
        "main terminal identity",
    )
    main_accounting(root, study, summary)
    expected = {**main["source_sha256"], main_name: file_sha256(source / main_name)}
    require(summary["source_sha256"] == expected, "actual main used original immutable admission")
    for name, digest in expected.items():
        candidate = study / "source_snapshot" / name
        if not candidate.exists() and name == main_name:
            candidate = study / "admission_snapshot.json"
        require(file_sha256(candidate) == digest, "executed main source snapshot")
    local = read_json(root / config["cohort_manifest"])
    require(
        len(local["records"]) == 40 and len(local["eligible_candidates"]) >= 40, "local cohort JSON schema"
    )
    frozen.training_schema(study)
    frozen.evaluation_schema(study)
    latest_accounting(root, source, study, summary)
    return main, audit, config


def numerical(root: Path, study: Path, config: dict, states: dict, guard: Any) -> dict[str, Any]:
    """Reconstruct every paired model-background metric and the24 fixed CPU forward checks.

    Parameters
    ----------
    root, study : Path
        Baseline artifact root and new main output directory.
    config : dict
        Frozen observations, masks, fits and bootstrap/replay configuration.
    states : dict
        Complete independently verified fitted-state identities.
    guard : callable
        Whole-audit resource guard.

    Returns
    -------
    dict
        Full independent statistics, comparison counts and CPU replay discrepancies.
    """
    evaluation = root / config["evaluation_root"]
    observed = previous.array(evaluation / "inputs.npy", (2, 64, 6, 12, 5000), np.float32)
    masks = previous.array(evaluation / "support.npy", (2, 64, 4, 12, 5000), np.bool_)
    strict = previous.array(evaluation / "strict_support.npy", masks.shape, np.bool_)
    saved_rows = previous.rows(study / "record_rows.jsonl")
    saved = {(r["training_arm"], r["source"], r["slot"], r["fit_id"]): r for r in saved_rows}
    all_rows, cpu_replays = [], []
    scalar_checks = 0
    for arm in core.TRAINING_ARMS:
        directory = evaluation if arm == "generated" else study
        suffix = "B" if arm == "generated" else "mixed"
        for fit in (1, 2, 3):
            maps = previous.array(
                directory / "probabilities" / f"fit_{fit}_{suffix}.npy", (2, 64, 6, 12, 5000), np.float32
            )
            state = inputs.load_state(study / f"fit_{fit}/mixed/B/final_state.pt") if arm == "mixed" else None
            for sourcepos, source_name in enumerate(core.transfer.SOURCES):
                for slot in range(64):
                    guard()
                    delta = observed[sourcepos, slot, 2:].astype(np.float64) - observed[sourcepos, slot, 1]
                    require(
                        np.array_equal(delta != 0, strict[sourcepos, slot]), "complete original strict truth"
                    )
                    require(
                        np.array_equal(np.abs(delta) > 0.01, masks[sourcepos, slot]),
                        "complete final threshold truth",
                    )
                    row = core.transfer.score_record(
                        maps[sourcepos, slot],
                        observed[sourcepos, slot],
                        masks[sourcepos, slot],
                        source_name,
                        slot,
                        fit,
                        "B",
                        masks[sourcepos, (slot + 1) % 64],
                    )
                    row["training_arm"] = arm
                    scalar_checks += compare_values(
                        row,
                        saved[arm, source_name, slot, fit],
                        "complete independent model-background row",
                        tolerance=1e-12,
                    )
                    all_rows.append(row)
                    if arm == "mixed" and slot in (0, 63):
                        for variant in (2, 5):
                            replay = record_replay(state, observed[sourcepos, slot, variant])
                            difference = float(
                                np.max(np.abs(replay.astype(np.float64) - maps[sourcepos, slot, variant]))
                            )
                            require(difference <= 1e-5, "fixed independent mixed CPU model-record replay")
                            cpu_replays.append(
                                {
                                    "fit_id": fit,
                                    "source": source_name,
                                    "slot": slot,
                                    "variant": variant,
                                    "maximum_absolute_difference": difference,
                                }
                            )
            del maps, state
    require(len(cpu_replays) == 24, "exact declared CPU functional replay roster")
    statistics, draws = core.summarize(all_rows, config)
    scalar_checks += compare_values(
        statistics,
        read_json(study / "summary.json")["statistics"],
        "complete paired training report",
        tolerance=1e-12,
    )
    saved_draws = previous.array(study / "bootstrap.npy", (2000, 348), np.float64)
    maximum = float(np.max(np.abs(draws - saved_draws)))
    require(maximum <= 1e-12, "all2000x348 complete named-column bootstrap draws")
    return {
        "statistics": statistics,
        "scalar_checks": scalar_checks,
        "rows": len(all_rows),
        "positive_rows": len(all_rows) * 4,
        "bootstrap_shape": list(draws.shape),
        "maximum_bootstrap_absolute_difference": maximum,
        "cpu_replays": cpu_replays,
        "verified_tensor_states": states,
    }


def terminal(destination: Path, result: dict, clock: dict, costs: dict, timers: dict) -> dict[str, bool]:
    """Preserve any failure and append actual independent costs through final receipt bookkeeping.

    Parameters
    ----------
    destination : Path
        Exclusive audit directory; old main and failed outputs remain unchanged.
    result, clock, costs, timers : dict
        Outcome, whole-process clocks, verified immediate predecessor and active timers.

    Returns
    -------
    dict
        Actual terminal resource-gate conjunction.
    """
    receipt, digest, retained = "summary.json", None, None
    try:
        previous.write_new(destination / receipt, result)
        digest = file_sha256(destination / receipt)
        retained = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file() and not p.is_symlink())
    except Exception as error:
        for timer in timers:
            signal.setitimer(timer, 0)
        receipt = "finalization_failure.json"
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "retained_byte_count_exact": False,
        }
        previous.write_new(destination / receipt, result)
        digest = file_sha256(destination / receipt)
    cpu = time.process_time() - clock["cpu"]
    wall = time.perf_counter() - clock["wall"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "cpu": cpu < 600,
        "wall": wall < 900,
        "rss": rss < 2 * 1024**3,
        "output": retained is not None and retained + 16384 < 64 * 1024**2,
    }
    verified = bool(costs)
    ledger = {
        "stage": "mixed_background_independent_audit_v2",
        "receipt": receipt,
        "sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "peak_rss_bytes": rss,
        "resource_gates": gates,
        "prior_accounting_verified": verified,
        "prior_ledger_sha256": costs.get("prior_ledger_sha256"),
        "current_machine_stages": costs["stages_through_main"] + 1 if verified else None,
        "current_machine_cumulative_cpu_seconds": costs["cpu_through_main"] + cpu if verified else None,
        "current_machine_cumulative_gpu_wall_seconds": costs.get("gpu_through_main"),
        "output_bytes_before_terminal_ledger": retained,
        "historical_costs_verified": False,
        "measurement_endpoint": "after receipt hash and byte enumeration; before ledger serialization",
        "terminal_bookkeeping_reserve_seconds": 2,
        "terminal_bookkeeping_reserve_bytes": 16384,
        "technical_passed": result["all_passed"] and all(gates.values()),
    }
    for timer in timers:
        signal.setitimer(timer, 0)
    previous.write_new(destination / "prospective_ledger.json", ledger)
    return gates


def run(
    root: Path,
    source: Path,
    main_name: str,
    audit_name: str,
    main_sha: str,
    audit_sha: str,
    runner_sha: str,
    checker_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Execute one prospectively source-admitted full independent reconstruction and24 CPU replays.

    Parameters
    ----------
    root, source : Path
        Explicit baseline artifact and isolated immutable source roots.
    main_name, audit_name : str
        Original main and separate independent identity admissions.
    main_sha, audit_sha, runner_sha, checker_sha : str
        Literal expected actual main/admission/runtime/entry-point SHA256 values.
    clock : dict
        Whole-audit CPU and monotonic wall starts captured before scientific imports.

    Returns
    -------
    dict
        Output-derived independent report; failures and actual costs are retained before raising.
    """
    study = root / STUDY
    destination = study / "independent_audit_v2"
    destination.mkdir(exist_ok=False)
    costs, timers = {}, {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 598 and time.perf_counter() - clock["wall"] < 898,
            "audit receipt time reserve",
        )
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "audit RSS")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent mixed-background audit timer:{signum}")

    try:
        for timer, signum, ceiling, key in (
            (signal.ITIMER_REAL, signal.SIGALRM, 898, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 598, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            spent = (time.perf_counter() if key == "wall" else time.process_time()) - clock[key]
            signal.setitimer(timer, max(0.01, ceiling - spent))
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(
                os.environ.get(k) == "1"
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            ),
            "hidden CUDA and all numerical thread limits1",
        )
        require(
            file_sha256(Path(__file__)) == runner_sha
            and file_sha256(source / CHECKER) == checker_sha
            and file_sha256(source / audit_name) == audit_sha,
            "literal independent executable identities",
        )
        require(file_sha256(study / "summary.json") == main_sha, "literal actual main summary identity")
        main = read_json(study / "summary.json")
        audit = read_json(source / audit_name)
        require(
            file_sha256(source / RECOVERY) == audit["source_sha256"][RECOVERY],
            "prospectively admitted separate recovery config",
        )
        costs = latest_accounting(root, source, study, main)
        admission, audit, config = metadata_setup(root, source, main_name, audit_name)
        own = {
            **source_union(admission, audit),
            main_name: file_sha256(source / main_name),
            audit_name: audit_sha,
        }
        require(
            sum((source / name).stat().st_size for name in own) + 1024**2 < 64 * 1024**2,
            "bounded complete source snapshots plus report reserve",
        )
        for name, digest in own.items():
            guard()
            require(file_sha256(source / name, guard) == digest, "immutable independent runtime source")
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        bindings = identities(root, source, study, main, admission, guard)
        closure_checks = predecessor(root, source, study, config, guard)
        banks = inputs.training_banks(root, study, config, guard)
        models = inputs.training_models(root, study, config, guard)
        require(
            main["training_bank_qualification"] == banks["training_bank_qualification"],
            "main complete training-bank qualification matches independent reconstruction",
        )
        training = profile_training(study, guard)
        profile = profile_forecast(study, config, training)
        profile["scalar_checks"] = profile_maps(root, study, config, models, guard)
        values = numerical(root, study, config, models, guard)
        operation_checks(study)
        from PIL import Image

        with Image.open(study / "all_source_family_and_control_contrasts.png") as figure:
            require(
                figure.format == "PNG" and figure.size == (1680, 720), "figure byte/signature/layout scope"
            )
        bindings += identities(root, source, study, main, admission, guard)
        _, final_raw_checks = inputs.declared_cohort(root, config, guard)
        for name, digest in own.items():
            require(
                file_sha256(source / name, guard) == digest,
                "complete independent source unchanged at termination",
            )
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "scientific_passed": values["statistics"]["scientific_passed"],
            "main_summary_sha256": main_sha,
            "main_ledger_sha256": audit["main_ledger_sha256"],
            "audit_admission_sha256": audit_sha,
            "source_sha256": own,
            "binding_checks": bindings,
            "predecessor_scalar_checks": closure_checks,
            "training_inputs": banks,
            "final_raw_identity_checks": final_raw_checks,
            "profile": profile,
            "numerical": values,
            "gpu_used": False,
            "new_training_updates": 0,
            "raw_training_record_decodes": 40,
            "raw_LUDB_reads": 0,
            "clinical_labels_interpreted": 0,
            "clinical_claim": False,
            "figure_scope": "hashes and PNG signature/layout, no pixel rerender or clinical interpretation",
        }
    except Exception as error:
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        gates = terminal(destination, result, clock, costs, timers)
    finally:
        for timer, (signum, previous_handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous_handler)
    require(
        result["all_passed"] and all(gates.values()),
        "retained independent mixed-background recovery audit failure",
    )
    return result
