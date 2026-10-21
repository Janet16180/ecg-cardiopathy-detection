"""Retain audit32 and normalize saved timing record order before unchanged independent reconstruction."""

from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_localizer_audit as frozen
from ecg_experiment import ssl_synthetic_ludb_localizer_audit_v2 as previous

CONFIG = frozen.CONFIG
RUNNER = "ecg_experiment/ssl_synthetic_ludb_localizer_audit_v3.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_localizer_v3.py"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-localizer-independent-audit-v3.md"
TESTS = "tests/test_ssl_synthetic_ludb_localizer_audit_v3.py"


def canonical_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize record-object order only, retaining every object's event and landmark lists.

    Parameters
    ----------
    rows : list
        All200 unchanged saved record objects, possibly in parser-first order.

    Returns
    -------
    list
        Identical objects ordered by numeric ID1..200 with no exclusions or duplication.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require

    ordered = sorted(rows, key=lambda row: row["record_id"])
    require(
        [row["record_id"] for row in ordered] == list(range(1, 201)),
        "all200 original record objects exactly once",
    )
    return ordered


def main_accounting(
    root: Any, study: Any, admission: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate actual GPU-main additive costs and the immutable output catalog before own scoring.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    admission : dict[str, Any]
        Declared frozen input values.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    main = read_json(study / "summary.json")
    path = study / "prospective_ledger.json"
    ledger = read_json(path)
    prior = admission["prior_ledger"]
    require(
        ledger["receipt_sha256"] == file_sha256(study / "summary.json")
        and ledger["prior_ledger_sha256"] == prior["sha256"]
        and ledger["prior_accounting_verified"]
        and ledger["current_machine_stages"] == 31,
        "actual main31 receipt and immediately verified prior30",
    )
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + suffix
        require(
            ledger[key] == prior[key] + ledger[suffix] and ledger[suffix] >= 0,
            "exact once-only GPU-main actual costs",
        )
    require(
        ledger["technical_passed"]
        and ledger["status"] == "PASS"
        and all(ledger["resource_gates"].values())
        and main["status"] == "PASS"
        and main["technical_passed_before_independent_audit"],
        "completed main31 technical resource conjunction",
    )
    limits = config["resources"]
    require(
        ledger["cpu_seconds"] < limits["cpu_seconds"]
        and ledger["wall_seconds"] < limits["wall_seconds"]
        and ledger["gpu_wall_seconds"] < limits["gpu_lock_wall_seconds"]
        and ledger["peak_rss_bytes"] < limits["peak_rss_bytes"],
        "main measured resource ceilings",
    )
    files = {
        path.relative_to(study).as_posix(): path
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not {"independent_audit_v1", "independent_audit_v2", "independent_audit_v3"}.intersection(
            path.relative_to(study).parts
        )
    }
    require(
        set(files) == set(main["output_sha256"]) | {"summary.json", "prospective_ledger.json"},
        "complete main retained file catalog",
    )
    require(
        sum(path.stat().st_size for name, path in files.items() if name != "prospective_ledger.json")
        == ledger["output_bytes_before_terminal_ledger"],
        "main byte arithmetic excludes only its terminal ledger",
    )
    require(
        ledger["output_bytes_before_terminal_ledger"] + 16384 < limits["output_bytes"],
        "main retained byte reserve",
    )
    return {
        "prior_ledger_sha256": file_sha256(path),
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
        "stages_through_main": 31,
    }


def predecessor_closures(root: Any, study: Any, guard: Any) -> dict[str, Any]:
    """Reconstruct all old synthetic metrics and both real-study bootstrap arrays independently.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_ludb_coherence_audit import summarize
    from ecg_experiment.ssl_synthetic_ludb_landmark_audit import aggregate as point_aggregate
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.ssl_synthetic_replication_saved_audit import (
        hotspot_rows,
        panel_controls,
        panel_metrics,
    )
    from ecg_experiment.ssl_synthetic_suppression_saved_audit import control_top_values, suppression_summary
    from ecg_experiment.synthetic_audit_io import compare_values, read_json

    parent = root / frozen.MODELS
    observed = np.load(parent / "fresh_observed.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(parent / "fresh_support.npy", mmap_mode="r", allow_pickle=False)
    prior = np.load(
        root / "outputs/ssl_synthetic_localization/randomized_localization_v1/spatial_prior.npy",
        allow_pickle=False,
    )
    fits, checks = [], 0
    for fit in frozen.FITS:
        record = {"fit_id": fit}
        for arm in frozen.ARMS:
            guard()
            directory = parent / f"fit_{fit}" / arm
            probability = np.load(directory / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False)
            cases = panel_metrics(probability, observed, support, prior, list(range(2000, 2064)))
            hotspots = hotspot_rows(probability, support, list(range(2000, 2064)))
            means = {
                "sham_nuisance_persistence": float(
                    np.mean([row["sham_nuisance_persistence"] for row in hotspots])
                )
            }
            for mode in ("focal", "repeated"):
                means[mode] = {
                    key: float(np.mean([row[mode][key] for row in hotspots])) for key in hotspots[0][mode]
                }
            metrics = {
                "cases": cases,
                "controls": panel_controls(probability),
                "hotspots": {"cases": hotspots, "means": means},
                "top600_control": control_top_values(probability),
            }
            checks += compare_values(
                metrics, read_json(directory / "fresh_metrics.json"), "independent complete synthetic panel"
            )
            record[arm] = metrics
        fits.append(record)
    synthetic = suppression_summary(fits)
    checks += compare_values(synthetic, read_json(parent / "summary.json")["aggregate"], "old full aggregate")
    checks += compare_values(
        synthetic,
        read_json(study / "synthetic_predecessor_closure.json")["aggregate"],
        "new saved complete synthetic closure",
    )
    timing = point_aggregate(canonical_records(frozen.read_rows(root / frozen.TIMING / "record_rows.jsonl")))
    point_draws = timing.pop("bootstrap_gains")
    checks += compare_values(
        timing,
        read_json(root / frozen.TIMING / "summary.json")["statistics"],
        "independent exact timing closure",
        0,
    )
    checks += compare_values(
        timing, read_json(study / "timing_predecessor_closure.json")["statistics"], "new timing closure", 0
    )
    require(
        np.array_equal(
            point_draws, np.load(root / frozen.TIMING / "bootstrap_gains.npy", allow_pickle=False)
        ),
        "all2000 original released-N draws",
    )
    coherence, coherence_draws = summarize(
        frozen.read_rows(root / frozen.COHERENCE / "generated_rows.jsonl"),
        frozen.read_rows(root / frozen.COHERENCE / "ludb_rows.jsonl"),
    )
    checks += compare_values(
        coherence,
        read_json(root / frozen.COHERENCE / "summary.json")["statistics"],
        "independent coherence closure",
        0,
    )
    checks += compare_values(
        coherence,
        read_json(study / "coherence_predecessor_closure.json")["statistics"],
        "new coherence closure",
        0,
    )
    require(
        np.array_equal(
            coherence_draws,
            np.load(root / frozen.COHERENCE / "bootstrap_cohort_means_and_contrasts.npy", allow_pickle=False),
        ),
        "all2000 original coherence draws",
    )
    return {
        "scalar_checks": checks,
        "synthetic_model_panels": 12,
        "synthetic_case_rows": 1536,
        "timing_bootstrap_draws": 2000,
        "coherence_bootstrap_draws": 2000,
        "original_synthetic_scientific_failure_preserved": True,
    }


def run(
    root: Any,
    source: Any,
    admission_name: str,
    main_sha: str,
    entry_sha: str,
    runner_sha: str,
    recovery_name: str,
    recovery_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Run one fully admitted independent CPU audit and preserve actual failure and measured cost.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    source : Any
        Isolated checkout containing the committed executable sources.
    admission_name : str
        Repository-relative committed identity admission.
    main_sha : str
        Literal SHA256 of the completed main summary.
    entry_sha : str
        Literal SHA256 of the independent CLI.
    runner_sha : str
        Literal SHA256 of this independent module.
    clock : dict[str, float]
        Whole-audit CPU and monotonic wall start values.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    import os
    import resource
    import signal
    import time
    from pathlib import Path

    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    config = read_json(source / CONFIG)
    study = root / config["output_root"]
    destination = study / "independent_audit_v3"
    destination.mkdir(exist_ok=False)
    costs, timers = {}, {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 898 and time.perf_counter() - clock["wall"] < 1198,
            "independent localizer receipt time reserve",
        )
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "independent RSS")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent localizer timer {signum}")

    try:
        for timer, signum, ceiling, kind in (
            (signal.ITIMER_REAL, signal.SIGALRM, 1198, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 898, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            elapsed = (time.process_time() if kind == "cpu" else time.perf_counter()) - clock[kind]
            signal.setitimer(timer, max(0.01, ceiling - elapsed))
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
            "hidden CUDA and four single-thread runtime limits",
        )
        require(
            file_sha256(Path(__file__)) == runner_sha and file_sha256(source / CHECKER) == entry_sha,
            "literal independent executable identities",
        )
        admission, config = metadata_setup(root, source, admission_name, recovery_name, recovery_sha)
        require(file_sha256(study / "summary.json") == main_sha, "literal actual main31 summary")
        recovery = read_json(source / recovery_name)
        costs = verify_chain(root, study, source, admission, config, recovery)
        own_sources = {
            **admission["source_sha256"],
            **recovery["source_sha256"],
            admission_name: file_sha256(source / admission_name),
            recovery_name: recovery_sha,
        }
        for name in own_sources:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        identity_checks = frozen.input_identities(root, admission, guard)
        identity_checks += frozen.output_identities(study, source, admission, admission_name, guard)
        predecessors = predecessor_closures(root, study, guard)
        profile = frozen.profile_checks(root, study, config, guard)
        values = frozen.numerical(root, study, config, guard)
        main = read_json(study / "summary.json")
        require(
            main["generator_forwards"]
            == main["optimizer_updates"]
            == main["original_annotation_stream_reads"]
            == main["parser_records_scored"]
            == 0
            and main["real_record_reads"] == 168
            and main["clinical_claim"] is False
            and main["no_training"] is True,
            "fixed nonclinical inference-only operation scope",
        )
        require(
            main["operation_counts"]
            == {
                "synthetic_replay_forwards": 12,
                "profile_batch8_forwards": 12,
                "full_batch8_forwards": 240,
                "model_record_rows": 1920,
            },
            "complete fixed inference operation counts",
        )
        require(
            main["tensor_state_sha256"]
            == {
                name: row["initial_tensor_state_sha256"]
                for name, row in read_json(study / "model_states.json").items()
            },
            "main receipt contains the same twelve original tensor digests",
        )
        require(
            main["runtime"]["torch_threads"] == main["runtime"]["interop_threads"] == 1,
            "single-thread recorded neural runtime",
        )
        require(
            "scientific_passed" not in main and "scientific_passed" not in main["statistics"],
            "descriptive localizer diagnostic has no post-hoc scientific-success threshold",
        )
        from PIL import Image

        for name, dimensions in (
            ("profile/profile_fixed_maps.png", (1920, 960)),
            ("all_model_landmark_and_activity.png", (2160, 720)),
            ("fixed_input_map_N_overlays.png", (2160, 2400)),
        ):
            with Image.open(study / name) as figure:
                require(figure.format == "PNG" and figure.size == dimensions, "fixed figure signature/layout")
        identity_checks += frozen.input_identities(root, admission, guard)
        identity_checks += frozen.output_identities(study, source, admission, admission_name, guard)
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "main_summary_sha256": main_sha,
            "source_sha256": own_sources,
            "predecessors": predecessors,
            "profile": profile,
            "numerical": values,
            "identity_checks": identity_checks,
            "clinical_claim": False,
            "gpu_used": False,
            "new_annotation_stream_reads": 0,
            "recovery_config_sha256": recovery_sha,
            "retained_failed_audit_ledger_sha256": recovery["failed_audit"]["ledger_sha256"],
            "figure_scope": "hashes, PNG signatures and fixed dimensions; no pixel rerender",
        }
    except Exception as error:
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    gates = _terminal(destination, clock, costs, result, timers)
    require(result["all_passed"] and all(gates.values()), "retained independent localizer failure")
    return result


def _terminal(
    destination: Any,
    clock: dict[str, float],
    costs: dict[str, Any],
    result: dict[str, Any],
    timers: dict[int, Any],
) -> dict[str, bool]:
    """Preserve receipt failures and charge actual self-cost once through final byte enumeration."""
    import resource
    import signal
    import time

    from ecg_experiment.synthetic_audit_io import file_sha256

    def write(path: Any, value: dict[str, Any]) -> None:
        import json

        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    try:
        receipt, digest, retained = "summary.json", None, None
        try:
            write(destination / receipt, result)
            digest = file_sha256(destination / receipt)
            retained = sum(
                path.stat().st_size
                for path in destination.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
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
            write(destination / receipt, result)
            digest = file_sha256(destination / receipt)
        cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gates = {
            "cpu": cpu < 900,
            "wall": wall < 1200,
            "rss": rss < 2 * 1024**3,
            "output": retained is not None and retained + 16384 < 128 * 1024**2,
        }
        verified = bool(costs)
        ledger = {
            "stage": "ludb_localizer_independent_audit_v3",
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
            "measurement_endpoint": "after receipt hash and byte enumeration, before ledger serialization",
            "terminal_bookkeeping_reserve_seconds": 2,
            "terminal_bookkeeping_reserve_bytes": 16384,
            "technical_passed": result["all_passed"] and all(gates.values()),
        }
        for timer in timers:
            signal.setitimer(timer, 0)
        write(destination / "prospective_ledger.json", ledger)
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
    return gates


def verify_chain(
    root: Any,
    study: Any,
    source: Any,
    admission: dict[str, Any],
    config: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """Verify actual main31 and failed32 and return actual32 as the once-only recovery prior.

    Parameters
    ----------
    root, study, source : Path
        Artifact root, immutable main directory and committed source checkout.
    admission, config, recovery : dict
        Original main admission/config and separate prospective recovery declarations.

    Returns
    -------
    dict
        Verified immediate failed32 identity and actual cumulative costs.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    require(
        file_sha256(study / "summary.json") == recovery["main_summary_sha256"]
        and file_sha256(study / "prospective_ledger.json") == recovery["main_ledger_sha256"],
        "unchanged actual main31 identities",
    )
    through_main = main_accounting(root, study, admission, config)
    spec = recovery["failed_audit"]
    directory = root / spec["directory"]
    require(
        file_sha256(directory / "summary.json") == spec["summary_sha256"]
        and file_sha256(directory / "prospective_ledger.json") == spec["ledger_sha256"],
        "immutable actual failed32 identities",
    )
    failure, ledger = read_json(directory / "summary.json"), read_json(directory / "prospective_ledger.json")
    require(
        failure
        == {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": spec["error_type"],
            "error": spec["error"],
        },
        "retained exact timing-order failure scope",
    )
    actual = append_failure(through_main, ledger, spec)
    require(
        ledger["sha256"] == spec["summary_sha256"]
        and ledger["technical_passed"] is False
        and all(ledger["resource_gates"].values()),
        "bound actual failure and resource gates",
    )
    retained = sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "prospective_ledger.json"
    )
    require(retained == ledger["output_bytes_before_terminal_ledger"], "retained failed32 exact bytes")
    snapshots = {
        **admission["source_sha256"],
        recovery["main_admission"]["path"]: recovery["main_admission"]["sha256"],
    }
    for name, digest in snapshots.items():
        require(
            file_sha256(directory / "source_snapshot" / name) == digest,
            "retained actual failed32 source snapshot",
        )
    prior = recovery["prior_ledger"]
    require(
        prior["path"] == spec["directory"] + "/prospective_ledger.json"
        and prior["sha256"] == spec["ledger_sha256"]
        and prior["current_machine_stages"] == 32,
        "frozen actual failed32 immediate prior",
    )
    require(
        prior["current_machine_cumulative_cpu_seconds"] == actual["cpu_through_main"]
        and prior["current_machine_cumulative_gpu_wall_seconds"] == actual["gpu_through_main"],
        "frozen exact once-only actual32 cumulative costs",
    )
    return actual


def append_failure(
    previous_costs: dict[str, Any], ledger: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Reject skipped or duplicated actual failure costs without fabricating a metadata stage.

    Parameters
    ----------
    previous_costs, ledger, spec : dict
        Verified actual31 costs, retained failure ledger and frozen32 binding.

    Returns
    -------
    dict
        Immediate actual32 costs after adding its own measured CPU exactly once.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require

    require(
        ledger["prior_accounting_verified"]
        and ledger["prior_ledger_sha256"] == previous_costs["prior_ledger_sha256"]
        and ledger["current_machine_stages"] == previous_costs["stages_through_main"] + 1 == 32
        and ledger["cpu_seconds"] == spec["cpu_seconds"]
        and ledger["gpu_wall_seconds"] == 0,
        "actual failed32 immediately follows31 with measured CPU",
    )
    require(
        ledger["current_machine_cumulative_cpu_seconds"]
        == previous_costs["cpu_through_main"] + ledger["cpu_seconds"]
        and ledger["current_machine_cumulative_gpu_wall_seconds"] == previous_costs["gpu_through_main"],
        "failed32 CPU counted exactly once and GPU unchanged",
    )
    return {
        "prior_ledger_sha256": spec["ledger_sha256"],
        "stages_through_main": 32,
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def metadata_setup(
    root: Any, source: Any, admission_name: str, recovery_name: str, recovery_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all source, catalog and chain schemas without raw/map/score reconstruction.

    Parameters
    ----------
    root, source : Path
        Artifact checkout and isolated source checkout.
    admission_name, recovery_name, recovery_sha : str
        Immutable original main admission and independently pinned new recovery config identity.

    Returns
    -------
    tuple
        Original main admission and its unchanged scientific config.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    require(file_sha256(source / recovery_name) == recovery_sha, "literal separate recovery config identity")
    recovery = read_json(source / recovery_name)
    require(
        recovery["main_admission"]["path"] == admission_name
        and file_sha256(source / admission_name) == recovery["main_admission"]["sha256"],
        "recovery retains original actual main admission",
    )
    admission, config = previous.metadata_setup(root, source, admission_name)
    required = {RUNNER, CHECKER, TESTS, PROTOCOL}
    require(required.issubset(recovery["source_sha256"]), "complete new recovery source roster")
    for name, digest in recovery["source_sha256"].items():
        require(file_sha256(source / name) == digest, "literal new recovery executable source: " + name)
    require(
        recovery["output_directory"] == "independent_audit_v3"
        and recovery["next_stage"] == 33
        and recovery["method_change"] is False,
        "prospective runtime-only recovery33",
    )
    verify_chain(root, root / config["output_root"], source, admission, config, recovery)
    return admission, config
