"""Pre-execution terminal-accounting successor; frozen localizer audit mathematics are unchanged."""

from typing import Any

from ecg_experiment import ssl_synthetic_ludb_localizer_audit as frozen

CONFIG = frozen.CONFIG
RUNNER = "ecg_experiment/ssl_synthetic_ludb_localizer_audit_v2.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_localizer_v2.py"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-localizer-independent-audit-v2.md"
TESTS = "tests/test_ssl_synthetic_ludb_localizer_audit_v2.py"


def metadata_setup(root: Any, source: Any, admission_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify both immutable v1 and runtime-only v2 source declarations without numerical reads.

    Parameters
    ----------
    root : Path
        Local artifact checkout.
    source : Path
        Isolated admitted source checkout.
    admission_name : str
        Committed repository-relative admission path.

    Returns
    -------
    tuple
        Verified prospective admission and unchanged scientific configuration.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256

    admission, config = frozen.metadata_setup(root, source, admission_name)
    for name in (RUNNER, CHECKER, PROTOCOL, TESTS):
        require(
            name in admission["source_sha256"]
            and file_sha256(source / name) == admission["source_sha256"][name],
            "new runtime-only independent audit successor identity",
        )
    return admission, config


def run(
    root: Any,
    source: Any,
    admission_name: str,
    main_sha: str,
    entry_sha: str,
    runner_sha: str,
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
    destination = study / "independent_audit_v2"
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
        admission, config = metadata_setup(root, source, admission_name)
        require(file_sha256(study / "summary.json") == main_sha, "literal actual main31 summary")
        costs = frozen.main_accounting(root, study, admission, config)
        own_sources = {**admission["source_sha256"], admission_name: file_sha256(source / admission_name)}
        for name in own_sources:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        identity_checks = frozen.input_identities(root, admission, guard)
        identity_checks += frozen.output_identities(study, source, admission, admission_name, guard)
        predecessors = frozen.predecessor_closures(root, study, guard)
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
            "stage": "ludb_localizer_independent_audit_v2",
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
