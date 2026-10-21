"""Read typed JSON arrays without changing frozen localizer numerical helpers or preserved failures."""

from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_localizer_audit as frozen
from ecg_experiment import ssl_synthetic_ludb_localizer_audit_v3 as previous

CONFIG = frozen.CONFIG
RUNNER = "ecg_experiment/ssl_synthetic_ludb_localizer_audit_v4.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_localizer_v4.py"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-localizer-independent-audit-v4.md"
TESTS = "tests/test_ssl_synthetic_ludb_localizer_audit_v4.py"


def read_json_array(path: Any) -> list[Any]:
    """Read only an admitted JSON array while retaining object-only readers elsewhere.

    Parameters
    ----------
    path : Path
        Local already admitted array-valued metadata path.

    Returns
    -------
    list
        Complete unchanged JSON array; object/scalar containers raise.
    """
    import json

    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require

    value = json.loads(path.read_text())
    require(isinstance(value, list), "audit array metadata must be an array")
    return value


def profile_checks(root: Any, study: Any, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct profile forecasts, all saved old-map replays and exact real first batches.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json

    profile = read_json(study / "profile/summary.json")
    require(
        profile["profile_scores_computed"] == 0 and profile["record_ids"] == config["profile"]["record_ids"],
        "unscored fixed actual-path profile",
    )
    require(
        profile["operations"]
        == {
            "checkpoint_loads": 12,
            "synthetic_forwards": 12,
            "real_batch8_forwards": 12,
            "real_record_reads": 8,
            "profile_figures": 1,
        },
        "complete prospective actual-path operations",
    )
    fixed, repeated, actual = profile["fixed_phase"], profile["repeated_phase"], profile["actual_at_forecast"]
    setup = {name: max(0.0, actual[name] - fixed[name] - repeated[name]) for name in fixed}
    checks = compare_values(setup, profile["setup_outside_phases"], "profile setup arithmetic", 0)
    forecast = {name: setup[name] + 2 * fixed[name] + 40 * repeated[name] + 180 for name in fixed}
    forecast["gpu_lock_wall_seconds"] = 2 * fixed["wall_seconds"] + 40 * repeated["wall_seconds"] + 30
    closures = {
        "synthetic_predecessor_closure.json",
        "timing_predecessor_closure.json",
        "coherence_predecessor_closure.json",
        "admission_snapshot.json",
    }
    before = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and (
            path.relative_to(study).as_posix() in closures
            or path.relative_to(study).parts[0] == "source_snapshot"
            or path.relative_to(study).parts[0] == "profile"
            and path.name != "summary.json"
        )
    )
    forecast["output_bytes"] = (
        before + 460800000 + 38400000 + 51200000 + 72 * 1024**2 + config["profile"]["output_reserve_bytes"]
    )
    checks += compare_values(forecast, profile["forecast"], "complete fixed/repeated forecast", 0)
    limits = config["resources"]
    gates = {
        name: forecast[name] < limits[name]
        for name in ("cpu_seconds", "wall_seconds", "gpu_lock_wall_seconds", "output_bytes")
    }
    gates["cuda_allocated_memory"] = (
        profile["peak_allocated_bytes_at_profile"] < limits["maximum_gpu_allocated_bytes"]
    )
    require(
        gates == profile["forecast_gates"] and all(gates.values()), "actual-path admission gate conjunction"
    )
    require(
        profile["gpu"]["free_bytes_at_admission"] >= limits["minimum_free_gpu_bytes"],
        "actual CUDA free memory admission",
    )
    states = read_json(study / "model_states.json")
    replays = {row["model"]: row for row in profile["synthetic_replays"]}
    inputs = np.load(study / "inputs.npy", mmap_mode="r", allow_pickle=False)
    require(
        np.array_equal(inputs[:8], np.load(study / "profile/inputs.npy", allow_pickle=False)),
        "exact retained input profile/full first batch",
    )
    require(
        read_json_array(study / "input_metadata.json")[:8]
        == read_json_array(study / "profile/input_metadata.json"),
        "exact profile/full structural metadata",
    )
    differences = {}
    for fit in frozen.FITS:
        for arm in frozen.ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            replay = np.load(study / "profile" / (name + "_synthetic_replay.npy"), allow_pickle=False)
            reference = np.load(
                root / frozen.MODELS / f"fit_{fit}" / arm / "fresh_probabilities.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[0]
            require(
                replay.dtype == np.float32 and replay.shape == (4, 12, 5000) and np.isfinite(replay).all(),
                "complete synthetic replay evidence",
            )
            difference = float(np.max(np.abs(replay.astype(np.float64) - reference.astype(np.float64))))
            row = replays[name]
            require(
                difference <= 1e-6
                and row["maximum_absolute_difference"] == difference
                and row["initial_tensor_state_sha256"] == states[name]["initial_tensor_state_sha256"]
                and row["all_passed"]
                and row["atol"] == 1e-6
                and row["rtol"] == 0,
                "prospective old CUDA map replay tolerance and original state",
            )
            require(
                read_json(study / "profile" / (name + "_synthetic_replay.json")) == row,
                "retained per-model replay receipt",
            )
            ppath = study / "profile" / (name + ".npy")
            fullpath = study / "probabilities" / (name + ".npy")
            require(
                file_sha256(ppath) == profile["profile_map_sha256"][name]
                and file_sha256(fullpath) == states[name]["probabilities_sha256"],
                "saved profile/full map hashes",
            )
            require(
                np.array_equal(
                    np.load(ppath, mmap_mode="r", allow_pickle=False),
                    np.load(fullpath, mmap_mode="r", allow_pickle=False)[:8],
                )
                and states[name]["profile_first_batch_exact"]
                and states[name]["full_batches"] == 20
                and states[name]["unchanged"],
                "exact fixed profile versus full first batch",
            )
            differences[name] = difference
    require(
        file_sha256(study / "profile/profile_fixed_maps.png") == profile["figure_sha256"],
        "profile fixed figure identity",
    )
    return {
        "scalar_checks": checks,
        "synthetic_replays": 12,
        "exact_profile_batches": 12,
        "old_map_maximum_absolute_differences": differences,
        "forecast": forecast,
        "profile_retained_bytes_before_summary": before,
    }


def numerical(root: Any, study: Any, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct all native scores and24 independent functional CPU forwards without GPU access.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    import torch

    from ecg_experiment.ssl_synthetic_ludb_audit import waveform
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import checkpoint_digest, require
    from ecg_experiment.synthetic_audit_io import compare_values, read_json

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    identifiers = [identifier for identifier in range(1, 201) if identifier % 5]
    require(read_json_array(study / "record_ids.json") == identifiers, "complete fixed160 record order")
    inputs = np.load(study / "inputs.npy", mmap_mode="r", allow_pickle=False)
    energy = np.load(study / "energy.npy", mmap_mode="r", allow_pickle=False)
    require(inputs.dtype == np.float32 and inputs.shape == (160, 12, 5000), "full float32 observed inputs")
    require(energy.dtype == np.float64 and energy.shape == (160, 8, 5000), "full float64 native energy")
    metadata = read_json_array(study / "input_metadata.json")
    require(len(metadata) == 160, "complete160 input metadata rows")
    scalar_checks = 0
    for position, identifier in enumerate(identifiers):
        guard()
        physical, signal = waveform(root / config["raw_alias"] / str(identifier))
        expected, transformation, expected_energy = frozen.prepare(physical)
        require(np.array_equal(expected, inputs[position]), "exact physical-to-training input")
        require(np.array_equal(expected_energy, energy[position]), "exact standardized native energy")
        scalar_checks += compare_values(
            {"record_id": identifier, "transformation": transformation, "signal_metadata": signal},
            metadata[position],
            "independent nonclinical input metadata",
            0,
        )
    structure = {
        row["record_id"]: row for row in frozen.read_rows(root / frozen.TIMING / "record_rows.jsonl")
    }
    saved = frozen.read_rows(study / "model_rows.jsonl")
    indexed = {(row["record_id"], row["fit_id"], row["arm"]): row for row in saved}
    require(len(saved) == len(indexed) == 1920, "complete1920 model-record rows")
    states = read_json(study / "model_states.json")
    parent = read_json(root / frozen.MODELS / "summary.json")
    reconstructed, replay_rows = [], []
    for fit in frozen.FITS:
        for arm in frozen.ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            probabilities = np.load(
                study / "probabilities" / (name + ".npy"), mmap_mode="r", allow_pickle=False
            )
            require(
                probabilities.shape == (160, 12, 5000) and probabilities.dtype == np.float32,
                "complete full-resolution float32 probability map",
            )
            require(
                np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1,
                "all twelve full output channels admitted without clipping",
            )
            state = torch.load(
                root / frozen.MODELS / f"fit_{fit}" / arm / "final_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            digest, count = checkpoint_digest(state)
            require(
                count == 35660 and digest == parent["final_state_sha256"][str(fit)][arm],
                "actual frozen checkpoint named-tensor identity",
            )
            require(
                states[name]["initial_tensor_state_sha256"] == digest
                and states[name]["final_tensor_state_sha256"] == digest,
                "saved unchanged model tensor state",
            )
            positions = [identifiers.index(identifier) for identifier in (1, 199)]
            replay = frozen.functional_replay(state, inputs[positions])
            difference = float(
                np.max(np.abs(replay.astype(np.float64) - probabilities[positions].astype(np.float64)))
            )
            require(difference <= 1e-5, "fixed independent CPU real replay absolute tolerance")
            replay_rows.append(
                {
                    "model": name,
                    "record_ids": [1, 199],
                    "maximum_absolute_difference": difference,
                    "tensor_state_sha256": digest,
                }
            )
            require(checkpoint_digest(state)[0] == digest, "CPU replay leaves frozen state unchanged")
            for position, identifier in enumerate(identifiers):
                guard()
                row = frozen.record_row(
                    probabilities[position], energy[position], structure[identifier], fit, arm
                )
                scalar_checks += compare_values(
                    row, indexed[identifier, fit, arm], "independent native endpoint"
                )
                reconstructed.append(row)
    statistics, draws = frozen.aggregate(reconstructed, config)
    scalar_checks += compare_values(
        statistics, read_json(study / "summary.json")["statistics"], "independent descriptive reductions"
    )
    require(
        np.array_equal(draws, np.load(study / "bootstrap.npy", allow_pickle=False)),
        "exact all2000 shared record draws across four arms and paired contrast",
    )
    return {
        "scalar_checks": scalar_checks,
        "records": 160,
        "model_record_rows": 1920,
        "native_lead_rows": 15360,
        "bootstrap_draws": 2000,
        "cpu_functional_real_forwards": 24,
        "cpu_replays": replay_rows,
        "statistics": statistics,
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
    destination = study / "independent_audit_v4"
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
        predecessors = previous.predecessor_closures(root, study, guard)
        profile = profile_checks(root, study, config, guard)
        values = numerical(root, study, config, guard)
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
            "stage": "ludb_localizer_independent_audit_v4",
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


def schema_preflight(study: Any) -> dict[str, int]:
    """Exercise remaining metadata containers and NumPy headers without loading numeric payloads.

    Parameters
    ----------
    study : Path
        Completed immutable main output directory.

    Returns
    -------
    dict
        Validated metadata/header counts; no waveform/map/energy payload is scored.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import read_json

    identifiers = [identifier for identifier in range(1, 201) if identifier % 5]
    require(read_json_array(study / "record_ids.json") == identifiers, "actual160 JSON array identifiers")
    full = read_json_array(study / "input_metadata.json")
    profile = read_json_array(study / "profile/input_metadata.json")
    signal_keys = {
        "fs",
        "samples",
        "source_lead_order",
        "canonical_lead_order",
        "source_units",
        "source_adc_gain",
        "source_baseline",
        "canonical_units",
        "canonical_adc_gain",
        "canonical_baseline",
    }
    for metadata, expected_ids in ((full, identifiers), (profile, identifiers[:8])):
        require([row["record_id"] for row in metadata] == expected_ids, "actual input metadata roster")
        for row in metadata:
            require(
                set(row) == {"record_id", "transformation", "signal_metadata"}
                and set(row["transformation"]) == {"median", "q95_minus_q05"}
                and all(len(values) == 8 for values in row["transformation"].values())
                and set(row["signal_metadata"]) == signal_keys,
                "actual metadata matches independent waveform/standardization schemas",
            )
            signal = row["signal_metadata"]
            require(
                signal["fs"] == 500
                and signal["samples"] == 5000
                and all(len(signal[name]) == 12 for name in signal_keys - {"fs", "samples"}),
                "complete twelve-lead structural metadata",
            )
    require(full[:8] == profile, "actual metadata profile/full equality")
    states = read_json(study / "model_states.json")
    names = [f"fit_{fit}_{arm}" for fit in frozen.FITS for arm in frozen.ARMS]
    require(set(states) == set(names), "actual twelve-model state roster")
    fields = {
        "initial_tensor_state_sha256",
        "final_tensor_state_sha256",
        "unchanged",
        "full_batches",
        "profile_first_batch_exact",
        "probabilities_sha256",
    }
    require(all(set(row) == fields for row in states.values()), "actual model-state fields")
    profile_summary = read_json(study / "profile/summary.json")
    require(
        set(profile_summary["profile_map_sha256"]) == set(names)
        and {row["model"] for row in profile_summary["synthetic_replays"]} == set(names),
        "actual complete profile roster",
    )
    for name in (
        "synthetic_predecessor_closure.json",
        "timing_predecessor_closure.json",
        "coherence_predecessor_closure.json",
        "summary.json",
        "prospective_ledger.json",
    ):
        read_json(study / name)
    headers = {
        "inputs.npy": ((160, 12, 5000), np.float32),
        "energy.npy": ((160, 8, 5000), np.float64),
        "bootstrap.npy": ((2000, 5), np.float64),
        "profile/inputs.npy": ((8, 12, 5000), np.float32),
    }
    for name in names:
        headers.update(
            {
                f"probabilities/{name}.npy": ((160, 12, 5000), np.float32),
                f"profile/{name}.npy": ((8, 12, 5000), np.float32),
                f"profile/{name}_synthetic_replay.npy": ((4, 12, 5000), np.float32),
            }
        )
        read_json(study / "profile" / (name + "_synthetic_replay.json"))
    for name, (shape, dtype) in headers.items():
        values = np.load(study / name, mmap_mode="r", allow_pickle=False)
        require(values.shape == shape and values.dtype == dtype, "actual NumPy header shape/dtype: " + name)
        del values
    return {
        "array_valued_json_containers": 3,
        "input_metadata_rows": 168,
        "numpy_headers": len(headers),
        "waveform_payload_reads": 0,
        "new_score_reconstructions": 0,
    }


def append_failure(
    previous_costs: dict[str, Any], ledger: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Verify the actual second failure is counted once after actual32.

    Parameters
    ----------
    previous_costs, ledger, spec : dict
        Verified actual32 costs, actual33 ledger and its frozen binding.

    Returns
    -------
    dict
        Actual33 immediate identity and cumulative costs.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require

    require(
        ledger["prior_accounting_verified"]
        and ledger["prior_ledger_sha256"] == previous_costs["prior_ledger_sha256"]
        and ledger["current_machine_stages"] == previous_costs["stages_through_main"] + 1 == 33
        and ledger["cpu_seconds"] == spec["cpu_seconds"]
        and ledger["gpu_wall_seconds"] == 0,
        "actual33 immediately follows32 with its measured CPU",
    )
    require(
        ledger["current_machine_cumulative_cpu_seconds"]
        == previous_costs["cpu_through_main"] + ledger["cpu_seconds"]
        and ledger["current_machine_cumulative_gpu_wall_seconds"] == previous_costs["gpu_through_main"],
        "failed33 CPU counted exactly once and GPU unchanged",
    )
    return {
        "prior_ledger_sha256": spec["ledger_sha256"],
        "stages_through_main": 33,
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def verify_chain(
    root: Any,
    study: Any,
    source: Any,
    admission: dict[str, Any],
    config: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """Verify immutable31→32→33 source/receipt/byte/cost lineage without numerical payload reads.

    Parameters
    ----------
    root, study, source : Path
        Local artifacts, immutable main directory and isolated source root.
    admission, config, recovery : dict
        Original main admission/config and separate v4 recovery declarations.

    Returns
    -------
    dict
        Genuine failed33 immediate prior, preserving both failures once.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    prior_spec = recovery["previous_recovery"]
    require(
        file_sha256(source / prior_spec["path"]) == prior_spec["sha256"], "immutable previous recovery config"
    )
    prior_recovery = read_json(source / prior_spec["path"])
    through32 = previous.verify_chain(root, study, source, admission, config, prior_recovery)
    spec = recovery["failed_audit"]
    directory = root / spec["directory"]
    require(
        file_sha256(directory / "summary.json") == spec["summary_sha256"]
        and file_sha256(directory / "prospective_ledger.json") == spec["ledger_sha256"],
        "immutable actual failed33 identities",
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
        "exact actual JSON-container failure",
    )
    actual = append_failure(through32, ledger, spec)
    require(
        ledger["sha256"] == spec["summary_sha256"]
        and ledger["technical_passed"] is False
        and all(ledger["resource_gates"].values()),
        "actual failed33 bound receipt/resource gates",
    )
    size = sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "prospective_ledger.json"
    )
    require(size == ledger["output_bytes_before_terminal_ledger"], "actual failed33 retained bytes")
    snapshots = {
        **admission["source_sha256"],
        **prior_recovery["source_sha256"],
        recovery["main_admission"]["path"]: recovery["main_admission"]["sha256"],
        prior_spec["path"]: prior_spec["sha256"],
    }
    for name, digest in snapshots.items():
        require(
            file_sha256(directory / "source_snapshot" / name) == digest, "actual failed33 executable snapshot"
        )
    prior = recovery["prior_ledger"]
    require(
        prior["path"] == spec["directory"] + "/prospective_ledger.json"
        and prior["sha256"] == spec["ledger_sha256"]
        and prior["current_machine_stages"] == 33
        and prior["current_machine_cumulative_cpu_seconds"] == actual["cpu_through_main"]
        and prior["current_machine_cumulative_gpu_wall_seconds"] == actual["gpu_through_main"],
        "frozen exact actual33 immediate prior",
    )
    return actual


def metadata_setup(
    root: Any, source: Any, admission_name: str, recovery_name: str, recovery_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise every remaining metadata schema before one measured payload reconstruction.

    Parameters
    ----------
    root, source : Path
        Artifact and source checkout roots.
    admission_name, recovery_name, recovery_sha : str
        Actual original main admission and separate literal recovery identity.

    Returns
    -------
    tuple
        Original main declarations, never a later source overlay disguised as the main admission.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    require(file_sha256(source / recovery_name) == recovery_sha, "literal JSON-container recovery identity")
    recovery = read_json(source / recovery_name)
    require(
        recovery["main_admission"]["path"] == admission_name
        and file_sha256(source / admission_name) == recovery["main_admission"]["sha256"],
        "unchanged actual main admission",
    )
    prior = recovery["previous_recovery"]
    admission, config = previous.metadata_setup(root, source, admission_name, prior["path"], prior["sha256"])
    require(
        {RUNNER, CHECKER, TESTS, PROTOCOL}.issubset(recovery["source_sha256"]), "complete v4 source overlay"
    )
    for name, digest in recovery["source_sha256"].items():
        require(file_sha256(source / name) == digest, "literal v4 executable/protocol identity")
    require(
        recovery["next_stage"] == 34
        and recovery["output_directory"] == "independent_audit_v4"
        and recovery["method_change"] is False,
        "prospective runtime-only recovery34",
    )
    study = root / config["output_root"]
    verify_chain(root, study, source, admission, config, recovery)
    schema_preflight(study)
    return admission, config
