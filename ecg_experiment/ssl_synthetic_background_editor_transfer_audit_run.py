"""Measured independent audit of saved background/editor transfer arrays and actual accounting."""

import json
import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_background_editor_transfer_audit as core
from ecg_experiment.ssl_synthetic_ludb_coherence_audit import standardize
from ecg_experiment.ssl_synthetic_ludb_localizer_audit import aggregate as localizer_aggregate
from ecg_experiment.ssl_synthetic_ludb_localizer_audit import functional_replay
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import checkpoint_digest, require
from ecg_experiment.ssl_synthetic_suppression_saved_audit import suppression_summary
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_background_editor_transfer_v1.json"
RUNNER = "ecg_experiment/ssl_synthetic_background_editor_transfer_audit_run.py"
CHECKER = "scripts/validation/ssl_synthetic_background_editor_transfer_audit.py"
PROTOCOL = "docs/ssl-synthetic-localization-background-editor-transfer-independent-audit-v1.md"
TESTS = "tests/test_ssl_synthetic_background_editor_transfer_audit_runtime.py"
STUDY = "outputs/ssl_synthetic_localization/background_editor_transfer_v1"
LOCALIZER = "outputs/ssl_synthetic_localization/ludb_localizer_v1"
PARENT = "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"
PRIOR34_SHA = "ccac6c1b21691705592cd6425107a2647aec7fcbcd7e082a4fac692eabaa864b"
PRIOR34_RECEIPT = "e229edb9f0b410531e91ea2ad9170d38b10bde40f32f5bd3eb542edb53a5f942"
NATIVE = (0, 1, 6, 7, 8, 9, 10, 11)


AUDIT_CORE = "ecg_experiment/ssl_synthetic_background_editor_transfer_audit.py"
CORE_TESTS = "tests/test_ssl_synthetic_background_editor_transfer_audit.py"
SHARED_SOURCES = {
    "ecg_experiment/intervals.py",
    "ecg_experiment/ssl_synthetic_quality_saved_audit.py",
    "ecg_experiment/ssl_synthetic_randomized_saved_audit.py",
    "ecg_experiment/ssl_synthetic_ludb_coherence_audit.py",
    "ecg_experiment/ssl_synthetic_ludb_localizer_audit.py",
    "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
    "ecg_experiment/ssl_synthetic_suppression_saved_audit.py",
    "ecg_experiment/synthetic_audit_io.py",
}


def source_union(main: dict[str, Any], audit: dict[str, Any]) -> dict[str, str]:
    """Require explicit independent math bindings and prohibit overriding any frozen main source.

    Parameters
    ----------
    main, audit : dict
        Original main and separate audit identity declarations.

    Returns
    -------
    dict
        Complete consistent mandatory independent/core/shared source bindings.
    """
    first, second = main["source_sha256"], audit["source_sha256"]
    require(
        {RUNNER, CHECKER, PROTOCOL, TESTS, AUDIT_CORE, CORE_TESTS}.issubset(second),
        "complete explicit independent core, runtime, protocol and test source roster",
    )
    require(
        all(first[name] == digest for name, digest in second.items() if name in first),
        "audit admission cannot override any immutable main source",
    )
    combined = {**first, **second}
    require(SHARED_SOURCES.issubset(combined), "all imported shared numerical helpers are source-bound")
    return combined


def json_array(path: Path) -> list[Any]:
    """Read only a JSON array; object metadata remains subject to its separate strict reader.

    Parameters
    ----------
    path : Path
        Local declared file path.

    Returns
    -------
    list[Any]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    value = json.loads(path.read_text())
    require(isinstance(value, list), "declared metadata must be an array")
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    """Read complete local JSONL objects without interpreting clinical or waveform content.

    Parameters
    ----------
    path : Path
        Local declared file path.

    Returns
    -------
    list[dict[str, Any]]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    values = [json.loads(line) for line in path.read_text().splitlines()]
    require(all(isinstance(value, dict) for value in values), "every JSONL record must be an object")
    return values


def array(path: Path, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """Validate a standard read-only NumPy header without reading its scientific payload.

    Parameters
    ----------
    path : Path
        Local declared file path.
    shape : tuple[int, ...]
        Exact declared complete array geometry.
    dtype : Any
        Exact float32, float64, integer or Boolean representation.

    Returns
    -------
    np.ndarray
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    require(values.shape == shape and values.dtype == dtype, "declared NumPy shape/dtype: " + path.name)
    return values


def prior34(root: Path) -> dict[str, Any]:
    """Verify the actual successful immediate predecessor, without reusing stale catalog wrappers.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    directory = root / LOCALIZER / "independent_audit_v4"
    require(
        file_sha256(directory / "prospective_ledger.json") == PRIOR34_SHA
        and file_sha256(directory / "summary.json") == PRIOR34_RECEIPT,
        "actual34 immutable identities",
    )
    ledger, summary = read_json(directory / "prospective_ledger.json"), read_json(directory / "summary.json")
    require(
        summary["all_passed"]
        and summary["mismatch_count"] == 0
        and ledger["technical_passed"]
        and ledger["prior_accounting_verified"]
        and all(ledger["resource_gates"].values())
        and ledger["sha256"] == PRIOR34_RECEIPT
        and ledger["current_machine_stages"] == 34
        and ledger["current_machine_cumulative_cpu_seconds"] == 759.933143906
        and ledger["current_machine_cumulative_gpu_wall_seconds"] == 473.0495132012293,
        "actual34 completed verification and exact latest costs",
    )
    return ledger


def parent_schema(root: Path, config: dict[str, Any]) -> None:
    """Exercise actual parent metadata containers, train membership and all array header contracts.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    config : dict[str, Any]
        Frozen prospective scientific and resource configuration.

    Returns
    -------
    None
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    identifiers = json_array(root / LOCALIZER / "record_ids.json")
    require(identifiers == [n for n in range(1, 201) if n % 5], "original160 real ID order")
    metadata = json_array(root / LOCALIZER / "input_metadata.json")
    require(
        len(metadata) == 160 and [r["record_id"] for r in metadata] == identifiers,
        "actual real metadata array schema",
    )
    array(root / LOCALIZER / "inputs.npy", (160, 12, 5000), np.float32)
    array(root / LOCALIZER / "bootstrap.npy", (2000, 5), np.float64)
    for filename, shape, dtype in (
        ("fresh_raw.npy", (64, 8, 5000), np.float32),
        ("fresh_observed.npy", (64, 4, 12, 5000), np.float32),
        ("fresh_support.npy", (64, 4, 12, 5000), bool),
        ("fresh_strict_support.npy", (64, 2, 12, 5000), bool),
    ):
        array(root / PARENT / filename, shape, dtype)
    cases = rows(root / PARENT / "fresh_cases.jsonl")
    require(
        [r["index"] for r in cases] == list(range(2000, 2064))
        and [r["seed"] for r in cases] == list(range(2026105000, 2026105064)),
        "actual independent-seed bank",
    )
    parent_catalog = read_json(root / PARENT / "summary.json")["output_sha256"]
    for fit in core.FITS:
        schedule_name = f"fit_{fit}/training_schedule.npy"
        require(
            file_sha256(root / PARENT / schedule_name) == parent_catalog[schedule_name],
            "training membership bound through original immutable parent catalog",
        )
        schedule = array(root / PARENT / f"fit_{fit}/training_schedule.npy", (400, 8), np.int64)
        require(
            schedule.min() >= 0 and schedule.max() <= 319, "localizer TRAIN80 membership excludes fresh64"
        )
        for arm in core.ARMS:
            read_json(root / PARENT / f"fit_{fit}/{arm}/fresh_metrics.json")
    require(
        config["sources"] == list(core.SOURCES)
        and config["variants"] == ["sham", "nuisance", *core.POSITIVES]
        and config["donor_mask_offset"] == 1
        and config["backgrounds_per_source"] == 64
        and config["bootstrap"]
        == {
            "draws": 2000,
            "seed": 2026106601,
            "unit": "shared_geometry_slot",
            "method": "linear",
            "fit_resampling": False,
        },
        "unchanged prospective geometry",
    )


def metadata_setup(
    root: Path, source: Path, main_admission_name: str, audit_admission_name: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify actual sources/parents before main execution, or add separate audit-source admission later.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    source : Path
        Isolated source checkout containing immutable committed code/configs.
    main_admission_name : str
        Repository-relative original main identity admission.
    audit_admission_name : str | None
        Separate prospective audit admission, never substituted for the original main roster.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    config, main = read_json(source / CONFIG), read_json(source / main_admission_name)
    require(
        main["source_sha256"][CONFIG] == main["frozen_config_sha256"]
        and main["prior_ledger"]["path"] == config["prior_ledger"]
        and main["prior_ledger"]["sha256"] == PRIOR34_SHA,
        "original prospective main admission",
    )
    for name, digest in main["source_sha256"].items():
        require(file_sha256(source / name) == digest, "complete main scientific source identity: " + name)
    previous = prior34(root)
    for key in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        require(main["prior_ledger"][key] == previous[key], "actual34 immediate prior fields")
    for parent in (PARENT, LOCALIZER):
        require(
            file_sha256(root / parent / "summary.json") == main["input_sha256"][parent + "/summary.json"],
            "immutable original parent summary identity",
        )
        receipt = read_json(root / parent / "summary.json")
        for name, digest in main["input_sha256"].items():
            prefix = parent + "/"
            if name.startswith(prefix):
                relative = name[len(prefix) :]
                if relative not in {"summary.json", "prospective_ledger.json"} and not relative.startswith(
                    "independent_audit"
                ):
                    require(
                        receipt["output_sha256"].get(relative) == digest,
                        "original parent output catalog anchor",
                    )
    parent_schema(root, config)
    if audit_admission_name is not None:
        audit = read_json(source / audit_admission_name)
        require(
            audit["main_admission"]
            == {"path": main_admission_name, "sha256": file_sha256(source / main_admission_name)},
            "separate audit admission preserves actual original main roster",
        )
        source_union(main, audit)
        for name, digest in audit["source_sha256"].items():
            require(file_sha256(source / name) == digest, "complete separate audit source identity: " + name)
        main_schema(root / config["output_root"], config)
    return main, config


def main_schema(study: Path, config: dict[str, Any]) -> None:
    """Exercise every actual remaining metadata container and saved NumPy header before scoring.

    Parameters
    ----------
    study : Path
        Completed exclusive main output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource configuration.

    Returns
    -------
    None
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    identifiers = read_json(study / "record_ids.json")
    require(
        identifiers
        == {"real": [n for n in range(1, 201) if n % 5][:64], "generated": list(range(2000, 2064))},
        "both fixed source rosters",
    )
    geometry = rows(study / "geometry.jsonl")
    require(
        [(r["source"], r["slot"]) for r in geometry] == [(s, n) for s in core.SOURCES for n in range(64)],
        "complete128 geometry metadata rows",
    )
    array(study / "inputs.npy", (2, 64, 6, 12, 5000), np.float32)
    for name in ("support.npy", "strict_support.npy"):
        array(study / name, (2, 64, 4, 12, 5000), bool)
    main, states = read_json(study / "summary.json"), read_json(study / "model_states.json")
    require(set(states) == {"profile", "full", "all_unchanged"}, "complete state evidence object")
    names = {f"fit_{f}_{a}" for f in core.FITS for a in core.ARMS}
    require(set(states["profile"]) == set(states["full"]) == names, "complete12-model state roster")
    require(len(rows(study / "record_rows.jsonl")) == 1536, "complete1536 model-background rows")
    require(len(rows(study / "profile/record_rows.jsonl")) == 192, "complete192 profile rows")
    array(study / "bootstrap.npy", (2000, len(main["statistics"]["endpoint_order"])), np.float64)
    for name in names:
        array(study / "probabilities" / (name + ".npy"), (2, 64, 6, 12, 5000), np.float32)
        array(study / "profile/probabilities" / (name + ".npy"), (2, 8, 6, 12, 5000), np.float32)
        array(study / "profile" / (name + "_synthetic_replay.npy"), (4, 12, 5000), np.float32)
        read_json(study / "profile" / (name + "_synthetic_replay.json"))
    for name in (
        "profile/summary.json",
        "profile/gpu_phase.json",
        "full_gpu_phase.json",
        "localizer_predecessor_closure.json",
        "prospective_ledger.json",
    ):
        read_json(study / name)
    require(
        config["independent_audit"]["cpu_replay_variants"] == [2, 5]
        and config["independent_audit"]["cpu_replay_slots"] == [0, 63]
        and config["independent_audit"]["cpu_replay_sources"] == list(core.SOURCES),
        "frozen96 replay roster",
    )


def identities(
    root: Path,
    source: Path,
    study: Path,
    main: dict[str, Any],
    admission: dict[str, Any],
    admission_name: str,
    guard: Any,
) -> int:
    """Verify all declared saved input/output/source bytes without changing their immutable histories.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    source : Path
        Isolated source checkout containing immutable committed code/configs.
    study : Path
        Completed exclusive main output directory.
    main : dict[str, Any]
        Completed main summary object.
    admission : dict[str, Any]
        Original main source/input/ledger identity declarations.
    admission_name : str
        Repository-relative original main admission path.
    guard : Any
        Callable enforcing live measured CPU/wall/RSS bounds.

    Returns
    -------
    int
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    expected_sources = {**admission["source_sha256"], admission_name: file_sha256(source / admission_name)}
    require(main["source_sha256"] == expected_sources, "actual main used its original source admission")
    for name, digest in admission["input_sha256"].items():
        guard()
        require(file_sha256(relative_file(root, name), guard) == digest, "unchanged saved input: " + name)
    for name, digest in admission["source_sha256"].items():
        guard()
        require(
            file_sha256(source / name) == digest and file_sha256(study / "source_snapshot" / name) == digest,
            "unchanged executed source snapshots",
        )
    require(
        file_sha256(study / "admission_snapshot.json") == expected_sources[admission_name],
        "actual main admission snapshot",
    )
    for name, digest in main["output_sha256"].items():
        guard()
        require(file_sha256(study / name, guard) == digest, "unchanged complete main output catalog: " + name)
    return len(admission["input_sha256"]) + len(expected_sources) + len(main["output_sha256"])


def accounting(root: Path, study: Path, main: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    """Verify actual GPU-main costs and all retained bytes, excluding this audit's own subtree.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    study : Path
        Completed exclusive main output directory.
    main : dict[str, Any]
        Completed main summary object.
    admission : dict[str, Any]
        Original main source/input/ledger identity declarations.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    previous = prior34(root)
    path = study / "prospective_ledger.json"
    ledger = read_json(path)
    require(
        ledger["receipt_sha256"] == file_sha256(study / "summary.json")
        and ledger["prior_ledger_sha256"] == admission["prior_ledger"]["sha256"] == PRIOR34_SHA
        and ledger["current_machine_stages"] == 35
        and ledger["prior_accounting_verified"],
        "actual35 immediate receipt/ledger",
    )
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + suffix
        require(
            ledger[suffix] >= 0 and ledger[key] == previous[key] + ledger[suffix],
            "exact once-only actual main costs",
        )
    require(
        main["status"] == ledger["status"] == "PASS"
        and ledger["technical_passed"]
        and main["technical_passed_before_independent_audit"]
        and all(ledger["resource_gates"].values()),
        "actual completed main technical/resource conjunction",
    )
    require(
        ledger["cpu_seconds"] < 1800
        and ledger["wall_seconds"] < 2400
        and ledger["gpu_wall_seconds"] < 300
        and ledger["peak_rss_bytes"] < 2 * 1024**3,
        "independently recomputed main CPU/wall/GPU/RSS hard ceilings",
    )
    require(
        ledger["output_bytes_before_terminal_ledger"] + 16384 < 3 * 1024**3,
        "independently recomputed main storage ceiling",
    )
    paths = {
        p.relative_to(study).as_posix(): p
        for p in study.rglob("*")
        if p.is_file() and not p.is_symlink() and "independent_audit_v1" not in p.relative_to(study).parts
    }
    require(
        set(paths) == set(main["output_sha256"]) | {"summary.json", "prospective_ledger.json"},
        "complete main file membership",
    )
    require(
        sum(p.stat().st_size for name, p in paths.items() if name != "prospective_ledger.json")
        == ledger["output_bytes_before_terminal_ledger"],
        "exact main retained bytes before its ledger",
    )
    return {
        "prior_ledger_sha256": file_sha256(path),
        "stages_through_main": 35,
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def predecessors(root: Path, source: Path, study: Path) -> int:
    """Independently reproduce the old synthetic aggregate and exact saved localizer statistics/draws.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    source : Path
        Isolated source checkout containing immutable committed code/configs.
    study : Path
        Completed exclusive main output directory.

    Returns
    -------
    int
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    fits = [
        {
            "fit_id": fit,
            **{arm: read_json(root / PARENT / f"fit_{fit}/{arm}/fresh_metrics.json") for arm in core.ARMS},
        }
        for fit in core.FITS
    ]
    summary = suppression_summary(fits)
    checks = compare_values(
        summary, read_json(root / PARENT / "summary.json")["aggregate"], "old full synthetic aggregate"
    )
    checks += compare_values(
        summary,
        read_json(study / "synthetic_predecessor_closure.json")["aggregate"],
        "saved synthetic closure",
    )
    require(
        summary["scientific_passed"] is False and len(summary["scientific_gates"]) == 15,
        "original15-gate FAIL preserved",
    )
    config = read_json(source / "configs/ssl_synthetic_ludb_localizer_v1.json")
    statistics, draws = localizer_aggregate(rows(root / LOCALIZER / "model_rows.jsonl"), config)
    checks += compare_values(
        statistics, read_json(root / LOCALIZER / "summary.json")["statistics"], "old localizer aggregate", 0
    )
    checks += compare_values(
        statistics,
        read_json(study / "localizer_predecessor_closure.json")["statistics"],
        "saved real diagnostic closure",
        0,
    )
    require(
        np.array_equal(draws, np.load(root / LOCALIZER / "bootstrap.npy", allow_pickle=False)),
        "all2000 old localizer draws",
    )
    return checks


def reconstruct_inputs(root: Path, study: Path, guard: Any) -> int:
    """Rebuild every input, support cell and geometry declaration from the unchanged admitted parents.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    study : Path
        Completed exclusive main output directory.
    guard : Any
        Callable enforcing live measured CPU/wall/RSS bounds.

    Returns
    -------
    int
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    inputs = array(study / "inputs.npy", (2, 64, 6, 12, 5000), np.float32)
    strict = array(study / "strict_support.npy", (2, 64, 4, 12, 5000), bool)
    support = array(study / "support.npy", (2, 64, 4, 12, 5000), bool)
    raw = array(root / PARENT / "fresh_raw.npy", (64, 8, 5000), np.float32)
    real = array(root / LOCALIZER / "inputs.npy", (160, 12, 5000), np.float32)
    old = array(root / PARENT / "fresh_observed.npy", (64, 4, 12, 5000), np.float32)
    oldstrict = array(root / PARENT / "fresh_strict_support.npy", (64, 2, 12, 5000), bool)
    oldsupport = array(root / PARENT / "fresh_support.npy", (64, 4, 12, 5000), bool)
    geometry = rows(study / "geometry.jsonl")
    real_ids = read_json(study / "record_ids.json")["real"]
    checks = 0
    for sourcepos, source in enumerate(core.SOURCES):
        for slot in range(64):
            guard()
            native, transformation = (
                standardize(raw[slot])
                if source == "generated"
                else (np.array(real[slot, list(NATIVE)], dtype=np.float32), None)
            )
            observed, exact, target, declaration = core.make_case(native, 2000 + slot)
            require(
                np.array_equal(observed, inputs[sourcepos, slot])
                and np.array_equal(exact, strict[sourcepos, slot])
                and np.array_equal(target, support[sourcepos, slot]),
                "every final input and mask cell",
            )
            if source == "generated":
                require(
                    np.array_equal(observed[:4], old[slot])
                    and np.array_equal(exact[:2], oldstrict[slot])
                    and np.array_equal(target[:2], oldsupport[slot, 2:]),
                    "original generated4 variants/targets exactly retained",
                )
            else:
                require(
                    np.array_equal(observed[0], real[slot]), "real input unchanged and standardized only once"
                )
            expected = {
                **declaration,
                "source": source,
                "slot": slot,
                "record_id": real_ids[slot] if source == "real" else None,
                "standardization": transformation,
                "canonical_sham_exact": True,
                "original_generated_four_exact": source == "generated",
            }
            checks += compare_values(
                expected, geometry[sourcepos * 64 + slot], "independent geometry and support declarations"
            )
    return checks


def aggregate_rows(values: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Add independent per-fit/mode reporting and canonical final bootstrap column placement.

    Parameters
    ----------
    values : list[dict[str, Any]]
        Complete independently reconstructed model/source/background rows.
    config : dict[str, Any]
        Frozen prospective scientific and resource configuration.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    summary, original = core.summarize(values, config)
    oldnames = summary["endpoint_order"]
    controls = [name for name in oldnames if name.split("__")[2] in {"sham", "nuisance"}]
    names = [name for name in oldnames if name not in controls] + controls
    draws = original[:, [oldnames.index(name) for name in names]]
    summary["endpoint_order"] = names
    indexed = {(r["source"], r["slot"], r["fit_id"], r["arm"]): r for r in values}
    points = []
    for source in core.SOURCES:
        for arm in core.ARMS:
            for fit in core.FITS:
                for variant in core.POSITIVES:
                    records = [indexed[source, slot, fit, arm]["positives"][variant] for slot in range(64)]
                    points.append(
                        {
                            "source": source,
                            "arm": arm,
                            "fit_id": fit,
                            "variant": variant,
                            "points": {
                                name: float(np.mean([r[name] for r in records])) for name in records[0]
                            },
                        }
                    )
    summary["per_fit_mode"] = points
    return summary, draws


def numerical(root: Path, study: Path, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct every saved full-map endpoint and exactly96 fixed CPU model-record forwards.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    study : Path
        Completed exclusive main output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource configuration.
    guard : Any
        Callable enforcing live measured CPU/wall/RSS bounds.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs = array(study / "inputs.npy", (2, 64, 6, 12, 5000), np.float32)
    support = array(study / "support.npy", (2, 64, 4, 12, 5000), bool)
    saved = rows(study / "record_rows.jsonl")
    indexed = {(r["source"], r["slot"], r["fit_id"], r["arm"]): r for r in saved}
    require(len(saved) == len(indexed) == 1536, "complete nonduplicated endpoint table")
    profile_rows = rows(study / "profile/record_rows.jsonl")
    profile_index = {(r["source"], r["slot"], r["fit_id"], r["arm"]): r for r in profile_rows}
    require(len(profile_rows) == len(profile_index) == 192, "complete nonduplicated profile endpoint table")
    parent = read_json(root / PARENT / "summary.json")
    states = read_json(study / "model_states.json")
    require(
        states["all_unchanged"] and states["profile"] == states["full"], "unchanged whole-run named states"
    )
    replay_cases = [
        (sourcepos, slot, variant) for sourcepos in range(2) for slot in (0, 63) for variant in (2, 5)
    ]
    replay_input = np.stack([inputs[sourcepos, slot, variant] for sourcepos, slot, variant in replay_cases])
    reconstructed, replays = [], []
    checks = 0
    for fit in core.FITS:
        for arm in core.ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            maps = array(study / "probabilities" / (name + ".npy"), (2, 64, 6, 12, 5000), np.float32)
            first = array(study / "profile/probabilities" / (name + ".npy"), (2, 8, 6, 12, 5000), np.float32)
            require(np.array_equal(first, maps[:, :8]), "exact entire first8-slots profile/full maps")
            del first
            state = torch.load(
                root / PARENT / f"fit_{fit}/{arm}/final_state.pt", map_location="cpu", weights_only=True
            )
            digest, count = checkpoint_digest(state)
            require(
                count == 35660
                and digest
                == parent["final_state_sha256"][str(fit)][arm]
                == states["profile"][name]
                == states["full"][name],
                "original named checkpoint identity",
            )
            cpu = functional_replay(state, replay_input)
            for position, (sourcepos, slot, variant) in enumerate(replay_cases):
                difference = float(
                    np.max(
                        np.abs(
                            cpu[position].astype(np.float64)
                            - maps[sourcepos, slot, variant].astype(np.float64)
                        )
                    )
                )
                require(
                    difference <= config["independent_audit"]["cpu_replay_atol"],
                    "fixed CPU/CUDA replay tolerance",
                )
                replays.append(
                    {
                        "model": name,
                        "source": core.SOURCES[sourcepos],
                        "slot": slot,
                        "variant": variant,
                        "maximum_absolute_difference": difference,
                        "tensor_state_sha256": digest,
                    }
                )
            require(checkpoint_digest(state)[0] == digest, "CPU functional replay retains exact model state")
            del state, cpu
            for sourcepos, source in enumerate(core.SOURCES):
                for slot in range(64):
                    guard()
                    value = core.score_record(
                        maps[sourcepos, slot],
                        inputs[sourcepos, slot],
                        support[sourcepos, slot],
                        source,
                        slot,
                        fit,
                        arm,
                        support[sourcepos, (slot + 1) % 64],
                    )
                    checks += compare_values(
                        value, indexed[source, slot, fit, arm], "all independent full-grid endpoints"
                    )
                    if slot < 8:
                        checks += compare_values(
                            value, profile_index[source, slot, fit, arm], "exact profile score path"
                        )
                    reconstructed.append(value)
            del maps
    summary, draws = aggregate_rows(reconstructed, config)
    checks += compare_values(
        summary, read_json(study / "summary.json")["statistics"], "all independent summaries and gates"
    )
    actual = array(study / "bootstrap.npy", draws.shape, np.float64)
    require(
        np.isfinite(actual).all() and np.allclose(draws, actual, atol=1e-12, rtol=0),
        "all ordered2000 bootstrap draws",
    )
    checks += draws.size
    return {
        "scalar_checks": checks,
        "model_background_rows": len(reconstructed),
        "positive_rows": len(reconstructed) * 4,
        "profile_rows": len(profile_rows),
        "bootstrap_draws": len(draws),
        "bootstrap_columns": draws.shape[1],
        "bootstrap_maximum_absolute_difference": float(np.abs(draws - actual).max()),
        "cpu_model_record_forwards": len(replays),
        "cpu_replays": replays,
        "statistics": summary,
    }


def profile_checks(
    root: Path, study: Path, config: dict[str, Any], main_ledger: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct complete profile forecasts, immutable parent replays and additive two-lock evidence.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    study : Path
        Completed exclusive main output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource configuration.
    main_ledger : dict[str, Any]
        Actual terminal main accounting and resource evidence.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    profile = read_json(study / "profile/summary.json")
    first, full = read_json(study / "profile/gpu_phase.json"), read_json(study / "full_gpu_phase.json")
    actual, infer, scoring = (
        profile["actual_at_forecast"],
        profile["recurring_inference"],
        profile["recurring_scoring"],
    )
    require(infer == first["recurring_inference"], "same recurring CUDA profile timings")
    factor = config["profile"]["factor"] * config["profile"]["recurrence_multiplier"]
    forecast = {
        key: actual[key] + factor * (infer[key] + scoring[key]) + config["profile"][reserve]
        for key, reserve in (("cpu_seconds", "cpu_reserve_seconds"), ("wall_seconds", "wall_reserve_seconds"))
    }
    forecast["gpu_lock_seconds"] = (
        first["gpu_lock_seconds"] + factor * infer["wall_seconds"] + config["profile"]["gpu_reserve_seconds"]
    )
    allocation = (
        2 * 64 * 6 * 12 * 5000 * 4 * 12
        + 2 * 8 * 6 * 12 * 5000 * 4 * 12
        + 2 * 64 * 6 * 12 * 5000 * 4
        + 2 * 2 * 64 * 4 * 12 * 5000
        + 12 * 4 * 12 * 5000 * 4
        + 128 * 1024**2
    )
    before_paths = (
        "synthetic_predecessor_closure.json",
        "timing_predecessor_closure.json",
        "coherence_predecessor_closure.json",
        "localizer_predecessor_closure.json",
        "admission_snapshot.json",
    )
    before = sum(
        p.stat().st_size
        for p in study.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and (
            p.relative_to(study).as_posix() in before_paths
            or p.relative_to(study).parts[0] == "source_snapshot"
        )
    )
    forecast["output_bytes"] = before + allocation
    checks = compare_values(forecast, profile["forecast"], "complete bounded-path forecast arithmetic", 0)
    gates = {name: forecast[name] < config["resources"][name] for name in forecast}
    gates["cuda_allocated_bytes"] = (
        first["peak_allocated_bytes"] < config["resources"]["cuda_allocated_bytes"]
    )
    require(
        gates == profile["forecast_gates"] and all(gates.values()),
        "actual-path full continuation conjunction",
    )
    require(
        main_ledger["gpu_wall_seconds"] == first["gpu_lock_seconds"] + full["gpu_lock_seconds"],
        "both actual GPU-lock phases counted exactly once",
    )
    for phase in (first, full):
        require(
            phase["free_cuda_bytes_at_admission"] >= config["resources"]["minimum_free_cuda_bytes"]
            and phase["peak_allocated_bytes"] < config["resources"]["cuda_allocated_bytes"],
            "actual CUDA memory gates",
        )
    for fit in core.FITS:
        for arm in core.ARMS:
            name = f"fit_{fit}_{arm}"
            predicted = array(study / "profile" / (name + "_synthetic_replay.npy"), (4, 12, 5000), np.float32)
            old = array(
                root / PARENT / f"fit_{fit}/{arm}/fresh_probabilities.npy", (64, 4, 12, 5000), np.float32
            )
            difference = float(np.max(np.abs(predicted.astype(np.float64) - old[0].astype(np.float64))))
            row = read_json(study / "profile" / (name + "_synthetic_replay.json"))
            require(
                difference <= 1e-6
                and row["maximum_absolute_difference"] == difference
                and row["all_passed"]
                and row["initial_tensor_state_sha256"] == first["tensor_state_sha256"][name],
                "actual original CUDA replay evidence",
            )
            require(
                file_sha256(study / "profile/probabilities" / (name + ".npy")) == first["map_sha256"][name]
                and file_sha256(study / "probabilities" / (name + ".npy")) == full["map_sha256"][name],
                "actual profile/full map identities",
            )
    require(
        profile["profile_rows"] == 192
        and profile["profile_batch8_forwards"] == 144
        and profile["full_batch8_forwards"] == 1152
        and profile["parent_batch4_replays"] == 12,
        "fixed actual-path operation roster",
    )
    return {
        "scalar_checks": checks,
        "profile_rows": 192,
        "parent_replays": 12,
        "forecast": forecast,
        "two_phase_gpu_lock_seconds": main_ledger["gpu_wall_seconds"],
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Preserve one local metadata object exclusively, never overwriting a prior attempt.

    Parameters
    ----------
    path : Path
        Local declared file path.
    value : dict[str, Any]
        Finite serializable local metadata object.

    Returns
    -------
    None
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def terminal(
    destination: Path,
    result: dict[str, Any],
    clock: dict[str, float],
    costs: dict[str, Any],
    timers: dict[int, Any],
) -> dict[str, bool]:
    """Retain receipt failures and measured once-only actual audit costs through final byte counting.

    Parameters
    ----------
    destination : Path
        New audit-only output directory preserving failures.
    result : dict[str, Any]
        Success or retained failure metadata.
    clock : dict[str, float]
        Whole-audit process CPU and monotonic wall start values.
    costs : dict[str, Any]
        Verified immediate main accounting anchor and exact cumulative costs.
    timers : dict[int, Any]
        Active timer and prior signal-handler declarations.

    Returns
    -------
    dict[str, bool]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    receipt, digest, retained = "summary.json", None, None
    try:
        write_new(destination / receipt, result)
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
        write_new(destination / receipt, result)
        digest = file_sha256(destination / receipt)
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "cpu": cpu < 600,
        "wall": wall < 900,
        "rss": rss < 2 * 1024**3,
        "output": retained is not None and retained + 16384 < 64 * 1024**2,
    }
    verified = bool(costs)
    ledger = {
        "stage": "background_editor_transfer_independent_audit_v1",
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
        "measurement_endpoint": "after receipt hash and byte enumeration before ledger serialization",
        "terminal_bookkeeping_reserve_seconds": 2,
        "terminal_bookkeeping_reserve_bytes": 16384,
        "technical_passed": result["all_passed"] and all(gates.values()),
    }
    for timer in timers:
        signal.setitimer(timer, 0)
    write_new(destination / "prospective_ledger.json", ledger)
    return gates


def run(
    root: Path,
    source: Path,
    main_admission_name: str,
    audit_admission_name: str,
    main_sha: str,
    audit_admission_sha: str,
    runner_sha: str,
    checker_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Execute one separately source-admitted CPU reconstruction of the complete saved experiment.

    Parameters
    ----------
    root : Path
        Local artifact checkout containing admitted saved outputs.
    source : Path
        Isolated source checkout containing immutable committed code/configs.
    main_admission_name : str
        Repository-relative original main identity admission.
    audit_admission_name : str
        Separate prospective audit admission, never substituted for the original main roster.
    main_sha : str
        Literal actual main summary SHA256.
    audit_admission_sha : str
        Literal separately committed audit admission SHA256.
    runner_sha : str
        Literal this-module SHA256.
    checker_sha : str
        Literal independent CLI SHA256.
    clock : dict[str, float]
        Whole-audit process CPU and monotonic wall start values.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed results, verified declarations or explicit validation failure.
    """
    study = root / STUDY
    destination = study / "independent_audit_v1"
    destination.mkdir(exist_ok=False)
    costs, timers = {}, {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 598 and time.perf_counter() - clock["wall"] < 898,
            "independent receipt time reserve",
        )
        require(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "independent audit RSS"
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent transfer audit timer {signum}")

    try:
        for timer, signum, ceiling, key in (
            (signal.ITIMER_REAL, signal.SIGALRM, 898, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 598, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            elapsed = (time.perf_counter() if key == "wall" else time.process_time()) - clock[key]
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
            "hidden CUDA and all numerical thread limits1",
        )
        require(
            file_sha256(Path(__file__)) == runner_sha
            and file_sha256(source / CHECKER) == checker_sha
            and file_sha256(source / audit_admission_name) == audit_admission_sha,
            "literal independent runtime, CLI and separate admission identities",
        )
        audit, admission = read_json(source / audit_admission_name), read_json(source / main_admission_name)
        require(
            file_sha256(study / "summary.json") == main_sha == audit["main_summary_sha256"],
            "actual main summary identity",
        )
        main = read_json(study / "summary.json")
        costs = accounting(root, study, main, admission)
        require(
            file_sha256(study / "prospective_ledger.json") == audit["main_ledger_sha256"],
            "actual main ledger identity",
        )
        admission, config = metadata_setup(root, source, main_admission_name, audit_admission_name)
        own = {
            **source_union(admission, audit),
            main_admission_name: file_sha256(source / main_admission_name),
            audit_admission_name: audit_admission_sha,
        }
        for name, digest in own.items():
            require(file_sha256(source / name) == digest, "immutable complete audit runtime source")
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        bindings = identities(root, source, study, main, admission, main_admission_name, guard)
        closure_checks = predecessors(root, source, study)
        input_checks = reconstruct_inputs(root, study, guard)
        profile = profile_checks(root, study, config, read_json(study / "prospective_ledger.json"))
        values = numerical(root, study, config, guard)
        require(
            main["new_training_updates"] == main["new_generator_forwards"] == main["raw_record_reads"] == 0
            and main["clinical_claim"] is False,
            "no new training/generator/raw/clinical operation",
        )
        require(
            main["operation_counts"]
            == {
                "parent_batch4_replays": 12,
                "profile_batch8_forwards": 144,
                "full_batch8_forwards": 1152,
                "metric_rows": 1536,
            },
            "complete full-path operation roster",
        )
        from PIL import Image

        for name, dimensions in (
            ("profile/fixed_examples.png", (1200, 700)),
            ("all_source_family_arm_results.png", (2520, 600)),
        ):
            with Image.open(study / name) as figure:
                require(
                    figure.format == "PNG" and figure.size == dimensions,
                    "fixed figure hash/signature/layout scope",
                )
        bindings += identities(root, source, study, main, admission, main_admission_name, guard)
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "scientific_passed": values["statistics"]["scientific_passed"],
            "main_summary_sha256": main_sha,
            "main_ledger_sha256": audit["main_ledger_sha256"],
            "audit_admission_sha256": audit_admission_sha,
            "source_sha256": own,
            "binding_checks": bindings,
            "predecessor_scalar_checks": closure_checks,
            "input_geometry_scalar_checks": input_checks,
            "profile": profile,
            "numerical": values,
            "gpu_used": False,
            "raw_record_reads": 0,
            "clinical_claim": False,
            "figure_scope": "hashes and PNG signature/dimensions, no pixel rerender or clinical reading",
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
        for timer, (signum, previous) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous)
    require(
        result["all_passed"] and all(gates.values()), "retained independent numerical-transfer audit failure"
    )
    return result
