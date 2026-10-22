"""Measured independent verification of matched generated and mixed-background fits."""

import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_background_editor_transfer_audit_run as previous
from ecg_experiment import ssl_synthetic_mixed_background_audit as core
from ecg_experiment import ssl_synthetic_mixed_background_audit_inputs as inputs
from ecg_experiment.ssl_synthetic_ludb_localizer_audit import functional_replay
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_mixed_background_v1.json"
RUNNER = "ecg_experiment/ssl_synthetic_mixed_background_audit_run.py"
CHECKER = "scripts/validation/ssl_synthetic_mixed_background_audit.py"
PROTOCOL = "docs/ssl-synthetic-localization-mixed-background-independent-audit-v1.md"
TESTS = "tests/test_ssl_synthetic_mixed_background_audit_runtime.py"
CORE = "ecg_experiment/ssl_synthetic_mixed_background_audit.py"
INPUTS = "ecg_experiment/ssl_synthetic_mixed_background_audit_inputs.py"
CORE_TESTS = "tests/test_ssl_synthetic_mixed_background_audit.py"
STUDY = "outputs/ssl_synthetic_localization/mixed_background_v1"
PRIOR37_SHA = "73758dc5c2f8b20543a63549d61fc6ce669286cd988c2dd135c6453e1a024d4f"
PRIOR37_RECEIPT = "c4d2b4cd77bbdabbebfd8b7dfea261a1bb5cd595abb6aefa4af3509006744923"
AUDIT36_SHA = "d64d6ee087dd5547ae02f02a3adc58e141be0d53b9f1d2474a4f2f01aaef1b12"
AUDIT36_RECEIPT = "e4f60657ab5c19007bdf91a53b4d27b58408d51dd732ebbc7b7ac3aa9c03afd6"
SHARED_SOURCES = previous.SHARED_SOURCES | {
    previous.RUNNER,
    previous.AUDIT_CORE,
    "ecg_experiment/ssl_synthetic_ludb_audit.py",
}


def source_union(main: dict[str, Any], audit: dict[str, Any]) -> dict[str, str]:
    """Bind every independent imported helper without overriding the executed main.

    Parameters
    ----------
    main, audit : dict
        Executed main admission and separate prospective independent admission.

    Returns
    -------
    dict
        Complete immutable source identity roster.
    """
    required = {RUNNER, CHECKER, PROTOCOL, TESTS, CORE, INPUTS, CORE_TESTS}
    require(required.issubset(audit["source_sha256"]), "explicit complete independent source roster")
    first, second = main["source_sha256"], audit["source_sha256"]
    require(all(first[k] == v for k, v in second.items() if k in first), "no executed-source override")
    combined = {**first, **second}
    require(SHARED_SOURCES.issubset(combined), "all imported independent helpers source-bound")
    return combined


def binding_path(root: Path, source: Path, name: str) -> Path:
    """Choose the declared lexical source or artifact root without resolving raw aliases.

    Parameters
    ----------
    root, source : Path
        Explicit artifact and immutable source roots.
    name : str
        Repository-relative admitted filename.

    Returns
    -------
    Path
        Confined source/artifact path; raw aliases stay read-only.
    """
    value = Path(name)
    require(not value.is_absolute() and ".." not in value.parts, "repository-relative binding")
    return (root if name.startswith(("outputs/", "data/")) else source) / value


def accounting(root: Path, study: Path, main: dict[str, Any]) -> dict[str, Any]:
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
        and "independent_audit_v1" not in p.relative_to(study).parts
        and p.name != "prospective_ledger.json"
    )
    require(retained == ledger["output_bytes_before_terminal_ledger"], "exact retained main bytes")
    return {
        "prior_ledger_sha256": file_sha256(study / "prospective_ledger.json"),
        "stages_through_main": ledger["current_machine_stages"],
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def training_schema(study: Path) -> None:
    """Validate all nested training JSON containers and saved array headers before numerical work.

    Parameters
    ----------
    study : Path
        Executed new main output directory.

    Returns
    -------
    None
        Raises before the audit on any incompatible typed artifact schema.
    """
    previous.array(study / "mixed_observed.npy", (80, 4, 12, 5000), np.float32)
    previous.array(study / "mixed_support.npy", (80, 4, 12, 5000), np.bool_)
    previous.array(study / "mixed_strict_support.npy", (80, 2, 12, 5000), np.bool_)
    metadata = previous.json_array(study / "training_input_metadata.json")
    require(
        len(metadata) == 40 and [r["geometry_index"] for r in metadata] == list(range(40, 80)),
        "complete training metadata roster",
    )
    for row in metadata:
        require(
            isinstance(row["signal_metadata"], dict) and isinstance(row["original_editor"], dict),
            "typed physical/editor metadata",
        )
        require(
            {
                "source_lead_order",
                "canonical_lead_order",
                "source_units",
                "source_adc_gain",
                "source_baseline",
                "canonical_units",
                "canonical_adc_gain",
                "canonical_baseline",
                "fs",
                "samples",
            }.issubset(row["signal_metadata"]),
            "physical metadata schema",
        )
    for fit in (1, 2, 3):
        common = study / f"fit_{fit}"
        previous.array(common / "training_schedule.npy", (400, 8), np.int64)
        previous.array(common / "background_schedule.npy", (400, 2), np.int64)
        read_json(common / "fit_design.json")
        require((common / "initial_state.pt").is_file(), "common initial state present")
        for arm in core.TRAINING_ARMS:
            directory = common / arm / "B"
            previous.array(directory / "losses.npy", (400,), np.float64)
            previous.array(directory / "loss_components.npy", (400, 5), np.float64)
            summary = read_json(directory / "summary.json")
            require(
                {
                    "training_arm",
                    "fit_id",
                    "updates",
                    "positive_weight",
                    "final_tensor_state_sha256",
                    "baseline_replay",
                    "output_sha256",
                }.issubset(summary),
                "fit summary fields",
            )
            for name in ("initial_state.pt", "final_state.pt", "optimizer.pt"):
                require((directory / name).is_file(), "complete model/Adam artifacts")
    for arm in core.TRAINING_ARMS:
        profile = study / "training_profile" / arm
        for phase, count in (("original", 3), ("replay", 2)):
            previous.array(profile / phase / "losses.npy", (count,), np.float64)
            previous.array(profile / phase / "loss_components.npy", (count, 5), np.float64)
        previous.array(profile / "step1_logits.npy", (8, 12, 5000), np.float32)
        previous.array(profile / "step1_case_indices.npy", (8,), np.int64)
        summary = read_json(profile / "summary.json")
        require(
            isinstance(summary["original_step_timings"], list)
            and len(summary["original_step_timings"]) == 3
            and isinstance(summary["restored_steps"], list)
            and len(summary["restored_steps"]) == 2,
            "complete profile step/timing schema",
        )
        for phase in ("original_step1", "original_step2", "original_step3", "replay_step2", "replay_step3"):
            for kind in ("model", "optimizer"):
                require((profile / f"{phase}_{kind}.pt").is_file(), "complete profile recovery trees")


def evaluation_schema(study: Path) -> None:
    """Validate every evaluation artifact container/geometry without recalculating saved scores.

    Parameters
    ----------
    study : Path
        Executed main output root.

    Returns
    -------
    None
        Raises on any incompatible nested schema before the numerical attempt.
    """
    values = previous.rows(study / "record_rows.jsonl")
    expected = {
        (arm, source, slot, fit)
        for arm in core.TRAINING_ARMS
        for source in core.transfer.SOURCES
        for slot in range(64)
        for fit in (1, 2, 3)
    }
    require(
        len(values) == 768
        and {(r["training_arm"], r["source"], r["slot"], r["fit_id"]) for r in values} == expected,
        "complete distinct evaluation identity roster",
    )
    for row in values:
        require(
            row["arm"] == "B" and set(row["positives"]) == set(core.transfer.POSITIVES),
            "fixed B/four-positive nested schema",
        )
        require(
            all(set(core.METRICS).issubset(p) for p in row["positives"].values()), "positive metric fields"
        )
        require(
            set(row["controls"]) == {"sham", "nuisance"}
            and all(set(core.CONTROLS) == set(p) for p in row["controls"].values()),
            "control schema",
        )
    previous.array(study / "bootstrap.npy", (2000, 348), np.float64)
    for fit in (1, 2, 3):
        previous.array(study / "probabilities" / f"fit_{fit}_mixed.npy", (2, 64, 6, 12, 5000), np.float32)
        previous.array(
            study / "mixed_subset_probabilities" / f"fit_{fit}_mixed.npy", (2, 8, 6, 12, 5000), np.float32
        )
        previous.array(
            study / "profile/frozen_B_probabilities" / f"fit_{fit}_B.npy", (2, 8, 6, 12, 5000), np.float32
        )
        read_json(study / f"fit_{fit}_mixed_inference_integrity.json")
        read_json(study / "profile" / f"fit_{fit}_B_replay.json")
    for name in ("resource_profile_gpu_phase.json", "full_training_inference_gpu_phase.json"):
        phase = read_json(study / name)
        require(
            {"cpu_seconds", "wall_seconds", "gpu_lock_seconds", "cuda", "peak_allocated_bytes"}.issubset(
                phase
            ),
            "measured GPU phase schema",
        )
    profile = read_json(study / "profile/summary.json")
    require(
        {"actual_at_forecast", "recurring_CPU_scoring", "forecast", "forecast_gates"}.issubset(profile),
        "profile forecast schema",
    )


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
    accounting(root, study, summary)
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
    training_schema(study)
    evaluation_schema(study)
    return main, audit, config


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
        if p.is_file() and not p.is_symlink() and "independent_audit_v1" not in p.relative_to(study).parts
    }
    require(
        actual == set(main["output_sha256"]) | {"summary.json", "prospective_ledger.json"},
        "complete immutable main artifact membership",
    )
    return count


def predecessor(root: Path, source: Path, study: Path, config: dict, guard: Any) -> int:
    """Close the actual spent transfer panel, original catalogs and every conditional parent draw.

    Parameters
    ----------
    root, source, study : Path
        Artifact, source and new main directories.
    config : dict
        Frozen original evaluation root/config identities.
    guard : callable
        Whole-audit resource guard.

    Returns
    -------
    int
        Complete independent parent scalar comparison count.
    """
    directory = root / config["evaluation_root"]
    expected = {
        "summary.json": "c3ea28105ef99659b0f8854fd32430169ecb48f456b02b8b709fec306da93ba7",
        "prospective_ledger.json": "8cf80790df2ef3ce3a5c0d2cf539b1e6311697a01a4394e672c7f6b1d819e304",
        "independent_audit_v1/summary.json": AUDIT36_RECEIPT,
        "independent_audit_v1/prospective_ledger.json": AUDIT36_SHA,
    }
    for name, digest in expected.items():
        require(file_sha256(directory / name, guard) == digest, "actual35/36 immutable predecessor")
    saved = read_json(directory / "summary.json")
    audit = read_json(directory / "independent_audit_v1/summary.json")
    require(
        audit["all_passed"] and audit["mismatch_count"] == 0, "actual36 completed independent verification"
    )
    names = [
        "inputs.npy",
        "support.npy",
        "strict_support.npy",
        "geometry.jsonl",
        "record_ids.json",
        "record_rows.jsonl",
        "bootstrap.npy",
    ] + [f"probabilities/fit_{fit}_B.npy" for fit in (1, 2, 3)]
    for name in names:
        require(
            file_sha256(directory / name, guard) == saved["output_sha256"][name],
            "original35 output catalog anchors referenced evaluation asset",
        )
    rows = previous.rows(directory / "record_rows.jsonl")
    statistics, draws = previous.aggregate_rows(rows, read_json(source / config["evaluation_config"]))
    count = compare_values(
        statistics, saved["statistics"], "complete actual35 parent report", tolerance=1e-12
    )
    old_draws = previous.array(directory / "bootstrap.npy", draws.shape, np.float64)
    require(np.max(np.abs(draws - old_draws)) <= 1e-12, "complete original2000 parent draws")
    closure = read_json(study / "predecessor_closure.json")
    require(
        closure["all_exact"] and closure["all2000draws_exact"] and closure["B_map_rows"] == 384,
        "recorded complete parent closure",
    )
    count += compare_values(statistics, closure["statistics"], "retained parent closure", tolerance=1e-12)
    return count


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
                            replay = functional_replay(state, observed[sourcepos, slot, variant])
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


def profile_training(study: Path, guard: Any) -> dict[str, Any]:
    """Verify both measured training paths and exact complete next-update restore evidence.

    Parameters
    ----------
    study : Path
        Executed main output directory.
    guard : callable
        Whole-audit time/RSS guard.

    Returns
    -------
    dict
        Both independently checked five-update disposable profiles and all timings.
    """
    reports = {}
    schedule = previous.array(study / "fit_1/training_schedule.npy", (400, 8), np.int64)
    for arm in core.TRAINING_ARMS:
        guard()
        directory = study / "training_profile" / arm
        report = read_json(directory / "summary.json")
        original = previous.array(directory / "original/losses.npy", (3,), np.float64)
        replay = previous.array(directory / "replay/losses.npy", (2,), np.float64)
        components = previous.array(directory / "original/loss_components.npy", (3, 5), np.float64)
        restored = previous.array(directory / "replay/loss_components.npy", (2, 5), np.float64)
        require(
            np.array_equal(original[1:], replay) and np.array_equal(components[1:], restored),
            "every scalar/component restored update exactly reproduces",
        )
        require(np.array_equal(original, components[:, -1]), "profile loss/component identity")
        require(
            np.array_equal(previous.array(directory / "step1_case_indices.npy", (8,), np.int64), schedule[0]),
            "profile uses actual first scheduled batch",
        )
        require(
            np.isfinite(previous.array(directory / "step1_logits.npy", (8, 12, 5000), np.float32)).all(),
            "retained complete actual first-profile logits",
        )
        for step in (1, 2, 3):
            model = inputs.load_state(directory / f"original_step{step}_model.pt")
            optimizer = inputs.load_state(directory / f"original_step{step}_optimizer.pt")
            inputs.verify_optimizer(optimizer, model, step)
            if step > 1:
                require(
                    inputs.exact_tree(model, inputs.load_state(directory / f"replay_step{step}_model.pt")),
                    "restored full model exactly reproduces next update",
                )
                require(
                    inputs.exact_tree(
                        optimizer, inputs.load_state(directory / f"replay_step{step}_optimizer.pt")
                    ),
                    "restored full Adam exactly reproduces next update",
                )
        require(
            report["complete_state_replay_exact"] and report["fit_id"] == 1 and report["objective"] == "B",
            "complete saved training recovery report",
        )
        compare_values(original.tolist(), report["original_losses"], "original profile losses", tolerance=0)
        compare_values(replay.tolist(), report["replayed_losses"], "restored profile losses", tolerance=0)
        reports[arm] = report
    return reports


def profile_maps(root: Path, study: Path, config: dict, states: dict, guard: Any) -> int:
    """Verify known-parent profile maps and every retained mixed-subset/full identity.

    Parameters
    ----------
    root, study : Path
        Artifact and executed main roots.
    config, states : dict
        Frozen evaluation declarations and independently verified fitted states.
    guard : callable
        Whole-audit resource guard.

    Returns
    -------
    int
        Complete saved frozen-profile row scalar comparison count.
    """
    evaluation = root / config["evaluation_root"]
    observed = previous.array(evaluation / "inputs.npy", (2, 64, 6, 12, 5000), np.float32)
    masks = previous.array(evaluation / "support.npy", (2, 64, 4, 12, 5000), np.bool_)
    saved = {
        (r["fit_id"], r["source"], r["slot"]): r for r in previous.rows(study / "profile/record_rows.jsonl")
    }
    require(len(saved) == 48, "all48 frozen-profile model-background rows")
    scalars = 0
    for fit in (1, 2, 3):
        guard()
        known = previous.array(
            study / "profile/frozen_B_probabilities" / f"fit_{fit}_B.npy", (2, 8, 6, 12, 5000), np.float32
        )
        parent = previous.array(
            evaluation / "probabilities" / f"fit_{fit}_B.npy", (2, 64, 6, 12, 5000), np.float32
        )
        maximum = 0.0
        for sourcepos, source_name in enumerate(core.transfer.SOURCES):
            for slot in range(8):
                guard()
                maximum = max(
                    maximum,
                    float(
                        np.max(np.abs(known[sourcepos, slot].astype(np.float64) - parent[sourcepos, slot]))
                    ),
                )
                row = core.transfer.score_record(
                    known[sourcepos, slot],
                    observed[sourcepos, slot],
                    masks[sourcepos, slot],
                    source_name,
                    slot,
                    fit,
                    "B",
                    masks[sourcepos, (slot + 1) % 64],
                )
                row["training_arm"] = "generated"
                scalars += compare_values(
                    row, saved[fit, source_name, slot], "known-profile row", tolerance=1e-12
                )
        proof = read_json(study / "profile" / f"fit_{fit}_B_replay.json")
        require(
            maximum == proof["maximum_absolute_difference"]
            and proof["atol"] == 1e-6
            and proof["passed"]
            and maximum <= 1e-6,
            "all known-profile outputs match admitted parent",
        )
        subset = previous.array(
            study / "mixed_subset_probabilities" / f"fit_{fit}_mixed.npy", (2, 8, 6, 12, 5000), np.float32
        )
        complete = previous.array(
            study / "probabilities" / f"fit_{fit}_mixed.npy", (2, 64, 6, 12, 5000), np.float32
        )
        require(np.array_equal(subset, complete[:, :8]), "every mixed subset/full map value exact")
        proof = read_json(study / f"fit_{fit}_mixed_inference_integrity.json")
        require(
            proof["subset_full_exact"]
            and proof["state_unchanged"]
            and proof["final_tensor_state_sha256"] == states[f"fit_{fit}_mixed"]["tensor_state_sha256"],
            "all mixed model states unchanged during inference",
        )
        require(
            file_sha256(study / "mixed_subset_probabilities" / f"fit_{fit}_mixed.npy", guard)
            == proof["subset_sha256"]
            and file_sha256(study / "probabilities" / f"fit_{fit}_mixed.npy", guard) == proof["full_sha256"],
            "complete subset/full map identities",
        )
    return scalars


def profile_forecast(study: Path, config: dict, training: dict) -> dict[str, Any]:
    """Independently recompute conservative full-path forecasts and additive two-lock costs.

    Parameters
    ----------
    study : Path
        Executed main output directory.
    config, training : dict
        Frozen resource formula and independently verified training-profile reports.

    Returns
    -------
    dict
        Complete reconstructed forecasts, gate outcomes and exact actual GPU accounting.
    """
    phase = read_json(study / "resource_profile_gpu_phase.json")
    full = read_json(study / "full_training_inference_gpu_phase.json")
    profile = read_json(study / "profile/summary.json")
    require(phase["training"] == training, "saved phase contains both complete verified profiles")
    forecast = {}
    for cost, reserve in (("cpu_seconds", "cpu_reserve_seconds"), ("wall_seconds", "wall_reserve_seconds")):
        recurring = max(t[cost] for bank in training.values() for t in bank["original_step_timings"][1:])
        inference = (phase["recurring_inference"][cost] + profile["recurring_CPU_scoring"][cost]) / 36
        forecast[cost] = (
            profile["actual_at_forecast"][cost]
            + 2 * 2400 * recurring
            + 2 * 324 * inference
            + config["profile"][reserve]
        )
    recurring_wall = max(
        t["wall_seconds"] for bank in training.values() for t in bank["original_step_timings"][1:]
    )
    forecast["gpu_lock_seconds"] = (
        phase["gpu_lock_seconds"]
        + 2 * 2400 * recurring_wall
        + 2 * 324 * phase["recurring_inference"]["wall_seconds"] / 36
        + config["profile"]["gpu_reserve_seconds"]
    )
    snapshot_bytes = sum(p.stat().st_size for p in (study / "source_snapshot").rglob("*") if p.is_file())
    initial_bytes = snapshot_bytes + (study / "admission_snapshot.json").stat().st_size
    initial_bytes += (study / "predecessor_closure.json").stat().st_size
    forecast["output_bytes"] = (
        initial_bytes
        + 80 * 4 * 12 * 5000 * 4
        + 80 * 4 * 12 * 5000
        + 80 * 2 * 12 * 5000
        + 3 * 2 * 64 * 6 * 12 * 5000 * 4
        + 2 * 3 * 2 * 8 * 6 * 12 * 5000 * 4
        + 32 * 1024**2
        + config["profile"]["output_reserve_bytes"]
    )
    compare_values(
        forecast, profile["forecast"], "independent complete actual-path forecast", tolerance=1e-12
    )
    gates = {key: value < config["resources"][key] for key, value in forecast.items()}
    gates["cuda_allocated_bytes"] = (
        phase["peak_allocated_bytes"] < config["resources"]["cuda_allocated_bytes"]
    )
    require(gates == profile["forecast_gates"] and all(gates.values()), "independent full forecast gates")
    ledger = read_json(study / "prospective_ledger.json")
    require(
        phase["gpu_lock_seconds"] + full["gpu_lock_seconds"] == ledger["gpu_wall_seconds"],
        "all actual shared-lock occupancy counted exactly once",
    )
    for value in (phase, full):
        require(
            0 < value["gpu_lock_seconds"] <= value["wall_seconds"]
            and value["cuda"]["free_bytes"] >= 2 * 1024**3
            and value["peak_allocated_bytes"] < 2 * 1024**3,
            "actual CUDA admission/allocation/lock gates",
        )
    return {
        "forecast": forecast,
        "all_gates": gates,
        "two_phase_gpu_lock_seconds": ledger["gpu_wall_seconds"],
    }


def operation_checks(study: Path) -> None:
    """Verify the full declared operation roster and honest nonclinical scope.

    Parameters
    ----------
    study : Path
        Executed new main output root.

    Returns
    -------
    None
        Raises for any missing/extra update, decode or inference count.
    """
    summary = read_json(study / "summary.json")
    require(
        summary["operation_counts"]
        == {
            "profile_update_attempts": 10,
            "profile_updates": 10,
            "generated_update_attempts": 1200,
            "generated_updates": 1200,
            "mixed_update_attempts": 1200,
            "mixed_updates": 1200,
            "frozen_B_profile_batches": 36,
            "mixed_subset_batches": 36,
            "mixed_full_batches": 288,
        },
        "all2400 scientific updates plus10 profiles and full inference roster",
    )
    require(
        summary["raw_record_decodes"] == summary["raw_record_decode_attempts"] == 40
        and summary["annotation_reads"] == summary["clinical_labels_interpreted"] == 0
        and summary["new_generator_forwards"] == 0
        and summary["clinical_claim"] is False
        and summary["final_input_reverification_completed"],
        "bounded raw/scope/termination declarations",
    )
    full = read_json(study / "full_training_inference_gpu_phase.json")
    require(
        summary["training_and_inference"] == full and len(full["training_results"]) == 6,
        "all six training summaries and inference integrity preserved",
    )
    runtime = summary["runtime"]
    require(
        runtime["deterministic"]
        and runtime["cudnn_deterministic"]
        and not runtime["cudnn_benchmark"]
        and not runtime["matmul_tf32"]
        and not runtime["cudnn_tf32"]
        and runtime["threads"] == runtime["interop_threads"] == 1,
        "actual deterministic runtime",
    )


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
        "stage": "mixed_background_independent_audit_v1",
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
    destination = study / "independent_audit_v1"
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
        costs = accounting(root, study, main)
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
        result["all_passed"] and all(gates.values()), "retained independent mixed-background audit failure"
    )
    return result
