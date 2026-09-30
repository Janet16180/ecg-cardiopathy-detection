"""Frozen encoder/context factorial using the audited Experiment 038 machinery.

The inherited per-cell decision flags are descriptive historical helper fields.
All eighteen cells (thirty-six fits) are scheduled independently of those flags.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import xlstm_study as base
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

ENCODERS = ("cnn", "multiscale", "patch")
SEEDS = (39042, 39043, 39044)
TIERS = (25, 50)
STAGES = ("prepare", "profile", "train", "readout", "audit")
NAME = "experiment039_encoder_context"
PROTOCOL = "docs/experiment-039-cpc-encoder-context.md"
DAY_CEILING_SECONDS = 28_800.0
CORRECTION_RESERVE_SECONDS = 3600.0
REPORT_RESERVE_SECONDS = 900.0
SOURCE_FILES = (PROTOCOL, "ecg_experiment/cpc_encoder_variants039.py",
                "ecg_experiment/cpc_xlstm_native_gru.py",
                "ecg_experiment/encoder_context_diagnostics039.py",
                "ecg_experiment/encoder_context_study039.py",
                "scripts/experiments/run_cpc_encoder_context039.py",
                "ecg_experiment/encoder_context_analysis039.py",
                "scripts/reports/report_cpc_encoder_context039.py",
                "scripts/coordination/run_encoder_context_day039.py")


def protocol_commit(root: Path) -> str:
    """Require the committed Experiment 039 protocol before execution.

    Parameters
    ----------
    root : Path
        Repository containing the committed protocol.

    Returns
    -------
    str
        Commit that last changed the unchanged protocol.
    """
    committed = subprocess.run(["git", "show", f"HEAD:{PROTOCOL}"], cwd=root,
                               check=True, capture_output=True).stdout
    if committed != (root / PROTOCOL).read_bytes():
        raise ValueError("Experiment 039 protocol must be committed unchanged before any score")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", PROTOCOL],
                          cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def verify_v4(root: Path, tier: int) -> dict[str, str]:
    """Reject cohort drift before opening any cached waveform.

    Parameters
    ----------
    root : Path
        Repository containing local cohort manifests.
    tier : int
        Cohort size in thousands: 25 or 50.

    Returns
    -------
    dict[str, str]
        Verified manifest, metadata and published receipt hashes.
    """
    directory = root / f"data/processed/clean_{tier}k_v4"
    manifest = directory / "train_manifest.csv"
    metadata = directory / "metadata.json"
    receipt = root / "outputs/data_quality/clean_cohorts_v4/receipt.json"
    hashes = {"manifest_sha256": sha256_file(manifest),
              "metadata_sha256": sha256_file(metadata)}
    published = json.loads(receipt.read_text())["tiers"][f"{tier}k"]
    if any(hashes[key] != published[key] for key in hashes):
        raise ValueError("Published v4 cohort bytes changed")
    if hashes["manifest_sha256"] != sha256_file(
            root / f"data/processed/clean_{tier}k_v3/train_manifest.csv"):
        raise ValueError("v4 and v3 manifests differ; the v3 waveform cache cannot be reused")
    hashes["published_receipt_sha256"] = sha256_file(receipt)
    return hashes


def audited_result(directory: Path) -> dict[str, Any]:
    """Check the unchanged saved result and prediction bytes against its audit.

    Parameters
    ----------
    directory : Path
        Cell output directory containing the saved audit.

    Returns
    -------
    dict[str, Any]
        Verified unchanged development result.
    """
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    audit = json.loads((directory / "audit.json").read_text())
    if (audit["status"] != "passed_development_only"
            or result["status"] != "complete_development_only"
            or audit["result_sha256"] != sha256_file(result_path)
            or result["predictions_sha256"] != sha256_file(directory / "development_predictions.npz")):
        raise ValueError("An unchanged audited development result is required")
    for filename, key in (("training.json", "training_sha256"),
                          ("features.npz", "features_sha256"),
                          ("head_parameters.npz", "head_parameters_sha256")):
        if key in result and sha256_file(directory / filename) != result[key]:
            raise ValueError(f"Audited {filename} changed")
    return result


def prior_integrity(root: Path, replay019: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    """Exactly replay both audited 038 tiers and the historical 019 readout.

    Parameters
    ----------
    root : Path
        Repository containing predecessor outputs.
    replay019 : Callable
        Original Experiment 019 integrity replay function.

    Returns
    -------
    dict[str, Any]
        Exactly replayed predecessor metrics and artifact hashes.
    """
    history = replay019(root)
    old_pool = base.pool(root)
    train, _ = base.training_examples(old_pool)
    dev = base.development_only_examples(old_pool, train)
    expected = {"record_ids": [row.record_id for row in dev],
                "patient_ids": [row.patient_id for row in dev],
                "targets": [row.target for row in dev]}
    replayed = {}
    for tier in TIERS:
        directory = root / f"outputs/experiment038_cpc_xlstm_v2/{tier}k"
        result = audited_result(directory)
        with np.load(directory / "development_predictions.npz", allow_pickle=False) as saved:
            names = set(expected) | {f"{budget}_{arm}" for budget in ("limited", "full")
                                     for arm in base.ARMS}
            if set(saved.files) != names:
                raise ValueError("Experiment 038 prediction schema changed")
            if any(not np.array_equal(saved[key], value) for key, value in expected.items()):
                raise ValueError("Experiment 038 development identities or targets changed")
            scores = {budget: {arm: base.metrics(saved["targets"], saved[f"{budget}_{arm}"])
                               for arm in base.ARMS} for budget in ("limited", "full")}
        if any(value != result["scores"][budget][arm][metric]
               for budget, arms in scores.items() for arm, metrics in arms.items()
               for metric, value in metrics.items()):
            raise ValueError("Experiment 038 exact metric replay mismatch")
        replayed[str(tier)] = {"scores": scores, "result_sha256": sha256_file(directory / "result.json"),
                              "predictions_sha256": result["predictions_sha256"]}
    return {"status": "passed_historical_integrity", "experiment019": history,
            "experiment038": replayed, "development_records": len(dev),
            "development_patients": len({row.patient_id for row in dev})}


def day_ledger(root: Path) -> dict[str, Any]:
    """Load all charged successful and failed study attempts without resetting.

    Parameters
    ----------
    root : Path
        Repository containing study outputs.

    Returns
    -------
    dict[str, Any]
        All charged attempts and their cumulative elapsed seconds.
    """
    path = root / "outputs" / NAME / "day_ledger.json"
    return json.loads(path.read_text()) if path.exists() else {"attempts": [], "total_seconds": 0.0}


def used_seconds(root: Path, tier: int) -> float:
    """Charge only new attempts, excluding historical cache construction.

    Parameters
    ----------
    root : Path
        Repository containing study outputs.
    tier : int
        Cohort size in thousands.

    Returns
    -------
    float
        Charged seconds for the configured encoder, seed and tier.
    """
    path = base.output(root, tier) / "stage_walltime.json"
    return float(sum(item["elapsed_seconds"] for item in json.loads(path.read_text())["attempts"])) \
        if path.exists() else 0.0


def record_stage(root: Path, tier: int, encoder: str, seed: int, stage: str,
                 seconds: float, status: str, original: Callable[..., None]) -> None:
    """Append the same attempt to the inherited cell ledger and the day ledger.

    Parameters
    ----------
    root : Path
        Repository containing study outputs.
    tier : int
        Cohort size in thousands.
    encoder : str
        Waveform encoder identifier.
    seed : int
        Initialization seed.
    stage : str
        Stage or diagnostic attempt name.
    seconds : float
        Measured elapsed wall time.
    status : str
        Attempt status: complete or failed.
    original : Callable
        Inherited cell-ledger writer.

    Returns
    -------
    None
        Both ledgers are updated atomically per file.
    """
    original(root, tier, stage, seconds, status)
    ledger = day_ledger(root)
    ledger["attempts"].append({"encoder": encoder, "seed": seed, "tier": tier,
                               "stage": stage, "elapsed_seconds": seconds, "status": status})
    ledger["total_seconds"] = sum(item["elapsed_seconds"] for item in ledger["attempts"])
    ledger["ceiling_seconds"] = DAY_CEILING_SECONDS
    write_json_atomic(root / "outputs" / NAME / "day_ledger.json", ledger, sort_keys=True)


def historical_cpu_seconds(root: Path, tier: int) -> float:
    """Reserve 1.5 times predecessor readout and audit measurements per cell.

    Parameters
    ----------
    root : Path
        Repository containing predecessor timing receipts.
    tier : int
        Cohort size in thousands.

    Returns
    -------
    float
        Conservative per-cell readout and audit time reserve.
    """
    path = root / f"outputs/experiment038_cpc_xlstm_v2/{tier}k/stage_walltime.json"
    attempts = json.loads(path.read_text())["attempts"]
    selected = {stage: max(item["elapsed_seconds"] for item in attempts
                           if item["stage"] == stage and item["status"] == "complete")
                for stage in ("readout", "audit")}
    return 1.5 * sum(selected.values())


def remaining_projection(root: Path, fallback: dict[str, Any]) -> float:
    """Reserve measured conservative work for every unfinished scheduled cell.

    Parameters
    ----------
    root : Path
        Repository containing study receipts.
    fallback : dict[str, Any]
        Measured profile used for cells awaiting a profile.

    Returns
    -------
    float
        Projected seconds for all unfinished scheduled cells.
    """
    projected = 0.0
    for encoder in ENCODERS:
        for seed in SEEDS:
            for tier in TIERS:
                directory = root / "outputs" / NAME / encoder / f"seed{seed}/{tier}k"
                if (directory / "audit.json").exists():
                    audited_result(directory)
                    continue
                profile_path = directory / "profile.json"
                profile = json.loads(profile_path.read_text()) if profile_path.exists() else fallback
                if not (directory / "result.json").exists():
                    training_path = directory / "training.json"
                    trained = (training_path.exists()
                               and json.loads(training_path.read_text())["status"] == "complete")
                    if not trained:
                        projected += (profile["projected_training_seconds"]
                                      + profile["projected_checkpoint_seconds"])
                    projected += profile["projected_feature_seconds"]
                projected += historical_cpu_seconds(root, tier)
                if not profile_path.exists():
                    projected += profile["profile_wall_seconds"] + profile["preflight_seconds"]
    return projected


def admit_schedule(root: Path, profile: dict[str, Any], active_seconds: float = 0.0,
                   extra_training_seconds: float = 0.0) -> None:
    """Gate the complete remaining schedule against the executable day ceiling.

    Parameters
    ----------
    root : Path
        Repository containing study receipts.
    profile : dict[str, Any]
        Measured profile for unfinished cells without their own profile.
    active_seconds : float, optional
        Elapsed current-stage work not yet recorded in the day ledger.
    extra_training_seconds : float, optional
        Additional remaining training cost from observed slower updates.

    Returns
    -------
    None
        Raises RuntimeError when projected work exceeds the ceiling.
    """
    projection = (day_ledger(root)["total_seconds"] + active_seconds
                  + remaining_projection(root, profile) + CORRECTION_RESERVE_SECONDS
                  + REPORT_RESERVE_SECONDS + extra_training_seconds)
    if projection > DAY_CEILING_SECONDS:
        raise RuntimeError(f"Full remaining Experiment 039 schedule projects {projection:.1f}s above 28800s")


def predecessor_hashes(root: Path) -> dict[str, str]:
    """Pin predecessor result, audit, predictions and timing used in admission.

    Parameters
    ----------
    root : Path
        Repository containing audited predecessor receipts.

    Returns
    -------
    dict[str, str]
        Repository-relative predecessor artifact hashes.
    """
    hashes = {}
    for tier in TIERS:
        directory = root / f"outputs/experiment038_cpc_xlstm_v2/{tier}k"
        audited_result(directory)
        for filename in ("result.json", "audit.json", "development_predictions.npz", "stage_walltime.json"):
            path = directory / filename
            hashes[to_stored(path)] = sha256_file(path)
    return hashes


@contextmanager
def configured(root: Path, encoder: str, seed: int) -> Iterator[None]:
    """Temporarily specialize the frozen shared library, restoring every hook.

    Parameters
    ----------
    root : Path
        Repository containing study receipts.
    encoder : str
        One of cnn, multiscale or patch.
    seed : int
        One of the three prespecified initialization seeds.

    Yields
    ------
    None
        Shared library uses this cell configuration until context exit.
    """
    if encoder not in ENCODERS or seed not in SEEDS:
        raise ValueError("Unknown Experiment 039 encoder or seed")
    from ecg_experiment.cpc_encoder_variants039 import architecture_spec, create_model

    originals = {name: getattr(base, name) for name in (
        "create_model", "OUTPUT_NAME", "SEED", "ORDER_SEED", "INTERVAL_SEED", "SOURCE_FILES",
        "protocol_commit", "identity", "require_50k_trigger", "prior_integrity", "used_seconds",
        "_elapsed_guard", "_pace_guard")}
    stage_started = time.monotonic()
    charged_day = day_ledger(root)["total_seconds"]

    def identity(study_root: Path, tier: int, receipt: dict[str, Any]) -> dict[str, Any]:
        v4 = verify_v4(study_root, tier)
        current = originals["identity"](study_root, tier, receipt)
        for path in (study_root / f"data/processed/clean_{tier}k_v4/metadata.json",
                     study_root / f"data/processed/clean_{tier}k_v4/train_manifest.csv",
                     study_root / "outputs/data_quality/clean_cohorts_v4/receipt.json"):
            current["files_sha256"][to_stored(path)] = sha256_file(path)
        current["files_sha256"].update(predecessor_hashes(study_root))
        current.update({"cohort_version": 4, "v4_cache_reuse": v4, "encoder": encoder,
                        "architectures": {arm: architecture_spec(encoder, arm) for arm in base.ARMS},
                        "schedule": "all_encoders_seeds_tiers_unconditional"})
        return current

    def require_50k(study_root: Path) -> None:
        audited_result(base.output(study_root, 25))

    def elapsed_guard(charged: float, started: float) -> None:
        originals["_elapsed_guard"](charged, started)
        if charged_day + time.monotonic() - stage_started > DAY_CEILING_SECONDS:
            raise RuntimeError("Experiment 039 exceeded its 28800-second executable day ceiling")

    def pace_guard(profile: dict[str, Any], charged: float, started: float,
                   arm: str, update: int, seconds_per_update: float) -> None:
        originals["_pace_guard"](profile, charged, started, arm, update, seconds_per_update)
        elapsed_guard(charged, started)
        extra_training = (1.5 * (base.UPDATES - update)
                          * max(0.0, seconds_per_update - profile["arms"][arm]["seconds_per_update"]))
        admit_schedule(root, profile, time.monotonic() - stage_started,
                       extra_training_seconds=extra_training)

    replacements = {
        "create_model": partial(create_model, encoder), "OUTPUT_NAME": f"{NAME}/{encoder}/seed{seed}",
        "SEED": seed, "ORDER_SEED": seed + 1000, "INTERVAL_SEED": 39045,
        "SOURCE_FILES": originals["SOURCE_FILES"] + SOURCE_FILES,
        "protocol_commit": protocol_commit, "identity": identity, "require_50k_trigger": require_50k,
        "prior_integrity": partial(prior_integrity, replay019=originals["prior_integrity"]),
        "used_seconds": used_seconds, "_elapsed_guard": elapsed_guard, "_pace_guard": pace_guard,
    }
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def _execute_stage(stage: str, root: Path, tier: int, device: str,
                   directory: Path, started: float) -> dict[str, Any]:
    """Validate reusable intermediate receipts and dispatch a library stage."""
    if stage == "readout" and (directory / "result.json").exists():
        raise ValueError("Development readout already exists; run audit instead")
    if stage == "prepare":
        result = base.prepare(root, tier)
    elif stage == "audit":
        result = base.audit(root, tier)
    elif stage == "profile" and (directory / "profile.json").exists():
        current = base.ensure_manifest(root, tier)
        result = base._passed_profile(root, tier, current)
        admit_schedule(root, result, time.monotonic() - started)
    elif (stage == "train" and (directory / "training.json").exists()
          and json.loads((directory / "training.json").read_text())["status"] == "complete"):
        current = base.ensure_manifest(root, tier)
        result = base._training_receipt(root, tier, current)
    else:
        if stage == "train":
            profile = json.loads((directory / "profile.json").read_text())
            admit_schedule(root, profile)
        result = getattr(base, stage)(root, tier, device)
        if stage == "profile":
            admit_schedule(root, result, time.monotonic() - started)
    return result


def execute(stage: str, root: Path, tier: int, encoder: str, seed: int,
            device: str = "cuda") -> dict[str, Any]:
    """Run a measured cell stage, charging failures and preserving completed scores.

    Parameters
    ----------
    stage : str
        One explicit prepare, profile, train, readout or audit stage.
    root : Path
        Repository containing the committed protocol and local data.
    tier : int
        Cohort size in thousands: 25 or 50.
    encoder : str
        Prespecified waveform encoder.
    seed : int
        Prespecified initialization seed.
    device : str, optional
        Profiled training device, normally cuda.

    Returns
    -------
    dict[str, Any]
        Validated stage receipt or development result.
    """
    if stage not in STAGES or tier not in TIERS:
        raise ValueError("Unknown Experiment 039 stage or tier")
    with configured(root, encoder, seed):
        started = time.monotonic()
        status = "failed"
        try:
            if day_ledger(root)["total_seconds"] >= DAY_CEILING_SECONDS:
                raise RuntimeError("Experiment 039 day ceiling exhausted")
            base.configure_runtime()
            protocol_commit(root)
            verify_v4(root, tier)
            directory = base.output(root, tier)
            if (directory / "audit.json").exists():
                audited_result(directory)
                if stage == "audit":
                    status = "complete"
                    return json.loads((directory / "audit.json").read_text())
                raise ValueError("Audited cell is immutable; run the convenience sequence to skip it")
            result = _execute_stage(stage, root, tier, device, directory, started)
            status = "complete"
            return result
        finally:
            record_stage(root, tier, encoder, seed, stage, time.monotonic() - started,
                         status, base.record_stage)


def run(root: Path, tier: int, encoder: str, seed: int,
        device: str = "cuda") -> dict[str, Any]:
    """Run one cell; completed audited cells are checked and skipped unchanged.

    Parameters
    ----------
    root : Path
        Repository containing the committed protocol and local data.
    tier : int
        Cohort size in thousands: 25 or 50.
    encoder : str
        Prespecified waveform encoder.
    seed : int
        Prespecified initialization seed.
    device : str, optional
        Profiled training device, normally cuda.

    Returns
    -------
    dict[str, Any]
        Unchanged audited development result for this cell.
    """
    directory = root / "outputs" / NAME / encoder / f"seed{seed}/{tier}k"
    if (directory / "result.json").exists():
        if not (directory / "audit.json").exists():
            execute("audit", root, tier, encoder, seed, device)
        with configured(root, encoder, seed):
            protocol_commit(root)
            verify_v4(root, tier)
            base.ensure_manifest(root, tier)
            return audited_result(directory)
    for stage in STAGES:
        execute(stage, root, tier, encoder, seed, device)
    return audited_result(directory)
