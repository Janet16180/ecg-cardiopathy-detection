"""Cold-start references and training quality with fresh administrative roles and optional verified caches."""

from __future__ import annotations

import json
import resource
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import ROOT
from .age_cohort_assembly import verify_unchanged
from .age_cohort_inputs import RECOVERY, SPLIT_SHA256, parse_boolean, stored, verified_table
from .challenge_training_recovery import (
    RELEASES,
    assess_training_record,
    exclude_reference_overlaps,
    supervised_window_duplicates,
    verified_training_rows,
)
from .downloads import parse_checksums
from .files import sha256_file, write_json_atomic
from .heldout_window_recovery import (
    challenge_references,
    hash_challenge_reference,
    hash_ptb_reference,
    ptb_reference_jobs,
    validate_reference_table,
)
from .raw_role_bootstrap import OUTPUT as RAW_ROLES

OUTPUT = RECOVERY / "cold_start"
PROTOCOL = Path("docs/cold-start-cohort-protocol-2026-10-28.md")
SOURCE_FILES = (
    "ecg_experiment/cold_data_stages.py", "scripts/data/bootstrap_cold_data_stages.py",
    "ecg_experiment/challenge_training_recovery.py", "ecg_experiment/heldout_window_recovery.py",
    "ecg_experiment/recovery_views.py", "ecg_experiment/raw_role_bootstrap.py",
    "ecg_experiment/ecg_quality.py", "ecg_experiment/public_sources.py", "ecg_experiment/files.py",
)


def raw_roles(input_root: Path) -> tuple[dict, dict[str, str]]:
    """Require new raw-derived administrative provenance, independently of historical scientific stages."""
    path = RAW_ROLES / "receipt.json"
    receipt = json.loads((input_root / path).read_text())
    if not receipt.get("complete") or receipt.get("scope") != "administrative_roles_raw_quality":
        raise ValueError("Fresh completed raw administrative roles are required")
    verify_unchanged(input_root, receipt["input_sha256"] | receipt["artifacts"])
    verify_unchanged(ROOT, receipt["recipe_sha256"])
    return receipt, {stored(path): sha256_file(input_root / path), **receipt["artifacts"]}


def source_recipe() -> dict[str, str]:
    """Capture executable and protocol bytes before any data-stage use."""
    return {stored(Path(name)): sha256_file(ROOT / name) for name in (*SOURCE_FILES, str(PROTOCOL))}


def fresh_sph_references(input_root: Path) -> tuple[list[dict], dict[str, str]]:
    """Read only newly derived administrative IDs, roles and hashes, without SPH outcomes."""
    _, inputs = raw_roles(input_root)
    frame = pd.read_csv(input_root / RAW_ROLES / "sph_rows.csv", dtype=str, keep_default_na=False,
                        usecols=["ecg_id", "patient_id", "role", "signal_sha256"])
    rows = [{
        "record_id": "sph:" + row.ecg_id, "patient_id": "sph:" + row.patient_id,
        "patient_identity_known": True, "source": "sph", "role": row.role, "path": "",
        "canonical_sha256": row.signal_sha256, "full_signal_sha256": "",
        "view_policy": "fresh_raw_sph_first5000", "source_samples": 5000,
        "window_start": 0, "skip_reason": "",
    } for row in frame.itertuples(index=False)]
    return rows, inputs


def verified_cache(
    input_root: Path, directory: Path, scope: str,
) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """Reuse a cache only when available code, input and table bindings all pass; otherwise decode raw."""
    path = directory / "receipt.json"
    if not (input_root / path).exists():
        return None
    receipt = json.loads((input_root / path).read_text())
    if any(not (input_root / name).exists() for name in receipt.get("input_sha256", {})):
        return None
    verify_unchanged(input_root, receipt.get("input_sha256", {}))
    verify_unchanged(ROOT, receipt["source_sha256"])
    frame, _, hashes = verified_table(input_root, path, scope, True)
    return frame, hashes


def finish(input_root: Path, output: Path, frame: pd.DataFrame, inputs: dict[str, str],
           recipe: dict[str, str], scope: str, extra: dict) -> dict:
    """Publish a new explicit data-stage receipt after all input and source bindings remain unchanged."""
    verify_unchanged(input_root, inputs)
    verify_unchanged(ROOT, recipe)
    destination = input_root / output
    destination.mkdir(parents=True, exist_ok=False)
    table = output / ("reference_rows.csv" if "canonical_sha256" in frame else "training_rows.csv")
    frame.to_csv(input_root / table, index=False)
    receipt = {
        "complete": True, "scope": scope, "original_split_sha256": SPLIT_SHA256,
        "input_sha256": inputs, "source_sha256": {name: recipe[name] for name in SOURCE_FILES},
        "recipe_sha256": recipe, "records": len(frame), "table_path": stored(table),
        "table_sha256": sha256_file(input_root / table), "profile_gate_passed": True,
        "model_scores_read_or_computed": 0, **extra,
    }
    verify_unchanged(input_root, inputs)
    verify_unchanged(ROOT, recipe)
    write_json_atomic(destination / "receipt.json", receipt, sort_keys=True)
    return receipt


def profile_gate(input_root: Path, stage: str) -> dict[str, str]:
    """Require a completed raw-path resource profile with matching producer before full processing."""
    path = OUTPUT / (stage + "_profile") / "receipt.json"
    receipt = json.loads((input_root / path).read_text())
    if receipt.get("scope") != "resource_profile" or not receipt.get("profile_gate_passed"):
        raise ValueError("Cold data stage requires its real raw-path profile")
    verify_unchanged(ROOT, receipt["recipe_sha256"])
    if sha256_file(input_root / receipt["table_path"]) != receipt["table_sha256"]:
        raise ValueError("Cold data resource profile changed")
    return {stored(path): sha256_file(input_root / path)}


def collected(rows: Iterable[dict], count: int, label: str) -> list[dict]:
    """Retain bounded-worker results in order and report meaningful raw-stage progress."""
    result = []
    for index, row in enumerate(rows, 1):
        result.append(row)
        if index % 2048 == 0:
            print(f"{label} {index}/{count}", flush=True)
    return result


def references(input_root: Path, workers: int = 4, limit: int | None = None,
               reuse_cache: bool = False) -> dict:
    """Build complete label-free references from fresh SPH roles and optional verified raw hashes."""
    output = OUTPUT / ("heldout_references_profile" if limit else "heldout_references")
    if (input_root / output).exists():
        raise FileExistsError(input_root / output)
    recipe = source_recipe()
    gate = profile_gate(input_root, "heldout_references") if limit is None else {}
    sph_rows, inputs = fresh_sph_references(input_root)
    inputs.update(gate)
    challenge, challenge_jobs, challenge_inputs = challenge_references(input_root)
    ptb, ptb_jobs, ptb_inputs = ptb_reference_jobs(input_root)
    inputs.update(challenge_inputs | ptb_inputs)
    intended = pd.Series([row["source"] for row in challenge + ptb + sph_rows] +
                         [job[3]["source"] for job in challenge_jobs] + ["ptbxl"] * len(ptb_jobs))
    expected = intended.value_counts().sort_index().to_dict()
    cached = verified_cache(input_root, RECOVERY / "heldout_references", "mechanical_reference_recovery") \
        if reuse_cache and limit is None else None
    if cached is not None:
        frame, cache_inputs = cached
        frame = frame[frame.source != "sph"]
        frame = pd.concat([frame, pd.DataFrame(sph_rows)], ignore_index=True)
        inputs.update(cache_inputs)
    else:
        if limit is not None:
            groups = pd.Series([job[3]["source"] for job in challenge_jobs])
            challenge_jobs = [challenge_jobs[index] for index in groups.groupby(groups).head(limit).index]
            ptb_jobs = ptb_jobs[:limit]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            computed = collected(pool.map(hash_challenge_reference, challenge_jobs, chunksize=16),
                                 len(challenge_jobs), "Cold Challenge held-out canonical references")
            computed += collected(pool.map(hash_ptb_reference, ptb_jobs, chunksize=16),
                                  len(ptb_jobs), "Cold PTB held-out canonical references")
        frame = pd.DataFrame(challenge + ptb + sph_rows + computed).drop(columns="worker_peak_rss_kib")
    frame = frame.sort_values(["source", "record_id"], kind="stable").reset_index(drop=True)
    actual = validate_reference_table(frame, expected, limit is None)
    return finish(input_root, output, frame, inputs, recipe,
                  "resource_profile" if limit else "mechanical_reference_recovery",
                  {"coverage_complete": limit is None, "intended_source_counts": expected,
                   "actual_source_counts": actual, "cache_used": cached is not None,
                   "canonical_hashes": int(frame.canonical_sha256.ne("").sum()),
                   "skip_reasons": frame.loc[frame.skip_reason.ne(""),
                                             "skip_reason"].value_counts().to_dict()})


def reference_table(input_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Require completed fresh full reference coverage before any training-window admission."""
    path = OUTPUT / "heldout_references/receipt.json"
    frame, receipt, hashes = verified_table(input_root, path, "mechanical_reference_recovery", True)
    if not receipt.get("coverage_complete"):
        raise ValueError("Incomplete cold held-out references")
    return frame, hashes


def quality(input_root: Path, workers: int = 4, limit: int | None = None,
            reuse_cache: bool = False) -> dict:
    """Reconstruct training quality through frozen pure helpers and apply new complete references."""
    output = OUTPUT / ("challenge_training_quality_profile" if limit else "challenge_training_quality")
    if (input_root / output).exists():
        raise FileExistsError(input_root / output)
    recipe = source_recipe()
    gate = profile_gate(input_root, "challenge_training_quality") if limit is None else {}
    refs, inputs = reference_table(input_root)
    inputs.update(gate)
    cached = verified_cache(input_root, RECOVERY / "challenge_training_quality", "training_quality") \
        if reuse_cache and limit is None else None
    if cached is not None:
        frame, cache_inputs = cached
        inputs.update(cache_inputs)
        for name in ("ssl_eligible", "supervised_eligible", "patient_identity_known"):
            frame[name] = parse_boolean(frame[name])
        frame["samples"] = pd.to_numeric(frame.samples)
    else:
        rows, row_inputs = verified_training_rows(input_root)
        inputs.update(row_inputs)
        selected = rows if limit is None else rows.groupby("source", sort=False).head(limit)
        checksums = {}
        for release in set(RELEASES.values()):
            path = Path("data/raw") / release / "SHA256SUMS.txt"
            checksums[release] = parse_checksums((input_root / path).read_text())
            inputs[stored(path)] = sha256_file(input_root / path)
        jobs = [(str(input_root), row,
                 {suffix: checksums[row["release"]][row["path"] + suffix] for suffix in (".hea", ".mat")})
                for row in selected.to_dict("records")]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frame = pd.DataFrame(collected(pool.map(assess_training_record, jobs, chunksize=16),
                                           len(jobs), "Cold Challenge training QC"))
        frame = supervised_window_duplicates(frame).drop(columns="worker_peak_rss_kib")
    frame = exclude_reference_overlaps(frame, set(refs.canonical_sha256) - {""},
                                       set(refs.full_signal_sha256) - {""})
    scope = "resource_profile" if limit else "training_quality"
    return finish(input_root, output, frame, inputs, recipe, scope,
                  {"cache_used": cached is not None,
                   "supervised_eligible": int(frame.supervised_eligible.sum()),
                   "heldout_overlap_excluded": int(frame.heldout_overlap_reasons.ne("").sum()),
                   "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
