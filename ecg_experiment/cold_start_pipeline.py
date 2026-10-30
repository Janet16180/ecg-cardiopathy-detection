"""Raw-only reproducible data pipeline with fresh administrative and cohort receipts."""

from __future__ import annotations

import json
from pathlib import Path

from . import ROOT
from .age_cohort_inputs import stored
from .cohort_reproduction import ensure_mappings, execute_stage, stages, verify_sources, verify_stage
from .cold_data_stages import OUTPUT
from .files import sha256_file, write_json_atomic
from .raw_role_bootstrap import OUTPUT as RAW_ROLES
from .raw_role_bootstrap import PROFILE as RAW_PROFILE

SOURCE_FILES = ("ecg_experiment/cold_start_pipeline.py", "scripts/data/reproduce_cold_start_cohorts.py")


def plan(input_root: Path, workers: int = 4, seed: int = 42, reuse_cache: bool = False) -> list[dict]:
    """Declare raw prerequisites, profiles and fresh reference/cohort stages without old scientific inputs."""
    old = stages(input_root, OUTPUT / "cohorts_25k", 25000, seed, workers)
    required = {"ptb_labels_full", "ptb_labels_limited", "ptb_profile", "ptb_quality",
                "challenge_profile", "challenge_split"}
    result = [stage for stage in old if stage["name"] in required]
    shared = ["--input-root", str(input_root), "--workers", str(workers)]
    raw = "scripts.data.bootstrap_raw_roles"
    result.extend([
        {"name": "fresh_raw_roles_profile", "kind": "receipt", "module": raw,
         "args": shared + ["--profile-limit", "64", "--output", str(RAW_PROFILE)],
         "receipt": str(RAW_PROFILE / "receipt.json"), "scope": "resource_profile"},
        {"name": "fresh_raw_roles_full", "kind": "receipt", "module": raw, "args": shared,
         "receipt": str(RAW_ROLES / "receipt.json"), "scope": "administrative_roles_raw_quality"},
    ])
    for stage, directory, scope in (("references", "heldout_references", "mechanical_reference_recovery"),
                                     ("quality", "challenge_training_quality", "training_quality")):
        result.append({
            "name": "fresh_" + stage + "_profile", "kind": "receipt",
            "module": "scripts.data.bootstrap_cold_data_stages",
            "args": [stage, *shared, "--limit", "16"], "scope": "resource_profile",
            "receipt": str(OUTPUT / (directory + "_profile") / "receipt.json"),
        })
        result.append({
            "name": "fresh_" + stage + "_full", "kind": "receipt",
            "module": "scripts.data.bootstrap_cold_data_stages",
            "args": [stage, *shared, *(["--reuse-verified-caches"] if reuse_cache else [])], "scope": scope,
            "receipt": str(OUTPUT / directory / "receipt.json"),
        })
    result.extend([
        {"name": "fresh_25k_cohorts", "kind": "receipt", "module": "scripts.data.assemble_cold_cohorts",
         "args": ["cohorts", "--input-root", str(input_root), "--seed", str(seed)],
         "receipt": str(OUTPUT / "cohorts_25k/receipt.json"), "scope": "new_supervised_age_label_cohorts",
         "seed": seed, "size": 25000},
        {"name": "fresh_strict_matched", "kind": "receipt", "module": "scripts.data.assemble_cold_cohorts",
         "args": ["strict", "--input-root", str(input_root), "--seed", str(seed)],
         "receipt": str(OUTPUT / "strict_matched/receipt.json"), "scope": "strict_young_matched_cohorts",
         "seed": seed, "parent": str(OUTPUT / "cohorts_25k/receipt.json")},
    ])
    return result


def reproduce(input_root: Path, workers: int = 4, seed: int = 42, reuse_cache: bool = False) -> dict:
    """
    Reproduce the data pipeline from raw public releases without historical scientific intermediates.

    Parameters
    ----------
    input_root : Path
        Checkout containing local raw public releases and their public manifests.
    workers : int
        One through four CPU processes, each with one numerical-library thread.
    seed : int
        Fixed record-selection seed. Original PTB and split seeds remain frozen.
    reuse_cache : bool
        Optionally reuse verified raw-only window/QC caches. False exercises cold raw decoding.

    Returns
    -------
    dict
        New completion receipt with every honest executed/reused stage and immutable artifact bindings.
    """
    if not 1 <= workers <= 4:
        raise ValueError("Use one through four CPU workers")
    recipe = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    maps = ensure_mappings(input_root)
    completed = []
    for stage in plan(input_root, workers, seed, reuse_cache):
        existing = (input_root / stage["receipt"]).exists()
        log = "" if existing else execute_stage(input_root, stage)
        digest = verify_stage(input_root, stage)
        completed.append({"stage": stage["name"], "receipt": stage["receipt"], "receipt_sha256": digest,
                          "status": "verified_existing" if existing else "executed", "log": log})
    verify_sources(input_root, recipe)
    receipt = {
        "complete": True, "scope": "cold_start_cohort_data_pipeline", "source_sha256": recipe,
        "stages": completed, "mapping_sha256": maps, "raw_input_only_prerequisites": True,
        "optional_cache_reuse_requested": reuse_cache,
        "seeds": {"ptb_label_sampling": 42, "challenge_original_split": 20260929,
                  "sph_patient_roles": 20261018, "cohort_selection": seed, "python_hash_seed": 0},
        "cohort_output": stored(OUTPUT / "cohorts_25k"), "strict_output": stored(OUTPUT / "strict_matched"),
        "scientific_model_or_prediction_prerequisites": [], "model_scores_read_or_computed": 0,
    }
    path = input_root / OUTPUT / "pipeline_receipt.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["source_sha256"] != recipe or previous["seeds"] != receipt["seeds"]:
            raise ValueError("Existing cold pipeline has different code or selection seed")
        return previous
    write_json_atomic(path, receipt, sort_keys=True)
    return receipt
