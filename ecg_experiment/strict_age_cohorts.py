"""Derive strict-age records and a source/evidence/target matched control from frozen candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import ROOT
from .age_cohort_assembly import (
    OUTPUT as PARENT,
)
from .age_cohort_assembly import (
    cohort_overlap,
    evaluation_overlap,
    summarize,
    verify_unchanged,
    write_artifacts,
)
from .age_cohort_inputs import parse_boolean, read_references, relative_path, stored
from .age_priority_cohorts import select_cohort
from .files import sha256_file, write_json_atomic

OUTPUT = Path("outputs/cohort_strict_young_matched_20261028")
PROTOCOL = Path("docs/cohort-strict-young-matched-protocol-2026-10-28.md")
SOURCE_FILES = (
    "ecg_experiment/strict_age_cohorts.py", "scripts/data/derive_strict_age_cohorts.py",
    "ecg_experiment/age_cohort_assembly.py", "ecg_experiment/age_cohort_inputs.py",
    "ecg_experiment/age_priority_cohorts.py", "ecg_experiment/files.py", "ecg_experiment/paths.py",
)


def derived_tables(candidates: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require exact 18–25 membership without fallback and an equal-size evidence-first general control."""
    exact = candidates["age_interpretation"].eq("exact")
    strict = candidates[exact & candidates["age"].between(18, 25)].copy()
    if strict.empty:
        raise ValueError("No eligible exact-age 18–25 candidates")
    young = select_cohort(strict, len(strict), seed, prefer_young=False)
    strata = ["source", "label_evidence_rank", "target"]
    parts = []
    for key, group in strict.groupby(strata, sort=True):
        mask = pd.Series(True, index=candidates.index)
        for column, value in zip(strata, key, strict=True):
            mask &= candidates[column].eq(value)
        parts.append(select_cohort(candidates[mask], len(group), seed, prefer_young=False))
    general = select_cohort(pd.concat(parts, ignore_index=True), len(strict), seed, prefer_young=False)
    if not strict.groupby(strata).size().equals(general.groupby(strata).size()):
        raise ValueError("General control does not match source, evidence and target strata")
    if len(young) != len(general) or any(frame["target"].nunique() != 2 for frame in (young, general)):
        raise ValueError("Equal-size cohorts must each retain both original classes")
    return young, general


def load_parent(input_root: Path, parent: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Require the completed parent's exact admitted-candidate table and preserve all original evidence."""
    receipt_path = parent / "receipt.json"
    candidate_path = parent / "candidates.csv"
    receipt = json.loads((input_root / receipt_path).read_text())
    if not receipt.get("complete") or receipt.get("scope") != "new_supervised_age_label_cohorts":
        raise ValueError("Completed parent admitted candidates are required")
    expected = receipt["artifacts"][stored(candidate_path)]
    if sha256_file(input_root / candidate_path) != expected:
        raise ValueError("Parent candidate bytes changed")
    rows = pd.read_csv(input_root / candidate_path, dtype=str, keep_default_na=False)
    for column in ("quality_passed", "age_right_censored", "age_missing", "patient_identity_known"):
        rows[column] = parse_boolean(rows[column])
    for column in ("age", "age_lower_bound"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in ("target", "label_evidence_rank", "samples", "window_start"):
        rows[column] = pd.to_numeric(rows[column], errors="raise").astype(int)
    hashes = {stored(receipt_path): sha256_file(input_root / receipt_path), stored(candidate_path): expected}
    return rows, hashes


def derive(input_root: Path, output: Path = OUTPUT, seed: int = 42, parent: Path = PARENT) -> dict:
    """
    Publish strict-age and matched-size training cohorts after all parent and reference gates pass.

    Parameters
    ----------
    input_root : Path
        Shared checkout holding completed local cohort artifacts.
    output : Path
        New root-relative destination; existing directories are refused.
    seed : int
        Explicit deterministic ordering and general-control selection seed.
    parent : Path
        Completed parent admitted-cohort directory, relative to input_root.

    Returns
    -------
    dict
        Completion receipt, aggregate age/label/source summaries and exact parent provenance.
    """
    output = relative_path(output)
    parent = relative_path(parent)
    destination = input_root / output
    if destination.exists():
        raise FileExistsError(destination)
    recipe = {stored(Path(name)): sha256_file(ROOT / name) for name in (*SOURCE_FILES, str(PROTOCOL))}
    candidates, inputs = load_parent(input_root, parent)
    references, reference_inputs = read_references(input_root)
    inputs.update(reference_inputs)
    young, general = derived_tables(candidates, seed)
    overlaps = {name: evaluation_overlap(frame, references) for name, frame in
                (("strict_18_25", young), ("general_matched_size", general))}
    if any(value for group in overlaps.values() for value in group.values()):
        raise ValueError("Derived training cohort intersects held-out references")
    summary = {
        "strict_available": len(young), "strict_18_25": summarize(young),
        "general_matched_size": summarize(general),
        "overlap_between_training_cohorts": cohort_overlap(young, general),
        "evaluation_overlap": overlaps,
    }
    verify_unchanged(input_root, inputs)
    verify_unchanged(ROOT, recipe)
    tables = {"strict_18_25.csv": young, "general_matched_size.csv": general}
    outputs = write_artifacts(destination, tables, summary)
    verify_unchanged(input_root, inputs)
    verify_unchanged(ROOT, recipe)
    receipt = {
        "complete": True, "scope": "strict_young_matched_cohorts", "seed": seed,
        "records_per_cohort": len(young), "matched_strata": ["source", "label_evidence_rank", "target"],
        "parent_candidates": stored(parent / "candidates.csv"),
        "parent_receipt": stored(parent / "receipt.json"),
        "input_sha256": inputs, "source_sha256": {name: recipe[name] for name in SOURCE_FILES},
        "recipe_sha256": recipe, "protocol_path": stored(PROTOCOL),
        "protocol_sha256": recipe[stored(PROTOCOL)],
        "artifacts": {stored(output / name): digest for name, digest in outputs.items()}, "summary": summary,
        "waveforms_read": 0, "model_scores_read_or_computed": 0,
        "limitation": "Control matches size, source, evidence rank and target; patient mix and disease "
                      "severity can differ; age effects are not causal; "
                      "Challenge patient identity remains unknown",
    }
    write_json_atomic(destination / "receipt.json", receipt, sort_keys=True)
    return receipt
