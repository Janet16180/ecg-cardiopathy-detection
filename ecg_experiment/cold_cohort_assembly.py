"""Successor cohort assembly using fresh raw administrative roles and new data-stage receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import ROOT
from .age_cohort_assembly import (
    admit_candidates,
    cohort_overlap,
    evaluation_overlap,
    summarize,
    validate_policy,
    verify_unchanged,
    write_artifacts,
)
from .age_cohort_inputs import (
    POLICY,
    PTB_EVIDENCE_COLUMNS,
    PTB_LABELS,
    PTB_LABELS_MD5,
    PTB_QUALITY,
    PTB_RAW,
    SPLIT_SHA256,
    canonical_ptb_identity,
    normalize_age,
    parse_boolean,
    ptb_initial_exclusions,
    ptb_label_evidence,
    read_selected_csv,
    stored,
    verified_table,
)
from .age_priority_cohorts import select_cohort
from .cold_data_stages import OUTPUT, PROTOCOL, raw_roles, reference_table
from .files import sha256_file, write_json_atomic
from .raw_role_bootstrap import OUTPUT as ROLE_OUTPUT
from .strict_age_cohorts import derived_tables, load_parent

SOURCE_FILES = (
    "ecg_experiment/cold_cohort_assembly.py", "scripts/data/assemble_cold_cohorts.py",
    "ecg_experiment/cold_data_stages.py", "ecg_experiment/age_cohort_assembly.py",
    "ecg_experiment/age_cohort_inputs.py", "ecg_experiment/age_priority_cohorts.py",
    "ecg_experiment/strict_age_cohorts.py", "ecg_experiment/ecg_quality.py",
)


def recipe_snapshot() -> dict[str, str]:
    """Bind the immutable successor implementation, label-evidence policy and prospective protocol."""
    paths = (*SOURCE_FILES, str(POLICY), str(PROTOCOL))
    return {stored(Path(name)): sha256_file(ROOT / name) for name in paths}


def fresh_ptb(input_root: Path, policy: dict) -> tuple[pd.DataFrame, dict[str, str], dict, pd.DataFrame]:
    """Join frozen strict targets and quality with newly raw-derived fitting roles and superclass targets."""
    _, hashes = raw_roles(input_root)
    roles = pd.read_csv(input_root / ROLE_OUTPUT / "ptb_rows.csv", dtype=str, keep_default_na=False)
    roles = canonical_ptb_identity(roles)
    fit = roles[pd.to_numeric(roles.strat_fold).between(1, 7)].rename(
        columns={"target": "fresh_superclass_target"},
    )
    allowed = set(fit.ecg_id)
    quality, receipt, quality_hashes = verified_table(input_root, PTB_QUALITY, "training_quality", False)
    verify_unchanged(input_root, receipt["input_sha256"])
    hashes.update(receipt["input_sha256"] | quality_hashes)
    label_path = input_root / PTB_LABELS
    if hashlib.md5(label_path.read_bytes()).hexdigest() != PTB_LABELS_MD5:
        raise ValueError("Original strict PTB manifest bytes changed")
    hashes[stored(PTB_LABELS)] = sha256_file(label_path)
    labels = canonical_ptb_identity(read_selected_csv(
        label_path, "ecg_id", allowed, ("ecg_id", "patient_id", "filename_hr", "target"),
    ))
    metadata = canonical_ptb_identity(read_selected_csv(
        input_root / PTB_RAW, "ecg_id", set(labels.ecg_id), PTB_EVIDENCE_COLUMNS,
    ))
    rows = labels.merge(metadata, on=["ecg_id", "patient_id", "filename_hr"], validate="one_to_one")
    rows = rows.merge(fit, on=["ecg_id", "patient_id", "filename_hr", "strat_fold"], validate="one_to_one")
    same_target = pd.to_numeric(rows.target).eq(pd.to_numeric(rows.fresh_superclass_target))
    if len(rows) != len(labels) or not same_target.all():
        raise ValueError("Fresh fitting roles disagree with original strict targets or identities")
    rows["record_id"] = "ptbxl:" + rows.ecg_id
    rows = rows.merge(quality, on="record_id", suffixes=("", "_quality"), validate="one_to_one")
    if (not ("ptbxl:" + rows.patient_id).eq(rows.patient_id_quality).all() or
            not rows.strat_fold.eq(rows.strat_fold_quality).all() or
            not rows.path.eq("data/raw/ptb-xl/1.0.3/" + rows.filename_hr).all()):
        raise ValueError("Fresh PTB quality identity alignment failed")
    administrative = pd.read_csv(input_root / ROLE_OUTPUT / "ptb_administrative.csv", dtype=str)
    canonical = administrative.set_index("record_id").signal_sha256
    if not rows.record_id.map(canonical).eq(rows.signal_sha256).all():
        raise ValueError("New raw canonical PTB hashes disagree with training quality")
    rows = ptb_label_evidence(rows, policy["ptbxl"])
    rows = rows.join(normalize_age(rows.pop("age"), True))
    rows = rows.assign(source="ptbxl", family="ptbxl", split="train", record=rows.ecg_id,
                       patient_id="ptbxl:" + rows.patient_id, patient_identity_known=True,
                       quality_passed=rows.exclusion_reasons.eq(""), full_signal_sha256="",
                       waveform_path=rows.path, backend="ptbxl_wfdb", window_start=0, samples=5000,
                       label_definition="diagnostic_abnormality", target=pd.to_numeric(rows.target))
    ledger = ptb_initial_exclusions(quality, {"ptbxl:" + key for key in allowed}, set(rows.record_id))
    return rows, hashes, {"common_strict_target_rows": len(rows), "target_disagreements": 0}, ledger


def fresh_challenge(input_root: Path, policy: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Admit only completed fresh-reference training quality with strict supervised-window eligibility."""
    rows, receipt, hashes = verified_table(input_root, OUTPUT / "challenge_training_quality/receipt.json",
                                           "training_quality", True)
    verify_unchanged(input_root, receipt["input_sha256"])
    hashes.update(receipt["input_sha256"])
    if len(rows) != receipt["records"] or not rows.split.eq("train").all():
        raise ValueError("Invalid fresh Challenge training coverage")
    allowed = parse_boolean(rows.supervised_eligible)
    if not rows.loc[allowed, "samples"].eq("5000").all():
        raise ValueError("Cropped record labels cannot enter supervised cold cohorts")
    rows = rows.assign(target=pd.to_numeric(rows.supervised_target, errors="coerce"),
                       quality_passed=rows.quality_status.eq("passed"), supervised_admitted=allowed)
    rows["signal_sha256"] = rows.signal_window_sha256
    raw_age = rows.pop("age_raw")
    rows = rows.drop(columns="age").join(normalize_age(raw_age, False))
    evidence = rows.source.map(policy["sources"])
    rows["label_evidence_rank"] = evidence.map(lambda item: item["rank"])
    rows["label_evidence"] = evidence.map(lambda item: item["evidence"])
    rows["label_evidence_reference"] = evidence.map(lambda item: item["reference"])
    rows["label_evidence_scope"] = "source_process"
    rows["patient_identity_known"] = parse_boolean(rows.patient_identity_known)
    rows["label_definition"] = "challenge_primary_MI_STTC_CD_HYP_vs_sinus_only"
    return rows, hashes


def publish(input_root: Path, output: Path, tables: dict[str, pd.DataFrame], summary: dict,
            inputs: dict[str, str], recipe: dict[str, str], seed: int, scope: str, extra: dict) -> dict:
    """Publish honest fresh-input cohort provenance without modifying completed historical outputs."""
    verify_unchanged(input_root, inputs)
    verify_unchanged(ROOT, recipe)
    artifacts = write_artifacts(input_root / output, tables, summary)
    verify_unchanged(input_root, inputs)
    verify_unchanged(ROOT, recipe)
    receipt = {
        "complete": True, "scope": scope, "seed": seed, "original_split_sha256": SPLIT_SHA256,
        "role_provenance": "fresh_raw_administrative_bootstrap", "input_sha256": inputs,
        "source_sha256": {name: recipe[name] for name in SOURCE_FILES}, "recipe_sha256": recipe,
        "artifacts": {stored(output / name): digest for name, digest in artifacts.items()},
        "summary": summary, "waveforms_read": 0, "model_scores_read_or_computed": 0,
        "protocol_path": stored(PROTOCOL), "protocol_sha256": recipe[stored(PROTOCOL)], **extra,
    }
    write_json_atomic(input_root / output / "receipt.json", receipt, sort_keys=True)
    return receipt


def assemble(input_root: Path, seed: int = 42, size: int = 25000) -> dict:
    """Construct both 25k selections using fresh raw-role and reference provenance."""
    output = OUTPUT / "cohorts_25k"
    if (input_root / output).exists():
        raise FileExistsError(input_root / output)
    recipe = recipe_snapshot()
    policy = json.loads((ROOT / POLICY).read_text())
    validate_policy(policy)
    refs, inputs = reference_table(input_root)
    ptb, ptb_inputs, counts, initial = fresh_ptb(input_root, policy)
    challenge, challenge_inputs = fresh_challenge(input_root, policy)
    inputs.update(ptb_inputs | challenge_inputs)
    candidates, ledger = admit_candidates(ptb, challenge, refs)
    younger = select_cohort(candidates, size, seed, True)
    general = select_cohort(candidates, size, seed, False)
    if any(frame.target.nunique() != 2 for frame in (younger, general)):
        raise ValueError("Both cohorts require both original classes")
    summary = {"candidates": summarize(candidates), "younger": summarize(younger),
               "all_age": summarize(general), "ptb_target_integrity": counts,
               "evaluation_overlap_after_admission": evaluation_overlap(candidates, refs),
               "overlap_between_training_cohorts": cohort_overlap(younger, general)}
    tables = {"candidates.csv": candidates, "younger.csv": younger, "all_age.csv": general,
              "exclusions.csv": pd.concat([initial, ledger], ignore_index=True)}
    return publish(input_root, output, tables, summary, inputs, recipe, seed,
                   "new_supervised_age_label_cohorts",
                   {"size_ceiling": size, "policy_version": policy["policy_version"]})


def strict(input_root: Path, seed: int = 42) -> dict:
    """Create the same strict-age/source-evidence-target matched comparison under fresh provenance."""
    output = OUTPUT / "strict_matched"
    if (input_root / output).exists():
        raise FileExistsError(input_root / output)
    recipe = recipe_snapshot()
    candidates, inputs = load_parent(input_root, OUTPUT / "cohorts_25k")
    refs, reference_inputs = reference_table(input_root)
    inputs.update(reference_inputs)
    younger, general = derived_tables(candidates, seed)
    overlaps = {key: evaluation_overlap(frame, refs) for key, frame in
                (("strict_18_25", younger), ("general_matched_size", general))}
    if any(value for group in overlaps.values() for value in group.values()):
        raise ValueError("Cold strict cohort intersects a held-out reference")
    summary = {"strict_available": len(younger), "strict_18_25": summarize(younger),
               "general_matched_size": summarize(general), "evaluation_overlap": overlaps,
               "overlap_between_training_cohorts": cohort_overlap(younger, general)}
    return publish(input_root, output, {"strict_18_25.csv": younger, "general_matched_size.csv": general},
                   summary, inputs, recipe, seed, "strict_young_matched_cohorts",
                   {"records_per_cohort": len(younger),
                    "matched_strata": ["source", "label_evidence_rank", "target"],
                    "parent_receipt": stored(OUTPUT / "cohorts_25k/receipt.json")})
