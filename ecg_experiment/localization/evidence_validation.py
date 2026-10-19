"""Validate evidence structure and signal-bound semantics without judging clinical truth."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from ecg_experiment import ROOT

SCHEMA = ROOT / "schemas/ecg-evidence-annotation-v1.schema.json"
UNITS = frozenset({"mV", "uV", "ms", "s", "bpm", "mm", "degree", "mV*s", "ratio"})
EXPERT_KINDS = frozenset({"independent_expert", "expert_adjudication"})


@dataclass(frozen=True)
class SignalBinding:
    """Trusted signal identity supplied separately from an imported annotation."""

    record_ref: str
    signal_sha256: str
    sampling_rate_hz: float
    n_samples: int
    lead_order: tuple[str, ...]
    patient_ref: str | None = None


@dataclass(frozen=True)
class ReviewerBinding:
    """Externally established reviewer attestation, never inferred from annotation flags.

    The caller is responsible for establishing the actual reviewer identity, expertise,
    timestamp and policy attestation. This object is not evidence that review occurred.
    """

    annotator_ref: str
    record_ref: str
    signal_sha256: str
    annotation_timestamp_utc: str
    policy_ref: str
    expert_attested: bool
    independent_of_model: bool
    independent_of_support_bank: bool
    model_output_visible: bool


@dataclass(frozen=True)
class BeatBinding:
    """Frozen original-sample beat anchors and their detector or reviewer revision."""

    sample_positions: tuple[int, ...]
    source_revision: str
    signal_sha256: str
    sha256: str


def beat_digest(positions: tuple[int, ...], revision: str, signal_sha256: str) -> str:
    """Hash canonical JSON of original sample anchors, revision and bound signal."""
    value = {"sample_positions": positions, "source_revision": revision, "signal_sha256": signal_sha256}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def date_time_checker() -> FormatChecker:
    """Check RFC3339 dates even when optional jsonschema format packages are absent."""
    checker = FormatChecker()

    @checker.checks("date-time", raises=ValueError)
    def valid_date_time(value: Any) -> bool:
        if not isinstance(value, str):
            return True
        pattern = r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})"
        if re.fullmatch(pattern, value) is None:
            return False
        if value[-1].upper() != "Z" and (int(value[-5:-3]) >= 24 or int(value[-2:]) >= 60):
            return False
        datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
        return True

    return checker


def finite_numbers(value: Any) -> None:
    """Reject NaN and infinity recursively, including values ignored by comparisons."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite annotation number")
    if isinstance(value, dict):
        for item in value.values():
            finite_numbers(item)
    if isinstance(value, list):
        for item in value:
            finite_numbers(item)


def check_signal(annotation: dict, signal: SignalBinding) -> None:
    """Require exact external signal binding rather than trusting imported metadata."""
    if not math.isfinite(signal.sampling_rate_hz) or signal.sampling_rate_hz <= 0:
        raise ValueError("Invalid trusted sampling rate")
    expected = {
        "record_ref": signal.record_ref,
        "signal_sha256": signal.signal_sha256,
        "sampling_rate_hz": signal.sampling_rate_hz,
        "n_samples": signal.n_samples,
        "lead_order": list(signal.lead_order),
    }
    if signal.patient_ref is not None:
        expected["patient_ref"] = signal.patient_ref
    if any(annotation.get(key) != value for key, value in expected.items()):
        raise ValueError("Annotation differs from trusted signal binding")


def check_beats(beats: BeatBinding | None, signal: SignalBinding) -> BeatBinding:
    """Require an exact signal-bound frozen beat list before accepting ordinals."""
    if beats is None or not beats.source_revision or beats.signal_sha256 != signal.signal_sha256:
        raise ValueError("Ordinal coordinates require a frozen signal-bound beat list")
    positions = beats.sample_positions
    if not positions or any(
        type(value) is not int or not 0 <= value < signal.n_samples for value in positions
    ):
        raise ValueError("Beat anchors must be integer samples inside the record")
    if any(left >= right for left, right in zip(positions[:-1], positions[1:], strict=True)):
        raise ValueError("Beat anchors must be strictly increasing")
    if beats.sha256 != beat_digest(positions, beats.source_revision, beats.signal_sha256):
        raise ValueError("Frozen beat list hash differs")
    return beats


def check_coordinates(coordinates: dict, signal: SignalBinding, beats: BeatBinding | None) -> None:
    """Check original coordinates without converting or replacing their representation."""
    kind = coordinates["kind"]
    if kind in {"seconds", "samples"}:
        start, stop = coordinates["start"], coordinates["stop"]
        limit = signal.n_samples / signal.sampling_rate_hz if kind == "seconds" else signal.n_samples
        if not 0 <= start < stop <= limit:
            raise ValueError("Coordinates violate half-open record bounds")
        if kind == "samples" and (type(start) is not int or type(stop) is not int):
            raise ValueError("Sample coordinates must be integers")
    if kind == "beat_ordinals":
        bound = check_beats(beats, signal)
        first, last = coordinates["first"], coordinates["last"]
        if (
            type(first) is not int
            or type(last) is not int
            or not 1 <= first <= last <= len(bound.sample_positions)
        ):
            raise ValueError("Ordinal coordinates outside frozen beat list")


def check_measurements(evidence_set: dict, signal: SignalBinding, units: frozenset[str]) -> None:
    """Require measurements to reference their own set and compatible lead combinations."""
    regions = {region["region_id"]: region for region in evidence_set["regions"]}
    for value in evidence_set.get("measurements", []):
        if not value["unit"].strip() or value["unit"] not in units or not value["method"].strip():
            raise ValueError("Measurement requires a declared unit and nonblank method")
        references = value.get("region_ids", list(regions))
        if not references or any(key not in regions for key in references):
            raise ValueError("Measurement region references must resolve inside the same evidence set")
        covered = set().union(*(set(regions[key]["leads"]) for key in references))
        if not set(value["leads"]) <= covered or not set(value["leads"]) <= set(signal.lead_order):
            raise ValueError("Measurement leads differ from its referenced regions or signal")


def check_provenance_flags(provenance: dict) -> None:
    """Reject incompatible source-kind, visibility and independence claims."""
    if provenance["model_output_visible"] and provenance["independent_of_model"]:
        raise ValueError("Visible model output contradicts model-independent provenance")
    if provenance["kind"] == "model_prediction" and provenance["independent_of_model"]:
        raise ValueError("Model prediction cannot claim independence from the model")


def check_reviewer(annotation: dict, signal: SignalBinding, reviewer: ReviewerBinding | None) -> None:
    """Validate provenance claims against externally supplied reviewer and policy binding."""
    provenance = annotation["provenance"]
    expert = provenance["kind"] in EXPERT_KINDS
    check_provenance_flags(provenance)
    actual = annotation["review_status"] in {"reviewed", "adjudicated"} or expert
    if not actual:
        return
    if reviewer is None or not reviewer.policy_ref.strip() or not reviewer.annotator_ref.strip():
        raise ValueError("Reviewed or expert annotations require an external reviewer/policy binding")
    expected = {
        "annotator_ref": reviewer.annotator_ref,
        "annotation_timestamp_utc": reviewer.annotation_timestamp_utc,
        "independent_of_model": reviewer.independent_of_model,
        "independent_of_support_bank": reviewer.independent_of_support_bank,
        "model_output_visible": reviewer.model_output_visible,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Reviewer attestation differs from annotation provenance")
    reviewed_time = datetime.fromisoformat(reviewer.annotation_timestamp_utc.upper().replace("Z", "+00:00"))
    if reviewed_time.utcoffset().total_seconds() != 0:
        raise ValueError("Review timestamp must be recorded in UTC")
    if reviewer.record_ref != signal.record_ref or reviewer.signal_sha256 != signal.signal_sha256:
        raise ValueError("Reviewer attestation is bound to another signal")
    if expert and not reviewer.expert_attested:
        raise ValueError("Expert provenance requires externally attested expertise")
    if provenance["kind"] == "independent_expert" and not (
        reviewer.independent_of_model
        and reviewer.independent_of_support_bank
        and not reviewer.model_output_visible
    ):
        raise ValueError("Independent blinded expert provenance contradicts the review conditions")
    if provenance["kind"] == "expert_adjudication" and annotation["review_status"] != "adjudicated":
        raise ValueError("Expert adjudication requires adjudicated review status")


def validate_evidence(
    annotation: dict,
    signal: SignalBinding,
    reviewer: ReviewerBinding | None = None,
    beats: BeatBinding | None = None,
    *,
    strict_blinded_expert: bool = False,
    allowed_units: frozenset[str] = UNITS,
    schema_path: Path = SCHEMA,
) -> dict:
    """Validate structure and technical semantics, preserving alternatives and unknowns.

    Parameters
    ----------
    annotation : dict
        One imported finding-specific annotation. Pending review forms are a different contract.
    signal : SignalBinding
        Trusted local signal binding, independently obtained from the canonical signal bytes.
    reviewer : ReviewerBinding or None
        Trusted caller-supplied reviewer/policy attestation, required for review and expert claims.
    beats : BeatBinding or None
        Frozen original sample anchors, required for ordinal coordinates.
    strict_blinded_expert : bool
        Require completed independent expert provenance and direct support in every alternative set.
    allowed_units : frozenset of str
        Explicit vocabulary for physical measurement units; no implicit unit conversion occurs.
    schema_path : Path
        Unchanged draft evidence schema.

    Returns
    -------
    dict
        Technical validation counts and context-only warnings, never a clinical truth judgment.
    """
    finite_numbers(annotation)
    Draft202012Validator(json.loads(schema_path.read_text()), format_checker=date_time_checker()).validate(
        annotation
    )
    check_signal(annotation, signal)
    check_reviewer(annotation, signal, reviewer)
    if strict_blinded_expert and (
        annotation["provenance"]["kind"] not in EXPERT_KINDS
        or annotation["review_status"] not in {"reviewed", "adjudicated"}
        or not annotation["provenance"]["independent_of_model"]
        or not annotation["provenance"]["independent_of_support_bank"]
        or annotation["provenance"]["model_output_visible"]
    ):
        raise ValueError("Strict evaluation requires completed independent blinded expert provenance")
    sets, region_ids, set_ids, warnings = annotation["acceptable_evidence_sets"], set(), set(), []
    for evidence_set in sets:
        if evidence_set["set_id"] in set_ids:
            raise ValueError("Duplicate evidence set ID")
        set_ids.add(evidence_set["set_id"])
        for region in evidence_set["regions"]:
            if region["region_id"] in region_ids:
                raise ValueError("Duplicate region ID")
            region_ids.add(region["region_id"])
            if not set(region["leads"]) <= set(signal.lead_order):
                raise ValueError("Region lead absent from bound signal")
            check_coordinates(region["coordinates"], signal, beats)
        if not any(region["role"] == "support" for region in evidence_set["regions"]):
            warnings.append(
                f"Set {evidence_set['set_id']} has no direct support; sufficiency needs clinical review"
            )
            if strict_blinded_expert:
                raise ValueError("Strict evaluation rejects sets without direct support")
        check_measurements(evidence_set, signal, allowed_units)
    if (
        annotation["finding"]["state"] == "present"
        and not sets
        and not annotation.get("unlocalized_reason", "").strip()
    ):
        raise ValueError("Present finding requires evidence or an explicit nonblank unlocalized reason")
    return {
        "structurally_and_semantically_valid": True,
        "evidence_sets": len(sets),
        "regions": len(region_ids),
        "warnings": warnings,
        "clinical_truth_certified": False,
        "coordinates_converted": False,
    }
