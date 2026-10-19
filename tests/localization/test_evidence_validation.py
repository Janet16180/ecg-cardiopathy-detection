"""Synthetic semantic failures, without source ECGs or clinical truth claims."""

from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import ValidationError

from ecg_experiment.localization.evidence_validation import (
    BeatBinding,
    ReviewerBinding,
    SignalBinding,
    beat_digest,
    validate_evidence,
)
from ecg_experiment.localization.review_forms import pending_form, validate_pending_form


@pytest.fixture
def signal():
    return SignalBinding("invented-case", "a" * 64, 500, 5000, ("I", "II", "V1", "V6"), "invented-patient")


@pytest.fixture
def annotation(signal):
    return {
        "schema_version": "ecg-evidence-1.0-draft",
        "record_ref": signal.record_ref,
        "patient_ref": signal.patient_ref,
        "signal_sha256": signal.signal_sha256,
        "sampling_rate_hz": 500,
        "n_samples": 5000,
        "lead_order": list(signal.lead_order),
        "finding": {"name": "Invented finding", "state": "present", "source_label_retained": False},
        "provenance": {
            "kind": "synthetic_example",
            "source": "Invented synthetic fixture",
            "source_revision": "1",
            "independent_of_model": False,
            "independent_of_support_bank": False,
            "model_output_visible": False,
        },
        "acceptable_evidence_sets": [
            {
                "set_id": "first",
                "rationale": "Invented jointly considered pair",
                "regions": [
                    {
                        "region_id": "support",
                        "leads": ["V1"],
                        "role": "support",
                        "coordinates": {"kind": "seconds", "start": 1.0, "stop": 1.3},
                    },
                    {
                        "region_id": "comparison",
                        "leads": ["V6"],
                        "role": "comparison",
                        "coordinates": {"kind": "samples", "start": 500, "stop": 650},
                    },
                ],
                "measurements": [
                    {
                        "name": "Invented duration",
                        "value": 100,
                        "unit": "ms",
                        "leads": ["V1", "V6"],
                        "method": "Invented manual method",
                        "region_ids": ["support", "comparison"],
                    }
                ],
            }
        ],
        "exhaustive_abnormality_annotation": False,
        "unmarked_samples": "unknown",
        "review_status": "draft",
    }


def expert(annotation, signal):
    reviewer = ReviewerBinding(
        "invented-reviewer",
        signal.record_ref,
        signal.signal_sha256,
        "2026-10-19T12:00:00Z",
        "synthetic-test-policy",
        True,
        True,
        True,
        False,
    )
    provenance = annotation["provenance"]
    provenance.update(
        kind="independent_expert",
        annotator_ref=reviewer.annotator_ref,
        annotation_timestamp_utc=reviewer.annotation_timestamp_utc,
        independent_of_model=True,
        independent_of_support_bank=True,
    )
    annotation["review_status"] = "reviewed"
    return reviewer


def test_joint_regions_and_alternatives_preserved(annotation, signal):
    alternative = deepcopy(annotation["acceptable_evidence_sets"][0])
    alternative["set_id"] = "second"
    for region in alternative["regions"]:
        region["region_id"] += "-second"
    alternative["measurements"][0]["region_ids"] = ["support-second", "comparison-second"]
    annotation["acceptable_evidence_sets"].append(alternative)
    original = deepcopy(annotation)
    result = validate_evidence(annotation, signal)
    assert result["evidence_sets"] == 2
    assert result["regions"] == 4
    assert result["clinical_truth_certified"] is False
    assert annotation == original


@pytest.mark.parametrize("state", ["absent", "uncertain"])
def test_record_judgments_do_not_create_negative_samples(annotation, signal, state):
    annotation["finding"]["state"] = state
    annotation["acceptable_evidence_sets"] = []
    assert validate_evidence(annotation, signal)["regions"] == 0
    assert annotation["unmarked_samples"] == "unknown"


def test_present_unlocalized_reason(annotation, signal):
    annotation["acceptable_evidence_sets"] = []
    with pytest.raises(ValidationError):
        validate_evidence(annotation, signal)
    annotation["unlocalized_reason"] = "Not localized in this invented fixture"
    assert validate_evidence(annotation, signal)["regions"] == 0
    annotation["unlocalized_reason"] = "  "
    with pytest.raises(ValueError, match="nonblank"):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("signal_sha256", "b" * 64),
        ("record_ref", "wrong"),
        ("patient_ref", "wrong"),
        ("n_samples", 4999),
        ("sampling_rate_hz", 250),
        ("lead_order", ["I", "V1"]),
    ],
)
def test_binding_failures(annotation, signal, field, value):
    annotation[field] = value
    with pytest.raises(ValueError, match="signal binding"):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numbers(annotation, signal, value):
    annotation["acceptable_evidence_sets"][0]["measurements"][0]["value"] = value
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"kind": "seconds", "start": 1, "stop": 1},
        {"kind": "seconds", "start": 2, "stop": 1},
        {"kind": "seconds", "start": 0, "stop": 10.001},
        {"kind": "seconds", "start": -0.1, "stop": 1},
        {"kind": "samples", "start": 0, "stop": 5001},
        {"kind": "samples", "start": 4999, "stop": 4999},
        {"kind": "samples", "start": 0.5, "stop": 100},
        {"kind": "samples", "start": True, "stop": 100},
    ],
)
def test_coordinate_failures(annotation, signal, coordinates):
    annotation["acceptable_evidence_sets"][0]["regions"][0]["coordinates"] = coordinates
    with pytest.raises((ValueError, ValidationError)):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"kind": "seconds", "start": 0, "stop": 10},
        {"kind": "samples", "start": 0, "stop": 5000},
        {"kind": "whole_record"},
    ],
)
def test_coordinate_boundaries(annotation, signal, coordinates):
    annotation["acceptable_evidence_sets"][0]["regions"][0]["coordinates"] = coordinates
    validate_evidence(annotation, signal)


def test_ordinal_coordinates_require_exact_bound_list(annotation, signal):
    annotation["acceptable_evidence_sets"][0]["regions"][0]["coordinates"] = {
        "kind": "beat_ordinals",
        "first": 1,
        "last": 3,
    }
    positions = (400, 900, 1400)
    beats = BeatBinding(
        positions,
        "invented-detector-v1",
        signal.signal_sha256,
        beat_digest(positions, "invented-detector-v1", signal.signal_sha256),
    )
    with pytest.raises(ValueError, match="beat list"):
        validate_evidence(annotation, signal)
    validate_evidence(annotation, signal, beats=beats)
    for wrong in [
        replace(beats, sha256="b" * 64),
        replace(beats, signal_sha256="b" * 64),
        replace(beats, sample_positions=(400, 400, 1400)),
        replace(beats, sample_positions=(400, 5000)),
        replace(beats, source_revision=""),
    ]:
        with pytest.raises(ValueError, match="beat|Beat|Measurement|Reviewer|reviewer|expert|Expert"):
            validate_evidence(annotation, signal, beats=wrong)
    annotation["acceptable_evidence_sets"][0]["regions"][0]["coordinates"]["last"] = 4
    with pytest.raises(ValueError, match="outside"):
        validate_evidence(annotation, signal, beats=beats)


def test_duplicate_ids(annotation, signal):
    duplicate = deepcopy(annotation["acceptable_evidence_sets"][0])
    annotation["acceptable_evidence_sets"].append(duplicate)
    with pytest.raises(ValueError, match="Duplicate evidence"):
        validate_evidence(annotation, signal)
    duplicate["set_id"] = "different"
    with pytest.raises(ValueError, match="Duplicate region"):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region_ids", ["other-set-region"]),
        ("region_ids", []),
        ("leads", ["I"]),
        ("unit", "pixels"),
        ("method", " "),
    ],
)
def test_measurement_semantics(annotation, signal, field, value):
    annotation["acceptable_evidence_sets"][0]["measurements"][0][field] = value
    with pytest.raises(ValueError, match="beat|Beat|Measurement|Reviewer|reviewer|expert|Expert"):
        validate_evidence(annotation, signal)


def test_region_lead_membership(annotation, signal):
    annotation["acceptable_evidence_sets"][0]["regions"][0]["leads"] = ["III"]
    with pytest.raises(ValueError, match="lead absent"):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize("kind", ["rule_generated_silver", "synthetic_example", "source_record_label"])
def test_independent_nonexpert_provenance_is_not_promoted(annotation, signal, kind):
    annotation["provenance"].update(kind=kind, independent_of_model=True, independent_of_support_bank=True)
    validate_evidence(annotation, signal)
    with pytest.raises(ValueError, match="Strict evaluation"):
        validate_evidence(annotation, signal, strict_blinded_expert=True)


def test_model_prediction_cannot_claim_independence_from_model(annotation, signal):
    annotation["provenance"].update(kind="model_prediction", independent_of_model=True)
    with pytest.raises(ValueError, match="Model prediction"):
        validate_evidence(annotation, signal)


def test_expert_requires_trusted_binding(annotation, signal):
    reviewer = expert(annotation, signal)
    with pytest.raises(ValueError, match="external reviewer"):
        validate_evidence(annotation, signal)
    validate_evidence(annotation, signal, reviewer, strict_blinded_expert=True)
    for wrong in [
        replace(reviewer, signal_sha256="b" * 64),
        replace(reviewer, expert_attested=False),
        replace(reviewer, policy_ref=""),
        replace(reviewer, annotator_ref="different"),
        replace(reviewer, model_output_visible=True),
    ]:
        with pytest.raises(ValueError, match="beat|Beat|Measurement|Reviewer|reviewer|expert|Expert"):
            validate_evidence(annotation, signal, wrong)


def test_visible_model_cannot_be_independent_blinded_expert(annotation, signal):
    reviewer = expert(annotation, signal)
    annotation["provenance"]["model_output_visible"] = True
    with pytest.raises(ValueError, match="contradicts"):
        validate_evidence(annotation, signal, replace(reviewer, model_output_visible=True))


def test_invalid_datetime_rejected(annotation, signal):
    reviewer = expert(annotation, signal)
    annotation["provenance"]["annotation_timestamp_utc"] = "2026-99-99"
    with pytest.raises(ValidationError, match="date-time"):
        validate_evidence(annotation, signal, replace(reviewer, annotation_timestamp_utc="2026-99-99"))


def test_reviewed_silver_requires_actual_reviewer_but_not_expert(annotation, signal):
    annotation["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="external reviewer"):
        validate_evidence(annotation, signal)


def test_context_policy_is_separate_from_clinical_sufficiency(annotation, signal):
    for region in annotation["acceptable_evidence_sets"][0]["regions"]:
        region["role"] = "context"
    assert validate_evidence(annotation, signal)["warnings"]
    reviewer = expert(annotation, signal)
    with pytest.raises(ValueError, match="without direct support"):
        validate_evidence(annotation, signal, reviewer, strict_blinded_expert=True)


def test_strict_policy_rejects_synthetic_even_structurally_valid(annotation, signal):
    with pytest.raises(ValueError, match="Strict evaluation"):
        validate_evidence(annotation, signal, strict_blinded_expert=True)


def test_pending_forms_cannot_become_annotations(signal):
    form = pending_form("case-001")
    validate_pending_form(form)
    assert all(item["state"] is None for item in form["review_prompts"])
    with pytest.raises(ValidationError):
        validate_evidence(form, signal)
    form["review_prompts"][0]["state"] = "uncertain"
    with pytest.raises(ValidationError):
        validate_pending_form(form)


@pytest.mark.parametrize(
    "timestamp", ["2026-10-19", "2026-10-19T12:00:00", "2026-02-30T12:00:00Z", "bad", "2026-10-19T25:00:00Z"]
)
def test_missing_or_invalid_timezone_and_calendar(annotation, signal, timestamp):
    reviewer = expert(annotation, signal)
    annotation["provenance"]["annotation_timestamp_utc"] = timestamp
    with pytest.raises(ValidationError, match="date-time"):
        validate_evidence(annotation, signal, replace(reviewer, annotation_timestamp_utc=timestamp))


def test_review_timestamp_requires_utc(annotation, signal):
    reviewer = expert(annotation, signal)
    timestamp = "2026-10-19T12:00:00+02:00"
    annotation["provenance"]["annotation_timestamp_utc"] = timestamp
    with pytest.raises(ValueError, match="UTC"):
        validate_evidence(annotation, signal, replace(reviewer, annotation_timestamp_utc=timestamp))


def test_adjudication_requires_completed_status_and_attestation(annotation, signal):
    reviewer = expert(annotation, signal)
    annotation["provenance"]["kind"] = "expert_adjudication"
    with pytest.raises(ValueError, match="adjudicated"):
        validate_evidence(annotation, signal, reviewer)
    annotation["review_status"] = "adjudicated"
    assert (
        validate_evidence(annotation, signal, reviewer, strict_blinded_expert=True)[
            "clinical_truth_certified"
        ]
        is False
    )


def test_measurement_cannot_reference_another_existing_set(annotation, signal):
    other = {
        "set_id": "other",
        "rationale": "Invented alternative",
        "regions": [
            {
                "region_id": "other-region",
                "leads": ["V1"],
                "role": "support",
                "coordinates": {"kind": "whole_record"},
            }
        ],
    }
    annotation["acceptable_evidence_sets"].append(other)
    annotation["acceptable_evidence_sets"][0]["measurements"][0]["region_ids"] = ["other-region"]
    with pytest.raises(ValueError, match="same evidence set"):
        validate_evidence(annotation, signal)


@pytest.mark.parametrize("offset", ["+00:99", "-00:60", "+24:00", "+99:00"])
def test_invalid_timezone_components_rejected(annotation, signal, offset):
    reviewer = expert(annotation, signal)
    timestamp = "2026-10-19T12:00:00" + offset
    annotation["provenance"]["annotation_timestamp_utc"] = timestamp
    with pytest.raises(ValidationError, match="date-time"):
        validate_evidence(annotation, signal, replace(reviewer, annotation_timestamp_utc=timestamp))


def test_reviewed_independent_silver_remains_nonexpert(annotation, signal):
    reviewer = expert(annotation, signal)
    annotation["provenance"]["kind"] = "rule_generated_silver"
    validate_evidence(annotation, signal, reviewer)
    with pytest.raises(ValueError, match="Strict evaluation"):
        validate_evidence(annotation, signal, reviewer, strict_blinded_expert=True)
