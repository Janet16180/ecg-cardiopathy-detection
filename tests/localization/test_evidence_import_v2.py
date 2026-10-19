"""Strict import regressions using invented evidence and attestations only."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from ecg_experiment import ROOT
from ecg_experiment.localization.evidence_import_v2 import (
    import_completed_evidence,
    validate_completed_evidence,
)
from ecg_experiment.localization.evidence_validation import (
    BeatBinding,
    ReviewerBinding,
    SignalBinding,
    beat_digest,
)


@pytest.fixture
def invented():
    annotation = json.loads((ROOT / "schemas/ecg-evidence-synthetic-example.json").read_text())
    annotation["signal_sha256"] = "a" * 64
    annotation["review_status"] = "reviewed"
    annotation["provenance"].update(
        kind="independent_expert",
        annotator_ref="invented-reviewer-not-a-real-clinician",
        annotation_timestamp_utc="2026-10-19T12:00:00Z",
        independent_of_model=True,
        independent_of_support_bank=True,
        model_output_visible=False,
    )
    signal = SignalBinding(
        annotation["record_ref"],
        annotation["signal_sha256"],
        annotation["sampling_rate_hz"],
        annotation["n_samples"],
        tuple(annotation["lead_order"]),
        annotation.get("patient_ref"),
    )
    reviewer = ReviewerBinding(
        annotation["provenance"]["annotator_ref"],
        signal.record_ref,
        signal.signal_sha256,
        annotation["provenance"]["annotation_timestamp_utc"],
        "invented-test-policy",
        True,
        True,
        True,
        False,
    )
    positions = tuple(range(200, signal.n_samples, 400))
    revision = "invented-detector"
    beats = BeatBinding(
        positions, revision, signal.signal_sha256, beat_digest(positions, revision, signal.signal_sha256)
    )
    return annotation, signal, reviewer, beats


def test_known_utc_import_preserves_alternatives_and_unknowns(invented):
    """A technical pass never authenticates the fictional reviewer or certifies clinical truth."""
    annotation, signal, reviewer, beats = invented
    original = deepcopy(annotation)
    result = validate_completed_evidence(annotation, signal, reviewer, beats)
    assert result["structurally_and_semantically_valid"]
    assert not result["clinical_truth_certified"]
    assert not result["reviewer_authenticated"]
    assert result["evidence_sets"] == len(annotation["acceptable_evidence_sets"])
    assert annotation == original


@pytest.mark.parametrize("suffix", ["Z", "+00:00", "-00:00"])
def test_utc_instants_keep_original_local_offset_semantics(invented, suffix):
    """Unknown local offset does not make the represented UTC instant unknown."""
    annotation, signal, reviewer, beats = invented
    timestamp = "2026-10-19T12:00:00" + suffix
    annotation["provenance"]["annotation_timestamp_utc"] = timestamp
    reviewer = replace(reviewer, annotation_timestamp_utc=timestamp)
    assert validate_completed_evidence(annotation, signal, reviewer, beats)[
        "structurally_and_semantically_valid"
    ]
    assert annotation["provenance"]["annotation_timestamp_utc"] == timestamp


@pytest.mark.parametrize("kind", ["expert_adjudication", "independent_expert"])
def test_adjudication_waits_for_lineage_policy(invented, kind):
    """An expertise flag cannot substitute for verified original adjudication records."""
    annotation, signal, reviewer, beats = invented
    annotation["provenance"]["kind"] = kind
    annotation["review_status"] = "adjudicated"
    with pytest.raises(ValueError, match="lineage-aware"):
        validate_completed_evidence(annotation, signal, reviewer, beats)


def test_strict_import_retains_original_signal_and_provenance_checks(invented):
    """The forward policy cannot bypass the frozen validator's signal and independence gates."""
    annotation, signal, reviewer, beats = invented
    with pytest.raises(ValueError, match="signal binding"):
        validate_completed_evidence(annotation, replace(signal, signal_sha256="b" * 64), reviewer, beats)
    annotation["provenance"]["kind"] = "synthetic_example"
    with pytest.raises(ValueError, match="completed independent"):
        validate_completed_evidence(annotation, signal, reviewer, beats)


def test_local_import_does_not_rewrite_binding_files(invented, tmp_path):
    """CLI-facing local loading preserves exact annotation and attestation bytes."""
    annotation, signal, reviewer, beats = invented
    paths = []
    for name, value in (
        ("annotation", annotation),
        ("signal", asdict(signal)),
        ("reviewer", asdict(reviewer)),
        ("beats", asdict(beats)),
    ):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        paths.append(path)
    before = [path.read_bytes() for path in paths]
    assert import_completed_evidence(*paths)["import_policy"] == "independent-expert-import-v2"
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_import_rejects_nonfinite_json_numbers(invented, value):
    """NaN and infinity cannot enter a completed annotation through the Python API."""
    annotation, signal, reviewer, beats = invented
    annotation["sampling_rate_hz"] = value
    with pytest.raises(ValueError, match="Out of range float"):
        validate_completed_evidence(annotation, signal, reviewer, beats)


def test_import_rejects_numpy_scalars_before_schema_type_checks(invented):
    """A NumPy NaN cannot bypass the frozen validator's built-in-float check."""
    annotation, signal, reviewer, beats = invented
    region = annotation["acceptable_evidence_sets"][0]["regions"][0]
    annotation["acceptable_evidence_sets"][0]["measurements"] = [
        {
            "name": "Invented API-only measurement",
            "value": np.float32(np.nan),
            "unit": "ms",
            "method": "invented",
            "leads": region["leads"],
            "region_ids": [region["region_id"]],
        }
    ]
    with pytest.raises(TypeError, match="JSON serializable"):
        validate_completed_evidence(annotation, signal, reviewer, beats)
