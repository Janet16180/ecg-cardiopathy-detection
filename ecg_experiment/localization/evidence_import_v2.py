"""Forward strict admission for completed independent expert evidence imports."""

from __future__ import annotations

import json
from pathlib import Path

from ecg_experiment.localization.evidence_validation import (
    BeatBinding,
    ReviewerBinding,
    SignalBinding,
    validate_evidence,
)


def validate_completed_evidence(
    annotation: dict,
    signal: SignalBinding,
    reviewer: ReviewerBinding,
    beats: BeatBinding | None = None,
) -> dict:
    """Admit completed independent expert evidence with a separate lineage policy gate.

    Parameters
    ----------
    annotation : dict
        A completed finding-specific annotation containing finite standard JSON values,
        never a pending packet form.
    signal : SignalBinding
        Separately verified signal-byte and coordinate binding.
    reviewer : ReviewerBinding
        Externally established reviewer and policy attestation. This function cannot
        authenticate identity, expertise, independence or the fact that review occurred.
    beats : BeatBinding or None
        Frozen signal-bound beat anchors when ordinal coordinates are used.

    Returns
    -------
    dict
        Technical validation counts and the admission-policy version. Clinical truth
        and sufficient evidence remain unverified by software.
    """
    json.dumps(annotation, allow_nan=False)
    provenance = annotation.get("provenance", {})
    if provenance.get("kind") == "expert_adjudication" or annotation.get("review_status") == "adjudicated":
        raise ValueError("Strict adjudication import requires a future lineage-aware policy")
    result = validate_evidence(annotation, signal, reviewer, beats, strict_blinded_expert=True)
    return {**result, "import_policy": "independent-expert-import-v2", "reviewer_authenticated": False}


def import_completed_evidence(
    annotation_path: Path,
    signal_path: Path,
    reviewer_path: Path,
    beat_path: Path | None = None,
) -> dict:
    """Load local completed evidence and separate trusted binding files without mutation.

    Parameters
    ----------
    annotation_path, signal_path, reviewer_path : Path
        Local annotation and externally established trust bindings.
    beat_path : Path or None
        Optional externally bound original-sample beat anchors.

    Returns
    -------
    dict
        Technical admission counts, with no clinical validity or authentication claim.
    """
    signal = json.loads(signal_path.read_text())
    signal["lead_order"] = tuple(signal["lead_order"])
    reviewer = ReviewerBinding(**json.loads(reviewer_path.read_text()))
    beats = json.loads(beat_path.read_text()) if beat_path is not None else None
    if beats is not None:
        beats["sample_positions"] = tuple(beats["sample_positions"])
    return validate_completed_evidence(
        json.loads(annotation_path.read_text()),
        SignalBinding(**signal),
        reviewer,
        BeatBinding(**beats) if beats is not None else None,
    )
