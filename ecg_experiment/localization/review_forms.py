"""Create pending forms that cannot be mistaken for completed evidence annotations."""

from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator, FormatChecker

from ecg_experiment import ROOT

FORM_SCHEMA = ROOT / "schemas/localization/pending-review-form-v1.schema.json"
PROMPTS = (
    "Sinus rhythm",
    "Complete right bundle branch block",
    "Complete left bundle branch block",
    "Other finding",
)


def pending_form(case_ref: str) -> dict:
    """Return identical blank review prompts without imported patient or model labels."""
    if not re.fullmatch(r"case-[0-9]{3}", case_ref):
        raise ValueError("Invalid blinded case reference")
    return {
        "schema_version": "ecg-pending-review-1.0",
        "case_ref": case_ref,
        "status": "pending",
        "round": 1,
        "rubric_status": "awaiting_clinician_approval",
        "reviewer_ref": None,
        "review_timestamp_utc": None,
        "model_output_shown": False,
        "source_labels_shown": False,
        "review_prompts": [{"name": name, "state": None, "evidence_annotation": None} for name in PROMPTS],
        "note": None,
    }


def validate_pending_form(value: dict) -> None:
    """Require a genuinely blank form, separately from the finding annotation schema."""
    schema = json.loads(FORM_SCHEMA.read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    if value != pending_form(value["case_ref"]):
        raise ValueError("Pending form differs from the identical blank review template")
