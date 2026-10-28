"""Pair fixed aggregate, cohort and model artifacts for each trusted startup mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactMode:
    """One immutable execution namespace; no fields come from an HTTP request."""

    name: str
    evaluation_dir: Path
    primary_cohorts: Path
    secondary_cohorts: Path
    inference_module: str


CACHED = ArtifactMode(
    "cached", Path("outputs/pr74_corrective_evaluation_20261028/evaluate"),
    Path("outputs/cohort_strict_young_matched_20261028"), Path("outputs/cohort_age_label_priority_20261028"),
    "ecg_experiment.pr74_screening_inference",
)
FROM_RAW = ArtifactMode(
    "from_raw", Path("outputs/pr74_from_raw_validation_20261028/evaluate"),
    Path("outputs/cohort25_recovery_20261028/cold_start/strict_matched"),
    Path("outputs/cohort25_recovery_20261028/cold_start/cohorts_25k"),
    "ecg_experiment.pr74_raw_inference",
)


def artifact_mode(name: str) -> ArtifactMode:
    """Require an explicit allowlisted mode instead of accepting artifact paths or fallbacks."""
    if name not in {"cached", "from_raw"}:
        raise ValueError("ECG_APP_ARTIFACT_MODE must be cached or from_raw")
    return CACHED if name == "cached" else FROM_RAW
