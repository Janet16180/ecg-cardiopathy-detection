"""Serve the NBR-01 normal-beat replacement examples and the study summary exported with them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactUnavailableError, read_json

NBR01 = Path("outputs/nbr01_normal_beat_replacement_20261028")
MODELS = ("all_age_blend", "ptb_baseline_blend")


def load_localization(root: Path) -> dict[str, Any]:
    """Return the exported examples after checking the schema the page reads."""
    examples = read_json(root / NBR01 / "examples.json")
    summary = examples.get("summary")
    if (examples.get("schema_version") != 1 or set(examples["models"]) != set(MODELS)
            or not isinstance(summary, dict) or not {"statistics", "n", "decisions"} <= set(summary)):
        raise ArtifactUnavailableError("The NBR-01 export does not match the expected schema.")
    return examples
