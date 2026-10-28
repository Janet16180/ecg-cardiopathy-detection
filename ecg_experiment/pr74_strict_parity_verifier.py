"""Compare complete saved corrective evaluation payloads without fitting or scoring models."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import ROOT
from .files import sha256_file
from .paths import to_stored

PUBLISHED = Path("reports/pr74-corrective-evaluation-2026-10-28/aggregate.json")
SAVED = {
    "warm": Path("outputs/pr74_corrective_evaluation_20261028/evaluate/screening_evaluation.json"),
    "fresh": Path("outputs/pr74_from_raw_validation_20261028/evaluate/screening_evaluation.json"),
}
SOURCE_FILES = (
    "ecg_experiment/pr74_strict_parity_verifier.py",
    "scripts/reports/verify_pr74_strict_parity.py",
    "tests/test_pr74_strict_parity_verifier.py",
)
MODEL_IDS = {cohort + "_" + kind
             for cohort in ("strict_young", "matched_general", "younger", "all_age", "ptb_baseline")
             for kind in ("hgb", "xecg", "blend")}


def canonical(value: object) -> str:
    """Compare every JSON field and numeric representation while rejecting nonfinite values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def inventory(artifact: dict) -> dict[tuple[str, str], dict]:
    """Require exactly the registered 15 models and both development hospitals, without duplicate points."""
    if artifact.get("complete") is not True or len(artifact["models"]) != len(MODEL_IDS):
        raise ValueError("Expected a complete 15-model aggregate")
    points = {}
    models = []
    for model in artifact["models"]:
        models.append(model["id"])
        if len(model["evaluations"]) != 2:
            raise ValueError("Each model requires exactly two hospital evaluations")
        for evaluation in model["evaluations"]:
            key = (model["id"], evaluation["pool"])
            if key in points or key[1] not in {"ptb", "sph"}:
                raise ValueError("Repeated or unexpected model/hospital evaluation")
            points[key] = evaluation
    if set(models) != MODEL_IDS or len(points) != 30:
        raise ValueError("Registered model/hospital inventory differs")
    return points


def compare_full_payloads(actual: dict, published: dict) -> dict:
    """Reject changes to any evaluation field or any field of the complete primary comparison."""
    first, second = inventory(actual), inventory(published)
    for key in sorted(first):
        if canonical(first[key]) != canonical(second[key]):
            raise ValueError(f"Complete evaluation payload differs: {key[0]}/{key[1]}")
    if canonical(actual["primary"]) != canonical(published["primary"]):
        raise ValueError("Complete primary comparison differs")
    return {"model_site_points": 30, "all_complete_evaluation_payloads_exact": True,
            "complete_primary_exact": True}


def committed_sources() -> dict[str, str]:
    """Bind the new verifier and its mutation tests to committed bytes before aggregate access."""
    result = {}
    for name in SOURCE_FILES:
        blob = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Strict verifier source and tests must be committed")
        result[name] = digest
    return result


def verify_saved(input_root: Path) -> dict:
    """Verify warm and fresh saved aggregates against the published aggregate with no prediction reads.

    Parameters
    ----------
    input_root : Path
        Local checkout holding the completed warm and fresh evaluation JSON artifacts.

    Returns
    -------
    dict
        Aggregate-only exact-parity evidence with committed verifier and immutable input bindings.
    """
    sources = committed_sources()
    published_path = ROOT / PUBLISHED
    published_sha = sha256_file(published_path)
    published = json.loads(published_path.read_text())
    inputs, checked = {}, {}
    for mode, path in SAVED.items():
        completion_path = path.parent / "completion.json"
        completion = json.loads((input_root / completion_path).read_text())
        if completion.get("status") != "completed" or completion.get("stage") != "evaluate":
            raise ValueError("Saved aggregate requires a completed evaluation receipt")
        digest = sha256_file(input_root / path)
        name = to_stored(ROOT / path)
        if completion["artifacts"].get(name) != digest:
            raise ValueError("Saved aggregate differs from its completed artifact binding")
        inputs[name] = digest
        inputs[to_stored(ROOT / completion_path)] = sha256_file(input_root / completion_path)
        actual = json.loads((input_root / path).read_text())
        checked[mode] = compare_full_payloads(actual, published)
    if any(sha256_file(input_root / name) != digest for name, digest in inputs.items()):
        raise ValueError("Saved inputs changed during strict verification")
    if sha256_file(published_path) != published_sha or sources != committed_sources():
        raise ValueError("Published aggregate or verifier changed during strict verification")
    return {
        "complete": True, "scope": "strict_saved_aggregate_parity", "checked": checked,
        "source_sha256": sources, "input_sha256": inputs,
        "published_aggregate": {"path": str(PUBLISHED), "sha256": published_sha},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "new_model_fits": 0, "new_model_scores": 0, "prediction_array_reads": 0,
        "raw_waveform_reads": 0, "closed_test_reads": 0,
    }
