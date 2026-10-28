"""Immutable local artifact gates for the corrective PR 74 development evaluation."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from . import ROOT
from .files import sha256_file, write_json_atomic
from .paths import to_stored

OUTPUT = Path("outputs/pr74_corrective_evaluation_20261028")
COHORT = Path("outputs/cohort_age_label_priority_20261028")
PRIOR = Path("outputs/screen01_matched_transfer_20261027/cuda_runtime_execution")
CPU_PRIOR = Path("outputs/meeting_20261027_recovered/experiment041_cpu_screening_v2")
MEASURE_PRIOR = Path("outputs/meeting_20261027_recovered/experiment042_measurement_transfer_v1")
PROTOCOL = Path("docs/pr74-corrective-evaluation-protocol-2026-10-28.md")
SPLIT_SHA256 = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
FROZEN_SOURCES = (
    "cpu_screening041", "measurement_extractor042", "measurement_screening042", "external_encoders",
    "full_development", "intervals", "ecg_quality", "public_sources", "recovery_views", "waveforms",
    "xecg", "gpu", "paths", "files", "age_cohort_inputs",
)


def relative(path: str | Path) -> Path:
    """Require a local root-relative artifact name without traversal."""
    path = Path(path)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("Unsafe artifact path")
    return path


def stored(path: str | Path) -> str:
    """Use the shared portable receipt path convention for an explicit input root."""
    return to_stored(ROOT / relative(path))


def read_json(path: Path) -> dict[str, Any]:
    """Read a local structured receipt."""
    return json.loads(path.read_text())


def verify_hashes(root: Path, hashes: dict[str, str]) -> None:
    """Reject changed or unsafe evidence paths before using or completing a stage."""
    for name, expected in hashes.items():
        if sha256_file(root / relative(name)) != expected:
            raise ValueError(f"Artifact changed: {name}")


def binding() -> dict[str, Any]:
    """Capture committed scientific code and protocol before any numerical stage starts."""
    own = sorted((ROOT / "ecg_experiment").glob("pr74_screening_*.py"))
    own += sorted((ROOT / "scripts/experiments").glob("run_pr74_screening*.py"))
    own += sorted((ROOT / "tests").glob("test_pr74_screening*.py"))
    own.append(ROOT / PROTOCOL)
    for path in own:
        blob = subprocess.check_output(["git", "show", "HEAD:" + to_stored(path)], cwd=ROOT)
        if blob != path.read_bytes():
            raise ValueError(f"Scientific source must be committed before execution: {path.name}")
    sources = own + [ROOT / f"ecg_experiment/{name}.py" for name in FROZEN_SOURCES]
    return {
        "source_sha256": {to_stored(path): sha256_file(path) for path in sources},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(ROOT / PROTOCOL)},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(), "scope": "reused_development", "threads": 2,
        "versions": {name: version(name) for name in
                     ("numpy", "scipy", "pandas", "scikit-learn", "torch", "wfdb", "joblib")},
    }


def completed(input_root: Path, directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify every artifact of a local predecessor or previously completed stage."""
    path = relative(directory) / "completion.json"
    receipt = read_json(input_root / path)
    legacy = directory.is_relative_to(MEASURE_PRIOR)
    if receipt.get("status") != "completed" and not (legacy and receipt.get("artifacts")):
        raise ValueError(f"Stage is incomplete: {directory}")
    hashes = {stored(path): sha256_file(input_root / path)}
    for name, digest in receipt["artifacts"].items():
        artifact = Path(name) if Path(name).parts[0] == "outputs" else directory / name
        hashes[stored(artifact)] = digest
    verify_hashes(input_root, hashes)
    return receipt, hashes


def training_pool(input_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read the entire admitted candidate union so later prespecified subsets reuse one extraction."""
    _, hashes = cohort_tables(input_root)
    rows = pd.read_csv(input_root / COHORT / "candidates.csv", dtype=str, keep_default_na=False)
    if (not rows["split"].eq("train").all() or not rows["quality_passed"].eq("True").all() or
            not rows["samples"].eq("5000").all() or rows["record_id"].duplicated().any() or
            rows["signal_sha256"].duplicated().any()):
        raise ValueError("Admitted training pool is invalid")
    return rows.sort_values("record_id", kind="stable").reset_index(drop=True), hashes


def cohort_tables(input_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Require completed verified cohort assembly and unmodified training-only manifests."""
    path = COHORT / "receipt.json"
    receipt = read_json(input_root / path)
    if (receipt.get("complete") is not True or receipt.get("scope") != "new_supervised_age_label_cohorts" or
            receipt.get("original_split_sha256") != SPLIT_SHA256):
        raise ValueError("Completed original-split cohort assembly is required")
    verify_hashes(input_root, receipt["artifacts"])
    verify_hashes(input_root, receipt["input_sha256"])
    verify_hashes(ROOT, receipt["source_sha256"])
    hashes = {stored(path): sha256_file(input_root / path), **receipt["artifacts"]}
    tables = {}
    for name in ("younger", "all_age"):
        rows = pd.read_csv(input_root / COHORT / (name + ".csv"), dtype=str, keep_default_na=False)
        if (not rows["split"].eq("train").all() or not rows["quality_passed"].eq("True").all() or
                not rows["samples"].eq("5000").all() or rows["record_id"].duplicated().any() or
                rows["signal_sha256"].duplicated().any()):
            raise ValueError("Cohort admission, quality, duration or identity gate failed")
        rows["target"] = pd.to_numeric(rows["target"], errors="raise").astype(int)
        if set(rows["target"]) != {0, 1}:
            raise ValueError("Each training cohort must contain both defined classes")
        tables[name] = rows
    return tables, hashes


def finish(input_root: Path, output: Path, stage: str, receipt: dict[str, Any],
           inputs: dict[str, str], details: dict[str, Any]) -> dict[str, Any]:
    """Recheck captured evidence before publishing a stage's completion marker."""
    verify_hashes(input_root, inputs)
    verify_hashes(ROOT, receipt["source_sha256"])
    directory = input_root / relative(output) / stage
    artifacts = {stored(output / stage / path.name): sha256_file(path)
                 for path in sorted(directory.iterdir()) if path.is_file() and path.name != "completion.json"}
    value = {**receipt, **details, "status": "completed", "stage": stage,
             "input_sha256": inputs, "artifacts": artifacts}
    write_json_atomic(directory / "completion.json", value)
    return value
