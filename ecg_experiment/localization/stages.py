"""Shared stage receipts for the NBR studies from NBR-03 on: source binding, completion and verification."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .. import ROOT
from ..files import read_json, sha256_file, write_json_atomic
from ..paths import to_stored
from ..pr74_screening_io import verify_hashes

FROZEN = (
    "normative_map062",
    "replacement070",
    "combination066",
    "pr74_fresh_baseline",
    "pr74_screening_features",
    "pr74_screening_io",
    "external_encoders",
    "xecg",
    "measurement_extractor042",
    "cpu_screening041",
    "full_development",
    "intervals",
    "gpu",
    "files",
    "paths",
    "heads01_features",
    "locbench_cache",
    "locbench_data",
    "eda/ptbxl",
)


def binding(prefix: str, protocol: Path, require_committed: bool = True) -> dict[str, Any]:
    """
    Hash a study's own sources, every NBR localization module and the frozen pipeline.

    Parameters
    ----------
    prefix : str
        File-name prefix of the study's own modules, script and tests, such as ``"nbr03"``.
    protocol : Path
        Repository-relative protocol path.
    require_committed : bool
        Require the study's own sources and protocol to equal ``HEAD``.

    Returns
    -------
    dict[str, Any]
        Source hashes, protocol hash and git revision.

    Raises
    ------
    ValueError
        If an own source or the protocol differs from ``HEAD``.
    """
    own = sorted((ROOT / "ecg_experiment/localization").glob(f"{prefix}_*.py"))
    own += sorted((ROOT / "scripts/experiments").glob(f"run_{prefix}.py"))
    own += sorted((ROOT / "tests/localization").glob(f"test_{prefix}*.py"))
    if require_committed:
        for path in [*own, ROOT / protocol]:
            blob = subprocess.check_output(["git", "show", "HEAD:" + to_stored(path)], cwd=ROOT)
            if blob != path.read_bytes():
                raise ValueError(f"Source must be committed before execution: {path.name}")
    sources = sorted((ROOT / "ecg_experiment/localization").glob("*.py"))
    sources += [ROOT / f"ecg_experiment/{name}.py" for name in FROZEN]
    sources += [path for path in own if path not in sources]
    return {
        "source_sha256": {to_stored(path): sha256_file(path) for path in sources},
        "protocol": {
            "path": str(protocol),
            "sha256": sha256_file(ROOT / protocol) if (ROOT / protocol).exists() else None,
        },
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def stage_dir(input_root: Path, output: Path, stage: str) -> Path:
    """Create and return one stage's output directory."""
    directory = input_root / output / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def complete(
    input_root: Path, output: Path, stage: str, receipt: dict[str, Any], details: dict[str, Any]
) -> dict[str, Any]:
    """Bind a stage's artifacts and sources in its completion receipt."""
    directory = input_root / output / stage
    verify_hashes(ROOT, receipt["source_sha256"])
    artifacts = {
        to_stored(path): sha256_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "completion.json"
    }
    value = {**receipt, "details": details, "status": "completed", "stage": stage, "artifacts": artifacts}
    write_json_atomic(directory / "completion.json", value)
    return value


def require(input_root: Path, output: Path, stage: str) -> dict[str, Any]:
    """Verify a completed stage's artifacts and return its details."""
    receipt = read_json(input_root / output / stage / "completion.json")
    if receipt.get("status") != "completed":
        raise ValueError(f"Stage {stage} is not complete")
    verify_hashes(input_root, receipt["artifacts"])
    return receipt["details"]
