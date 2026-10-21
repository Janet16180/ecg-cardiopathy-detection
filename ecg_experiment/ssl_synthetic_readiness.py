"""Inspect prerequisite paths without reading ECG or annotation payloads."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from ecg_experiment import ROOT


def checked_relative(value: str) -> Path:
    """
    Validate a repository-relative prerequisite path.

    Parameters
    ----------
    value : str
        POSIX path inside the repository.

    Returns
    -------
    Path
        Relative path without parent traversal.

    Raises
    ------
    ValueError
        If the path is empty, absolute or contains parent traversal.
    """
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"Expected a repository-relative path: {value!r}")
    return Path(*path.parts)


def validate_spec(spec: dict[str, Any]) -> None:
    """
    Validate a nonempty readiness specification before inspecting any target.

    Parameters
    ----------
    spec : dict
        Version, stream name and named file or directory checks.

    Raises
    ------
    ValueError
        If the schema, names, paths, kinds or required flags are invalid.
    """
    if spec.get("schema_version") != 1 or spec.get("stream") != "ssl_synthetic_localization":
        raise ValueError("Unsupported readiness schema or stream")
    checks = spec.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("Expected a nonempty list of prerequisite checks")
    names = set()
    for check in checks:
        name = check.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Each prerequisite must have a unique nonempty name")
        names.add(name)
        checked_relative(check["path"])
        if check.get("kind") not in {"file", "directory"} or type(check.get("required")) is not bool:
            raise ValueError(f"Invalid kind or required flag for {name}")
    if not any(check["required"] for check in checks):
        raise ValueError("At least one prerequisite must be required")


def inspect_prerequisites(spec: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """
    Inspect existence and type only, never opening target file contents.

    Parameters
    ----------
    spec : dict
        Validated by this function before any filesystem target is inspected.
    root : Path
        Repository root, optionally replaced by a temporary directory for tests.

    Returns
    -------
    dict
        Availability report. Present paths do not establish scientific or runtime readiness.
    """
    validate_spec(spec)
    checks = []
    for check in spec["checks"]:
        target = root / checked_relative(check["path"])
        present = target.is_file() if check["kind"] == "file" else target.is_dir()
        checks.append({**check, "present": present})
    missing = [check["name"] for check in checks if check["required"] and not check["present"]]
    return {
        "schema_version": 1,
        "stream": spec["stream"],
        "scope": "Path existence and type only; no ECG or annotation payload was opened",
        "required_paths_present": not missing,
        "missing_required": missing,
        "checks": checks,
        "scientific_readiness": "not_assessed",
        "performance_scored": False,
        "training_started": False,
        "not_verified": [
            "dataset completeness and waveform checksums",
            "patient separation and pretraining exposure",
            "clinical annotation correctness or completeness",
            "installed dependencies and runtime compatibility",
            "generator qualification and predecessor reproduction",
            "GPU profile contents and resource gate",
        ],
    }


def load_spec(path: Path) -> dict[str, Any]:
    """
    Read and validate the versioned prerequisite configuration.

    Parameters
    ----------
    path : Path
        JSON configuration file, containing no patient payload.

    Returns
    -------
    dict
        Valid specification.
    """
    spec = json.loads(path.read_text())
    validate_spec(spec)
    return spec
