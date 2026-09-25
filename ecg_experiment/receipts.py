"""Fingerprinted JSON receipts that prove a run finished with the requested inputs.

Runners use three historical ``completion`` layouts, kept as they are on disk:
``complete.json`` with a ``sha256`` map (``verified_completion``),
``completion.json`` with an ``artifacts`` map (``check_completed_stage``), and
``completion.json`` with one named digest field per artifact
(``write_completion``/``check_completion``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .files import sha256_file, write_json_atomic


def artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    """
    Hash the named artifacts of a run directory.

    Parameters
    ----------
    directory : Path
        Run directory.
    names : Iterable[str]
        Artifact file names inside ``directory``.

    Returns
    -------
    dict[str, str]
        SHA-256 digest keyed by file name.
    """
    return {name: sha256_file(directory / name) for name in names}


def verified_completion(directory: Path, fingerprint: dict[str, Any],
                        names: Iterable[str]) -> dict[str, Any] | None:
    """
    Read a run's ``complete.json`` and verify its inputs and artifacts.

    Parameters
    ----------
    directory : Path
        Run directory.
    fingerprint : dict[str, Any]
        Inputs the completed run must have used.
    names : Iterable[str]
        Artifacts whose digests the receipt records.

    Returns
    -------
    dict[str, Any] | None
        The receipt, or None when the run has not completed.

    Raises
    ------
    ValueError
        If the completed run used other inputs or an artifact changed.
    """
    marker = directory / "complete.json"
    if not marker.is_file():
        return None
    receipt = json.loads(marker.read_text())
    if receipt.get("fingerprint") != fingerprint:
        raise ValueError(f"Completed run differs from requested inputs: {directory}")
    if receipt.get("sha256") != artifact_hashes(directory, names):
        raise ValueError(f"Completed artifact checksum mismatch: {directory}")
    return receipt


def check_completed_stage(directory: Path, fingerprint: str) -> None:
    """
    Check a stage's ``completion.json`` fingerprint and its recorded artifact digests.

    Parameters
    ----------
    directory : Path
        Stage directory holding ``completion.json``.
    fingerprint : str
        Digest the completed stage must match.

    Raises
    ------
    ValueError
        If the fingerprint differs or a recorded artifact changed.
    """
    saved = json.loads((directory / "completion.json").read_text())
    if saved["fingerprint"] != fingerprint:
        raise ValueError(f"Completed training fingerprint mismatch: {directory}")
    for name, expected_hash in saved["artifacts"].items():
        if sha256_file(directory / name) != expected_hash:
            raise ValueError(f"Completed artifact changed: {directory / name}")


def check_completion(directory: Path, fingerprint: str, artifacts: dict[str, str]) -> list[dict[str, Any]]:
    """
    Verify a completed arm and return its history.

    Parameters
    ----------
    directory : Path
        Arm directory holding ``completion.json``.
    fingerprint : str
        Identity the completed arm must match.
    artifacts : dict[str, str]
        Completion field for each artifact file, such as
        ``{"history.json": "history_sha256"}``.

    Returns
    -------
    list[dict[str, Any]]
        Saved per-epoch history.

    Raises
    ------
    ValueError
        If the fingerprint or an artifact digest differs.
    """
    done = json.loads((directory / "completion.json").read_text())
    if done["fingerprint"] != fingerprint:
        raise ValueError("Completed arm fingerprint mismatch")
    for name, key in artifacts.items():
        if sha256_file(directory / name) != done[key]:
            raise ValueError(f"Completed artifact changed: {name}")
    return json.loads((directory / "history.json").read_text())


def write_completion(directory: Path, fingerprint: str, best: dict[str, Any],
                     artifacts: dict[str, str]) -> None:
    """
    Record a completed arm with its best epoch and artifact digests.

    Parameters
    ----------
    directory : Path
        Arm directory.
    fingerprint : str
        Arm identity.
    best : dict[str, Any]
        Best ``auc`` and ``epoch``.
    artifacts : dict[str, str]
        Completion field for each artifact file, in output order.
    """
    write_json_atomic(directory / "completion.json", {
        "fingerprint": fingerprint, "best_epoch": best["epoch"], "best_development_auroc": best["auc"],
        **{key: sha256_file(directory / name) for name, key in artifacts.items()}})


def check_existing_config(path: Path, fingerprint: str) -> None:
    """
    Refuse to reuse an arm directory created from other inputs.

    Parameters
    ----------
    path : Path
        The arm's ``config.json``, which may not exist yet.
    fingerprint : str
        Identity of this run.

    Raises
    ------
    ValueError
        If the saved configuration has a different fingerprint.
    """
    if path.exists() and json.loads(path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing output uses different inputs or code")


def require_receipt(path: Path, fingerprint: str, message: str) -> dict[str, Any]:
    """
    Load a receipt that must exist and match the current provenance.

    Parameters
    ----------
    path : Path
        Receipt file.
    fingerprint : str
        Required ``fingerprint`` value.
    message : str
        Error message when the receipt is missing or differs.

    Returns
    -------
    dict[str, Any]
        The receipt.

    Raises
    ------
    ValueError
        If the receipt is missing or its fingerprint differs.
    """
    receipt = json.loads(path.read_text()) if path.exists() else {}
    if receipt.get("fingerprint") != fingerprint:
        raise ValueError(message)
    return receipt
