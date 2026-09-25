"""Verified waveform identities of the frozen SSL pools and curated Challenge candidates.

Every loader receives a ``pin`` recorder so each metadata dependency it reads is
hashed into the caller's input receipt.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from . import ROOT
from .files import SHA256_HEX_LENGTH, read_csv, sha256_file, sha256_json

PTB_REFERENCE = ROOT / "outputs/data_quality/ptbxl_reference_hashes_v2.json"
PTB_PILOT_METADATA = ROOT / "data/processed/public_ecg_quality/astra_v2_pilot32/metadata.json"
MIMIC_POOL = ROOT / "data/processed/mimic_ssl_40k_cpc"
CHALLENGE_VIEWS = ROOT / "data/processed/challenge_ecg_views"
VIEW_FIELDS = ("source", "signal_sha256", "shard_index", "window_start", "source_samples",
               "window_samples", "sampling_rate_hz", "units", "lead_order")

PinInput = Callable[[Path], Path]


def load_candidates(candidate_dir: Path, pin: PinInput) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """
    Load the curated SSL candidate manifest and enforce its declared eligibility.

    Parameters
    ----------
    candidate_dir : Path
        Directory with ``challenge_ssl_curated_manifest.csv`` and its receipt.
    pin : PinInput
        Records each input file read.

    Returns
    -------
    tuple[list[dict[str, str]], dict[str, Any]]
        Candidate rows and the curation receipt.

    Raises
    ------
    ValueError
        If the manifest hash or count differs, or a row claims supervised or
        patient-independent evaluation eligibility.
    """
    candidate_path = pin(candidate_dir / "challenge_ssl_curated_manifest.csv")
    receipt = json.loads(pin(candidate_dir / "challenge_ssl_curated_receipt.json").read_text())
    if sha256_file(candidate_path) != receipt["curated_manifest_sha256"]:
        raise ValueError("Curated candidate manifest mismatch")
    rows = read_csv(candidate_path)
    if not rows or len(rows) != receipt["counts"]["selected_total"]:
        raise ValueError("Candidate count mismatch")
    for row in rows:
        if (row["endpoint_supervised_eligible"] != "false" or row["label_scope"] != "ssl_only" or
                row["patient_independent_eval_eligible"] != "false"):
            raise ValueError("Candidate claims unsupported supervised/evaluation eligibility")
    return rows, receipt


def verify_candidate_views(rows: list[dict[str, str]], receipt: dict[str, Any], pin: PinInput) -> None:
    """
    Reconcile each candidate pointer with the published source view.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Candidate rows from :func:`load_candidates`.
    receipt : dict[str, Any]
        Curation receipt with the expected view manifest hashes.
    pin : PinInput
        Records each input file read.

    Raises
    ------
    ValueError
        If a view is unpublished or changed, or a candidate differs from it.
    """
    views = {}
    for view, receipt_key in (("strict_10s", "strict_manifest_sha256"),
                              ("cpsc_ssl_center_crop", "centered_manifest_sha256")):
        directory = CHALLENGE_VIEWS / view
        manifest_path = pin(directory / "manifest.csv")
        materialized = json.loads(pin(directory / "metadata.json").read_text())
        if (not materialized.get("complete") or sha256_file(manifest_path) != receipt[receipt_key] or
                sha256_file(manifest_path) != materialized["manifest_sha256"]):
            raise ValueError("Published Challenge view identity mismatch")
        views[view] = {row["ecg_id"]: row for row in read_csv(manifest_path)}
    for row in rows:
        original = views.get(row["view"], {}).get(row["ecg_id"])
        if original is None or any(row[key] != original[key] for key in VIEW_FIELDS):
            raise ValueError("Candidate differs from published waveform reference")
        expected_path = CHALLENGE_VIEWS / row["view"] / original["shard"]
        if (ROOT / row["shard_path"]).resolve() != expected_path.resolve():
            raise ValueError("Candidate shard path mismatch")


def load_ptb_reference(pin: PinInput) -> set[str]:
    """
    Load PTB-XL waveform hashes bound to a completed official-byte verification.

    Parameters
    ----------
    pin : PinInput
        Records each input file read.

    Returns
    -------
    set[str]
        Signal hashes of every PTB-XL record, including held-out folds.

    Raises
    ------
    ValueError
        If the reference is not the verified schema-2 snapshot.
    """
    reference_path = pin(PTB_REFERENCE)
    reference = json.loads(reference_path.read_text())
    pilot = json.loads(pin(PTB_PILOT_METADATA).read_text())
    if (reference.get("schema_version") != 2 or
            pilot["ptb_reference_receipt_sha256"] != sha256_file(reference_path) or
            reference["hashes_sha256"] != sha256_json(reference["hashes"])):
        raise ValueError("PTB reference lacks matching completed official-byte verification")
    return set(reference["hashes"])


def load_mimic_reference(pin: PinInput) -> tuple[set[str], set[str]]:
    """
    Read frozen MIMIC identities without opening or changing raw MIMIC records.

    Parameters
    ----------
    pin : PinInput
        Records each input file read.

    Returns
    -------
    tuple[set[str], set[str]]
        Accepted signal hashes and ECG IDs.

    Raises
    ------
    ValueError
        If the manifest, its closed SQLite audit or their identities disagree.
    """
    metadata = json.loads(pin(MIMIC_POOL / "metadata.json").read_text())
    manifest = pin(MIMIC_POOL / "ssl_manifest.csv")
    if sha256_file(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Frozen MIMIC manifest mismatch")
    rows = read_csv(manifest)
    database = pin(MIMIC_POOL / "audit.sqlite3")
    if Path(str(database) + "-wal").exists():
        raise ValueError("MIMIC audit is not a closed immutable SQLite snapshot")
    uri = database.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        state = dict(connection.execute("SELECT key,value FROM state"))
        accepted = dict(connection.execute("SELECT name,signal_sha256 FROM outcomes WHERE status='accepted'"))
    if (state.get("selection_sha256") != metadata["selection_sha256"] or
            set(accepted) != {r["filename_hr"] for r in rows} or
            len(accepted) != metadata["accepted_records"] or
            len(set(accepted.values())) != len(accepted) or
            any(not isinstance(h, str) or len(h) != SHA256_HEX_LENGTH for h in accepted.values())):
        raise ValueError("Frozen MIMIC audit identities do not match its accepted manifest")
    return set(accepted.values()), {row["ecg_id"] for row in rows}
