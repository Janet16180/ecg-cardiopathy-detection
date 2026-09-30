"""Versioned Experiment 038 runner using a checkpointable native GRU backend."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ecg_experiment import xlstm_study as base
from ecg_experiment.cpc_xlstm_native_gru import create_model as native_create_model
from ecg_experiment.files import sha256_file, write_json_atomic

OUTPUT_NAME = "experiment038_cpc_xlstm_v2"
PROTOCOL = "docs/experiment-038-cpc-xlstm-v2.md"
EXPECTED_V1_SECONDS = 118.02747250895482
NEW_SOURCES = (
    PROTOCOL,
    "ecg_experiment/cpc_xlstm_native_gru.py",
    "ecg_experiment/xlstm_study_v2.py",
    "scripts/experiments/run_cpc_xlstm038_v2.py",
    "ecg_experiment/gru_replay_diagnostic.py",
)
_BASE_IDENTITY = base.identity


def protocol_commit(root: Path) -> str:
    """Require the committed v2 protocol and return its stable origin commit.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    str
        Commit that last changed the v2 protocol.
    """
    committed = subprocess.run(
        ["git", "show", f"HEAD:{PROTOCOL}"], cwd=root, check=True, capture_output=True
    ).stdout
    if committed != (root / PROTOCOL).read_bytes():
        raise ValueError("Experiment 038 v2 protocol must be committed unchanged before scoring")
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", PROTOCOL],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def identity(root: Path, tier: int, cache_receipt: dict[str, Any]) -> dict[str, Any]:
    """Extend the frozen v1 input identity with v1 failure provenance.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    cache_receipt : dict[str, Any]
        Fully verified CPC signal cache receipt.

    Returns
    -------
    dict[str, Any]
        Versioned execution identity with the predecessor failure bound.
    """
    current = _BASE_IDENTITY(root, tier, cache_receipt)
    provenance = root / "outputs" / OUTPUT_NAME / "25k/predecessor_v1.json"
    if not provenance.exists():
        raise ValueError("Experiment 038 v1 failure provenance is required")
    current["predecessor_v1_sha256"] = sha256_file(provenance)
    current["gru_backend"] = "native_pytorch_gru_cudnn_disabled_in_forward"
    return current


@contextmanager
def configured() -> Iterator[None]:
    """Temporarily bind v2 model, source, protocol, identity and output hooks.

    Yields
    ------
    None
        Frozen v1 study helpers configured for the v2 execution identity.
    """
    old = (base.create_model, base.SOURCE_FILES, base.OUTPUT_NAME,
           base.protocol_commit, base.identity)
    base.create_model = native_create_model
    base.SOURCE_FILES = (*old[1], *NEW_SOURCES)
    base.OUTPUT_NAME = OUTPUT_NAME
    base.protocol_commit = protocol_commit
    base.identity = identity
    try:
        yield
    finally:
        (base.create_model, base.SOURCE_FILES, base.OUTPUT_NAME,
         base.protocol_commit, base.identity) = old


def carry_v1_attempts(root: Path, tier: int) -> dict[str, Any] | None:
    """Charge v1's failed 25k work once while pinning the failure artifacts.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.

    Returns
    -------
    dict[str, Any] | None
        Pinned failure provenance for 25k, or None for 50k.
    """
    if tier != 25:
        return None
    original = root / "outputs/experiment038_cpc_xlstm"
    previous_ledger = original / "25k/stage_walltime.json"
    recorded = json.loads(previous_ledger.read_text())
    attempts = recorded["attempts"]
    seconds = sum(item["elapsed_seconds"] for item in attempts)
    if (not math.isclose(seconds, EXPECTED_V1_SECONDS, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(seconds, recorded["total_seconds"], rel_tol=0, abs_tol=1e-9)
            or attempts[-1]["stage"] != "profile" or attempts[-1]["status"] != "failed"):
        raise ValueError("The recorded v1 failure and elapsed work changed")
    names = (
        "25k/stage_walltime.json", "25k/manifest.json", "25k/prior_integrity.json",
        "25k/profile_resume_gru.pt", "profile25k.log", "prepare25k.log",
        "cache25k-build.json", "cache25k-launch.json", "predecessor-replay.json",
    )
    hashes = {f"outputs/experiment038_cpc_xlstm/{name}": sha256_file(original / name)
              for name in names}
    result_exists = (original / "25k/result.json").exists()
    if result_exists:
        raise ValueError("v1 unexpectedly has a completed development result")
    provenance = {
        "status": "stopped_before_new_scores_after_gru_checkpoint_replay_failure",
        "v1_attempt_seconds": seconds,
        "v1_profile_error": "Exact checkpoint replay failed for gru",
        "v1_result_exists": False,
        "files_sha256": hashes,
    }
    destination = root / "outputs" / OUTPUT_NAME / "25k"
    provenance_path = destination / "predecessor_v1.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Experiment 038 v1 provenance changed")
    else:
        write_json_atomic(provenance_path, provenance, sort_keys=True)
    ledger = destination / "stage_walltime.json"
    copied = [{**item, "predecessor_version": "v1", "cache_build_included": False}
              for item in attempts]
    if ledger.exists():
        saved = json.loads(ledger.read_text())["attempts"]
        if saved[:len(copied)] != copied:
            raise ValueError("Experiment 038 v1 attempted work was not carried exactly once")
    else:
        write_json_atomic(ledger, {"attempts": copied, "total_seconds": seconds}, sort_keys=True)
    return provenance
