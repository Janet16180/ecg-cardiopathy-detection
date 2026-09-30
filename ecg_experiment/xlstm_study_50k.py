"""Bind the audited 25k result before the conditional CPC xLSTM 50k tier."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ecg_experiment import xlstm_study as base
from ecg_experiment import xlstm_study_v2 as version_two
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.paths import to_stored

PROTOCOL = "docs/experiment-038-cpc-xlstm-50k.md"
NEW_SOURCES = (
    PROTOCOL,
    "ecg_experiment/xlstm_study_50k.py",
    "scripts/experiments/run_cpc_xlstm038_50k.py",
)
PREDECESSOR_FILES = (
    "manifest.json", "result.json", "audit.json", "development_predictions.npz",
)


def protocol_commit(root: Path) -> str:
    """Require the frozen 50k protocol and return its stable commit.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    str
        Commit that last changed the 50k protocol.
    """
    committed = subprocess.run(
        ["git", "show", f"HEAD:{PROTOCOL}"], cwd=root, check=True, capture_output=True,
    ).stdout
    if committed != (root / PROTOCOL).read_bytes():
        raise ValueError("Experiment 038 50k protocol must be committed unchanged before scoring")
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", PROTOCOL],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def identity(root: Path, tier: int, cache_receipt: dict[str, Any]) -> dict[str, Any]:
    """Add the four audited 25k comparison inputs to the 50k manifest.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Required 50k cohort tier.
    cache_receipt : dict[str, Any]
        Fully verified 50k CPC cache receipt.

    Returns
    -------
    dict[str, Any]
        Versioned identity bound to the exact 25k result and predictions.
    """
    if tier != 50:
        raise ValueError("This successor is restricted to the conditional 50k tier")
    base.require_50k_trigger(root)
    version_two.protocol_commit(root)
    current = version_two.identity(root, tier, cache_receipt)
    predecessor = base.output(root, 25)
    current["audited_25k_sha256"] = {
        to_stored(predecessor / name): sha256_file(predecessor / name)
        for name in PREDECESSOR_FILES
    }
    audit = json.loads((predecessor / "audit.json").read_text())
    manifest = json.loads((predecessor / "manifest.json").read_text())
    result = json.loads((predecessor / "result.json").read_text())
    if sha256_json(manifest) != result["identity_sha256"]:
        raise ValueError("The 25k result no longer binds its input manifest")
    if audit["result_sha256"] != current["audited_25k_sha256"][
            to_stored(predecessor / "result.json")]:
        raise ValueError("The 25k audit no longer binds its result")
    return current


@contextmanager
def configured() -> Iterator[None]:
    """Compose the frozen v2 recipe with 50k-only provenance hooks.

    Yields
    ------
    None
        Frozen v2 helpers with the 50k input identity and protocol.
    """
    with version_two.configured():
        old = base.SOURCE_FILES, base.protocol_commit, base.identity
        base.SOURCE_FILES = (*old[0], *NEW_SOURCES)
        base.protocol_commit = protocol_commit
        base.identity = identity
        try:
            yield
        finally:
            base.SOURCE_FILES, base.protocol_commit, base.identity = old


def require_successful_train(root: Path) -> None:
    """Require a successful latest 50k train attempt within the tier ceiling.

    Parameters
    ----------
    root : Path
        Repository root containing the 50k stage ledger.
    """
    ledger_path = base.output(root, 50) / "stage_walltime.json"
    attempts = json.loads(ledger_path.read_text())["attempts"]
    training = [item for item in attempts if item["stage"] == "train"]
    if not training or training[-1]["status"] != "complete":
        raise ValueError("A successful latest 50k train stage is required before readout")
    if base.used_seconds(root, 50) >= base.CEILING_SECONDS:
        raise RuntimeError("The 50k tier exhausted its 7200-second work ceiling before readout")
