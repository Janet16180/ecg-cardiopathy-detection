"""Run the committed NTT-01 public annotation audit without model or waveform scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.reference_label_audit import audit_local_labels, audit_reasoning, identifier_table

UPSTREAM_COMMIT = "90a53f107023c7c196cbf08de183b871b2a5798c"
UPSTREAM_BLOB = "da62062c76828d53297e76f00ff02153e5713246"
SOURCE_FILES = (
    "ecg_experiment/reference_label_audit.py",
    "scripts/reports/audit_reference_labels.py",
    "tests/test_reference_label_audit.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "docs/ntt-01-reference-label-audit.md",
    "docs/ntt-01-amendment-safe-grounding.md",
)


def frozen_sources() -> dict[str, str]:
    """Verify every declared source against Git before reading audit inputs.

    Returns
    -------
    dict
        Repository-relative source paths and their SHA-256 values.
    """
    hashes = {}
    for name in SOURCE_FILES:
        committed = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, check=True, capture_output=True)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"Uncommitted source: {name}")
        hashes[name] = digest
    return hashes


def run_audit(annotations: Path) -> dict[str, Any]:
    """Bind inputs and return one completed aggregate audit.

    Parameters
    ----------
    annotations : Path
        Frozen public PTB annotation file.

    Returns
    -------
    dict
        Aggregate findings and provenance without patient identifiers.
    """
    started = time.perf_counter()
    source_hashes = frozen_sources()
    raw = annotations.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != UPSTREAM_BLOB:
        raise ValueError("Public annotations differ from the frozen upstream blob")
    metadata = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    statements = metadata.with_name("scp_statements.csv")
    inputs = {to_stored(p): sha256_file(p) for p in (annotations, metadata, statements)}
    labels = audit_local_labels(metadata, statements)
    reasoning = audit_reasoning(annotations, identifier_table(metadata))
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_information_audit",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_git_blob": blob,
        "source_sha256": source_hashes,
        "input_sha256": inputs,
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "wfdb")},
        "local_labels": labels,
        "reasoning_annotations": reasoning,
        "clinical_or_model_score_computed": False,
        "waveforms_read": False,
        "closed_or_spent_evaluation_content_interpreted": False,
        "historical_localization_metrics_reproduced": False,
        "historical_reproduction_limit": "Original ECGAD marks and score tables are unavailable here",
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    """Execute one source-bound run and preserve completion or failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data/raw/ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ntt01_reference_label_audit")
    args = parser.parse_args()
    command = [
        "python",
        "-m",
        "scripts.reports.audit_reference_labels",
        "--annotations",
        to_stored(args.annotations),
        "--output",
        to_stored(args.output),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / "status.json"
    status = {
        "status": "running",
        "command": command,
        "thread_environment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        result = run_audit(args.annotations)
        result.update(command=command, thread_environment=status["thread_environment"])
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise
    status.update(status="complete", result_sha256=sha256_file(args.output / "result.json"))
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": to_stored(args.output),
                "seconds": result["seconds"],
                "safe_cases": result["reasoning_annotations"]["safe_cases"],
                "safe_ecgs": result["reasoning_annotations"]["safe_unique_ecgs"],
            }
        )
    )


if __name__ == "__main__":
    main()
