#!/usr/bin/env python3
"""Read-only exact-identity gate for appending Challenge candidates to frozen SSL data.

The novel output is a candidate manifest, not a scheduled union or a patient split.
MIMIC hashes come from its frozen preparation audit; raw MIMIC is not redecoded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ecg_experiment import pool_overlap
from ecg_experiment.downloads import parse_checksums
from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic
from ecg_experiment.pool_overlap import (
    PinInput,
    load_candidates,
    load_mimic_reference,
    load_ptb_reference,
    verify_candidate_views,
)
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.staging import published_directory
from ecg_experiment.waveforms import read_record

ROOT = Path(__file__).resolve().parents[2]
GEORGIA_POOL = ROOT / "data/processed/georgia_ssl_g1"
CHALLENGE_RAW = ROOT / "data/raw/challenge-2020/1.0.2"
OVERLAP_FIELDS = (
    "ecg_id", "source", "signal_sha256", "exact_signal_reference_pools",
    "existing_record_id",
)


def classify_overlap(rows: list[dict[str, str]], references: dict[str, set[str]],
                     existing_ids: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Split candidates into novel records and exact overlaps with reference pools.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Candidate rows with ``ecg_id``, ``source`` and ``signal_sha256``.
    references : dict[str, set[str]]
        Signal hashes of each reference pool keyed by pool name.
    existing_ids : set[str]
        Record IDs already used by a frozen pool.

    Returns
    -------
    tuple[list[dict[str, str]], list[dict[str, str]]]
        Novel candidate rows and overlap rows in ``OVERLAP_FIELDS`` form.

    Raises
    ------
    ValueError
        If the candidate repeats an ECG ID or signal hash.
    """
    if len({r["ecg_id"] for r in rows}) != len(rows) or len({r["signal_sha256"] for r in rows}) != len(rows):
        raise ValueError("Candidate contains duplicate identity")
    novel, overlaps = [], []
    for row in rows:
        matches = sorted(name for name, hashes in references.items() if row["signal_sha256"] in hashes)
        same_id = row["ecg_id"] in existing_ids
        if matches or same_id:
            overlaps.append({"ecg_id": row["ecg_id"], "source": row["source"],
                             "signal_sha256": row["signal_sha256"],
                             "exact_signal_reference_pools": ";".join(matches),
                             "existing_record_id": str(same_id).lower()})
        else:
            novel.append(row)
    return novel, overlaps


def load_georgia_reference(pin: PinInput) -> tuple[set[str], set[str]]:
    """
    Verify official raw bytes and decode the historical Georgia pilot.

    Parameters
    ----------
    pin : PinInput
        Records each input file read.

    Returns
    -------
    tuple[set[str], set[str]]
        Decoded signal hashes and ECG IDs of the frozen Georgia pool.

    Raises
    ------
    ValueError
        If the manifest, its count, or any official raw checksum differs.
    """
    metadata = json.loads(pin(GEORGIA_POOL / "metadata.json").read_text())
    manifest = pin(GEORGIA_POOL / "ssl_manifest.csv")
    if sha256_file(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Frozen Georgia manifest mismatch")
    rows = read_csv(manifest)
    if len(rows) != metadata["accepted_records"]:
        raise ValueError("Frozen Georgia count mismatch")
    sums_file = pin(CHALLENGE_RAW / "SHA256SUMS.txt")
    if sha256_file(sums_file) != metadata["checksums_sha256"]:
        raise ValueError("Georgia official checksum reference mismatch")
    sums = parse_checksums(sums_file.read_text())
    raw = CHALLENGE_RAW.resolve()
    hashes = set()
    for row in rows:
        stem = row["filename_hr"]
        if Path(row["raw_dir"]).resolve() != raw or not stem.startswith("training/georgia/g1/"):
            raise ValueError("Georgia raw source mismatch")
        for suffix in (".hea", ".mat"):
            name = stem + suffix
            path = (CHALLENGE_RAW / name).resolve()
            if not path.is_relative_to(raw) or sha256_file(path) != sums.get(name):
                raise ValueError("Georgia official raw checksum mismatch")
        hashes.add(signal_sha256(read_record(CHALLENGE_RAW, stem)))
    return hashes, {row["ecg_id"] for row in rows}


def overlap_receipt(rows: list[dict[str, str]], novel: list[dict[str, str]],
                    overlap: list[dict[str, str]], pools: dict[str, set[str]]) -> dict[str, Any]:
    """
    Summarize the overlap classification.

    Parameters
    ----------
    rows : list[dict[str, str]]
        All candidate rows.
    novel, overlap : list[dict[str, str]]
        Output of :func:`classify_overlap`.
    pools : dict[str, set[str]]
        Reference pool hashes keyed by name.

    Returns
    -------
    dict[str, Any]
        Receipt fields describing counts and the append gate.
    """
    return {
        "status": "exact_overlap_audited_candidate_only_not_scheduled",
        "candidate_records": len(rows),
        "overlapping_records": len(overlap),
        "novel_records": len(novel),
        "unfiltered_candidate_append_gate_passed": not overlap,
        "reference_unique_hashes": {pool: len(hashes) for pool, hashes in pools.items()},
        "overlap_by_reference_pool": {
            pool: sum(pool in row["exact_signal_reference_pools"].split(";") for row in overlap)
            for pool in pools
        },
        "overlap_by_record_id": sum(row["existing_record_id"] == "true" for row in overlap),
        "novel_source_counts": dict(Counter(row["source"] for row in novel)),
    }


def publish_audit(output_dir: Path, rows: list[dict[str, str]], novel: list[dict[str, str]],
                  overlap: list[dict[str, str]], pools: dict[str, set[str]],
                  inputs: dict[str, str]) -> dict[str, Any]:
    """
    Publish both manifests and their receipt as one immutable directory.

    Parameters
    ----------
    output_dir : Path
        New directory to publish.
    rows : list[dict[str, str]]
        All candidate rows; their columns define the novel manifest.
    novel, overlap : list[dict[str, str]]
        Output of :func:`classify_overlap`.
    pools : dict[str, set[str]]
        Reference pool hashes keyed by name.
    inputs : dict[str, str]
        SHA-256 of every pinned input keyed by resolved path.

    Returns
    -------
    dict[str, Any]
        The published receipt.
    """
    with published_directory(output_dir) as stage:
        write_csv_atomic(stage / "overlaps.csv", overlap, OVERLAP_FIELDS)
        write_csv_atomic(stage / "novel_challenge_ssl_manifest.csv", novel, tuple(rows[0]))
        result = {
            **overlap_receipt(rows, novel, overlap, pools),
            "input_sha256": inputs,
            "source_sha256": sha256_file(Path(__file__)),
            "library_source_sha256": sha256_file(Path(pool_overlap.__file__)),
            "output_sha256": {
                name: sha256_file(stage / name)
                for name in ("overlaps.csv", "novel_challenge_ssl_manifest.csv")
            },
            "limitations": [
                "MIMIC identities rely on its frozen preparation audit; raw MIMIC was not redecoded.",
                "PTB identities refer to the completed official-checksum-verified v2 preparation snapshot.",
                ("Exact identities do not detect shifted/resampled/partial near-duplicates or "
                 "unknown patient overlap."),
                "No labels mapped, patient split assigned, frozen pool replaced or training scheduled.",
            ],
        }
        (stage / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit(candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Validate inputs, classify overlap, then publish without changing any input.

    Parameters
    ----------
    candidate_dir : Path
        Directory with the curated Challenge candidate.
    output_dir : Path
        New directory to publish, outside ``data/raw``.

    Returns
    -------
    dict[str, Any]
        The published receipt.

    Raises
    ------
    FileExistsError
        If the output already exists.
    ValueError
        If the output is inside raw data or any input check fails or changes.
    """
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if output_dir.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Output must be outside raw data")

    inputs: dict[str, str] = {}

    def pin(path: Path) -> Path:
        inputs[str(path.resolve())] = sha256_file(path)
        return path

    rows, receipt = load_candidates(candidate_dir, pin)
    verify_candidate_views(rows, receipt, pin)
    pools = {"ptbxl_all_folds": load_ptb_reference(pin)}
    pools["mimic_frozen_40k"], mimic_ids = load_mimic_reference(pin)
    pools["georgia_frozen_g1"], georgia_ids = load_georgia_reference(pin)
    novel, overlap = classify_overlap(rows, pools, mimic_ids | georgia_ids)

    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Input changed during audit: {path}")
    return publish_audit(output_dir, rows, novel, overlap, pools, inputs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir", type=Path, default=ROOT / "outputs/data_quality/processed_eda",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Run the overlap audit and print its receipt.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    print(json.dumps(audit(args.candidate_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
