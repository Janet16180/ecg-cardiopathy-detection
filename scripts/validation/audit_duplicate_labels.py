#!/usr/bin/env python3
"""Compare diagnosis codes on exact duplicate Challenge waveforms, read-only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ecg_experiment.files import read_csv, write_csv_atomic
from ecg_experiment.public_sources import SOURCE, load_view, signal_sha256
from ecg_experiment.wfdb_records import header_label_codes

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("excluded_ecg_id", "retained_ecg_id", "excluded_label_codes", "retained_label_codes",
          "same_label_set")


def _duplicate_header(row: dict[str, str]) -> tuple[Path, Path]:
    """Find the one raw header of an excluded duplicate; return it and its release root."""
    project, version, prefix = SOURCE[row["source"]]
    raw = ROOT / "data/raw" / project / version
    record_id = row["ecg_id"].split(":", 1)[1]
    headers = list((raw / prefix).glob(f"*/{record_id}.hea"))
    if len(headers) != 1:
        raise ValueError(f"Expected exactly one header for {row['ecg_id']}: {headers}")
    return headers[0], raw


def compare_duplicate(row: dict[str, str],
                      retained: dict[str, dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Compare an excluded duplicate's annotation with its retained copy.

    Label sets are compared by plain comma splitting, so an empty annotation is
    the set ``{""}``. ``materialize_challenge_ecg`` drops empty codes instead;
    the difference is deliberate and kept.

    Parameters
    ----------
    row : dict[str, str]
        Exclusion row with reason ``exact_pool_duplicate``.
    retained : dict[str, dict[str, str]]
        Accepted manifest rows keyed by signal hash.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Comparison row with ``FIELDS`` columns, and the retained manifest row.

    Raises
    ------
    ValueError
        If the raw header is ambiguous or no retained copy has the same waveform.
    """
    header, raw = _duplicate_header(row)
    stem = str(header.relative_to(raw).with_suffix(""))
    signal, _, _, _ = load_view(raw, stem, "strict_10s")
    kept = retained.get(signal_sha256(signal))
    if kept is None:
        raise ValueError(f"Duplicate reference missing for {row['ecg_id']}")
    excluded_codes = header_label_codes(header)
    retained_codes = kept["label_codes"]
    # Unlike materialize_challenge_ecg, empty codes are kept: "" vs "1" is a different set.
    same = set(excluded_codes.split(",")) == set(retained_codes.split(","))
    comparison = {"excluded_ecg_id": row["ecg_id"], "retained_ecg_id": kept["ecg_id"],
                  "excluded_label_codes": excluded_codes, "retained_label_codes": retained_codes,
                  "same_label_set": str(same).lower()}
    return comparison, kept


def audit(prepared_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Compare annotations of every exact duplicate excluded by the strict audit.

    Parameters
    ----------
    prepared_dir : Path
        Published strict ``prepare_public_ecg`` directory under ``ROOT``.
    output_dir : Path
        Directory for the comparison table and summary.

    Returns
    -------
    dict[str, Any]
        Summary counts, also written as ``duplicate_label_summary.json``.

    Raises
    ------
    ValueError
        If the audit is not strict or a duplicate cannot be matched.
    """
    metadata = json.loads((prepared_dir / "metadata.json").read_text())
    if metadata["policy"] != "strict_10s":
        raise ValueError("This audit requires the strict ten-second manifest")
    retained = {row["signal_sha256"]: row for row in read_csv(prepared_dir / "manifest.csv")}
    duplicates = [row for row in read_csv(prepared_dir / "exclusions.csv")
                  if row["reason"] == "exact_pool_duplicate"]
    comparisons = []
    counts = Counter()
    for row in duplicates:
        comparison, kept = compare_duplicate(row, retained)
        comparisons.append(comparison)
        same = comparison["same_label_set"] == "true"
        counts["same_label_set" if same else "different_label_set"] += 1
        counts[f"{row['source']}_different_label_set"] += not same
        if row["source"] != kept["source"]:
            counts["cross_source_duplicate"] += 1
            counts["cross_source_label_conflict"] += not same
    write_csv_atomic(output_dir / "duplicate_label_comparisons.csv", comparisons, FIELDS)
    result = {"prepared_dir": str(prepared_dir.relative_to(ROOT)),
              "comparisons": len(comparisons), "counts": dict(counts),
              "interpretation": "Different original SNOMED sets on identical decoded waveforms; "
                                "no adjudication or label mapping performed"}
    (output_dir / "duplicate_label_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    """Run the audit from the command line and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/public_ecg_quality/strict_10s")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_quality")
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared_dir.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
