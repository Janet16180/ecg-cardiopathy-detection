#!/usr/bin/env python3
"""Compare diagnosis codes on exact duplicate Challenge waveforms, read-only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.data.prepare_public_ecg import ROOT, SOURCE, load_view, signal_sha256


FIELDS = ("excluded_ecg_id", "retained_ecg_id", "excluded_label_codes", "retained_label_codes",
          "same_label_set")


def labels(header: Path) -> str:
    return next((line.split(":", 1)[1].strip() for line in header.read_text().splitlines()
                 if line.startswith("# Dx:")), "")


def audit(prepared_dir: Path, output_dir: Path) -> dict:
    metadata = json.loads((prepared_dir / "metadata.json").read_text())
    if metadata["policy"] != "strict_10s":
        raise ValueError("This audit requires the strict ten-second manifest")
    with (prepared_dir / "manifest.csv").open(newline="") as handle:
        retained = {row["signal_sha256"]: row for row in csv.DictReader(handle)}
    with (prepared_dir / "exclusions.csv").open(newline="") as handle:
        duplicates = [row for row in csv.DictReader(handle)
                      if row["reason"] == "exact_pool_duplicate"]
    comparisons = []
    counts = Counter()
    for row in duplicates:
        source = row["source"]
        project, version, prefix = SOURCE[source]
        raw = ROOT / "data/raw" / project / version
        record_id = row["ecg_id"].split(":", 1)[1]
        headers = list((raw / prefix).glob(f"*/{record_id}.hea"))
        if len(headers) != 1:
            raise ValueError(f"Expected exactly one header for {row['ecg_id']}: {headers}")
        stem = str(headers[0].relative_to(raw).with_suffix(""))
        signal, _, _, _ = load_view(raw, stem, "strict_10s")
        kept = retained.get(signal_sha256(signal))
        if kept is None:
            raise ValueError(f"Duplicate reference missing for {row['ecg_id']}")
        excluded_codes = labels(headers[0])
        retained_codes = kept["label_codes"]
        same = set(excluded_codes.split(",")) == set(retained_codes.split(","))
        comparisons.append({"excluded_ecg_id": row["ecg_id"],
                            "retained_ecg_id": kept["ecg_id"],
                            "excluded_label_codes": excluded_codes,
                            "retained_label_codes": retained_codes,
                            "same_label_set": str(same).lower()})
        counts["same_label_set" if same else "different_label_set"] += 1
        counts[f"{source}_different_label_set"] += not same
        if source != kept["source"]:
            counts["cross_source_duplicate"] += 1
            counts["cross_source_label_conflict"] += not same
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = output_dir / "duplicate_label_comparisons.csv"
    with detail.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(comparisons)
    result = {"prepared_dir": str(prepared_dir.relative_to(ROOT)),
              "comparisons": len(comparisons), "counts": dict(counts),
              "interpretation": "Different original SNOMED sets on identical decoded waveforms; no adjudication or label mapping performed"}
    (output_dir / "duplicate_label_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/public_ecg_quality/strict_10s")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_quality")
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared_dir.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
