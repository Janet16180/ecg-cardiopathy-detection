"""Compare PR 74 split identities with the restored original Challenge assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
SOURCES = {"CPSC 2018": "cpsc_2018", "CPSC 2018 extra": "cpsc_2018_extra",
           "Ningbo": "ningbo", "Chapman-Shaoxing": "chapman_shaoxing", "Georgia": "georgia"}


def audit(pr_root: Path, original_split: Path) -> dict[str, object]:
    """Compare identifiers and split assignments without reading diagnosis or score columns.

    Parameters
    ----------
    pr_root : Path
        Checkout of the reviewed PR head, containing its committed partition table.
    original_split : Path
        Byte-verified original Challenge rows table.

    Returns
    -------
    dict[str, object]
        Aggregate assignment counts and source hashes, with no individual identifiers.
    """
    digest = hashlib.sha256(original_split.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError("Original Challenge split is not reproduced")
    original = pd.read_csv(original_split, usecols=["source", "record", "split"], dtype=str)
    pr_path = pr_root / "data/processed/balanceado_jovenes_v2/particion.csv"
    pr = pd.read_csv(pr_path, usecols=["registro", "dataset", "split"], dtype=str)
    pr = pr[pr.dataset.isin(SOURCES)].copy()
    pr["source"] = pr.dataset.map(SOURCES)
    pr = pr.rename(columns={"registro": "record", "split": "pr_split"})
    if pr.duplicated(["source", "record"]).any():
        raise ValueError("Duplicate PR record identity")
    matched = pr.merge(original.rename(columns={"split": "original_split"}),
                       on=["source", "record"], how="left", validate="one_to_one")
    if matched.original_split.isna().any():
        raise ValueError("PR record identities cannot all be mapped to original split")
    counts = matched.groupby(["source", "original_split", "pr_split"]).size().rename("records")
    totals = matched.groupby(["original_split", "pr_split"]).size().rename("records")
    return {"scope": "assignment_audit_only_no_diagnoses_predictions_or_waveforms",
            "original_split_sha256": digest,
            "pr_partition_sha256": hashlib.sha256(pr_path.read_bytes()).hexdigest(),
            "matched_challenge_records": len(matched),
            "counts_by_source": counts.reset_index().to_dict("records"),
            "totals": totals.reset_index().to_dict("records")}


def main() -> None:
    """Print aggregate assignment overlap for an exact restored split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-root", type=Path, required=True)
    parser.add_argument("--original-split", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.pr_root, args.original_split), indent=2))


if __name__ == "__main__":
    main()
