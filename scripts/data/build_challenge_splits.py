"""Publish the frozen record-level train, calibration and test split of the five Challenge sources.

Every record with a signal file gets a split. Labels come from ``challenge_labels``; exact duplicates are
resolved over all five sources with the Ningbo family rule, which must reproduce the clean Ningbo manifest
and its Chapman receipt. No waveform is read: signal hashes come from the clean Ningbo manifest and the
Challenge EDA hash cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecg_experiment import challenge_labels, challenge_splits, ningbo
from ecg_experiment.eda import challenge
from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
NINGBO = ROOT / "data/processed/ningbo_clean_v1"
RECEIPT = ROOT / "outputs/data_quality/ningbo_v1/exact_overlap_receipt.json"
OUTPUT = ROOT / "data/processed/challenge_splits_v1"
CODE = ("ecg_experiment/challenge_splits.py", "ecg_experiment/challenge_labels.py",
        "ecg_experiment/ningbo.py", "ecg_experiment/eda/challenge.py",
        "scripts/data/build_challenge_splits.py")
COLUMNS = ("source", "family", "record", "path", "duration_s", "primary", "secondary",
           *ningbo.LABEL_COLUMNS[2:], "signal_sha256", "window_sha256", "duplicate_group", "duplicate_status",
           "evaluable", "split")
STRATA = "source x primary label (positive, negative, undefined) of each duplicate group's lowest record"


def ningbo_rows() -> pd.DataFrame:
    """
    Rows of the clean Ningbo manifest, after checking it against its metadata.

    Returns
    -------
    pd.DataFrame
        Indexed by record name, with labels, sex, hashes and the manifest's duplicate status.

    Raises
    ------
    ValueError
        If ``rows.csv`` differs from its metadata.
    """
    metadata = json.loads((NINGBO / "metadata.json").read_text())
    if sha256_file(NINGBO / "rows.csv") != metadata["rows_sha256"]:
        raise ValueError("Ningbo manifest differs from its metadata")
    columns = ["record", "path", "male", *ningbo.LABEL_COLUMNS, "signal_sha256", "window_sha256",
               "duplicate_status"]
    rows = pd.read_csv(NINGBO / "rows.csv", usecols=columns,
                       dtype={"record": str, "path": str, "signal_sha256": str, "window_sha256": str,
                              "duplicate_status": str})
    rows = rows.set_index("record")
    rows["source"] = "ningbo"
    rows["duration_s"] = 10.0
    return rows.rename(columns={"duplicate_status": "manifest_duplicate_status"})


def other_rows(official: pd.DataFrame) -> pd.DataFrame:
    """
    Labels, sex and signal hashes of every Chapman, Georgia, CPSC 2018 and CPSC-Extra record.

    Parameters
    ----------
    official : pd.DataFrame
        ``challenge_labels.load_official`` output.

    Returns
    -------
    pd.DataFrame
        Indexed by record name.
    """
    headers = challenge.load_headers()
    hashes = challenge.signal_hashes(headers)
    labels = challenge_labels.label_table(headers["dx_codes"], official)
    rows = pd.DataFrame({"source": headers["source"], "path": headers["path"],
                         "duration_s": headers["duration_s"], "male": ningbo.male(headers["sex"]),
                         "signal_sha256": hashes, "window_sha256": ""}, index=headers.index).join(labels)
    return rows.set_index(headers["record"].rename("record"))


def check_family_rule(rows: pd.DataFrame, receipt: dict[str, object]) -> None:
    """
    Require the duplicate resolution to match the clean Ningbo manifest and the Chapman receipt.

    Parameters
    ----------
    rows : pd.DataFrame
        All records with ``source``, ``duplicate_status`` and, for Ningbo, ``manifest_duplicate_status``.
    receipt : dict[str, object]
        The Ningbo exact-overlap receipt.

    Raises
    ------
    ValueError
        If any status differs.
    """
    ningbo_part = rows[rows["source"] == "ningbo"]
    chapman = rows[rows["source"] == "chapman_shaoxing"]
    dropped = {status: sorted(chapman.index[chapman["duplicate_status"] == status])
               for status in ("dropped_copy", "dropped_conflict")}
    if not (ningbo_part["duplicate_status"] == ningbo_part["manifest_duplicate_status"]).all() or \
            dropped["dropped_copy"] != receipt["chapman_records_dropped_as_copy"] or \
            dropped["dropped_conflict"] != receipt["chapman_records_dropped_for_conflict"]:
        raise ValueError("Duplicate resolution differs from the clean Ningbo manifest or its receipt")


def build(output: Path, seed: int) -> dict[str, object]:
    """
    Resolve duplicates, assign splits and write ``rows.csv`` and ``metadata.json``.

    Parameters
    ----------
    output : Path
        New directory; it must not exist.
    seed : int
        Split seed.

    Returns
    -------
    dict[str, object]
        The written metadata.

    Raises
    ------
    ValueError
        If the output exists, a record name repeats, or a duplicate group crosses splits.
    """
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    official = challenge_labels.load_official()
    rows = pd.concat([ningbo_rows(), other_rows(official)])
    if rows.index.duplicated().any():
        raise ValueError("A record name appears twice")
    rows["family"] = rows["source"].map(challenge_splits.FAMILIES)
    rows["label_key"] = ningbo.label_key(rows)
    status = ningbo.duplicate_status(rows[["signal_sha256", "label_key", "male"]])
    rows["duplicate_status"] = status["duplicate_status"]
    check_family_rule(rows, json.loads(RECEIPT.read_text()))

    rows["duplicate_group"] = challenge_splits.duplicate_groups(rows[["signal_sha256", "window_sha256"]])
    if rows.groupby("duplicate_group")["signal_sha256"].nunique().max() > 1:
        raise ValueError("A window hash joins records with different signals")
    strata = rows["source"] + ":" + challenge_splits.label_category(rows["primary"])
    rows["split"] = challenge_splits.assign_splits(rows["duplicate_group"], strata, seed)
    if rows.groupby("duplicate_group")["split"].nunique().max() > 1:
        raise ValueError("A duplicate group crosses splits")
    rows["evaluable"] = rows["duplicate_status"].isin(["unique", "kept"]) & rows["primary"].notna()
    rows = rows.rename_axis("record").reset_index().sort_values(["source", "record"])[list(COLUMNS)]

    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    rows.to_csv(stage / "rows.csv", index=False)
    group_sizes = rows.groupby("duplicate_group").size()
    group_sources = rows.groupby("duplicate_group")["source"].nunique()
    metadata = {
        "complete": True, "seed": seed, "proportions": challenge_splits.PROPORTIONS,
        "strata": STRATA,
        "families": challenge_splits.FAMILIES, "records": len(rows),
        "duplicate_groups": {"groups": len(group_sizes), "multi_record_groups": int((group_sizes > 1).sum()),
                             "records_in_multi_record_groups": int(group_sizes[group_sizes > 1].sum()),
                             "cross_source_groups": int((group_sources > 1).sum())},
        "duplicate_status": rows.groupby("source")["duplicate_status"].value_counts().unstack(fill_value=0)
        .to_dict(orient="index"),
        "counts_all": challenge_splits.split_counts(rows),
        "counts_evaluable": challenge_splits.split_counts(rows[rows["evaluable"]]),
        "inputs_sha256": {
            "ningbo_rows_csv": sha256_file(NINGBO / "rows.csv"),
            "ningbo_metadata_json": sha256_file(NINGBO / "metadata.json"),
            "ningbo_overlap_receipt": sha256_file(RECEIPT),
            "challenge_headers_parquet": sha256_file(challenge.HEADER_CACHE),
            "challenge_hashes_parquet": sha256_file(challenge.HASH_CACHE),
            **challenge_labels.MAPPING_SHA256,
        },
        "rows_sha256": sha256_file(stage / "rows.csv"),
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def main() -> None:
    """Parse arguments and build the split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=challenge_splits.SEED)
    args = parser.parse_args()
    metadata = build(args.output_dir, args.seed)
    print(json.dumps({key: metadata[key] for key in ("records", "duplicate_groups", "counts_evaluable")},
                     indent=1))


if __name__ == "__main__":
    main()
