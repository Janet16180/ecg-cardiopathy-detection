"""Describe the duplicate groups of the MLC-01 dataset: which labels conflict, and how SPH copies relate.

Writes ``<dataset>/supplementary/duplicate_conflicts.csv`` (one row per group dropped for conflicting labels)
and ``<dataset>/supplementary/duplicates.json`` (counts, SPH copy patients, and agreement with the duplicate
statuses of the earlier SPH baseline table). Run from the repository root with ``PYTHONPATH=.``:
``.venv/bin/python -m scripts.reports.describe_mlc01_duplicates``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecg_experiment import ROOT, mlc01_dataset
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc01_dataset import SPH_ROWS

DATASET = ROOT / "outputs/mlc01_condition_dataset_20261110"
CPSC_PARTIAL = frozenset({"a_STTC", "a_CD", "b_af_or_flutter"})


def conflict_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Labels on which the copies of each conflicting group disagree.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.

    Returns
    -------
    pd.DataFrame
        Group, sources, roles, size, conflicting labels and ``only_cpsc_2018_partial`` (every conflicting
        label is one that CPSC 2018 annotates only in part).
    """
    labels = mlc01_dataset.label_columns(manifest)
    groups = manifest[manifest["duplicate_status"] == "dropped_conflict"]["duplicate_group"].unique()
    rows = []
    for group in groups:
        members = manifest[manifest["duplicate_group"] == group]
        values = members[labels].astype(float)
        conflicting = [name for name in labels if (values[name] == 1).any() and (values[name] == 0).any()]
        partial = "cpsc_2018" in set(members["source"]) and set(conflicting) <= CPSC_PARTIAL
        rows.append({"duplicate_group": group, "sources": "+".join(sorted(set(members["source"]))),
                     "roles": "+".join(sorted(set(members["role"]))), "size": len(members),
                     "conflicting_labels": ";".join(conflicting), "only_cpsc_2018_partial": partial})
    return pd.DataFrame(rows)


def sph_summary(manifest: pd.DataFrame) -> dict[str, object]:
    """
    Patients behind the SPH duplicate groups and agreement with the earlier SPH duplicate statuses.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.

    Returns
    -------
    dict[str, object]
        Group sizes, patients per group, groups whose raw codes differ, and a status cross-table over the
        records of ``SPH_ROWS``.
    """
    sph = manifest[manifest["source"] == "sph"]
    copies = sph[sph["duplicate_size"].fillna(0) > 1]
    groups = copies.groupby("duplicate_group").agg(patients=("group_id", "nunique"),
                                                   size=("group_id", "size"), codes=("raw_codes", "nunique"))
    earlier = pd.read_csv(SPH_ROWS, usecols=["ecg_id", "duplicate_status"], dtype=str).set_index("ecg_id")
    joined = earlier.join(sph.set_index("record")["duplicate_status"].rename("mlc01"))
    table = pd.crosstab(joined["duplicate_status"], joined["mlc01"])
    return {"groups": len(groups), "group_sizes": groups["size"].value_counts().to_dict(),
            "patients_per_group": groups["patients"].value_counts().to_dict(),
            "groups_with_different_raw_codes": int((groups["codes"] > 1).sum()),
            "earlier_table": str(SPH_ROWS.relative_to(ROOT)), "earlier_table_records": len(earlier),
            "status_crosstab_earlier_by_mlc01": {row: {col: int(n) for col, n in values.items()}
                                                 for row, values in table.to_dict(orient="index").items()}}


def cross_source_records(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Count groups and records per source in the duplicate groups that span two sources.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.

    Returns
    -------
    dict[str, dict[str, int]]
        Keyed by the group's sources joined with ``+``: ``groups`` and the record count of each source.
    """
    copies = manifest[manifest["duplicate_size"].fillna(0) > 1]
    sources = copies.groupby("duplicate_group")["source"].agg(lambda names: "+".join(sorted(set(names))))
    found: dict[str, dict[str, int]] = {}
    for combination, members in copies.groupby(copies["duplicate_group"].map(sources)):
        if "+" in combination:
            found[combination] = {"groups": int(members["duplicate_group"].nunique()),
                                  **{name: int(n) for name, n in members["source"].value_counts().items()}}
    return found


def main() -> None:
    """Write the duplicate description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    manifest = pd.read_csv(args.dataset / "manifest.csv", index_col="record_key", low_memory=False,
                           dtype={"record": str, "raw_codes": str})
    manifest["duplicate_status"] = manifest["duplicate_status"].fillna("")
    out = args.dataset / "supplementary"
    out.mkdir(exist_ok=True)
    conflicts = conflict_table(manifest)
    conflicts.to_csv(out / "duplicate_conflicts.csv", index=False)
    by_sources = conflicts.groupby("sources").agg(groups=("size", "size"), records=("size", "sum"),
                                                  only_cpsc_2018_partial=("only_cpsc_2018_partial", "sum"))
    labels = conflicts["conflicting_labels"].str.split(";").explode().value_counts()
    summary = {"conflict_groups_by_sources": {key: {name: int(value) for name, value in row.items()}
                                              for key, row in by_sources.to_dict(orient="index").items()},
               "conflict_groups_per_label": {name: int(n) for name, n in labels.items()},
               "records_per_source_in_cross_source_groups": cross_source_records(manifest),
               "sph": sph_summary(manifest),
               "manifest_sha256": sha256_file(args.dataset / "manifest.csv"),
               "script_sha256": sha256_file(Path(__file__)),
               "duplicate_conflicts_sha256": sha256_file(out / "duplicate_conflicts.csv")}
    write_json_atomic(out / "duplicates.json", summary, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
