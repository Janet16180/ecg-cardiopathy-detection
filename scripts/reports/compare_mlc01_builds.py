"""Compare two MLC-01 builds: label counts per evaluation set, state transitions and record-set changes.

Writes ``<after>/comparison_with_before/``: ``before_after_<space>.csv`` (positives, negatives and undefined
per label, role and unit, with the support flags of both builds), ``state_transitions.csv`` (records included
in both builds whose state changed) and ``record_set_changes.csv``. Run from the repository root with
``PYTHONPATH=.``: ``.venv/bin/python -m scripts.reports.compare_mlc01_builds --before <dir> --after <dir>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecg_experiment import mlc01_dataset, mlc_labels
from ecg_experiment.files import sha256_file, write_json_atomic

SPACES = ("A", "B", "B_lik50", "C")
KEYS = ["space", "label", "role", "level", "unit"]


def support_change(before: Path, after: Path, space: str) -> pd.DataFrame:
    """
    Join the support tables of two builds.

    Parameters
    ----------
    before, after : Path
        Build directories.
    space : str
        ``A``, ``B``, ``B_lik50`` or ``C``.

    Returns
    -------
    pd.DataFrame
        One row per label, role and unit with ``_before`` and ``_after`` counts and flags, and
        ``negatives_below_20``.
    """
    tables = []
    for name, directory in (("before", before), ("after", after)):
        table = pd.read_csv(directory / f"support_{space}.csv")
        table["negatives_below_20"] = table["negative"] < 20
        columns = ["records", "positive", "negative", "undefined", "below_20", "below_50", "below_100",
                   "negatives_below_20", "enters_macro"]
        tables.append(table.set_index(KEYS)[columns].add_suffix(f"_{name}"))
    return tables[0].join(tables[1], how="outer").reset_index()


def manifest(directory: Path) -> pd.DataFrame:
    """
    Identity, inclusion and label columns of a build's manifest.

    Parameters
    ----------
    directory : Path
        Build directory.

    Returns
    -------
    pd.DataFrame
        Indexed by record key.
    """
    return pd.read_csv(directory / "manifest.csv", index_col="record_key", low_memory=False,
                       dtype={"record": str, "raw_codes": str})


def transitions(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """
    Count state changes of records included in both builds.

    Parameters
    ----------
    before, after : pd.DataFrame
        Outputs of ``manifest``.

    Returns
    -------
    pd.DataFrame
        ``label``, ``source``, ``role``, ``before``, ``after`` and ``records``.
    """
    keys = before.index[before["included"]].intersection(after.index[after["included"]])
    rows = []
    for column in mlc01_dataset.label_columns(after):
        old = before.loc[keys, column].astype(float)
        new = after.loc[keys, column].astype(float)
        changed = keys[~((old == new) | (old.isna() & new.isna())).to_numpy()]
        if len(changed):
            part = pd.DataFrame({"source": after.loc[changed, "source"], "role": after.loc[changed, "role"],
                                 "before": old[changed].map(mlc_labels.state_name),
                                 "after": new[changed].map(mlc_labels.state_name)})
            counts = part.groupby(["source", "role", "before", "after"]).size().rename("records")
            counts = counts.reset_index()
            rows.append(counts.assign(label=column))
    return pd.concat(rows, ignore_index=True)[["label", "source", "role", "before", "after", "records"]]


def record_set_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """
    Count records included in only one of the two builds, per source, role and duplicate status.

    Parameters
    ----------
    before, after : pd.DataFrame
        Outputs of ``manifest``.

    Returns
    -------
    pd.DataFrame
        ``source``, ``role``, ``change``, ``duplicate_status`` (in the build that excludes the record) and
        ``records``.
    """
    columns = ["source", "role", "duplicate_status"]
    later = after["included"].reindex(before.index).astype(bool)
    earlier = before["included"].reindex(after.index).astype(bool)
    only_before = before.index[before["included"] & ~later]
    only_after = after.index[after["included"] & ~earlier]
    parts = [after.loc[only_before, columns].assign(change="included only before"),
             before.loc[only_after, columns].assign(change="included only after")]
    table = pd.concat(parts)
    counts = table.groupby(["source", "role", "change", "duplicate_status"]).size()
    return counts.rename("records").reset_index()


def main() -> None:
    """Write the comparison tables and their receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    out = args.after / "comparison_with_before"
    out.mkdir(exist_ok=False)
    for space in SPACES:
        support_change(args.before, args.after, space).to_csv(out / f"before_after_{space}.csv", index=False)
    before, after = manifest(args.before), manifest(args.after)
    transitions(before, after).to_csv(out / "state_transitions.csv", index=False)
    changes = record_set_changes(before, after)
    changes.to_csv(out / "record_set_changes.csv", index=False)
    names = [f"before_after_{space}.csv" for space in SPACES]
    names += ["state_transitions.csv", "record_set_changes.csv"]
    receipt = {"before_manifest_sha256": sha256_file(args.before / "manifest.csv"),
               "after_manifest_sha256": sha256_file(args.after / "manifest.csv"),
               "script_sha256": sha256_file(Path(__file__)),
               "outputs_sha256": {name: sha256_file(out / name) for name in names},
               "included": {"before": int(before["included"].sum()), "after": int(after["included"].sum())}}
    write_json_atomic(out / "receipt.json", receipt, sort_keys=True)
    print(json.dumps(receipt["included"]))
    print(changes.to_string(index=False))


if __name__ == "__main__":
    main()
