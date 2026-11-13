"""Compare the MLC-01 dataset with the metadata inventory that preceded its protocol; explain each difference.

The inventory directory is not part of the repository. Its aggregate tables and draft vocabulary are copied
into ``<dataset>/inventory_comparison/inventory_snapshot`` with their SHA-256. The comparison has three steps,
each on named records:

1. The draft mapping is re-applied to the inventory's record sets and must reproduce its counts exactly.
   For the Challenge groups those sets are the split's ``unique`` and ``kept`` records, rebuilt here with the
   frozen Ningbo-family rule from the whole-record hashes (the split's ``signal_sha256`` convention), without
   reading any column of the split table beyond identity and group.
2. The frozen mapping is applied to the same records; each changed state is attributed to a protocol rule.
3. The final dataset is compared with those records; each record-set difference is attributed to an
   exclusion or a duplicate rule.

Run from the repository root with ``PYTHONPATH=.``:
``.venv/bin/python -m scripts.reports.compare_mlc01_inventory --inventory <labelspace directory>``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT, challenge_labels, mlc01_dataset, mlc_labels, ningbo, sph
from ecg_experiment.eda import ptbxl as ptbxl_eda
from ecg_experiment.files import sha256_file, write_json_atomic

DATASET = ROOT / "outputs/mlc01_condition_dataset_20261110"
SNAPSHOT = ("counts_long.csv", "label_space_A.csv", "label_spaces_BC.csv", "summary.json", "patients.csv",
            "vocabulary.py", "count_labels.py", "mapping.csv")
SPLIT_GROUPS = {"train": "train", "calibration": "calibration"}
PTBXL_GROUPS = {"folds_1_8": [f"fold_{n}" for n in range(1, 9)], "fold_9": ["fold_9"], "fold_10": ["fold_10"]}
SPH_GROUPS = {"all": ["half_calibration", "half_evaluation"], "half_calibration": ["half_calibration"],
              "half_evaluation": ["half_evaluation"]}
FINAL_ROLES = {"folds_1_8": "train", "fold_9": "validation", "fold_10": "test_same_source", "train": "train",
               "calibration": "validation", "half_evaluation": "test_external",
               "half_calibration": "sph_calibration"}
UNSPECIFIED_MI = frozenset({"164865005", "164867002", "57054005"})


def manifest_column(name: str, manifest: pd.DataFrame) -> str:
    """
    Manifest column holding a draft condition or a space-A group.

    Parameters
    ----------
    name : str
        Condition key or ``a_<group>``.
    manifest : pd.DataFrame
        The MLC-01 manifest.

    Returns
    -------
    str
        ``a_<group>``, ``c_<key>`` when the condition is in space C, else ``b_<key>``.
    """
    if name.startswith("a_"):
        return name
    return f"c_{name}" if f"c_{name}" in manifest.columns else f"b_{name}"


def load_vocabulary(path: Path) -> list[dict]:
    """
    Load the draft condition list of the inventory.

    Parameters
    ----------
    path : Path
        The inventory's ``vocabulary.py``.

    Returns
    -------
    list[dict]
        Its ``CONDITIONS``.
    """
    spec = importlib.util.spec_from_file_location("inventory_vocabulary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONDITIONS


def draft_state(item: dict, source: str, record: object) -> float:
    """
    State of one draft condition on one record, as the inventory computed it.

    Parameters
    ----------
    item : dict
        One draft condition.
    source : str
        One of ``mlc_labels.SOURCES``.
    record : object
        The record's codes (``mlc01_dataset.parsed_codes``).

    Returns
    -------
    float
        1.0, 0.0 or NaN; the draft had no contradictory-pair rule.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    positive, ambiguous = frozenset(item["codes"][vocabulary]), frozenset(item["ambiguous"][vocabulary])
    status = item["annotation"].get(source, ("", ""))[0]
    if not positive or status == "not annotated":
        return np.nan
    if mlc_labels.listed(positive, record, vocabulary):
        return 1.0
    return np.nan if mlc_labels.listed(ambiguous, record, vocabulary) else 0.0


def draft_groups(source: str, record: object, classes: dict[str, str]) -> dict[str, float]:
    """
    Space-A groups as the inventory computed them: existing definitions, no undefined state.

    Parameters
    ----------
    source : str
        One of ``mlc_labels.SOURCES``.
    record : object
        The record's codes.
    classes : dict[str, str]
        PTB-XL diagnostic statement to superclass.

    Returns
    -------
    dict[str, float]
        1.0 or 0.0 per group; SPH code 153 counts for STTC as in ``ecg_experiment/sph.py``.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    if vocabulary == "aha":
        bases = {base for base, _ in record}
        return {name: float(bool(bases & members)) for name, members in sph.SUPERCLASSES.items()}
    return mlc_labels.group_flags(source, record, classes)


def records_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the codes of every manifest record.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.

    Returns
    -------
    pd.Series
        Codes per record key.
    """
    items = zip(manifest["source"], manifest["raw_codes"], manifest["heart_axis"], strict=True)
    return pd.Series([mlc01_dataset.parsed_codes(source, raw, axis) for source, raw, axis in items],
                     index=manifest.index)


def split_duplicate_status(manifest: pd.DataFrame, codes: pd.Series) -> pd.Series:
    """
    Rebuild the split's duplicate status of the open Challenge records with the frozen Ningbo-family rule.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.
    codes : pd.Series
        Output of ``records_frame``.

    Returns
    -------
    pd.Series
        ``unique``, ``kept``, ``dropped_copy`` or ``dropped_conflict`` per open Challenge record.
    """
    part = manifest[manifest["group_kind"] == "record"]
    labels = challenge_labels.label_table(codes[part.index].apply(sorted), challenge_labels.load_official())
    frame = pd.DataFrame({"signal_sha256": part["sha256_full"], "label_key": ningbo.label_key(labels),
                          "male": ningbo.male(part["sex"].map({"M": "Male", "F": "Female"}))})
    frame.index = part["record"].to_numpy()
    status = ningbo.duplicate_status(frame)["duplicate_status"]
    return pd.Series(status.to_numpy(), index=part.index)


def inventory_sets(manifest: pd.DataFrame, split_status: pd.Series) -> dict[tuple[str, str], pd.Index]:
    """
    Record keys of every inventory group.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.
    split_status : pd.Series
        Output of ``split_duplicate_status``.

    Returns
    -------
    dict[tuple[str, str], pd.Index]
        Keyed by ``(source, group)`` with the inventory's group names.
    """
    sets = {}
    for group, origins in PTBXL_GROUPS.items():
        mask = (manifest["source"] == "ptbxl") & manifest["origin"].isin(origins)
        sets[("ptbxl", group)] = manifest.index[mask]
    kept = split_status.index[split_status.isin(["unique", "kept"])]
    for source in mlc01_dataset.RELEASES:
        for group in SPLIT_GROUPS:
            mask = (manifest["source"] == source) & (manifest["origin"] == group) & manifest.index.isin(kept)
            sets[(source, group)] = manifest.index[mask]
    for group, origins in SPH_GROUPS.items():
        mask = (manifest["source"] == "sph") & manifest["origin"].isin(origins)
        sets[("sph", group)] = manifest.index[mask]
    return sets


def draft_table(manifest: pd.DataFrame, codes: pd.Series, conditions: list[dict], classes: dict[str, str]
                ) -> pd.DataFrame:
    """
    Draft states of every record for the conditions of spaces B and C and the space-A groups.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.
    codes : pd.Series
        Output of ``records_frame``.
    conditions : list[dict]
        Output of ``load_vocabulary``.
    classes : dict[str, str]
        PTB-XL diagnostic statement to superclass.

    Returns
    -------
    pd.DataFrame
        One column per condition key and per ``a_<group>``.
    """
    wanted = [item for item in conditions if item["key"] in set(mlc_labels.SPACE_B) | set(mlc_labels.SPACE_C)]
    rows = []
    for source, record in zip(manifest["source"], codes, strict=True):
        row = {item["key"]: draft_state(item, source, record) for item in wanted}
        row |= {f"a_{name}": value for name, value in draft_groups(source, record, classes).items()}
        rows.append(row)
    return pd.DataFrame(rows, index=manifest.index)


def cause(column: str, source: str, record: object) -> str:
    """
    Protocol rule that explains a state differing from the draft.

    Parameters
    ----------
    column : str
        Condition key or ``a_<group>``.
    source : str
        Record source.
    record : object
        The record's codes.

    Returns
    -------
    str
        Rule name, or ``unexplained``.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    in_pair = [pair for pair in mlc_labels.CONTRADICTORY_PAIRS if column in pair]
    lvhv = mlc_labels.listed(mlc_labels.LV_HIGH_VOLTAGE[vocabulary], record, vocabulary)
    has_153 = vocabulary == "aha" and mlc_labels.aha_match("153", record)
    contradiction = any(all(mlc_labels.listed(mlc_labels.CONDITIONS[key]["positive"][vocabulary], record,
                                              vocabulary) for key in pair) for pair in in_pair)
    rules = [
        ("contradictory_pair", contradiction),
        ("georgia_st_depression_undefined", column == "st_depression" and source == "georgia"),
        ("lv_high_voltage_only_undefined", column in ("lvh", "a_HYP") and lvhv),
        ("sph_153_sets_no_b_condition", column == "lvh" and has_153),
        ("sph_153_counts_for_hyp_not_sttc", column in ("a_STTC", "a_HYP") and has_153),
        ("cpsc_2018_group_not_annotated", column in ("a_MI", "a_HYP") and source == "cpsc_2018"),
        ("chapman_lpfb_not_annotated", column == "lpfb" and source == "chapman_shaoxing"),
        ("anterior_mi_unspecified_territory", column == "anterior_mi" and vocabulary == "snomed"
         and bool(record & UNSPECIFIED_MI)),
    ]
    return next((name for name, applies in rules if applies), "unexplained")


def transitions(manifest: pd.DataFrame, codes: pd.Series, draft: pd.DataFrame,
                sets: dict[tuple[str, str], pd.Index]) -> pd.DataFrame:
    """
    Count records whose state differs between the draft and the frozen mapping, by group and rule.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.
    codes : pd.Series
        Output of ``records_frame``.
    draft : pd.DataFrame
        Output of ``draft_table``.
    sets : dict[tuple[str, str], pd.Index]
        Output of ``inventory_sets``.

    Returns
    -------
    pd.DataFrame
        ``source``, ``group``, ``label``, ``draft``, ``frozen``, ``rule`` and ``records``.
    """
    rows = []
    for (source, group), keys in sets.items():
        for column in draft.columns:
            mine = manifest.loc[keys, manifest_column(column, manifest)].astype(float)
            theirs = draft.loc[keys, column]
            changed = keys[~((mine == theirs) | (mine.isna() & theirs.isna())).to_numpy()]
            for key in changed:
                rows.append({"source": source, "group": group, "label": column,
                             "draft": mlc_labels.state_name(theirs[key]),
                             "frozen": mlc_labels.state_name(mine[key]),
                             "rule": cause(column, source, codes[key])})
    table = pd.DataFrame(rows, columns=["source", "group", "label", "draft", "frozen", "rule"])
    return table.groupby(list(table.columns)).size().rename("records").reset_index()


def counts(values: pd.DataFrame) -> pd.DataFrame:
    """
    Count records, positives and undefined per column.

    Parameters
    ----------
    values : pd.DataFrame
        States (1.0, 0.0, NaN).

    Returns
    -------
    pd.DataFrame
        Indexed by column.
    """
    return pd.DataFrame({"records": len(values), "positive": (values == 1).sum(),
                         "undefined": values.isna().sum()})


def comparison(manifest: pd.DataFrame, draft: pd.DataFrame, sets: dict[tuple[str, str], pd.Index],
               inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Set inventory counts beside the draft reproduction, the frozen mapping on its records and the final data.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.
    draft : pd.DataFrame
        Output of ``draft_table``.
    sets : dict[tuple[str, str], pd.Index]
        Output of ``inventory_sets``.
    inventory : pd.DataFrame
        Inventory counts indexed by ``(label, source, group)`` with ``n_records``, ``n_positive`` and
        ``n_undefined``.

    Returns
    -------
    pd.DataFrame
        One row per label and group.
    """
    rows = []
    for (source, group), keys in sets.items():
        columns = {name: manifest_column(name, manifest) for name in draft.columns}
        reproduced = counts(draft.loc[keys])
        frozen = counts(manifest.loc[keys, list(columns.values())].astype(float)
                        .set_axis(list(columns), axis=1))
        role = FINAL_ROLES.get(group)
        final_mask = (manifest["source"] == source) & manifest["included"]
        if role is not None:
            final_mask &= manifest["role"] == role
        final = counts(manifest.loc[final_mask, list(columns.values())].astype(float)
                       .set_axis(list(columns), axis=1))
        for name in columns:
            index = (name, source, group)
            listed = inventory.loc[index] if index in inventory.index else None
            row = {"label": name, "source": source, "group": group}
            row |= {f"inventory_{key}": (np.nan if listed is None else listed[f"n_{key}"])
                    for key in ("records", "positive", "undefined")}
            for stage, table in (("draft_reproduced", reproduced), ("frozen_same_records", frozen),
                                 ("final", final)):
                row |= {f"{stage}_{key}": int(table.loc[name, key])
                        for key in ("records", "positive", "undefined")}
            rows.append(row)
    return pd.DataFrame(rows)


def inventory_counts(directory: Path) -> pd.DataFrame:
    """
    Inventory counts of spaces B and C and of space A, in one long table.

    Parameters
    ----------
    directory : Path
        Snapshot directory.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(label, source, group)``; space A has no undefined count, so zero is used.
    """
    long = pd.read_csv(directory / "counts_long.csv")
    long = long.rename(columns={"condition": "label"})[["label", "source", "group", "n_records", "n_positive",
                                                        "n_undefined"]]
    space_a = pd.read_csv(directory / "label_space_A.csv")
    parts = [long]
    for name in mlc_labels.SPACE_A:
        parts.append(pd.DataFrame({"label": f"a_{name}", "source": space_a["source"],
                                   "group": space_a["group"],
                                   "n_records": space_a["n_records"], "n_positive": space_a[name],
                                   "n_undefined": 0}))
    return pd.concat(parts).set_index(["label", "source", "group"])


def record_set_changes(manifest: pd.DataFrame, sets: dict[tuple[str, str], pd.Index]) -> pd.DataFrame:
    """
    Why records of each inventory group are absent from the final dataset, or present only there.

    Parameters
    ----------
    manifest : pd.DataFrame
        The MLC-01 manifest.
    sets : dict[tuple[str, str], pd.Index]
        Output of ``inventory_sets``.

    Returns
    -------
    pd.DataFrame
        ``source``, ``group``, ``change`` and ``records``.
    """
    rows = []
    for (source, group), keys in sets.items():
        role = FINAL_ROLES.get(group)
        if role is None:
            continue
        mask = (manifest["source"] == source) & (manifest["role"] == role) & manifest["included"]
        final = manifest.index[mask]
        dropped = manifest.loc[keys.difference(final)]
        reasons = dropped["exclusion_reason"].where(dropped["exclusion_reason"] != "",
                                                    dropped["duplicate_status"])
        for reason, n in reasons.fillna("").value_counts().items():
            rows.append({"source": source, "group": group, "change": f"absent from final: {reason}",
                         "records": n})
        added = final.difference(keys)
        if len(added):
            rows.append({"source": source, "group": group,
                         "change": "in final, not in the inventory set (split rule dropped it)",
                         "records": len(added)})
    return pd.DataFrame(rows, columns=["source", "group", "change", "records"])


def main() -> None:
    """Write the comparison tables and their receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    out = args.dataset / "inventory_comparison"
    snapshot = out / "inventory_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in SNAPSHOT:
        shutil.copy2(args.inventory / name, snapshot / name)
    manifest = pd.read_csv(args.dataset / "manifest.csv", index_col="record_key", low_memory=False,
                           dtype={"record": str, "raw_codes": str, "heart_axis": str})
    for name in ("heart_axis", "exclusion_reason", "duplicate_status"):
        manifest[name] = manifest[name].fillna("")
    codes = records_frame(manifest)
    classes = ptbxl_eda.diagnostic_classes(ptbxl_eda.load_statements())
    split_status = split_duplicate_status(manifest, codes)
    sets = inventory_sets(manifest, split_status)
    draft = draft_table(manifest, codes, load_vocabulary(snapshot / "vocabulary.py"), classes)
    table = comparison(manifest, draft, sets, inventory_counts(snapshot))
    table.to_csv(out / "counts.csv", index=False)
    transitions(manifest, codes, draft, sets).to_csv(out / "rule_transitions.csv", index=False)
    record_set_changes(manifest, sets).to_csv(out / "record_set_changes.csv", index=False)
    listed = table.dropna(subset=["inventory_records"])
    residual = {key: int((listed[f"inventory_{key}"] != listed[f"draft_reproduced_{key}"]).sum())
                for key in ("records", "positive", "undefined")}
    summary = {"split_rule_status": split_status.value_counts().to_dict(),
               "draft_reproduction_mismatches": residual,
               "compared_rows": len(listed)}
    receipt = {"script_sha256": sha256_file(Path(__file__)),
               "manifest_sha256": sha256_file(args.dataset / "manifest.csv"),
               "inventory_snapshot_sha256": {name: sha256_file(snapshot / name) for name in SNAPSHOT},
               "summary": summary,
               "outputs_sha256": {name: sha256_file(out / name) for name in
                                  ("counts.csv", "rule_transitions.csv", "record_set_changes.csv")}}
    write_json_atomic(out / "receipt.json", receipt, sort_keys=True)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
