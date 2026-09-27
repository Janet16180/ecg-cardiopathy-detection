"""Audit the PTB-XL binary labels and patient splits used by the project."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecg_experiment.eda.ptbxl import (
    QUALITY_COLUMNS,
    diagnostic_classes,
    load_manifest,
    load_metadata,
    load_statements,
    project_label,
    superclass_label,
    superclasses,
)

LABEL_NAMES = {0.0: "negative", 1.0: "positive"}
MANIFESTS = ["labeled_train", "unlabeled_train", "validation", "test"]


def labeled_metadata() -> pd.DataFrame:
    """
    Load metadata and attach split, both label rules and superclass summary.

    Returns
    -------
    pd.DataFrame
        Metadata with ``split``, ``project``, ``superclass`` and ``classes``
        columns. Label columns hold ``negative``, ``positive`` or ``unlabeled``.
    """
    meta = load_metadata()
    classes = diagnostic_classes(load_statements())
    meta["split"] = np.select([meta["strat_fold"] <= 8, meta["strat_fold"] == 9],
                              ["train", "validation"], "test")
    meta["project"] = meta["scp_codes"].apply(project_label, classes=classes)
    meta["superclass"] = meta["scp_codes"].apply(superclass_label, classes=classes)
    meta["classes"] = meta["scp_codes"].apply(superclasses, classes=classes)
    for column in ("project", "superclass"):
        meta[column] = meta[column].map(LABEL_NAMES).fillna("unlabeled")
    return meta


def reproduction_mismatches(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compare independently derived labels with every project manifest.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``labeled_metadata``.

    Returns
    -------
    pd.DataFrame
        Per manifest: row count, rows whose target differs from our rule,
        and rows whose fold does not match the manifest's intended split.
    """
    expected_split = {"labeled_train": "train", "unlabeled_train": "train",
                      "validation": "validation", "test": "test"}
    rows = []
    for fraction in ("1", "0.1"):
        for name in MANIFESTS:
            manifest = load_manifest(name, fraction)
            ours = meta.loc[manifest.index]
            label_mismatch = 0
            if "target" in manifest:
                label_mismatch = int((manifest["target"].map(LABEL_NAMES) != ours["project"]).sum())
            rows.append({"fraction": fraction, "manifest": name, "records": len(manifest),
                         "label_mismatch": label_mismatch,
                         "wrong_split": int((ours["split"] != expected_split[name]).sum())})
    return pd.DataFrame(rows)


def split_integrity() -> dict[str, int]:
    """
    Check patient disjointness and nesting of the limited-label subset.

    Returns
    -------
    dict[str, int]
        Violation counts; every value should be zero.
    """
    patients = {name: set(load_manifest(name)["patient_id"])
                for name in ("all_train_ssl", "validation", "test")}
    full = load_manifest("labeled_train", "1")
    limited = load_manifest("labeled_train", "0.1")
    selected = set(limited["patient_id"])
    partially_selected = full[full["patient_id"].isin(selected) & ~full.index.isin(limited.index)]
    return {
        "train_validation_shared_patients": len(patients["all_train_ssl"] & patients["validation"]),
        "train_test_shared_patients": len(patients["all_train_ssl"] & patients["test"]),
        "validation_test_shared_patients": len(patients["validation"] & patients["test"]),
        "limited_labels_not_in_full": int((~limited.index.isin(full.index)).sum()),
        "selected_patients_missing_records": len(partially_selected),
    }


def label_rule_comparison(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate the project rule against the superclass rule on held-out folds.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``labeled_metadata``.

    Returns
    -------
    pd.DataFrame
        Counts with project labels as rows and superclass labels as columns.
    """
    heldout = meta[meta["split"] != "train"]
    return pd.crosstab(heldout["project"], heldout["superclass"], margins=True)


def unlabeled_composition(meta: pd.DataFrame, statements: pd.DataFrame) -> pd.DataFrame:
    """
    Describe what the project rule leaves unlabeled.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``labeled_metadata``.
    statements : pd.DataFrame
        Output of ``load_statements``.

    Returns
    -------
    pd.DataFrame
        Record counts per non-NORM, non-SR code among unlabeled records, with
        the code's statement type and description.
    """
    unlabeled = meta[meta["project"] == "unlabeled"]
    codes = pd.Series([code for listed in unlabeled["scp_codes"] for code in listed
                       if code not in ("NORM", "SR")])
    counts = codes.value_counts().rename("records").to_frame()
    kind = np.select([statements["diagnostic"] == 1, statements["rhythm"] == 1],
                     ["diagnostic", "rhythm"], "form")
    counts["type"] = pd.Series(kind, index=statements.index).reindex(counts.index)
    counts["description"] = statements["description"].reindex(counts.index)
    return counts


def positive_rates(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Positive rate across candidate confounders among labeled records.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``labeled_metadata``.

    Returns
    -------
    dict[str, pd.DataFrame]
        One table per variable with record count and positive rate.
    """
    labeled = meta[meta["project"] != "unlabeled"].copy()
    labeled["positive"] = labeled["project"] == "positive"
    labeled["age_bin"] = pd.cut(labeled["age_capped"], [0, 30, 45, 60, 75, 90], right=False)
    labeled["year"] = labeled["recording_date"].dt.year
    labeled["any_quality_flag"] = labeled[QUALITY_COLUMNS[:4]].notna().any(axis=1)
    labeled["sex_name"] = labeled["sex"].map({0: "male", 1: "female"})
    tables = {}
    labeled["site"] = labeled["site"].where(labeled["site"] < 4, other=4).map(
        {0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"})
    for variable in ("split", "sex_name", "age_bin", "device", "site", "year", "any_quality_flag",
                     "validated_by_human"):
        grouped = labeled.groupby(variable, observed=True)["positive"]
        tables[variable] = pd.DataFrame({"records": grouped.size(), "positive_rate": grouped.mean().round(3)})
    return tables


def metadata_baselines(meta: pd.DataFrame) -> pd.DataFrame:
    """
    AUC of single metadata variables as label predictors on held-out folds.

    Group positive rates are fitted on training records only. These are the
    floors a waveform model should beat, and the shortcuts it could learn.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``labeled_metadata``.

    Returns
    -------
    pd.DataFrame
        AUC per variable (rows) and held-out split (columns).
    """
    labeled = meta[meta["project"] != "unlabeled"].copy()
    labeled["y"] = (labeled["project"] == "positive").astype(int)
    labeled["age_filled"] = labeled["age_capped"].fillna(90)
    train = labeled[labeled["split"] == "train"]
    scores = {"age": labeled["age_filled"]}
    for variable in ("sex", "device", "site"):
        rates = train.groupby(variable)["y"].mean()
        scores[variable] = labeled[variable].map(rates).fillna(train["y"].mean())
    rows = {}
    for variable, score in scores.items():
        rows[variable] = {split: round(roc_auc_score(labeled.loc[subset.index, "y"], score[subset.index]), 3)
                          for split, subset in labeled[labeled["split"] != "train"].groupby("split")}
    return pd.DataFrame(rows).T


def patient_repeats(meta: pd.DataFrame) -> dict[str, int]:
    """
    Summarize repeated ECGs per patient and label changes within patients.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``labeled_metadata``.

    Returns
    -------
    dict[str, int]
        Counts of patients with repeats and with both labels.
    """
    labeled = meta[meta["project"] != "unlabeled"]
    per_patient = meta.groupby("patient_id").size()
    labels_per_patient = labeled.groupby("patient_id")["project"].nunique()
    return {
        "patients": len(per_patient),
        "patients_with_repeat_ecgs": int((per_patient > 1).sum()),
        "max_ecgs_per_patient": int(per_patient.max()),
        "patients_with_both_labels": int((labels_per_patient > 1).sum()),
    }
