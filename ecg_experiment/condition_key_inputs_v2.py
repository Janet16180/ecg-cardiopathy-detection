"""Strict numeric parsing successor for the pre-score KEY-01 qualification stop."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import ROOT
from .condition_key_inputs import CACHE, COHORT, SPH, cached_arrays, positions
from .condition_keys import GROUPS


def memberships(table: pd.DataFrame) -> np.ndarray:
    """
    Parse numeric binary group memberships, including decimal string representations.

    Parameters
    ----------
    table : pd.DataFrame
        Table containing all four group columns.

    Returns
    -------
    np.ndarray
        Integer zero/one [records, groups] array.
    """
    values = table[list(GROUPS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("Membership flags must be finite and exactly binary")
    return values.astype(int)


def load_inputs() -> dict[str, Any]:
    """
    Align unchanged permitted cohorts using strict numeric membership parsing.

    Returns
    -------
    dict[str, Any]
        Training/query metadata, integer flags and aligned original feature caches.
    """
    train = pd.read_csv(ROOT / COHORT, dtype=str, keep_default_na=False)
    query = pd.read_csv(ROOT / SPH, dtype=str, keep_default_na=False)
    query = query.loc[query["evaluation"].eq("True")].reset_index(drop=True)
    if len(train) != 25000 or len(query) != 10434:
        raise ValueError("Frozen cohort counts changed")
    if not train["split"].eq("train").all() or train["source"].eq("sph").any():
        raise ValueError("Ineligible training source or role")
    if not query["source"].eq("sph").all() or query["fit"].eq("True").any():
        raise ValueError("Ineligible query source or role")
    labels = pd.read_csv(ROOT / CACHE / "qualify/labels.csv").set_index("record_id")
    if not np.array_equal(labels.loc[train["record_id"], "target"], train["target"].to_numpy(dtype=int)):
        raise ValueError("Training target identities changed")
    result: dict[str, Any] = {"train": train, "query": query}
    for role, frame, label_table, part, expected in (
        ("train", train, labels.loc[train["record_id"]], "training", 28568),
        ("query", query, query, "evaluation", 21677),
    ):
        flags = memberships(label_table)
        if not np.array_equal(flags.max(axis=1), frame["target"].to_numpy(dtype=int)):
            raise ValueError("Superclass union no longer equals the binary target")
        rows = pd.read_csv(ROOT / CACHE / f"extract_{part}/rows.csv")
        index = positions(rows["record_id"], frame["record_id"])
        pooled, tokens = cached_arrays(part, expected)
        result.update({f"{role}_flags": flags, f"{role}_positions": index,
                       f"{role}_pooled": np.asarray(pooled[index]), f"{role}_tokens": tokens})
    known = set(train.loc[train["patient_identity_known"].eq("True"), "patient_id"])
    if known & set(query["patient_key"]) or set(train["record_id"]) & set(query["record_id"]):
        raise ValueError("Training and query identities overlap")
    return result
