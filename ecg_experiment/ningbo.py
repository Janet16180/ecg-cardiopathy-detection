"""Duplicate resolution and demographics for Ningbo and its source family, Chapman/Shaoxing.

Chapman (JS00001-JS10646) and Ningbo (JS10647 onward) were released together, and the Ningbo EDA found
exact copies between them and within Ningbo (``notebooks/09-jr-ningbo.ipynb``). The two sources are
therefore resolved as one family. Record names do not collide across the two ranges, so the lowest record
name, always the Chapman copy in a cross-source group, is the one kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS = ("primary", "secondary", "MI", "STTC", "CD", "HYP")


def clean_age(age: pd.Series) -> pd.Series:
    """
    Ages in years with the missing-value placeholders removed.

    Ningbo stores ``NaN`` for 55 records and 0 for 247; no record is aged 1 to 3, and most age-0 records that
    also appear in Chapman carry an adult age there, so 0 is a placeholder.

    Parameters
    ----------
    age : pd.Series
        Header age as text or number.

    Returns
    -------
    pd.Series
        Float ages, NaN where missing or 0.
    """
    years = pd.to_numeric(age, errors="coerce")
    return years.where(years > 0)


def male(sex: pd.Series) -> pd.Series:
    """
    Header sex as 1.0 (male), 0.0 (female) or NaN (unknown).

    Parameters
    ----------
    sex : pd.Series
        ``Male``, ``Female`` or anything else.

    Returns
    -------
    pd.Series
        Float indicator.
    """
    return sex.map({"Male": 1.0, "Female": 0.0}).astype(float)


def label_key(labels: pd.DataFrame) -> pd.Series:
    """
    One comparable text key per record from the mapped labels.

    Parameters
    ----------
    labels : pd.DataFrame
        ``challenge_labels.label_table`` output.

    Returns
    -------
    pd.Series
        Keys such as ``"1,1,0,1,0,0"``; undefined labels appear as ``nan``.
    """
    values = labels[list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    return pd.Series([",".join(f"{value:g}" for value in row) for row in values], index=labels.index)


def duplicate_status(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve records whose samples are bit-identical, within and across the family.

    In each identical group, the lowest record name is kept when every copy has the same mapped labels and
    the same known sex. Otherwise the copies disagree about the recording, and every copy is dropped.

    Parameters
    ----------
    frame : pd.DataFrame
        Indexed by record name, with ``signal_sha256``, ``label_key`` and ``male`` (NaN when unknown).

    Returns
    -------
    pd.DataFrame
        ``duplicate_status`` (``unique``, ``kept``, ``dropped_copy`` or ``dropped_conflict``) and
        ``duplicate_of`` (the other record names of the group, ``;``-separated), on the input index.
    """
    ordered = frame.sort_index()
    grouped = ordered.groupby("signal_sha256")
    size = grouped["label_key"].transform("size")
    conflict = (grouped["label_key"].transform("nunique") > 1) | (grouped["male"].transform("nunique") > 1)
    first = ~ordered["signal_sha256"].duplicated(keep="first")
    status = pd.Series("unique", index=ordered.index)
    status[(size > 1) & first & ~conflict] = "kept"
    status[(size > 1) & ~first & ~conflict] = "dropped_copy"
    status[(size > 1) & conflict] = "dropped_conflict"
    members = ordered[size > 1].groupby("signal_sha256").groups
    others = [";".join(name for name in members.get(sha, []) if name != record)
              for record, sha in ordered["signal_sha256"].items()]
    result = pd.DataFrame({"duplicate_status": status, "duplicate_of": others}, index=ordered.index)
    return result.reindex(frame.index)
