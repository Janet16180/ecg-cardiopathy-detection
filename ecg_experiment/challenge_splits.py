"""A frozen record-level train, calibration and test split of the PhysioNet Challenge sources.

The Challenge sources carry no patient identifiers, so records are split with a fixed seed. Bit-identical
signals form one duplicate group, and a group is assigned as a whole, so no signal appears in two splits.
Groups are stratified by source and by the primary label. Chapman/Shaoxing and Ningbo are one source family
(one combined release, shared copies), and so are CPSC 2018 and CPSC-Extra (the same challenge collection,
with 344 signals shared between them).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SOURCES = ("ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
FAMILIES = {"ningbo": "chapman_ningbo", "chapman_shaoxing": "chapman_ningbo", "georgia": "georgia",
            "cpsc_2018": "cpsc", "cpsc_2018_extra": "cpsc"}
SPLITS = ("train", "calibration", "test")
PROPORTIONS = {"train": 0.6, "calibration": 0.2, "test": 0.2}
SEED = 20260929


def duplicate_groups(hashes: pd.DataFrame) -> pd.Series:
    """
    Group records that share any hash value, transitively.

    Parameters
    ----------
    hashes : pd.DataFrame
        One row per record, indexed by unique record name, with one column per hash kind. Empty strings and
        NaN mean no hash of that kind.

    Returns
    -------
    pd.Series
        The lowest record name of each record's group, on the input index.
    """
    records = hashes.index.to_numpy(dtype=str)
    long = hashes.reset_index(names="record").melt(id_vars="record", var_name="kind", value_name="value")
    long = long[long["value"].notna() & (long["value"].astype(str) != "")]
    record_codes = pd.Index(records).get_indexer(long["record"])
    value_codes, _ = pd.factorize(long["kind"].astype(str) + ":" + long["value"].astype(str))
    size = len(records) + len(np.unique(value_codes))
    edges = coo_matrix((np.ones(len(long)), (record_codes, len(records) + value_codes)), shape=(size, size))
    _, components = connected_components(edges, directed=False)
    component = pd.Series(components[:len(records)], index=hashes.index)
    lowest = pd.Series(records, index=hashes.index).groupby(component).transform("min")
    return lowest.rename("duplicate_group")


def label_category(primary: pd.Series) -> pd.Series:
    """
    Primary label as a stratum name.

    Parameters
    ----------
    primary : pd.Series
        1.0, 0.0 or NaN.

    Returns
    -------
    pd.Series
        ``positive``, ``negative`` or ``undefined``.
    """
    names = primary.map({1.0: "positive", 0.0: "negative"})
    return names.fillna("undefined").astype(str)


def split_sizes(count: int) -> dict[str, int]:
    """
    Count the groups per split in one stratum, rounding half up.

    Parameters
    ----------
    count : int
        Groups in the stratum.

    Returns
    -------
    dict[str, int]
        Calibration and test take their rounded shares; train takes the rest.
    """
    calibration = int(count * PROPORTIONS["calibration"] + 0.5)
    test = int(count * PROPORTIONS["test"] + 0.5)
    return {"train": count - calibration - test, "calibration": calibration, "test": test}


def assign_splits(groups: pd.Series, strata: pd.Series, seed: int = SEED) -> pd.Series:
    """
    Assign each duplicate group to a split, stratified by the stratum of the group's lowest record.

    Strata are visited in sorted order and their groups in sorted order, and one generator shuffles each
    stratum, so the assignment depends only on the records, their strata and the seed.

    Parameters
    ----------
    groups : pd.Series
        Output of ``duplicate_groups``; each value is a record name of the index.
    strata : pd.Series
        Stratum name of each record, on the same index.
    seed : int
        Seed of the generator.

    Returns
    -------
    pd.Series
        Split name of each record, on the input index.
    """
    representatives = pd.Index(sorted(groups.unique()))
    group_strata = strata.loc[representatives]
    rng = np.random.default_rng(seed)
    assigned = {}
    for stratum in sorted(group_strata.unique()):
        members = representatives[(group_strata == stratum).to_numpy()]
        shuffled = members[rng.permutation(len(members))]
        labels = np.repeat(SPLITS, [split_sizes(len(members))[name] for name in SPLITS])
        assigned.update(zip(shuffled, labels, strict=True))
    return groups.map(assigned).rename("split")


def split_counts(rows: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    """
    Count records per source, split and primary label category.

    Parameters
    ----------
    rows : pd.DataFrame
        With ``source``, ``split`` and ``primary``.

    Returns
    -------
    dict[str, dict[str, dict[str, int]]]
        Counts keyed by source, split, then ``positive``, ``negative``, ``undefined`` and ``records``.
    """
    counts = {}
    for source, frame in rows.groupby("source"):
        counts[source] = {}
        for split in SPLITS:
            part = frame[frame["split"] == split]
            categories = label_category(part["primary"]).value_counts()
            counts[source][split] = {"records": len(part),
                                     **{name: int(categories.get(name, 0))
                                        for name in ("positive", "negative", "undefined")}}
    return counts
