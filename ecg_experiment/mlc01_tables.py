"""Count tables of the MLC-01 condition dataset, computed from its manifest.

Every table describes the data; none holds a model score. Counts are within a source (or a multi-source
family) and a role, never pooled across sources.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd

from ecg_experiment import mlc_labels
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.mlc01_dataset import EXCLUSIONS, ROLES

MULTI_SOURCE_FAMILIES = ("chapman_ningbo", "cpsc")
EVALUATION_ROLES = ("validation", "test_same_source", "test_external", "sph_calibration")
SUPPORT_LEVELS = (20, 50, 100)
NEAR_LEADS = 6
LEAD_GROUP_CAP = 50


def units(frame: pd.DataFrame) -> list[tuple[str, str, str, np.ndarray]]:
    """
    Every role by source, and role by multi-source family, of the included records.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``included``, ``role``, ``source`` and ``family``.

    Returns
    -------
    list[tuple[str, str, str, np.ndarray]]
        ``(role, level, unit, mask)`` with ``level`` ``source`` or ``family``; empty units are left out.
    """
    included = frame["included"].to_numpy(dtype=bool)
    found = []
    for role in ROLES:
        in_role = included & (frame["role"] == role).to_numpy()
        for level, names in (("source", mlc_labels.SOURCES), ("family", MULTI_SOURCE_FAMILIES)):
            for name in names:
                mask = in_role & (frame[level] == name).to_numpy()
                if mask.any():
                    found.append((role, level, name, mask))
    return found


def state_counts(frame: pd.DataFrame, columns: list[str], space: str) -> pd.DataFrame:
    """
    Positive, negative and undefined records per label and unit.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with the label columns (1, 0 or missing).
    columns : list[str]
        Label columns to count.
    space : str
        Name written in the ``space`` column.

    Returns
    -------
    pd.DataFrame
        One row per label, role and unit.
    """
    rows = []
    for role, level, unit, mask in units(frame):
        values = frame.loc[mask, columns].astype(float)
        for column in columns:
            states = values[column]
            rows.append({"space": space, "label": column, "role": role, "level": level, "unit": unit,
                         "records": int(mask.sum()), "positive": int((states == 1).sum()),
                         "negative": int((states == 0).sum()), "undefined": int(states.isna().sum())})
    return pd.DataFrame(rows)


def sensitivity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Space-B states of the likelihood sensitivity analysis.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``b_<condition>`` and ``lik50_<condition>``.

    Returns
    -------
    pd.DataFrame
        The identity columns of ``units`` and one ``b_<condition>`` column per condition, undefined where
        the PTB-XL positive rests only on likelihoods below 50.
    """
    states = {f"b_{key}": frame[f"b_{key}"].astype(float).mask(frame[f"lik50_{key}"].astype(bool))
              for key in mlc_labels.SPACE_B}
    identity = {name: frame[name] for name in ("included", "role", "source", "family")}
    return pd.DataFrame({**identity, **states})


def support_table(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Positives and negatives of each condition in each evaluation set, against the support levels.

    Parameters
    ----------
    counts : pd.DataFrame
        Output of ``state_counts``.

    Returns
    -------
    pd.DataFrame
        Evaluation-role rows with ``below_20``, ``below_50``, ``below_100`` (positives) and ``enters_macro``
        (at least 20 positives and 20 negatives, the protocol's macro rule).
    """
    table = counts[counts["role"].isin(EVALUATION_ROLES)].copy()
    for level in SUPPORT_LEVELS:
        table[f"below_{level}"] = table["positive"] < level
    table["enters_macro"] = (table["positive"] >= 20) & (table["negative"] >= 20)
    return table


def exclusion_flow(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Count records per source and role at each step from enumeration to the final dataset.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``exclusion_reason``, ``duplicate_status`` and ``included``.

    Returns
    -------
    pd.DataFrame
        One row per source and role, plus a ``all`` role row per source.
    """
    rows = []
    for source in mlc_labels.SOURCES:
        part = frame[frame["source"] == source]
        for role in [*ROLES, "all"]:
            group = part if role == "all" else part[part["role"] == role]
            if group.empty:
                continue
            row = {"source": source, "role": role, "start": len(group)}
            row |= {name: int((group["exclusion_reason"] == name).sum()) for name in EXCLUSIONS}
            row["eligible"] = int((group["exclusion_reason"] == "").sum())
            for status in ("removed_cross_role", "dropped_copy", "dropped_conflict"):
                row[status] = int((group["duplicate_status"] == status).sum())
            row["final"] = int(group["included"].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def group_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Count records and patient or record groups per role and source of the final dataset.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``included`` and ``group_id``.

    Returns
    -------
    pd.DataFrame
        ``records``, ``groups``, ``groups_with_several_records``, ``records_in_those_groups`` and
        ``max_records_per_group``.
    """
    rows = []
    for role, level, unit, mask in units(frame):
        if level != "source":
            continue
        per = frame.loc[mask, "group_id"].value_counts()
        rows.append({"role": role, "source": unit, "group_kind": frame.loc[mask, "group_kind"].iloc[0],
                     "records": int(mask.sum()), "groups": len(per), "groups_with_several_records":
                     int((per > 1).sum()), "records_in_those_groups": int(per[per > 1].sum()),
                     "max_records_per_group": int(per.max())})
    return pd.DataFrame(rows)


def patients_in_several_roles(frame: pd.DataFrame) -> dict[str, int]:
    """
    Patients of PTB-XL and SPH whose records fall in more than one role, over every enumerated record.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``source``, ``group_id`` and ``role``.

    Returns
    -------
    dict[str, int]
        Count per source; the protocol requires zero.
    """
    found = {}
    for source in ("ptbxl", "sph"):
        part = frame[frame["source"] == source]
        found[source] = int((part.groupby("group_id")["role"].nunique() > 1).sum())
    return found


def cooccurrence(frame: pd.DataFrame, space: tuple[str, ...], prefix: str
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count positive conditions per record, and records positive for each pair, per role and source.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with the label columns.
    space : tuple[str, ...]
        Conditions of the space.
    prefix : str
        Column prefix, ``b_`` or ``c_``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Distribution of the positive count (``5`` stands for five or more) and pair counts with the share
        of the unit's records.
    """
    distribution, pairs = [], []
    columns = [f"{prefix}{key}" for key in space]
    for role, level, unit, mask in units(frame):
        if level != "source":
            continue
        positive = frame.loc[mask, columns].astype(float).to_numpy() == 1
        count = positive.sum(axis=1).clip(max=5)
        for value, n in sorted(Counter(count.tolist()).items()):
            distribution.append({"role": role, "source": unit, "positive_conditions": int(value),
                                 "records": n, "share": n / len(count)})
        for (i, first), (j, second) in itertools.combinations(enumerate(space), 2):
            both = int((positive[:, i] & positive[:, j]).sum())
            pairs.append({"role": role, "source": unit, "condition_a": first, "condition_b": second,
                          "records": both, "share": both / len(count)})
    return pd.DataFrame(distribution), pd.DataFrame(pairs)


def lead_flag_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Count records with flat or missing leads in the first 10 s, per role and source of the final dataset.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``flat_<lead>`` and ``missing_<lead>``.

    Returns
    -------
    pd.DataFrame
        Records with at least one flat lead, with all twelve flat, with at least one missing sample, and the
        count per lead.
    """
    rows = []
    flat = [f"flat_{lead}" for lead in LEADS]
    missing = [f"missing_{lead}" for lead in LEADS]
    for role, level, unit, mask in units(frame):
        if level != "source":
            continue
        part = frame.loc[mask]
        row = {"role": role, "source": unit, "records": len(part),
               "any_flat_lead": int(part[flat].any(axis=1).sum()),
               "all_leads_flat": int(part[flat].all(axis=1).sum()),
               "any_missing_samples": int(part[missing].any(axis=1).sum())}
        row |= {name: int(part[name].sum()) for name in flat + missing}
        rows.append(row)
    return pd.DataFrame(rows)


def duplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """
    One row per group of eligible records with an identical first 10 s.

    Parameters
    ----------
    frame : pd.DataFrame
        Manifest with ``duplicate_group``, ``duplicate_size`` and ``duplicate_status``.

    Returns
    -------
    pd.DataFrame
        Group key, size, sources, roles, scope (``within_source``, ``across_sources`` in one role,
        ``across_roles``), the number of distinct whole-record hashes and the statuses.
    """
    part = frame[frame["duplicate_size"].fillna(0) > 1]
    rows = []
    for group, members in part.groupby("duplicate_group"):
        roles = sorted(set(members["role"]))
        sources = sorted(set(members["source"]))
        scope = "across_sources" if len(sources) > 1 else "within_source"
        scope = "across_roles" if len(roles) > 1 else scope
        status = Counter(members["duplicate_status"])
        rows.append({"duplicate_group": group, "size": len(members), "sources": "+".join(sources),
                     "roles": "+".join(roles), "scope": scope,
                     "distinct_full_hashes": int(members["sha256_full"].nunique()),
                     **{name: status.get(name, 0) for name in ("kept", "dropped_copy", "dropped_conflict",
                                                               "removed_cross_role")}})
    return pd.DataFrame(rows)


def duplicate_accounting(groups: pd.DataFrame) -> pd.DataFrame:
    """
    Count groups and records per scope and source combination, with the records each rule removed.

    Parameters
    ----------
    groups : pd.DataFrame
        Output of ``duplicate_groups``.

    Returns
    -------
    pd.DataFrame
        One row per scope, sources and roles combination.
    """
    columns = ["size", "kept", "dropped_copy", "dropped_conflict", "removed_cross_role"]
    table = groups.groupby(["scope", "sources", "roles"])[columns].sum().reset_index()
    table.insert(3, "groups", groups.groupby(["scope", "sources", "roles"]).size().to_numpy())
    return table.rename(columns={"size": "records"})


def shape_near_duplicates(eligible: pd.DataFrame) -> pd.DataFrame:
    """
    Find groups with the same sign pattern of successive differences but a different first-10-s hash.

    Records with six or more flat leads are left out, as their sign patterns are mostly zero.

    Parameters
    ----------
    eligible : pd.DataFrame
        Eligible records with ``shape_sha256_10s``, ``sha256_10s``, ``source``, ``role`` and flat flags.

    Returns
    -------
    pd.DataFrame
        One row per such group: size, distinct exact hashes, sources and roles.
    """
    flat = eligible[[f"flat_{lead}" for lead in LEADS]].sum(axis=1)
    part = eligible[flat < NEAR_LEADS]
    rows = []
    for shape, members in part.groupby("shape_sha256_10s"):
        if members["sha256_10s"].nunique() > 1:
            rows.append({"evidence": "same_shape_different_exact_hash", "key": shape, "records": len(members),
                         "distinct_exact_hashes": int(members["sha256_10s"].nunique()),
                         "sources": "+".join(sorted(set(members["source"]))),
                         "roles": "+".join(sorted(set(members["role"])))})
    return pd.DataFrame(rows, columns=["evidence", "key", "records", "distinct_exact_hashes", "sources",
                                       "roles"])


def lead_near_duplicates(eligible: pd.DataFrame) -> pd.DataFrame:
    """
    Record pairs sharing at least six identical non-flat leads but not the whole first 10 s.

    Parameters
    ----------
    eligible : pd.DataFrame
        Eligible records indexed by record key, with ``lead_sha256_10s``, ``sha256_10s``, ``source``,
        ``role`` and flat flags.

    Returns
    -------
    pd.DataFrame
        One row per pair: keys, shared leads, sources and roles. Lead values shared by more than 50 records
        are left out as degenerate.
    """
    hashes = eligible["lead_sha256_10s"].str.split(",", expand=True)
    hashes.columns = list(LEADS)
    flat = eligible[[f"flat_{lead}" for lead in LEADS]].to_numpy(dtype=bool)
    shared: Counter = Counter()
    for index, lead in enumerate(LEADS):
        values = hashes[lead][~flat[:, index]]
        groups = values.groupby(values).groups
        for members in groups.values():
            if 1 < len(members) <= LEAD_GROUP_CAP:
                shared.update(itertools.combinations(sorted(members), 2))
    rows = []
    for (first, second), n in shared.items():
        pair = eligible.loc[[first, second]]
        if n >= NEAR_LEADS and pair["sha256_10s"].nunique() > 1:
            rows.append({"record_a": first, "record_b": second, "shared_leads": n,
                         "sources": "+".join(sorted(set(pair["source"]))),
                         "roles": "+".join(sorted(set(pair["role"])))})
    return pd.DataFrame(rows, columns=["record_a", "record_b", "shared_leads", "sources", "roles"])
