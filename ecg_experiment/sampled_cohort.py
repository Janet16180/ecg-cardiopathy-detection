"""Deterministic source sampling for a manifest-backed ECG training cohort."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def source_quotas(
    capacities: Mapping[str, int], total: int, mimic_fraction: float = 0.7
) -> dict[str, int]:
    """Allocate an exact sample, capping MIMIC and sharing the rest by source size.

    Parameters
    ----------
    capacities : Mapping[str, int]
        Available deduplicated records per source.
    total : int
        Number of unlabeled ECGs to select.
    mimic_fraction : float
        Maximum planned MIMIC share, relaxed only if other sources lack capacity.

    Returns
    -------
    dict[str, int]
        Exact per-source sample sizes.
    """
    if total < 1 or not 0 <= mimic_fraction <= 1:
        raise ValueError("Invalid sample size or MIMIC fraction")
    if sum(capacities.values()) < total or "mimic" not in capacities:
        raise ValueError("Insufficient candidates or missing MIMIC source")

    other = {source: count for source, count in capacities.items() if source != "mimic"}
    mimic_count = min(capacities["mimic"], round(total * mimic_fraction))
    mimic_count = max(mimic_count, total - sum(other.values()))
    remaining = total - mimic_count
    quotas = dict.fromkeys(capacities, 0)
    quotas["mimic"] = mimic_count
    if not remaining:
        return quotas

    available = sum(other.values())
    ideal = {source: remaining * count / available for source, count in other.items()}
    for source, amount in ideal.items():
        quotas[source] = min(other[source], int(amount))
    missing = total - sum(quotas.values())
    order = sorted(other, key=lambda source: (-(ideal[source] % 1), source))
    for source in order:
        if missing == 0:
            break
        if quotas[source] < other[source]:
            quotas[source] += 1
            missing -= 1
    if missing:
        raise ValueError("Could not allocate exact source quotas")
    return quotas


def sample_sources(
    candidates: pd.DataFrame, total: int, seed: int, mimic_fraction: float = 0.7
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sample each source independently with stable source-specific RNG streams.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Unique candidate rows with ``record_id`` and ``source``.
    total : int
        Exact unlabeled sample size.
    seed : int
        Nonnegative selection seed.
    mimic_fraction : float
        Planned MIMIC share.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, int]]
        Shuffled selected rows and exact source quotas.
    """
    if seed < 0 or candidates["record_id"].duplicated().any():
        raise ValueError("Seed must be nonnegative and record IDs unique")
    capacities = candidates["source"].value_counts().to_dict()
    quotas = source_quotas(capacities, total, mimic_fraction)
    streams = np.random.SeedSequence(seed).spawn(len(quotas) + 1)
    selected = []
    for position, source in enumerate(sorted(quotas)):
        count = quotas[source]
        rows = candidates.loc[candidates["source"] == source].sort_values("record_id")
        if count:
            state = int(streams[position].generate_state(1)[0])
            selected.append(rows.sample(n=count, random_state=state))
    shuffled = pd.concat(selected, ignore_index=True)
    state = int(streams[-1].generate_state(1)[0])
    return shuffled.sample(frac=1, random_state=state).reset_index(drop=True), quotas
