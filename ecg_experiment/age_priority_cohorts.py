"""Select reproducible training cohorts by label evidence, then age preference."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

REQUIRED = {
    "record_id", "source", "target", "age", "label_evidence_rank", "label_evidence",
    "split", "quality_passed", "signal_sha256",
}


def verify_identities(rows: pd.DataFrame) -> None:
    """Require unique record and canonical waveform identities before selection."""
    if rows["record_id"].isna().any() or rows["record_id"].duplicated().any():
        raise ValueError("Candidate identities must be present and unique")
    if rows["signal_sha256"].isna().any() or rows["signal_sha256"].duplicated().any():
        raise ValueError("Candidate waveforms must already be deduplicated")
    if not rows["signal_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Missing or malformed canonical waveform identity")


def select_cohort(candidates: pd.DataFrame, size: int, seed: int,
                  prefer_young: bool) -> pd.DataFrame:
    """Select only admitted records, prioritizing label evidence before age.

    Parameters
    ----------
    candidates : pd.DataFrame
        Already verified, deduplicated training records. Evidence ranks are assigned
        from documented annotation provenance upstream; smaller ranks have priority.
        Only ranks 0, 1 and 2 are admitted. A rank is not a measured label error rate.
    size : int
        Maximum number of selected ECGs; never fills with ineligible records.
    seed : int
        Seed for deterministic record-ID hash ordering within equal priorities.
    prefer_young : bool
        Prefer ages 18 through 25 within each evidence rank, then older adults in
        ascending age. Unknown ages and minors are excluded from the younger arm.
        The comparison arm has no age-based selection or exclusion.

    Returns
    -------
    pd.DataFrame
        Selected original columns plus a one-based selection rank.

    Raises
    ------
    ValueError
        If input records are unverified, duplicated, reserved, or have invalid targets
        or evidence. This function does not replace upstream eligibility checks.
    """
    if size < 1 or not set(candidates.columns) >= REQUIRED:
        raise ValueError("Positive size and complete candidate schema required")
    rows = candidates.copy()
    if rows.empty:
        return rows.assign(selection_rank=pd.Series(dtype=int))
    verify_identities(rows)
    if not rows["split"].eq("train").all():
        raise ValueError("Reserved records cannot enter cohort selection")
    if not rows["quality_passed"].isin([True]).all():
        raise ValueError("Signal quality eligibility must be established upstream")
    if not rows["target"].isin([0, 1]).all():
        raise ValueError("Only defined binary proxy labels can enter supervised cohorts")
    if not rows["label_evidence_rank"].isin([0, 1, 2]).all():
        raise ValueError("Unadmitted label evidence rank")
    if rows["label_evidence"].isna().any() or rows["label_evidence"].eq("").any():
        raise ValueError("Label provenance must accompany every candidate")
    age = pd.to_numeric(rows["age"], errors="raise")
    if ((age < 0) | (age > 120) | np.isinf(age)).any():
        raise ValueError("Age placeholders must be normalized upstream")
    rows["_tie"] = rows["record_id"].map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    order = ["label_evidence_rank"]
    if prefer_young:
        rows = rows[age.ge(18) & age.notna()].copy()
        selected_age = age.loc[rows.index]
        rows["_age_group"] = (~selected_age.between(18, 25)).astype(int)
        rows["_older_age"] = selected_age.where(selected_age > 25, 0)
        order.extend(["_age_group", "_older_age"])
    order.extend(["_tie", "record_id"])
    selected = rows.sort_values(order, kind="stable").head(size)
    selected = selected[list(candidates.columns)].reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected
