"""Cohorts v3: the v2 quality-first order without any Challenge test or calibration record.

Cohorts v2 were built before the Challenge record split (``docs/challenge-splits-v1.md``) and hold records of
its test and calibration groups. V3 removes them from the curated candidates before the v2 order is built, so
every v2 tier rule still holds: the MIMIC top-up still fills the 100k tier and CODE-15 still enters at 150k.
Duplicate groups never span split groups, so dropping every evaluation record by ID also drops every exact
copy of one inside the Challenge sources; a candidate of any source whose waveform hash equals an evaluation
record's hash is dropped as well.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import clean_cohorts_v2 as v2

CHALLENGE_SPLITS = v2.ROOT / "data/processed/challenge_splits_v1/rows.csv"
CHALLENGE_SOURCES = ("ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
SPLIT_COLUMNS = ("source", "record", "split", "duplicate_group", "signal_sha256", "window_sha256")


def read_splits(path: Path = CHALLENGE_SPLITS) -> pd.DataFrame:
    """
    Read the Challenge split without its label columns, so no evaluation label is ever loaded.

    Parameters
    ----------
    path : Path
        The split's ``rows.csv``.

    Returns
    -------
    pd.DataFrame
        ``SPLIT_COLUMNS`` as strings.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=list(SPLIT_COLUMNS))


def challenge_references(rows: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    """
    Record IDs of the split, and record IDs and waveform hashes of its test and calibration groups.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``read_splits``.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        Every split record ID, the evaluation record IDs, and the evaluation waveform hashes.

    Raises
    ------
    ValueError
        If a duplicate group spans more than one split group.
    """
    if (rows.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("A Challenge duplicate group spans split groups")
    ids = rows["source"] + ":" + rows["record"]
    held = rows["split"] != "train"
    hashes = (set(rows.loc[held, "signal_sha256"]) | set(rows.loc[held, "window_sha256"])) - {""}
    return set(ids), set(ids[held]), hashes


def exclude_challenge_evaluation(curated: pd.DataFrame, known: set[str], held_ids: set[str],
                                 held_hashes: set[str]) -> pd.DataFrame:
    """
    Drop curated candidates in the Challenge test or calibration groups, or with one of their waveforms.

    Parameters
    ----------
    curated : pd.DataFrame
        Output of ``clean_cohorts_v2.curated_rows``.
    known : set[str]
        Every record ID of the Challenge split.
    held_ids : set[str]
        Record IDs of the test and calibration groups.
    held_hashes : set[str]
        Waveform hashes of the test and calibration groups.

    Returns
    -------
    pd.DataFrame
        The kept candidates.

    Raises
    ------
    ValueError
        If a Challenge candidate is missing from the split.
    """
    challenge = curated["source"].isin(CHALLENGE_SOURCES)
    if (challenge & ~curated["record_id"].isin(known)).any():
        raise ValueError("A Challenge candidate is missing from the Challenge split")
    held = curated["record_id"].isin(held_ids) | curated["signal_sha256"].isin(held_hashes)
    return curated[~held].reset_index(drop=True)


def candidate_order(seed: int, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble, order and deduplicate every candidate with the v2 rules, after the Challenge exclusion.

    Parameters
    ----------
    seed : int
        Selection seed.
    splits : pd.DataFrame
        Output of ``read_splits``.

    Returns
    -------
    pd.DataFrame
        The global order (``clean_cohorts_v2.COLUMNS`` plus ``block`` and ``order``).
    """
    known, held_ids, held_hashes = challenge_references(splits)
    curated = exclude_challenge_evaluation(v2.curated_rows(), known, held_ids, held_hashes)
    ordered = v2.global_order(curated, v2.mimic_local_rows(), v2.code15_rows(), v2.mimic_pending_rows(seed),
                              seed)
    ordered = ordered[~ordered["signal_sha256"].isin(held_hashes)]
    return v2.remove_overlaps(ordered, *v2.heldout_references())
