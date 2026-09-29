"""Quality-first nested cohorts from 25k to 1M ECGs, as manifests that reference the source waveforms.

One global order over every candidate decides all tiers: a tier of size N is the first N rows, so every
smaller tier is contained in every larger one. The order follows the user's decisions of 28-29 September 2026:

1. The curated human-read 500 Hz sources: every clean PTB-XL training row first (so the PTB-XL label files are
   the same in every tier), then Ningbo, Chapman, Georgia, CPSC 2018 and CPSC-Extra interleaved in proportion
   to their clean records.
2. MIMIC records downloaded locally, only as many as the 100k tier needs.
3. CODE-15, from the 150k tier on.
4. The remaining local MIMIC records.
5. MIMIC records not downloaded yet, whole patients at a time in a seeded order, marked ``pending`` until the
   quality policy has been applied to them.

Within a source, records without a review flag come first, each group in a seeded random order. SPH is never a
candidate: it is the external test hospital. CODE-15 and MIMIC rows are unlabeled SSL data only.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .challenge_labels import label_table, load_official
from .clean_cohorts import ROOT, UNION, candidate_pool, mimic_rows, union_rows
from .eda import challenge, ptbxl

QUALITY_V1 = ROOT / "outputs/data_quality/clean_cohorts_v1/record_quality.csv.gz"
NINGBO = ROOT / "data/processed/ningbo_clean_v1/rows.csv"
NINGBO_RECEIPT = ROOT / "outputs/data_quality/ningbo_v1/exact_overlap_receipt.json"
CODE15 = ROOT / "data/processed/code15_clean_v1/rows.csv"
MIMIC_RECORDS = ROOT / "outputs/eda/features/mimic_records.parquet"
PTBXL_REFERENCE = ROOT / "outputs/data_quality/ptbxl_reference_hashes_v2.json"
CHALLENGE_ROOT = "data/raw/challenge-2021/1.0.3"
MIMIC_ROOT = "data/raw/mimic-iv-ecg/1.0"
CURATED = ("ptbxl", "ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
CURATED_TIER = 100_000
COLUMNS = ("record_id", "patient_id", "source", "split", "label_scope", "backend", "path", "index",
           "window_start", "signal_sha256", "quality_status", "review_flags", "label_available", "age",
           "male")


def seeded_key(values: pd.Series, seed: int) -> pd.Series:
    """
    Compute a reproducible pseudo-random sort key per value, independent of row order.

    Parameters
    ----------
    values : pd.Series
        Identifiers as text.
    seed : int
        Selection seed.

    Returns
    -------
    pd.Series
        Hex SHA-256 of ``"{seed}:{value}"``.
    """
    return values.map(lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def source_order(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Order one source's rows: records without a review flag first, each group in seeded order.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of one source with ``record_id`` and ``review_flags``.
    seed : int
        Selection seed.

    Returns
    -------
    pd.DataFrame
        The rows in selection order.
    """
    keys = pd.DataFrame({"flagged": rows["review_flags"] != "", "key": seeded_key(rows["record_id"], seed)},
                        index=rows.index)
    return rows.loc[keys.sort_values(["flagged", "key"]).index]


def interleave(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge ordered sources so that every prefix holds them in proportion to their size.

    Row ``i`` of a source with ``n`` rows gets position ``(i + 0.5) / n``; rows are merged by position, with
    ties broken by source name.

    Parameters
    ----------
    parts : dict[str, pd.DataFrame]
        Ordered rows keyed by source.

    Returns
    -------
    pd.DataFrame
        All rows in merged order.
    """
    frames = []
    for name, rows in parts.items():
        position = (np.arange(len(rows)) + 0.5) / len(rows)
        frames.append(rows.assign(_position=position, _source=name))
    merged = pd.concat(frames, ignore_index=True).sort_values(["_position", "_source"], kind="stable")
    return merged.drop(columns=["_position", "_source"]).reset_index(drop=True)


def curated_rows() -> pd.DataFrame:
    """
    Clean curated candidates: PTB-XL training rows, the union's Challenge rows, the Chapman view and Ningbo.

    The v1 quality table scores every union and Chapman row with the same policy. Chapman records dropped by
    the Ningbo family duplicate resolution are removed, and Ningbo contributes its ``use_training`` rows.

    Returns
    -------
    pd.DataFrame
        ``COLUMNS`` rows with ``quality_status`` ``passed``.

    Raises
    ------
    ValueError
        If a v1 quality row is missing.
    """
    labeled, candidates, _ = candidate_pool()
    rows = pd.concat([labeled, candidates[candidates["source"] != "mimic"]], ignore_index=True)
    quality = pd.read_csv(QUALITY_V1, dtype=str, keep_default_na=False).set_index("record_id")
    if not rows["record_id"].isin(quality.index).all():
        raise ValueError("A curated row has no v1 quality score")
    rows = rows.join(quality[["exclusion", "review_flags"]], on="record_id")
    rows = rows[rows["exclusion"] == ""].drop(columns="exclusion")
    receipt = json.loads(NINGBO_RECEIPT.read_text())
    dropped = {f"chapman_shaoxing:{name}" for name in
               receipt["chapman_records_dropped_for_conflict"] + receipt["chapman_records_dropped_as_copy"]}
    rows = rows[~rows["record_id"].isin(dropped)]
    rows = rows.assign(window_start="0", quality_status="passed").join(
        demographics(rows["record_id"], set(labeled["record_id"])), on="record_id")

    ningbo = pd.read_csv(NINGBO, dtype=str, keep_default_na=False)
    ningbo = ningbo[ningbo["use_training"] == "True"]
    ningbo = pd.DataFrame({
        "record_id": "ningbo:" + ningbo["record"], "patient_id": "", "source": "ningbo", "split": "train",
        "label_scope": "ssl_only", "backend": "challenge_wfdb", "path": CHALLENGE_ROOT + "/" + ningbo["path"],
        "index": "", "window_start": "0", "signal_sha256": ningbo["window_sha256"],
        "quality_status": "passed", "review_flags": ningbo["review_flags"],
        "label_available": ningbo["primary"] != "", "age": pd.to_numeric(ningbo["age"], errors="coerce"),
        "male": pd.to_numeric(ningbo["male"], errors="coerce"),
    })
    return pd.concat([rows, ningbo], ignore_index=True)


def demographics(record_ids: pd.Series, labeled_ids: set[str]) -> pd.DataFrame:
    """
    Age, sex and label availability of PTB-XL and Challenge rows from their source metadata.

    Parameters
    ----------
    record_ids : pd.Series
        ``ptbxl:<ecg_id>`` or ``<challenge source>:<record>`` identifiers.
    labeled_ids : set[str]
        PTB-XL rows with a resolved training label.

    Returns
    -------
    pd.DataFrame
        ``age`` (NaN when missing, 0 or the PTB-XL placeholder), ``male`` (1, 0 or NaN) and
        ``label_available`` (a PTB-XL training label, or a defined mapped primary label), indexed by
        record ID.
    """
    meta = ptbxl.load_metadata()
    ptb = pd.DataFrame({"age": meta["age_capped"], "male": (meta["sex"] == 0).astype(float)})
    ptb.index = "ptbxl:" + meta.index.astype(str)
    ptb["label_available"] = ptb.index.isin(labeled_ids)
    headers = challenge.load_headers()
    mapped = label_table(headers["dx_codes"], load_official())["primary"]
    other = pd.DataFrame({"age": headers["age_years"].where(headers["age_years"] > 0),
                          "male": headers["sex"].map({"Male": 1.0, "Female": 0.0}),
                          "label_available": mapped.notna()}, index=headers.index)
    return pd.concat([ptb, other]).reindex(record_ids.unique())


def mimic_local_rows() -> pd.DataFrame:
    """
    Select the accepted MIMIC 200k audit records that pass the v1 quality policy.

    Returns
    -------
    pd.DataFrame
        ``COLUMNS`` rows with ``quality_status`` ``passed``.
    """
    _, _, lookup = union_rows()
    rows = mimic_rows(lookup)
    quality = pd.read_csv(QUALITY_V1, dtype=str, keep_default_na=False).set_index("record_id")
    rows = rows.join(quality[["exclusion", "review_flags"]], on="record_id")
    rows = rows[rows["exclusion"] == ""].drop(columns="exclusion")
    return rows.assign(window_start="0", quality_status="passed", label_available=False, age=np.nan,
                       male=np.nan)


def code15_rows() -> pd.DataFrame:
    """
    CODE-15 exams usable for training after trimming, the 10 s rule, the policy and duplicate resolution.

    Returns
    -------
    pd.DataFrame
        ``COLUMNS`` rows. ``path`` is the HDF5 member of the export archive, ``index`` the storage index and
        ``window_start`` the first observed sample at 400 Hz.
    """
    rows = pd.read_csv(CODE15, dtype=str, keep_default_na=False)
    rows = rows[rows["use_training"] == "True"]
    return pd.DataFrame({
        "record_id": "code15:" + rows["exam_id"], "patient_id": "code15:" + rows["patient_id"],
        "source": "code15", "split": "train", "label_scope": "ssl_only", "backend": "code15_archive",
        "path": rows["hdf5_member"], "index": rows["storage_index"], "window_start": rows["window_start"],
        "signal_sha256": rows["window_sha256"], "quality_status": "passed",
        "review_flags": rows["review_flags"],
        "label_available": False, "age": pd.to_numeric(rows["age"]),
        "male": rows["is_male"].map({"True": 1.0, "False": 0.0}),
    })


def mimic_pending_rows(seed: int) -> pd.DataFrame:
    """
    MIMIC records not downloaded yet, whole patients at a time in seeded order.

    Only patients with no downloaded record are candidates, so no patient is split between the local 200k
    selection and the pending rows.

    Parameters
    ----------
    seed : int
        Selection seed.

    Returns
    -------
    pd.DataFrame
        ``COLUMNS`` rows with ``quality_status`` ``pending`` and no hash, in selection order.
    """
    records = pd.read_parquet(MIMIC_RECORDS).reset_index()
    local = set(records.loc[records["downloaded"], "subject_id"])
    pending = records[~records["subject_id"].isin(local)].copy()
    pending["key"] = seeded_key(pending["subject_id"].astype(str), seed)
    pending = pending.sort_values(["key", "study_id"])
    return pd.DataFrame({
        "record_id": "mimic:" + pending["study_id"].astype(str),
        "patient_id": "mimic:" + pending["subject_id"].astype(str), "source": "mimic", "split": "train",
        "label_scope": "ssl_only", "backend": "mimic_wfdb", "path": MIMIC_ROOT + "/" + pending["path"],
        "index": "", "window_start": "0", "signal_sha256": "", "quality_status": "pending",
        "review_flags": "",
        "label_available": False, "age": np.nan, "male": np.nan,
    }).reset_index(drop=True)


def global_order(curated: pd.DataFrame, mimic_local: pd.DataFrame, code15: pd.DataFrame,
                 pending: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Put every candidate into the single selection order that defines all tiers.

    Parameters
    ----------
    curated : pd.DataFrame
        Output of ``curated_rows``.
    mimic_local : pd.DataFrame
        Output of ``mimic_local_rows``.
    code15 : pd.DataFrame
        Output of ``code15_rows``.
    pending : pd.DataFrame
        Output of ``mimic_pending_rows``, already in order.
    seed : int
        Selection seed.

    Returns
    -------
    pd.DataFrame
        ``COLUMNS`` plus ``order`` (0-based rank) and ``block`` (``curated``, ``mimic_top_up``, ``code15``,
        ``mimic_local``, ``mimic_pending``).
    """
    ptbxl = source_order(curated[curated["source"] == "ptbxl"], seed)
    others = {name: source_order(curated[curated["source"] == name], seed)
              for name in CURATED if name != "ptbxl"}
    curated_ordered = pd.concat([ptbxl, interleave(others)], ignore_index=True)
    mimic_ordered = source_order(mimic_local, seed)
    top_up = max(0, CURATED_TIER - len(curated_ordered))
    blocks = [("curated", curated_ordered), ("mimic_top_up", mimic_ordered.iloc[:top_up]),
              ("code15", source_order(code15, seed)), ("mimic_local", mimic_ordered.iloc[top_up:]),
              ("mimic_pending", pending)]
    ordered = pd.concat([rows[list(COLUMNS)].assign(block=name) for name, rows in blocks], ignore_index=True)
    return ordered.assign(order=np.arange(len(ordered)))


def remove_overlaps(ordered: pd.DataFrame, heldout_hashes: set[str], heldout_ids: set[str],
                    heldout_patients: set[str]) -> pd.DataFrame:
    """
    Drop PTB-XL held-out records and patients, copies of any PTB-XL record elsewhere, and repeated waveforms.

    Parameters
    ----------
    ordered : pd.DataFrame
        Output of ``global_order``.
    heldout_hashes : set[str]
        Window hashes of every PTB-XL record.
    heldout_ids : set[str]
        Record IDs of the PTB-XL development, calibration and test records.
    heldout_patients : set[str]
        Patient IDs of those records.

    Returns
    -------
    pd.DataFrame
        The kept rows with ``order`` renumbered.
    """
    copies_of_ptbxl = (ordered["source"] != "ptbxl") & ordered["signal_sha256"].isin(heldout_hashes)
    repeated = (ordered["signal_sha256"] != "") & ordered["signal_sha256"].duplicated(keep="first")
    heldout = ordered["record_id"].isin(heldout_ids) | ordered["patient_id"].isin(heldout_patients)
    keep = ~copies_of_ptbxl & ~repeated & ~heldout
    kept = ordered[keep].reset_index(drop=True)
    return kept.assign(order=np.arange(len(kept)))


def resolve_pending(ordered: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality results of newly downloaded MIMIC records, keeping the order of everything else.

    Failed records leave the order and the next candidates move up, so tiers stay nested and are still the
    first N rows. Records without a result stay ``pending``.

    Parameters
    ----------
    ordered : pd.DataFrame
        Candidate order with ``quality_status`` and ``signal_sha256``.
    results : pd.DataFrame
        ``record_id``, ``exclusion_reasons`` (empty when passing), ``review_flags`` and ``signal_sha256`` of
        scored pending records; a record whose file could not be read has a reason and no hash.

    Returns
    -------
    pd.DataFrame
        The resolved order with ``order`` renumbered.
    """
    resolved = ordered.set_index("record_id")
    results = results.set_index("record_id")
    scored = resolved.index.intersection(results.index)
    passed = scored[results.loc[scored, "exclusion_reasons"] == ""]
    resolved.loc[passed, "quality_status"] = "passed"
    resolved.loc[passed, "signal_sha256"] = results.loc[passed, "signal_sha256"]
    resolved.loc[passed, "review_flags"] = results.loc[passed, "review_flags"]
    failed = scored.difference(passed)
    resolved = resolved.drop(index=failed).reset_index()
    repeated = (resolved["signal_sha256"] != "") & resolved["signal_sha256"].duplicated(keep="first")
    resolved = resolved[~repeated].reset_index(drop=True)
    return resolved.assign(order=np.arange(len(resolved)))


def heldout_references() -> tuple[set[str], set[str], set[str]]:
    """
    Hashes of all PTB-XL records, and record and patient IDs of every PTB-XL record outside folds 1-8.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        Window hashes of all 21,799 PTB-XL records, and record and patient IDs of folds 9 and 10, which hold
        the development, calibration and test partitions.
    """
    hashes = set(json.loads(PTBXL_REFERENCE.read_text())["hashes"])
    meta = ptbxl.load_metadata()
    held = meta[meta["strat_fold"] >= 9]
    references = pd.read_csv(ROOT / UNION / "heldout_references.csv", dtype=str)
    ids = set("ptbxl:" + held.index.astype(str)) | set(references["record_id"])
    patients = set("ptbxl:" + held["patient_id"].astype(str)) | set(references["patient_id"])
    return hashes, ids, patients
