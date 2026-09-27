"""Quality-filtered, nested training cohorts built from the verified source manifests.

The candidate pool is the one behind ``sampled_100k_plus_labels_v1``: the
verified 500 Hz training union, the strict ten-second Chapman view and the
accepted MIMIC 200k audit, after exact-waveform deduplication. Every candidate
and every labeled PTB-XL training record is scored with
``ecg_experiment.ecg_quality.assess``; records with an exclusion reason are
dropped before sampling. Smaller cohorts are strict subsets of larger ones and
keep the same source mix, so only the number of records changes between them.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .ecg_quality import assess
from .files import sha256_file
from .public_sources import signal_sha256
from .sampled_cohort import source_quotas
from .waveforms import read_record

ROOT = Path(__file__).resolve().parents[1]
UNION = Path("data/processed/training_union_500hz_v1")
CHAPMAN = Path("data/processed/challenge_ecg_views/chapman_strict_10s_v1")
MIMIC = Path("data/processed/mimic_ssl_200k")
MIMIC_RAW = Path("data/raw/mimic-iv-ecg/1.0")
FIELDS = (
    "record_id", "patient_id", "source", "split", "label_scope", "backend",
    "path", "index", "signal_sha256",
)
INPUTS = (
    UNION / "metadata.json",
    UNION / "train_manifest.csv",
    UNION / "labels_fraction1.csv",
    UNION / "labels_fraction0.1.csv",
    UNION / "heldout_references.csv",
    CHAPMAN / "metadata.json",
    CHAPMAN / "manifest.csv",
    MIMIC / "metadata.json",
    MIMIC / "ssl_manifest.csv",
    MIMIC / "audit.sqlite3",
)


def table(path: Path) -> pd.DataFrame:
    """
    Read a repository CSV, keeping patient and record identifiers as text.

    Parameters
    ----------
    path : Path
        Path relative to the repository root.

    Returns
    -------
    pd.DataFrame
        All columns as strings.
    """
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)


def union_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    """
    Split the verified union into labeled PTB-XL rows and other candidates.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]
        Labeled PTB-XL rows, non-MIMIC unlabeled candidates, and the union's
        MIMIC shard pointers keyed by record ID.

    Raises
    ------
    ValueError
        If a union table changed or the clean label selection differs.
    """
    metadata = json.loads((ROOT / UNION / "metadata.json").read_text())
    for name in ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv",
                 "heldout_references.csv"):
        if sha256_file(ROOT / UNION / name) != metadata["table_sha256"][name]:
            raise ValueError(f"Union table hash mismatch: {name}")
    rows = table(UNION / "train_manifest.csv")
    labeled_ids = set(table(UNION / "labels_fraction1.csv")["record_id"])
    if len(labeled_ids) != 15359 or not labeled_ids <= set(rows["record_id"]):
        raise ValueError("Clean PTB label selection changed")

    rows = rows.assign(backend="shard", path=str(UNION) + "/" + rows["shard"], index=rows["shard_index"])
    is_labeled = rows["record_id"].isin(labeled_ids)
    labeled = rows.loc[is_labeled, list(FIELDS)].copy()
    candidates = rows.loc[(rows["source"] != "mimic") & ~is_labeled, list(FIELDS)].copy()
    mimic_shards = rows.loc[rows["source"] == "mimic", ["record_id", "path", "index", "signal_sha256"]]
    return labeled, candidates, mimic_shards.set_index("record_id").to_dict("index")


def mimic_rows(union_lookup: dict[str, dict[str, str]]) -> pd.DataFrame:
    """
    Describe every record accepted by the MIMIC 200k audit.

    Records already stored in the union point to its shard; the rest point to
    the original WFDB files.

    Parameters
    ----------
    union_lookup : dict[str, dict[str, str]]
        Union shard pointers of MIMIC records, from ``union_rows``.

    Returns
    -------
    pd.DataFrame
        One SSL-only row per accepted MIMIC record.

    Raises
    ------
    ValueError
        If the audit tables disagree or a stored waveform hash differs.
    """
    metadata = json.loads((ROOT / MIMIC / "metadata.json").read_text())
    if sha256_file(ROOT / MIMIC / "ssl_manifest.csv") != metadata["manifest_sha256"]:
        raise ValueError("MIMIC accepted manifest hash mismatch")
    rows = table(MIMIC / "ssl_manifest.csv")
    database = ROOT / MIMIC / "audit.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        hashes = dict(connection.execute("SELECT name, signal_sha256 FROM outcomes WHERE status='accepted'"))
    if len(rows) != metadata["accepted_records"] or len(rows) != len(hashes):
        raise ValueError("MIMIC audit and accepted manifest disagree")

    records = []
    for row in rows.itertuples(index=False):
        digest = hashes[row.filename_hr]
        stored = union_lookup.get(row.ecg_id)
        if stored and stored["signal_sha256"] != digest:
            raise ValueError(f"MIMIC union/audit waveform mismatch: {row.ecg_id}")
        records.append({
            "record_id": row.ecg_id, "patient_id": row.patient_id, "source": "mimic", "split": "train",
            "label_scope": "ssl_only", "backend": "shard" if stored else "mimic_wfdb",
            "path": stored["path"] if stored else str(MIMIC_RAW / row.filename_hr),
            "index": stored["index"] if stored else "", "signal_sha256": digest,
        })
    return pd.DataFrame.from_records(records, columns=FIELDS)


def chapman_rows() -> pd.DataFrame:
    """
    Describe the verified strict ten-second Chapman/Shaoxing view.

    Returns
    -------
    pd.DataFrame
        One SSL-only row per accepted Chapman record.

    Raises
    ------
    ValueError
        If the manifest hash or record count changed.
    """
    metadata = json.loads((ROOT / CHAPMAN / "metadata.json").read_text())
    if sha256_file(ROOT / CHAPMAN / "manifest.csv") != metadata["manifest_sha256"]:
        raise ValueError("Chapman manifest hash mismatch")
    rows = table(CHAPMAN / "manifest.csv")
    if len(rows) != metadata["accepted_records"]:
        raise ValueError("Chapman accepted count changed")
    return pd.DataFrame({
        "record_id": rows["ecg_id"], "patient_id": "", "source": "chapman_shaoxing", "split": "train",
        "label_scope": "ssl_only", "backend": "shard", "path": str(CHAPMAN) + "/" + rows["shard"],
        "index": rows["shard_index"], "signal_sha256": rows["signal_sha256"],
    })


def candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Assemble labeled PTB-XL rows and the deduplicated unlabeled candidates.

    Exact waveform copies are removed, preferring curated views over raw
    MIMIC files and never keeping a copy of a labeled record.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, int]
        Labeled rows, unique candidates, and the number of copies removed.

    Raises
    ------
    ValueError
        If two sources reuse a record ID.
    """
    labeled, union_pool, lookup = union_rows()
    candidates = pd.concat([union_pool, chapman_rows(), mimic_rows(lookup)], ignore_index=True)
    if candidates["record_id"].duplicated().any():
        raise ValueError("Source record ID collision")
    candidates = candidates.loc[~candidates["signal_sha256"].isin(labeled["signal_sha256"])]
    before = len(candidates)
    candidates = candidates.drop_duplicates("signal_sha256", keep="first").reset_index(drop=True)
    return labeled.reset_index(drop=True), candidates, before - len(candidates)


@lru_cache(maxsize=8)
def _shard(relative: str) -> np.ndarray:
    """Memory-map one source shard; each worker keeps a few open."""
    return np.load(ROOT / relative, mmap_mode="r", allow_pickle=False)


def read_signal(backend: str, path: str, index: str) -> np.ndarray:
    """
    Read one canonical ECG the same way the sampled-cohort loader does.

    Parameters
    ----------
    backend : str
        ``"shard"`` or ``"mimic_wfdb"``.
    path : str
        Shard or WFDB record path relative to the repository root.
    index : str
        Row inside the shard; empty for WFDB records.

    Returns
    -------
    np.ndarray
        Float32 ``[12, 5000]`` signal in mV.
    """
    if backend == "shard":
        return np.array(_shard(path)[int(index)], copy=True)
    return read_record(ROOT / MIMIC_RAW, str(Path(path).relative_to(MIMIC_RAW)))


def _score(row: tuple[str, str, str, str, str]) -> tuple[str, str, str]:
    """Assess one ``(record_id, backend, path, index, signal_sha256)`` row."""
    record_id, backend, path, index, digest = row
    signal = read_signal(backend, path, index)
    if signal_sha256(signal) != digest:
        raise ValueError(f"Waveform changed: {record_id}")
    reasons, flags = assess(signal)
    return record_id, ";".join(reasons), ";".join(flags)


def score_rows(rows: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    """
    Apply the quality policy to every row, reading and hash-checking each waveform.

    Parameters
    ----------
    rows : pd.DataFrame
        Manifest rows with ``FIELDS`` columns.
    workers : int
        Number of processes.

    Returns
    -------
    pd.DataFrame
        ``record_id``, ``source``, semicolon-separated ``exclusion`` reasons
        and ``review_flags``, in the input order.
    """
    ordered = rows.sort_values(["backend", "path", "index"])
    items = list(ordered[["record_id", "backend", "path", "index", "signal_sha256"]].itertuples(
        index=False, name=None))
    with Pool(workers) as pool:
        scored = pool.map(_score, items, chunksize=256)
    result = pd.DataFrame(scored, columns=["record_id", "exclusion", "review_flags"])
    result = result.set_index("record_id").loc[rows["record_id"]].reset_index()
    result = result.assign(source=rows["source"].to_numpy())
    return result[["record_id", "source", "exclusion", "review_flags"]]


def nested_quotas(capacities: dict[str, int], sizes: list[int],
                  mimic_fraction: float) -> dict[int, dict[str, int]]:
    """
    Source quotas for each cohort size, checked to be nested.

    Parameters
    ----------
    capacities : dict[str, int]
        Clean candidates available per source.
    sizes : list[int]
        Unlabeled cohort sizes, in any order.
    mimic_fraction : float
        Planned MIMIC share, as in ``sampled_cohort.source_quotas``.

    Returns
    -------
    dict[int, dict[str, int]]
        Exact per-source counts keyed by size.

    Raises
    ------
    ValueError
        If a smaller cohort would need more records of a source than a larger one.
    """
    quotas = {size: source_quotas(capacities, size, mimic_fraction) for size in sorted(sizes)}
    ordered = sorted(quotas)
    for smaller, larger in zip(ordered, ordered[1:], strict=False):
        if any(quotas[smaller][source] > quotas[larger][source] for source in capacities):
            raise ValueError(f"Quotas of {smaller} are not nested in {larger}")
    return quotas


def nested_samples(candidates: pd.DataFrame, sizes: list[int], seed: int,
                   mimic_fraction: float) -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, int]]]:
    """
    Draw nested source-stratified samples.

    Each source's candidates are sorted by record ID and permuted once with a
    source-specific random stream. A cohort takes the first ``quota`` records of
    every permutation, so each smaller cohort is contained in the larger ones.

    Parameters
    ----------
    candidates : pd.DataFrame
        Unique clean candidates with ``record_id`` and ``source``.
    sizes : list[int]
        Unlabeled cohort sizes.
    seed : int
        Nonnegative selection seed.
    mimic_fraction : float
        Planned MIMIC share.

    Returns
    -------
    tuple[dict[int, pd.DataFrame], dict[int, dict[str, int]]]
        Shuffled selected rows and per-source quotas, both keyed by size.

    Raises
    ------
    ValueError
        If the seed is negative or record IDs repeat.
    """
    if seed < 0 or candidates["record_id"].duplicated().any():
        raise ValueError("Seed must be nonnegative and record IDs unique")
    capacities = candidates["source"].value_counts().to_dict()
    quotas = nested_quotas(capacities, sizes, mimic_fraction)
    sources = sorted(capacities)
    streams = np.random.SeedSequence(seed).spawn(len(sources) + 1)
    permuted = {}
    for stream, source in zip(streams, sources, strict=False):
        rows = candidates.loc[candidates["source"] == source].sort_values("record_id")
        permuted[source] = rows.iloc[np.random.default_rng(stream).permutation(len(rows))]

    samples = {}
    for size, quota in quotas.items():
        selected = pd.concat([permuted[source].iloc[:quota[source]] for source in sources])
        order = np.random.default_rng(streams[-1]).permutation(len(selected))
        samples[size] = selected.iloc[order].reset_index(drop=True)
    return samples, quotas


def source_shards(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    """
    Bind every referenced source shard to its published SHA-256 and shape.

    Parameters
    ----------
    rows : pd.DataFrame
        Selected manifest rows.

    Returns
    -------
    dict[str, dict[str, object]]
        ``{"sha256", "shape"}`` keyed by shard path.

    Raises
    ------
    ValueError
        If a shard belongs to neither the union nor the Chapman view.
    """
    union_meta = json.loads((ROOT / UNION / "metadata.json").read_text())
    chapman_meta = json.loads((ROOT / CHAPMAN / "metadata.json").read_text())
    chapman_shards = {item["file"]: item for item in chapman_meta["shards"]}
    result = {}
    for relative in sorted(rows.loc[rows["backend"] == "shard", "path"].unique()):
        path = Path(relative)
        if path.parent == UNION:
            result[relative] = union_meta["shards"][path.name]
        elif path.parent == CHAPMAN:
            item = chapman_shards[path.name]
            result[relative] = {"sha256": item["sha256"], "shape": [item["records"], 12, 5000]}
        else:
            raise ValueError(f"Unexpected shard source: {relative}")
    return result
