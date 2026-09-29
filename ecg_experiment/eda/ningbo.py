"""Load Ningbo (PhysioNet Challenge 2021) headers and signals and compute the cached analyses.

Ningbo First Hospital contributed 34,905 twelve-lead ECGs to the Challenge 2021 release, recorded at 500 Hz
and labeled with SNOMED codes by physicians (one labels, a second validates, a senior physician resolves
disagreements, per the source database). Records are WFDB ``.hea``/``.mat`` pairs under
``training/ningbo/g1`` to ``g35``. The headers carry no patient identifier.
"""

import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import wfdb

from ecg_experiment.ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS, assess
from ecg_experiment.eda.challenge import load_headers as challenge_headers
from ecg_experiment.eda.challenge import parse_header
from ecg_experiment.eda.challenge import read_signal as challenge_read_signal
from ecg_experiment.eda.ptbxl import OUTPUT_DIR, ROOT
from ecg_experiment.eda.signals import LEADS, canonical_order, compute_features, record_summary
from ecg_experiment.public_sources import signal_sha256

RAW_ROOT = ROOT / "data/raw/challenge-2021/1.0.3"
NINGBO_DIR = RAW_ROOT / "training/ningbo"
HEADER_CACHE = OUTPUT_DIR / "features" / "ningbo_headers.parquet"
FEATURE_CACHE = OUTPUT_DIR / "features" / "ningbo.parquet"
QUALITY_CACHE = OUTPUT_DIR / "features" / "ningbo_quality.parquet"
LEAD_CACHE = OUTPUT_DIR / "features" / "ningbo_leads.parquet"
FINGERPRINT_CACHE = OUTPUT_DIR / "features" / "ningbo_fingerprints.npy"
CHAPMAN_FINGERPRINT_CACHE = OUTPUT_DIR / "features" / "chapman_fingerprints.npy"


def load_headers() -> pd.DataFrame:
    """
    Parse every Ningbo header, caching the result.

    Returns
    -------
    pd.DataFrame
        One row per record indexed by record name, with the ``parse_header`` fields, ``path`` relative to
        the release root, ``group`` (``g1`` to ``g35``), ``has_signal`` (the ``.mat`` file exists),
        ``age_years``, ``duration_s`` and ``dx_codes``.
    """
    if HEADER_CACHE.exists():
        table = pd.read_parquet(HEADER_CACHE)
        table["dx_codes"] = table["dx_codes"].apply(list)
        return table
    rows = []
    for header in sorted(NINGBO_DIR.rglob("*.hea")):
        row = parse_header(header)
        row["path"] = str(header.with_suffix("").relative_to(RAW_ROOT))
        row["group"] = header.parent.name
        row["has_signal"] = header.with_suffix(".mat").exists()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("record")
    table["age_years"] = pd.to_numeric(table["age"], errors="coerce")
    table["duration_s"] = table["samples"] / table["fs"]
    table["dx_codes"] = table["dx"].str.split(",").apply(
        lambda codes: [code.strip() for code in codes if code])
    HEADER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(HEADER_CACHE)
    return table


def read_signal(path: str) -> tuple[np.ndarray, int]:
    """
    Read one Ningbo record in mV and canonical lead order.

    Parameters
    ----------
    path : str
        Record path relative to the release root, without extension.

    Returns
    -------
    tuple[np.ndarray, int]
        Array of shape ``(samples, 12)`` and the sampling rate.
    """
    signal, fields = wfdb.rdsamp(str(RAW_ROOT / path))
    return canonical_order(signal, fields["sig_name"]), int(fields["fs"])


def ningbo_summary(headers: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """
    Per-record signal summary of every record, joined with the header fields.

    Parameters
    ----------
    headers : pd.DataFrame
        Output of ``load_headers``, restricted to records whose signal file exists.
    workers : int
        Number of processes.

    Returns
    -------
    pd.DataFrame
        ``record_summary`` columns plus header fields, indexed by record name.
    """
    items = [(record, row.path) for record, row in headers.iterrows()]
    summary = record_summary(compute_features(items, read_signal, FEATURE_CACHE, workers))
    return summary.join(headers.drop(columns=["samples", "fs", "duration_s"]))


def constant_run(lead: np.ndarray) -> tuple[int, int, float]:
    """
    Locate the longest run of identical consecutive samples in one lead.

    Parameters
    ----------
    lead : np.ndarray
        One-dimensional signal.

    Returns
    -------
    tuple[int, int, float]
        Start sample, length in samples and the repeated value of the first longest run.
    """
    starts = np.flatnonzero(np.r_[True, lead[1:] != lead[:-1]])
    lengths = np.diff(np.r_[starts, len(lead)])
    best = int(lengths.argmax())
    return int(starts[best]), int(lengths[best]), float(lead[starts[best]])


def edge_zero_runs(signal: np.ndarray) -> tuple[int, int]:
    """
    Count the leading and trailing samples at which every lead is exactly zero.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, leads)``.

    Returns
    -------
    tuple[int, int]
        Leading and trailing all-lead zero samples; both equal the length for an all-zero record.
    """
    nonzero = np.flatnonzero((signal != 0).any(axis=1))
    if len(nonzero) == 0:
        return len(signal), len(signal)
    return int(nonzero[0]), int(len(signal) - 1 - nonzero[-1])


def fingerprint(signal: np.ndarray, step: int = 10) -> np.ndarray:
    """
    Scale- and offset-invariant summary of lead II for near-duplicate search.

    Parameters
    ----------
    signal : np.ndarray
        Canonical array of shape ``(5000, 12)`` in mV.
    step : int
        Samples averaged into one value (10 gives 50 Hz at 500 Hz).

    Returns
    -------
    np.ndarray
        Float16 z-scored lead II means of length ``5000 // step``; all zero for a constant lead.
    """
    lead = np.nan_to_num(signal[:, 1]).reshape(-1, step).mean(axis=1)
    spread = lead.std()
    scaled = (lead - lead.mean()) / spread if spread > 0 else np.zeros_like(lead)
    return scaled.astype(np.float16)


def _scan(item: tuple[str, str]) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    record, path = item
    signal, _ = read_signal(path)
    window = signal[:5000].T.astype(np.float32)
    reasons, flags = assess(window) if window.shape == (12, 5000) else (["short"], [])
    microvolts = np.round(np.nan_to_num(signal, nan=-99999) * 1000).astype(np.int32)
    leading, trailing = edge_zero_runs(signal)
    row = {"record": record, "signal_sha256": hashlib.sha256(microvolts.tobytes()).hexdigest(),
           "window_sha256": signal_sha256(window), "edge_zero_leading": leading,
           "edge_zero_trailing": trailing,
           **{name: name in reasons for name in (*EXCLUSION_REASONS, "short")},
           **{name: name in flags for name in REVIEW_FLAGS}}
    leads = []
    for index, name in enumerate(LEADS):
        start, length, value = constant_run(signal[:, index])
        leads.append({"record": record, "lead": name, "run_start": start, "run_length": length,
                      "run_value": value, "zero_samples": int((signal[:, index] == 0).sum())})
    return row, leads, fingerprint(signal)


def record_scan(headers: pd.DataFrame, workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Scan every record once: quality policy, hashes, constant runs and fingerprints, or load the caches.

    The quality policy is applied to the first 10 s, which for Ningbo is the whole record.
    ``signal_sha256`` hashes the int32 microvolt samples like ``ecg_experiment.eda.challenge.signal_hash``;
    ``window_sha256`` hashes the float32 ``(12, 5000)`` window like ``ecg_experiment.public_sources`` and so
    joins the training union, the Chapman view, the MIMIC audit and the PTB-XL reference hashes.

    Parameters
    ----------
    headers : pd.DataFrame
        Output of ``load_headers``, restricted to records whose signal file exists.
    workers : int
        Number of processes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
        Record table (one boolean column per exclusion reason, ``short`` and review flag, ``excluded``,
        both hashes and the edge zero runs) indexed by record; lead table with each lead's longest
        constant run and its exact-zero sample count; and fingerprints in record-table order.
    """
    if QUALITY_CACHE.exists():
        records = pd.read_parquet(QUALITY_CACHE)
        return records, pd.read_parquet(LEAD_CACHE), np.load(FINGERPRINT_CACHE)
    items = [(record, row.path) for record, row in headers.iterrows()]
    with Pool(workers) as pool:
        results = pool.map(_scan, items, chunksize=64)
    records = pd.DataFrame([result[0] for result in results]).set_index("record")
    records["excluded"] = records[[*EXCLUSION_REASONS, "short"]].any(axis=1)
    leads = pd.DataFrame([lead for result in results for lead in result[1]])
    fingerprints = np.stack([result[2] for result in results])
    QUALITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(QUALITY_CACHE)
    leads.to_parquet(LEAD_CACHE)
    np.save(FINGERPRINT_CACHE, fingerprints)
    return records, leads, fingerprints


def _chapman_fingerprint(path: str) -> np.ndarray:
    signal, _ = challenge_read_signal(path)
    return fingerprint(signal[:5000])


def chapman_fingerprints(workers: int = 4) -> tuple[pd.Index, np.ndarray]:
    """
    Fingerprints of every Chapman/Shaoxing record, cached.

    Parameters
    ----------
    workers : int
        Number of processes.

    Returns
    -------
    tuple[pd.Index, np.ndarray]
        Challenge record IDs (``chapman_shaoxing:JS...``) and fingerprints in the same order.
    """
    headers = challenge_headers()
    headers = headers[headers["source"] == "chapman_shaoxing"]
    if CHAPMAN_FINGERPRINT_CACHE.exists():
        return headers.index, np.load(CHAPMAN_FINGERPRINT_CACHE)
    paths = [f"{row.source}:{row.path}" for row in headers.itertuples()]
    with Pool(workers) as pool:
        fingerprints = np.stack(pool.map(_chapman_fingerprint, paths, chunksize=64))
    np.save(CHAPMAN_FINGERPRINT_CACHE, fingerprints)
    return headers.index, fingerprints


def near_duplicates(names: pd.Index, fingerprints: np.ndarray, threshold: float = 0.99,
                    chunk: int = 2000) -> pd.DataFrame:
    """
    Pairs of records whose fingerprints correlate at or above a threshold at zero lag.

    Parameters
    ----------
    names : pd.Index
        Record name of each fingerprint row.
    fingerprints : np.ndarray
        Output of ``fingerprint`` stacked per record.
    threshold : float
        Smallest Pearson correlation reported.
    chunk : int
        Rows compared per matrix product.

    Returns
    -------
    pd.DataFrame
        ``first``, ``second`` (first before second in ``names`` order) and ``correlation``.
    """
    scaled = fingerprints.astype(np.float32) / np.sqrt(fingerprints.shape[1])
    pairs = []
    for start in range(0, len(scaled), chunk):
        block = scaled[start:start + chunk] @ scaled.T
        rows, columns = np.nonzero(block >= threshold)
        keep = columns > rows + start
        pairs.append(pd.DataFrame({"first": names[rows[keep] + start], "second": names[columns[keep]],
                                   "correlation": block[rows[keep], columns[keep]]}))
    return pd.concat(pairs, ignore_index=True)
