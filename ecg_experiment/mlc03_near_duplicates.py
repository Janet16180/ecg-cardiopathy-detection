"""
MLC-03 Part A item 8: near-duplicate waveforms beyond exact hashes.

Two searches, both on fingerprints: each lead high-passed at 0.5 Hz (zero-phase second order), resampled
from 500 to 50 Hz in one polyphase pass, and z-scored; a lead without variation becomes zeros.

1. Zero lag, every pair. The fingerprint of leads I and II of the first 10 s (2 x 500 values) is scaled to
   unit length, so the dot product of two records is the mean of their two lead correlations when both
   leads vary. All pairs are scored by matrix products in blocks of rows; every pair at or above a floor
   and each record's nearest neighbours are kept. This finds copies that were rescaled, offset, filtered,
   requantised or had some leads changed, which an exact hash misses.
2. Shifted copies. A copy that starts at another time shares runs of RR intervals. Each record's R peaks
   (``measurement_extractor042.detect_r_peaks``; the whole raw recording for records longer than 10 s) give
   keys of three consecutive RR intervals in 8 ms bins, with a second key set shifted by half a bin. Two
   records that share at least ``MIN_VOTES`` keys at a consistent lag become a candidate, and the candidate
   is verified by the correlation of their fingerprints at that lag over at least ``MIN_OVERLAP_S`` seconds.
   Keys held by more than ``MAX_BUCKET`` records (very regular rhythms) are dropped, so shifted copies of
   such rhythms can be missed.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from ecg_experiment.eda.signals import LEADS, canonical_order
from ecg_experiment.measurement_extractor042 import SOS_CLEAN, detect_r_peaks
from ecg_experiment.mlc01_dataset import read_signal
from ecg_experiment.mlc02_waveforms import closed_keys, guard_path, init_worker
from ecg_experiment.mlc03_signal_quality import FS, SAMPLES, fill_missing
from ecg_experiment.mlc03_signal_runs import check_roles, chunk_tasks, open_block, read_checked
from ecg_experiment.resample import resample_full

FINGERPRINT_RATE = 50
DECIMATION = FS // FINGERPRINT_RATE
DENSE_LEADS = ("I", "II")
VERIFY_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
SOS_FINGERPRINT = butter(2, 0.5, btype="high", fs=FS, output="sos")
MIN_STD_MV = 1e-3
RR_QUANTUM = 4
RR_SHIFTS = (0, 2)
MAX_BUCKET = 200
MIN_VOTES = 2
LAG_TOLERANCE = 8
MIN_OVERLAP_S = 4.0


def fingerprint(signal: np.ndarray, leads: Iterable[str] = VERIFY_LEADS) -> np.ndarray:
    """
    Z-scored 50 Hz fingerprint of selected leads.

    Parameters
    ----------
    signal : np.ndarray
        Float ``(12, samples)`` in mV at 500 Hz, ``LEADS`` order, NaN for missing samples; ``samples`` is a
        multiple of 10.
    leads : Iterable[str]
        Leads to keep, in order.

    Returns
    -------
    np.ndarray
        Float32 ``(len(leads), samples // 10)``; a lead whose 50 Hz standard deviation is below 1 uV is zeros.
    """
    rows = [LEADS.index(lead) for lead in leads]
    high = sosfiltfilt(SOS_FINGERPRINT, fill_missing(np.asarray(signal, dtype=np.float64)[rows]), axis=1)
    low = resample_full(high, 1, DECIMATION).astype(np.float64)
    centred = low - low.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return np.where(spread >= MIN_STD_MV, centred / np.where(spread >= MIN_STD_MV, spread, 1.0), 0.0).astype(
        np.float32)


def dense_vector(window: np.ndarray) -> np.ndarray:
    """
    Unit-length zero-lag fingerprint of leads I and II of a 10 s window.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` in mV.

    Returns
    -------
    np.ndarray
        Float32 vector of 1,000 values with norm 1, or zeros when neither lead varies.
    """
    vector = fingerprint(window, DENSE_LEADS).ravel()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def dense_pairs(queries: np.ndarray, references: np.ndarray, floor: float, same_set: bool,
                neighbours: int = 5, block: int = 2048) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Score every query against every reference at zero lag.

    Parameters
    ----------
    queries, references : np.ndarray
        Float32 ``(n, d)`` unit vectors of ``dense_vector``.
    floor : float
        Keep every pair with a score at or above this value.
    same_set : bool
        True when ``queries`` and ``references`` are the same rows: self-pairs are ignored and each pair is
        kept once (query index below reference index).
    neighbours : int
        Nearest references to keep per query.
    block : int
        Query rows per matrix product.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        Pairs (``query``, ``reference``, ``score``), and the indices and scores of each query's nearest
        references, best first.
    """
    found = []
    nearest_index = np.empty((len(queries), neighbours), dtype=np.int64)
    nearest_score = np.empty((len(queries), neighbours), dtype=np.float32)
    for start in range(0, len(queries), block):
        scores = queries[start : start + block] @ references.T
        rows = np.arange(start, start + len(scores))
        if same_set:
            scores[np.arange(len(scores)), rows] = -np.inf
        top = np.argpartition(-scores, neighbours, axis=1)[:, :neighbours]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        nearest_index[rows] = np.take_along_axis(top, order, axis=1)
        nearest_score[rows] = np.take_along_axis(top_scores, order, axis=1)
        query, reference = np.nonzero(scores >= floor)
        query = query + start
        keep = reference > query if same_set else np.ones(len(query), dtype=bool)
        found.append(pd.DataFrame({"query": query[keep], "reference": reference[keep],
                                   "score": scores[query[keep] - start, reference[keep]]}))
    empty = pd.DataFrame(columns=["query", "reference", "score"])
    pairs = pd.concat(found, ignore_index=True) if found else empty
    return pairs, nearest_index, nearest_score


def r_peaks(signal: np.ndarray) -> np.ndarray:
    """
    R peaks of a recording of any length by the multi-lead detector.

    Parameters
    ----------
    signal : np.ndarray
        Float ``(12, samples)`` in mV at 500 Hz, NaN for missing samples.

    Returns
    -------
    np.ndarray
        Sorted sample locations.
    """
    clean = sosfiltfilt(SOS_CLEAN, fill_missing(np.asarray(signal, dtype=np.float64)).T, axis=0)
    return detect_r_peaks(clean)


def rr_keys(peaks: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    """
    Keys of three consecutive RR intervals, in two quantisations, with the time of their first beat.

    Parameters
    ----------
    peaks : np.ndarray
        Sorted R-peak locations.

    Returns
    -------
    list[tuple[tuple[int, int, int, int], int]]
        ``((shift, bin1, bin2, bin3), first_peak)`` for every run of three intervals and both shifts.
    """
    rr = np.diff(peaks)
    keys = []
    for position in range(len(rr) - 2):
        triple = rr[position : position + 3]
        for shift in RR_SHIFTS:
            bins = tuple(int(value) for value in (triple + shift) // RR_QUANTUM)
            keys.append(((shift, *bins), int(peaks[position])))
    return keys


def shifted_candidates(keys: dict[int, list[tuple[tuple[int, int, int, int], int]]],
                       max_bucket: int = MAX_BUCKET, min_votes: int = MIN_VOTES) -> pd.DataFrame:
    """
    Record pairs that share RR keys at a consistent lag.

    Parameters
    ----------
    keys : dict[int, list[tuple[tuple[int, int, int, int], int]]]
        ``rr_keys`` per record id.
    max_bucket : int
        Keys held by more records than this are dropped.
    min_votes : int
        Least number of shared keys at one lag.

    Returns
    -------
    pd.DataFrame
        ``first``, ``second`` (record ids, first below second), ``lag`` (samples to add to a time of
        ``first`` to reach the same beat in ``second``) and ``votes``.
    """
    buckets: dict[tuple[int, int, int, int], dict[int, int]] = defaultdict(dict)
    for record, entries in keys.items():
        for key, position in entries:
            buckets[key].setdefault(record, position)
    votes: dict[tuple[int, int, int], int] = defaultdict(int)
    for members in buckets.values():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        items = sorted(members.items())
        for first_index, (first, first_position) in enumerate(items):
            for second, second_position in items[first_index + 1 :]:
                lag = second_position - first_position
                votes[(first, second, int(round(lag / LAG_TOLERANCE)))] += 1
    rows = [{"first": first, "second": second, "lag": lag_bin * LAG_TOLERANCE, "votes": count}
            for (first, second, lag_bin), count in votes.items() if count >= min_votes]
    frame = pd.DataFrame(rows, columns=["first", "second", "lag", "votes"])
    return frame.sort_values(["first", "second", "votes"], ascending=[True, True, False]).drop_duplicates(
        ["first", "second"]).reset_index(drop=True)


def lagged_correlation(first: np.ndarray, second: np.ndarray, lag: int) -> tuple[float, float, int]:
    """
    Mean lead correlation of two fingerprints with ``second`` shifted by a lag.

    Sample ``t`` of ``first`` is compared with sample ``t + lag`` of ``second``. Only leads that vary in
    both fingerprints over the overlap count.

    Parameters
    ----------
    first, second : np.ndarray
        Float ``(leads, samples)`` fingerprints at 50 Hz with the same leads.
    lag : int
        Lag in 50 Hz samples.

    Returns
    -------
    tuple[float, float, int]
        Mean Pearson correlation (NaN when no lead varies in both), the ratio of standard deviations of
        ``second`` to ``first`` averaged over those leads, and the overlap in samples.
    """
    start = max(0, -lag)
    stop = min(first.shape[1], second.shape[1] - lag)
    if stop - start < 2:
        return np.nan, np.nan, max(stop - start, 0)
    a = first[:, start:stop].astype(np.float64)
    b = second[:, start + lag : stop + lag].astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    spread_a, spread_b = a.std(axis=1), b.std(axis=1)
    valid = (spread_a > 1e-6) & (spread_b > 1e-6)
    if not valid.any():
        return np.nan, np.nan, stop - start
    correlation = (a[valid] * b[valid]).mean(axis=1) / (spread_a[valid] * spread_b[valid])
    return float(correlation.mean()), float((spread_b[valid] / spread_a[valid]).mean()), stop - start


def best_lag(first: np.ndarray, second: np.ndarray, centre: int,
             radius: int) -> tuple[int, float, float, int]:
    """
    Lag with the highest mean correlation near an expected lag.

    Parameters
    ----------
    first, second : np.ndarray
        Fingerprints at 50 Hz.
    centre : int
        Expected lag in 50 Hz samples.
    radius : int
        Lags ``centre - radius`` to ``centre + radius`` are tried.

    Returns
    -------
    tuple[int, float, float, int]
        Lag, correlation, standard-deviation ratio and overlap of the best lag.
    """
    best = (centre, -np.inf, np.nan, 0)
    for lag in range(centre - radius, centre + radius + 1):
        correlation, ratio, overlap = lagged_correlation(first, second, lag)
        if np.isfinite(correlation) and correlation > best[1]:
            best = (lag, correlation, ratio, overlap)
    return best


def record_fingerprints(signal: np.ndarray) -> dict[str, Any]:
    """
    Every fingerprint of one recording.

    Parameters
    ----------
    signal : np.ndarray
        Float ``(12, samples)`` in mV at 500 Hz; for a cached window ``samples`` is 5,000.

    Returns
    -------
    dict[str, Any]
        ``dense`` (``dense_vector`` of the first 10 s), ``verify`` (``fingerprint`` of every sample, float16)
        and ``keys`` (``rr_keys`` of every sample).
    """
    return {"dense": dense_vector(signal[:, :SAMPLES]),
            "verify": fingerprint(signal).astype(np.float16),
            "keys": rr_keys(r_peaks(signal))}


def fingerprint_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """
    Fingerprints of one chunk of cached windows (the format of ``mlc03_signal_runs.chunk_tasks``).

    Parameters
    ----------
    task : dict[str, Any]
        Chunk task with ``block``, ``rows``, ``positions``, ``keys``, ``hashes`` and ``allow_test``.

    Returns
    -------
    dict[str, Any]
        ``positions``, ``dense`` ``(n, 1000)``, ``verify`` ``(n, 8, 500)`` and the RR-key table ``rr``
        (columns position, shift, three bins, first peak).
    """
    array = open_block(task["block"], task["allow_test"])
    dense, verify, keys = [], [], []
    for row, position, key, expected in zip(task["rows"], task["positions"], task["keys"], task["hashes"],
                                            strict=True):
        prints = record_fingerprints(read_checked(array, int(row), expected, key))
        dense.append(prints["dense"])
        verify.append(prints["verify"])
        keys.extend((int(position), *entry[0], entry[1]) for entry in prints["keys"])
    del array
    return {"positions": np.asarray(task["positions"], dtype=np.int64), "dense": np.stack(dense),
            "verify": np.stack(verify), "rr": np.asarray(keys, dtype=np.int64).reshape(-1, 6)}


def run_fingerprint_chunk(task: dict[str, Any]) -> str:
    """
    Write the fingerprints of one chunk unless its file exists.

    Parameters
    ----------
    task : dict[str, Any]
        Chunk task plus ``directory``.

    Returns
    -------
    str
        Chunk name.
    """
    path = Path(task["directory"]) / f"{task['name']}.npz"
    if not path.exists():
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez(temporary, **fingerprint_chunk(task))
        os.replace(temporary, path)
    return task["name"]


def run_fingerprints(directory: Path, index: pd.DataFrame, workers: int, allow_test: bool = False) -> None:
    """
    Fingerprint every cached window of an index in parallel, resuming from finished chunks.

    Parameters
    ----------
    directory : Path
        Chunk directory.
    index : pd.DataFrame
        Output of ``mlc03_signal_runs.read_index``.
    workers : int
        Processes.
    allow_test : bool
        Passed to the role checks.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [{**task, "directory": str(directory)} for task in chunk_tasks(index, allow_test=allow_test)]
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(run_fingerprint_chunk, tasks):
            pass


def load_fingerprints(directory: Path, index: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Read the chunk files back in the order of the index.

    Parameters
    ----------
    directory : Path
        Chunk directory.
    index : pd.DataFrame
        The index that was fingerprinted.

    Returns
    -------
    dict[str, np.ndarray]
        ``dense`` and ``verify`` in index order, and ``rr`` with the index row (not the cache position) in
        its first column.

    Raises
    ------
    ValueError
        If the chunks do not cover the index exactly.
    """
    parts = [dict(np.load(path)) for path in sorted(directory.glob("*.npz")) if ".tmp" not in path.name]
    row_of = pd.Series(np.arange(len(index)), index=index["position"].to_numpy())
    positions = np.concatenate([part["positions"] for part in parts])
    if len(positions) != len(index) or not set(positions) == set(row_of.index):
        raise ValueError("fingerprint chunks do not cover the index")
    rows = row_of.loc[positions].to_numpy()
    dense = np.empty((len(index), parts[0]["dense"].shape[1]), dtype=np.float32)
    verify = np.empty((len(index), *parts[0]["verify"].shape[1:]), dtype=np.float16)
    dense[rows] = np.concatenate([part["dense"] for part in parts])
    verify[rows] = np.concatenate([part["verify"] for part in parts])
    rr = np.concatenate([part["rr"] for part in parts])
    rr[:, 0] = row_of.loc[rr[:, 0]].to_numpy()
    return {"dense": dense, "verify": verify, "rr": rr}


def long_record_job(task: dict[str, Any]) -> dict[str, Any]:
    """
    Fingerprints of the whole raw recording of one long record, through the closed-data guard.

    Parameters
    ----------
    task : dict[str, Any]
        ``record_key``, ``raw_path``, ``source``, ``role`` and ``allow_test``.

    Returns
    -------
    dict[str, Any]
        ``record_key``, ``verify`` and ``keys`` of ``record_fingerprints`` (the dense vector is not used).
    """
    check_roles((task["role"],), task["allow_test"])
    path = guard_path(task["raw_path"], task["source"])
    job = {"format": "h5" if task["source"] == "sph" else "wfdb", "stem": str(path)}
    signal, rate, names = read_signal(job)
    if rate != FS:
        raise ValueError(f"not 500 Hz: {task['record_key']}")
    signal = canonical_order(signal, names).T
    signal = signal[:, : signal.shape[1] - signal.shape[1] % DECIMATION]
    prints = record_fingerprints(signal)
    return {"record_key": task["record_key"], "verify": prints["verify"], "keys": prints["keys"]}


def run_long_records(identity: pd.DataFrame, workers: int, allow_test: bool = False) -> list[dict[str, Any]]:
    """
    Fingerprint the whole raw recording of every record longer than 10 s.

    Parameters
    ----------
    identity : pd.DataFrame
        Output of ``mlc03_signal_runs.read_identity``.
    workers : int
        Processes.
    allow_test : bool
        Passed to the role checks.

    Returns
    -------
    list[dict[str, Any]]
        Outputs of ``long_record_job``.
    """
    long = identity[identity["n_samples"] > SAMPLES]
    tasks = [{**row, "allow_test": allow_test}
             for row in long[["record_key", "raw_path", "source", "role"]].to_dict("records")]
    guard = (closed_keys(), frozenset(identity["raw_path"]))
    with Pool(workers, initializer=init_worker, initargs=guard) as pool:
        return pool.map(long_record_job, tasks, chunksize=8)


def key_table(rr: np.ndarray, long_records: dict[int, list[tuple[tuple[int, int, int, int], int]]]
              ) -> dict[int, list[tuple[tuple[int, int, int, int], int]]]:
    """
    RR keys per record: from the cached window, replaced by the whole recording for long records.

    Parameters
    ----------
    rr : np.ndarray
        ``rr`` of ``load_fingerprints`` (record row, shift, three bins, first peak).
    long_records : dict[int, list[tuple[tuple[int, int, int, int], int]]]
        Whole-recording keys by record row.

    Returns
    -------
    dict[int, list[tuple[tuple[int, int, int, int], int]]]
        Keys by record row.
    """
    keys: dict[int, list[tuple[tuple[int, int, int, int], int]]] = defaultdict(list)
    for row, shift, first, second, third, peak in rr.tolist():
        if row not in long_records:
            keys[row].append(((shift, first, second, third), peak))
    keys.update(long_records)
    return dict(keys)


def verify_candidates(candidates: pd.DataFrame, prints: dict[int, np.ndarray],
                      radius: int = 2) -> pd.DataFrame:
    """
    Correlation of every shifted candidate pair at its RR lag.

    Parameters
    ----------
    candidates : pd.DataFrame
        Output of ``shifted_candidates``.
    prints : dict[int, np.ndarray]
        Verification fingerprint of every record row (whole recording for long records).
    radius : int
        Lags within this many 50 Hz samples of the RR lag are tried.

    Returns
    -------
    pd.DataFrame
        The candidates with ``fp_lag`` (50 Hz samples), ``correlation``, ``std_ratio`` and ``overlap_s``.
    """
    rows = []
    for first, second, lag in candidates[["first", "second", "lag"]].itertuples(index=False):
        best = best_lag(np.asarray(prints[first], dtype=np.float32),
                        np.asarray(prints[second], dtype=np.float32), int(round(lag / DECIMATION)), radius)
        rows.append({"fp_lag": best[0], "correlation": best[1], "std_ratio": best[2],
                     "overlap_s": best[3] / FINGERPRINT_RATE})
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(rows, index=None)], axis=1)
