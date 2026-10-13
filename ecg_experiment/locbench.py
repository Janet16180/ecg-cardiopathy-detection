"""
LocBench v1: localization benchmark sets, targets, metrics and patient bootstrap intervals.

Frozen in ``docs/experiment-061-locbench.md`` (with its Amendment 1). A method supplies a finite map of shape
(n, 12, 5000) for the PTB-XL ``ecg_id``s returned by ``scored_ecg_ids``; higher means more abnormal. Only
samples 100-4899 are scored. ``evaluate_method`` returns a summary with intervals and per-record tables;
``save_tables`` and ``load_tables`` keep the tables so that later experiments can compute paired differences
against the Experiment 061 baselines with ``paired_difference`` without recomputing their maps.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.stats import rankdata

from ecg_experiment import ROOT
from ecg_experiment.ecgad import load_labels, load_split
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.locbench_data import CLEAN_RHYTHMS, REGION_TABLE, load_partition
from ecg_experiment.waveforms import LEADS

SEED = 61061
DRAWS = 2000
SAMPLES = 5000
START, STOP = 100, 4900
FS = 500
LEAD_PERCENTILE = 95.0
RECORD_PERCENTILE = 99.0
TOP_FRACTION = 0.05
A1_BINS = 65536
A1_TOLERANCE = 1e-4
MIN_FAMILY = 20
FIDUCIAL_DIR = ROOT / "data/raw/ptb-xl-plus/1.0.1/fiducial_points/ecgdeli"
REGION_RHYTHMS = CLEAN_RHYTHMS | {"STACH"}
DETECTION_CLASSES = ("MI", "STTC", "CD", "HYP")
FAMILIES = {
    "IMI": "IMI",
    "ASMI": "anterior MI",
    "AMI": "anterior MI",
    "ALMI": "lateral MI",
    "LMI": "lateral MI",
    "CLBBB": "CLBBB",
    "CRBBB": "CRBBB",
    "LAFB": "LAFB",
    "1AVB": "1AVB",
    "WPW": "WPW",
    "LNGQT": "LNGQT",
    "INJAS": "ST injury",
    "INJAL": "ST injury",
    "INJIN": "ST injury",
    "ISCAS": "ischemia",
    "ISCAL": "ischemia",
    "ISCIN": "ischemia",
    "ISCLA": "ischemia",
}
# Component -> (start mark, end mark, longest interval in samples). "Same beat": no other start mark between.
PAIRS = {
    "P": ("p-wave onset", "p-wave offset", 150),
    "PR": ("p-wave onset", "QRS onset", 200),
    "QRS": ("QRS onset", "QRS offset", 125),
    "ST": ("QRS offset", "t-wave onset", 200),
    "T": ("t-wave onset", "t-wave offset", 300),
}
ECGAD_COLUMNS = ("A2", "A3", "A4", "A6", "marked_share")
REGION_COLUMNS = ("B1", "B2", "B3")


@dataclass
class Benchmark:
    """
    LocBench v1 sets and their targets.

    Attributes
    ----------
    ecgad : pd.DataFrame
        ECGAD development records: ``ecg_id``, ``patient_id``, ``bench_index``, ``marked``.
    ecgad_labels : np.ndarray
        Boolean cardiologist marks (n, 12, 5000) in ``ecgad`` row order.
    region : pd.DataFrame
        Region set: ``ecg_id``, ``patient_id``, ``statement``, ``family``, ``strict``, ``all_leads``.
    region_leads : np.ndarray
        Boolean expected leads (n, 12) in ``region`` row order.
    region_time : np.ndarray
        Boolean expected-component samples (n, 5000) in ``region`` row order.
    noise : pd.DataFrame
        Noise set: ``ecg_id``, ``patient_id``, ``noisy_leads`` (``;``-joined).
    noise_leads : np.ndarray
        Boolean named noisy leads (n, 12) in ``noise`` row order.
    clean_normals : pd.DataFrame
        ``evaluation`` clean normals: ``ecg_id``, ``patient_id``.
    detection : pd.DataFrame
        Detection set: ``ecg_id``, ``patient_id``, ``y`` (1 for MI, STTC, CD or HYP at likelihood >= 50).
    counts : dict[str, Any]
        Set sizes and exclusions.
    """

    ecgad: pd.DataFrame
    ecgad_labels: np.ndarray
    region: pd.DataFrame
    region_leads: np.ndarray
    region_time: np.ndarray
    noise: pd.DataFrame
    noise_leads: np.ndarray
    clean_normals: pd.DataFrame
    detection: pd.DataFrame
    counts: dict[str, Any]


def fiducial_path(ecg_id: int) -> Path:
    """
    ECGdeli global fiducial annotation stem of a PTB-XL record (no extension).

    Parameters
    ----------
    ecg_id : int
        PTB-XL record ID.

    Returns
    -------
    Path
        ``fiducial_points/ecgdeli/NNNNN/NNNNN_points_global``.
    """
    return FIDUCIAL_DIR / f"{ecg_id // 1000 * 1000:05d}" / f"{ecg_id:05d}_points_global"


def pair_marks(samples: np.ndarray, notes: list[str], start: str, end: str, window: int) -> np.ndarray:
    """
    Pair each start mark with the next end mark of the same beat.

    The end mark must come strictly after the start mark, at most ``window`` samples later, with no other
    start mark of the same kind in between. Unpaired marks are dropped.

    Parameters
    ----------
    samples : np.ndarray
        Annotation sample positions.
    notes : list[str]
        Annotation text of each mark.
    start, end : str
        Start and end mark names.
    window : int
        Longest interval in samples.

    Returns
    -------
    np.ndarray
        Integer array (k, 2) of half-open intervals [start, end).
    """
    samples = np.asarray(samples, dtype=np.int64)
    notes = np.asarray(notes, dtype=object)
    starts = np.sort(samples[notes == start])
    ends = np.sort(samples[notes == end])
    intervals = []
    for index, onset in enumerate(starts):
        following = np.searchsorted(ends, onset, side="right")
        if following == len(ends):
            continue
        offset = ends[following]
        next_start = starts[index + 1] if index + 1 < len(starts) else np.iinfo(np.int64).max
        if next_start < offset or offset - onset > window:
            continue
        intervals.append((onset, offset))
    return np.array(intervals, dtype=np.int64).reshape(-1, 2)


def component_intervals(samples: np.ndarray, notes: list[str]) -> dict[str, np.ndarray]:
    """
    P, PR, QRS, ST and T intervals from one record's ECGdeli marks.

    Parameters
    ----------
    samples : np.ndarray
        Annotation sample positions at 500 Hz.
    notes : list[str]
        Annotation text of each mark.

    Returns
    -------
    dict[str, np.ndarray]
        Half-open intervals (k, 2) per component.
    """
    return {
        name: pair_marks(samples, notes, start, end, window) for name, (start, end, window) in PAIRS.items()
    }


def read_components(ecg_id: int) -> dict[str, np.ndarray]:
    """
    Read a record's ECGdeli global fiducials and pair them into components.

    Parameters
    ----------
    ecg_id : int
        PTB-XL record ID.

    Returns
    -------
    dict[str, np.ndarray]
        Output of ``component_intervals``.
    """
    annotation = wfdb.rdann(str(fiducial_path(ecg_id)), "atr")
    if annotation.fs not in (None, FS):
        raise ValueError(f"Unexpected fiducial sampling rate for {ecg_id}: {annotation.fs}")
    return component_intervals(annotation.sample, list(annotation.aux_note))


def interval_mask(intervals: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Boolean sample mask of half-open intervals.

    Parameters
    ----------
    intervals : np.ndarray
        Integer array (k, 2).
    samples : int
        Record length.

    Returns
    -------
    np.ndarray
        Boolean mask (samples,).
    """
    mask = np.zeros(samples, dtype=bool)
    for onset, offset in intervals:
        mask[max(int(onset), 0) : min(int(offset), samples)] = True
    return mask


def _statement_sets() -> tuple[pd.DataFrame, set[str], set[str], set[str], pd.Series]:
    """Metadata and the rhythm, diagnostic and form statement sets."""
    meta = load_metadata()
    statements = load_statements()
    rhythm = set(statements.index[statements["rhythm"] == 1])
    diagnostic = set(statements.index[statements["diagnostic"] == 1])
    form = set(statements.index[statements["form"] == 1])
    return meta, rhythm, diagnostic, form, statements["diagnostic_class"]


def _ecgad_development() -> tuple[pd.DataFrame, np.ndarray]:
    """ECGAD development records and their labels; holdout labels are never indexed."""
    split = load_split()
    table = split[split["partition"] == "development"].sort_values("bench_index").reset_index(drop=True)
    labels_file = load_labels()
    labels = np.stack([np.asarray(labels_file[int(index)]).T > 0.5 for index in table["bench_index"]])
    table = table[["ecg_id", "patient_id", "bench_index"]].copy()
    table["marked"] = labels[:, :, START:STOP].any(axis=(1, 2))
    return table, labels


def _region_set(
    evaluation: pd.DataFrame, meta: pd.DataFrame, rhythm: set[str], diagnostic: set[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Region-set records, expected leads and component samples, and exclusion counts."""
    rows, lead_masks, time_masks = [], [], []
    exclusions = {"no_ecgdeli_file": [], "no_component_interval": []}
    candidates = 0
    for record in evaluation.itertuples(index=False):
        codes = [code for code in record.region_statements.split(";") if code]
        if len(codes) != 1:
            continue
        scp = meta.at[record.ecg_id, "scp_codes"]
        if any(code in rhythm and code not in REGION_RHYTHMS for code in scp):
            continue
        candidates += 1
        code = codes[0]
        if not Path(str(fiducial_path(record.ecg_id)) + ".atr").exists():
            exclusions["no_ecgdeli_file"].append(int(record.ecg_id))
            continue
        components = read_components(record.ecg_id)
        leads, names = REGION_TABLE[code]
        masks = [interval_mask(components[name]) for name in names]
        if not all(mask[START:STOP].any() for mask in masks):
            exclusions["no_component_interval"].append(int(record.ecg_id))
            continue
        rows.append(
            {
                "ecg_id": int(record.ecg_id),
                "patient_id": int(record.patient_id),
                "statement": code,
                "family": FAMILIES[code],
                "strict": not any(other in diagnostic and other != code for other in scp),
                "all_leads": len(leads) == len(LEADS),
            }
        )
        lead_masks.append(np.isin(LEADS, leads))
        time_masks.append(np.logical_or.reduce(masks))
    table = pd.DataFrame(rows)
    counts = {
        "candidates": candidates,
        "excluded_no_ecgdeli_file": exclusions["no_ecgdeli_file"],
        "excluded_no_component_interval": exclusions["no_component_interval"],
        "records": len(table),
        "families": table["family"].value_counts().sort_index().to_dict(),
        "strict_families": table[table["strict"]]["family"].value_counts().sort_index().to_dict(),
    }
    return table, np.array(lead_masks), np.array(time_masks), counts


def _noise_set(
    evaluation: pd.DataFrame, meta: pd.DataFrame, rhythm: set[str], diagnostic: set[str], form: set[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Noise-set records and their named noisy leads."""
    rows, masks = [], []
    for record in evaluation.itertuples(index=False):
        scp = meta.at[record.ecg_id, "scp_codes"]
        if scp.get("NORM", 0) != 100:
            continue
        if any(code != "NORM" and (code in diagnostic or code in form) for code in scp):
            continue
        if any(code in rhythm and code not in CLEAN_RHYTHMS for code in scp):
            continue
        if meta.loc[record.ecg_id, ["extra_beats", "pacemaker"]].notna().any():
            continue
        leads = [lead for lead in record.noisy_leads.split(";") if lead]
        if not 1 <= len(leads) <= len(LEADS) - 1:
            continue
        rows.append(
            {
                "ecg_id": int(record.ecg_id),
                "patient_id": int(record.patient_id),
                "noisy_leads": record.noisy_leads,
            }
        )
        masks.append(np.isin(LEADS, leads))
    return pd.DataFrame(rows), np.array(masks)


def build_benchmark() -> Benchmark:
    """
    Build every LocBench v1 set from the frozen partition, the ECGAD split and the ECGdeli fiducials.

    Returns
    -------
    Benchmark
        Sets, targets and counts.
    """
    meta, rhythm, diagnostic, form, classes = _statement_sets()
    partition = load_partition()
    evaluation = partition[partition["partition"] == "evaluation"]
    ecgad, labels = _ecgad_development()
    region, region_leads, region_time, region_counts = _region_set(evaluation, meta, rhythm, diagnostic)
    noise, noise_leads = _noise_set(evaluation, meta, rhythm, diagnostic, form)
    clean = evaluation[evaluation["clean_normal"]][["ecg_id", "patient_id"]].reset_index(drop=True)
    positive = np.array(
        [
            any(
                likelihood >= 50 and classes.get(code) in DETECTION_CLASSES
                for code, likelihood in meta.at[ecg_id, "scp_codes"].items()
            )
            for ecg_id in evaluation["ecg_id"]
        ]
    )
    keep = positive | evaluation["clean_normal"].to_numpy()
    detection = evaluation[keep][["ecg_id", "patient_id"]].reset_index(drop=True)
    detection["y"] = positive[keep].astype(int)
    counts = {
        "ecgad_development": {
            "records": len(ecgad),
            "patients": int(ecgad["patient_id"].nunique()),
            "marked": int(ecgad["marked"].sum()),
        },
        "region": region_counts,
        "noise": {"records": len(noise), "patients": int(noise["patient_id"].nunique())},
        "clean_normals": {"records": len(clean), "patients": int(clean["patient_id"].nunique())},
        "detection": {
            "records": len(detection),
            "positives": int(detection["y"].sum()),
            "patients": int(detection["patient_id"].nunique()),
        },
    }
    return Benchmark(
        ecgad, labels, region, region_leads, region_time, noise, noise_leads, clean, detection, counts
    )


@cache
def load_benchmark() -> Benchmark:
    """
    Build the benchmark once per process.

    Returns
    -------
    Benchmark
        Output of ``build_benchmark``.
    """
    return build_benchmark()


def scored_ecg_ids(bench: Benchmark | None = None) -> np.ndarray:
    """
    Every ``ecg_id`` that a method must map, sorted.

    Parameters
    ----------
    bench : Benchmark or None
        Benchmark; built if None.

    Returns
    -------
    np.ndarray
        Union of the ECGAD development, region, noise, clean-normal and detection records.
    """
    bench = load_benchmark() if bench is None else bench
    frames = (bench.ecgad, bench.region, bench.noise, bench.clean_normals, bench.detection)
    return np.unique(np.concatenate([frame["ecg_id"].to_numpy(np.int64) for frame in frames]))


def patient_ids(bench: Benchmark) -> pd.Series:
    """
    Patient ID of every scored record.

    Parameters
    ----------
    bench : Benchmark
        Benchmark.

    Returns
    -------
    pd.Series
        Indexed by ``ecg_id``.
    """
    frames = (bench.ecgad, bench.region, bench.noise, bench.clean_normals, bench.detection)
    table = pd.concat([frame[["ecg_id", "patient_id"]] for frame in frames]).drop_duplicates("ecg_id")
    return table.set_index("ecg_id")["patient_id"].sort_index()


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """
    Mann-Whitney AUROC with ties counted as half.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    scores : np.ndarray
        Scores, same shape.

    Returns
    -------
    float
        AUROC.

    Raises
    ------
    ValueError
        If only one class is present.
    """
    y = np.asarray(y, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC needs both classes")
    ranks = rankdata(scores)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def lead_scores(record_map: np.ndarray) -> np.ndarray:
    """
    95th percentile of each lead over samples 100-4899.

    Parameters
    ----------
    record_map : np.ndarray
        Map (12, 5000).

    Returns
    -------
    np.ndarray
        Lead scores (12,).
    """
    return np.percentile(record_map[:, START:STOP], LEAD_PERCENTILE, axis=1)


def time_scores(record_map: np.ndarray) -> np.ndarray:
    """
    Maximum over leads of each sample in 100-4899.

    Parameters
    ----------
    record_map : np.ndarray
        Map (12, 5000).

    Returns
    -------
    np.ndarray
        Time scores (4800,).
    """
    return record_map[:, START:STOP].max(axis=0)


def record_score(record_map: np.ndarray) -> float:
    """
    99th percentile over all leads and samples in 100-4899.

    Parameters
    ----------
    record_map : np.ndarray
        Map (12, 5000).

    Returns
    -------
    float
        Record score.
    """
    return float(np.percentile(record_map[:, START:STOP], RECORD_PERCENTILE))


def precision_at_top(scores: np.ndarray, positive: np.ndarray, fraction: float = TOP_FRACTION) -> float:
    """
    Share of the top ``fraction`` of points that are positive, with ties at the cut counted by expectation.

    Parameters
    ----------
    scores : np.ndarray
        Point scores.
    positive : np.ndarray
        Boolean point targets, same shape.
    fraction : float
        Share of points kept.

    Returns
    -------
    float
        Expected precision under random tie breaking.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positive = np.asarray(positive, dtype=bool).ravel()
    k = int(round(fraction * len(scores)))
    cut = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > cut
    tied = scores == cut
    hits = positive[above].sum() + (k - above.sum()) * positive[tied].sum() / tied.sum()
    return float(hits / k)


def ecgad_record_metrics(record_map: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    A2, A3, A4, A6 and the marked share of one ECGAD record; NaN where a metric does not apply.

    Parameters
    ----------
    record_map : np.ndarray
        Map (12, 5000).
    labels : np.ndarray
        Boolean marks (12, 5000).

    Returns
    -------
    dict[str, float]
        Within-record metrics.
    """
    values = record_map[:, START:STOP]
    marks = labels[:, START:STOP]
    result = dict.fromkeys(ECGAD_COLUMNS, np.nan)
    if not marks.any():
        return result
    result["marked_share"] = float(marks.mean())
    result["A6"] = precision_at_top(values, marks)
    if not marks.all():
        result["A2"] = auroc(marks, values)
    marked_leads = marks.any(axis=1)
    if 1 <= marked_leads.sum() <= len(LEADS) - 1:
        result["A3"] = auroc(marked_leads, lead_scores(record_map))
    marked_times = marks.any(axis=0)
    if not marked_times.all():
        result["A4"] = auroc(marked_times, time_scores(record_map))
    return result


def region_record_metrics(record_map: np.ndarray, leads: np.ndarray, times: np.ndarray) -> dict[str, float]:
    """
    B1, B2 and B3 of one region-set record; B2 is NaN when every lead is expected.

    Parameters
    ----------
    record_map : np.ndarray
        Map (12, 5000).
    leads : np.ndarray
        Boolean expected leads (12,).
    times : np.ndarray
        Boolean expected-component samples (5000,).

    Returns
    -------
    dict[str, float]
        Within-record metrics.
    """
    window = times[START:STOP]
    region = leads[:, None] & window[None, :]
    return {
        "B1": auroc(region, record_map[:, START:STOP]),
        "B2": np.nan if leads.all() else auroc(leads, lead_scores(record_map)),
        "B3": auroc(window, time_scores(record_map)),
    }


def a1_histograms(
    scores: np.ndarray, labels: np.ndarray, bins: int = A1_BINS
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Per-record positive and negative counts on a global rank grid, and the exact pooled AUROC.

    All points are ranked once (average ranks). Ranks are cut into ``bins`` equal-count bins, so tied values
    share a bin. A bootstrap draw sums the weighted record histograms and counts pairs within a bin as ties.

    Parameters
    ----------
    scores : np.ndarray
        Point scores (records, points).
    labels : np.ndarray
        Boolean point targets (records, points).
    bins : int
        Number of rank bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Positive and negative counts (records, bins) as uint16, and the exact Mann-Whitney AUROC.
    """
    records, points = scores.shape
    flat = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    ranks = rankdata(flat)
    positives = int(y.sum())
    negatives = len(y) - positives
    exact = float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    codes = np.minimum(((ranks - 1) * bins / len(flat)).astype(np.int64), bins - 1)
    codes += np.repeat(np.arange(records, dtype=np.int64) * bins, points)
    positive = np.bincount(codes[y], minlength=records * bins).reshape(records, bins)
    negative = np.bincount(codes[~y], minlength=records * bins).reshape(records, bins)
    if max(positive.max(), negative.max()) > np.iinfo(np.uint16).max:
        raise ValueError("Histogram count exceeds uint16")
    return positive.astype(np.uint16), negative.astype(np.uint16), exact


def binned_auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """
    AUROC from positive and negative counts on an ordered grid, pairs in the same bin counted as ties.

    Parameters
    ----------
    positive, negative : np.ndarray
        Counts per bin, ascending score order.

    Returns
    -------
    float
        AUROC, or NaN if a class is empty.
    """
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive == 0 or total_negative == 0:
        return np.nan
    below = np.cumsum(negative) - negative
    return float((positive * (below + 0.5 * negative)).sum() / (total_positive * total_negative))


def _record_map(maps: np.ndarray, position: int) -> np.ndarray:
    """One record's map in float32, checked for shape and finiteness."""
    record_map = np.asarray(maps[position], dtype=np.float32)
    if record_map.shape != (len(LEADS), SAMPLES) or not np.isfinite(record_map).all():
        raise ValueError(f"Map at position {position} is not a finite (12, 5000) array")
    return record_map


def evaluate_method(
    maps: np.ndarray, ecg_ids: np.ndarray, name: str, bench: Benchmark | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Score a method's maps on every LocBench v1 set.

    Parameters
    ----------
    maps : np.ndarray
        Maps (n, 12, 5000), any float dtype, possibly memory-mapped; higher means more abnormal.
    ecg_ids : np.ndarray
        ``ecg_id`` of each map row; must include every ``scored_ecg_ids`` record.
    name : str
        Method name, stored in the tables.
    bench : Benchmark or None
        Benchmark; built if None.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Summary (every metric with its interval) and per-record tables (see ``record_tables``).
    """
    tables = record_tables(maps, ecg_ids, name, bench)
    return summarize(tables), tables


def record_tables(
    maps: np.ndarray, ecg_ids: np.ndarray, name: str, bench: Benchmark | None = None
) -> dict[str, Any]:
    """
    Per-record metrics and scores of one method.

    Parameters
    ----------
    maps : np.ndarray
        Maps (n, 12, 5000).
    ecg_ids : np.ndarray
        ``ecg_id`` of each map row.
    name : str
        Method name.
    bench : Benchmark or None
        Benchmark; built if None.

    Returns
    -------
    dict[str, Any]
        ``name``; ``records`` (``ecg_id``, ``patient_id``, ``record_score`` and one lead score per lead, for
        every scored record); ``ecgad``, ``region``, ``noise_control`` (noise set ``y`` = 1 and clean normals
        ``y`` = 0, with ``C1`` on noise rows) and ``detection`` tables; ``a1_positive`` and ``a1_negative``
        histograms; ``a1_exact``; ``ecgad_maps`` (float32, ECGAD development order).
    """
    bench = load_benchmark() if bench is None else bench
    ecg_ids = np.asarray(ecg_ids, dtype=np.int64)
    if len(ecg_ids) != len(maps) or len(np.unique(ecg_ids)) != len(ecg_ids):
        raise ValueError("ecg_ids must be unique and aligned with maps")
    position = pd.Series(np.arange(len(ecg_ids)), index=ecg_ids)
    scored = scored_ecg_ids(bench)
    missing = np.setdiff1d(scored, ecg_ids)
    if len(missing):
        raise ValueError(f"{len(missing)} scored records have no map, first {missing[:5].tolist()}")
    patients = patient_ids(bench)
    leads = np.empty((len(scored), len(LEADS)))
    records = np.empty(len(scored))
    for row, ecg_id in enumerate(scored):
        record_map = _record_map(maps, int(position[ecg_id]))
        leads[row] = lead_scores(record_map)
        records[row] = record_score(record_map)
    record_table = pd.DataFrame(
        {"ecg_id": scored, "patient_id": patients.loc[scored].to_numpy(), "record_score": records}
    )
    for index, lead in enumerate(LEADS):
        record_table[f"lead_{lead}"] = leads[:, index]
    score_of = record_table.set_index("ecg_id")["record_score"]

    ecgad = bench.ecgad[["ecg_id", "patient_id", "marked"]].copy()
    ecgad_maps = np.stack([_record_map(maps, int(position[ecg_id])) for ecg_id in ecgad["ecg_id"]])
    rows = [
        ecgad_record_metrics(record_map, labels)
        for record_map, labels in zip(ecgad_maps, bench.ecgad_labels, strict=True)
    ]
    for column in ECGAD_COLUMNS:
        ecgad[column] = [row[column] for row in rows]
    ecgad["record_score"] = score_of.loc[ecgad["ecg_id"]].to_numpy()
    positive, negative, exact = a1_histograms(
        ecgad_maps[:, :, START:STOP].reshape(len(ecgad), -1),
        bench.ecgad_labels[:, :, START:STOP].reshape(len(ecgad), -1),
    )

    region = bench.region.copy()
    rows = [
        region_record_metrics(_record_map(maps, int(position[ecg_id])), lead_mask, time_mask)
        for ecg_id, lead_mask, time_mask in zip(
            region["ecg_id"], bench.region_leads, bench.region_time, strict=True
        )
    ]
    for column in REGION_COLUMNS:
        region[column] = [row[column] for row in rows]

    lead_columns = [f"lead_{lead}" for lead in LEADS]
    noise_scores = record_table.set_index("ecg_id").loc[bench.noise["ecg_id"], lead_columns].to_numpy()
    noise = bench.noise[["ecg_id", "patient_id"]].copy()
    noise["y"] = 1
    noise["C1"] = [auroc(mask, values) for mask, values in zip(bench.noise_leads, noise_scores, strict=True)]
    clean = bench.clean_normals[["ecg_id", "patient_id"]].copy()
    clean["y"] = 0
    clean["C1"] = np.nan
    control = pd.concat([noise, clean], ignore_index=True)
    control["record_score"] = score_of.loc[control["ecg_id"]].to_numpy()

    detection = bench.detection.copy()
    detection["record_score"] = score_of.loc[detection["ecg_id"]].to_numpy()
    return {
        "name": name,
        "records": record_table,
        "ecgad": ecgad,
        "region": region,
        "noise_control": control,
        "detection": detection,
        "a1_positive": positive,
        "a1_negative": negative,
        "a1_exact": exact,
        "ecgad_maps": ecgad_maps,
    }


def eligible_families(region: pd.DataFrame, strict: bool, metric: str) -> list[str]:
    """
    Families with at least 20 records (strict records for the strict variant) that the metric applies to.

    Parameters
    ----------
    region : pd.DataFrame
        Region table with ``family``, ``strict`` and ``all_leads``.
    strict : bool
        Strict variant.
    metric : str
        ``B1``, ``B2`` or ``B3``; B2 skips families whose every record expects all 12 leads.

    Returns
    -------
    list[str]
        Sorted family names.
    """
    rows = region[region["strict"]] if strict else region
    counts = rows["family"].value_counts()
    families = sorted(counts.index[counts >= MIN_FAMILY])
    if metric == "B2":
        families = [family for family in families if not rows[rows["family"] == family]["all_leads"].all()]
    return families


def metric_names(tables: dict[str, Any]) -> list[str]:
    """
    Every metric reported for a method.

    Parameters
    ----------
    tables : dict[str, Any]
        Output of ``record_tables`` or ``load_tables``.

    Returns
    -------
    list[str]
        ``A1``-``A6``, ``marked_share``, ``B1/<family>``, ``B1/macro`` (and B2, B3), their ``/strict``
        variants, ``C1``, ``C2`` and ``D1``.
    """
    names = ["A1", "A2", "A3", "A4", "A5", "A6", "marked_share"]
    for strict in (False, True):
        suffix = "/strict" if strict else ""
        for metric in REGION_COLUMNS:
            families = eligible_families(tables["region"], strict, metric)
            names += [f"{metric}/{family}{suffix}" for family in families] + [f"{metric}/macro{suffix}"]
    return names + ["C1", "C2", "D1"]


def _record_auroc(y: np.ndarray, scores: np.ndarray) -> Callable[[np.ndarray], float]:
    """Statistic: record AUROC of the drawn rows, NaN for a single-class draw."""

    def statistic(rows: np.ndarray) -> float:
        drawn = y[rows]
        if drawn.all() or not drawn.any():
            return np.nan
        return auroc(drawn, scores[rows])

    return statistic


def _mean(values: np.ndarray) -> Callable[[np.ndarray], float]:
    """Statistic: mean of the drawn rows."""
    return lambda rows: float(values[rows].mean())


def _macro(values: np.ndarray, families: np.ndarray, names: list[str]) -> Callable[[np.ndarray], float]:
    """Statistic: unweighted mean over families of the family means of the drawn rows."""
    codes = np.array([names.index(family) for family in families])

    def statistic(rows: np.ndarray) -> float:
        counts = np.bincount(codes[rows], minlength=len(names))
        if (counts == 0).any():
            return np.nan
        return float((np.bincount(codes[rows], weights=values[rows], minlength=len(names)) / counts).mean())

    return statistic


def _a1(positive: np.ndarray, negative: np.ndarray) -> Callable[[np.ndarray], float]:
    """Statistic: binned pooled AUROC of the drawn records."""
    positive = positive.astype(np.float64)
    negative = negative.astype(np.float64)

    def statistic(rows: np.ndarray) -> float:
        weights = np.bincount(rows, minlength=len(positive)).astype(np.float64)
        return binned_auroc(weights @ positive, weights @ negative)

    return statistic


def metric_statistic(tables: dict[str, Any], metric: str) -> tuple[np.ndarray, np.ndarray, Callable, float]:
    """
    Build the population, statistic and point estimate of one metric.

    Parameters
    ----------
    tables : dict[str, Any]
        Output of ``record_tables`` or ``load_tables``.
    metric : str
        A name from ``metric_names``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Callable, float]
        ``ecg_id`` and ``patient_id`` of the population rows, a statistic of drawn row positions (NaN for an
        invalid draw), and the point estimate.
    """
    ecgad = tables["ecgad"]
    if metric == "A1":
        statistic = _a1(tables["a1_positive"], tables["a1_negative"])
        point = float(tables["a1_exact"])
        return ecgad["ecg_id"].to_numpy(), ecgad["patient_id"].to_numpy(), statistic, point
    if metric == "A5":
        frame = ecgad
        statistic = _record_auroc(frame["marked"].to_numpy(bool), frame["record_score"].to_numpy())
    elif metric in ECGAD_COLUMNS:
        frame = ecgad[ecgad[metric].notna()]
        statistic = _mean(frame[metric].to_numpy())
    elif metric == "C1":
        control = tables["noise_control"]
        frame = control[control["y"] == 1]
        statistic = _mean(frame["C1"].to_numpy())
    elif metric == "C2":
        frame = tables["noise_control"]
        statistic = _record_auroc(frame["y"].to_numpy(bool), frame["record_score"].to_numpy())
    elif metric == "D1":
        frame = tables["detection"]
        statistic = _record_auroc(frame["y"].to_numpy(bool), frame["record_score"].to_numpy())
    else:
        column, family, *rest = metric.split("/")
        strict = rest == ["strict"]
        region = tables["region"]
        region = region[region["strict"]] if strict else region
        if family == "macro":
            names = eligible_families(tables["region"], strict, column)
            frame = region[region["family"].isin(names)]
            statistic = _macro(frame[column].to_numpy(), frame["family"].to_numpy(), names)
        else:
            frame = region[region["family"] == family]
            statistic = _mean(frame[column].to_numpy())
    ids, patients = frame["ecg_id"].to_numpy(), frame["patient_id"].to_numpy()
    return ids, patients, statistic, statistic(np.arange(len(frame)))


def _bootstrap(
    patients: np.ndarray, statistics: list[Callable], draws: int, seed: int
) -> tuple[np.ndarray, int]:
    """Values of each statistic (draws, k) on the same whole-patient draws, and the count of skipped draws."""
    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    values, skipped = [], 0
    for _ in range(draws):
        rows = patient_resample(groups, rng)
        drawn = [statistic(rows) for statistic in statistics]
        if any(np.isnan(value) for value in drawn):
            skipped += 1
            continue
        values.append(drawn)
    if not values:
        raise ValueError("Every bootstrap draw was skipped")
    return np.array(values), skipped


def metric_interval(
    tables: dict[str, Any], metric: str, draws: int = DRAWS, seed: int = SEED
) -> dict[str, Any]:
    """
    Point estimate and 95% whole-patient bootstrap interval of one metric.

    Parameters
    ----------
    tables : dict[str, Any]
        Output of ``record_tables`` or ``load_tables``.
    metric : str
        A name from ``metric_names``.
    draws : int
        Bootstrap draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci_low``, ``ci_high``, ``records``, ``patients``, ``skipped_draws``.
    """
    ids, patients, statistic, point = metric_statistic(tables, metric)
    values, skipped = _bootstrap(patients, [statistic], draws, seed)
    low, high = np.percentile(values[:, 0], [2.5, 97.5])
    return {
        "value": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "records": len(ids),
        "patients": int(len(np.unique(patients))),
        "skipped_draws": skipped,
    }


def summarize(tables: dict[str, Any], draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    """
    Every metric of one method with its interval, and the A1 binning check.

    Parameters
    ----------
    tables : dict[str, Any]
        Output of ``record_tables`` or ``load_tables``.
    draws : int
        Bootstrap draws.
    seed : int
        Seed.

    Returns
    -------
    dict[str, Any]
        ``name``, ``metrics`` keyed by ``metric_names`` and ``a1_check`` (exact, binned, absolute difference).
    """
    binned = binned_auroc(
        tables["a1_positive"].sum(axis=0).astype(np.float64),
        tables["a1_negative"].sum(axis=0).astype(np.float64),
    )
    difference = abs(binned - float(tables["a1_exact"]))
    if difference > A1_TOLERANCE:
        raise ValueError(f"Binned A1 differs from exact by {difference}")
    return {
        "name": tables["name"],
        "metrics": {metric: metric_interval(tables, metric, draws, seed) for metric in metric_names(tables)},
        "a1_check": {"exact": float(tables["a1_exact"]), "binned": binned, "absolute_difference": difference},
    }


def paired_difference(
    tables_a: dict[str, Any], tables_b: dict[str, Any], metric: str, draws: int = DRAWS, seed: int = SEED
) -> dict[str, Any]:
    """
    Metric of method A minus method B, with a paired interval on the same whole-patient draws.

    Parameters
    ----------
    tables_a, tables_b : dict[str, Any]
        Per-record tables of the two methods.
    metric : str
        A name from ``metric_names``.
    draws : int
        Bootstrap draws.
    seed : int
        Seed.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci_low``, ``ci_high``, ``a``, ``b``, ``records``, ``patients``, ``skipped_draws``.

    Raises
    ------
    ValueError
        If the two methods' populations differ.
    """
    ids_a, patients, statistic_a, point_a = metric_statistic(tables_a, metric)
    ids_b, _, statistic_b, point_b = metric_statistic(tables_b, metric)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError(f"Populations of {metric} differ between methods")
    values, skipped = _bootstrap(patients, [statistic_a, statistic_b], draws, seed)
    low, high = np.percentile(values[:, 0] - values[:, 1], [2.5, 97.5])
    return {
        "value": point_a - point_b,
        "ci_low": float(low),
        "ci_high": float(high),
        "a": point_a,
        "b": point_b,
        "records": len(ids_a),
        "patients": int(len(np.unique(patients))),
        "skipped_draws": skipped,
    }


TABLE_FRAMES = ("records", "ecgad", "region", "noise_control", "detection")


def save_tables(tables: dict[str, Any], folder: Path) -> None:
    """
    Write a method's per-record tables: CSVs, A1 histograms (npz) and ECGAD maps (float32 npy).

    Parameters
    ----------
    tables : dict[str, Any]
        Output of ``record_tables``.
    folder : Path
        Output folder, created if missing.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for frame in TABLE_FRAMES:
        tables[frame].to_csv(folder / f"{frame}.csv", index=False, float_format="%.17g")
    np.savez_compressed(
        folder / "a1_histograms.npz", positive=tables["a1_positive"], negative=tables["a1_negative"]
    )
    np.save(folder / "ecgad_maps.npy", tables["ecgad_maps"].astype(np.float32))
    meta = {
        "name": tables["name"],
        "a1_exact": float(tables["a1_exact"]),
        "ecgad_ecg_ids": tables["ecgad"]["ecg_id"].astype(int).tolist(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def load_tables(folder: Path, maps: bool = False) -> dict[str, Any]:
    """
    Read tables written by ``save_tables``.

    Parameters
    ----------
    folder : Path
        Folder of one method.
    maps : bool
        Also memory-map the ECGAD development maps.

    Returns
    -------
    dict[str, Any]
        Same keys as ``record_tables`` (``ecgad_maps`` only when ``maps``).
    """
    meta = json.loads((folder / "meta.json").read_text())
    tables: dict[str, Any] = {"name": meta["name"], "a1_exact": meta["a1_exact"]}
    for frame in TABLE_FRAMES:
        tables[frame] = pd.read_csv(folder / f"{frame}.csv")
    with np.load(folder / "a1_histograms.npz") as histograms:
        tables["a1_positive"] = histograms["positive"]
        tables["a1_negative"] = histograms["negative"]
    if maps:
        tables["ecgad_maps"] = np.load(folder / "ecgad_maps.npy", mmap_mode="r")
    return tables


def save_maps(path: Path, maps: np.ndarray, ecg_ids: np.ndarray, dtype: type = np.float16) -> None:
    """
    Write maps as an .npy file with an ``ecg_id`` order file beside it (``<path>.ids.json``).

    Parameters
    ----------
    path : Path
        Target .npy path.
    maps : np.ndarray
        Maps (n, 12, 5000).
    ecg_ids : np.ndarray
        ``ecg_id`` of each row.
    dtype : type
        Stored dtype.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(path.with_suffix(".npy.part"), mode="w+", dtype=dtype, shape=maps.shape)
    for start in range(0, len(maps), 256):
        out[start : start + 256] = maps[start : start + 256]
    out.flush()
    del out
    os.replace(path.with_suffix(".npy.part"), path)
    Path(str(path) + ".ids.json").write_text(json.dumps([int(ecg_id) for ecg_id in ecg_ids]))
