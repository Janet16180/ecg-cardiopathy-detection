"""
Replacement faithfulness test of localization maps for Experiment 070.

A map's top lead-samples are replaced by the record's normal counterpart (the 062 population mean beat for
its heart rate, age and sex, warped to the record's own beats), and a detector's logit drop is compared with
the drop after replacing matched random regions: same lead-samples per lead and per wave label, moved to
random leads and random positions inside the same wave label. See ``docs/experiment-070-replacement.md``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional

from ecg_experiment import combination066 as cb
from ecg_experiment import mil065 as mil
from ecg_experiment import normative_map062 as nm
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.locbench import DETECTION_CLASSES, START, STOP, lead_scores
from ecg_experiment.waveforms import LEADS

SAMPLES = 5000
SCORED = STOP - START
SEED = 70070
NORMAL_SEED = 70071
FRACTIONS = (0.05, 0.10, 0.20)
CONTROL_DRAWS = 8
LEAD_DRAWS = 20
TOP_LEADS = 3
TAPER = 11
RAMP = 30
LABELS = ("P", "PQ", "QRS", "ST", "T", "other")
LABEL_PRIORITY = (0, 1, 4, 3, 2)
CLIP_MV = 10.0
CROP = 4500
DETECTOR_WIDTHS = (32, 64, 128, 128)
DETECTOR_STRIDES = (1, 2, 2, 2)
MAPS = (
    "random",
    "amplitude",
    "qrs_prior",
    "U_B",
    "residual056",
    "normative_v1",
    "qrs_lexicographic",
    "raw",
    "dino_dist",
    "mil_lse",
    "factor",
    "textbook",
    "units_learned",
    "units_v2",
)


def one_per_patient(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """
    Draw ``count`` patients and one record of each.

    Parameters
    ----------
    frame : pd.DataFrame
        Eligible records with ``ecg_id`` and ``patient_id``.
    count : int
        Patients to keep (all if fewer).
    seed : int
        Seed of ``numpy.random.default_rng``; the sorted unique patients are permuted, then one record of each
        kept patient is drawn with the same generator.

    Returns
    -------
    pd.DataFrame
        ``ecg_id`` and ``patient_id`` of the chosen records, in draw order.
    """
    rng = np.random.default_rng(seed)
    patients = np.unique(frame["patient_id"].to_numpy())
    chosen = rng.permutation(patients)[:count]
    rows = []
    for patient in chosen:
        ids = np.sort(frame.loc[frame["patient_id"] == patient, "ecg_id"].to_numpy())
        rows.append({"ecg_id": int(ids[rng.integers(0, len(ids))]), "patient_id": int(patient)})
    return pd.DataFrame(rows)


def detection_positive(ecg_ids: np.ndarray) -> np.ndarray:
    """
    LocBench detection rule: a diagnostic statement of superclass MI, STTC, CD or HYP at likelihood >= 50.

    Parameters
    ----------
    ecg_ids : np.ndarray
        PTB-XL records.

    Returns
    -------
    np.ndarray
        Boolean per record.
    """
    meta = load_metadata()
    classes = load_statements()["diagnostic_class"]
    return np.array(
        [
            any(
                likelihood >= 50 and classes.get(code) in DETECTION_CLASSES
                for code, likelihood in meta.at[int(ecg_id), "scp_codes"].items()
            )
            for ecg_id in ecg_ids
        ]
    )


def composition_labels(raw: np.ndarray) -> np.ndarray:
    """
    One wave label per sample from the frozen 066 delineation.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV.

    Returns
    -------
    np.ndarray
        (5000,) int8 index into ``LABELS``; where components overlap QRS > ST > T > PQ > P.
    """
    masks = cb.component_masks(cb.delineated_components(raw), raw.shape[1])
    labels = np.full(raw.shape[1], len(LABELS) - 1, np.int8)
    for component in LABEL_PRIORITY:
        labels[masks[component]] = component
    return labels


def template_beat(normative: nm.Normative, heart_rate: float, age: float, sex: float) -> np.ndarray:
    """
    Return the population mean beat of fit clean normals for one heart rate, age and sex.

    Parameters
    ----------
    normative : nm.Normative
        Frozen 062 normative model.
    heart_rate, age, sex : float
        Covariates (age NaN is treated as 90, as in 062).

    Returns
    -------
    np.ndarray
        (12, BINS) mean beat in mV on the 062 grid, PR-segment baseline at zero.
    """
    row = nm.design(np.array([heart_rate]), np.array([age]), np.array([sex]))[0]
    return (row @ normative.models["rank0"].coefficients).reshape(len(LEADS), nm.BINS)


def template_timeline(template: np.ndarray, knots: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Warp the template onto every beat's own boundaries and blend overlapping beat windows.

    Parameters
    ----------
    template : np.ndarray
        (12, BINS) beat on the 062 grid.
    knots : np.ndarray
        (beats, 6) aligned warp knots of the record.
    samples : int
        Record length.

    Returns
    -------
    np.ndarray
        (12, samples) float64; each beat window has a trapezoid weight with ``RAMP``-sample ramps, overlapping
        windows are averaged, and a summed weight below 1 fades the template to zero.
    """
    total = np.zeros((template.shape[0], samples))
    weight = np.zeros(samples)
    grid = np.arange(nm.BINS, dtype=float)
    for row in knots:
        start, end = max(math.ceil(row[0]), 0), min(math.floor(row[-1]), samples - 1)
        if end < start:
            continue
        times = np.arange(start, end + 1)
        positions = np.interp(times, row, nm.GRID_KNOTS)
        ramp = np.clip(np.minimum(times - row[0], row[-1] - times) / RAMP, 0.0, 1.0)
        for lead in range(template.shape[0]):
            total[lead, times] += ramp * np.interp(positions, grid, template[lead])
        weight[times] += ramp
    return total / np.maximum(weight, 1.0)


def counterpart_difference(
    raw: np.ndarray, age: float, sex: float, normative: nm.Normative
) -> np.ndarray | None:
    """
    Return the normal counterpart minus the record, ``c - x = T - clean``.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV.
    age, sex : float
        Covariates.
    normative : nm.Normative
        Frozen 062 normative model.

    Returns
    -------
    np.ndarray or None
        (12, 5000) float32 difference, or None when 062 finds fewer than two complete beats.
    """
    raw = np.asarray(raw, np.float64)
    features = nm.record_features(raw)
    if features is None:
        return None
    template = template_beat(normative, float(features.intervals[0]), age, sex)
    timeline = template_timeline(template, features.knots, raw.shape[1])
    clean = nm.preprocess(raw, nm.delineate(raw)["qon"])
    return (timeline - clean).astype(np.float32)


def selection_ranks(record_map: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Rank of every scored lead-sample, 0 for the highest map value, ties broken by a random key.

    Parameters
    ----------
    record_map : np.ndarray
        (12, 5000) map.
    key : np.ndarray
        (12, SCORED) uniform tiebreak key; a larger key ranks first among equal values.

    Returns
    -------
    np.ndarray
        (12, SCORED) uint16 ranks.
    """
    values = np.asarray(record_map[:, START:STOP], np.float64).ravel()
    order = np.lexsort((-key.ravel(), -values))
    ranks = np.empty(order.size, np.uint16)
    ranks[order] = np.arange(order.size)
    return ranks.reshape(len(LEADS), SCORED)


def top_leads(record_map: np.ndarray, key: np.ndarray, count: int = TOP_LEADS) -> np.ndarray:
    """
    Leads with the highest 061 lead score, ties broken by a random key.

    Parameters
    ----------
    record_map : np.ndarray
        (12, 5000) map.
    key : np.ndarray
        (12,) uniform tiebreak key.
    count : int
        Leads to return.

    Returns
    -------
    np.ndarray
        (count,) lead indices, highest first.
    """
    return np.lexsort((-key, -lead_scores(np.asarray(record_map, np.float64))))[:count]


def control_draws(
    labels: np.ndarray, rng: np.random.Generator, draws: int = CONTROL_DRAWS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lead permutations and within-label cyclic sources of the matched random control.

    The control mask is ``M[inverse[r]][:, sources[r]]``: the selection of lead ``l`` moves to lead
    ``perm[l]`` and the selection at the j-th sample of a label moves to the (j + offset)-th sample of
    that label.

    Parameters
    ----------
    labels : np.ndarray
        (SCORED,) wave labels of the scored samples.
    rng : np.random.Generator
        Generator.
    draws : int
        Number of draws.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``inverse`` (draws, 12) inverse lead permutations and ``sources`` (draws, SCORED) int64 sample
        sources.
    """
    inverse = np.zeros((draws, len(LEADS)), np.int64)
    sources = np.zeros((draws, len(labels)), np.int64)
    for r in range(draws):
        inverse[r] = np.argsort(rng.permutation(len(LEADS)))
        source = np.arange(len(labels))
        for label in range(len(LABELS)):
            members = np.flatnonzero(labels == label)
            if not len(members):
                continue
            offset = rng.integers(0, len(members))
            source[members] = members[(np.arange(len(members)) - offset) % len(members)]
        sources[r] = source
    return inverse, sources


def record_plan(ecg_id: int, maps: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """
    Everything random or map-derived that the scoring of one record needs.

    Parameters
    ----------
    ecg_id : int
        Record ID; seeds ``default_rng([SEED, ecg_id])``.
    maps : np.ndarray
        (maps, 12, 5000) maps in the declared order.
    labels : np.ndarray
        (5000,) wave labels.

    Returns
    -------
    dict[str, np.ndarray]
        ``ranks`` (maps, 12, SCORED) uint16, ``leads`` (maps, 3), ``inverse`` (draws, 12), ``sources``
        (draws, SCORED), ``triplets`` (LEAD_DRAWS, 3) and ``composition`` (maps, labels) top-10% shares.
    """
    rng = np.random.default_rng([SEED, ecg_id])
    key = rng.random((len(LEADS), SCORED))
    lead_key = rng.random(len(LEADS))
    scored = labels[START:STOP]
    inverse, sources = control_draws(scored, rng)
    triplets = np.stack([rng.choice(len(LEADS), TOP_LEADS, replace=False) for _ in range(LEAD_DRAWS)])
    ranks = np.stack([selection_ranks(m, key) for m in maps])
    count = selection_count(FRACTIONS[1])
    composition = np.stack(
        [
            np.bincount(np.broadcast_to(scored, r.shape)[r < count], minlength=len(LABELS)) / count
            for r in ranks
        ]
    )
    return {
        "ranks": ranks,
        "leads": np.stack([top_leads(m, lead_key) for m in maps]),
        "inverse": inverse,
        "sources": sources,
        "triplets": triplets,
        "composition": composition,
    }


def selection_count(fraction: float) -> int:
    """Return the number of lead-samples in a top-``fraction`` region of the scored range."""
    return round(fraction * len(LEADS) * SCORED)


def pass_layout(maps: int) -> dict[str, int]:
    """
    Positions of the passes of one record.

    Order: original (zero mask), whole scored range, then per map and fraction the map region followed by its
    ``CONTROL_DRAWS`` controls, then each map's top-3 leads, then the ``LEAD_DRAWS`` random 3-lead sets.

    Parameters
    ----------
    maps : int
        Number of maps.

    Returns
    -------
    dict[str, int]
        ``original``, ``full``, ``regions`` (first region pass), ``leads``, ``lead_controls`` and ``passes``.
    """
    regions = 2
    leads = regions + maps * len(FRACTIONS) * (1 + CONTROL_DRAWS)
    controls = leads + maps
    return {
        "original": 0,
        "full": 1,
        "regions": regions,
        "leads": leads,
        "lead_controls": controls,
        "passes": controls + LEAD_DRAWS,
    }


def pass_masks(plan: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    """
    Binary replacement masks of every pass of one record, in ``pass_layout`` order.

    Parameters
    ----------
    plan : dict[str, torch.Tensor]
        ``record_plan`` arrays as tensors (``ranks`` as int32).
    device : str
        Torch device.

    Returns
    -------
    torch.Tensor
        (passes, 12, 5000) float32 masks.
    """
    ranks = plan["ranks"].to(device)
    maps = ranks.shape[0]
    layout = pass_layout(maps)
    counts = torch.tensor([selection_count(f) for f in FRACTIONS], device=device)
    selected = ranks[:, None] < counts[None, :, None, None]
    inverse = plan["inverse"].to(device)
    sources = plan["sources"].to(device)
    moved = selected[:, :, inverse]
    index = sources[None, None, :, None, :].expand(maps, len(FRACTIONS), -1, len(LEADS), -1)
    controls = torch.gather(moved, -1, index)
    regions = torch.cat([selected[:, :, None], controls], dim=2).reshape(-1, len(LEADS), SCORED)
    masks = torch.zeros((layout["passes"], len(LEADS), SAMPLES), dtype=torch.float32, device=device)
    masks[layout["full"], :, START:STOP] = 1.0
    masks[layout["regions"] : layout["leads"], :, START:STOP] = regions.float()
    lead_sets = torch.cat([plan["leads"].to(device), plan["triplets"].to(device)])
    rows = torch.arange(len(lead_sets), device=device)[:, None].expand_as(lead_sets) + layout["leads"]
    masks[rows.reshape(-1), lead_sets.reshape(-1), START:STOP] = 1.0
    return masks


def taper(masks: torch.Tensor, length: int = TAPER) -> torch.Tensor:
    """
    Convolve binary masks along time with a normalized Hann window.

    Parameters
    ----------
    masks : torch.Tensor
        (..., samples) masks.
    length : int
        Odd window length in samples.

    Returns
    -------
    torch.Tensor
        Weights in [0, 1] of the same shape.
    """
    window = torch.hann_window(length + 2, periodic=False, dtype=masks.dtype, device=masks.device)[1:-1]
    window = (window / window.sum()).view(1, 1, -1)
    flat = masks.reshape(-1, 1, masks.shape[-1])
    return functional.conv1d(flat, window, padding=length // 2).reshape(masks.shape).clamp(0.0, 1.0)


def replaced(signal: torch.Tensor, difference: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Blend the counterpart into the record: ``x + w * (c - x)``.

    Parameters
    ----------
    signal : torch.Tensor
        (12, samples) record.
    difference : torch.Tensor
        (12, samples) counterpart minus record.
    weights : torch.Tensor
        (passes, 12, samples) tapered weights.

    Returns
    -------
    torch.Tensor
        (passes, 12, samples) replaced signals.
    """
    return signal[None] + weights * difference[None]


class Block(nn.Module):
    """Residual block: two kernel-9 convolutions with batch norm and a projection shortcut when needed."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, 9, stride, 4, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, 9, 1, 4, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.shortcut = (
            nn.Identity()
            if c_in == c_out and stride == 1
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return functional.relu(self.body(h) + self.shortcut(h))


class Detector(nn.Module):
    """
    Small 1D ResNet abnormal-ECG detector on 250 Hz input with global average pooling.

    The input is the raw (batch, 12, samples) signal in mV at 500 Hz; it is averaged in pairs and clipped to
    +-10 mV inside ``forward``, so any length works.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(len(LEADS), DETECTOR_WIDTHS[0], 15, 2, 7, bias=False),
            nn.BatchNorm1d(DETECTOR_WIDTHS[0]),
            nn.ReLU(),
        )
        widths = (DETECTOR_WIDTHS[0], *DETECTOR_WIDTHS)
        self.blocks = nn.Sequential(
            *(Block(widths[k], widths[k + 1], DETECTOR_STRIDES[k]) for k in range(len(DETECTOR_STRIDES)))
        )
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(DETECTOR_WIDTHS[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abnormal logit per record.

        Parameters
        ----------
        x : torch.Tensor
            (batch, 12, samples) raw signals in mV at 500 Hz.

        Returns
        -------
        torch.Tensor
            (batch,) logits.
        """
        h = functional.avg_pool1d(x, 2).clamp(-CLIP_MV, CLIP_MV)
        h = self.blocks(self.stem(h)).mean(dim=-1)
        return self.head(self.dropout(h)).squeeze(-1)


def filter_matrix(samples: int = SAMPLES) -> np.ndarray:
    """
    065's zero-phase 0.5-40 Hz band-pass as a matrix: ``filtered = signal @ matrix``.

    ``scipy.signal.sosfiltfilt`` with its default odd padding is linear in the signal, so row i is the
    response to the i-th unit impulse.

    Parameters
    ----------
    samples : int
        Signal length.

    Returns
    -------
    np.ndarray
        (samples, samples) float64.
    """
    sos = butter(2, [0.5, 40.0], btype="band", fs=mil.FS, output="sos")
    return sosfiltfilt(sos, np.eye(samples), axis=-1)


class MilDetector(nn.Module):
    """
    065 ``mil_lse`` ensemble as an abnormal detector: minus the mean NORM bag logit of the networks.

    Parameters
    ----------
    networks : list[mil.LeadMIL]
        The three 065 networks.
    norm_index : int
        Index of the NORM head.
    matrix : np.ndarray
        ``filter_matrix`` output.
    """

    def __init__(self, networks: list[mil.LeadMIL], norm_index: int, matrix: np.ndarray) -> None:
        super().__init__()
        self.networks = nn.ModuleList(networks)
        self.norm_index = norm_index
        self.register_buffer("matrix", torch.from_numpy(matrix.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abnormal logit per record.

        Parameters
        ----------
        x : torch.Tensor
            (batch, 12, 5000) raw signals in mV.

        Returns
        -------
        torch.Tensor
            (batch,) logits.
        """
        filtered = torch.matmul(x, self.matrix).clamp(-mil.CLIP_MV, mil.CLIP_MV)
        bags = torch.stack([network(filtered)[0][:, self.norm_index] for network in self.networks])
        return -bags.mean(dim=0)


@torch.no_grad()
def record_logits(
    detector: nn.Module,
    signal: np.ndarray,
    difference: np.ndarray,
    plan: dict[str, np.ndarray],
    device: str,
    batch: int = 128,
) -> np.ndarray:
    """
    Detector logits of every pass of one record.

    Parameters
    ----------
    detector : nn.Module
        Detector in evaluation mode.
    signal : np.ndarray
        (12, 5000) record in mV.
    difference : np.ndarray
        (12, 5000) counterpart minus record.
    plan : dict[str, np.ndarray]
        ``record_plan`` output.
    device : str
        Torch device.
    batch : int
        Passes per forward call.

    Returns
    -------
    np.ndarray
        (passes,) float32 logits in ``pass_layout`` order.
    """
    tensors = {
        name: torch.from_numpy(np.asarray(plan[name]).astype(np.int64))
        for name in plan
        if name != "composition"
    }
    weights = taper(pass_masks(tensors, device))
    x = torch.from_numpy(np.asarray(signal, np.float32)).to(device)
    d = torch.from_numpy(np.array(difference, np.float32)).to(device)
    out = []
    for start in range(0, len(weights), batch):
        out.append(detector(replaced(x, d, weights[start : start + batch])).float().cpu())
    return torch.cat(out).numpy()


def readings(logits: np.ndarray, maps: int) -> dict[str, np.ndarray]:
    """
    Per-record drops and scores from the pass logits.

    Parameters
    ----------
    logits : np.ndarray
        (records, passes) logits.
    maps : int
        Number of maps.

    Returns
    -------
    dict[str, np.ndarray]
        ``original`` (records,), ``full`` drop (records,), ``drop_map`` and ``drop_control`` and ``S``
        (records, maps, fractions), ``AOPC_raw`` and ``AOPC_matched`` (records, maps), ``drop_leads``,
        ``drop_lead_control`` and ``W`` (records, maps).
    """
    layout = pass_layout(maps)
    original = logits[:, layout["original"]]
    drops = original[:, None] - logits
    regions = drops[:, layout["regions"] : layout["leads"]].reshape(len(logits), maps, len(FRACTIONS), -1)
    drop_map = regions[..., 0]
    drop_control = regions[..., 1:].mean(axis=-1)
    score = drop_map - drop_control
    drop_leads = drops[:, layout["leads"] : layout["lead_controls"]]
    drop_lead_control = drops[:, layout["lead_controls"] :].mean(axis=1)
    return {
        "original": original,
        "full": drops[:, layout["full"]],
        "drop_map": drop_map,
        "drop_control": drop_control,
        "S": score,
        "AOPC_raw": drop_map.mean(axis=-1),
        "AOPC_matched": score.mean(axis=-1),
        "drop_leads": drop_leads,
        "drop_lead_control": np.broadcast_to(drop_lead_control[:, None], drop_leads.shape),
        "W": drop_leads - drop_lead_control[:, None],
    }


def bootstrap_means(values: np.ndarray, patients: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """
    Column means on whole-patient bootstrap draws (``intervals`` convention).

    Parameters
    ----------
    values : np.ndarray
        (records, columns) values.
    patients : np.ndarray
        Patient of each record.
    draws : int
        Number of draws.
    seed : int
        Seed of a fresh ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        (draws, columns) means.
    """
    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    return np.stack([values[patient_resample(groups, rng)].mean(axis=0) for _ in range(draws)])


def interval(point: float, draws: np.ndarray) -> dict[str, float]:
    """
    Point estimate and 2.5 and 97.5 percentiles of bootstrap draws.

    Parameters
    ----------
    point : float
        Full-sample value.
    draws : np.ndarray
        (draws,) bootstrap values.

    Returns
    -------
    dict[str, float]
        ``value``, ``ci_low`` and ``ci_high``.
    """
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": float(point), "ci_low": float(low), "ci_high": float(high)}


def verdict(result: dict[str, float]) -> str:
    """Return ``above``, ``below`` or ``contains 0`` for an interval."""
    if result["ci_low"] > 0:
        return "above"
    if result["ci_high"] < 0:
        return "below"
    return "contains 0"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, NaN when either vector is constant."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def joint_spearman(
    scores: np.ndarray,
    score_patients: np.ndarray,
    marks: np.ndarray,
    mark_patients: np.ndarray,
    draws: int,
    seeds: tuple[int, int],
) -> dict[str, float]:
    """
    Spearman correlation across maps of mean replacement score and mean A2, with independent patient draws.

    Parameters
    ----------
    scores : np.ndarray
        (records, maps) replacement scores.
    score_patients : np.ndarray
        Patient of each score record.
    marks : np.ndarray
        (records, maps) per-record A2 (no NaN).
    mark_patients : np.ndarray
        Patient of each A2 record.
    draws : int
        Bootstrap draws.
    seeds : tuple[int, int]
        Seeds of the score and A2 draws.

    Returns
    -------
    dict[str, float]
        ``value``, ``ci_low``, ``ci_high`` and ``skipped_draws``.
    """
    a = bootstrap_means(scores, score_patients, draws, seeds[0])
    b = bootstrap_means(marks, mark_patients, draws, seeds[1])
    values = np.array([spearman(x, y) for x, y in zip(a, b, strict=True)])
    kept = values[np.isfinite(values)]
    out = interval(spearman(scores.mean(axis=0), marks.mean(axis=0)), kept)
    return {**out, "skipped_draws": int(len(values) - len(kept))}


def map_spearman(scores: np.ndarray, marks: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    """
    Spearman correlation of two per-map vectors with a bootstrap over maps.

    Parameters
    ----------
    scores, marks : np.ndarray
        (maps,) values.
    draws : int
        Bootstrap draws.
    seed : int
        Seed.

    Returns
    -------
    dict[str, float]
        ``value``, ``ci_low``, ``ci_high`` and ``skipped_draws``.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        rows = rng.integers(0, len(scores), len(scores))
        values.append(spearman(scores[rows], marks[rows]))
    values = np.array(values)
    kept = values[np.isfinite(values)]
    return {**interval(spearman(scores, marks), kept), "skipped_draws": int(len(values) - len(kept))}


def detector_crops(
    signals: np.ndarray, rows: np.ndarray, rng: np.random.Generator, device: str, crop: int = CROP
) -> torch.Tensor:
    """
    One random time crop per record (a random shift of up to ``SAMPLES - crop`` samples).

    Parameters
    ----------
    signals : np.ndarray
        (n, 12, 5000) signals in mV.
    rows : np.ndarray
        Rows to take.
    rng : np.random.Generator
        Generator for the crop starts.
    device : str
        Torch device.
    crop : int
        Crop length in samples.

    Returns
    -------
    torch.Tensor
        (len(rows), 12, crop) float32.
    """
    starts = rng.integers(0, signals.shape[-1] - crop + 1, len(rows))
    batch = np.stack([signals[r, :, s : s + crop] for r, s in zip(rows, starts, strict=True)])
    return torch.from_numpy(np.ascontiguousarray(batch, np.float32)).to(device)


@torch.no_grad()
def full_logits(
    detector: nn.Module, signals: np.ndarray, rows: np.ndarray, device: str, batch: int = 256
) -> np.ndarray:
    """
    Detector logits of full-length records.

    Parameters
    ----------
    detector : nn.Module
        Detector.
    signals : np.ndarray
        (n, 12, 5000) signals in mV.
    rows : np.ndarray
        Rows to score.
    device : str
        Torch device.
    batch : int
        Records per forward call.

    Returns
    -------
    np.ndarray
        (len(rows),) float32 logits.
    """
    detector.eval()
    out = []
    for start in range(0, len(rows), batch):
        x = torch.from_numpy(np.ascontiguousarray(signals[rows[start : start + batch]], np.float32)).to(
            device
        )
        out.append(detector(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_detector(
    signals: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    held_rows: np.ndarray,
    device: str,
    seed: int = SEED,
    max_epochs: int = 40,
    patience: int = 6,
    batch: int = 64,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    """
    Train the detector with random time crops and early stopping on the held-out AUROC.

    Parameters
    ----------
    signals : np.ndarray
        (n, 12, 5000) signals in mV.
    y : np.ndarray
        (n,) binary labels.
    train_rows, held_rows : np.ndarray
        Training and held-out rows.
    device : str
        Torch device.
    seed : int
        Seed of the weights, batch order and crops.
    max_epochs : int
        Largest number of epochs.
    patience : int
        Epochs without a better held-out AUROC before stopping.
    batch : int
        Batch size.

    Returns
    -------
    tuple[dict[str, torch.Tensor], list[dict[str, float]]]
        Best state dict (CPU) and per-epoch history (loss, held-out AUROC, seconds).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    detector = Detector().to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-3, weight_decay=1e-2)
    target = torch.from_numpy(np.asarray(y, np.float32))
    best, best_state, history, stale = -np.inf, None, [], 0
    for epoch in range(max_epochs):
        began = time.time()
        detector.train()
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order) - batch + 1, batch):
            rows = order[start : start + batch]
            logits = detector(detector_crops(signals, rows, rng, device))
            loss = functional.binary_cross_entropy_with_logits(logits, target[rows].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        auroc = float(roc_auc_score(y[held_rows], full_logits(detector, signals, held_rows, device)))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "held_auroc": auroc,
                "seconds": time.time() - began,
            }
        )
        if auroc > best:
            best, stale = auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    return best_state, history
