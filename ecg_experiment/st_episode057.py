"""Label-free waveform changes and independent episode metrics for Experiment 057."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_experiment.intervals import patient_groups, patient_resample

FS = 250
SEED = 57057
BEAT_SYMBOLS = frozenset(
    ("N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?")
)
REPEATED_SUBJECTS = (
    ("e0118", "e0119", "e0121", "e0122"),
    ("e0123", "e0124", "e0125", "e0126"),
    ("e0129", "e0133"),
    ("e0136", "e0139"),
    ("e0147", "e0148"),
    ("e0154", "e0155"),
    ("e0162", "e0163"),
)
EPISODE_PATTERN = re.compile(r"(\()?([Aa])?(ST|T|st|t)([01])([+-]{1,2})(\d+)?(\))?")


def subject_id(record: str) -> str:
    """Return the published repeated-subject grouping.

    Parameters
    ----------
    record : str
        European ST-T record name.

    Returns
    -------
    str
        First record name of the subject's group.
    """
    return next((group[0] for group in REPEATED_SUBJECTS if record in group), record)


def change_windows(signal: np.ndarray, anchors: np.ndarray) -> dict[str, np.ndarray]:
    """Score shared windows without receiving episode or beat-identity labels.

    Parameters
    ----------
    signal : np.ndarray
        Original two-channel waveform in mV, samples by channels, at 250 Hz.
    anchors : np.ndarray
        All annotated beat timestamps, with beat identity discarded.

    Returns
    -------
    dict
        Beat and shared window features on the original time axis.
    """
    if signal.ndim != 2 or signal.shape[1] != 2 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite two-channel waveform")
    offsets = np.arange(round(-0.25 * FS), round(0.45 * FS))
    anchors = np.asarray(anchors, dtype=int)
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("Beat anchors must be strictly increasing")
    anchors = anchors[(anchors + offsets[0] >= 0) & (anchors + offsets[-1] < len(signal))]
    times = anchors / FS
    reference_rows = times < 30
    if reference_rows.sum() < 3:
        raise ValueError("Fewer than three complete early-reference beats")
    beats = signal[anchors[:, None] + offsets[None, :]].transpose(0, 2, 1).copy()
    baseline = (offsets >= round(-0.10 * FS)) & (offsets < round(-0.06 * FS))
    beats -= np.median(beats[:, :, baseline], axis=2, keepdims=True)
    reference = np.median(beats[reference_rows], axis=0)
    whole = (offsets >= round(-0.06 * FS)) & (offsets < round(0.45 * FS))
    st = (offsets >= round(0.12 * FS)) & (offsets < round(0.16 * FS))
    beat_scores = {
        "whole": np.sqrt(np.mean((beats[:, :, whole] - reference[None, :, whole]) ** 2, axis=2)),
        "st": np.abs(np.mean(beats[:, :, st], axis=2) - np.mean(reference[:, st], axis=1)),
    }
    bins = np.floor(times / 30).astype(int)
    candidates = np.arange(1, len(signal) // (FS * 30))
    counts = np.bincount(bins, minlength=len(candidates) + 1)[candidates]
    candidates = candidates[counts >= 3]
    counts = counts[counts >= 3]
    scores = {}
    for name, values in beat_scores.items():
        scores[name] = np.stack(
            [
                np.bincount(bins, weights=values[:, channel], minlength=241)[candidates] / counts
                for channel in range(2)
            ],
            axis=1,
        )
    if not len(candidates):
        raise ValueError("No shared candidate windows")
    return {
        "anchors": anchors,
        "reference": reference,
        "offsets": offsets,
        "beat_whole": beat_scores["whole"],
        "beat_st": beat_scores["st"],
        "centers": candidates * 30 + 15,
        "counts": counts,
        **scores,
    }


def parse_episodes(
    samples: np.ndarray,
    notes: list[str],
    *,
    allow_censored: bool = False,
) -> list[dict[str, Any]]:
    """Pair case-sensitive expert episodes without conflating positional shifts.

    Parameters
    ----------
    samples : np.ndarray
        Native annotation timestamps at 250 Hz.
    notes : list of str
        Expert auxiliary annotations, supplied only after feature computation.
    allow_censored : bool
        Preserve unfinished episodes with unknown end for conservative evaluation censoring.

    Returns
    -------
    list of dict
        Episodes with kind, channel, sign, onset/end and optional extrema.
    """
    opened: dict[tuple[str, int, str], dict[str, Any]] = {}
    episodes = []
    for sample, raw in zip(samples, notes, strict=True):
        note = raw.split("\x00", 1)[0].strip()
        match = EPISODE_PATTERN.fullmatch(note)
        if match is None:
            if re.match(r"\(?[Aa]?(?:ST|T|st|t)[01][+-]", note):
                raise ValueError(f"Malformed episode annotation: {note!r}")
            continue
        _episode_marker(match, int(sample) / FS, opened, episodes)
    if opened and not allow_censored:
        raise ValueError(f"Unclosed episodes: {list(opened)}")
    for episode in opened.values():
        episode.update(end=None, censored=True)
        episodes.append(episode)
    return episodes


def _episode_marker(
    match: re.Match[str],
    time: float,
    opened: dict[tuple[str, int, str], dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> None:
    start, extremum, kind, channel_text, sign, displacement, end = match.groups()
    channel = int(channel_text)
    key = (kind, channel, sign)
    if start:
        if key in opened or extremum or end:
            raise ValueError(f"Invalid episode start: {match.group()}")
        opened[key] = {"kind": kind, "channel": channel, "sign": sign, "start": time, "peaks": []}
        return
    if key not in opened:
        raise ValueError(f"Unpaired episode marker: {match.group()}")
    if extremum:
        if displacement is None:
            raise ValueError(f"Unpaired episode extremum: {match.group()}")
        opened[key]["peaks"].append({"time": time, "microvolts": int(displacement)})
        return
    if not end:
        raise ValueError(f"Episode marker has no role: {match.group()}")
    episode = opened.pop(key)
    if time <= episode["start"]:
        raise ValueError("Nonpositive episode duration")
    episode["end"] = time
    episodes.append(episode)


def episode_targets(centers: np.ndarray, episodes: list[dict[str, Any]], kind: str) -> np.ndarray:
    """Mark same-channel candidate centers inside the specified expert episodes.

    Parameters
    ----------
    centers : np.ndarray
        Original-time shared window centers in seconds.
    episodes : list of dict
        Independently parsed expert targets.
    kind : str
        Exact case-sensitive episode kind, such as ST or st.

    Returns
    -------
    np.ndarray
        Boolean window-by-channel target array.
    """
    target = np.zeros((len(centers), 2), dtype=bool)
    for episode in episodes:
        if episode["kind"] == kind and episode["end"] is not None:
            target[:, episode["channel"]] |= (centers >= episode["start"]) & (centers <= episode["end"])
    return target


def location_summary(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    """Compute tie-aware and displayed hits on an identical candidate domain.

    Parameters
    ----------
    scores : np.ndarray
        Window-by-channel scores.
    targets : np.ndarray
        Same-shaped independent expert target mask.

    Returns
    -------
    dict
        Expected top hit, deterministic hit, chance and tie count.
    """
    if scores.shape != targets.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid aligned window scores and targets")
    tied = np.isclose(scores, scores.max(), rtol=1e-10, atol=1e-10)
    return {
        "hit": float(targets[tied].mean()),
        "displayed_hit": float(targets.ravel()[np.argmax(scores)]),
        "chance": float(targets.mean()),
        "ties": int(tied.sum()),
    }


def mean_interval(values: np.ndarray, patients: np.ndarray) -> dict[str, float | int]:
    """Average records within subjects and bootstrap those subject means.

    Parameters
    ----------
    values : np.ndarray
        Finite metric per record, aligned with patient identifiers.
    patients : np.ndarray
        Published subject identity for every record.

    Returns
    -------
    dict
        Subject-macro estimate and paired-compatible 95% interval.
    """
    groups = patient_groups(patients)
    macro = np.array([np.mean(values[group]) for group in groups])
    unique_groups = patient_groups(np.arange(len(macro)))
    rng = np.random.default_rng(SEED)
    draws = np.array([macro[patient_resample(unique_groups, rng)].mean() for _ in range(2000)])
    return {
        "value": float(macro.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "patients": len(macro),
        "draws": 2000,
        "seed": SEED,
    }


def subject_auroc(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report subject-macro ST-versus-other-window ranking with skip counts.

    Parameters
    ----------
    windows : list of dict
        Per-record targets, scores and subject identities.

    Returns
    -------
    dict
        Both methods' subject-macro AUROC and excluded single-class subjects.
    """
    patients = np.asarray([row["patient"] for row in windows])
    output: dict[str, Any] = {"skipped_single_class": 0, "subjects": []}
    for group in patient_groups(patients):
        y = np.concatenate([windows[index]["target"].ravel() for index in group])
        if len(np.unique(y)) < 2:
            output["skipped_single_class"] += 1
            continue
        row = {"patient": str(patients[group[0]])}
        for method in ("st", "whole"):
            score = np.concatenate([windows[index][method].ravel() for index in group])
            row[method] = float(roc_auc_score(y, score))
        output["subjects"].append(row)
    for method in ("st", "whole"):
        output[method] = (
            float(np.mean([row[method] for row in output["subjects"]])) if output["subjects"] else None
        )
    return output
