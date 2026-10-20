"""Independent binary decoding and endpoint reconstruction for public LUDB qualification."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
FRACTIONS = (.17, .31, .43, .59, .71, .83)
SYMBOLS = {1: "N", 24: "p", 27: "t", 39: "(", 40: ")"}


def annotation_events(payload: bytes) -> list[tuple[int, str]]:
    """Decode MIT annotation timing words without reading auxiliary text fields."""
    position, sample = 0, 0
    events = []
    while position < len(payload):  # noqa: C901
        if position + 2 > len(payload):
            raise ValueError("Truncated annotation word")
        word = int.from_bytes(payload[position : position + 2], "little")
        position += 2
        code, delta = word >> 10, word & 1023
        if word == 0:
            if any(payload[position:]):
                raise ValueError("Unexpected bytes after annotation EOF")
            return events
        if code == 59:
            sample, position = _skip(payload, sample, position, delta)
        elif code == 63:
            position += delta + delta % 2
            if position > len(payload):
                raise ValueError("Truncated annotation auxiliary field")
        elif code in (60, 61, 62):
            continue
        else:
            if code not in SYMBOLS:
                raise ValueError("Unexpected annotation type")
            sample += delta
            events.append((sample, SYMBOLS[code]))
    raise ValueError("Annotation EOF missing")


def _skip(payload: bytes, sample: int, position: int, delta: int) -> tuple[int, int]:
    """Decode the signed high-word-first PDP-11 long sample increment."""
    if delta != 0 or position + 4 > len(payload):
        raise ValueError("Invalid annotation SKIP")
    high, low = struct.unpack_from("<HH", payload, position)
    increment = (high << 16) | low
    if increment >= 2**31:
        increment -= 2**32
    return sample + increment, position + 4


def wave_triples(events: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """Require ordered complete onset/peak/offset wave triples in 5000 samples."""
    if not events or len(events) % 3:
        raise ValueError("Incomplete wave triples")
    points = [point for point, _ in events]
    if min(points) < 0 or max(points) >= 5000 or points != sorted(points):
        raise ValueError("Annotation timing invalid")
    output = []
    types = {"p": "P", "N": "QRS", "t": "T"}
    for index in range(0, len(events), 3):
        (onset, left), (peak, middle), (offset, right) = events[index : index + 3]
        if left != "(" or right != ")" or middle not in types:
            raise ValueError("Unexpected wave symbols")
        if not onset <= peak <= offset or onset == offset:
            raise ValueError("Degenerate wave timing")
        if output and onset < output[-1]["offset"]:
            raise ValueError("Strictly overlapping wave triples")
        output.append({"wave": types[middle], "onset": onset, "peak": peak, "offset": offset})
    return output


def waveform(record: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read thirteen nonclinical header lines and independently decode format-16 samples."""
    with record.with_suffix(".hea").open() as stream:
        header = [stream.readline().rstrip("\n") for _ in range(13)]
    fields = header[0].split()
    if len(fields) < 4 or fields[:4] != [record.name, "12", "500", "5000"]:
        raise ValueError("Unexpected recording geometry")
    specs = [_signal_spec(line, record.name) for line in header[1:]]
    names = [spec["lead"] for spec in specs]
    if len(set(names)) != 12 or set(names) != set(LEADS):
        raise ValueError("Missing or duplicate canonical lead")
    raw = np.frombuffer(record.with_suffix(".dat").read_bytes(), dtype="<i2")
    if raw.size != 60000 or np.any(raw == -32768):
        raise ValueError("Unexpected format-16 waveform or missing samples")
    gains = np.asarray([spec["gain"] for spec in specs])
    baseline = np.asarray([spec["baseline"] for spec in specs])
    values = (raw.reshape(5000, 12).astype(np.float64) - baseline) / gains
    output = values[:, [names.index(lead) for lead in LEADS]].T
    metadata = {"fs": 500, "samples": 5000, "source_lead_order": names,
                "canonical_lead_order": list(LEADS), "units": [spec["units"] for spec in specs],
                "adc_gain": gains.tolist(), "baseline": baseline.tolist()}
    return output, metadata


def _signal_spec(line: str, name: str) -> dict[str, Any]:
    """Decode exactly one admitted format-16 signal metadata line."""
    fields = line.split()
    if len(fields) < 9 or fields[0] != name + ".dat" or fields[1] != "16":
        raise ValueError("Unsupported signal layout")
    match = re.fullmatch(r"([\d.eE+\-]+)(?:\((-?\d+)\))?/([^\s]+)", fields[2])
    if match is None:
        raise ValueError("Invalid gain/baseline/unit metadata")
    gain = float(match.group(1))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Invalid signal gain")
    baseline = int(match.group(2)) if match.group(2) is not None else int(fields[4])
    return {"lead": fields[-1].lower(), "gain": gain, "baseline": baseline, "units": match.group(3)}


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """Compute noninterpolated AP by positive counts at complete descending score ties."""
    target, score = np.asarray(target), np.asarray(score)
    if target.ndim != 1 or target.shape != score.shape or not np.isfinite(score).all():
        raise ValueError("AP inputs invalid")
    if not np.isin(target, [0, 1]).all() or not 0 < target.sum() < len(target):
        raise ValueError("AP requires two target classes")
    order = np.argsort(-score, kind="stable")
    sorted_scores, sorted_y = score[order], target[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(target)]
    positives = np.cumsum(sorted_y, dtype=np.int64)[ends - 1]
    added = np.diff(np.r_[0, positives])
    return float(np.sum(added * positives / ends, dtype=np.float64) / target.sum())


def coverage_row(values: np.ndarray, waves: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    """Reconstruct coverage and inclusive QRS target without calculating a score."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.shape != (5000,) or not np.isfinite(samples).all():
        raise ValueError("Invalid waveform samples")
    mask = np.zeros(5000, dtype=np.uint8)
    counts = dict.fromkeys(("P", "QRS", "T"), 0)
    for wave in waves:
        counts[wave["wave"]] += 1
        if wave["wave"] == "QRS":
            mask[wave["onset"] : wave["offset"] + 1] = 1
    first, stop = waves[0]["onset"], waves[-1]["offset"] + 1
    target = mask[first:stop]
    if not 0 < target.sum() < len(target):
        raise ValueError("Coverage requires both QRS and non-QRS samples")
    return {"start": first, "end_exclusive": stop, "coverage_samples": stop - first,
            "qrs_samples": int(target.sum()), "prevalence": float(target.mean()), "wave_counts": counts,
            "flat_signal": bool(np.ptp(samples) == 0)}, target


def lead_row(values: np.ndarray, waves: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently reconstruct fixed derivative score, coverage targets and all controls."""
    metadata, target = coverage_row(values, waves)
    samples = np.asarray(values, dtype=np.float64)
    difference = np.r_[0., samples[1:] - samples[:-1]]
    energy = np.convolve(difference * difference, np.full(20, .05), mode="same")
    if energy.max() > 0:
        energy /= energy.max()
    score = energy[metadata["start"] : metadata["end_exclusive"]]
    offsets = [int(len(score) * fraction) for fraction in FRACTIONS]
    controls = [average_precision(target, np.roll(score, offset)) for offset in offsets]
    observed, shifted = average_precision(target, score), float(np.mean(controls))
    return {**metadata, "AP": observed, "shift_AP": controls, "shift_samples": offsets,
            "mean_shift_AP": shifted, "gain": observed - shifted}


def record_row(record: Path) -> dict[str, Any]:
    """Reconstruct one complete per-record row using no main parser or endpoint code."""
    values, metadata = waveform(record)
    number = int(record.name)
    role = "parser_qualification" if number % 5 == 0 else "measurement"
    rows = []
    for lead, signal in zip(LEADS, values, strict=True):
        waves = wave_triples(annotation_events(record.with_suffix("." + lead).read_bytes()))
        metrics = (coverage_row(signal, waves)[0] if role == "parser_qualification"
                   else lead_row(signal, waves))
        rows.append({"lead": lead, **metrics, "waves": waves})
    result = {"record_id": number, "role": role, "metadata": metadata, "leads": rows}
    if role == "measurement":
        result.update({key: float(np.mean([row[key] for row in rows]))
                       for key in ("AP", "mean_shift_AP", "gain")})
    return result


def summary(rows: list[dict[str, Any]], draws: int = 2000, seed: int = 2026106101) -> dict[str, Any]:
    """Reconstruct equal-record endpoints and paired whole-record bootstrap without main code."""
    if [row["record_id"] for row in rows] != list(range(1, 201)):
        raise ValueError("Require all200 numeric records in order")
    selected = [row for row in rows if row["record_id"] % 5]
    groups = patient_groups(np.asarray([row["record_id"] for row in selected]))
    random = np.random.default_rng(seed)
    gains = np.asarray([row["gain"] for row in selected])
    boot = [float(gains[patient_resample(groups, random)].mean()) for _ in range(draws)]
    lower, upper = np.percentile(boot, [2.5, 97.5], method="linear")
    point = float(gains.mean())
    return {"measurement_records": 160, "parser_qualification_records": 40, "measurement_leads": 1920,
            "AP": float(np.mean([row["AP"] for row in selected])),
            "mean_shift_AP": float(np.mean([row["mean_shift_AP"] for row in selected])),
            "gain": point, "gain_interval": [float(lower), float(upper)], "bootstrap_draws": draws,
            "bootstrap_seed": seed, "skipped_draws": 0,
            "scientific_passed": bool(lower > 0 and point >= .05),
            "per_lead": {lead: {key: float(np.mean([row["leads"][index][key] for row in selected]))
                                for key in ("AP", "mean_shift_AP", "gain", "prevalence")}
                         for index, lead in enumerate(LEADS)}}
