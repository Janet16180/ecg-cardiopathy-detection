"""Qualify a frozen scale-free activity proxy against public LUDB wave annotations."""

from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from sklearn.metrics import average_precision_score

from ecg_experiment.intervals import patient_groups, patient_resample

LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
SHIFTS = (.17, .31, .43, .59, .71, .83)
PEAKS = {"p": "P", "N": "QRS", "t": "T"}


def parse_waves(samples: np.ndarray, symbols: list[str], length: int = 5000) -> list[dict[str, Any]]:
    """Parse complete ordered onset/peak/offset triplets without repairing malformed annotations.

    Parameters
    ----------
    samples : np.ndarray
        WFDB integer event positions.
    symbols : list[str]
        Corresponding unmodified WFDB symbols.
    length : int
        Recording sample count.

    Returns
    -------
    list[dict]
        Every original valid triplet, with inclusive offsets.
    """
    points = np.asarray(samples)
    if points.ndim != 1 or points.dtype.kind not in "iu" or len(points) != len(symbols):
        raise ValueError("Annotation samples and symbols must be aligned integer vectors")
    if not len(points) or len(points) % 3 or np.any(np.diff(points) < 0):
        raise ValueError("Annotations must be complete ordered triplets")
    if points.min() < 0 or points.max() >= length:
        raise ValueError("Annotation sample outside recording")
    waves = []
    for index in range(0, len(points), 3):
        onset, peak, offset = map(int, points[index : index + 3])
        symbol = symbols[index + 1]
        if symbols[index] != "(" or symbols[index + 2] != ")" or symbol not in PEAKS:
            raise ValueError("Unexpected onset/peak/offset symbols")
        if not onset <= peak <= offset or onset == offset:
            raise ValueError("Invalid onset/peak/offset ordering")
        waves.append({"wave": PEAKS[symbol], "onset": onset, "peak": peak, "offset": offset})
    return waves


def targets(waves: list[dict[str, Any]], length: int = 5000) -> tuple[np.ndarray, tuple[int, int]]:
    """Rasterize inclusive QRS spans inside the first-through-last annotated wave envelope.

    Parameters
    ----------
    waves : list[dict]
        Complete parsed annotation triples.
    length : int
        Full recording sample count.

    Returns
    -------
    tuple
        Full binary QRS mask and half-open annotation envelope.
    """
    target = np.zeros(length, dtype=np.uint8)
    for wave in waves:
        if wave["wave"] == "QRS":
            target[wave["onset"] : wave["offset"] + 1] = 1
    start, end = waves[0]["onset"], waves[-1]["offset"] + 1
    if not 0 < target[start:end].sum() < end - start:
        raise ValueError("Each annotated lead envelope requires QRS positives and negatives")
    return target, (start, end)


def activity_proxy(signal: np.ndarray) -> np.ndarray:
    """Return a full-record normalized20-sample mean squared prepend-first difference.

    Parameters
    ----------
    signal : np.ndarray
        One physical lead with exactly5000 finite samples.

    Returns
    -------
    np.ndarray
        Float64 full-record activity ranks scaled by their positive maximum.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) != 5000 or not np.isfinite(values).all():
        raise ValueError("Activity proxy requires5000 finite samples")
    derivative = np.diff(values, prepend=values[0])
    energy = np.convolve(derivative**2, np.ones(20, dtype=np.float64) / 20, mode="same")
    if not np.isfinite(energy).all():
        raise ValueError("Activity proxy overflow or nonfinite arithmetic")
    maximum = float(energy.max())
    return energy / maximum if maximum > 0 else energy


def coverage_metrics(waves: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe admitted annotation coverage without computing any activity or AP.

    Parameters
    ----------
    waves : list[dict]
        Complete valid original annotation triples.

    Returns
    -------
    dict
        Coverage, QRS occupancy and all three wave counts.
    """
    mask, (start, end) = targets(waves)
    return {"start": start, "end_exclusive": end, "coverage_samples": end - start,
            "coverage_fraction": (end - start) / 5000, "qrs_samples": int(mask[start:end].sum()),
            "prevalence": float(mask[start:end].mean()),
            "wave_counts": {name: sum(wave["wave"] == name for wave in waves)
                            for name in ["P", "QRS", "T"]}}


def lead_metrics(signal: np.ndarray, waves: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    """Score one lead using its annotation envelope and six fixed within-envelope circular shifts.

    Parameters
    ----------
    signal : np.ndarray
        One complete physical lead.
    waves : list[dict]
        Independently parsed annotations for this lead.

    Returns
    -------
    tuple
        Original and six control APs, coverage metadata and full activity score.
    """
    mask, (start, end) = targets(waves)
    proxy = activity_proxy(signal)
    y, score = mask[start:end], proxy[start:end]
    shifts = [int(np.floor(len(score) * fraction)) for fraction in SHIFTS]
    if len(set(shifts)) != 6 or not all(0 < shift < len(score) for shift in shifts):
        raise ValueError("Fixed circular shifts must be distinct and nonzero")
    observed = float(average_precision_score(y, score))
    controls = [float(average_precision_score(y, np.roll(score, shift))) for shift in shifts]
    tolerance = 8 * np.finfo(np.float64).eps
    if not all(np.isfinite(value) and -tolerance <= value <= 1 + tolerance
               for value in [observed, *controls]):
        raise ValueError("AP outside frozen eight-epsilon admission envelope")
    metrics = {
        **coverage_metrics(waves),
        "AP": observed, "shift_AP": controls, "shift_samples": shifts,
        "mean_shift_AP": float(np.mean(controls)), "gain": observed - float(np.mean(controls)),
        "flat_signal": bool(np.ptp(signal) == 0), "constant_score": bool(np.ptp(score) == 0),
    }
    return metrics, proxy


def read_signal(record: Path, scratch: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only signal metadata lines, then load samples through a sanitized temporary WFDB header.

    Parameters
    ----------
    record : Path
        Authorized local WFDB record path without suffix.
    scratch : Path
        New local directory for a structural-only header and waveform symlink.

    Returns
    -------
    tuple
        Canonical12-lead physical samples and structural signal metadata.
    """
    with record.with_suffix(".hea").open() as stream:
        first = stream.readline().rstrip("\n")
        fields = first.split()
        if len(fields) < 4 or fields[0] != record.name or fields[1:4] != ["12", "500", "5000"]:
            raise ValueError("Expected one500Hz5000-sample12-lead WFDB record")
        lines = [first, *[stream.readline().rstrip("\n") for _ in range(12)]]
    if any(not line or line.startswith("#") for line in lines):
        raise ValueError("Signal metadata header is incomplete")
    if any(line.split()[1] != "16" for line in lines[1:]):
        raise ValueError("Require interleaved16-bit WFDB signals")
    if record.with_suffix(".dat").stat().st_size != 120000:
        raise ValueError("Waveform byte length differs from12x5000int16")
    names = [line.split()[-1].lower() for line in lines[1:]]
    if len(set(names)) != 12 or set(names) != set(LEADS):
        raise ValueError("Canonical lead mapping is incomplete or duplicated")
    files = {line.split()[0] for line in lines[1:]}
    if files != {record.name + ".dat"}:
        raise ValueError("Unexpected waveform filename in header")
    scratch.mkdir(parents=True, exist_ok=False)
    sanitized = scratch / record.name
    sanitized.with_suffix(".hea").write_text("\n".join(lines) + "\n")
    sanitized.with_suffix(".dat").symlink_to(record.with_suffix(".dat"))
    loaded = wfdb.rdrecord(str(sanitized), physical=True)
    if (not np.isfinite(loaded.adc_gain).all() or min(loaded.adc_gain) <= 0
            or not np.isfinite(loaded.baseline).all()):
        raise ValueError("Invalid waveform gains or baselines")
    order = [names.index(name) for name in LEADS]
    values = np.asarray(loaded.p_signal, dtype=np.float64).T[order]
    if values.shape != (12, 5000) or not np.isfinite(values).all():
        raise ValueError("Waveform is not finite canonical12x5000")
    metadata = {"fs": loaded.fs, "samples": loaded.sig_len, "source_lead_order": names,
                "canonical_lead_order": list(LEADS), "source_units": list(loaded.units),
                "source_adc_gain": list(loaded.adc_gain), "source_baseline": list(loaded.baseline),
                "canonical_units": [loaded.units[index] for index in order],
                "canonical_adc_gain": [loaded.adc_gain[index] for index in order],
                "canonical_baseline": [loaded.baseline[index] for index in order]}
    return values, metadata


def evaluate_record(record: Path, scratch: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    """Qualify all leads; compute AP only for prospectively assigned measurement records.

    Parameters
    ----------
    record : Path
        Authorized public numeric WFDB record.
    scratch : Path
        Fresh sanitized-header directory.

    Returns
    -------
    tuple
        Complete record metadata, physical signals and optional measurement scores.
    """
    signal, metadata = read_signal(record, scratch)
    role = "parser_qualification" if int(record.name) % 5 == 0 else "measurement"
    rows, proxies = [], []
    for lead, values in zip(LEADS, signal, strict=True):
        annotation = wfdb.rdann(str(record), lead)
        channels = np.asarray(annotation.chan)
        if channels.shape != annotation.sample.shape or not np.isin(channels, np.arange(12)).all():
            raise ValueError("Annotation channel metadata is invalid")
        waves = parse_waves(annotation.sample, annotation.symbol)
        row = coverage_metrics(waves)
        if role == "measurement":
            row, proxy = lead_metrics(values, waves)
            proxies.append(proxy)
        rows.append({"lead": lead, **row, "waves": waves})
    result = {"record_id": int(record.name), "role": role, "metadata": metadata, "leads": rows}
    if role == "measurement":
        controls = np.asarray([row["shift_AP"] for row in rows]).mean(axis=0)
        result.update({"AP": float(np.mean([row["AP"] for row in rows])),
                       "shift_AP": controls.tolist(), "mean_shift_AP": float(controls.mean()),
                       "prevalence": float(np.mean([row["prevalence"] for row in rows]))})
        result["gain"] = result["AP"] - result["mean_shift_AP"]
    return result, signal, np.asarray(proxies) if proxies else None


def aggregate(rows: list[dict[str, Any]], draws: int = 2000, seed: int = 2026106101) -> dict[str, Any]:
    """Aggregate fixed measurement records equally with paired whole-subject bootstrap draws.

    Parameters
    ----------
    rows : list[dict]
        All200 records with prospectively assigned roles.
    draws : int
        Frozen count of whole-subject draws.
    seed : int
        Frozen bootstrap seed.

    Returns
    -------
    dict
        All lead means, six controls, paired interval and every bootstrap gain.
    """
    if sorted(row["record_id"] for row in rows) != list(range(1, 201)):
        raise ValueError("Require exactly all200 prospectively assigned record IDs")
    selected = [row for row in rows if row["role"] == "measurement"]
    if any(row["role"] != ("parser_qualification" if row["record_id"] % 5 == 0 else "measurement")
           for row in rows):
        raise ValueError("Prospective record roles changed")
    if len(selected) != 160 or any(len(row["leads"]) != 12 for row in rows):
        raise ValueError("Require160 measurement subjects and12leads for every record")
    ids = np.asarray([row["record_id"] for row in selected])
    gains = np.asarray([row["gain"] for row in selected])
    groups, rng = patient_groups(ids), np.random.default_rng(seed)
    boot = np.asarray([gains[patient_resample(groups, rng)].mean() for _ in range(draws)])
    lower, upper = np.percentile(boot, [2.5, 97.5])
    point = float(gains.mean())
    return {"bootstrap_gains": boot.tolist(), "measurement_records": 160, "parser_qualification_records": 40,
            "measurement_leads": 1920,
            "constant_score_leads": sum(row["constant_score"] for record in selected
                                         for row in record["leads"]),
            "AP": float(np.mean([row["AP"] for row in selected])),
            "mean_shift_AP": float(np.mean([row["mean_shift_AP"] for row in selected])),
            "shift_AP": np.asarray([row["shift_AP"] for row in selected]).mean(axis=0).tolist(),
            "prevalence": float(np.mean([row["prevalence"] for row in selected])),
            "gain": point, "gain_interval": [float(lower), float(upper)],
            "bootstrap_draws": draws, "bootstrap_seed": seed, "skipped_draws": 0,
            "scientific_passed": bool(lower > 0 and point >= .05),
            "per_lead": {lead: {key: float(np.mean([row["leads"][index][key] for row in selected]))
                                for key in ["AP", "mean_shift_AP", "gain", "prevalence"]}
                         for index, lead in enumerate(LEADS)}}
