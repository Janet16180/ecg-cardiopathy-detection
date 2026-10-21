"""Qualify explicit public QRS-marker timing without inferring waveform boundaries or negative labels."""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from ecg_experiment import ssl_synthetic_ludb_qualification as original
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_ludb_raw_events import decode

LEADS, SHIFTS = original.LEADS, original.SHIFTS


def validate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Admit ordered explicit markers while preserving every missing-boundary flag."""
    samples = np.asarray([event["sample"] for event in events])
    if (not len(events) or samples.dtype.kind not in "iu" or samples.min() < 0 or samples.max() >= 5000
            or np.any(np.diff(samples) < 0)):
        raise ValueError("Require nonempty ordered integer events within5000samples")
    if any(not 0 <= event["channel"] <= 255 for event in events):
        raise ValueError("WFDB channel metadata outside0..255")
    positions = [event["sample"] for event in events if event["type_code"] == 1]
    start, end = int(samples[0]), int(samples[-1]) + 1
    shifts = [int(np.floor((end - start) * fraction)) for fraction in SHIFTS]
    if (not positions or end - start < 6 or len(set(shifts)) != 6
            or not all(0 < shift < end - start for shift in shifts)):
        raise ValueError("Each lead needs explicit N markers and six distinct nonzero reference shifts")
    return {"start": start, "end_exclusive": end, "span_samples": end - start,
            "span_fraction": (end - start) / 5000, "N_samples": positions, "events": events,
            "event_count": len(events),
            "symbol_counts": dict(Counter(str(event["symbol"]) for event in events)),
            "type_code_counts": dict(Counter(str(event["type_code"]) for event in events)),
            "duplicate_sample_code_counts": [{"sample": key[0], "type_code": key[1], "count": count}
                for key, count in Counter((event["sample"], event["type_code"]) for event in events).items()
                if count > 1],
            "duplicate_N_timestamp_excess_count": len(positions) - len(set(positions)),
            "N_without_immediately_preceding_onset_indices": [index for index, event in enumerate(events)
                if event["type_code"] == 1 and (index == 0 or events[index - 1]["symbol"] != "(")]}


def midranks(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return strict-less plus half-equal reference fractions; include the landmark sample itself."""
    sorted_values = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(sorted_values) or not np.isfinite(sorted_values).all() or not np.isfinite(values).all():
        raise ValueError("Finite nonempty reference and landmark values are required")
    lower = np.searchsorted(sorted_values, values, side="left")
    upper = np.searchsorted(sorted_values, values, side="right")
    return (lower + upper) / (2 * len(sorted_values))


def score_lead(values: np.ndarray, structural: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Rank only explicit N markers against the same lead's all-sample reference span."""
    proxy = original.activity_proxy(values)
    start, end = structural["start"], structural["end_exclusive"]
    reference = proxy[start:end]
    positions = np.asarray(structural["N_samples"]) - start
    shifts = [int(np.floor(len(reference) * fraction)) for fraction in SHIFTS]
    ranks = midranks(reference, reference)
    native = ranks[positions]
    controls = [np.roll(ranks, shift)[positions] for shift in shifts]
    means = [float(rank.mean()) for rank in controls]
    result = {"N_midrank_values": native.tolist(), "N_mean_midrank": float(native.mean()),
              "shift_N_midrank_values": [rank.tolist() for rank in controls],
              "shift_N_mean_midrank": means, "shift_samples": shifts,
              "mean_shift_N_midrank": float(np.mean(means)), "constant_score": bool(np.ptp(reference) == 0)}
    result["gain"] = result["N_mean_midrank"] - result["mean_shift_N_midrank"]
    return result, proxy


def evaluate_record(record: Path, scratch: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    """Read only structural headers and original annotation events beside sanitized local signal metadata."""
    values, metadata = original.read_signal(record, scratch)
    role = "parser_qualification" if int(record.name) % 5 == 0 else "measurement"
    rows, scores = [], []
    for lead, signal in zip(LEADS, values, strict=True):
        (scratch / (record.name + "." + lead)).symlink_to(record.with_suffix("." + lead))
        ann = wfdb.rdann(str(scratch / record.name), lead, return_label_elements=["symbol", "label_store"])
        if ann.sample.dtype.kind not in "iu":
            raise ValueError("WFDB timed event samples must be integers")
        events = [{"sample": int(sample),
                   "symbol": ann.symbol[index] if isinstance(ann.symbol[index], str) else None,
                   "symbol_lookup_status": "mapped" if isinstance(ann.symbol[index], str) else "unmapped",
                   "type_code": int(ann.label_store[index]), "subtype": int(ann.subtype[index]),
                   "channel": int(ann.chan[index]), "num": int(ann.num[index]),
                   "aux_length": len(ann.aux_note[index] or "")} for index, sample in enumerate(ann.sample)]
        raw = decode(record.with_suffix("." + lead).read_bytes())
        fields = ["sample", "type_code", "subtype", "channel", "num", "aux_length"]
        if (raw["errors"] or len(raw["events"]) != len(events)
                or any(any(first[key] != second[key] for key in fields)
                       or (not first["symbol"].startswith("unknown_code_")
                           and first["symbol"] != second["symbol"])
                       for first, second in zip(raw["events"], events, strict=True))):
            raise ValueError("Independent raw/WFDB timed event agreement failed")
        row = {"lead": lead, **validate_events(events),
               "independent_event_decode": {"all_numeric_fields_exact": True,
                   "known_symbols_exact": True, "raw_bytes": raw["bytes"], "raw_sha256": raw["sha256"],
                   "metadata_counts": raw["metadata_counts"]}}
        if role == "measurement":
            metrics, score = score_lead(signal, row)
            row.update(metrics)
            scores.append(score)
        rows.append(row)
    result = {"record_id": int(record.name), "role": role, "metadata": metadata, "leads": rows}
    if role == "measurement":
        controls = np.asarray([row["shift_N_mean_midrank"] for row in rows]).mean(axis=0)
        result.update(N_mean_midrank=float(np.mean([row["N_mean_midrank"] for row in rows])),
                      shift_N_mean_midrank=controls.tolist(), mean_shift_N_midrank=float(controls.mean()))
        result["gain"] = result["N_mean_midrank"] - result["mean_shift_N_midrank"]
    return result, values, np.asarray(scores) if scores else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve fixed record roles and use paired whole-subject draws for the engineering decision."""
    if sorted(row["record_id"] for row in rows) != list(range(1, 201)) or any(
            row["role"] != ("parser_qualification" if row["record_id"] % 5 == 0 else "measurement")
            or len(row["leads"]) != 12 for row in rows):
        raise ValueError("Require all200 prospectively assigned records and twelve leads")
    selected = sorted([row for row in rows if row["role"] == "measurement"], key=lambda row: row["record_id"])
    gains = np.asarray([row["gain"] for row in selected])
    groups, rng = patient_groups(np.arange(160)), np.random.default_rng(2026106101)
    draws = np.asarray([gains[patient_resample(groups, rng)].mean() for _ in range(2000)])
    low, high = np.percentile(draws, [2.5, 97.5], method="linear")
    return {"measurement_records": 160, "parser_qualification_records": 40, "measurement_leads": 1920,
            "N_mean_midrank": float(np.mean([row["N_mean_midrank"] for row in selected])),
            "shift_N_mean_midrank": (
                np.asarray([row["shift_N_mean_midrank"] for row in selected]).mean(0).tolist()),
            "mean_shift_N_midrank": float(np.mean([row["mean_shift_N_midrank"] for row in selected])),
            "gain": float(gains.mean()), "gain_interval": [float(low), float(high)],
            "bootstrap_draws": 2000, "bootstrap_seed": 2026106101, "skipped_draws": 0,
            "bootstrap_gains": draws.tolist(), "scientific_passed": bool(gains.mean() >= .05 and low > 0),
            "constant_score_leads": sum(lead["constant_score"] for row in selected for lead in row["leads"]),
            "reference_span_summary": {
                "lead_streams": 2400,
                "span_samples_quantiles": np.percentile([lead["span_samples"] for row in rows
                    for lead in row["leads"]], [0, 25, 50, 75, 100]).tolist(),
                "N_count_quantiles": np.percentile([len(lead["N_samples"]) for row in rows
                    for lead in row["leads"]], [0, 25, 50, 75, 100]).tolist()},
            "per_lead_distributions": {lead: {key: np.percentile([row["leads"][index][key]
                for row in selected], [0, 25, 50, 75, 100]).tolist()
                for key in ["N_mean_midrank", "mean_shift_N_midrank", "gain", "span_samples"]}
                for index, lead in enumerate(LEADS)},
            "per_lead": {lead: {key: float(np.mean([row["leads"][index][key] for row in selected]))
                                for key in ["N_mean_midrank", "mean_shift_N_midrank", "gain"]}
                         for index, lead in enumerate(LEADS)}}
