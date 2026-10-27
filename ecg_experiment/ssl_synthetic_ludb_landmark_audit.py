"""Independent empirical midrank and whole-subject landmark arithmetic."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample

STANDARD_SYMBOLS = dict(
    zip(
        (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
        ),
        (
            " ",
            "N",
            "L",
            "R",
            "a",
            "V",
            "F",
            "J",
            "A",
            "S",
            "E",
            "j",
            "/",
            "Q",
            "~",
            "|",
            "s",
            "T",
            "*",
            "D",
            '"',
            "=",
            "p",
            "B",
            "^",
            "t",
            "+",
            "u",
            "?",
            "!",
            "[",
            "]",
            "e",
            "n",
            "@",
            "x",
            "f",
            "(",
            ")",
            "r",
        ),
        strict=True,
    )
)
FRACTIONS = (0.17, 0.31, 0.43, 0.59, 0.71, 0.83)


def midranks(values: np.ndarray) -> np.ndarray:
    """Count strict-lower and tied reference samples independently without interpolated ranks."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite nonempty one-dimensional reference required")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    twice_midcounts = 2 * np.cumsum(counts) - counts
    return (twice_midcounts / (2 * len(scores)))[inverse]


def metrics(
    full_score: np.ndarray, start: int, stop: int, peaks: list[int]
) -> tuple[dict[str, Any], np.ndarray]:
    """Reconstruct N-point timing ranks and all six controls in the fixed unlabeled reference span."""
    score = np.asarray(full_score, dtype=np.float64)
    if score.shape != (5000,) or not 0 <= start < stop <= 5000:
        raise ValueError("Full score and inclusive-event reference geometry differ")
    points = np.asarray(peaks, dtype=np.int64) - start
    if not len(points) or np.any(points < 0) or np.any(points >= stop - start):
        raise ValueError("Original N landmarks must be nonempty and inside reference")
    ranks = midranks(score[start:stop])
    offsets = [int(len(ranks) * fraction) for fraction in FRACTIONS]
    if len(set(offsets)) != 6 or not all(0 < offset < len(ranks) for offset in offsets):
        raise ValueError("Invalid fixed reference-span offsets")
    observed = float(ranks[points].mean())
    shifted = [float(np.roll(ranks, offset)[points].mean()) for offset in offsets]
    control = float(np.mean(shifted))
    return {
        "observed": observed,
        "controls": shifted,
        "mean_control": control,
        "gain": observed - control,
        "shift_samples": offsets,
        "constant_score": bool(np.ptp(score[start:stop]) == 0),
    }, ranks


def bootstrap(rows: list[dict[str, Any]], key: str = "gain") -> tuple[np.ndarray, list[float]]:
    """Use equal subject gains and exactly two thousand common whole-subject draws."""
    if [row["record_id"] for row in rows] != [number for number in range(1, 201) if number % 5]:
        raise ValueError("Require all160 sorted measurement subjects without exclusions")
    groups = patient_groups(np.asarray([row["record_id"] for row in rows]))
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Finite subject means required")
    rng = np.random.default_rng(2026106101)
    draws = np.asarray([values[patient_resample(groups, rng)].mean() for _ in range(2000)])
    return draws, np.percentile(draws, [2.5, 97.5], method="linear").tolist()


def annotation_row(payload: bytes) -> dict[str, Any]:
    """Retain every original event while constructing only the unlabeled span and N point list."""
    from collections import Counter

    from ecg_experiment.ssl_synthetic_ludb_raw_events import decode

    raw = decode(payload)
    events = [
        {
            key: event[key]
            for key in ("sample", "symbol", "type_code", "subtype", "channel", "num", "aux_length")
        }
        for event in raw["events"]
    ]
    for event in events:
        event["symbol"] = STANDARD_SYMBOLS.get(event["type_code"])
        event["symbol_lookup_status"] = "mapped" if event["symbol"] is not None else "unmapped"
    points = [event["sample"] for event in events]
    if raw["errors"] or not raw["EOF"] or not points or points != sorted(points):
        raise ValueError("Original complete nondecreasing event stream required")
    if min(points) < 0 or max(points) >= 5000:
        raise ValueError("Original timed events inside5000 samples required")
    peaks = [event["sample"] for event in events if event["type_code"] == 1]
    if not peaks:
        raise ValueError("Every lead requires at least one original N landmark")
    start, stop = points[0], points[-1] + 1
    offsets = [int((stop - start) * fraction) for fraction in FRACTIONS]
    if len(set(offsets)) != 6 or not all(0 < shift < stop - start for shift in offsets):
        raise ValueError("Every reference span requires six distinct valid shifts")
    return {
        "start": start,
        "end_exclusive": stop,
        "span_samples": stop - start,
        "span_fraction": (stop - start) / 5000,
        "N_samples": peaks,
        "events": events,
        "symbol_counts": dict(Counter(str(event["symbol"]) for event in events)),
        "type_code_counts": dict(Counter(str(event["type_code"]) for event in events)),
        "duplicate_sample_code_counts": [
            {"sample": key[0], "type_code": key[1], "count": count}
            for key, count in Counter((event["sample"], event["type_code"]) for event in events).items()
            if count > 1
        ],
        "duplicate_N_timestamp_excess_count": len(peaks) - len(set(peaks)),
        "independent_event_decode": {
            "all_numeric_fields_exact": True,
            "known_symbols_exact": True,
            "raw_bytes": raw["bytes"],
            "raw_sha256": raw["sha256"],
            "metadata_counts": raw["metadata_counts"],
        },
        "event_count": len(events),
        "N_without_immediately_preceding_onset_indices": [
            index
            for index, event in enumerate(events)
            if event["type_code"] == 1 and (index == 0 or events[index - 1]["symbol"] != "(")
        ],
    }


def lead_metrics(full_score: np.ndarray, structural: dict[str, Any]) -> dict[str, Any]:
    """Expose all original and shifted N-point rank values in the prospectively fixed output schema."""
    result, ranks = metrics(
        full_score, structural["start"], structural["end_exclusive"], structural["N_samples"]
    )
    points = np.asarray(structural["N_samples"]) - structural["start"]
    return {
        "N_midrank_values": ranks[points].tolist(),
        "N_mean_midrank": result["observed"],
        "shift_N_midrank_values": [
            np.roll(ranks, shift)[points].tolist() for shift in result["shift_samples"]
        ],
        "shift_N_mean_midrank": result["controls"],
        "shift_samples": result["shift_samples"],
        "mean_shift_N_midrank": result["mean_control"],
        "gain": result["gain"],
        "constant_score": result["constant_score"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently reduce equal N/lead/subject values and the fixed paired engineering comparison."""
    if [row["record_id"] for row in rows] != list(range(1, 201)):
        raise ValueError("Require all200 original records")
    selected = [row for row in rows if row["record_id"] % 5]
    draws, interval = bootstrap(selected)
    gain = float(np.mean([row["gain"] for row in selected]))
    return {
        "measurement_records": 160,
        "parser_qualification_records": 40,
        "measurement_leads": 1920,
        "N_mean_midrank": float(np.mean([row["N_mean_midrank"] for row in selected])),
        "shift_N_mean_midrank": np.asarray([row["shift_N_mean_midrank"] for row in selected])
        .mean(0)
        .tolist(),
        "mean_shift_N_midrank": float(np.mean([row["mean_shift_N_midrank"] for row in selected])),
        "gain": gain,
        "gain_interval": interval,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 2026106101,
        "skipped_draws": 0,
        "bootstrap_gains": draws.tolist(),
        "scientific_passed": bool(gain >= 0.05 and interval[0] > 0),
        "constant_score_leads": sum(lead["constant_score"] for row in selected for lead in row["leads"]),
        "reference_span_summary": {
            "lead_streams": 2400,
            "span_samples_quantiles": np.percentile(
                [lead["span_samples"] for row in rows for lead in row["leads"]], [0, 25, 50, 75, 100]
            ).tolist(),
            "N_count_quantiles": np.percentile(
                [len(lead["N_samples"]) for row in rows for lead in row["leads"]], [0, 25, 50, 75, 100]
            ).tolist(),
        },
        "per_lead_distributions": {
            lead: {
                key: np.percentile(
                    [row["leads"][index][key] for row in selected], [0, 25, 50, 75, 100]
                ).tolist()
                for key in ("N_mean_midrank", "mean_shift_N_midrank", "gain", "span_samples")
            }
            for index, lead in enumerate(
                ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
            )
        },
        "per_lead": {
            lead: {
                key: float(np.mean([row["leads"][index][key] for row in selected]))
                for key in ("N_mean_midrank", "mean_shift_N_midrank", "gain")
            }
            for index, lead in enumerate(
                ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
            )
        },
    }
