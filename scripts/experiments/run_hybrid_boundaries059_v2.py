"""Execute prospectively frozen QTDB joint-channel wave-boundary comparison."""

from __future__ import annotations

import time

import numpy as np
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.hybrid_boundaries059 import (
    aggregate,
    evaluate_joint,
    read_qtdb,
    reproduce_055,
    summarize,
)
from ecg_experiment.hybrid_boundaries059_v2 import joint_hybrid
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head

DATA = ROOT / "data/raw/qtdb/1.0.0"
OUTPUT = ROOT / "outputs/experiment059_qtdb_hybrid_v2"
PROTOCOL = "docs/experiment-059-hybrid-wave-boundaries.md"
SOURCES = (
    "ecg_experiment/hybrid_boundaries059.py",
    "ecg_experiment/hybrid_boundaries059_v2.py",
    "scripts/experiments/run_hybrid_boundaries059_v2.py",
    "docs/experiment-059-v2-execution-note.md",
    "scripts/experiments/run_hybrid_boundaries059.py",
    "ecg_experiment/morphology_boundaries.py",
    "ecg_experiment/ludb_boundary_validation.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/calibrated_localization.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "pyproject.toml",
    "uv.lock",
    PROTOCOL,
)
FAMILIES = {
    "mit_arrhythmia": "100 102 103 104 114 116 117 123 213 221 223 230 231 232 233",
    "mit_st": "301 302 306 307 308 310",
    "mit_supraventricular": "803 808 811 820 821 840 847 853 871 872 873 883 891",
    "mit_normal_sinus": "16265 16272 16273 16420 16483 16539 16773 16786 16795 17453",
    "mit_long_term": "14046 14157 14172 15814",
    "sudden_death": "30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 17152",
}


def source_family(record_id: str) -> str:
    """
    Identify original dataset without reading clinical diagnoses.

    Parameters
    ----------
    record_id : str
        Official source-prefixed QTDB name.

    Returns
    -------
    str
        Published original source family.
    """
    if record_id.startswith("sele"):
        return "european_st_t"
    found = [name for name, identifiers in FAMILIES.items() if record_id[3:] in identifiers.split()]
    if len(found) != 1:
        raise ValueError("Unknown or ambiguous QTDB source family")
    return found[0]


def fixed_figures() -> None:
    """
    Render prespecified examples without selecting favorable results.

    Returns
    -------
    None
        Local-only raw waveform figures with joint expert boundaries.
    """
    for record_id in ("sel100", "sel232", "sel16265", "sel30", "sele0106"):
        signal, truth, _ = read_qtdb(DATA, record_id)
        with np.load(OUTPUT / "records" / f"{record_id}_predictions.npz") as saved:
            peaks = saved["peaks"]
            predictions = {name: saved[name] for name in ("fixed", "hybrid")}
        center = int(truth["QRS"][0, 1])
        start, end = max(0, center - 250), min(signal.shape[1], center + 750)
        figure = Figure(figsize=(15, 7), layout="constrained")
        axes = figure.subplots(2, 3, sharex=True)
        for channel in range(2):
            for wave_index, name in enumerate(("P", "QRS", "T")):
                axis = axes[channel, wave_index]
                axis.plot(
                    np.arange(start, end) / 250, signal[channel, start:end], color="black", linewidth=0.7
                )
                for onset, _, offset in truth[name]:
                    if onset >= start and offset <= end:
                        axis.axvspan(onset / 250, offset / 250, color="green", alpha=0.2)
                for method, color, style in (("fixed", "blue", ":"), ("hybrid", "red", "--")):
                    for interval in predictions[method][(peaks >= start) & (peaks <= end), wave_index]:
                        for endpoint in interval:
                            if start <= endpoint <= end:
                                axis.axvline(endpoint / 250, color=color, linestyle=style, linewidth=0.8)
                axis.set_title(f"Channel {channel + 1}; joint {name} truth")
                axis.set_xlabel("seconds")
        figure.suptitle(f"{record_id}: joint expert green, fixed blue, hybrid red; unknown onset unshaded")
        figure.savefig(OUTPUT / "figures" / f"{record_id}.png", dpi=110)
        figure.clear()


def run() -> None:
    """
    Verify predecessor and data, then score every frozen evaluation record once.

    Returns
    -------
    None
        Immutable launch receipt, per-record artifacts and complete aggregate result.
    """
    if OUTPUT.exists():
        raise FileExistsError("Preserve existing059 output; use a new successor identity")
    OUTPUT.mkdir()
    started = time.monotonic()
    identity = {
        "source_commit": git_head(ROOT),
        "execution_predecessor": "43c5cff; v1 fixed-P recording-edge guard failed before QTDB scores",
        "protocol_commits": ["c4dd7d5", "ec7c041"],
        "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        "data_version": "QTDB1.0.0",
        "fitting_records": [],
        "bootstrap_seed": 59059,
        "bootstrap_draws": 2000,
        "data_source": "https://physionet.org/content/qtdb/1.0.0/",
    }
    write_json_atomic(OUTPUT / "launch.json", identity)
    try:
        integrity = reproduce_055(ROOT)
        write_json_atomic(OUTPUT / "predecessor_integrity.json", integrity)
        print("Predecessor055 exact and immutable hashes verified", flush=True)
        records = (DATA / "RECORDS").read_text().split()
        if len(records) != 105 or sum(name.startswith("sele") for name in records) != 33:
            raise ValueError("QTDB source populations differ")
        manifest = {
            line.split(maxsplit=1)[1].strip().lstrip("*"): line.split()[0]
            for line in (DATA / "SHA256SUMS.txt").read_text().splitlines()
        }
        inputs = {
            to_stored(DATA / "SHA256SUMS.txt"): sha256_file(DATA / "SHA256SUMS.txt"),
            to_stored(DATA / "RECORDS"): sha256_file(DATA / "RECORDS"),
        }
        for record_id in records:
            for extension in ("dat", "hea", "q1c"):
                name = f"{record_id}.{extension}"
                actual = sha256_file(DATA / name)
                if actual != manifest[name]:
                    raise ValueError(f"Published checksum mismatch: {name}")
                inputs[to_stored(DATA / name)] = actual
        audits, summaries, coverage = {}, {}, {}
        (OUTPUT / "records").mkdir()
        (OUTPUT / "figures").mkdir()
        for count, record_id in enumerate(records, 1):
            signal, truth, audit = read_qtdb(DATA, record_id)
            audit["source_family"] = source_family(record_id)
            peaks, predictions = joint_hybrid(signal)
            rows, diagnostic = evaluate_joint(peaks, predictions, truth)
            summaries[record_id] = summarize(rows)
            audits[record_id], coverage[record_id] = audit, diagnostic
            write_json_atomic(
                OUTPUT / "records" / f"{record_id}.json",
                {
                    "record_id": record_id,
                    "summary": summaries[record_id],
                    "measurements": rows,
                    "coverage": diagnostic,
                },
            )
            write_npz_atomic(OUTPUT / "records" / f"{record_id}_predictions.npz", peaks=peaks, **predictions)
            if count % 10 == 0:
                print(f"Scored {count}/105 frozen records", flush=True)
        fixed_figures()
        primary = {key: value for key, value in summaries.items() if not key.startswith("sele")}
        edb = {key: value for key, value in summaries.items() if key.startswith("sele")}
        result = {
            "experiment": 59,
            "status": "complete",
            "identity": identity,
            "predecessor_integrity": integrity,
            "input_hashes": inputs,
            "primary_non_edb": aggregate(primary),
            "all105_descriptive": aggregate(summaries),
            "edb33_descriptive": aggregate(edb),
            "source_family_descriptive": {},
            "format_audit": audits,
            "coverage": coverage,
            "seconds": time.monotonic() - started,
            "closed_test_access": False,
            "annotation_tuning": False,
            "synthetic_role": "unit invariants only; no synthetic result promoted",
            "p_identity_verified_every_record": True,
        }
        for family in (*FAMILIES, "european_st_t"):
            chosen = {
                key: value for key, value in summaries.items() if audits[key]["source_family"] == family
            }
            result["source_family_descriptive"][family] = aggregate(chosen)
        result["output_hashes"] = {
            to_stored(path): sha256_file(path)
            for path in OUTPUT.rglob("*")
            if path.is_file() and path.name != "run.log"
        }
        write_json_atomic(OUTPUT / "result.json", result)
        print("Complete059; frozen gates recorded", flush=True)
    except Exception as error:
        write_json_atomic(
            OUTPUT / "failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "source_commit": identity["source_commit"],
            },
        )
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
