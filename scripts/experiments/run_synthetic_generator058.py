"""Run the prospectively frozen synthetic generator qualification on two CPU threads."""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.incart_localization056 import METHODS, score_signal
from ecg_experiment.paths import to_stored
from ecg_experiment.synthetic_generator058 import FS, KINDS, LEADS, SEED, generate_subject, intervene
from ecg_experiment.synthetic_localization058 import (
    grouped_summary,
    localization_metrics,
    reproduce_predecessors,
)

OUTPUT = ROOT / "outputs/experiment058_synthetic_generator_v1"
PROTOCOL = ROOT / "docs/experiment-058-synthetic-generator-localization.md"
REFERENCE = ROOT / "outputs/experiment056_incart_localization_v1/normal_reference.pkl"
ARCHIVE = ROOT / "third_party/ecgsyn058/ecgsyn.tar.gz"
LOG = logging.getLogger("experiment058")


def _limb_error(signal: np.ndarray) -> float:
    """Check the four exact limb algebra constraints.

    Parameters
    ----------
    signal : np.ndarray
        Twelve physical lead rows in canonical order.

    Returns
    -------
    float
        Largest absolute algebra error in mV.
    """
    first, second = signal[:2]
    differences = (
        signal[2] - (second - first),
        signal[3] + (first + second) / 2,
        signal[4] - (first - second / 2),
        signal[5] - (second - first / 2),
    )
    return float(np.max(np.abs(differences)))


def _qualify(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate paired subject caches and complete invariants before any method score.

    Parameters
    ----------
    directory : Path
        Fresh partial output directory.

    Returns
    -------
    tuple
        Full prospective manifest and generator qualification fields.
    """
    began = time.perf_counter()
    manifest = []
    max_limb = 0.0
    max_sham = 0.0
    uninformative = {}
    for family in ("ecgsyn", "compact"):
        cache = directory / "subjects" / family
        cache.mkdir(parents=True)
        for subject in range(600):
            cohort = "calibration" if subject < 200 else "iid" if subject < 400 else "shifted"
            generated = generate_subject(subject, cohort, family)
            sham = intervene(generated, "sham")
            max_sham = max(max_sham, float(np.max(np.abs(sham["signal"] - generated["signal"]))))
            max_limb = max(max_limb, _limb_error(generated["signal"]))
            for kind in KINDS:
                changed = intervene(generated, kind)
                expected = np.abs(changed["noiseless"] - generated["clean"]) >= 0.01
                expected &= (expected.sum(axis=1) >= 10)[:, None]
                if not np.array_equal(expected, changed["mask"]):
                    raise ValueError("Observable mask differs from actual paired difference")
                max_limb = max(max_limb, _limb_error(changed["signal"]))
                if not changed["informative"] and kind in ("qrs", "st", "t", "persistent_st", "persistent_t"):
                    key = f"{family}/{cohort}/{kind}"
                    uninformative[key] = uninformative.get(key, 0) + 1
            fields = {key: value for key, value in generated.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(cache / f"{subject}.npz", **fields, scale=generated["scale"])
            manifest.append({**generated["parameters"], "family": family})
            if subject % 100 == 0:
                LOG.info("generator qualification %s %s/600", family, subject)
    if max_sham != 0 or max_limb >= 1e-10:
        raise ValueError("Paired generator qualification failed")
    return manifest, {
        "max_processed_sham_difference": max_sham,
        "max_limb_error": max_limb,
        "all_masks_match_noiseless_counterfactual": True,
        "uninformative": uninformative,
        "elapsed_seconds": time.perf_counter() - began,
        "subjects_per_generator": 600,
    }


def _load_subject(directory: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Load one qualified cache without rerendering or changing paired noise.

    Parameters
    ----------
    directory : Path
        Partial output directory.
    row : dict
        Prospective subject manifest row.

    Returns
    -------
    dict
        Original generator inputs and arrays needed for deterministic intervention.
    """
    with np.load(directory / "subjects" / row["family"] / f"{row['subject']}.npz") as saved:
        values = {key: saved[key] for key in saved.files}
    values["scale"] = float(values["scale"])
    return {**values, "parameters": row, "family": row["family"]}


def _score(signal: np.ndarray, references: dict) -> tuple[dict, str | None]:
    """Run a frozen localizer receiving only observed voltage and frozen normal references.

    Parameters
    ----------
    signal : np.ndarray
        Observed noisy ECG; no clean counterpart, mask or latent timing is passed.
    references : dict
        Frozen PTB normal U_B reference.

    Returns
    -------
    tuple
        Both equal-support maps, or explicit unscorable reason.
    """
    try:
        _beats, maps = score_signal(signal, references)
    except ValueError as error:
        if str(error) != "Fewer than three complete beats":
            raise
        return dict.fromkeys(METHODS), str(error)
    return maps, None


def _thresholds(directory: Path, manifest: list[dict], references: dict) -> tuple[dict, pd.DataFrame]:
    """Fit only normal record-max thresholds before evaluating any test intervention.

    Parameters
    ----------
    directory : Path
        Qualified cache directory.
    manifest : list of dict
        All prospective subject identities.
    references : dict
        Frozen normal reference.

    Returns
    -------
    tuple
        Generator/method thresholds and all calibration rows, including failures.
    """
    rows = []
    for row in manifest:
        if row["cohort"] != "calibration":
            continue
        subject = _load_subject(directory, row)
        maps, failure = _score(subject["signal"], references)
        for method, unit_map in maps.items():
            rows.append(
                {
                    "subject": row["subject"],
                    "family": row["family"],
                    "map": method,
                    "score": float(unit_map.scores.max()) if unit_map is not None else 0,
                    "inference_failure": failure is not None,
                }
            )
    table = pd.DataFrame(rows)
    thresholds = {
        f"{family}/{method}": float(values.score.quantile(0.95))
        for (family, method), values in table.groupby(["family", "map"])
    }
    return thresholds, table


def _plot(subject: dict, changed: dict, maps: dict, path: Path) -> None:
    """Show both fixed-area marks with independently observed change support.

    Parameters
    ----------
    subject : dict
        Original clean and observed traces.
    changed : dict
        Paired intervention and evaluation-only observable mask.
    maps : dict
        Both frozen inference outputs, already computed without targets.
    path : Path
        Local figure destination.
    """
    fig, axes = plt.subplots(12, 2, figsize=(17, 17), sharex=True)
    times = np.arange(subject["signal"].shape[1]) / FS
    for column, method in enumerate(METHODS):
        metrics = localization_metrics(maps[method], changed["mask"], changed["energy"])
        for lead in range(12):
            axis = axes[lead, column]
            axis.plot(times, changed["signal"][lead], lw=0.6, color="black")
            axis.plot(times, subject["clean"][lead], lw=0.5, color="gray", alpha=0.5)
            indices = np.flatnonzero(changed["mask"][lead])
            for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if len(segment):
                    axis.axvspan(segment[0] / FS, (segment[-1] + 1) / FS, color="green", alpha=0.15)
            if lead == metrics["display_lead"]:
                axis.axvspan(
                    metrics["display_start"], metrics["display_start"] + 0.14, color="red", alpha=0.3
                )
            if column == 0:
                axis.set_ylabel(LEADS[lead])
        axes[0, column].set_title(f"{method}: expected exact center+lead hit {metrics['hit']:.3f}")
    fig.suptitle(
        f"{subject['family']} subject {subject['parameters']['subject']} {changed['kind']}: "
        "green observable change; red 140 ms mark"
    )
    axes[-1, 0].set_xlabel("original seconds")
    axes[-1, 1].set_xlabel("original seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def _test_cases(directory: Path, manifest: list[dict], references: dict, thresholds: dict) -> pd.DataFrame:
    """Evaluate every held-out trace once, retaining failures and deterministic local figures.

    Parameters
    ----------
    directory : Path
        Partial outputs.
    manifest : list of dict
        Prospectively selected subjects.
    references : dict
        Frozen PTB reference.
    thresholds : dict
        Already frozen calibration-only record-max cutoffs.

    Returns
    -------
    pd.DataFrame
        Complete per-case and per-method metrics.
    """
    rows = []
    chosen = set()
    for row in manifest:
        if row["cohort"] == "calibration":
            continue
        subject = _load_subject(directory, row)
        output = directory / "cases" / row["family"] / str(row["subject"])
        output.mkdir(parents=True)
        for kind in KINDS:
            changed = intervene(subject, kind)
            maps, failure = _score(changed["signal"], references)
            arrays = {
                "signal": changed["signal"],
                "noiseless": changed["noiseless"],
                "difference": changed["difference"],
                "energy": changed["energy"],
                "mask": changed["mask"],
            }
            readings = {}
            for method, unit_map in maps.items():
                metrics = localization_metrics(unit_map, changed["mask"], changed["energy"])
                rows.append(
                    {
                        "subject": row["subject"],
                        "family": row["family"],
                        "cohort": row["cohort"],
                        "kind": kind,
                        "map": method,
                        "informative": changed["informative"],
                        "inference_failure": failure is not None,
                        "failure_reason": failure,
                        "target_area": int(changed["mask"].sum()),
                        "target_occupancy": float(changed["mask"].mean()),
                        "target_energy": float(changed["energy"].sum()),
                        "flag": int(metrics["score"] > thresholds[f"{row['family']}/{method}"]),
                        **metrics,
                    }
                )
                readings[method] = metrics
                if unit_map is not None:
                    for field in ("scores", "leads", "starts", "ends"):
                        arrays[f"{method}_{field}"] = getattr(unit_map, field)
            np.savez_compressed(output / f"{kind}.npz", **arrays)
            difference = readings[METHODS[0]]["hit"] - readings[METHODS[1]]["hit"]
            category = "win" if difference > 0 else "loss" if difference < 0 else "tie"
            key = (row["family"], row["cohort"], kind, category)
            if kind in ("st", "qrs", "t", "persistent_st", "persistent_t") and key not in chosen:
                _plot(
                    subject,
                    changed,
                    maps,
                    directory
                    / "figures"
                    / f"{row['family']}_{row['cohort']}_{kind}_{category}_{row['subject']}.png",
                )
                chosen.add(key)
        if row["subject"] % 50 == 0:
            LOG.info("test scoring %s subject %s", row["family"], row["subject"])
    return pd.DataFrame(rows)


def main() -> None:
    """Qualify generators, verify predecessors, fit normal cutoffs, then score locked tests."""
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    if OUTPUT.exists() or partial.exists():
        raise FileExistsError(OUTPUT)
    partial.mkdir()
    (partial / "figures").mkdir()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(partial / "run.log"), logging.StreamHandler()]
    )
    began = time.perf_counter()
    with threadpool_limits(limits=2):
        integrity = reproduce_predecessors()
        sources = [
            PROTOCOL,
            ROOT / "ecg_experiment/synthetic_generator058.py",
            ROOT / "ecg_experiment/synthetic_localization058.py",
            ROOT / "scripts/experiments/run_synthetic_generator058.py",
            ROOT / "ecg_experiment/incart_localization056.py",
            ROOT / "ecg_experiment/raw_residual054.py",
            ROOT / "ecg_experiment/raw_residual052.py",
            ROOT / "ecg_experiment/st_episode057.py",
            ROOT / "ecg_experiment/intervals.py",
            ROOT / "ecg_experiment/paths.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
        identity = {
            "sources": {to_stored(path): sha256_file(path) for path in sources},
            "inputs": {to_stored(path): sha256_file(path) for path in (ARCHIVE, REFERENCE)},
            "seed": SEED,
            "cpu_threads": 2,
            "generator_samples": 6000,
            "sample_rate": FS,
            "source_url": "https://physionet.org/content/ecgsyn/1.0.0/",
            "classifier_changed": False,
        }
        if (
            identity["inputs"][to_stored(REFERENCE)]
            != json.loads((REFERENCE.parent / "result.json").read_text())["outputs_sha256"][REFERENCE.name]
        ):
            raise ValueError("Frozen normal reference changed")
        with REFERENCE.open("rb") as source:
            references = pickle.load(source)
        manifest, qualification = _qualify(partial)
        write_json_atomic(partial / "subject_manifest.json", manifest)
        write_json_atomic(partial / "qualification.json", qualification)
        thresholds, calibration = _thresholds(partial, manifest, references)
        calibration.to_csv(partial / "calibration.csv", index=False)
        write_json_atomic(partial / "thresholds.json", thresholds)
        LOG.info("calibration cutoffs frozen; scoring held-out cases")
        table = _test_cases(partial, manifest, references, thresholds)
        table.to_csv(partial / "locations.csv", index=False)
        summary = grouped_summary(table)
        result = {
            "identity": identity,
            "integrity": integrity,
            "qualification": qualification,
            "thresholds": thresholds,
            "benchmark": summary,
            "clinical_promotion": False,
            "provisional_synthetic_robustness_pass": bool(
                all(row["provisional_transient_robustness_pass"] for row in summary.values())
            ),
            "elapsed_seconds": time.perf_counter() - began,
        }
        result["outputs_sha256"] = {
            path.relative_to(partial).as_posix(): sha256_file(path)
            for path in sorted(partial.rglob("*"))
            if path.is_file() and path.name != "run.log"
        }
        write_json_atomic(partial / "result.json", result)
    partial.rename(OUTPUT)
    LOG.info(
        "complete %.1fs; provisional robustness=%s, clinical promotion=false",
        result["elapsed_seconds"],
        result["provisional_synthetic_robustness_pass"],
    )


if __name__ == "__main__":
    main()
