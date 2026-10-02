"""Execute the frozen conditional European ST-T localization comparison."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import wfdb
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.raw_residual054 import primary_summary, synthetic_summary
from ecg_experiment.st_episode057 import (
    BEAT_SYMBOLS,
    FS,
    change_windows,
    episode_targets,
    location_summary,
    mean_interval,
    parse_episodes,
    subject_auroc,
    subject_id,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = ROOT / "data/raw/edb/1.0.0"
OUTPUT = ROOT / "outputs/experiment057_st_episode_v1"
PROTOCOL = "docs/experiment-057-st-episode-localization.md"
METHODS = ("st", "whole")
LOG = logging.getLogger("experiment057")


def _predecessor() -> dict[str, Any]:
    directory = ROOT / "outputs/experiment054_aligned_phase_residual_v1"
    prior = json.loads((directory / "result.json").read_text())
    verified = 0
    for group in ("sources", "waveforms"):
        for path, expected in prior["identity"][group].items():
            if sha256_file(ROOT / path) != expected:
                raise ValueError(f"054 pinned file changed: {path}")
            verified += 1
    for path, expected in prior["outputs_sha256"].items():
        if sha256_file(directory / path) != expected:
            raise ValueError(f"054 output changed: {path}")
    development = pd.read_csv(directory / "development_locations.csv", float_precision="round_trip")
    found = primary_summary(development, ("aligned_phase", "aligned_only", "U_B_fixed"), 54054)
    found["excluded"] = prior["development"]["excluded"]
    found["saturated_records"] = int(
        (development[development["map"] == "aligned_phase"]["saturated_windows"] > 0).sum()
    )
    if found != prior["development"]:
        raise ValueError("054 primary metrics did not reproduce exactly")
    synthetic = pd.read_csv(directory / "synthetic_locations.csv", float_precision="round_trip")
    controls = pd.read_csv(directory / "threshold_controls.csv", float_precision="round_trip")
    summaries = {}
    for method in ("aligned_phase", "aligned_only"):
        selected = synthetic[synthetic["map"] == method]
        summary = synthetic_summary(selected, 54054)
        summary["records"] = int(selected["ecg_id"].nunique())
        summary["excluded_fewer_than_five_complete_beats"] = []
        nuisance = selected[selected["kind"] == "nuisance"]
        threshold = float(np.quantile(controls[controls["map"] == method]["score"], 0.95))
        before = float((nuisance["control_score"] > threshold).mean())
        after = float((nuisance["score"] > threshold).mean())
        summary["nuisance"] = {
            "threshold": threshold,
            "unchanged_flag_share": before,
            "nuisance_flag_share": after,
            "increase": after - before,
            "score_change": bootstrap_mean(
                nuisance["patient_id"].to_numpy(),
                (nuisance["score"] - nuisance["control_score"]).to_numpy(),
                2000,
                54054,
            ),
        }
        summaries[method] = summary
    if summaries != prior["synthetic"]:
        raise ValueError("054 synthetic summaries did not reproduce exactly")
    return {
        "verified_sources_and_waveforms": verified,
        "054_primary_exact": True,
        "054_synthetic_exact": True,
        "receipt_sha256": sha256_file(directory / "result.json"),
    }


def _data_integrity() -> tuple[list[str], dict[str, str]]:
    records = (DATA / "RECORDS").read_text().split()
    if len(records) != 90 or len({subject_id(record) for record in records}) != 79:
        raise ValueError("European ST-T record/subject count changed")
    expected = {
        line.split()[1]: line.split()[0] for line in (DATA / "SHA256SUMS.txt").read_text().splitlines()
    }
    hashes = {}
    for record in records:
        for extension in (".hea", ".dat", ".atr"):
            path = DATA / (record + extension)
            digest = sha256_file(path)
            if digest != expected[path.name]:
                raise ValueError(f"Published checksum mismatch: {path.name}")
            hashes[to_stored(path)] = digest
    for name in ("RECORDS", "SHA256SUMS.txt"):
        hashes[to_stored(DATA / name)] = sha256_file(DATA / name)
    return records, hashes


def _coverage(scores: np.ndarray, centers: np.ndarray, episodes: list[dict[str, Any]]) -> float | None:
    selected = np.zeros(scores.size, dtype=bool)
    selected[np.argsort(-scores.ravel(), kind="stable")[: int(np.ceil(scores.size * 0.05))]] = True
    selected = selected.reshape(scores.shape)
    hits = []
    for episode in episodes:
        if episode["kind"] == "ST" and episode["end"] is not None:
            inside = (centers >= episode["start"]) & (centers <= episode["end"])
            if inside.any():
                hits.append(bool(np.any(selected[inside, episode["channel"]])))
    return float(np.mean(hits)) if hits else None


def _extremum_distance(
    scores: np.ndarray, centers: np.ndarray, episodes: list[dict[str, Any]]
) -> float | None:
    window, channel = np.unravel_index(np.argmax(scores), scores.shape)
    peaks = [
        peak["time"]
        for episode in episodes
        if episode["kind"] == "ST"
        and episode["channel"] == channel
        and episode["end"] is not None
        and np.any((centers >= episode["start"]) & (centers <= episode["end"]))
        for peak in episode["peaks"]
    ]
    return float(min(abs(centers[window] - peak) for peak in peaks)) if peaks else None


def _plot(
    record: str,
    features: dict[str, np.ndarray],
    episodes: list[dict[str, Any]],
    channels: list[str],
    directory: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    mask = features.get("evaluation_mask", np.ones(len(features["centers"]), dtype=bool))
    for channel, axis in enumerate(axes):
        for method in METHODS:
            axis.plot(features["centers"] / 60, features[method][:, channel], label=method, linewidth=1)
            index, top_channel = np.unravel_index(
                np.argmax(features[method][mask]), features[method][mask].shape
            )
            if top_channel == channel:
                axis.axvline(
                    features["centers"][mask][index] / 60,
                    linestyle="--",
                    linewidth=1,
                    label=f"{method} top mark",
                )
        if not mask.all():
            axis.axvspan(
                (features["centers"][~mask][0] - 15) / 60,
                120,
                color="gray",
                alpha=0.2,
                label="Unknown annotation tail",
            )
        for episode in episodes:
            if (
                episode["channel"] == channel
                and episode["kind"] in ("ST", "st")
                and episode["end"] is not None
            ):
                axis.axvspan(
                    episode["start"] / 60,
                    episode["end"] / 60,
                    color="green" if episode["kind"] == "ST" else "orange",
                    alpha=0.18,
                )
        axis.set_ylabel(f"{channels[channel]} change (mV)")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Original recording time (minutes)")
    fig.suptitle(f"{record}: conditional ST-change localization; green expert ST, orange positional shift")
    fig.tight_layout()
    fig.savefig(directory / f"{record}.png", dpi=120)
    plt.close(fig)


def _one_record(record: str, directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    waveform = wfdb.rdrecord(str(DATA / record))
    if waveform.fs != FS or waveform.sig_len != 1800000:
        raise ValueError("Unexpected European ST-T sample rate or length")
    annotation = wfdb.rdann(str(DATA / record), "atr")
    anchors = annotation.sample[np.array([symbol in BEAT_SYMBOLS for symbol in annotation.symbol])]
    features = change_windows(waveform.p_signal, anchors)
    episodes = parse_episodes(annotation.sample, annotation.aux_note, allow_censored=True)
    incomplete = [episode for episode in episodes if episode["end"] is None]
    censor_at = min(
        (episode["start"] for episode in incomplete if episode["kind"] in ("ST", "st")), default=7200.0
    )
    evaluation_mask = features["centers"] < censor_at
    if not evaluation_mask.any():
        raise ValueError("No windows before unknown annotation tail")
    evaluated = {name: features[name][evaluation_mask] for name in ("st", "whole", "centers")}
    targets = episode_targets(features["centers"], episodes, "ST")
    shifts = episode_targets(features["centers"], episodes, "st")
    summary = {
        "record": record,
        "patient": subject_id(record),
        "channels": waveform.sig_name,
        "complete_beats": len(features["anchors"]),
        "windows_per_channel": len(features["centers"]),
        "st_episodes": sum(episode["kind"] == "ST" for episode in episodes),
        "axis_episodes": sum(episode["kind"] == "st" for episode in episodes),
        "episode_counts": {
            kind: sum(episode["kind"] == kind and episode["end"] is not None for episode in episodes)
            for kind in ("ST", "T", "st", "t")
        },
        "eligible_st": bool(targets.any()),
        "eligible_axis": bool(shifts.any()),
    }
    summary.update(
        eligible_st=bool(targets[evaluation_mask].any()),
        eligible_axis=bool(shifts[evaluation_mask].any()),
        censor_at_seconds=censor_at,
        incomplete_episodes=len(incomplete),
        censored_windows_per_channel=int((~evaluation_mask).sum()),
    )
    features["evaluation_mask"] = evaluation_mask
    for method in METHODS:
        summary[method] = location_summary(evaluated[method], targets[evaluation_mask])
        summary[method]["axis_hit"] = location_summary(evaluated[method], shifts[evaluation_mask])["hit"]
        summary[method]["episode_coverage_top5pct"] = _coverage(
            evaluated[method], evaluated["centers"], episodes
        )
        summary[method]["extremum_distance_seconds"] = _extremum_distance(
            evaluated[method], evaluated["centers"], episodes
        )
    write_npz_atomic(
        directory / "features" / f"{record}.npz",
        **features,
        targets=targets,
        axis_targets=shifts,
    )
    write_json_atomic(directory / "episodes" / f"{record}.json", episodes)
    if record in ("e0161", "e0509", "e0601", "e0611", "e0613", "e0615"):
        _plot(record, features, episodes, waveform.sig_name, directory / "figures")
    windows = {
        "patient": subject_id(record),
        "target": targets[evaluation_mask],
        **{method: evaluated[method] for method in METHODS},
    }
    return summary, windows


def _summaries(
    records: list[dict[str, Any]], windows: list[dict[str, Any]], excluded: list[dict[str, str]]
) -> dict[str, Any]:
    eligible = [row for row in records if row["eligible_st"]]
    patients = np.asarray([row["patient"] for row in eligible])
    primary = {}
    for method in METHODS:
        hits = np.array([row[method]["hit"] for row in eligible])
        primary[method] = mean_interval(hits, patients)
        primary[method]["displayed_hit"] = mean_interval(
            np.array([row[method]["displayed_hit"] for row in eligible]), patients
        )
    gain = np.array([row["st"]["hit"] - row["whole"]["hit"] for row in eligible])
    paired = mean_interval(gain, patients)
    axis = [row for row in records if row["eligible_axis"]]
    axis_gain = (
        mean_interval(
            np.array([row["st"]["axis_hit"] - row["whole"]["axis_hit"] for row in axis]),
            np.asarray([row["patient"] for row in axis]),
        )
        if axis
        else None
    )
    primary_pass = paired["value"] >= 0.10 and paired["ci_low"] > 0 and primary["st"]["value"] >= 0.70
    quality_pass = len(records) >= 85 and paired["patients"] >= 30
    axis_pass = axis_gain is None or axis_gain["value"] <= 0.05
    for method in METHODS:
        primary[method]["chance"] = mean_interval(
            np.array([row[method]["chance"] for row in eligible]), patients
        )
        primary[method]["hit_minus_chance"] = mean_interval(
            np.array([row[method]["hit"] - row[method]["chance"] for row in eligible]), patients
        )
        primary[method]["tie_records"] = sum(row[method]["ties"] > 1 for row in eligible)
        primary[method]["mean_episode_coverage_top5pct"] = float(
            np.mean([row[method]["episode_coverage_top5pct"] for row in eligible])
        )
    return {
        "records": len(records),
        "patients": len({row["patient"] for row in records}),
        "excluded": excluded,
        "primary_records": len(eligible),
        "complete_episode_counts": {
            kind: sum(row["episode_counts"][kind] for row in records) for kind in ("ST", "T", "st", "t")
        },
        "censored_records": sum(row["censored_windows_per_channel"] > 0 for row in records),
        "censored_windows_per_channel": sum(row["censored_windows_per_channel"] for row in records),
        "primary": primary,
        "paired_gain": paired,
        "axis_records": len(axis),
        "axis_paired_hit_increase": axis_gain,
        "window_auroc": subject_auroc(windows),
        "passes_primary": primary_pass,
        "passes_counts": quality_pass,
        "passes_axis_stress": axis_pass,
        "promote_st_change_prototype_for_review": primary_pass and quality_pass,
        "promote_general_st_only_localizer": primary_pass and quality_pass and axis_pass,
        "clinical_single_ecg_localization_established": False,
    }


def main() -> None:
    """Run the prospective CPU-only benchmark and write its complete receipt."""
    start = time.monotonic()
    if OUTPUT.exists() or OUTPUT.with_suffix(".partial").exists():
        raise FileExistsError("057 outputs already exist; use a successor identity")
    directory = OUTPUT.with_suffix(".partial")
    directory.mkdir(parents=True)
    for name in ("features", "episodes", "figures"):
        (directory / name).mkdir()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(directory / "run.log")]
    )
    sources = (
        PROTOCOL,
        "ecg_experiment/st_episode057.py",
        "scripts/experiments/run_st_episode057.py",
        "ecg_experiment/intervals.py",
        "ecg_experiment/paths.py",
        "ecg_experiment/files.py",
        "ecg_experiment/raw_residual054.py",
        "ecg_experiment/fragment_localization.py",
        "docs/experiment-057-annotation-audit-amendment.md",
        "tests/test_st_episode057.py",
        "pyproject.toml",
        "uv.lock",
    )
    identity = {"git_head": git_head(ROOT), "sources": {path: sha256_file(ROOT / path) for path in sources}}
    integrity = _predecessor()
    LOG.info("054 primary and synthetic metrics reproduced exactly")
    names, hashes = _data_integrity()
    identity["data"] = hashes
    write_json_atomic(directory / "identity.json", identity)
    records, windows, excluded = [], [], []
    with threadpool_limits(limits=2):
        for index, record in enumerate(names):
            try:
                row, window = _one_record(record, directory)
            except ValueError as error:
                if str(error) not in (
                    "Fewer than three complete early-reference beats",
                    "No windows before unknown annotation tail",
                ):
                    raise
                excluded.append({"record": record, "reason": str(error)})
                continue
            records.append(row)
            windows.append(window)
            LOG.info("records %s/90", index + 1)
    result = _summaries(records, windows, excluded)
    write_json_atomic(directory / "per_record.json", records)
    _review_examples(records, directory)
    result.update({"identity": identity, "integrity": integrity, "elapsed_seconds": time.monotonic() - start})
    result["outputs_sha256"] = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "run.log"
    }
    write_json_atomic(directory / "result.json", result)
    directory.rename(OUTPUT)
    LOG.info(
        "primary gain %.4f; prototype review %s",
        result["paired_gain"]["value"],
        result["promote_st_change_prototype_for_review"],
    )


def _review_examples(records: list[dict[str, Any]], directory: Path) -> None:
    for success in (True, False):
        candidates = [
            row for row in records if row["eligible_st"] and bool(row["st"]["displayed_hit"]) == success
        ]
        for row in sorted(candidates, key=lambda item: item["record"])[:3]:
            record = row["record"]
            with np.load(directory / "features" / f"{record}.npz") as saved:
                features = {name: saved[name] for name in saved.files}
            episodes = json.loads((directory / "episodes" / f"{record}.json").read_text())
            _plot(record, features, episodes, row["channels"], directory / "figures")


if __name__ == "__main__":
    main()
