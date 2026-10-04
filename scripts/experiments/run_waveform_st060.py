"""Execute the prospective waveform-only ST localization comparison."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import wfdb
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.st_episode057 import BEAT_SYMBOLS, FS, episode_targets, parse_episodes, subject_auroc
from ecg_experiment.waveform_st060 import (
    METHODS,
    benchmark_summary,
    detector_coverage,
    record_metrics,
    reproduce057,
    waveform_features,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = ROOT / "data/raw/edb/1.0.0"
PRIOR = ROOT / "outputs/experiment057_st_episode_v1"
OUTPUT = ROOT / "outputs/experiment060_waveform_st_v1"
LOG = logging.getLogger("experiment060")


def _one_record(prior: dict[str, Any], directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = prior["record"]
    waveform = wfdb.rdrecord(str(DATA / record))
    if waveform.fs != FS or waveform.sig_len != 1800000:
        raise ValueError("Unexpected waveform sample rate/length")
    features = waveform_features(waveform.p_signal)
    write_npz_atomic(directory / "features" / f"{record}.npz", **features)
    with np.load(PRIOR / "features" / f"{record}.npz") as saved:
        gold = {name: saved[name] for name in ("centers", "evaluation_mask", "targets", "axis_targets")}
    annotation = wfdb.rdann(str(DATA / record), "atr")
    episodes = parse_episodes(annotation.sample, annotation.aux_note, allow_censored=True)
    for name, kind in (("targets", "ST"), ("axis_targets", "st")):
        if not np.array_equal(episode_targets(gold["centers"], episodes, kind), gold[name]):
            raise ValueError(f"057 expert target changed: {record}/{kind}")
    cutoff = min(
        (
            episode["start"]
            for episode in episodes
            if episode["end"] is None and episode["kind"] in ("ST", "st")
        ),
        default=7200.0,
    )
    if not np.array_equal(gold["centers"] < cutoff, gold["evaluation_mask"]):
        raise ValueError("Annotation-completeness mask changed")
    row = record_metrics(features, gold, prior)
    expert_beats = annotation.sample[np.array([symbol in BEAT_SYMBOLS for symbol in annotation.symbol])]
    row["detector"] = detector_coverage(features["detected_anchors"], expert_beats)
    write_npz_atomic(directory / "evaluation" / f"{record}.npz", **gold)
    mask = gold["evaluation_mask"]
    windows = {"patient": prior["patient"], "target": gold["targets"][mask]}
    windows.update({name: features[name][mask] for name in METHODS})
    return row, windows


def _plot_examples(rows: list[dict[str, Any]], directory: Path) -> None:
    groups = {
        "corrected": [row for row in rows if row["eligible_st"] and row["st"]["hit"] > row["supplied_st"]],
        "regressed": [row for row in rows if row["eligible_st"] and row["st"]["hit"] < row["supplied_st"]],
    }
    if not groups["regressed"]:
        groups["failure"] = [row for row in rows if row["eligible_st"] and row["st"]["hit"] < 1]
    for group, examples in groups.items():
        for row in sorted(examples, key=lambda item: item["record"])[:3]:
            _plot(row, directory, group)


def _plot(row: dict[str, Any], directory: Path, group: str) -> None:
    record = row["record"]
    with np.load(directory / "features" / f"{record}.npz") as saved:
        features = {name: saved[name] for name in saved.files}
    with np.load(directory / "evaluation" / f"{record}.npz") as saved:
        gold = {name: saved[name] for name in saved.files}
    mask = gold["evaluation_mask"]
    with np.load(PRIOR / "features" / f"{record}.npz") as saved:
        supplied = saved["st"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for channel, axis in enumerate(axes):
        for name in METHODS:
            axis.plot(features["centers"][mask] / 60, features[name][mask, channel], label=f"Waveform {name}")
        axis.plot(
            features["centers"][mask] / 60, supplied[mask, channel], label="Supplied-anchor ST", alpha=0.6
        )
        axis.fill_between(
            gold["centers"][mask] / 60,
            0,
            1,
            where=gold["targets"][mask, channel],
            transform=axis.get_xaxis_transform(),
            alpha=0.12,
            color="green",
            label="Expert ST episode",
        )
        axis.set_ylabel(f"{row['channels'][channel]} change (mV)")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Original time (minutes)")
    fig.suptitle(f"{record}: {group}; waveform-derived anchors")
    fig.tight_layout()
    fig.savefig(directory / "figures" / f"{group}_{record}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the fixed CPU comparison and write complete source-backed evidence."""
    start = time.monotonic()
    directory = OUTPUT.with_suffix(".partial")
    if OUTPUT.exists() or directory.exists():
        raise FileExistsError("060 outputs exist; use a successor identity")
    directory.mkdir(parents=True)
    for name in ("features", "evaluation", "figures"):
        (directory / name).mkdir()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(directory / "run.log")]
    )
    sources = (
        "docs/experiment-060-waveform-st-localization.md",
        "ecg_experiment/waveform_st060.py",
        "scripts/experiments/run_waveform_st060.py",
        "tests/test_waveform_st060.py",
        "ecg_experiment/fragment_localization.py",
        "ecg_experiment/st_episode057.py",
        "ecg_experiment/incart_localization056.py",
        "ecg_experiment/intervals.py",
        "ecg_experiment/files.py",
        "ecg_experiment/paths.py",
        "ecg_experiment/provenance.py",
        "pyproject.toml",
        "uv.lock",
    )
    identity = {"git_head": git_head(ROOT), "sources": {path: sha256_file(ROOT / path) for path in sources}}
    integrity = reproduce057(PRIOR, ROOT)
    prior = json.loads((PRIOR / "result.json").read_text())
    identity["data"] = prior["identity"]["data"]
    write_json_atomic(directory / "identity.json", identity)
    previous = json.loads((PRIOR / "per_record.json").read_text())
    if len(previous) != 90 or sum(row["eligible_st"] for row in previous) != 85:
        raise ValueError("Predecessor population changed")
    rows, windows = [], []
    with threadpool_limits(limits=2):
        for index, row in enumerate(previous):
            found, window = _one_record(row, directory)
            rows.append(found)
            windows.append(window)
            LOG.info("records %s/90", index + 1)
    result = benchmark_summary(rows)
    result["window_auroc"] = subject_auroc(windows)
    result.update(
        identity=identity,
        integrity=integrity,
        elapsed_seconds=time.monotonic() - start,
        output_dir=to_stored(OUTPUT),
    )
    write_json_atomic(directory / "per_record.json", rows)
    _plot_examples(rows, directory)
    result["outputs_sha256"] = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "run.log"
    }
    write_json_atomic(directory / "result.json", result)
    directory.rename(OUTPUT)
    LOG.info("Review gate %s; gain %.4f", result["promote_for_review"], result["paired_gain"]["value"])


if __name__ == "__main__":
    main()
