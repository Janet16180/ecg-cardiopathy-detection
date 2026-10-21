#!/usr/bin/env python3
"""Pretrain a larger SimDINOv2 encoder (for a GPU machine) and evaluate it with the 2,000-label recipe.

Edit only the block of variables below. The script needs, in the same paths as on the Mac:

- ``outputs/experiment050_ptbxl_final_pipeline/ptbxl_100hz_hp0.5_float16.npy`` and its
  ``interval_features_100hz_hp0.5.npy`` (PTB-XL, open data), plus ``data/raw/ptb-xl/1.0.3/ptbxl_database.csv``;
- ``outputs/experiment053_challenge_pool/`` caches, ``manifest.csv`` and ``prueba_interval_features_100hz.npy``
  (Challenge, open data);
- optionally ``outputs/experiment052_unlabeled_pool/`` (EchoNext). EchoNext is credentialed: the project rules say
  its data stay on this machine, so ``USE_ECHONEXT`` is off unless the data use agreement allows the VM.

It pretrains on the unlabelled pool, then reports, for the new encoder, the frozen recipe (embedding + interval
features + logistic head) on PTB-XL (three 2,000-label subsets and every label) and on the external Challenge test,
and on EchoNext structural heart disease when enabled. Results go to ``OUTPUT_DIR/results.json``.

Run from the repository root:

    python -u -m scripts.experiments.run_large_encoder_vm
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ecg_experiment.challenge_pool import EXTERNAL_CODES, evaluable_pairs, pair_aurocs
from ecg_experiment.echonext_signal_clusters import load_open_splits
from ecg_experiment.ptbxl_label_head import fit_head, head_scores, label_matrix
from ecg_experiment.ptbxl_pipeline_v2 import (
    ECGEncoder, EncoderSettings, PretrainNetwork, PretrainSettings, encode, make_teacher, pretrain_simdino,
    settings_record, standardize_features,
)
from ecg_experiment.ptbxl_simdino_memory import CODES, class_aurocs, load_ptbxl

ROOT = Path(__file__).resolve().parents[2]

# ---- Configuración: cambia solo esta parte ----
OUTPUT_DIR = ROOT / "outputs" / "experiment057_large_encoder"
DEVICE_NAME = None                      # None: CUDA, luego MPS, luego CPU
USE_PTBXL, USE_CHALLENGE, USE_ECHONEXT = True, True, False
MAX_PRETRAIN_ECGS = None                # un número para una prueba rápida
ENCODER = EncoderSettings(sampling_rate=100, patch_seconds=0.2, token_layout="time", width=384, depth=12, heads=6,
                          registers=4, drop_path=0.1)
PRETRAIN = PretrainSettings(epochs=100, batch_size=512, learning_rate=7e-4, warmup_epochs=10, local_views=10,
                            momentum_start=0.996, mixed_precision=True, seed=57)
LABELS_PER_TASK, REPEATS, HEAD_C = 2_000, 3, 0.1
# ------------------------------------------------

PTBXL_DATA = ROOT / "outputs" / "experiment050_ptbxl_final_pipeline"
CHALLENGE_DIR = ROOT / "outputs" / "experiment053_challenge_pool"
ECHONEXT_DIR = ROOT / "outputs" / "experiment052_unlabeled_pool"


def challenge_part(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rows and signals of one part of the local Challenge split, sorted by source and cache row.

    Parameters
    ----------
    split : str
        ``train``, ``calibracion`` or ``prueba``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Manifest rows and float16 signals ``(records, 12, 1000)``.
    """
    manifest = pd.read_csv(CHALLENGE_DIR / "manifest.csv", dtype={"record": str})
    rows = manifest[manifest.usable & manifest.split.eq(split)].sort_values(["source", "cache_row"])
    signals = np.concatenate([np.asarray(np.load(CHALLENGE_DIR / f"{source}_100hz_float16.npy", mmap_mode="r")
                                         [part.cache_row.to_numpy()]) for source, part in rows.groupby("source", sort=True)])
    return rows, signals


def ptbxl_subset(labels: np.ndarray, train: np.ndarray, repeat: int) -> np.ndarray:
    """
    The 2,000-label PTB-XL subset of Experiment 051 for one repeat.

    Parameters
    ----------
    labels : np.ndarray
        Boolean labels of every PTB-XL ECG for the 14 statements.
    train : np.ndarray
        Positions of the ``train`` ECGs.
    repeat : int
        Subset number.

    Returns
    -------
    np.ndarray
        Sorted positions with at least two positives per statement.
    """
    rng = np.random.default_rng(51 + 1000 * repeat + LABELS_PER_TASK)
    while True:
        chosen = np.sort(rng.choice(train, LABELS_PER_TASK, replace=False))
        if labels[chosen].sum(axis=0).min() >= 2:
            return chosen


def main() -> None:
    """Pretrain the encoder and evaluate the frozen recipe."""
    device = torch.device(DEVICE_NAME or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata, ptbxl = load_ptbxl(ROOT / "data" / "raw" / "ptb-xl" / "1.0.3", PTBXL_DATA / "ptbxl_100hz_hp0.5_float16.npy",
                                 rate=100)
    train = np.flatnonzero(metadata.split.eq("train").to_numpy())
    test = np.flatnonzero(metadata.split.eq("test").to_numpy())
    pool = []
    if USE_PTBXL:
        pool.append(np.asarray(ptbxl[train]))
    if USE_CHALLENGE:
        pool.append(challenge_part("train")[1])
    if USE_ECHONEXT:
        pool.append(np.asarray(np.load(ECHONEXT_DIR / "echonext_train_100hz_mv_float16.npy", mmap_mode="r")))
    pool = np.concatenate(pool)[:MAX_PRETRAIN_ECGS]
    print(f"ECG sin etiqueta: {len(pool):,}; dispositivo: {device}", flush=True)
    torch.manual_seed(PRETRAIN.seed)
    student = PretrainNetwork(ENCODER, PRETRAIN).to(device)
    teacher = make_teacher(student)
    pretrain_simdino(student, teacher, pool, ptbxl[np.flatnonzero(metadata.split.eq("val").to_numpy())[:1024]], ENCODER,
                     PRETRAIN, device=device, checkpoint_path=OUTPUT_DIR / "pretrain_checkpoint.pt")
    encoder = teacher.backbone
    torch.save(encoder.state_dict(), OUTPUT_DIR / "teacher_encoder.pt")

    def embed(signals: np.ndarray) -> np.ndarray:
        return np.concatenate(encode(encoder, signals, device=device, patches=True), axis=1)

    features = np.load(PTBXL_DATA / "interval_features_100hz_hp0.5.npy")
    test_rows, external = challenge_part("prueba")
    _, ptb_intervals, ext_intervals = standardize_features(features[train], features,
                                                           np.load(CHALLENGE_DIR / "prueba_interval_features_100hz.npy"))
    ptb_inputs = np.concatenate([embed(ptbxl), ptb_intervals], axis=1)
    ext_inputs = np.concatenate([embed(external), ext_intervals], axis=1)
    labels = label_matrix(metadata, CODES)
    ext_labels = test_rows[list(EXTERNAL_CODES)].to_numpy().astype(bool)
    ext_sources = test_rows.source.to_numpy()
    pairs = evaluable_pairs(ext_sources, ext_labels, EXTERNAL_CODES, 20)
    columns = [CODES.index(code) for code in EXTERNAL_CODES]
    results = {"settings": settings_record(ENCODER, PRETRAIN), "pretrain_ecgs": int(len(pool)), "ptbxl": {}, "external": {}}
    for repeat in [*range(REPEATS), "todas"]:
        chosen = train if repeat == "todas" else ptbxl_subset(labels, train, repeat)
        head = fit_head(ptb_inputs[chosen], labels[chosen], HEAD_C)
        results["ptbxl"][str(repeat)] = float(class_aurocs(labels[test], head_scores(head, ptb_inputs[test])).mean())
        results["external"][str(repeat)] = float(pair_aurocs(ext_sources, ext_labels, head_scores(head, ext_inputs)[:, columns],
                                                             pairs).mean())
    if USE_ECHONEXT:
        meta, _ = load_open_splits(ROOT)
        echo_labels = {split: meta.loc[meta.split.eq(split), "shd_moderate_or_greater_flag"].to_numpy().astype(bool)
                       for split in ("train", "val")}
        raw = np.load(ECHONEXT_DIR / "echonext_interval_features_100hz.npz")
        echo_train, echo_val = standardize_features(raw["train"], raw["val"])
        inputs = {split: np.concatenate([embed(np.load(ECHONEXT_DIR / f"echonext_{split}_100hz_mv_float16.npy", mmap_mode="r")),
                                         extra], axis=1) for split, extra in (("train", echo_train), ("val", echo_val))}
        results["echonext"] = {}
        for repeat in [*range(REPEATS), "todas"]:
            chosen = (np.arange(len(echo_labels["train"])) if repeat == "todas" else
                      np.sort(np.random.default_rng(52 + repeat).choice(len(echo_labels["train"]), LABELS_PER_TASK, replace=False)))
            head = fit_head(inputs["train"][chosen], echo_labels["train"][chosen, None], HEAD_C)
            results["echonext"][str(repeat)] = float(roc_auc_score(echo_labels["val"], head_scores(head, inputs["val"])[:, 0]))
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in results.items() if key != "settings"}, indent=2))


if __name__ == "__main__":
    main()
