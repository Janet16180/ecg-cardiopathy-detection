"""Experiment 043: ANN heads, an attention head on ECG-JEPA tokens and the tutor's CNN + transformer.

Stage 1 (CPU) trains MLP heads on xECG and on xECG + JEPA features, fits the logistic heads on the same
features, and gates Experiment 042's ``U_B`` map by the sign of its ``G_B`` shares. Stage 2 (GPU) caches
ECG-JEPA tokens and the canonical windows and trains an attention head on the tokens. Stage 3 (GPU) trains
the tutor's CNN + transformer from scratch on the cached windows. Every stage is compared with pipeline v2's
readout R on the same rows, split, evaluation sets and statistics.

The row logic lives in pinned runners (Experiments 030, 032, 037 and 042), which are imported as documented
in the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment import sph
from ecg_experiment.ann_heads import (
    SEEDS,
    VALIDATION_SHARE,
    AttentionHead,
    CNNTransformer,
    MLPHead,
    Recipe,
    RowFile,
    augment,
    create_row_file,
    detection_reading,
    gate_beat_units,
    grid_unit_map,
    map_reading,
    open_row_file,
    predict,
    ragged_unit_maps,
    read_rows,
    train,
    validation_mask,
    weighted_standardizer,
    write_rows,
)
from ecg_experiment.challenge_features import (
    RAW_ROOTS,
    canonical_window,
    read_verified,
    skip_reasons,
    window_start,
)
from ecg_experiment.external_encoders import (
    JEPA_CHECKPOINT,
    jepa_input,
    load_jepa,
    ptb_jepa_input,
    read_ptb_float64,
)
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.hard_subset import ARM_SPECS, arm_design
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    UnitMap,
    ecg_score,
    jepa_tokens,
    jepa_unit_map,
    premature_hit,
    top_lead,
    two_group_difference,
)
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.public_sources import signal_sha256
from scripts.data.extract_challenge_features import load_checksums, ningbo_items
from scripts.experiments.run_lead_wave_maps042 import blas_architectures, premature_targets, select_rows
from scripts.experiments.run_pipeline_v2_037 import receipt_checked
from scripts.experiments.run_pipeline_v2_037 import training_data as training_data037
from scripts.experiments.run_referral_budget030 import checked_csv
from scripts.experiments.run_rhythm_findings032 import (
    challenge_table,
    check_binary_rows,
    ptb_inputs,
    sph_inputs,
    training_sets,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment043_ann_heads_v1"
PRIOR032 = ROOT / "outputs/experiment032_rhythm_findings_v1"
PRIOR035 = ROOT / "outputs/experiment035_hard_subset_v1"
PRIOR037 = ROOT / "outputs/experiment037_pipeline_v2_v1"
PRIOR042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PROTOCOL = "docs/experiment-043-ann-heads.md"
SOURCES = (
    "ecg_experiment/ann_heads.py", "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/fragment_localization.py", "ecg_experiment/intervals.py",
    "ecg_experiment/multisource_readout.py", "ecg_experiment/hard_subset.py",
    "scripts/experiments/run_ann_heads043.py", "scripts/experiments/run_pipeline_v2_037.py",
    "scripts/experiments/run_rhythm_findings032.py", "scripts/experiments/run_referral_budget030.py",
    "scripts/experiments/run_lead_wave_maps042.py", "ecg_experiment/external_encoders.py",
    "ecg_experiment/challenge_features.py", "ecg_experiment/sph.py", "ecg_experiment/public_sources.py",
    "ecg_experiment/gpu.py", "scripts/data/extract_challenge_features.py", "pyproject.toml", "uv.lock",
    PROTOCOL,
)
ENCODERS = ("xecg", "jepa")
EXPECTED_TRAINING = {"binary": [39577, 27360], "dropped": 1724, "ptbxl_binary": 17083}
EXPECTED_SETS = {"full": [1572, 884, 1413], "ordinary": [1306, 843, 1173], "hard": [266, 41, 258],
                 "sph": [21008, 7190, 20364]}
EXPECTED_R_AUROC = {"full": 0.93056, "ordinary": 0.95356, "hard": 0.71241, "sph": 0.93869}
PRIMARY_SETS = ("sph", "full")
VALIDATION_SEED = 43043
DRAWS = 2000
SEED = 43043
REPRODUCTION_TOLERANCE = 1e-10
AUROC_TOLERANCE = 1e-12
LOCALIZATION_MARGIN = -0.10
THREADS = 4
MLP_RECIPE = Recipe(learning_rate=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=200, patience=10)
MLP_HIDDEN = 256
MLP_DROPOUT = 0.2
STAGE1_ARMS = {"mlp_xecg": ("mlp", "xecg"), "logistic_concat": ("logistic", "concat"),
               "mlp_concat": ("mlp", "concat"), "logistic_jepa": ("logistic", "jepa")}
MAP_POINT_FIELDS = ("ecgs_with_map", "evaluation_ecgs", "auroc", "average_precision", "any_red",
                    "mean_red_units", "pvc_ecgs", "hit_rate", "chance_rate")
SMOKE_ROWS = 4000
SMOKE_HELD_OUT = 0.25
SMOKE_EPOCHS = 2
SMOKE_DRAWS = 200
PRIOR_STAGE1 = OUTPUT / "stage1"
PRIOR_STAGE2 = OUTPUT / "stage2"
CNN_RECIPE = Recipe(learning_rate=5e-4, weight_decay=0.05, batch_size=64, max_epochs=40, patience=5, clip=1.0)
STAGE3_SAMPLES = 2500
STAGE3_PREDICT_BATCH = 128
STAGE3_CEILING = 14400.0
PREPROCESS_CHUNK = 1024
CACHE = Path("/tmp/claude-218201143/-home-janetrivera-ecg-cardiopathy-detection/"
             "64bd5d6e-e2f5-403b-8837-1c9210d4ec66/scratchpad/exp043_cache")
ATTENTION_RECIPE = Recipe(learning_rate=3e-4, weight_decay=1e-2, batch_size=64, max_epochs=30, patience=4)
TOKEN_SHAPE = (400, 768)
WINDOW_SHAPE = (12, 5000)
EXTRACT_CHUNK = 256
READER_THREADS = 16
WARMUP_STEPS = 3
PROFILE_RECORDS = 128
PROFILE_STEPS = 30
PREDICT_BATCH = 256
STAGE2_CEILING = 7200.0
FEATURE_TOLERANCE = 1e-4
CONTRIBUTION_TOLERANCE = 1e-4
SMOKE_TOKEN_ROWS = 300
LOG = logging.getLogger("experiment043")


def training_data() -> dict[str, Any]:
    """
    Build pipeline v2's training rows, labels and weights with xECG and JEPA features, as Experiment 037 does.

    Returns
    -------
    dict[str, Any]
        Stacked features per encoder (``x``), readout sets, 035's ``dropped_upweighted`` design, the
        validation ``groups`` of the binary rows, SPH and development rows with features per encoder, and the
        input identity.

    Raises
    ------
    ValueError
        If a row order or count differs from 032 or the protocol.
    """
    ptb, ptb_x, ptb_identity = ptb_inputs()
    sph, sph_x, sph_hashes = sph_inputs()
    rows, challenge_x = challenge_table()
    in_train = (rows["split"] == "train").to_numpy()
    train_rows = rows[in_train].copy()
    saved = checked_csv(PRIOR032, "training_rows.csv", dtype={"record": str}, keep_default_na=False)
    if not all(np.array_equal(saved[column].to_numpy(dtype=str), train_rows[column].to_numpy(dtype=str))
               for column in ("source", "record")):
        raise ValueError("Challenge training rows differ from 032's")
    train_rows["reasons"] = saved["reasons"].to_numpy(dtype=str)
    check_binary_rows(train_rows)
    kept = (train_rows["reasons"] == "").to_numpy()
    readouts = training_sets(ptb["train"], train_rows[kept])
    x = {name: np.concatenate([ptb_x[name]["train"], challenge_x[name][in_train][kept]]) for name in ENCODERS}
    families = np.concatenate([np.full(len(ptb["train"]), "ptbxl"),
                               train_rows.loc[kept, "family"].to_numpy(dtype=str)])
    dropped = np.concatenate([ptb["train"]["target"].isna().to_numpy(),
                              np.zeros(int(kept.sum()), dtype=bool)])
    groups = np.concatenate([ptb["train"]["patient_id"].to_numpy(dtype=str),
                             (train_rows.loc[kept, "source"] + ":" + train_rows.loc[kept, "record"])
                             .to_numpy(dtype=str)])
    binary_rows = readouts["binary"]["rows"]
    design = arm_design(ARM_SPECS["dropped_upweighted"], families[binary_rows],
                        {"primary": readouts["binary"]["y"].astype(np.float64)}, dropped[binary_rows])
    found = {"binary": [len(binary_rows), int(readouts["binary"]["y"].sum())],
             "dropped": int(dropped[binary_rows].sum()),
             "ptbxl_binary": int((families[binary_rows] == "ptbxl").sum())}
    if found != EXPECTED_TRAINING or not design["selected"].all():
        raise ValueError(f"Training counts differ from the protocol: {found}")
    identity = {**ptb_identity, "sph_manifest": sph_hashes}
    challenge = train_rows[kept]
    stacked = pd.DataFrame({
        "kind": np.concatenate([np.full(len(ptb["train"]), "ptbxl"), np.full(len(challenge), "challenge")]),
        "key": np.concatenate([("ptbxl:" + ptb["train"]["ecg_id"].astype(str)).to_numpy(dtype=str),
                               (challenge["source"] + ":" + challenge["record"]).to_numpy(dtype=str)]),
        "filename_hr": np.concatenate([ptb["train"]["filename_hr"].to_numpy(dtype=str),
                                       np.full(len(challenge), "")]),
        "source": np.concatenate([np.full(len(ptb["train"]), ""), challenge["source"].to_numpy(dtype=str)]),
        "path": np.concatenate([np.full(len(ptb["train"]), ""), challenge["path"].to_numpy(dtype=str)]),
        "window_start": np.concatenate([np.full(len(ptb["train"]), -1),
                                        challenge["window_start"].to_numpy(dtype=np.int64)]),
        "window_sha256": np.concatenate([np.full(len(ptb["train"]), ""),
                                         challenge["window_sha256"].fillna("").to_numpy(dtype=str)]),
    })
    return {"x": x, "readouts": readouts, "design": design, "groups": groups[binary_rows],
            "stacked": stacked.iloc[binary_rows].reset_index(drop=True),
            "families": families[binary_rows], "sph": sph, "sph_x": {name: sph_x[name] for name in ENCODERS},
            "development": ptb["development"],
            "development_x": {name: ptb_x[name]["development"] for name in ENCODERS},
            "identity": identity, "training_counts": found}


def check_against_037(data: dict[str, Any]) -> dict[str, Any]:
    """
    Require the xECG rows, labels, weights and evaluation rows to equal Experiment 037's ``training_data``.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    dict[str, Any]
        Shapes compared.

    Raises
    ------
    ValueError
        If anything differs.
    """
    prior = training_data037()
    checks = {
        "x": np.array_equal(prior["x"], data["x"]["xecg"]),
        "binary_rows": np.array_equal(prior["readouts"]["binary"]["rows"],
                                      data["readouts"]["binary"]["rows"]),
        "y": np.array_equal(prior["design"]["y"], data["design"]["y"]),
        "weights": np.array_equal(prior["design"]["weights"], data["design"]["weights"]),
        "sph_x": np.array_equal(prior["sph_x"], data["sph_x"]["xecg"]),
        "development_x": np.array_equal(prior["development_x"], data["development_x"]["xecg"]),
        "development_rows": np.array_equal(prior["development"].index.to_numpy(dtype=str),
                                           data["development"].index.to_numpy(dtype=str)),
        "sph_rows": np.array_equal(prior["sph"]["ecg_id"].to_numpy(dtype=str),
                                   data["sph"]["ecg_id"].to_numpy(dtype=str)),
    }
    if not all(checks.values()):
        raise ValueError(f"Rows differ from Experiment 037's training_data: {checks}")
    return {"equal": checks, "x_shape": list(prior["x"].shape)}


def binary_features(data: dict[str, Any], name: str, part: str) -> np.ndarray:
    """
    Return one feature set (``xecg``, ``jepa`` or ``concat``) of the binary training rows or a scored set.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    name : str
        Feature set.
    part : str
        ``train`` (binary training rows), ``development`` or ``sph``.

    Returns
    -------
    np.ndarray
        ``[rows, features]``.
    """
    def of(encoder: str) -> np.ndarray:
        if part == "train":
            return data["x"][encoder][data["readouts"]["binary"]["rows"]]
        return data[f"{part}_x"][encoder]

    if name == "concat":
        return np.concatenate([of("xecg"), of("jepa")], axis=1)
    return of(name)


def shared_validation(data: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    """
    Draw the validation split shared by every arm and seed over the binary training rows.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        Validation mask and its counts.
    """
    mask = validation_mask(data["groups"], VALIDATION_SHARE, VALIDATION_SEED)
    y = data["design"]["y"]
    counts = {"groups": int(len(np.unique(data["groups"]))),
              "validation_groups": int(len(np.unique(data["groups"][mask]))),
              "validation_rows": int(mask.sum()), "validation_positive": int(y[mask].sum()),
              "training_rows": int((~mask).sum()), "training_positive": int(y[~mask].sum()),
              "validation_ptbxl": int((data["families"][mask] == "ptbxl").sum())}
    return mask, counts


def logits_of(head: tuple, x: np.ndarray) -> np.ndarray:
    """Return the decision values of a fitted scaler and logistic head."""
    scaler, model = head
    return model.decision_function(scaler.transform(np.asarray(x, dtype=np.float64)))


def comparator(data: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Refit pipeline v2's readout R and require it to reproduce Experiment 037's saved probabilities.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        R's logits on ``development`` and ``sph``, and the reproduction differences.

    Raises
    ------
    ValueError
        If the saved rows differ or a probability differs by more than ``REPRODUCTION_TOLERANCE``.
    """
    design = data["design"]
    with threadpool_limits(limits=1):
        head = fit_readout(binary_features(data, "xecg", "train"), design["y"], design["weights"])
        logits = {part: logits_of(head, binary_features(data, "xecg", part))
                  for part in ("development", "sph")}
    with np.load(PRIOR037 / "predictions.npz") as saved:
        if not (np.array_equal(saved["sph_ecg_ids"], data["sph"]["ecg_id"].to_numpy(dtype=str))
                and np.array_equal(saved["development_record_ids"],
                                   data["development"].index.to_numpy(dtype=str))):
            raise ValueError("037 prediction rows differ")
        differences = {part: float(np.abs(1 / (1 + np.exp(-logits[part])) - saved[f"{part}_v2_binary"]).max())
                       for part in ("development", "sph")}
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"R does not reproduce 037: {differences}")
    return logits, {"probability_max_abs_difference": differences, "iterations": int(head[1].n_iter_[0])}


def evaluation_sets(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Experiment 037's AUROC sets: full, ordinary and hard PTB-XL development, and labeled SPH.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per set: ``part`` (``development`` or ``sph``), ``positions`` into that part's scores, ``y`` and
        ``patients``.

    Raises
    ------
    ValueError
        If a count differs from the protocol.
    """
    with np.load(PRIOR035 / "predictions.npz") as saved:
        record_ids = saved["development_record_ids"]
        labels, original = saved["development_labels"], saved["development_original"]
    development = data["development"]
    positions = pd.Index(development.index.to_numpy(dtype=str)).get_indexer(record_ids)
    if (positions < 0).any():
        raise ValueError("035 development rows are missing")
    patients = development["patient_id"].to_numpy(dtype=str)[positions]
    selections = {"full": np.ones(len(original), dtype=bool), "ordinary": original, "hard": ~original}
    sets = {name: {"part": "development", "positions": positions[mask], "y": labels[mask].astype(np.int64),
                   "patients": patients[mask]} for name, mask in selections.items()}
    sph = data["sph"]
    labeled = sph["primary"].notna().to_numpy()
    sets["sph"] = {"part": "sph", "positions": np.flatnonzero(labeled),
                   "y": sph.loc[labeled, "primary"].to_numpy(dtype=np.int64),
                   "patients": sph.loc[labeled, "patient_id"].to_numpy(dtype=str)}
    found = {name: [len(spec["y"]), int(spec["y"].sum()), len(np.unique(spec["patients"]))]
             for name, spec in sets.items()}
    if found != EXPECTED_SETS:
        raise ValueError(f"Evaluation counts differ from the protocol: {found}")
    return sets


def set_scores(spec: dict[str, Any], scores: dict[str, np.ndarray]) -> np.ndarray:
    """Return the scores of one evaluation set from per-part scores."""
    return scores[spec["part"]][spec["positions"]]


def check_comparator_aurocs(sets: dict[str, dict[str, Any]], reference: dict[str, np.ndarray]
                            ) -> dict[str, float]:
    """
    Require R's AUROC on every set to equal Experiment 037's and the protocol's rounded values.

    Parameters
    ----------
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.
    reference : dict[str, np.ndarray]
        R's scores per part.

    Returns
    -------
    dict[str, float]
        R's AUROC per set.

    Raises
    ------
    ValueError
        If an AUROC differs.
    """
    saved = json.loads((PRIOR037 / "result.json").read_text())["auroc"]
    found = {name: float(roc_auc_score(spec["y"], set_scores(spec, reference)))
             for name, spec in sets.items()}
    for name, value in found.items():
        if (abs(value - saved[name]["auroc"]["v2"]) > AUROC_TOLERANCE
                or round(value, 5) != EXPECTED_R_AUROC[name]):
            raise ValueError(f"R's {name} AUROC {value} differs from 037's")
    return found


def detection_statistics(sets: dict[str, dict[str, Any]], arms: dict[str, dict[str, np.ndarray]],
                         reference: dict[str, np.ndarray], draws: int, seed: int,
                         primary: tuple[str, ...] = PRIMARY_SETS) -> dict[str, Any]:
    """
    AUROC and AP of every arm on every set, and paired AUROC differences against R on the primary sets.

    Parameters
    ----------
    sets : dict[str, dict[str, Any]]
        Evaluation sets (``evaluation_sets`` or the smoke set).
    arms : dict[str, dict[str, np.ndarray]]
        Scores per arm and part.
    reference : dict[str, np.ndarray]
        R's scores per part.
    draws : int
        Bootstrap draws.
    seed : int
        Bootstrap seed.
    primary : tuple[str, ...]
        Sets with a paired difference.

    Returns
    -------
    dict[str, Any]
        Per arm: ``sets`` (AUROC and AP) and ``minus_R``.
    """
    result = {}
    for arm, scores in {"R": reference, **arms}.items():
        found: dict[str, Any] = {"sets": {}}
        for name, spec in sets.items():
            values = set_scores(spec, scores)
            found["sets"][name] = {"auroc": float(roc_auc_score(spec["y"], values)),
                                   "average_precision": float(average_precision_score(spec["y"], values))}
        if arm != "R":
            found["minus_R"] = {name: paired_auroc_difference(sets[name]["patients"], sets[name]["y"],
                                                              set_scores(sets[name], scores),
                                                              set_scores(sets[name], reference), draws, seed)
                                for name in primary if name in sets}
        result[arm] = found
    return result


def paired_contrast(sets: dict[str, dict[str, Any]], first: dict[str, np.ndarray],
                    second: dict[str, np.ndarray], draws: int, seed: int,
                    names: tuple[str, ...] = PRIMARY_SETS) -> dict[str, Any]:
    """Return the paired AUROC difference, first minus second, on the named sets."""
    return {name: paired_auroc_difference(sets[name]["patients"], sets[name]["y"],
                                          set_scores(sets[name], first), set_scores(sets[name], second),
                                          draws, seed)
            for name in names if name in sets}


def train_mlp_arm(x: np.ndarray, y: np.ndarray, weights: np.ndarray, validation: np.ndarray,
                  scored: dict[str, np.ndarray], recipe: Recipe, log: Any) -> dict[str, Any]:
    """
    Train the MLP head with every seed and score the scored parts.

    Inputs are standardized with the weighted mean and SD of the training part (the rows outside the
    validation split).

    Parameters
    ----------
    x : np.ndarray
        Binary training rows' features.
    y : np.ndarray
        Their targets.
    weights : np.ndarray
        Their design weights.
    validation : np.ndarray
        Validation mask over the rows.
    scored : dict[str, np.ndarray]
        Features per scored part.
    recipe : Recipe
        Optimizer and stopping settings.
    log : Any
        Receives one line per epoch.

    Returns
    -------
    dict[str, Any]
        ``logits`` (mean over seeds, per part), ``seed_logits`` (per seed and part) and ``seeds`` (best epoch,
        validation AUROC, epochs run, seconds).
    """
    fit_rows, check_rows = np.flatnonzero(~validation), np.flatnonzero(validation)
    mean, scale = weighted_standardizer(np.asarray(x[fit_rows], dtype=np.float64), weights[fit_rows])

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32))

    inputs = tensor(x)
    parts = {name: tensor(values) for name, values in scored.items()}
    seed_logits: dict[int, dict[str, np.ndarray]] = {}
    seeds = {}
    for seed in SEEDS:
        began = time.perf_counter()
        torch.manual_seed(seed)
        model = MLPHead(x.shape[1], MLP_HIDDEN, MLP_DROPOUT)
        fit = train(model, lambda index: inputs[index], fit_rows, y[fit_rows], weights[fit_rows], check_rows,
                    y[check_rows], recipe, seed, log=log)
        seed_logits[seed] = {name: predict(model, lambda index, values=values: values[index],
                                           np.arange(len(values)), 4096)[0]
                             for name, values in parts.items()}
        seeds[str(seed)] = {"best_epoch": fit.best_epoch, "validation_auroc": fit.best_auroc,
                            "epochs_run": len(fit.history), "seconds": time.perf_counter() - began,
                            "history": fit.history}
    logits = {name: np.mean([seed_logits[seed][name] for seed in SEEDS], axis=0) for name in parts}
    return {"logits": logits, "seed_logits": seed_logits, "seeds": seeds}


def stage1_arms(x: dict[str, np.ndarray], y: np.ndarray, weights: np.ndarray, validation: np.ndarray,
                scored: dict[str, dict[str, np.ndarray]], recipe: Recipe) -> dict[str, dict[str, Any]]:
    """
    Fit every Stage 1 detection arm and score the scored parts.

    Parameters
    ----------
    x : dict[str, np.ndarray]
        Training features per feature set (``xecg``, ``jepa``, ``concat``).
    y : np.ndarray
        Targets.
    weights : np.ndarray
        Design weights.
    validation : np.ndarray
        Validation mask (MLP arms only; logistic arms use every row).
    scored : dict[str, dict[str, np.ndarray]]
        Features per feature set, then scored part.
    recipe : Recipe
        MLP recipe.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per arm: ``logits`` per part, and the seeds or iterations.
    """
    arms = {}
    for arm, (kind, features) in STAGE1_ARMS.items():
        began = time.perf_counter()
        LOG.info("arm %s on %s features %s", arm, features, x[features].shape)
        if kind == "mlp":
            arms[arm] = train_mlp_arm(x[features], y, weights, validation, scored[features], recipe,
                                      log=lambda line, name=arm: LOG.info("%s %s", name, line))
        else:
            with threadpool_limits(limits=1):
                head = fit_readout(x[features], y, weights)
                arms[arm] = {"logits": {part: logits_of(head, values)
                                        for part, values in scored[features].items()},
                             "iterations": int(head[1].n_iter_[0])}
        arms[arm]["seconds"] = time.perf_counter() - began
        LOG.info("arm %s done in %.1f s", arm, arms[arm]["seconds"])
    return arms


def seed_statistics(sets: dict[str, dict[str, Any]], arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return every seed's AUROC per set, best epoch and validation AUROC, for the network arms."""
    result = {}
    for arm, found in arms.items():
        if "seed_logits" not in found:
            continue
        result[arm] = {seed: {"best_epoch": info["best_epoch"], "validation_auroc": info["validation_auroc"],
                              "epochs_run": info["epochs_run"],
                              "auroc": {name: float(roc_auc_score(spec["y"], set_scores(
                                  spec, found["seed_logits"][int(seed)]))) for name, spec in sets.items()}}
                       for seed, info in found["seeds"].items()}
    return result


def readings(statistics: dict[str, Any]) -> dict[str, str]:
    """Return the prespecified beats, matches or below reading of every detection arm."""
    return {arm: detection_reading(found["minus_R"]["sph"]["ci_low"], found["minus_R"]["full"]["difference"])
            for arm, found in statistics.items() if arm != "R"}


def map_inputs() -> dict[str, Any]:
    """
    Experiment 042's rows, premature-beat windows and saved ``U_B`` and ``G_B`` maps.

    Returns
    -------
    dict[str, Any]
        ``rows`` (042 ``select_rows``), ``windows``, ``exclusions``, ``maps`` (``U_B`` and ``G_B`` in scored
        order) and ``receipt`` hashes.

    Raises
    ------
    ValueError
        If the saved rows differ from 042's scored rows.
    """
    receipt = receipt_checked(PRIOR042, ("unit_scores.npz",))
    rows = select_rows(False)
    with np.load(PRIOR042 / "unit_scores.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], rows["scored"]["ecg_id"].to_numpy(np.int64)):
            raise ValueError("042 unit scores differ from 042's scored rows")
        arrays = {key: saved[key] for key in saved.files if key.startswith(("U_B_", "G_B_"))
                  and not key.startswith("U_B_median")}
    maps = {name: ragged_unit_maps(arrays, name) for name in ("U_B", "G_B")}
    windows, exclusions = premature_targets(rows["pvc"])
    return {"rows": rows, "windows": windows, "exclusions": exclusions, "maps": maps, "receipt": receipt}


def gated_map(beat_maps: list[UnitMap | None], share_maps: list[UnitMap | None]
              ) -> tuple[list[UnitMap | None], dict[str, float]]:
    """
    Gate every ECG's ``U_B`` units by the sign of its ``G_B`` share of the same lead and wave.

    Parameters
    ----------
    beat_maps : list[UnitMap | None]
        ``U_B`` per ECG.
    share_maps : list[UnitMap | None]
        ``G_B`` per ECG.

    Returns
    -------
    tuple[list[UnitMap | None], dict[str, float]]
        ``U_B_gated`` per ECG, and the share of units kept.

    Raises
    ------
    ValueError
        If the lead order of the two maps differs.
    """
    gated: list[UnitMap | None] = []
    kept, total = 0, 0
    for beats, shares in zip(beat_maps, share_maps, strict=True):
        if beats is None or shares is None:
            gated.append(None)
            continue
        if not np.array_equal(beats.leads.reshape(-1, shares.scores.size), np.tile(shares.leads, (
                len(beats.scores) // shares.scores.size, 1))):
            raise ValueError("U_B and G_B lead orders differ")
        scores = gate_beat_units(beats.scores, shares.scores)
        kept += int((scores != 0).sum())
        total += len(scores)
        gated.append(UnitMap(scores, beats.leads, beats.starts, beats.ends))
    return gated, {"units": total, "kept_share": kept / total}


def map_metrics(rows: dict[str, Any], maps: dict[str, list[UnitMap | None]],
                windows: dict[int, list[tuple[float, float]]], draws: int, seed: int) -> dict[str, Any]:
    """
    Compute Experiment 042's map metrics: red threshold, worst-unit AUROC, red shares, lead contrasts, hits.

    Parameters
    ----------
    rows : dict[str, Any]
        042 ``select_rows`` output.
    maps : dict[str, list[UnitMap | None]]
        Unit maps in ``rows["scored"]`` order.
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows keyed by ECG ID.
    draws : int
        Bootstrap draws.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``thresholds`` and per-map ``maps`` metrics, plus the per-ECG values used by ``map_contrasts``.
    """
    ids = rows["scored"]["ecg_id"].to_numpy()
    patients = dict(zip(ids, rows["scored"]["patient_id"], strict=True))
    evaluation = rows["evaluation"]
    labels = dict(zip(evaluation["ecg_id"], evaluation["standard"].astype(int), strict=True))
    normals = [i for i, y in labels.items() if y == 0]
    groups = {"normal": normals, "positive": [i for i, y in labels.items() if y == 1],
              "pvc": rows["pvc"]["ecg_id"].tolist(), "benign": rows["benign"]["ecg_id"].tolist()}
    result: dict[str, Any] = {"maps": {}, "thresholds": {}, "per_ecg": {}}
    for name, unit_maps in maps.items():
        by_id = {i: m for i, m in zip(ids, unit_maps, strict=True) if m is not None}
        threshold = float(np.quantile([ecg_score(by_id[i]) for i in normals if i in by_id], 0.95))
        result["thresholds"][name] = threshold
        scored_eval = [i for i in labels if i in by_id]
        y = np.array([labels[i] for i in scored_eval])
        score = np.array([ecg_score(by_id[i]) for i in scored_eval])
        red = {i: m.scores > threshold for i, m in by_id.items()}
        found: dict[str, Any] = {
            "ecgs_with_map": len(by_id), "evaluation_ecgs": len(scored_eval),
            "auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "any_red": {g: float(np.mean([red[i].any() for i in m if i in red])) for g, m in groups.items()},
            "mean_red_units": {g: float(np.mean([red[i].sum() for i in m if i in red]))
                               for g, m in groups.items()},
        }
        contrasts = {}
        for region, leads, first, second in (("anterior", ANTERIOR, "anterior", "inferior"),
                                             ("inferior", INFERIOR, "inferior", "anterior")):
            a = [i for i in rows[first]["ecg_id"] if i in by_id]
            b = [i for i in rows[second]["ecg_id"] if i in by_id]
            contrasts[region] = two_group_difference(
                np.array([patients[i] for i in a]), np.array([float(top_lead(by_id[i]) in leads) for i in a]),
                np.array([patients[i] for i in b]), np.array([float(top_lead(by_id[i]) in leads) for i in b]),
                draws, seed)
        found["lead_contrast"] = contrasts
        included = [i for i in windows if i in by_id]
        hit, chance = np.array([premature_hit(by_id[i], windows[i]) for i in included]).T
        pvc_patients = np.array([patients[i] for i in included])
        found.update({"pvc_ecgs": len(included), "hit_rate": float(hit.mean()),
                      "chance_rate": float(chance.mean()),
                      "hit_minus_chance": bootstrap_mean(pvc_patients, hit - chance, draws, seed)})
        result["maps"][name] = found
        result["per_ecg"][name] = {"evaluation": scored_eval, "y": y, "score": score,
                                   "excess": dict(zip(included, hit - chance, strict=True)),
                                   "benign_red": {i: bool(red[i].any())
                                                  for i in groups["benign"] if i in red}}
    result["patients"] = patients
    return result


def map_contrasts(metrics: dict[str, Any], name: str, baseline: str, draws: int, seed: int) -> dict[str, Any]:
    """
    Paired contrasts of one map minus a baseline map: hit - chance, benign any-red and worst-unit AUROC.

    Parameters
    ----------
    metrics : dict[str, Any]
        Output of ``map_metrics`` with both maps.
    name : str
        Map.
    baseline : str
        Baseline map.
    draws : int
        Bootstrap draws.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        The three contrasts, and the reading by 042's rule.
    """
    patients = metrics["patients"]
    mine, theirs = metrics["per_ecg"][name], metrics["per_ecg"][baseline]
    pvc = [i for i in mine["excess"] if i in theirs["excess"]]
    benign = [i for i in mine["benign_red"] if i in theirs["benign_red"]]
    if mine["evaluation"] != theirs["evaluation"]:
        raise ValueError("The two maps cover different evaluation ECGs")
    found = {
        "hit_minus_chance": bootstrap_mean(np.array([patients[i] for i in pvc]),
                                           np.array([mine["excess"][i] - theirs["excess"][i] for i in pvc]),
                                           draws, seed),
        "benign_any_red": bootstrap_mean(np.array([patients[i] for i in benign]),
                                         np.array([float(mine["benign_red"][i])
                                                   - float(theirs["benign_red"][i]) for i in benign]),
                                         draws, seed),
        "auroc": paired_auroc_difference(np.array([patients[i] for i in mine["evaluation"]]), mine["y"],
                                         mine["score"], theirs["score"], draws, seed),
    }
    found["reading"] = map_reading(found["hit_minus_chance"]["ci_low"],
                                   metrics["maps"][name]["lead_contrast"]["anterior"]["ci_low"],
                                   found["benign_any_red"]["ci_high"], found["auroc"]["ci_low"],
                                   LOCALIZATION_MARGIN)
    return found


def check_042_points(found: dict[str, Any], threshold: float) -> dict[str, bool]:
    """
    Require the recomputed ``U_B`` point estimates to equal Experiment 042's.

    Parameters
    ----------
    found : dict[str, Any]
        ``map_metrics`` metrics of ``U_B``.
    threshold : float
        Its recomputed red threshold.

    Returns
    -------
    dict[str, bool]
        Per field, whether it equals 042's.

    Raises
    ------
    ValueError
        If any differs.
    """
    prior = json.loads((PRIOR042 / "result.json").read_text())
    checks = {key: found[key] == prior["maps"]["U_B"][key] for key in MAP_POINT_FIELDS}
    checks["threshold"] = threshold == prior["thresholds"]["U_B"]
    checks["hit_minus_chance"] = (found["hit_minus_chance"]["value"]
                                  == prior["maps"]["U_B"]["hit_minus_chance"]["value"])
    for region in ("anterior", "inferior"):
        checks[f"{region}_contrast"] = all(found["lead_contrast"][region][key]
                                           == prior["maps"]["U_B"]["lead_contrast"][region][key]
                                           for key in ("value", "group_a", "group_b"))
    if not all(checks.values()):
        raise ValueError(f"U_B does not reproduce 042: {checks}")
    return checks


def public_map_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return ``map_metrics`` without its per-ECG working values."""
    return {"thresholds": metrics["thresholds"], "maps": metrics["maps"]}


def identity(receipts: dict[str, Any], data_identity: dict[str, Any]) -> dict[str, Any]:
    """
    Hash every source and the protocol, with the input receipts and the run environment.

    Parameters
    ----------
    receipts : dict[str, Any]
        Hashes of earlier outputs, checked against their receipts, keyed by repository-relative path.
    data_identity : dict[str, Any]
        Input identity from ``training_data``.

    Returns
    -------
    dict[str, Any]
        The identity block of ``result.json``.
    """
    return {"inputs": receipts, "data": data_identity,
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES}, "git_head": git_head(ROOT),
            "openblas_architectures": blas_architectures(), "torch": torch.__version__,
            "jepa_checkpoint_sha256": sha256_file(JEPA_CHECKPOINT),
            "torch_threads": torch.get_num_threads()}


def smoke_split(data: dict[str, Any]) -> dict[str, Any]:
    """
    Pick a few thousand binary training rows and hold out a quarter of their groups for scoring.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    dict[str, Any]
        ``fit`` and ``held_out`` positions into the binary rows, and the validation mask over ``fit``.
    """
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(data["groups"]), SMOKE_ROWS, replace=False))
    held = validation_mask(data["groups"][chosen], SMOKE_HELD_OUT, SEED + 1)
    fit, held_out = chosen[~held], chosen[held]
    return {"fit": fit, "held_out": held_out,
            "validation": validation_mask(data["groups"][fit], VALIDATION_SHARE, VALIDATION_SEED)}


def run_smoke(data: dict[str, Any], partial: Path) -> dict[str, Any]:
    """
    Run every Stage 1 code path on training rows only: no development or SPH ECG is scored.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    partial : Path
        Output folder.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    split = smoke_split(data)
    y, weights = data["design"]["y"], data["design"]["weights"]
    names = ("xecg", "jepa", "concat")
    train_x = {name: binary_features(data, name, "train") for name in names}
    fit_x = {name: values[split["fit"]] for name, values in train_x.items()}
    scored = {name: {"held_out": values[split["held_out"]]} for name, values in train_x.items()}
    recipe = Recipe(MLP_RECIPE.learning_rate, MLP_RECIPE.weight_decay, MLP_RECIPE.batch_size, SMOKE_EPOCHS,
                    MLP_RECIPE.patience)
    with threadpool_limits(limits=1):
        head = fit_readout(fit_x["xecg"], y[split["fit"]], weights[split["fit"]])
        reference = {"held_out": logits_of(head, scored["xecg"]["held_out"])}
    arms = stage1_arms(fit_x, y[split["fit"]], weights[split["fit"]], split["validation"], scored, recipe)
    sets = {"held_out": {"part": "held_out", "positions": np.arange(len(split["held_out"])),
                         "y": y[split["held_out"]], "patients": data["groups"][split["held_out"]]}}
    statistics = detection_statistics(sets, {arm: found["logits"] for arm, found in arms.items()}, reference,
                                      SMOKE_DRAWS, SEED, primary=("held_out",))
    contrast = paired_contrast(sets, arms["mlp_concat"]["logits"], arms["logistic_concat"]["logits"],
                               SMOKE_DRAWS, SEED, names=("held_out",))
    receipt = receipt_checked(PRIOR042, ("unit_scores.npz",))
    with np.load(PRIOR042 / "unit_scores.npz") as saved:
        arrays = {key: saved[key] for key in saved.files if key.startswith(("U_B_", "G_B_"))
                  and not key.startswith("U_B_median")}
    _, gating = gated_map(ragged_unit_maps(arrays, "U_B"), ragged_unit_maps(arrays, "G_B"))
    write_npz_atomic(partial / "predictions.npz", held_out_rows=split["held_out"],
                     **{f"held_out_{arm}": found["logits"]["held_out"] for arm, found in arms.items()})
    return {"split": {key: int(len(value)) for key, value in split.items()},
            "validation_rows": int(split["validation"].sum()), "detection": statistics,
            "seeds": seed_statistics(sets, arms), "mlp_concat_minus_logistic_concat": contrast,
            "gating_structure": gating, "experiment042_receipt": receipt,
            "note": "smoke: training rows only; maps checked for structure, not scored"}


def run_stage1(data: dict[str, Any], partial: Path) -> dict[str, Any]:
    """
    Run Stage 1 in full: comparator reproduction, the four detection arms and the gated beat-wave map.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    partial : Path
        Output folder.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    began = time.perf_counter()
    integrity: dict[str, Any] = {"training_rows_vs_037": check_against_037(data)}
    LOG.info("rows equal 037's: %s", integrity["training_rows_vs_037"])
    reference, integrity["comparator"] = comparator(data)
    LOG.info("R reproduces 037: %s", integrity["comparator"])
    sets = evaluation_sets(data)
    integrity["R_auroc"] = check_comparator_aurocs(sets, reference)
    LOG.info("R AUROC: %s", integrity["R_auroc"])
    validation, split_counts = shared_validation(data)
    LOG.info("validation split: %s", split_counts)
    y, weights = data["design"]["y"], data["design"]["weights"]
    names = ("xecg", "jepa", "concat")
    train_x = {name: binary_features(data, name, "train") for name in names}
    scored = {name: {part: binary_features(data, name, part) for part in ("development", "sph")}
              for name in names}
    arms = stage1_arms(train_x, y, weights, validation, scored, MLP_RECIPE)
    statistics = detection_statistics(sets, {arm: found["logits"] for arm, found in arms.items()}, reference,
                                      DRAWS, SEED)
    contrast = paired_contrast(sets, arms["mlp_concat"]["logits"], arms["logistic_concat"]["logits"],
                               DRAWS, SEED)
    LOG.info("detection done at %.1f s", time.perf_counter() - began)

    inputs = map_inputs()
    gated, gating = gated_map(inputs["maps"]["U_B"], inputs["maps"]["G_B"])
    metrics = map_metrics(inputs["rows"], {"U_B": inputs["maps"]["U_B"], "U_B_gated": gated},
                          inputs["windows"], DRAWS, SEED)
    integrity["U_B_vs_042"] = check_042_points(metrics["maps"]["U_B"], metrics["thresholds"]["U_B"])
    contrasts = map_contrasts(metrics, "U_B_gated", "U_B", DRAWS, SEED)
    LOG.info("maps done at %.1f s: %s", time.perf_counter() - began, contrasts["reading"])

    arrays = {"development_record_ids": data["development"].index.to_numpy(dtype=str),
              "sph_ecg_ids": data["sph"]["ecg_id"].to_numpy(dtype=str), "validation_mask": validation,
              **{f"{part}_R": reference[part] for part in ("development", "sph")}}
    for arm, found in arms.items():
        for part in ("development", "sph"):
            arrays[f"{part}_{arm}"] = found["logits"][part]
            for seed in found.get("seed_logits", {}):
                arrays[f"{part}_{arm}_seed{seed}"] = found["seed_logits"][seed][part]
    write_npz_atomic(partial / "predictions.npz", **arrays)
    arm_info = {arm: {key: value for key, value in found.items() if key not in ("logits", "seed_logits")}
                for arm, found in arms.items()}
    return {"integrity": integrity, "validation_split": split_counts,
            "evaluation_sets": {name: {"records": len(spec["y"]), "positives": int(spec["y"].sum()),
                                       "patients": int(len(np.unique(spec["patients"])))}
                                for name, spec in sets.items()},
            "detection": statistics, "seeds": seed_statistics(sets, arms),
            "mlp_concat_minus_logistic_concat": contrast, "readings": readings(statistics), "arms": arm_info,
            "maps": {**public_map_metrics(metrics), "gating": gating, "pvc_exclusions": inputs["exclusions"],
                     "U_B_gated_minus_U_B": contrasts},
            "map_reading": contrasts["reading"],
            "notebook": bool(any(value == "beats" for value in readings(statistics).values())
                             or contrasts["reading"]["improves_on_U_B"]),
            "inputs_042": inputs["receipt"]}


def extraction_table(data: dict[str, Any], sets: dict[str, dict[str, Any]]
                     ) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[int]]]:
    """
    Rows of the Stage 2 caches: the binary training rows, all development rows and the labeled SPH rows.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, dict[str, list[int]]]
        Rows (``kind``, ``key`` and the reader fields), their cached JEPA features, and the
        ``[start, stop)`` rows of each part.
    """
    development = data["development"]
    labeled = data["sph"].iloc[sets["sph"]["positions"]]
    frames = [data["stacked"],
              pd.DataFrame({"kind": "ptbxl", "key": development.index.to_numpy(dtype=str),
                            "filename_hr": development["filename_hr"].to_numpy(dtype=str)}),
              pd.DataFrame({"kind": "sph", "key": ("sph:" + labeled["ecg_id"]).to_numpy(dtype=str),
                            "ecg_id": labeled["ecg_id"].to_numpy(dtype=str),
                            "signal_sha256": labeled["signal_sha256"].to_numpy(dtype=str)})]
    table = pd.concat(frames, ignore_index=True)
    reference = np.concatenate([binary_features(data, "jepa", "train"), data["development_x"]["jepa"],
                                data["sph_x"]["jepa"][sets["sph"]["positions"]]])
    sizes = np.cumsum([0, *(len(frame) for frame in frames)])
    parts = {name: [int(sizes[k]), int(sizes[k + 1])]
             for k, name in enumerate(("train", "development", "sph"))}
    if table["key"].duplicated().any() or len(reference) != len(table):
        raise ValueError("Cache rows are not unique or do not match their features")
    return table, reference, parts


def read_inputs(row: dict[str, Any], checksums: dict[str, dict[str, str]], ningbo: dict[str, str]
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one cache row's canonical 500 Hz window and its ECG-JEPA input, as the cached features were made.

    Parameters
    ----------
    row : dict[str, Any]
        Row of ``extraction_table``.
    checksums : dict[str, dict[str, str]]
        Official Challenge checksums per source.
    ningbo : dict[str, str]
        Ningbo manifest window hash per record.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float32 ``[12, 5000]`` window and float32 ``[8, 2500]`` input.

    Raises
    ------
    ValueError
        If a window moved or differs from its manifest hash.
    """
    if row["kind"] == "ptbxl":
        return read_ptb_float64(row["filename_hr"]).astype(np.float32), ptb_jepa_input(row["filename_hr"])
    if row["kind"] == "challenge":
        source = row["source"]
        signal, sampling_rate, names = read_verified(RAW_ROOTS[source], row["path"], checksums[source])
        start = window_start(len(signal))
        if skip_reasons(sampling_rate, signal) or start != row["window_start"]:
            raise ValueError(f"Challenge window differs from its feature extraction: {row['key']}")
        window = canonical_window(signal, names, start).astype(np.float32)
        if source == "ningbo":
            expected = ningbo[row["key"].split(":", 1)[1]]
            if row["window_sha256"] != expected or signal_sha256(window) != expected:
                raise ValueError(f"Ningbo window differs from the manifest: {row['key']}")
        return window, jepa_input(window)
    window = sph.read_window(row["ecg_id"])
    if hashlib.sha256(window.tobytes()).hexdigest() != row["signal_sha256"]:
        raise ValueError(f"SPH window differs from the manifest: {row['key']}")
    return window, jepa_input(window)


def read_chunk(rows: list[dict[str, Any]], checksums: dict[str, dict[str, str]], ningbo: dict[str, str]
               ) -> tuple[np.ndarray, np.ndarray]:
    """Read the windows and ECG-JEPA inputs of several rows in parallel, in order."""
    with ThreadPoolExecutor(READER_THREADS) as pool:
        found = list(pool.map(lambda row: read_inputs(row, checksums, ningbo), rows))
    return np.stack([window for window, _ in found]), np.stack([inputs for _, inputs in found])


def profile_extraction(table: pd.DataFrame, encoder: torch.nn.Module, checksums: dict[str, dict[str, str]],
                       ningbo: dict[str, str]) -> dict[str, float]:
    """
    Time reading and encoding ``PROFILE_RECORDS`` rows spread over the cache rows.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``extraction_table``.
    encoder : torch.nn.Module
        ECG-JEPA encoder on the GPU.
    checksums : dict[str, dict[str, str]]
        Official Challenge checksums.
    ningbo : dict[str, str]
        Ningbo manifest window hashes.

    Returns
    -------
    dict[str, float]
        Seconds per record for reading and for the model.
    """
    picked = np.unique(np.linspace(0, len(table) - 1, min(PROFILE_RECORDS, len(table))).astype(np.int64))
    began = time.perf_counter()
    _, inputs = read_chunk(table.iloc[picked].to_dict("records"), checksums, ningbo)
    read = time.perf_counter() - began
    began = time.perf_counter()
    jepa_tokens(encoder, inputs)
    torch.cuda.synchronize()
    return {"records": len(picked), "read_per_record": read / len(picked),
            "model_per_record": (time.perf_counter() - began) / len(picked)}


def data_sha256(path: Path) -> str:
    """Return the SHA-256 of the data bytes of a ``.npy`` file (after its header)."""
    store = open_row_file(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(store.offset)
        while block := handle.read(1 << 26):
            digest.update(block)
    os.close(store.descriptor)
    return digest.hexdigest()


def cache_receipt(folder: Path, table: pd.DataFrame, verify: bool) -> dict[str, Any] | None:
    """
    Return a completed cache's receipt if its rows equal ``table``'s and (optionally) its data hashes hold.

    Parameters
    ----------
    folder : Path
        Cache folder.
    table : pd.DataFrame
        Expected rows.
    verify : bool
        Rehash both caches.

    Returns
    -------
    dict[str, Any] | None
        The receipt, or ``None`` when there is no completed cache of these rows.

    Raises
    ------
    ValueError
        If a completed cache of these rows fails its hash check.
    """
    path = folder / "receipt.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if not receipt.get("complete") or receipt["keys_sha256"] != sha256_json(table["key"].tolist()):
        return None
    if verify:
        for name in ("tokens", "windows"):
            if data_sha256(folder / f"{name}.npy") != receipt[name]["data_sha256"]:
                raise ValueError(f"Cache {name}.npy differs from its receipt")
    return receipt


def extract_cache(folder: Path, table: pd.DataFrame, reference: np.ndarray, parts: dict[str, list[int]],
                  encoder: torch.nn.Module, checksums: dict[str, dict[str, str]], ningbo: dict[str, str]
                  ) -> dict[str, Any]:
    """
    Write the float16 token cache and the float32 window cache, checking every token mean first.

    Parameters
    ----------
    folder : Path
        Cache folder (local disk).
    table : pd.DataFrame
        Output of ``extraction_table``.
    reference : np.ndarray
        Cached JEPA features of the rows.
    parts : dict[str, list[int]]
        Part boundaries.
    encoder : torch.nn.Module
        ECG-JEPA encoder on the GPU.
    checksums : dict[str, dict[str, str]]
        Official Challenge checksums.
    ningbo : dict[str, str]
        Ningbo manifest window hashes.

    Returns
    -------
    dict[str, Any]
        The cache receipt.

    Raises
    ------
    ValueError
        If a token mean differs from its cached feature by more than ``FEATURE_TOLERANCE``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "receipt.json").unlink(missing_ok=True)
    table[["key"]].to_csv(folder / "rows.csv", index=False)
    stores = {"tokens": create_row_file(folder / "tokens.npy", (len(table), *TOKEN_SHAPE), np.float16),
              "windows": create_row_file(folder / "windows.npy", (len(table), *WINDOW_SHAPE), np.float32)}
    digests = {name: hashlib.sha256() for name in stores}
    seconds = {"read_wait": 0.0, "model": 0.0, "check_and_write": 0.0}
    worst = 0.0
    records = table.to_dict("records")
    began_all = time.perf_counter()
    prefetch = ThreadPoolExecutor(1)
    pending = prefetch.submit(read_chunk, records[:EXTRACT_CHUNK], checksums, ningbo)
    for start in range(0, len(records), EXTRACT_CHUNK):
        began = time.perf_counter()
        windows, inputs = pending.result()
        if start + EXTRACT_CHUNK < len(records):
            pending = prefetch.submit(read_chunk, records[start + EXTRACT_CHUNK:start + 2 * EXTRACT_CHUNK],
                                      checksums, ningbo)
        seconds["read_wait"] += time.perf_counter() - began
        began = time.perf_counter()
        tokens = jepa_tokens(encoder, inputs)
        seconds["model"] += time.perf_counter() - began
        began = time.perf_counter()
        difference = float(np.abs(tokens.mean(axis=1, dtype=np.float64)
                                  - reference[start:start + len(tokens)]).max())
        worst = max(worst, difference)
        if difference > FEATURE_TOLERANCE:
            raise ValueError(f"Token mean differs from the cached feature by {difference} at row {start}")
        stored = tokens.astype(np.float16)
        if not np.isfinite(stored).all():
            raise ValueError(f"Tokens overflow float16 at row {start}")
        for name, values in (("tokens", stored), ("windows", windows.astype(np.float32))):
            write_rows(stores[name], start, values)
            digests[name].update(np.ascontiguousarray(values).tobytes())
        seconds["check_and_write"] += time.perf_counter() - began
        if (start // EXTRACT_CHUNK) % 20 == 0:
            LOG.info("extracted %d / %d at %.1f s", start + len(tokens), len(records),
                     time.perf_counter() - began_all)
    prefetch.shutdown()
    for store in stores.values():
        os.fsync(store.descriptor)
        os.close(store.descriptor)
    receipt = {"complete": True, "rows": len(table), "keys_sha256": sha256_json(table["key"].tolist()),
               "parts": parts, "token_check": {"rows_compared": len(table), "max_abs_difference": worst,
                                               "tolerance": FEATURE_TOLERANCE},
               "tokens": {"shape": [len(table), *TOKEN_SHAPE], "dtype": "float16",
                          "data_sha256": digests["tokens"].hexdigest()},
               "windows": {"shape": [len(table), *WINDOW_SHAPE], "dtype": "float32",
                           "data_sha256": digests["windows"].hexdigest()},
               "seconds": seconds, "rows_csv_sha256": sha256_file(folder / "rows.csv")}
    write_json_atomic(folder / "receipt.json", receipt)
    return receipt


def row_batch(store: RowFile, device: str) -> Any:
    """Return a batch function reading cache rows as float32 tensors on the device."""
    return lambda rows: torch.from_numpy(read_rows(store, rows)).to(device).float()


def array_batch(values: np.ndarray, device: str) -> Any:
    """Return a batch function taking rows of an in-memory array as tensors on the device."""
    return lambda rows: torch.from_numpy(values[np.asarray(rows)]).to(device)


def profile_training(build: Callable[[], torch.nn.Module], batch: Any, fit_rows: np.ndarray,
                     fit_y: np.ndarray, fit_weights: np.ndarray, recipe: Recipe, device: str,
                     augmentation: Callable[[torch.Tensor, torch.Generator], torch.Tensor] | None = None,
                     predict_batch: int = PREDICT_BATCH) -> dict[str, float]:
    """
    Time ``PROFILE_STEPS`` optimizer steps of a throwaway network (after warm-up) and one scoring pass.

    Parameters
    ----------
    build : Callable[[], torch.nn.Module]
        Builds the network on the CPU.
    batch : Any
        Maps rows to an input tensor on the device.
    fit_rows : np.ndarray
        Rows of the training part.
    fit_y : np.ndarray
        Their targets.
    fit_weights : np.ndarray
        Their weights.
    recipe : Recipe
        Training recipe.
    device : str
        Torch device.
    augmentation : Callable | None
        Applied to each training batch.
    predict_batch : int
        Rows per scoring pass.

    Returns
    -------
    dict[str, float]
        Seconds per optimizer step and per scored row, and the GPU memory peak in GiB.
    """
    torch.manual_seed(0)
    model = build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    generator = torch.Generator(device=device).manual_seed(0)
    rng = np.random.default_rng(0)
    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP_STEPS + PROFILE_STEPS):
        if step == WARMUP_STEPS:
            torch.cuda.synchronize()
            began = time.perf_counter()
        chosen = rng.choice(len(fit_rows), recipe.batch_size, replace=False)
        inputs = batch(fit_rows[chosen])
        if augmentation is not None:
            inputs = augmentation(inputs, generator)
        logit, _ = model(inputs)
        target = torch.as_tensor(fit_y[chosen], dtype=torch.float32, device=device)
        weight = torch.as_tensor(fit_weights[chosen], dtype=torch.float32, device=device)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
                * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if recipe.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
        optimizer.step()
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - began) / PROFILE_STEPS
    scored = fit_rows[:min(1024, len(fit_rows))]
    began = time.perf_counter()
    predict(model, batch, scored, predict_batch, keep_contributions=True)
    torch.cuda.synchronize()
    return {"step": per_step, "scored_row": (time.perf_counter() - began) / len(scored),
            "warmup_steps": WARMUP_STEPS, "timed_steps": PROFILE_STEPS,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}


def projected_training(profile: dict[str, float], fits: int, checks: int, scored: int,
                       recipe: Recipe) -> float:
    """Return the worst-case training seconds of all seeds: every epoch run, then scoring."""
    epoch = -(-fits // recipe.batch_size) * profile["step"] + checks * profile["scored_row"]
    return len(SEEDS) * (recipe.max_epochs * epoch + scored * profile["scored_row"])


def train_seeds(build: Callable[[], torch.nn.Module], batch: Any, fit_rows: np.ndarray, fit_y: np.ndarray,
                fit_weights: np.ndarray, check_rows: np.ndarray, check_y: np.ndarray,
                score_rows: dict[str, np.ndarray], recipe: Recipe, device: str, maps: tuple[str, ...],
                name: str,
                augmentation: Callable[[torch.Tensor, torch.Generator], torch.Tensor] | None = None,
                predict_batch: int = PREDICT_BATCH) -> dict[str, Any]:
    """
    Train a network with every seed and score the scored parts.

    Parameters
    ----------
    build : Callable[[], torch.nn.Module]
        Builds the network on the CPU (after the seed is set).
    batch : Any
        Maps rows to an input tensor on the device.
    fit_rows, check_rows : np.ndarray
        Rows of the training and validation parts.
    fit_y, fit_weights : np.ndarray
        Targets and weights of ``fit_rows``.
    check_y : np.ndarray
        Targets of ``check_rows``.
    score_rows : dict[str, np.ndarray]
        Rows per scored part.
    recipe : Recipe
        Training recipe.
    device : str
        Torch device.
    maps : tuple[str, ...]
        Parts whose per-token contributions are kept.
    name : str
        Arm name for the log.
    augmentation : Callable | None
        Applied to each training batch.
    predict_batch : int
        Rows per scoring pass.

    Returns
    -------
    dict[str, Any]
        ``logits`` (seed mean per part), ``seed_logits``, ``contributions`` (seed mean per kept part),
        ``seeds``, the largest contribution-sum check difference and the GPU memory peak.

    Raises
    ------
    ValueError
        If the contributions do not sum to the logit minus the bias.
    """
    seed_logits: dict[int, dict[str, np.ndarray]] = {}
    seed_contributions: dict[int, dict[str, np.ndarray]] = {}
    seeds, worst = {}, 0.0
    torch.cuda.reset_peak_memory_stats()
    for seed in SEEDS:
        began = time.perf_counter()
        torch.manual_seed(seed)
        model = build().to(device)
        fit = train(model, batch, fit_rows, fit_y, fit_weights, check_rows, check_y, recipe, seed,
                    augmentation=augmentation, log=lambda line: LOG.info("%s %s", name, line))
        bias = float(getattr(model, "head", model).bias)
        seed_logits[seed], seed_contributions[seed] = {}, {}
        for part, rows in score_rows.items():
            logits, units = predict(model, batch, rows, predict_batch, keep_contributions=True)
            worst = max(worst, float(np.abs(units.sum(axis=1, dtype=np.float64) + bias - logits).max()))
            seed_logits[seed][part] = logits
            if part in maps:
                seed_contributions[seed][part] = units
        seeds[str(seed)] = {"best_epoch": fit.best_epoch, "validation_auroc": fit.best_auroc,
                            "epochs_run": len(fit.history), "seconds": time.perf_counter() - began,
                            "history": fit.history}
        LOG.info("seed %d best epoch %d validation AUROC %.4f", seed, fit.best_epoch, fit.best_auroc)
        del model
    if worst > CONTRIBUTION_TOLERANCE:
        raise ValueError(f"Token contributions do not sum to the logit: {worst}")
    return {"logits": {part: np.mean([seed_logits[s][part] for s in SEEDS], axis=0) for part in score_rows},
            "seed_logits": seed_logits, "seed_contributions": seed_contributions,
            "contributions": {part: np.mean([seed_contributions[s][part] for s in SEEDS], axis=0)
                              for part in maps},
            "seeds": seeds, "contribution_max_abs_difference": worst,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}


def full_length(values: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
    """Return a NaN vector of ``length`` with ``values`` at ``positions``."""
    found = np.full(length, np.nan)
    found[positions] = values
    return found


def stage1_predictions(reference: dict[str, np.ndarray], validation: np.ndarray) -> dict[str, Any]:
    """
    Stage 1's saved logistic logits, after checking its receipt, validation split and R.

    Parameters
    ----------
    reference : dict[str, np.ndarray]
        R's refitted logits per part.
    validation : np.ndarray
        The shared validation mask.

    Returns
    -------
    dict[str, Any]
        ``logistic_jepa`` and ``logistic_concat`` logits per part, and the receipt hashes.

    Raises
    ------
    ValueError
        If the split or R differs from Stage 1's.
    """
    receipt = receipt_checked(PRIOR_STAGE1, ("predictions.npz",))
    with np.load(PRIOR_STAGE1 / "predictions.npz") as saved:
        if not np.array_equal(saved["validation_mask"], validation):
            raise ValueError("Validation split differs from Stage 1's")
        difference = max(float(np.abs(saved[f"{part}_R"] - reference[part]).max()) for part in reference)
        if difference > AUROC_TOLERANCE:
            raise ValueError(f"R differs from Stage 1's by {difference}")
        found = {arm: {part: saved[f"{part}_{arm}"] for part in ("development", "sph")}
                 for arm in ("logistic_jepa", "logistic_concat")}
    return {**found, "receipt": receipt, "R_max_abs_difference_vs_stage1": difference}


def stage2_predictions(data: dict[str, Any], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Stage 2's saved ``attention_jepa`` logits on 037's parts, after checking its receipt and rows.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        ``attention_jepa`` logits per part (SPH at full length), the receipt and Stage 2's cache receipt.

    Raises
    ------
    ValueError
        If Stage 2's rows differ.
    """
    receipt = receipt_checked(PRIOR_STAGE2, ("predictions.npz",))
    positions = sets["sph"]["positions"]
    with np.load(PRIOR_STAGE2 / "predictions.npz") as saved:
        if not (np.array_equal(saved["sph_ecg_ids"], data["sph"]["ecg_id"].to_numpy(dtype=str)[positions])
                and np.array_equal(saved["development_record_ids"],
                                   data["development"].index.to_numpy(dtype=str))):
            raise ValueError("Stage 2 prediction rows differ")
        attention = {"development": saved["development_attention_jepa"],
                     "sph": full_length(saved["sph_attention_jepa"], positions, len(data["sph"]))}
    cache = json.loads((PRIOR_STAGE2 / "result.json").read_text())["cache"]["receipt"]
    return {"attention_jepa": attention, "receipt": receipt, "cache_receipt": cache}


def token_smoke_split(data: dict[str, Any]) -> dict[str, Any]:
    """
    Pick the smoke cache's training rows and split them into fit, validation and held-out rows by group.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    dict[str, Any]
        ``chosen`` binary rows, cache ``table`` and features, part bounds, and cache-row positions of the
        ``fit``, ``check`` and ``held`` rows with targets and weights.
    """
    y, weights = data["design"]["y"], data["design"]["weights"]
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(data["groups"]), SMOKE_TOKEN_ROWS, replace=False))
    held = validation_mask(data["groups"][chosen], SMOKE_HELD_OUT, SEED + 1)
    pool = np.flatnonzero(~held)
    validation = validation_mask(data["groups"][chosen][pool], VALIDATION_SHARE, VALIDATION_SEED)
    fit_rows, check_rows, held_rows = pool[~validation], pool[validation], np.flatnonzero(held)
    return {"chosen": chosen, "table": data["stacked"].iloc[chosen].reset_index(drop=True),
            "reference": binary_features(data, "jepa", "train")[chosen], "parts": {"train": [0, len(chosen)]},
            "pool": pool, "fit_rows": fit_rows, "check_rows": check_rows, "held_rows": held_rows,
            "fit_y": y[chosen][fit_rows], "fit_w": weights[chosen][fit_rows],
            "check_y": y[chosen][check_rows]}


def smoke_reference(data: dict[str, Any], split: dict[str, Any]) -> tuple[dict, dict[str, np.ndarray]]:
    """
    Build the smoke held-out set and fit a smoke R on the other smoke rows.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    split : dict[str, Any]
        Output of ``token_smoke_split``.

    Returns
    -------
    tuple[dict, dict[str, np.ndarray]]
        The ``held_out`` evaluation set and R's logits on it.
    """
    y, weights = data["design"]["y"], data["design"]["weights"]
    chosen, pool, held_rows = split["chosen"], split["pool"], split["held_rows"]
    sets = {"held_out": {"part": "held_out", "positions": np.arange(len(held_rows)),
                         "y": y[chosen][held_rows], "patients": data["groups"][chosen][held_rows]}}
    x = binary_features(data, "xecg", "train")[chosen]
    with threadpool_limits(limits=1):
        head = fit_readout(x[pool], y[chosen][pool], weights[chosen][pool])
        reference = {"held_out": logits_of(head, x[held_rows])}
    return sets, reference


def full_integrity(data: dict[str, Any]) -> dict[str, Any]:
    """
    Check the rows against 037, reproduce R and its AUROCs, draw the shared split and check Stage 1.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    dict[str, Any]
        ``integrity`` checks, ``reference`` (R's logits), ``sets``, ``validation``, ``split_counts``,
        ``stage1`` predictions, and 042's map inputs.

    Raises
    ------
    ValueError
        If 042's scored rows differ from the development rows.
    """
    integrity: dict[str, Any] = {"training_rows_vs_037": check_against_037(data)}
    reference, integrity["comparator"] = comparator(data)
    sets = evaluation_sets(data)
    integrity["R_auroc"] = check_comparator_aurocs(sets, reference)
    validation, split_counts = shared_validation(data)
    prior = stage1_predictions(reference, validation)
    integrity["stage1"] = {"receipt": prior["receipt"],
                           "R_max_abs_difference": prior["R_max_abs_difference_vs_stage1"]}
    inputs042 = map_inputs()
    if not np.array_equal(inputs042["rows"]["scored"]["ecg_id"].to_numpy(np.int64),
                          data["development"]["ecg_id"].to_numpy(np.int64)):
        raise ValueError("042's scored rows differ from the development rows")
    LOG.info("integrity: %s", integrity)
    return {"integrity": integrity, "reference": reference, "sets": sets, "validation": validation,
            "split_counts": split_counts, "stage1": prior, "inputs042": inputs042}


def evaluate_network(name: str, arm: dict[str, Any], data: dict[str, Any], checks: dict[str, Any],
                     unit_map: Callable[[np.ndarray], UnitMap], secondary: dict[str, dict[str, np.ndarray]],
                     partial: Path) -> dict[str, Any]:
    """
    Compute one network's detection statistics, contrasts, map metrics and map contrasts; save its outputs.

    Parameters
    ----------
    name : str
        Arm name.
    arm : dict[str, Any]
        Output of ``train_seeds`` on the ``development`` and ``sph`` cache parts.
    data : dict[str, Any]
        Output of ``training_data``.
    checks : dict[str, Any]
        Output of ``full_integrity``.
    unit_map : Callable[[np.ndarray], UnitMap]
        Maps one ECG's contributions to units.
    secondary : dict[str, dict[str, np.ndarray]]
        Arms to contrast with, logits per part.
    partial : Path
        Output folder for ``predictions.npz`` and ``token_maps.npz``.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    began = time.perf_counter()
    sets, inputs042 = checks["sets"], checks["inputs042"]
    sph_positions, sph_length = sets["sph"]["positions"], len(data["sph"])
    scores = {"development": arm["logits"]["development"],
              "sph": full_length(arm["logits"]["sph"], sph_positions, sph_length)}
    arm_for_seeds = {"seeds": arm["seeds"], "seed_logits": {
        seed: {"development": found["development"],
               "sph": full_length(found["sph"], sph_positions, sph_length)}
        for seed, found in arm["seed_logits"].items()}}
    statistics = detection_statistics(sets, {name: scores}, checks["reference"], DRAWS, SEED)
    contrasts_detection = {f"{name}_minus_{other}": paired_contrast(sets, scores, values, DRAWS, SEED)
                           for other, values in secondary.items()}
    LOG.info("detection done in %.1f s", time.perf_counter() - began)
    maps = [unit_map(values) for values in arm["contributions"]["development"]]
    metrics = map_metrics(inputs042["rows"], {"U_B": inputs042["maps"]["U_B"], name: maps},
                          inputs042["windows"], DRAWS, SEED)
    integrity = {"U_B_vs_042": check_042_points(metrics["maps"]["U_B"], metrics["thresholds"]["U_B"])}
    contrasts = map_contrasts(metrics, name, "U_B", DRAWS, SEED)
    LOG.info("maps done in %.1f s: %s", time.perf_counter() - began, contrasts["reading"])
    sph_ids = data["sph"]["ecg_id"].to_numpy(dtype=str)[sph_positions]
    arrays = {"development_record_ids": data["development"].index.to_numpy(dtype=str), "sph_ecg_ids": sph_ids,
              f"development_{name}": arm["logits"]["development"], f"sph_{name}": arm["logits"]["sph"]}
    for seed, found in arm["seed_logits"].items():
        arrays.update({f"{part}_{name}_seed{seed}": values for part, values in found.items()})
    write_npz_atomic(partial / "predictions.npz", **arrays)
    write_npz_atomic(partial / "token_maps.npz", ecg_ids=data["development"]["ecg_id"].to_numpy(np.int64),
                     **{name: arm["contributions"]["development"].astype(np.float32)},
                     **{f"{name}_seed{seed}": found["development"].astype(np.float32)
                        for seed, found in arm["seed_contributions"].items()})
    reading = readings(statistics)
    return {"integrity_maps": integrity, "validation_split": checks["split_counts"],
            "evaluation_sets": {key: {"records": len(spec["y"]), "positives": int(spec["y"].sum()),
                                      "patients": int(len(np.unique(spec["patients"])))}
                                for key, spec in sets.items()},
            "detection": statistics, "seeds": seed_statistics(sets, {name: arm_for_seeds}),
            **contrasts_detection, "readings": reading, "arm": {"seeds": arm["seeds"]},
            "maps": {**public_map_metrics(metrics), "pvc_exclusions": inputs042["exclusions"],
                     f"{name}_minus_U_B": contrasts},
            "map_reading": contrasts["reading"],
            "notebook": bool(reading[name] == "beats" or contrasts["reading"]["improves_on_U_B"]),
            "inputs_042": inputs042["receipt"]}


def run_stage2(data: dict[str, Any], partial: Path, smoke: bool, cache: Path) -> dict[str, Any]:
    """
    Run Stage 2: cache ECG-JEPA tokens and windows, train the attention head, and evaluate it and its map.

    In smoke mode only a few hundred training rows are cached and scored, and the map is checked for
    structure only.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    partial : Path
        Output folder.
    smoke : bool
        Training-only smoke test.
    cache : Path
        Cache folder on local disk.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    began = time.perf_counter()
    integrity: dict[str, Any] = {}
    y, weights = data["design"]["y"], data["design"]["weights"]
    if smoke:
        split = token_smoke_split(data)
        table, reference, parts = split["table"], split["reference"], split["parts"]
        fit_rows, check_rows = split["fit_rows"], split["check_rows"]
        fit_y, fit_w, check_y = split["fit_y"], split["fit_w"], split["check_y"]
        score_rows = {"held_out": split["held_rows"]}
        recipe = Recipe(ATTENTION_RECIPE.learning_rate, ATTENTION_RECIPE.weight_decay,
                        ATTENTION_RECIPE.batch_size, SMOKE_EPOCHS, ATTENTION_RECIPE.patience)
        maps_parts: tuple[str, ...] = ("held_out",)
    else:
        checks = full_integrity(data)
        integrity = checks["integrity"]
        table, reference, parts = extraction_table(data, checks["sets"])
        validation = checks["validation"]
        fit_rows, check_rows = np.flatnonzero(~validation), np.flatnonzero(validation)
        fit_y, fit_w, check_y = y[fit_rows], weights[fit_rows], y[check_rows]
        score_rows = {part: np.arange(*parts[part]) for part in ("development", "sph")}
        recipe = ATTENTION_RECIPE
        maps_parts = ("development",)
    LOG.info("cache rows %s", parts)
    checksums = load_checksums()
    ningbo = {record: window for record, _, window in ningbo_items()[0]}
    device = "cuda"
    build = partial_build(AttentionHead, TOKEN_SHAPE[1])
    with gpu_lock(device):
        receipt = cache_receipt(cache, table, verify=True)
        profile: dict[str, Any] = {"extraction": None, "cache_reused": receipt is not None}
        if receipt is None:
            encoder = load_jepa()
            profile["extraction"] = profile_extraction(table, encoder, checksums, ningbo)
            per_record = profile["extraction"]["read_per_record"] + profile["extraction"]["model_per_record"]
            projected = time.perf_counter() - began + per_record * len(table)
            LOG.info("extraction profile %s, projected %.0f s", profile["extraction"], projected)
            if projected > STAGE2_CEILING:
                raise RuntimeError(f"Projected extraction {projected:.0f} s exceeds the stage ceiling")
            receipt = extract_cache(cache, table, reference, parts, encoder, checksums, ningbo)
            del encoder
            torch.cuda.empty_cache()
        LOG.info("cache ready at %.1f s: %s", time.perf_counter() - began, receipt["token_check"])
        store = open_row_file(cache / "tokens.npy")
        batch = row_batch(store, device)
        profile["training"] = profile_training(build, batch, fit_rows, fit_y, fit_w, recipe, device)
        scored = sum(len(rows) for rows in score_rows.values())
        profile["projected_training_seconds"] = projected_training(profile["training"], len(fit_rows),
                                                                   len(check_rows), scored, recipe)
        profile["projected_stage_seconds"] = (time.perf_counter() - began
                                              + profile["projected_training_seconds"])
        LOG.info("training profile %s", profile)
        if profile["projected_stage_seconds"] > STAGE2_CEILING:
            raise RuntimeError(f"Projected stage time {profile['projected_stage_seconds']:.0f} s "
                               "exceeds 2 hours")
        arm = train_seeds(build, batch, fit_rows, fit_y, fit_w, check_rows, check_y, score_rows, recipe,
                          device, maps_parts, "attention_jepa")
        torch.cuda.empty_cache()
    os.close(store.descriptor)
    integrity["contribution_max_abs_difference"] = arm["contribution_max_abs_difference"]
    LOG.info("training done at %.1f s", time.perf_counter() - began)
    cache_fields = {"folder": str(cache), "receipt": receipt, "rows_csv": "cache_rows.csv"}
    table[["kind", "key"]].to_csv(partial / "cache_rows.csv", index=False)
    if smoke:
        sets, reference_r = smoke_reference(data, split)
        statistics = detection_statistics(sets, {"attention_jepa": arm["logits"]}, reference_r, SMOKE_DRAWS,
                                          SEED, primary=("held_out",))
        unit_map = jepa_unit_map(arm["contributions"]["held_out"][0])
        write_npz_atomic(partial / "predictions.npz", held_out_rows=split["held_rows"],
                         held_out_attention_jepa=arm["logits"]["held_out"])
        return {"integrity": integrity, "profile": profile, "cache": cache_fields, "detection": statistics,
                "seeds": seed_statistics(sets, {"attention_jepa": arm}),
                "map_structure": {"contributions": list(arm["contributions"]["held_out"].shape),
                                  "units": len(unit_map.scores),
                                  "leads": sorted(set(unit_map.leads.tolist()))},
                "recipe": vars(recipe),
                "note": "smoke: training rows only; map checked for structure, not scored"}
    found = evaluate_network("attention_jepa", arm, data, checks, jepa_unit_map,
                             {"logistic_jepa": checks["stage1"]["logistic_jepa"]}, partial)
    integrity.update(found.pop("integrity_maps"))
    return {"integrity": integrity, "profile": profile, "cache": cache_fields, "recipe": vars(recipe),
            **found}


def partial_build(network: Callable[..., torch.nn.Module], *arguments: Any) -> Callable[[], torch.nn.Module]:
    """Return a function building the network with the given arguments."""
    return lambda: network(*arguments)


def preprocess_windows(path: Path, rows: np.ndarray, fit_rows: np.ndarray
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Decimate cached 500 Hz windows to 250 Hz, subtract each lead's median and divide by the training SD.

    The SD of each lead is unweighted, over every sample of the median-subtracted training-part records,
    around their pooled mean.

    Parameters
    ----------
    path : Path
        Window cache (``[rows, 12, 5000]`` float32).
    rows : np.ndarray
        Cache rows to preprocess, in output order.
    fit_rows : np.ndarray
        Output positions of the training part.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Float32 ``[len(rows), 12, 2500]`` inputs, and the lead SDs and means.
    """
    store = open_row_file(path)
    inputs = np.empty((len(rows), WINDOW_SHAPE[0], STAGE3_SAMPLES), dtype=np.float32)
    for start in range(0, len(rows), PREPROCESS_CHUNK):
        windows = read_rows(store, rows[start:start + PREPROCESS_CHUNK]).astype(np.float64)
        decimated = resample_poly(windows, 1, 2, axis=2)
        inputs[start:start + len(windows)] = decimated - np.median(decimated, axis=2, keepdims=True)
    os.close(store.descriptor)
    if inputs.shape[2] != STAGE3_SAMPLES:
        raise ValueError(f"Decimated windows have {inputs.shape[2]} samples")
    total = np.zeros(WINDOW_SHAPE[0])
    squares = np.zeros(WINDOW_SHAPE[0])
    for start in range(0, len(fit_rows), PREPROCESS_CHUNK):
        part = inputs[fit_rows[start:start + PREPROCESS_CHUNK]].astype(np.float64)
        total += part.sum(axis=(0, 2))
        squares += np.square(part).sum(axis=(0, 2))
    count = len(fit_rows) * STAGE3_SAMPLES
    mean = total / count
    scale = np.sqrt(squares / count - mean ** 2)
    for start in range(0, len(rows), PREPROCESS_CHUNK):
        inputs[start:start + PREPROCESS_CHUNK] /= scale[None, :, None].astype(np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError("Nonfinite preprocessed windows")
    return inputs, {"lead_sd": scale.tolist(), "lead_mean_after_median": mean.tolist(),
                    "sd_rows": len(fit_rows), "weighted": False}


def run_stage3(data: dict[str, Any], partial: Path, smoke: bool, cache: Path) -> dict[str, Any]:
    """
    Run Stage 3: train the tutor's CNN + transformer from scratch on Stage 2's window cache and evaluate it.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    partial : Path
        Output folder.
    smoke : bool
        Training-only smoke test (Stage 2 smoke cache rows), map checked for structure only.
    cache : Path
        Cache folder from Stage 2.

    Returns
    -------
    dict[str, Any]
        Result fields.

    Raises
    ------
    ValueError
        If the cache is missing or differs from its receipt or Stage 2's.
    """
    began = time.perf_counter()
    integrity: dict[str, Any] = {}
    y, weights = data["design"]["y"], data["design"]["weights"]
    if smoke:
        split = token_smoke_split(data)
        table, parts = split["table"], split["parts"]
        fit_rows, check_rows = split["fit_rows"], split["check_rows"]
        fit_y, fit_w, check_y = split["fit_y"], split["fit_w"], split["check_y"]
        score_rows = {"held_out": split["held_rows"]}
        recipe = Recipe(CNN_RECIPE.learning_rate, CNN_RECIPE.weight_decay, CNN_RECIPE.batch_size,
                        SMOKE_EPOCHS, CNN_RECIPE.patience, CNN_RECIPE.clip)
        maps_parts: tuple[str, ...] = ("held_out",)
    else:
        checks = full_integrity(data)
        integrity = checks["integrity"]
        prior2 = stage2_predictions(data, checks["sets"])
        integrity["stage2"] = prior2["receipt"]
        table, _, parts = extraction_table(data, checks["sets"])
        validation = checks["validation"]
        fit_rows, check_rows = np.flatnonzero(~validation), np.flatnonzero(validation)
        fit_y, fit_w, check_y = y[fit_rows], weights[fit_rows], y[check_rows]
        score_rows = {part: np.arange(*parts[part]) for part in ("development", "sph")}
        recipe = CNN_RECIPE
        maps_parts = ("development",)
    receipt = cache_receipt(cache, table, verify=False)
    if receipt is None:
        raise ValueError(f"No completed Stage 2 cache of these rows in {cache}")
    windows_hash = data_sha256(cache / "windows.npy")
    if windows_hash != receipt["windows"]["data_sha256"] or (
            not smoke and windows_hash != prior2["cache_receipt"]["windows"]["data_sha256"]):
        raise ValueError("Window cache differs from its receipt or from Stage 2's")
    integrity["windows_data_sha256"] = windows_hash
    inputs, preprocessing = preprocess_windows(cache / "windows.npy", np.arange(len(table)), fit_rows)
    LOG.info("preprocessed %s at %.1f s: %s", inputs.shape, time.perf_counter() - began, preprocessing)
    device = "cuda"
    build = partial_build(CNNTransformer)
    batch = array_batch(inputs, device)
    with gpu_lock(device):
        profile: dict[str, Any] = {"training": profile_training(
            build, batch, fit_rows, fit_y, fit_w, recipe, device, augment, STAGE3_PREDICT_BATCH)}
        scored = sum(len(rows) for rows in score_rows.values())
        profile["projected_training_seconds"] = projected_training(profile["training"], len(fit_rows),
                                                                   len(check_rows), scored, recipe)
        profile["projected_stage_seconds"] = (time.perf_counter() - began
                                              + profile["projected_training_seconds"])
        LOG.info("training profile %s", profile)
        if profile["projected_stage_seconds"] > STAGE3_CEILING:
            raise RuntimeError(f"Projected stage time {profile['projected_stage_seconds']:.0f} s "
                               "exceeds 4 hours")
        arm = train_seeds(build, batch, fit_rows, fit_y, fit_w, check_rows, check_y, score_rows, recipe,
                          device, maps_parts, "cnn_transformer", augment, STAGE3_PREDICT_BATCH)
        torch.cuda.empty_cache()
    integrity["contribution_max_abs_difference"] = arm["contribution_max_abs_difference"]
    epochs = [entry["seconds"] for info in arm["seeds"].values() for entry in info["history"]]
    timing = {"peak_allocated_gib": arm["peak_allocated_gib"], "peak_reserved_gib": arm["peak_reserved_gib"],
              "epoch_seconds_mean": float(np.mean(epochs)), "epoch_seconds_max": float(np.max(epochs)),
              "training_seconds": float(sum(info["seconds"] for info in arm["seeds"].values()))}
    LOG.info("training done at %.1f s: %s", time.perf_counter() - began, timing)
    cache_fields = {"folder": str(cache), "receipt": receipt}
    common = {"integrity": integrity, "profile": profile, "timing": timing, "cache": cache_fields,
              "preprocessing": preprocessing, "recipe": vars(recipe)}
    if smoke:
        sets, reference_r = smoke_reference(data, split)
        statistics = detection_statistics(sets, {"cnn_transformer": arm["logits"]}, reference_r, SMOKE_DRAWS,
                                          SEED, primary=("held_out",))
        unit_map = grid_unit_map(arm["contributions"]["held_out"][0], tuple(range(WINDOW_SHAPE[0])))
        write_npz_atomic(partial / "predictions.npz", held_out_rows=split["held_rows"],
                         held_out_cnn_transformer=arm["logits"]["held_out"])
        return {**common, "detection": statistics, "seeds": seed_statistics(sets, {"cnn_transformer": arm}),
                "map_structure": {"contributions": list(arm["contributions"]["held_out"].shape),
                                  "units": len(unit_map.scores),
                                  "leads": sorted(set(unit_map.leads.tolist()))},
                "note": "smoke: training rows only; map checked for structure, not scored"}
    found = evaluate_network("cnn_transformer", arm, data, checks,
                             lambda values: grid_unit_map(values, tuple(range(WINDOW_SHAPE[0]))),
                             {"attention_jepa": prior2["attention_jepa"],
                              "logistic_concat": checks["stage1"]["logistic_concat"]}, partial)
    integrity.update(found.pop("integrity_maps"))
    return {**common, **found}


def run(stage: int, smoke: bool, output: Path, cache: Path) -> None:
    """
    Run one stage end to end and write its outputs.

    Parameters
    ----------
    stage : int
        Stage number.
    smoke : bool
        Training-only smoke test.
    output : Path
        Final output folder; it must not exist.
    cache : Path
        Stage 2 and 3 cache folder on local disk.
    """
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(partial / "run.log")])
    torch.set_num_threads(THREADS)
    if stage == 1 and torch.cuda.is_available():
        raise RuntimeError("Stage 1 is CPU only: set CUDA_VISIBLE_DEVICES=")
    if stage in (2, 3) and not torch.cuda.is_available():
        raise RuntimeError(f"Stage {stage} needs a GPU")
    receipts = {to_stored(PRIOR037 / name): digest
                for name, digest in receipt_checked(PRIOR037, ("predictions.npz",)).items()}
    receipts.update({to_stored(PRIOR035 / name): digest
                     for name, digest in receipt_checked(PRIOR035, ("predictions.npz",)).items()})
    receipts.update({to_stored(PRIOR042 / name): digest
                     for name, digest in receipt_checked(PRIOR042, ("unit_scores.npz",)).items()})
    if stage in (2, 3) and not smoke:
        receipts.update({to_stored(PRIOR_STAGE1 / name): digest
                         for name, digest in receipt_checked(PRIOR_STAGE1, ("predictions.npz",)).items()})
    if stage == 3 and not smoke:
        receipts.update({to_stored(PRIOR_STAGE2 / name): digest
                         for name, digest in receipt_checked(PRIOR_STAGE2, ("predictions.npz",)).items()})
    data = training_data()
    LOG.info("loaded %s at %.1f s", data["training_counts"], time.perf_counter() - started)
    run_identity = identity(receipts, data["identity"])
    if stage == 1:
        found = run_smoke(data, partial) if smoke else run_stage1(data, partial)
        found.update({"recipe": vars(MLP_RECIPE), "hidden": MLP_HIDDEN, "dropout": MLP_DROPOUT})
    elif stage == 2:
        found = run_stage2(data, partial, smoke, cache)
    else:
        found = run_stage3(data, partial, smoke, cache)
    outputs = {path.name: sha256_file(path) for path in sorted(partial.iterdir())
               if path.is_file() and path.name not in ("result.json", "run.log")}
    write_json_atomic(partial / "result.json", {
        "status": "smoke" if smoke else "completed", "stage": stage, "identity": run_identity,
        "training_counts": data["training_counts"], **found,
        "training_seeds": list(SEEDS), "validation_share": VALIDATION_SHARE,
        "validation_seed": VALIDATION_SEED,
        "draws": SMOKE_DRAWS if smoke else DRAWS, "bootstrap_seed": SEED,
        "outputs_sha256": outputs, "total_seconds": time.perf_counter() - started,
        "calibration_test_evaluated": False, "challenge_test_read": False, "ptbxl_test_read": False,
    })
    partial.rename(output)
    LOG.info("done in %.1f s", time.perf_counter() - started)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--smoke", action="store_true", help="training-only smoke test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=CACHE, help="token and window cache on local disk")
    arguments = parser.parse_args()
    output = arguments.output or OUTPUT / f"stage{arguments.stage}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    run(arguments.stage, arguments.smoke, output, arguments.cache)


if __name__ == "__main__":
    main()
