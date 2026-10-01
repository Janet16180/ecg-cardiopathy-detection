"""Experiment 043: ANN heads against pipeline v2's readout R, and a label-gated beat-wave map.

Stage 1 (CPU) trains MLP heads on xECG and on xECG + JEPA features, fits the logistic heads on the same
features, and gates Experiment 042's ``U_B`` map by the sign of its ``G_B`` shares. Stages 2 and 3 (GPU) are
added later and reuse the rows, split, comparator, evaluation sets and statistics defined here.

The row logic lives in pinned runners (Experiments 030, 032, 037 and 042), which are imported as documented
in the protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.ann_heads import (
    SEEDS,
    VALIDATION_SHARE,
    MLPHead,
    Recipe,
    detection_reading,
    gate_beat_units,
    map_reading,
    predict,
    ragged_unit_maps,
    train,
    validation_mask,
    weighted_standardizer,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.hard_subset import ARM_SPECS, arm_design
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    UnitMap,
    ecg_score,
    premature_hit,
    top_lead,
    two_group_difference,
)
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
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
    "scripts/experiments/run_lead_wave_maps042.py", "pyproject.toml", "uv.lock", PROTOCOL,
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
    return {"x": x, "readouts": readouts, "design": design, "groups": groups[binary_rows],
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


def run(stage: int, smoke: bool, output: Path) -> None:
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
    """
    if stage != 1:
        raise NotImplementedError(f"Stage {stage} is not implemented yet")
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(partial / "run.log")])
    torch.set_num_threads(THREADS)
    if torch.cuda.is_available():
        raise RuntimeError("Stage 1 is CPU only: set CUDA_VISIBLE_DEVICES=")
    receipts = {to_stored(PRIOR037 / name): digest
                for name, digest in receipt_checked(PRIOR037, ("predictions.npz",)).items()}
    receipts.update({to_stored(PRIOR035 / name): digest
                     for name, digest in receipt_checked(PRIOR035, ("predictions.npz",)).items()})
    receipts.update({to_stored(PRIOR042 / name): digest
                     for name, digest in receipt_checked(PRIOR042, ("unit_scores.npz",)).items()})
    data = training_data()
    LOG.info("loaded %s at %.1f s", data["training_counts"], time.perf_counter() - started)
    run_identity = identity(receipts, data["identity"])
    found = run_smoke(data, partial) if smoke else run_stage1(data, partial)
    write_json_atomic(partial / "result.json", {
        "status": "smoke" if smoke else "completed", "stage": stage, "identity": run_identity,
        "training_counts": data["training_counts"], **found,
        "recipe": vars(MLP_RECIPE), "hidden": MLP_HIDDEN, "dropout": MLP_DROPOUT,
        "training_seeds": list(SEEDS), "validation_share": VALIDATION_SHARE,
        "validation_seed": VALIDATION_SEED,
        "draws": SMOKE_DRAWS if smoke else DRAWS, "bootstrap_seed": SEED,
        "outputs_sha256": {"predictions.npz": sha256_file(partial / "predictions.npz")},
        "total_seconds": time.perf_counter() - started,
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
    arguments = parser.parse_args()
    output = arguments.output or OUTPUT / f"stage{arguments.stage}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    run(arguments.stage, arguments.smoke, output)


if __name__ == "__main__":
    main()
