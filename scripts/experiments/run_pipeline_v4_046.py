"""Experiment 046: pipeline v4, pipeline v3 with its binary readout replaced by the ensemble E.

Retrains Experiment 043 Stage 2's attention head on frozen ECG-JEPA tokens (A) with its recipe, split and
seeds, saves its weights, extracts the tokens of the SPH evaluation ECGs that 043 did not cache, and
recomputes Experiment 044's operating numbers on SPH for pipelines v2, v3 and v4 (binary readout E, the mean
of v3's readout logit and A's logit) over 044's identical draws and resamples.

The row logic lives in pinned runners (Experiments 030, 032, 033, 037, 043 and 044); it is reached through the
043 and 044 runners, as documented in the protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import ecg_experiment
from ecg_experiment.ann_heads import SEEDS, AttentionHead, Recipe, open_row_file, predict, read_rows, train
from ecg_experiment.external_encoders import (
    JEPA_CHECKPOINT,
    jepa_input,
    load_jepa,
    load_xecg_backbone,
    xecg_features,
    xecg_input,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.finding_screen import split_thresholds
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import jepa_tokens
from ecg_experiment.paths import to_stored
from ecg_experiment.pipeline_v2 import percentile_interval, resample_counts, score_parameters
from ecg_experiment.pipeline_v4 import (
    candidate_reading,
    ensemble_logit,
    load_state,
    matched_rate_screen,
    retrain_agreement,
    state_arrays,
    summarize_screens,
)
from ecg_experiment.provenance import git_head
from scripts.experiments import run_ann_heads043 as exp043
from scripts.experiments import run_pipeline_v3_044 as exp044

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment046_pipeline_v4_v1"
PRIOR035 = exp044.PRIOR035
PRIOR037 = exp044.PRIOR037
PRIOR043_STAGE1 = exp043.PRIOR_STAGE1
PRIOR043_STAGE2 = exp043.PRIOR_STAGE2
PRIOR044 = ROOT / "outputs/experiment044_pipeline_v3_v1"
PRIOR044_RATE = ROOT / "outputs/experiment044_rate_matched_v1"
PRIOR045 = ROOT / "outputs/experiment045_two_layer_map_v1"
CACHE043 = exp043.CACHE
CACHE046 = CACHE043.parent / "exp046_cache"
SMOKE_CACHE = CACHE043.parent / "smoke046_cache"
PROTOCOL = "docs/experiment-046-pipeline-v4.md"
SOURCES = (
    "ecg_experiment/pipeline_v4.py", "ecg_experiment/pipeline_v3.py", "ecg_experiment/pipeline_v2.py",
    "ecg_experiment/finding_screen.py", "ecg_experiment/intervals.py", "ecg_experiment/referral_budget.py",
    "ecg_experiment/hard_subset.py", "ecg_experiment/multisource_readout.py",
    "ecg_experiment/full_development.py",
    "ecg_experiment/hybrid_score.py", "ecg_experiment/rhythm_findings.py", "ecg_experiment/ann_heads.py",
    "ecg_experiment/lead_wave_maps.py", "ecg_experiment/external_encoders.py", "ecg_experiment/sph.py",
    "ecg_experiment/challenge_features.py", "ecg_experiment/public_sources.py", "ecg_experiment/gpu.py",
    "scripts/experiments/run_pipeline_v4_046.py", "scripts/experiments/run_pipeline_v3_044.py",
    "scripts/experiments/run_ann_heads043.py", "scripts/experiments/run_pipeline_v2_037.py",
    "scripts/experiments/run_finding_screen033.py", "scripts/experiments/run_rhythm_findings032.py",
    "scripts/experiments/run_referral_budget030.py", "scripts/experiments/run_lead_wave_maps042.py",
    "scripts/data/extract_challenge_features.py", "pyproject.toml", "uv.lock", PROTOCOL,
)
CONTRASTS = (("v4_minus_v3", "v4", "v3"), ("v4_minus_v2", "v4", "v2"), ("v3_minus_v2", "v3", "v2"))
DEVICE = "cuda"
EXPECTED_NEW_SPH = 4569
RETRAIN_CORRELATION = 0.999
RETRAIN_AUROC = 0.001
WEIGHT_TOLERANCE = 1e-6
LOGIT_TOLERANCE = 1e-10
TOLERANCE = 1e-12
CEILING = 7200.0
THREADS = 4
RATE_OUTCOMES = ("normal", "composite", "binary_positive", "pvc", "other")
COST_RECORDS = 128
COST_WARMUP = 8
SMOKE_EXTRACT = 64
SMOKE_RATE_OUTCOMES = ("normal", "composite", "binary_positive", "pvc")
LOG = logging.getLogger("experiment046")


def input_receipts() -> dict[str, str]:
    """Return the hashes of the earlier outputs this run reads, each checked against its receipt."""
    found = {}
    for directory, names in ((PRIOR037, ("predictions.npz", "draws.csv", "pipeline_v2_heads.npz")),
                             (PRIOR043_STAGE1, ("predictions.npz",)), (PRIOR043_STAGE2, ("predictions.npz",)),
                             (PRIOR044, ("predictions.npz", "draws.csv", "pipeline_v3_heads.npz")),
                             (PRIOR035, ("predictions.npz",))):
        found.update({to_stored(directory / name): digest
                      for name, digest in exp044.receipt_checked(directory, names).items()})
    for directory in (PRIOR044_RATE, PRIOR045):
        found[to_stored(directory / "result.json")] = sha256_file(directory / "result.json")
    return found


def identity(receipts: dict[str, Any], data_identity: dict[str, Any]) -> dict[str, Any]:
    """
    Hash every source and the protocol, with the input receipts and the run environment.

    Parameters
    ----------
    receipts : dict[str, Any]
        Output of ``input_receipts``.
    data_identity : dict[str, Any]
        Input identity from Experiment 043's ``training_data``.

    Returns
    -------
    dict[str, Any]
        The identity block of ``result.json``.

    Raises
    ------
    RuntimeError
        If ``ecg_experiment`` is imported from outside this checkout.
    """
    if Path(ecg_experiment.__file__).resolve().parents[1] != ROOT:
        raise RuntimeError(f"ecg_experiment is imported from {ecg_experiment.__file__}, not {ROOT}")
    return {"inputs": receipts, "data": data_identity,
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES}, "git_head": git_head(ROOT),
            "openblas_architectures": exp044.blas_architectures(), "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "jepa_checkpoint_sha256": sha256_file(JEPA_CHECKPOINT),
            "torch_threads": torch.get_num_threads()}


def frozen_scores(data: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """
    Return the frozen v2 and v3 scores (037's and 044's saved files) and 044's saved arrays.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]
        ``v2`` and ``v3`` probabilities and ``v3_logit`` per part (``sph``, ``development``), and 044's
        arrays.

    Raises
    ------
    ValueError
        If a row order differs or 037's v2 scores differ from 044's copy.
    """
    sph_ids, development_ids = (data["sph"]["ecg_id"].to_numpy(dtype=str),
                                data["development"].index.to_numpy(dtype=str))
    with np.load(PRIOR037 / "predictions.npz") as saved:
        if not (np.array_equal(saved["sph_ecg_ids"], sph_ids)
                and np.array_equal(saved["development_record_ids"], development_ids)):
            raise ValueError("037 prediction rows differ")
        v2 = {part: saved[f"{part}_v2_binary"] for part in ("sph", "development")}
    with np.load(PRIOR044 / "predictions.npz") as saved:
        saved044 = {key: saved[key] for key in saved.files}
    if not (np.array_equal(saved044["sph_ecg_ids"], sph_ids)
            and np.array_equal(saved044["development_record_ids"], development_ids)):
        raise ValueError("044 prediction rows differ")
    if not all(np.array_equal(v2[part], saved044[f"{part}_v2_binary"]) for part in v2):
        raise ValueError("037's v2 scores differ from 044's copy")
    frozen = {"v2": v2, "v3": {part: saved044[f"{part}_v3_binary"] for part in v2},
              "v3_logit": {part: saved044[f"{part}_v3_logit"] for part in v2}}
    return frozen, saved044


def check_r3(data: dict[str, Any], frozen: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """
    Refit v3's readout R3 as Experiment 044 did and require it to reproduce 044's and 043's saved scores.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    frozen : dict[str, dict[str, np.ndarray]]
        Output of ``frozen_scores``.

    Returns
    -------
    dict[str, Any]
        Largest differences against 044 and 043, and the iterations.

    Raises
    ------
    ValueError
        If a difference exceeds its tolerance.
    """
    began = time.perf_counter()
    concat = {"stacked": np.concatenate([data["x"]["xecg"], data["x"]["jepa"]], axis=1),
              **{part: np.concatenate([data[f"{part}_x"]["xecg"], data[f"{part}_x"]["jepa"]], axis=1)
                 for part in ("sph", "development")}}
    sets = {"concat": concat}
    heads, scores, iterations = exp044.fit_heads(sets, {"v3": exp044.head_plan(data)["v3"]},
                                                 ("sph", "development"))
    logits, versus043 = exp044.check_v3(data, heads, sets)
    differences = {}
    for part in ("sph", "development"):
        differences[f"{part}_logit"] = float(np.abs(logits[part] - frozen["v3_logit"][part]).max())
        differences[f"{part}_probability"] = float(np.abs(scores["v3"][part] - frozen["v3"][part]).max())
    with np.load(PRIOR044 / "pipeline_v3_heads.npz") as saved:
        parameters = {key: saved[key] for key in saved.files}
    saved_parameters = float(np.abs(score_parameters(parameters, "v3", concat["sph"])
                                    - scores["v3"]["sph"]).max())
    if max(differences.values()) > LOGIT_TOLERANCE or saved_parameters > TOLERANCE:
        raise ValueError(f"R3 does not reproduce 044: {differences}, parameters {saved_parameters}")
    LOG.info("R3 refit reproduces 044 in %.1f s: %s", time.perf_counter() - began, differences)
    return {"vs_044": differences, "vs_043_logistic_concat": versus043,
            "saved_parameters_vs_refit": saved_parameters, "iterations": iterations["v3"]}


def new_sph_table(data: dict[str, Any], sets: dict[str, dict[str, Any]]
                  ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Return the SPH evaluation ECGs without a binary label, which Experiment 043 did not cache.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    sets : dict[str, dict[str, Any]]
        Output of Experiment 043's ``evaluation_sets``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        Cache rows in 043's reader format, 022's saved JEPA features of them and their SPH positions.

    Raises
    ------
    ValueError
        If the count differs from the protocol or a row has a binary label.
    """
    labeled = np.zeros(len(data["sph"]), dtype=bool)
    labeled[sets["sph"]["positions"]] = True
    positions = np.flatnonzero(~labeled)
    rows = data["sph"].iloc[positions]
    if len(positions) != EXPECTED_NEW_SPH or rows["primary"].notna().any():
        raise ValueError(f"Expected {EXPECTED_NEW_SPH} SPH rows without a binary label: {len(positions)}")
    table = pd.DataFrame({"kind": "sph", "key": ("sph:" + rows["ecg_id"]).to_numpy(dtype=str),
                          "ecg_id": rows["ecg_id"].to_numpy(dtype=str),
                          "signal_sha256": rows["signal_sha256"].to_numpy(dtype=str)})
    return table, data["sph_x"]["jepa"][positions], positions


def checked_cache043(data: dict[str, Any], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Require Experiment 043's token cache to hold 043's rows and to match its receipt (rehashed).

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    sets : dict[str, dict[str, Any]]
        Output of Experiment 043's ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        The cache receipt, with its part boundaries.

    Raises
    ------
    ValueError
        If the cache is missing, incomplete or differs from 043's recorded receipt.
    """
    began = time.perf_counter()
    table, _, parts = exp043.extraction_table(data, sets)
    receipt = exp043.cache_receipt(CACHE043, table, verify=True)
    recorded = json.loads((PRIOR043_STAGE2 / "result.json").read_text())["cache"]["receipt"]
    if receipt is None or receipt != recorded or receipt["parts"] != parts:
        raise ValueError("Experiment 043's token cache differs from its recorded receipt")
    LOG.info("043 cache verified in %.1f s", time.perf_counter() - began)
    return receipt


def train_and_score(batch: Any, fit: dict[str, np.ndarray], score: dict[str, tuple[Any, np.ndarray]],
                    recipe: Recipe) -> dict[str, Any]:
    """
    Train the attention head with every seed as Experiment 043's ``train_seeds``, keep the weights and score.

    Parameters
    ----------
    batch : Any
        Maps training-cache rows to a token tensor on the GPU.
    fit : dict[str, np.ndarray]
        ``rows``, ``y``, ``weights``, ``check_rows`` and ``check_y`` of the fit and validation parts.
    score : dict[str, tuple[Any, np.ndarray]]
        Per scored part, its batch function and rows.
    recipe : Recipe
        Training recipe.

    Returns
    -------
    dict[str, Any]
        ``logits`` (seed mean per part), ``seed_logits``, ``states`` (best weights per seed) and ``seeds``.
    """
    seed_logits, states, seeds = {}, {}, {}
    for seed in SEEDS:
        began = time.perf_counter()
        torch.manual_seed(seed)
        model = AttentionHead(exp043.TOKEN_SHAPE[1]).to(DEVICE)
        result = train(model, batch, fit["rows"], fit["y"], fit["weights"], fit["check_rows"], fit["check_y"],
                       recipe, seed, log=lambda line: LOG.info("attention_jepa %s", line))
        states[seed] = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
        seed_logits[seed] = {part: predict(model, part_batch, rows, exp043.PREDICT_BATCH)[0]
                             for part, (part_batch, rows) in score.items()}
        seeds[str(seed)] = {"best_epoch": result.best_epoch, "validation_auroc": result.best_auroc,
                            "epochs_run": len(result.history), "seconds": time.perf_counter() - began,
                            "history": result.history}
        LOG.info("seed %d best epoch %d validation AUROC %.4f", seed, result.best_epoch, result.best_auroc)
        del model
    return {"logits": {part: np.mean([seed_logits[seed][part] for seed in SEEDS], axis=0) for part in score},
            "seed_logits": seed_logits, "states": states, "seeds": seeds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}


def check_weights(states: dict[int, dict[str, np.ndarray]], batch: Any, rows: np.ndarray,
                  expected: dict[int, np.ndarray]) -> float:
    """
    Require fresh networks loaded from the saved arrays to reproduce each seed's logits.

    Parameters
    ----------
    states : dict[int, dict[str, np.ndarray]]
        Best weights per seed.
    batch : Any
        Maps rows to a token tensor on the GPU.
    rows : np.ndarray
        Rows to score.
    expected : dict[int, np.ndarray]
        Logits of those rows per seed.

    Returns
    -------
    float
        The largest absolute difference.

    Raises
    ------
    ValueError
        If it exceeds ``WEIGHT_TOLERANCE``.
    """
    arrays = state_arrays(states)
    worst = 0.0
    for seed in SEEDS:
        model = load_state(AttentionHead(exp043.TOKEN_SHAPE[1]), arrays, seed).to(DEVICE)
        logits, _ = predict(model, batch, rows, exp043.PREDICT_BATCH)
        worst = max(worst, float(np.abs(logits - expected[seed]).max()))
        del model
    if worst > WEIGHT_TOLERANCE:
        raise ValueError(f"Saved weights do not reproduce the logits: {worst}")
    return worst


def timed(function: Any, *arguments: Any) -> tuple[Any, float]:
    """Return a function's result and its wall time, synchronizing the GPU before and after."""
    torch.cuda.synchronize()
    began = time.perf_counter()
    found = function(*arguments)
    torch.cuda.synchronize()
    return found, time.perf_counter() - began


def inference_cost(cache: Path, encoder: torch.nn.Module, states: dict[int, dict[str, np.ndarray]],
                   references: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Time each part of pipeline v4 per ECG on cached windows, after a short warm-up.

    Parameters
    ----------
    cache : Path
        Cache folder whose ``windows.npy`` holds the windows.
    encoder : torch.nn.Module
        ECG-JEPA encoder on the GPU.
    states : dict[int, dict[str, np.ndarray]]
        Attention-head weights per seed.
    references : dict[str, np.ndarray]
        Saved ``jepa`` and ``xecg`` features of the cached rows, in order.

    Returns
    -------
    dict[str, Any]
        Seconds per ECG of each step and the largest feature differences from the saved ones.
    """
    store = open_row_file(cache / "windows.npy")
    records = min(COST_RECORDS, store.shape[0])
    windows = read_rows(store, np.arange(records))
    os.close(store.descriptor)

    def jepa_part(chosen: np.ndarray) -> np.ndarray:
        return jepa_tokens(encoder, np.stack([jepa_input(window) for window in chosen]))

    jepa_part(windows[:COST_WARMUP])
    tokens, jepa_seconds = timed(jepa_part, windows)
    arrays = state_arrays(states)
    models = [load_state(AttentionHead(exp043.TOKEN_SHAPE[1]), arrays, seed).to(DEVICE) for seed in SEEDS]
    batch = exp043.array_batch(tokens.astype(np.float16), DEVICE)

    def heads_part(rows: np.ndarray) -> list[np.ndarray]:
        return [predict(model, batch, rows, exp043.PREDICT_BATCH)[0] for model in models]

    heads_part(np.arange(COST_WARMUP))
    _, heads_seconds = timed(heads_part, np.arange(records))
    models.clear()
    backbone = load_xecg_backbone()

    def xecg_part(chosen: np.ndarray) -> np.ndarray:
        return xecg_features(backbone, np.stack([xecg_input(window) for window in chosen]))

    xecg_part(windows[:COST_WARMUP])
    features, xecg_seconds = timed(xecg_part, windows)
    torch.cuda.empty_cache()
    return {"records": records, "warmup_records": COST_WARMUP, "device": torch.cuda.get_device_name(0),
            "seconds_per_ecg": {"jepa_tokens_with_input": jepa_seconds / records,
                                "attention_heads_three_seeds": heads_seconds / records,
                                "xecg_features_with_input": xecg_seconds / records},
            "jepa_token_mean_vs_saved": float(np.abs(tokens.mean(axis=1, dtype=np.float64)
                                                     - references["jepa"][:records]).max()),
            "xecg_vs_saved": float(np.abs(features - references["xecg"][:records]).max())}


def gpu_stage(new: dict[str, Any], cache: Path, fit: dict[str, np.ndarray], scored043: dict[str, np.ndarray],
              recipe: Recipe, started: float) -> dict[str, Any]:
    """
    Profile, extract the new tokens, train and score A, check its saved weights and time inference (GPU).

    Parameters
    ----------
    new : dict[str, Any]
        ``table`` of rows to extract, their saved ``jepa`` and ``xecg`` features.
    cache : Path
        Folder of the new token cache.
    fit : dict[str, np.ndarray]
        Fit and validation rows of 043's cache, as ``train_and_score``.
    scored043 : dict[str, np.ndarray]
        Parts to score from 043's cache, as rows of it.
    recipe : Recipe
        Training recipe.
    started : float
        ``time.perf_counter()`` at the start of the run.

    Returns
    -------
    dict[str, Any]
        ``profile``, new cache ``receipt``, the trained ``arm`` (with ``new`` scores), ``weights_check`` and
        ``cost``.

    Raises
    ------
    RuntimeError
        If the projected time exceeds ``CEILING``.
    """
    readers = (exp043.load_checksums(), {record: window for record, _, window in exp043.ningbo_items()[0]})
    with gpu_lock(DEVICE):
        torch.cuda.reset_peak_memory_stats()
        store043 = open_row_file(CACHE043 / "tokens.npy")
        batch043 = exp043.row_batch(store043, DEVICE)
        encoder = load_jepa()
        profile: dict[str, Any] = {"extraction": exp043.profile_extraction(new["table"], encoder, *readers)}
        profile["training"] = exp043.profile_training(partial_attention, batch043, fit["rows"], fit["y"],
                                                      fit["weights"], recipe, DEVICE)
        scored = sum(len(rows) for rows in scored043.values()) + len(new["table"])
        per_record = profile["extraction"]["read_per_record"] + profile["extraction"]["model_per_record"]
        profile["projected_extraction_seconds"] = per_record * len(new["table"])
        profile["projected_training_seconds"] = exp043.projected_training(
            profile["training"], len(fit["rows"]), len(fit["check_rows"]), scored, recipe)
        profile["projected_run_seconds"] = (time.perf_counter() - started
                                            + profile["projected_extraction_seconds"]
                                            + profile["projected_training_seconds"])
        LOG.info("GPU profile %s", profile)
        if profile["projected_run_seconds"] > CEILING:
            raise RuntimeError(f"Projected time {profile['projected_run_seconds']:.0f} s exceeds 2 hours")
        began = time.perf_counter()
        receipt = exp043.cache_receipt(cache, new["table"], verify=True)
        profile["new_cache_reused"] = receipt is not None
        if receipt is None:
            receipt = exp043.extract_cache(cache, new["table"], new["jepa"], {"new": [0, len(new["table"])]},
                                           encoder, *readers)
        profile["extraction_seconds"] = time.perf_counter() - began
        LOG.info("new cache ready in %.1f s: %s", profile["extraction_seconds"], receipt["token_check"])
        store046 = open_row_file(cache / "tokens.npy")
        score = {**{part: (batch043, rows) for part, rows in scored043.items()},
                 "new": (exp043.row_batch(store046, DEVICE), np.arange(len(new["table"])))}
        began = time.perf_counter()
        arm = train_and_score(batch043, fit, score, recipe)
        profile["training_seconds"] = time.perf_counter() - began
        first = next(iter(scored043))
        weights_check = check_weights(arm["states"], batch043, scored043[first],
                                      {seed: values[first] for seed, values in arm["seed_logits"].items()})
        cost = inference_cost(cache, encoder, arm["states"], new)
        del encoder
        torch.cuda.empty_cache()
        profile["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    for store in (store043, store046):
        os.close(store.descriptor)
    LOG.info("GPU stage done at %.1f s: cost %s", time.perf_counter() - started, cost)
    return {"profile": profile, "receipt": receipt, "arm": arm, "weights_check": weights_check, "cost": cost}


def partial_attention() -> torch.nn.Module:
    """Build Experiment 043's attention head on the CPU."""
    return AttentionHead(exp043.TOKEN_SHAPE[1])


def retrain_check(arm: dict[str, Any], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Compare the retrained A with Experiment 043 Stage 2's saved logits, by the protocol's tolerances.

    Parameters
    ----------
    arm : dict[str, Any]
        Output of ``train_and_score`` with parts ``development`` and ``sph_labeled``.
    sets : dict[str, dict[str, Any]]
        Output of Experiment 043's ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        Agreement of the seed mean and of each seed on SPH and full development, and the best epochs.

    Raises
    ------
    ValueError
        If the seed mean's r is below ``RETRAIN_CORRELATION`` or an AUROC moves by more than
        ``RETRAIN_AUROC``.
    """
    full = sets["full"]["positions"]
    labels = {"sph": sets["sph"]["y"], "full": sets["full"]["y"]}
    with np.load(PRIOR043_STAGE2 / "predictions.npz") as saved:
        old = {"mean": {"sph": saved["sph_attention_jepa"],
                        "full": saved["development_attention_jepa"][full]},
               **{seed: {"sph": saved[f"sph_attention_jepa_seed{seed}"],
                         "full": saved[f"development_attention_jepa_seed{seed}"][full]} for seed in SEEDS}}
    new = {"mean": {"sph": arm["logits"]["sph_labeled"], "full": arm["logits"]["development"][full]},
           **{seed: {"sph": arm["seed_logits"][seed]["sph_labeled"],
                     "full": arm["seed_logits"][seed]["development"][full]} for seed in SEEDS}}
    found = {str(key): {part: retrain_agreement(new[key][part], old[key][part], labels[part])
                        for part in ("sph", "full")} for key in new}
    prior_seeds = json.loads((PRIOR043_STAGE2 / "result.json").read_text())["arm"]["seeds"]
    found["best_epochs"] = {str(seed): {"046": arm["seeds"][str(seed)]["best_epoch"],
                                        "043": prior_seeds[str(seed)]["best_epoch"]} for seed in SEEDS}
    passed = all(found["mean"][part]["r"] >= RETRAIN_CORRELATION
                 and abs(found["mean"][part]["auroc_difference"]) <= RETRAIN_AUROC
                 for part in ("sph", "full"))
    found["passed"] = passed
    if not passed:
        raise ValueError(f"The retrained A disagrees with 043's beyond tolerance: {found['mean']}")
    return found


def check_draws_044(draws: pd.DataFrame) -> dict[str, Any]:
    """
    Require the v2 and v3 draws to equal the v2 and v3 rows of Experiment 044's ``draws.csv`` exactly.

    Raises
    ------
    ValueError
        If a column set or any value differs.
    """
    prior = exp044.checked_csv(PRIOR044, "draws.csv", float_precision="round_trip")
    keys = ["rule", "m", "budget", "draw"]
    found = {}
    for pipeline in ("v2", "v3"):
        ours = draws[draws["pipeline"] == pipeline].drop(columns="pipeline")
        theirs = prior[prior["pipeline"] == pipeline].drop(columns="pipeline")
        if sorted(ours.columns) != sorted(theirs.columns):
            raise ValueError("Draw columns differ from 044's")
        columns = [column for column in theirs.columns if column not in keys]
        largest = exp044.largest_difference(ours, theirs, keys,
                                            [(column, f"{column}_prior") for column in columns])
        if largest != 0.0:
            raise ValueError(f"The {pipeline} draws differ from 044's by {largest}")
        found[pipeline] = {"rows": len(ours), "largest_difference": largest}
    return found


def contrast_row(contrasts: list[dict[str, Any]], label: str, outcome: str, size: int, rule: str
                 ) -> dict[str, Any]:
    """Return the one contrast row of a label, outcome and size at the primary budget."""
    (row,) = [row for row in contrasts if row["contrast"] == label and row["rule"] == rule
              and row["m"] == size and row["budget"] == exp044.PRIMARY_BUDGET / 1000
              and row["outcome"] == outcome]
    return row


def decide(contrasts: list[dict[str, Any]], size: int) -> dict[str, Any]:
    """
    Apply the pre-registered reading to the primary contrast, with the v4 - v2 and v3 - v2 readings beside it.

    Parameters
    ----------
    contrasts : list[dict[str, Any]]
        Output of ``summarize_screens``.
    size : int
        Number of local normals of the primary screen.

    Returns
    -------
    dict[str, Any]
        The primary contrast, the decision and the secondary readings (no decision).
    """
    primary = contrast_row(contrasts, "v4_minus_v3", "composite", size, exp044.PRIMARY_RULE)
    secondary = {label: contrast_row(contrasts, label, "composite", size, exp044.PRIMARY_RULE)
                 for label in ("v4_minus_v2", "v3_minus_v2")}
    return {"primary": primary,
            "decision": candidate_reading(primary["ci"], exp044.WORSE_MARGIN, "v4", "v3"),
            "margin": exp044.WORSE_MARGIN,
            "v4_minus_v2_reading": candidate_reading(secondary["v4_minus_v2"]["ci"], exp044.WORSE_MARGIN,
                                                     "v4", "v2"),
            "v3_minus_v2": secondary["v3_minus_v2"]}


def matched_rates(pipelines: dict[str, dict[str, np.ndarray]], evaluation: np.ndarray,
                  masks: dict[str, np.ndarray], local_normal: np.ndarray, patients: np.ndarray,
                  resamples: int, outcomes: tuple[str, ...]) -> dict[str, Any]:
    """
    Run Experiment 044's matched-rate check: thresholds on the (resampled) evaluation normals and the pool.

    Parameters
    ----------
    pipelines : dict[str, dict[str, np.ndarray]]
        Per pipeline the ``binary`` and ``combined`` scores of every row.
    evaluation : np.ndarray
        Evaluation mask over the rows.
    masks : dict[str, np.ndarray]
        Outcome masks over the evaluation rows.
    local_normal : np.ndarray
        Row positions of the local normals.
    patients : np.ndarray
        Patient of each evaluation row.
    resamples : int
        Number of whole-patient resamples.
    outcomes : tuple[str, ...]
        Outcomes to report.

    Returns
    -------
    dict[str, Any]
        ``oracle_evaluation_thresholds`` and ``whole_local_pool_thresholds`` rows in 044's format.
    """
    chosen = {name: masks[name] for name in outcomes}
    counts = resample_counts(patients, resamples, exp044.BOOTSTRAP_SEED)
    ones = np.ones(int(evaluation.sum()))
    oracle, pooled = [], []
    for rule in exp044.RULES:
        columns, shares = exp044.RULES033[rule]
        for budget in exp044.BUDGETS:
            observed, boot = {}, {}
            for pipeline, scores in pipelines.items():
                matrix = np.column_stack([scores["binary"], *(scores[column] for column in columns)])
                part = matrix[evaluation]
                observed[pipeline] = matched_rate_screen(part, ones, chosen["normal"], chosen, budget, shares)
                boot[pipeline] = [matched_rate_screen(part, row, chosen["normal"], chosen, budget, shares)
                                  for row in counts]
                referred = (part > split_thresholds(matrix[local_normal], budget, shares)).any(axis=1)
                pooled.append({"rule": rule, "budget": budget / 1000, "pipeline": pipeline,
                               **{name: float(referred[mask].mean()) for name, mask in chosen.items()}})
            row: dict[str, Any] = {"rule": rule, "budget": budget / 1000, "observed": observed,
                                   "contrasts": {}}
            for label, first, second in CONTRASTS:
                row["contrasts"][label] = {
                    name: {"difference": observed[first][name] - observed[second][name],
                           "ci": percentile_interval(np.array([a[name] - b[name]
                                                               for a, b in zip(boot[first], boot[second],
                                                                               strict=True)]))}
                    for name in outcomes}
            oracle.append(row)
    return {"oracle_evaluation_thresholds": oracle, "whole_local_pool_thresholds": pooled}


def check_matched_044(found: dict[str, Any]) -> float:
    """
    Require the v2 and v3 matched-rate numbers to reproduce Experiment 044's rate-matched ``result.json``.

    Raises
    ------
    ValueError
        If a value differs by more than ``TOLERANCE`` or a row is missing.
    """
    prior = json.loads((PRIOR044_RATE / "result.json").read_text())
    pairs = []
    for ours, theirs in zip(found["oracle_evaluation_thresholds"], prior["oracle_evaluation_thresholds"],
                            strict=True):
        if (ours["rule"], ours["budget"]) != (theirs["rule"], theirs["budget"]):
            raise ValueError("Matched-rate rows differ in order from 044's")
        for pipeline in ("v2", "v3"):
            pairs += [(ours["observed"][pipeline][name], value)
                      for name, value in theirs["observed"][pipeline].items()]
        for name, value in theirs["contrasts"]["v3_minus_v2"].items():
            mine = ours["contrasts"]["v3_minus_v2"][name]
            pairs += [(mine["difference"], value["difference"]), *zip(mine["ci"], value["ci"], strict=True)]
    prior_pooled = {(row["rule"], row["budget"], row["pipeline"]): row
                    for row in prior["whole_local_pool_thresholds"]}
    for row in found["whole_local_pool_thresholds"]:
        if row["pipeline"] in ("v2", "v3"):
            theirs = prior_pooled[(row["rule"], row["budget"], row["pipeline"])]
            pairs += [(row[name], theirs[name]) for name in RATE_OUTCOMES]
    largest = max(abs(first - second) for first, second in pairs)
    if largest > TOLERANCE:
        raise ValueError(f"The matched-rate check differs from 044's by {largest}")
    return largest


def auroc_tables(data: dict[str, Any], scores: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """
    Compute the AUROC of v2, v3, v4 and A on Experiment 037's sets, with the paired v4 contrasts.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    scores : dict[str, dict[str, np.ndarray]]
        Per readout (``v2``, ``v3``, ``v4``, ``A``), scores per part (``development``, ``sph``).

    Returns
    -------
    dict[str, Any]
        ``binary`` per set, ``reference`` (045's E and 043's A) and the checks.

    Raises
    ------
    ValueError
        If v3's AUROC differs from 043 Stage 1's.
    """
    sets = exp043.evaluation_sets(data)
    checks = {"v2_vs_037": exp043.check_comparator_aurocs(sets, scores["v2"])}
    prior043 = json.loads((PRIOR043_STAGE1 / "result.json").read_text())
    prior043 = prior043["detection"]["logistic_concat"]["sets"]
    binary = {}
    for name, spec in sets.items():
        found = {readout: float(roc_auc_score(spec["y"], exp043.set_scores(spec, values)))
                 for readout, values in scores.items()}
        if abs(found["v3"] - prior043[name]["auroc"]) > TOLERANCE:
            raise ValueError(f"v3's {name} AUROC differs from 043's")
        binary[name] = {"records": len(spec["y"]), "positives": int(spec["y"].sum()), "auroc": found,
                        **{f"v4_minus_{other}": paired_auroc_difference(
                            spec["patients"], spec["y"], exp043.set_scores(spec, scores["v4"]),
                            exp043.set_scores(spec, scores[other]), exp044.RESAMPLES, exp044.BOOTSTRAP_SEED)
                           for other in ("v3", "v2")}}
    checks["v3_vs_043"] = {name: binary[name]["auroc"]["v3"] for name in binary}
    ensemble045 = json.loads((PRIOR045 / "result.json").read_text())["ensemble"]["detection"]["E"]["sets"]
    attention043 = json.loads((PRIOR043_STAGE2 / "result.json").read_text())
    attention043 = attention043["detection"]["attention_jepa"]["sets"]
    reference = {name: {"E_045": ensemble045[name]["auroc"], "A_043": attention043[name]["auroc"]}
                 for name in sets}
    return {"binary": binary, "reference": reference, "checks": checks}


def screens(pipelines: dict[str, dict[str, np.ndarray]], evaluation: np.ndarray, masks: dict[str, np.ndarray],
            local_masks: dict[str, np.ndarray], local_normal: np.ndarray, patients: np.ndarray,
            sizes: tuple[int, ...], primary_size: int, resamples: int) -> dict[str, Any]:
    """
    Run 044's draws and resamples for every pipeline and summarize them with the 046 contrasts.

    Returns
    -------
    dict[str, Any]
        ``draws``, ``summary``, ``contrasts``, ``reading``, ``local_selection`` and ``empty_resamples``.
    """
    with threadpool_limits(limits=THREADS):
        draws, shares = exp044.run_draws(pipelines, evaluation, masks, local_masks, local_normal, sizes,
                                         primary_size)
        boot, keys, matrix, empty = exp044.bootstrap(patients, shares, masks, resamples)
    exp044.check_shares(draws, matrix, keys, masks["composite"])
    summary, contrasts = summarize_screens(draws, boot, keys, CONTRASTS, exp044.PREVALENCE)
    return {"draws": draws, "summary": summary, "contrasts": contrasts,
            "reading": decide(contrasts, primary_size),
            "local_selection": exp044.local_selection(draws, primary_size), "empty_resamples": empty}


def run_full(data: dict[str, Any], partial: Path, started: float) -> dict[str, Any]:
    """
    Run the experiment: checks, A retrained and scored, the three pipelines over 044's draws, and the AUROCs.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    partial : Path
        Output folder.
    started : float
        ``time.perf_counter()`` at the start of the run.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    checks: dict[str, Any] = {"training_rows_vs_037": exp043.check_against_037(data)}
    frozen, saved044 = frozen_scores(data)
    checks["r3_refit"] = check_r3(data, frozen)
    sets = exp043.evaluation_sets(data)
    validation, split_counts = exp043.shared_validation(data)
    with np.load(PRIOR043_STAGE1 / "predictions.npz") as saved:
        if not np.array_equal(saved["validation_mask"], validation):
            raise ValueError("Validation split differs from 043 Stage 1's")
    checks["cache043"] = checked_cache043(data, sets)
    table, jepa, positions = new_sph_table(data, sets)
    parts = checks["cache043"]["parts"]
    y, weights = data["design"]["y"], data["design"]["weights"]
    fit_rows, check_rows = np.flatnonzero(~validation), np.flatnonzero(validation)
    fit = {"rows": fit_rows, "y": y[fit_rows], "weights": weights[fit_rows], "check_rows": check_rows,
           "check_y": y[check_rows]}
    gpu = gpu_stage({"table": table, "jepa": jepa, "xecg": data["sph_x"]["xecg"][positions]}, CACHE046, fit,
                    {"development": np.arange(*parts["development"]),
                     "sph_labeled": np.arange(*parts["sph"])},
                    exp043.ATTENTION_RECIPE, started)
    arm = gpu["arm"]
    checks["retrain_vs_043"] = retrain_check(arm, sets)
    checks["weights_reload_max_abs_difference"] = gpu["weights_check"]
    LOG.info("retrain agreement %s", checks["retrain_vs_043"]["mean"])

    attention = {"development": arm["logits"]["development"], "sph": np.full(len(data["sph"]), np.nan)}
    attention["sph"][sets["sph"]["positions"]] = arm["logits"]["sph_labeled"]
    attention["sph"][positions] = arm["logits"]["new"]
    v4_logit = {part: ensemble_logit(frozen["v3_logit"][part], attention[part]) for part in attention}

    rows, site, local_normal = exp044.load_rows()
    counts = exp044.counts_of(rows)
    if json.loads(json.dumps(counts)) != exp044.EXPECTED033:
        raise ValueError("Counts differ from 033's")
    if not np.array_equal(rows["ecg_id"].to_numpy(dtype=str), data["sph"]["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("033's SPH rows differ from 032's")
    with threadpool_limits(limits=THREADS):
        scores033, checks["033_load_scores"] = exp044.load_scores(rows, site)
    checks["z_033_vs_044"] = max(float(np.abs(scores033[name] - saved044[f"sph_v2_z_{name}"]).max())
                                 for name in ("pvc", "wpw", "combined"))
    if checks["z_033_vs_044"] != 0.0:
        raise ValueError(f"033's z-scores differ from 044's by {checks['z_033_vs_044']}")
    findings = {name: scores033[name] for name in ("pvc", "wpw", "combined")}
    pipelines = {"v2": {"binary": frozen["v2"]["sph"], **findings},
                 "v3": {"binary": frozen["v3"]["sph"], **findings},
                 "v4": {"binary": expit(v4_logit["sph"]), **findings}}
    evaluation = rows["evaluation"].to_numpy()
    masks = exp044.outcome_masks(rows, evaluation)
    local_masks = {name: rows.loc[~evaluation, name].to_numpy() for name in exp044.LOCAL_OUTCOMES}
    patients = rows.loc[evaluation, "patient_id"].to_numpy(dtype=str)
    found = screens(pipelines, evaluation, masks, local_masks, local_normal, patients, exp044.SIZES,
                    exp044.PRIMARY_SIZE, exp044.RESAMPLES)
    checks["v2_draws_vs_037"] = exp044.check_v2_draws(found["draws"])
    checks["draws_vs_044"] = check_draws_044(found["draws"])
    prior_primary = json.loads((PRIOR044 / "result.json").read_text())["reading"]["primary"]
    if (abs(found["reading"]["v3_minus_v2"]["difference"] - prior_primary["difference"]) > TOLERANCE
            or max(abs(a - b) for a, b in zip(found["reading"]["v3_minus_v2"]["ci"], prior_primary["ci"],
                                               strict=True)) > TOLERANCE):
        raise ValueError("v3 - v2 does not reproduce 044's primary contrast")
    checks["v3_minus_v2_vs_044_primary"] = True
    for pipeline in ("v2", "v3"):
        if found["local_selection"][pipeline]["selected"] != exp044.PRIMARY_RULE:
            raise ValueError(f"The {pipeline} local selection does not reproduce 044's")
    LOG.info("screens done at %.1f s: %s", time.perf_counter() - started, found["reading"])
    matched = matched_rates(pipelines, evaluation, masks, local_normal, patients, exp044.RESAMPLES,
                            RATE_OUTCOMES)
    checks["matched_rate_vs_044"] = check_matched_044(matched)
    aurocs = auroc_tables(data, {"v2": frozen["v2"], "v3": frozen["v3"],
                                 "v4": {part: expit(values) for part, values in v4_logit.items()},
                                 "A": attention})
    LOG.info("matched rates and AUROCs done at %.1f s", time.perf_counter() - started)

    found["draws"].to_csv(partial / "draws.csv", index=False)
    write_npz_atomic(partial / "attention_jepa_weights.npz", **state_arrays(arm["states"]))
    write_npz_atomic(
        partial / "predictions.npz",
        sph_ecg_ids=rows["ecg_id"].to_numpy(dtype=str),
        sph_patient_ids=rows["patient_id"].to_numpy(dtype=str),
        sph_evaluation=evaluation, sph_in_043_cache=~np.isin(np.arange(len(rows)), positions),
        development_record_ids=data["development"].index.to_numpy(dtype=str),
        **{f"{part}_{name}": values[part] for name, values in (
            ("v2_binary", frozen["v2"]), ("v3_binary", frozen["v3"]), ("v3_logit", frozen["v3_logit"]),
            ("attention_jepa", attention), ("v4_logit", v4_logit)) for part in ("sph", "development")},
        sph_v4_binary=pipelines["v4"]["binary"],
        **{f"development_attention_jepa_seed{seed}": arm["seed_logits"][seed]["development"]
           for seed in SEEDS},
        **{f"sph_attention_jepa_seed{seed}": seed_sph(arm, seed, sets["sph"]["positions"], positions,
                                                     len(rows)) for seed in SEEDS})
    return {"checks": checks, "validation_split": split_counts, "profile": gpu["profile"],
            "cost": gpu["cost"], "new_cache": {"folder": str(CACHE046), "receipt": gpu["receipt"]},
            "seeds": arm["seeds"], "recipe": vars(exp043.ATTENTION_RECIPE), "counts": counts,
            "empty_resamples": found["empty_resamples"], "summary": found["summary"],
            "contrasts": found["contrasts"], "local_selection": found["local_selection"],
            "matched_rate": matched, "auroc": aurocs, "reading": found["reading"],
            "resamples": exp044.RESAMPLES, "sizes": list(exp044.SIZES)}


def seed_sph(arm: dict[str, Any], seed: int, labeled: np.ndarray, new: np.ndarray, length: int) -> np.ndarray:
    """Return one seed's logits on every SPH row, from the labeled and the new parts."""
    values = np.full(length, np.nan)
    values[labeled] = arm["seed_logits"][seed]["sph_labeled"]
    values[new] = arm["seed_logits"][seed]["new"]
    return values


def smoke_pipelines(data: dict[str, Any], split: dict[str, Any], attention: np.ndarray
                    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    """
    Fit the smoke readouts and finding heads as Experiment 044's smoke; build v2, v3 and v4 on the held part.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    split : dict[str, Any]
        Output of Experiment 043's ``smoke_split``.
    attention : np.ndarray
        A's logits on the held-out rows.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]
        Pipelines, the outcome masks over the held-out rows and the fit details.
    """
    binary_rows, y = data["readouts"]["binary"]["rows"], data["design"]["y"]
    held = binary_rows[split["held_out"]]
    allowed = np.ones(len(data["x"]["xecg"]), dtype=bool)
    allowed[held] = False
    fit_normals = binary_rows[split["fit"][y[split["fit"]] == 0]][:exp044.SMOKE_NORMALS]
    stacked = {"xecg": data["x"]["xecg"], "concat": np.concatenate([data["x"]["xecg"], data["x"]["jepa"]],
                                                                   axis=1)}
    sets = {name: {"stacked": values, "held_out": values[held], "normals": values[fit_normals]}
            for name, values in stacked.items()}
    heads, scores, iterations = exp044.fit_heads(sets, exp044.head_plan(data, (split["fit"], allowed)),
                                                 ("held_out", "normals"))
    _, z = exp044.standardization(scores, "normals", None, "held_out")
    labels = {}
    for name, group in exp044.FINDING_HEADS.items():
        values = np.zeros(len(stacked["xecg"]), dtype=bool)
        readout = data["readouts"][group]
        values[readout["rows"][readout["y"] == 1]] = True
        labels[name] = values[held]
    positive = y[split["held_out"]] == 1
    composite = positive | labels["pvc"] | labels["wpw"]
    outcomes = {"normal": ~positive, "binary_positive": positive, "composite": composite,
                "pvc": labels["pvc"], "wpw": labels["wpw"], "finding_only": composite & ~positive}
    findings = {"pvc": z["pvc"], "wpw": z["wpw"], "combined": np.maximum(z["pvc"], z["wpw"])}
    v3_logit = exp043.logits_of(heads["v3"], sets["concat"]["held_out"])
    pipelines = {"v2": {"binary": scores["v2"]["held_out"], **findings},
                 "v3": {"binary": scores["v3"]["held_out"], **findings},
                 "v4": {"binary": expit(ensemble_logit(v3_logit, attention)), **findings}}
    return pipelines, outcomes, {"iterations": iterations, "v3_logit_vs_probability": float(np.abs(
        expit(v3_logit) - scores["v3"]["held_out"]).max())}


def run_smoke(data: dict[str, Any], partial: Path, started: float) -> dict[str, Any]:
    """
    Run every code path on training rows only: no development, calibration or SPH ECG is read or scored.

    A is trained for a few epochs on 043's ``smoke_split`` fit rows (tokens from 043's cache); a few held-out
    training rows are extracted again into a smoke cache; the held-out part is split by group into a local
    pool and an evaluation half, as Experiment 044's smoke.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    partial : Path
        Output folder.
    started : float
        ``time.perf_counter()`` at the start of the run.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    split = exp043.smoke_split(data)
    sets = exp043.evaluation_sets(data)
    checks: dict[str, Any] = {"cache043": checked_cache043(data, sets)}
    y, weights = data["design"]["y"], data["design"]["weights"]
    fit_rows, check_rows = split["fit"][~split["validation"]], split["fit"][split["validation"]]
    fit = {"rows": fit_rows, "y": y[fit_rows], "weights": weights[fit_rows], "check_rows": check_rows,
           "check_y": y[check_rows]}
    spread = np.unique(np.linspace(0, len(split["held_out"]) - 1, SMOKE_EXTRACT).astype(int))
    picked = split["held_out"][spread]
    new = {"table": data["stacked"].iloc[picked].reset_index(drop=True),
           "jepa": exp043.binary_features(data, "jepa", "train")[picked],
           "xecg": exp043.binary_features(data, "xecg", "train")[picked]}
    recipe = Recipe(exp043.ATTENTION_RECIPE.learning_rate, exp043.ATTENTION_RECIPE.weight_decay,
                    exp043.ATTENTION_RECIPE.batch_size, exp043.SMOKE_EPOCHS, exp043.ATTENTION_RECIPE.patience)
    gpu = gpu_stage(new, SMOKE_CACHE, fit, {"held_out": split["held_out"]}, recipe, started)
    arm = gpu["arm"]
    where = np.searchsorted(split["held_out"], picked)
    checks["new_cache_vs_043_cache_logits"] = float(np.abs(arm["logits"]["new"]
                                                           - arm["logits"]["held_out"][where]).max())
    pipelines, outcomes, fitted = smoke_pipelines(data, split, arm["logits"]["held_out"])
    groups = data["groups"][split["held_out"]]
    evaluation = exp043.validation_mask(groups, 0.5, exp044.DRAW_SEED)
    masks = {name: values[evaluation] for name, values in outcomes.items()}
    local_masks = {name: outcomes[name][~evaluation] for name in exp044.LOCAL_OUTCOMES}
    local_normal = np.flatnonzero(~evaluation & outcomes["normal"])
    found = screens(pipelines, evaluation, masks, local_masks, local_normal, groups[evaluation],
                    exp044.SMOKE_SIZES, exp044.SMOKE_PRIMARY_SIZE, exp044.SMOKE_RESAMPLES)
    matched = matched_rates(pipelines, evaluation, masks, local_normal, groups[evaluation],
                            exp044.SMOKE_RESAMPLES, SMOKE_RATE_OUTCOMES)
    held_y = y[split["held_out"]][evaluation]
    auroc = {name: float(roc_auc_score(held_y, values["binary"][evaluation]))
             for name, values in pipelines.items()}
    auroc["A"] = float(roc_auc_score(held_y, arm["logits"]["held_out"][evaluation]))
    auroc["v4_minus_v3"] = paired_auroc_difference(
        groups[evaluation], held_y, pipelines["v4"]["binary"][evaluation],
        pipelines["v3"]["binary"][evaluation], exp044.SMOKE_RESAMPLES, exp044.BOOTSTRAP_SEED)
    found["draws"].to_csv(partial / "draws.csv", index=False)
    write_npz_atomic(partial / "attention_jepa_weights.npz", **state_arrays(arm["states"]))
    write_npz_atomic(partial / "predictions.npz", held_out_rows=split["held_out"],
                     held_out_attention_jepa=arm["logits"]["held_out"],
                     **{f"held_out_{name}_binary": values["binary"] for name, values in pipelines.items()})
    checks["weights_reload_max_abs_difference"] = gpu["weights_check"]
    return {"checks": checks, "fitted": fitted, "profile": gpu["profile"], "cost": gpu["cost"],
            "new_cache": {"folder": str(SMOKE_CACHE), "receipt": gpu["receipt"]}, "seeds": arm["seeds"],
            "recipe": vars(recipe),
            "held_out": {"evaluation": int(evaluation.sum()), "local_normals": int(len(local_normal)),
                         **{name: int(values.sum()) for name, values in masks.items()}},
            "empty_resamples": found["empty_resamples"], "summary": found["summary"],
            "contrasts": found["contrasts"], "local_selection": found["local_selection"],
            "matched_rate": matched, "auroc": auroc, "reading": found["reading"],
            "resamples": exp044.SMOKE_RESAMPLES, "sizes": list(exp044.SMOKE_SIZES),
            "note": "smoke: training rows only; the readings are not results"}


def run(smoke: bool, output: Path) -> None:
    """
    Run the experiment (or its smoke test) end to end and write its outputs.

    Parameters
    ----------
    smoke : bool
        Training-only smoke test.
    output : Path
        Final output folder; it must not exist.

    Raises
    ------
    RuntimeError
        If no GPU is visible.
    """
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    handler = logging.FileHandler(partial / "run.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), handler])
    torch.set_num_threads(THREADS)
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 046 needs a GPU")
    receipts = input_receipts()
    data = exp044.training_data043()
    LOG.info("loaded %s at %.1f s", data["training_counts"], time.perf_counter() - started)
    run_identity = identity(receipts, data["identity"])
    found = run_smoke(data, partial, started) if smoke else run_full(data, partial, started)
    outputs = {path.name: sha256_file(path) for path in sorted(partial.iterdir())
               if path.is_file() and path.name not in ("result.json", "run.log")}
    LOG.info("done in %.1f s; renaming %s to %s", time.perf_counter() - started, partial.name, output.name)
    handler.close()
    logging.getLogger().removeHandler(handler)
    write_json_atomic(partial / "result.json", {
        "status": "smoke" if smoke else "complete", "identity": run_identity,
        "training_counts": data["training_counts"],
        "pipelines": {"v2": "037 pipeline v2 (frozen scores)", "v3": "044 pipeline v3 (frozen scores)",
                      "v4": "v3 with the binary readout E = (R3 logit + A logit) / 2, ranked by expit(E)"},
        "rules": {name: {"columns": list(columns), "shares_per_mille": list(shares)}
                  for name, (columns, shares) in exp044.RULES033.items()},
        "training_seeds": list(SEEDS), "draw_seed": exp044.DRAW_SEED, "bootstrap_seed": exp044.BOOTSTRAP_SEED,
        "draws": exp044.DRAWS, "budgets": [budget / 1000 for budget in exp044.BUDGETS],
        "prevalence": exp044.PREVALENCE, **found, "outputs_sha256": outputs,
        "total_seconds": time.perf_counter() - started, "challenge_test_read": False,
        "challenge_calibration_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
    })
    partial.rename(output)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="training-only smoke test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_name(arguments.output.name + ".partial").exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    run(arguments.smoke, arguments.output)


if __name__ == "__main__":
    main()
