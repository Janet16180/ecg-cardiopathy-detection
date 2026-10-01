"""Experiment 050: attention finding heads (PVC and WPW) on frozen ECG-JEPA tokens.

Trains Experiment 043 Stage 2's attention head on 032's ``ventricular_ectopy`` and ``preexcitation`` labels,
compares its detection with the xECG finding heads of pipelines v2-v4 on SPH and PTB-XL development, tests
whether the PVC head's token map points at premature beats (042's test), and, if it does, scores 048's
explanation rule with the attention PVC head as switch and rhythm layer.

The row logic lives in pinned runners (Experiments 032, 037, 042, 043, 044 and 045), reached through the 043,
044 and 045 runners as documented in the protocol.
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
from sklearn.metrics import average_precision_score, roc_auc_score

import ecg_experiment
from ecg_experiment.ann_heads import AttentionHead, Recipe, open_row_file, predict, validation_mask
from ecg_experiment.attention_findings import (
    finding_reading,
    located_batch,
    token_locator,
    top_unit,
    train_seeds,
    unlabeled_sph_table,
)
from ecg_experiment.explanation_rule import (
    explanation_metrics,
    paired_difference,
    referral_threshold,
    referred_ids,
)
from ecg_experiment.external_encoders import JEPA_CHECKPOINT, load_jepa
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import jepa_unit_map, premature_hit
from ecg_experiment.paths import to_stored
from ecg_experiment.pipeline_v4 import load_state, state_arrays
from ecg_experiment.provenance import git_head
from ecg_experiment.pvc_switch import pvc_z, switch_threshold
from scripts.experiments import run_ann_heads043 as exp043
from scripts.experiments import run_pipeline_v3_044 as exp044
from scripts.experiments import run_two_layer_map045 as exp045

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment050_attention_findings_v1"
PRIOR044 = ROOT / "outputs/experiment044_pipeline_v3_v1"
PRIOR046 = ROOT / "outputs/experiment046_pipeline_v4_v1"
PRIOR048 = ROOT / "outputs/experiment048_pvc_switch_v1"
CACHE043 = exp043.CACHE
CACHE046 = CACHE043.parent / "exp046_cache"
CACHE050 = CACHE043.parent / "exp050_cache"
SMOKE_CACHE = CACHE043.parent / "smoke050_cache"
PROTOCOL = "docs/experiment-050-attention-finding-heads.md"
SOURCES = (
    "ecg_experiment/attention_findings.py", "ecg_experiment/ann_heads.py", "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/explanation_rule.py", "ecg_experiment/pvc_switch.py", "ecg_experiment/pipeline_v4.py",
    "ecg_experiment/pipeline_v2.py", "ecg_experiment/finding_screen.py", "ecg_experiment/intervals.py",
    "ecg_experiment/fragment_localization.py", "ecg_experiment/external_encoders.py", "ecg_experiment/sph.py",
    "ecg_experiment/challenge_features.py", "ecg_experiment/gpu.py",
    "scripts/experiments/run_attention_findings050.py", "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_pipeline_v3_044.py", "scripts/experiments/run_two_layer_map045.py",
    "scripts/experiments/run_pipeline_v2_037.py", "scripts/experiments/run_rhythm_findings032.py",
    "scripts/experiments/run_lead_wave_maps042.py", "pyproject.toml", "uv.lock", PROTOCOL,
)
DEVICE = "cuda"
HEADS = {"pvc": "ventricular_ectopy", "wpw": "preexcitation"}
SEEDS = (50050, 50051, 50052)
SEED = 50050
SEED048 = 48048
DRAWS = 2000
VALIDATION_SHARE = 0.10
RECIPE = Recipe(learning_rate=3e-4, weight_decay=1e-2, batch_size=64, max_epochs=30, patience=4)
EXPECTED_READOUTS = {"pvc": [54910, 2437], "wpw": [51229, 124]}
EXPECTED_EXTRA = 15434
EXPECTED_SETS = {"pvc": {"sph": [25577, 1058], "development": [1600, 84]},
                 "wpw": {"sph": [25566, 27], "development": [1604, 6]}}
Z_TOLERANCE = 1e-10
AUROC_TOLERANCE = 1e-12
WEIGHT_TOLERANCE = 1e-6
LOCALIZATION_MARGIN = -0.10
SWITCH_QUANTILE = 0.975
EXAMPLE_PVC = 219
CEILING = 7200.0
THREADS = 4
SMOKE_ROWS = 4000
SMOKE_EXTRACT = 64
SMOKE_EPOCHS = 2
SMOKE_DRAWS = 200
LOG = logging.getLogger("experiment050")


def input_receipts() -> dict[str, str]:
    """Return the hashes of the earlier outputs this run reads, each checked against its receipt."""
    found = {}
    for directory, names in ((exp043.PRIOR_STAGE1, ("predictions.npz",)),
                             (exp043.PRIOR_STAGE2, ("predictions.npz", "token_maps.npz")),
                             (PRIOR044, ("predictions.npz", "pipeline_v3_heads.npz")),
                             (PRIOR048, ("explanations.npz",)), (exp043.PRIOR042, ("unit_scores.npz",))):
        found.update({to_stored(directory / name): digest
                      for name, digest in exp044.receipt_checked(directory, names).items()})
    found[to_stored(PRIOR046 / "result.json")] = sha256_file(PRIOR046 / "result.json")
    return found


def identity(receipts: dict[str, Any], data_identity: dict[str, Any]) -> dict[str, Any]:
    """
    Hash every source and the protocol, with the input receipts and the run environment.

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
            "jepa_checkpoint_sha256": sha256_file(JEPA_CHECKPOINT), "torch_threads": torch.get_num_threads()}


def stacked_table(data: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rebuild Experiment 043's reader table and groups for all stacked rows, checked on the binary rows.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Reader rows and the validation group of every stacked row.

    Raises
    ------
    ValueError
        If the rows, the binary-row table or the groups differ from 043's.
    """
    ptb, _, _ = exp043.ptb_inputs()
    rows, _ = exp043.challenge_table()
    train_rows = rows[(rows["split"] == "train").to_numpy()].copy()
    saved = exp043.checked_csv(exp043.PRIOR032, "training_rows.csv", dtype={"record": str},
                               keep_default_na=False)
    if not all(np.array_equal(saved[column].to_numpy(dtype=str), train_rows[column].to_numpy(dtype=str))
               for column in ("source", "record")):
        raise ValueError("Challenge training rows differ from 032's")
    challenge = train_rows[saved["reasons"].to_numpy(dtype=str) == ""]
    train = ptb["train"]
    table = pd.DataFrame({
        "kind": np.concatenate([np.full(len(train), "ptbxl"), np.full(len(challenge), "challenge")]),
        "key": np.concatenate([("ptbxl:" + train["ecg_id"].astype(str)).to_numpy(dtype=str),
                               (challenge["source"] + ":" + challenge["record"]).to_numpy(dtype=str)]),
        "filename_hr": np.concatenate([train["filename_hr"].to_numpy(dtype=str),
                                       np.full(len(challenge), "")]),
        "source": np.concatenate([np.full(len(train), ""), challenge["source"].to_numpy(dtype=str)]),
        "path": np.concatenate([np.full(len(train), ""), challenge["path"].to_numpy(dtype=str)]),
        "window_start": np.concatenate([np.full(len(train), -1),
                                        challenge["window_start"].to_numpy(np.int64)]),
        "window_sha256": np.concatenate([np.full(len(train), ""),
                                         challenge["window_sha256"].fillna("").to_numpy(dtype=str)]),
    })
    groups = np.concatenate([train["patient_id"].to_numpy(dtype=str),
                             (challenge["source"] + ":" + challenge["record"]).to_numpy(dtype=str)])
    binary = data["readouts"]["binary"]["rows"]
    if (len(table) != len(data["x"]["xecg"])
            or not table.iloc[binary].reset_index(drop=True).equals(data["stacked"])
            or not np.array_equal(groups[binary], data["groups"])):
        raise ValueError("The rebuilt stacked table differs from 043's")
    return table, groups


def head_rows(data: dict[str, Any], validation: np.ndarray, allowed: np.ndarray | None = None
              ) -> dict[str, dict[str, Any]]:
    """
    Return each finding head's fit and validation rows and targets, as stacked positions.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    validation : np.ndarray
        Validation mask over the stacked rows.
    allowed : np.ndarray | None
        Mask of the stacked rows a head may use (smoke mode); ``None`` keeps all.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per head, ``train_seeds``'s ``fit`` fields and the counts.
    """
    found = {}
    for name, group in HEADS.items():
        rows, y = data["readouts"][group]["rows"], data["readouts"][group]["y"]
        if allowed is not None:
            rows, y = rows[allowed[rows]], y[allowed[rows]]
        check = validation[rows]
        found[name] = {"rows": rows[~check], "y": y[~check], "weights": np.ones(int((~check).sum())),
                       "check_rows": rows[check], "check_y": y[check],
                       "counts": {"fit": int((~check).sum()), "fit_positive": int(y[~check].sum()),
                                  "validation": int(check.sum()), "validation_positive": int(y[check].sum())}}
    return found


def checked_caches(data: dict[str, Any], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Require 043's and 046's token caches to match their recorded receipts (rehashed).

    Returns
    -------
    dict[str, Any]
        The receipts, 043's part boundaries and the SPH positions held in 046's cache.

    Raises
    ------
    ValueError
        If a cache differs from its recorded receipt.
    """
    began = time.perf_counter()
    table043, _, parts = exp043.extraction_table(data, sets)
    receipt043 = exp043.cache_receipt(CACHE043, table043, verify=True)
    recorded043 = json.loads((exp043.PRIOR_STAGE2 / "result.json").read_text())["cache"]["receipt"]
    table046, positions046 = unlabeled_sph_table(data["sph"], sets["sph"]["positions"])
    receipt046 = exp043.cache_receipt(CACHE046, table046, verify=True)
    recorded046 = json.loads((PRIOR046 / "result.json").read_text())["new_cache"]["receipt"]
    if receipt043 != recorded043 or receipt043["parts"] != parts or receipt046 != recorded046:
        raise ValueError("A token cache differs from its recorded receipt")
    LOG.info("043 and 046 caches verified in %.1f s", time.perf_counter() - began)
    return {"043": receipt043, "046": receipt046, "parts": parts, "positions046": positions046}


def gpu_stage(new: dict[str, Any], cache: Path, stores: list[Path], parts: list[tuple[int, Any, Any]],
              total: int, fits: dict[str, dict[str, Any]], score: dict[str, np.ndarray], keep: str,
              recipe: Recipe, started: float) -> dict[str, Any]:
    """
    Profile, extract the new tokens, train both heads with every seed and check the saved weights, on the GPU.

    Parameters
    ----------
    new : dict[str, Any]
        ``table`` and cached ``jepa`` features of the rows to extract, and their ``virtual`` rows.
    cache : Path
        New cache folder.
    stores : list[Path]
        Existing token files; the new cache is appended after them.
    parts : list[tuple[int, Any, Any]]
        Locator parts of the existing caches.
    total : int
        Number of virtual rows.
    fits : dict[str, dict[str, Any]]
        Output of ``head_rows``.
    score : dict[str, np.ndarray]
        Virtual rows per scored part.
    keep : str
        Part whose PVC contributions are kept.
    recipe : Recipe
        Training recipe.
    started : float
        ``time.perf_counter()`` at the start of the run.

    Returns
    -------
    dict[str, Any]
        ``profile``, the new cache ``receipt``, ``heads`` (``train_seeds`` per head) and ``weights_check``.

    Raises
    ------
    RuntimeError
        If the projected time exceeds ``CEILING``.
    """
    readers = (exp043.load_checksums(), {record: window for record, _, window in exp043.ningbo_items()[0]})
    with gpu_lock(DEVICE):
        opened = [open_row_file(path) for path in stores]
        partial_locator = token_locator(parts, total)
        encoder = load_jepa()
        profile: dict[str, Any] = {"extraction": exp043.profile_extraction(new["table"], encoder, *readers)}
        ready = partial_locator[fits["pvc"]["rows"], 0] >= 0
        profile["training"] = exp043.profile_training(
            lambda: AttentionHead(exp043.TOKEN_SHAPE[1]), located_batch(opened, partial_locator, DEVICE),
            fits["pvc"]["rows"][ready], fits["pvc"]["y"][ready], fits["pvc"]["weights"][ready], recipe,
            DEVICE)
        scored = sum(len(rows) for rows in score.values())
        per_record = profile["extraction"]["read_per_record"] + profile["extraction"]["model_per_record"]
        profile["projected_extraction_seconds"] = per_record * len(new["table"])
        profile["projected_training_seconds"] = sum(exp043.projected_training(
            profile["training"], len(fit["rows"]), len(fit["check_rows"]), scored, recipe)
            for fit in fits.values())
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
        del encoder
        torch.cuda.empty_cache()
        LOG.info("new cache ready in %.1f s: %s", profile["extraction_seconds"], receipt["token_check"])
        opened.append(open_row_file(cache / "tokens.npy"))
        locator = token_locator([*parts, (len(stores), new["virtual"], np.arange(len(new["table"])))], total)
        batch = located_batch(opened, locator, DEVICE)
        torch.cuda.reset_peak_memory_stats()
        heads, weights_check = {}, {}
        for name, fit in fits.items():
            began = time.perf_counter()
            heads[name] = train_seeds(lambda: AttentionHead(exp043.TOKEN_SHAPE[1]), batch, fit, score, recipe,
                                      SEEDS, DEVICE, keep=(keep,) if name == "pvc" else (),
                                      log=lambda line, head=name: LOG.info("attention_%s %s", head, line))
            heads[name]["seconds"] = time.perf_counter() - began
            weights_check[name] = check_weights(heads[name], batch, score[keep], keep)
            LOG.info("attention_%s trained in %.1f s: %s", name, heads[name]["seconds"],
                     {seed: info["best_epoch"] for seed, info in heads[name]["seeds"].items()})
        profile["training_seconds"] = sum(head["seconds"] for head in heads.values())
        profile["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    for store in opened:
        os.close(store.descriptor)
    return {"profile": profile, "receipt": receipt, "heads": heads, "weights_check": weights_check,
            "locator": locator}


def check_weights(head: dict[str, Any], batch: Any, rows: np.ndarray, part: str) -> float:
    """
    Require fresh networks loaded from the saved arrays to reproduce each seed's logits of one part.

    Raises
    ------
    ValueError
        If the largest difference exceeds ``WEIGHT_TOLERANCE``.
    """
    arrays = state_arrays(head["states"])
    worst = 0.0
    for seed in SEEDS:
        model = load_state(AttentionHead(exp043.TOKEN_SHAPE[1]), arrays, seed).to(DEVICE)
        logits, _ = predict(model, batch, rows, RECIPE.batch_size * 4)
        worst = max(worst, float(np.abs(logits - head["seed_logits"][seed][part]).max()))
        del model
    if worst > WEIGHT_TOLERANCE:
        raise ValueError(f"Saved weights do not reproduce the logits: {worst}")
    return worst


def comparator(data: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """
    Score the xECG PVC and WPW heads of pipelines v2-v4 as z-scores, checked against Experiment 044.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]
        z-scores per head and part (``sph``, ``development``), and the checks.

    Raises
    ------
    ValueError
        If the SPH z-scores or AUROCs differ from 044's.
    """
    with np.load(PRIOR044 / "pipeline_v3_heads.npz") as saved:
        parameters = {key: saved[key] for key in saved.files}
    with np.load(PRIOR044 / "predictions.npz") as saved:
        if not np.array_equal(saved["sph_ecg_ids"], data["sph"]["ecg_id"].to_numpy(dtype=str)):
            raise ValueError("044 prediction rows differ")
        saved_z = {name: saved[f"sph_v2_z_{name}"] for name in HEADS}
    prior = json.loads((PRIOR044 / "result.json").read_text())["auroc"]["findings"]
    z, checks = {}, {}
    for name, group in HEADS.items():
        z[name] = {part: pvc_z(parameters, data[f"{part}_x"]["xecg"], name)
                   for part in ("sph", "development")}
        defined = data["sph"][group].notna().to_numpy()
        y = (data["sph"].loc[defined, group] == 1).to_numpy()
        auroc = float(roc_auc_score(y, z[name]["sph"][defined]))
        checks[name] = {"z_max_difference": float(np.abs(z[name]["sph"] - saved_z[name]).max()),
                        "auroc_difference": abs(auroc - prior[name]["auroc"]["xecg"])}
        if (checks[name]["z_max_difference"] > Z_TOLERANCE
                or checks[name]["auroc_difference"] > AUROC_TOLERANCE):
            raise ValueError(f"The xECG {name} head does not reproduce 044: {checks[name]}")
    return z, checks


def detection(data: dict[str, Any], heads: dict[str, dict[str, Any]], z: dict[str, dict[str, np.ndarray]],
              draws: int) -> dict[str, Any]:
    """
    Compare each attention head with its xECG head on SPH and full development, with the readings.

    Returns
    -------
    dict[str, Any]
        Per head and set: counts, AUROC and AP of both heads, each seed's AUROC, the paired difference and
        its reading.

    Raises
    ------
    ValueError
        If a set's counts differ from the protocol.
    """
    found: dict[str, Any] = {}
    for name, group in HEADS.items():
        found[name] = {}
        for part in ("sph", "development"):
            frame = data[part]
            defined = frame[group].notna().to_numpy()
            y = (frame.loc[defined, group] == 1).to_numpy(dtype=np.int64)
            patients = frame.loc[defined, "patient_id"].to_numpy(dtype=str)
            if [len(y), int(y.sum())] != EXPECTED_SETS[name][part]:
                raise ValueError(f"{name} {part} counts differ from the protocol")
            scores = {"attention": heads[name]["logits"][part][defined], "xecg": z[name][part][defined]}
            difference = paired_auroc_difference(patients, y, scores["attention"], scores["xecg"], draws,
                                                 SEED)
            found[name][part] = {
                "records": len(y), "positives": int(y.sum()), "patients": int(len(np.unique(patients))),
                "auroc": {key: float(roc_auc_score(y, values)) for key, values in scores.items()},
                "average_precision": {key: float(average_precision_score(y, values))
                                      for key, values in scores.items()},
                "seed_auroc": {str(seed): float(roc_auc_score(
                    y, heads[name]["seed_logits"][seed][part][defined])) for seed in SEEDS},
                "attention_minus_xecg": difference, "reading": finding_reading(difference["ci_low"])}
    return found


def map_analysis(inputs: dict[str, Any], contributions: np.ndarray, draws: int) -> dict[str, Any]:
    """
    Run 042's map tests on the PVC head's map, against ``U_B`` and ``attention_jepa``, and the example ECG.

    Returns
    -------
    dict[str, Any]
        Thresholds and metrics of the three maps, both paired contrasts, the reading and the example's top
        units, plus the unit maps (``units``) for the secondary rule.
    """
    pvc_maps = [jepa_unit_map(values) for values in contributions]
    maps = {"U_B": inputs["U_B"], "attention_jepa": inputs["attention_jepa"], "attention_pvc": pvc_maps}
    metrics = exp043.map_metrics(inputs["rows"], maps, inputs["windows"], draws, SEED)
    versus = {base: exp043.map_contrasts(metrics, "attention_pvc", base, draws, SEED)
              for base in ("U_B", "attention_jepa")}
    low = versus["U_B"]["hit_minus_chance"]["ci_low"]
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    example = {}
    windows = inputs["windows"].get(EXAMPLE_PVC, [])
    for name, unit_maps in maps.items():
        unit_map = dict(zip(ids, unit_maps, strict=True))[EXAMPLE_PVC]
        hit, chance = premature_hit(unit_map, windows)
        example[name] = {**top_unit(unit_map), "on_premature_beat": bool(hit), "chance": chance}
    return {"thresholds": metrics["thresholds"], "maps": metrics["maps"],
            "attention_pvc_minus_U_B": versus["U_B"],
            "attention_pvc_minus_attention_jepa": versus["attention_jepa"],
            "keeps_premature_localization": bool(low > LOCALIZATION_MARGIN),
            "example_windows": windows, "example": {str(EXAMPLE_PVC): example},
            "units": {name: dict(zip(ids, unit_maps, strict=True)) for name, unit_maps in maps.items()}}


def rule_units(referred: list[int], score: dict[int, float], switch: float, rhythm: dict[int, Any],
               other: dict[int, Any]) -> tuple[dict[int, Any], dict[int, int]]:
    """Return each referred ECG's explaining units (rhythm layer when the switch is on) and its layer."""
    layers = {i: 1 if score[i] > switch else 2 for i in referred}
    return {i: rhythm[i] if layers[i] == 1 else other[i] for i in referred}, layers


def explanation_rules(inputs: dict[str, Any], data: dict[str, Any], attention_pvc: np.ndarray,
                      units: dict[str, dict[int, Any]], draws: int) -> dict[str, Any]:
    """
    Score 048's rule and the 050 rule on 048's referred ECGs, after reproducing 048's primary case.

    Raises
    ------
    ValueError
        If 048's primary case does not reproduce.
    """
    groups = exp045.groups_of(inputs["rows"])
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    if not np.array_equal(data["development"]["ecg_id"].to_numpy(np.int64), np.array(ids)):
        raise ValueError("Development rows differ from 042's scored rows")
    logits = inputs["scores"]["logistic_concat"]["development"]
    referred = referred_ids(ids, logits, referral_threshold(logits[np.isin(ids, groups["normal"])]))
    with np.load(PRIOR044 / "pipeline_v3_heads.npz") as saved:
        parameters = {key: saved[key] for key in saved.files}
    scores = {"048": dict(zip(ids, pvc_z(parameters, data["development_x"]["xecg"]).tolist(), strict=True)),
              "050": dict(zip(ids, attention_pvc.tolist(), strict=True))}
    switches = {name: switch_threshold(np.array([values[i] for i in groups["normal"]]), SWITCH_QUANTILE)
                for name, values in scores.items()}
    rules = {"048": rule_units(referred, scores["048"], switches["048"], units["U_B"],
                               units["attention_jepa"]),
             "050": rule_units(referred, scores["050"], switches["050"], units["attention_pvc"],
                               units["attention_jepa"])}
    args = (groups["patients"], inputs["windows"], groups["anterior"], groups["inferior"], draws)
    named = {"anterior": groups["anterior"], "inferior": groups["inferior"],
             "pvc_window": list(inputs["windows"])}
    chosen = set(referred)

    def sent(layers: dict[int, int]) -> dict[str, Any]:
        return {name: {"count": int(sum(layers[i] == 1 for i in members if i in chosen)),
                       "referred": int(sum(i in chosen for i in members))} for name, members in named.items()}

    prior = json.loads((PRIOR048 / "result.json").read_text())
    case = prior["cases"]["logistic_concat_q0.975"]
    reproduced = explanation_metrics(rules["048"][0], *args, SEED048)["metrics"]
    checks = {"referred": len(referred) == case["referred"],
              "switch": switches["048"] == prior["switch_thresholds"]["0.975"],
              "sent_to_U_B": {k: v["count"] for k, v in sent(rules["048"][1]).items()} == case["sent_to_U_B"],
              "maps_rule": exp045.plain(reproduced) == case["maps"]["rule"]}
    if not all(checks.values()):
        raise ValueError(f"048's primary case does not reproduce: {checks}")
    scored = {name: explanation_metrics(found[0], *args, SEED) for name, found in rules.items()}
    paired = paired_difference(scored["050"], scored["048"], groups["patients"], draws, SEED)
    reading = {"keeps_048_localization": bool(paired["hit_minus_chance"]["ci_low"] > LOCALIZATION_MARGIN),
               "anterior_gain": bool(paired["lead_contrast"]["anterior"]["ci_low"] > 0)}
    reading["improves_on_048"] = reading["keeps_048_localization"] and reading["anterior_gain"]
    return {"integrity_048": checks, "referred": len(referred), "switch_thresholds": switches,
            "metrics": {name: found["metrics"] for name, found in scored.items()},
            "sent_to_rhythm_layer": {name: sent(found[1]) for name, found in rules.items()},
            "rule050_minus_rule048": paired, "reading": reading}


def run_full(data: dict[str, Any], partial: Path, started: float) -> dict[str, Any]:
    """
    Run the experiment: checks, token extraction, both heads, detection, the map and the secondary rule.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    checks: dict[str, Any] = {"training_rows_vs_037": exp043.check_against_037(data)}
    table, groups = stacked_table(data)
    sets = exp043.evaluation_sets(data)
    caches = checked_caches(data, sets)
    z, checks["comparator"] = comparator(data)
    inputs = exp045.load_inputs()
    checks["U_B_vs_042"] = exp045.check_042(inputs)
    checks["maps_vs_043"] = exp045.check_043_maps(inputs)
    validation = validation_mask(groups, VALIDATION_SHARE, SEED)
    fits = head_rows(data, validation)
    readouts = {name: [len(data["readouts"][g]["rows"]), int(data["readouts"][g]["y"].sum())]
                for name, g in HEADS.items()}
    binary = data["readouts"]["binary"]["rows"]
    union = np.zeros(len(table), dtype=bool)
    for group in HEADS.values():
        union[data["readouts"][group]["rows"]] = True
    union[binary] = False
    extra = np.flatnonzero(union)
    if readouts != EXPECTED_READOUTS or len(extra) != EXPECTED_EXTRA:
        raise ValueError(f"Readout rows differ from the protocol: {readouts}, {len(extra)}")
    n_stacked, n_dev, n_sph = len(table), len(data["development"]), len(data["sph"])
    total = n_stacked + n_dev + n_sph
    parts043, labeled = caches["parts"], sets["sph"]["positions"]
    parts = [(0, binary, np.arange(len(binary))),
             (0, n_stacked + np.arange(n_dev), np.arange(*parts043["development"])),
             (0, n_stacked + n_dev + labeled, np.arange(*parts043["sph"])),
             (1, n_stacked + n_dev + caches["positions046"], np.arange(len(caches["positions046"])))]
    score = {"development": n_stacked + np.arange(n_dev), "sph": n_stacked + n_dev + np.arange(n_sph)}
    new = {"table": table.iloc[extra].reset_index(drop=True), "jepa": data["x"]["jepa"][extra],
           "virtual": extra}
    gpu = gpu_stage(new, CACHE050, [CACHE043 / "tokens.npy", CACHE046 / "tokens.npy"], parts, total, fits,
                    score, "development", RECIPE, started)
    heads = gpu["heads"]
    found = detection(data, heads, z, DRAWS)
    LOG.info("detection %s", {name: {part: (v["auroc"], v["attention_minus_xecg"]["ci_low"], v["reading"])
                                     for part, v in sets_.items()} for name, sets_ in found.items()})
    maps = map_analysis(inputs, heads["pvc"]["contributions"]["development"], DRAWS)
    units = maps.pop("units")
    LOG.info("map: keeps %s, %s", maps["keeps_premature_localization"],
             maps["attention_pvc_minus_U_B"]["hit_minus_chance"])
    secondary = (explanation_rules(inputs, data, heads["pvc"]["logits"]["development"], units, DRAWS)
                 if maps["keeps_premature_localization"] else None)
    write_npz_atomic(partial / "attention_finding_weights.npz",
                     **{f"{name}.{key}": value for name, head in heads.items()
                        for key, value in state_arrays(head["states"]).items()})
    write_npz_atomic(partial / "predictions.npz", sph_ecg_ids=data["sph"]["ecg_id"].to_numpy(dtype=str),
                     development_record_ids=data["development"].index.to_numpy(dtype=str),
                     **{f"{part}_attention_{name}": head["logits"][part] for name, head in heads.items()
                        for part in score},
                     **{f"{part}_attention_{name}_seed{seed}": head["seed_logits"][seed][part]
                        for name, head in heads.items() for seed in SEEDS for part in score},
                     **{f"{part}_xecg_z_{name}": z[name][part] for name in HEADS for part in score})
    write_npz_atomic(partial / "token_maps.npz", ecg_ids=data["development"]["ecg_id"].to_numpy(np.int64),
                     attention_pvc=heads["pvc"]["contributions"]["development"].astype(np.float32),
                     **{f"attention_pvc_seed{seed}": heads["pvc"]["seed_contributions"][seed]["development"]
                        .astype(np.float32) for seed in SEEDS})
    checks["weights_reload_max_abs_difference"] = gpu["weights_check"]
    return {"checks": checks,
            "caches": {"043": caches["043"]["keys_sha256"], "046": caches["046"]["keys_sha256"],
                       "050": {"folder": str(CACHE050), "receipt": gpu["receipt"]}},
            "profile": gpu["profile"], "readouts": readouts, "extra_rows": len(extra),
            "split": {name: fit["counts"] for name, fit in fits.items()},
            "seeds": {name: head["seeds"] for name, head in heads.items()},
            "detection": found, "map": maps, "explanation_rule": secondary, "recipe": vars(RECIPE)}


def run_smoke(data: dict[str, Any], partial: Path, started: float) -> dict[str, Any]:
    """
    Run every GPU and statistics path on training rows only: no development or SPH ECG is read or scored.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    table, groups = stacked_table(data)
    sets = exp043.evaluation_sets(data)
    caches = checked_caches(data, sets)
    binary = data["readouts"]["binary"]["rows"]
    pvc_rows = data["readouts"]["ventricular_ectopy"]["rows"]
    in_binary = np.isin(pvc_rows, binary)
    rng = np.random.default_rng(SEED)
    wpw = data["readouts"]["preexcitation"]
    chosen = np.union1d(rng.choice(pvc_rows[in_binary], SMOKE_ROWS, replace=False),
                        np.intersect1d(wpw["rows"][wpw["y"] == 1], binary))
    outside = pvc_rows[~in_binary]
    extra = outside[np.unique(np.linspace(0, len(outside) - 1, SMOKE_EXTRACT).astype(int))]
    held = validation_mask(groups[chosen], 0.25, SEED + 1)
    allowed = np.zeros(len(table), dtype=bool)
    allowed[chosen[~held]] = True
    fits = head_rows(data, validation_mask(groups, VALIDATION_SHARE, SEED), allowed)
    parts = [(0, binary, np.arange(len(binary)))]
    score = {"held_out": np.concatenate([chosen[held], extra])}
    new = {"table": table.iloc[extra].reset_index(drop=True), "jepa": data["x"]["jepa"][extra],
           "virtual": extra}
    recipe = Recipe(RECIPE.learning_rate, RECIPE.weight_decay, RECIPE.batch_size, SMOKE_EPOCHS,
                    RECIPE.patience)
    gpu = gpu_stage(new, SMOKE_CACHE, [CACHE043 / "tokens.npy"], parts, len(table), fits, score, "held_out",
                    recipe, started)
    heads = gpu["heads"]
    rows = score["held_out"]
    smoke_detection = {}
    for name, group in HEADS.items():
        readout = data["readouts"][group]
        position = pd.Series(np.arange(len(readout["rows"])), index=readout["rows"])
        defined = np.isin(rows, data["readouts"][group]["rows"])
        y = data["readouts"][group]["y"][position.loc[rows[defined]].to_numpy()]
        if y.min() == y.max():
            smoke_detection[name] = {"records": int(defined.sum()), "positives": int(y.sum())}
            continue
        xecg = data["x"]["xecg"][rows[defined]].mean(axis=1)
        attention = heads[name]["logits"]["held_out"][defined]
        difference = paired_auroc_difference(groups[rows[defined]], y, attention, xecg, SMOKE_DRAWS, SEED)
        smoke_detection[name] = {"records": int(defined.sum()), "positives": int(y.sum()),
                                 "auroc": float(roc_auc_score(y, heads[name]["logits"]["held_out"][defined])),
                                 "difference": difference, "reading": finding_reading(difference["ci_low"])}
    unit_maps = [jepa_unit_map(values) for values in heads["pvc"]["contributions"]["held_out"]]
    write_npz_atomic(partial / "predictions.npz", held_out_rows=rows,
                     **{f"held_out_attention_{name}": head["logits"]["held_out"]
                        for name, head in heads.items()})
    return {"checks": {"caches": {"043": caches["043"]["keys_sha256"], "046": caches["046"]["keys_sha256"]},
                       "weights_reload_max_abs_difference": gpu["weights_check"]},
            "profile": gpu["profile"], "new_cache": gpu["receipt"],
            "split": {name: fit["counts"] for name, fit in fits.items()},
            "seeds": {name: head["seeds"] for name, head in heads.items()}, "detection": smoke_detection,
            "map_structure": {"maps": len(unit_maps), "top": top_unit(unit_maps[0])}, "recipe": vars(recipe),
            "note": "smoke: training rows only; the readings are not results"}


def run(smoke: bool, output: Path) -> None:
    """
    Run the experiment (or its smoke test) end to end and write its outputs.

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
        raise RuntimeError("Experiment 050 needs a GPU")
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
        "status": "smoke" if smoke else "complete", "identity": run_identity, "seeds_used": list(SEEDS),
        "validation_share": VALIDATION_SHARE, "validation_seed": SEED, "bootstrap_seed": SEED, "draws": DRAWS,
        **found, "outputs_sha256": outputs, "total_seconds": time.perf_counter() - started,
        "challenge_test_read": False, "challenge_calibration_read": False, "ptbxl_calibration_read": False,
        "ptbxl_test_read": False})
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
