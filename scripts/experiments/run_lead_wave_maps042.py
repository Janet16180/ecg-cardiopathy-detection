"""Experiment 042: per-lead anomaly maps from ECG-JEPA patches (arm J) and beat-aligned waves (arm B)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.external_encoders import jepa_cache, load_jepa, ptb_jepa_input, read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import (
    bootstrap_mean,
    pointing,
    premature_windows,
    r_peaks,
    readout_contributions,
    section_overlap,
)
from ecg_experiment.full_development import cohorts, fit_logistic, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    MINIMUM_BEATS,
    WAVES,
    UnitMap,
    beat_pieces,
    beat_unit_map,
    block_contributions,
    ecg_score,
    fit_lead_references,
    fit_token_reference,
    fit_wave_references,
    jepa_order_check,
    jepa_tokens,
    jepa_unit_map,
    lead_token_scores,
    lead_wave_unit_map,
    median_features,
    median_pieces,
    plot_lead_marks,
    premature_hit,
    red_marks,
    token_scores,
    top_lead,
    two_group_difference,
    wave_lengths,
    wave_scores,
)
from ecg_experiment.normal_manifold import fit_mahalanobis, mahalanobis_scores
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PRIOR026 = ROOT / "outputs/experiment026_normal_manifold_v1"
PRIOR041 = ROOT / "outputs/experiment041_fragment_localization_v1"
CACHE_IDS = (
    ROOT / "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
    ROOT / "outputs/experiment016_xecg_probe_finetune/features/ecg_ids.npy",
    ROOT / "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
)
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
PTB_FILES = (PTB_RAW / "ptbxl_database.csv", PTB_RAW / "scp_statements.csv")
PROTOCOL = "docs/experiment-042-lead-wave-maps.md"
SOURCES = (
    "ecg_experiment/lead_wave_maps.py", "ecg_experiment/fragment_localization.py",
    "ecg_experiment/external_encoders.py", "ecg_experiment/normal_manifold.py",
    "ecg_experiment/full_development.py", "ecg_experiment/intervals.py", "ecg_experiment/eda/ptbxl.py",
    "scripts/experiments/run_lead_wave_maps042.py", "pyproject.toml", "uv.lock", PROTOCOL,
)
EXPECTED = {"fit": (5872, 5537), "pool": (15359, 9487), "evaluation": (1306, 843), "development": 1604,
            "pvc": (84, 81), "benign": 52, "anterior": (146, 133), "inferior": (149, 139)}
BENIGN_CODES = {"NORM", "SR", "SBRAD", "SARRH"}
OTHER_MI = ("LMI", "PMI")
PRIMARY = {"J": "U_J", "B": "U_B"}
TIMED = ("U_J", "U_J_kmeans", "G_J", "U_B")
FIGURE_MAPS = ("U_J", "U_B", "G_J")
ORDER_RECORDS = 4
ORDER_CELLS = ((0, 0), (1, 25), (7, 49))
FS = 500
SEED = 42042
DRAWS = 2000
PROFILE_RECORDS = 128
CHUNK = 256
READER_THREADS = 4
PROBE_THREADS = 1
ANALYSIS_THREADS = 4
ANALYSIS_RESERVE_SECONDS = 1200.0
CEILING_SECONDS = 3600.0
FEATURE_TOLERANCE = 1e-4
REPRODUCTION_TOLERANCE = 1e-8
STREAMING_TOLERANCE = 1e-6
LOGIT_TOLERANCE = 1e-3
SHARE_TOLERANCE = 1e-8
LOCALIZATION_MARGIN = -0.10
BLAS_ARCHITECTURE = "Haswell"
LOG = logging.getLogger("experiment042")


def blas_architectures() -> list[str]:
    """
    Require the OpenBLAS kernels that reproduce Experiment 026 bit for bit (see Experiment 041).

    Returns
    -------
    list[str]
        Architecture of every loaded OpenBLAS library.

    Raises
    ------
    RuntimeError
        If any OpenBLAS library uses other kernels.
    """
    found = [str(item.get("architecture")) for item in threadpool_info()
             if item.get("internal_api") == "openblas"]
    if not found or set(found) != {BLAS_ARCHITECTURE}:
        raise RuntimeError(f"Set OPENBLAS_CORETYPE={BLAS_ARCHITECTURE}; OpenBLAS kernels are {found}")
    return found


@cache
def opened_cache() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Open the ECG-JEPA cache, checked against its receipt once per process."""
    return jepa_cache()


def cached(ecg_ids: np.ndarray) -> np.ndarray:
    """Return the cached pooled ECG-JEPA features of the given ECGs, in order."""
    features, ids, _ = opened_cache()
    position = pd.Series(np.arange(len(ids)), index=ids)
    return np.asarray(features[position.loc[ecg_ids].to_numpy()], dtype=np.float64)


def identity() -> dict[str, Any]:
    """
    Hash every input and source; the caches and prior outputs must match their receipts.

    Returns
    -------
    dict[str, Any]
        Input and source hashes keyed by repository-relative path.

    Raises
    ------
    ValueError
        If an input differs from its receipt.
    """
    _, _, cache_hashes = opened_cache()
    prior026 = json.loads((PRIOR026 / "result.json").read_text())
    prior041 = json.loads((PRIOR041 / "result.json").read_text())
    checks = {PRIOR026 / "development_scores.csv": prior026["development_scores_sha256"],
              PRIOR041 / "section_scores.npz": prior041["outputs_sha256"]["section_scores.npz"]}
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"{to_stored(path)} differs from its receipt")
    files = (*CACHE_IDS, *PTB_FILES, *checks, PRIOR026 / "result.json", PRIOR041 / "result.json")
    return {"inputs": {to_stored(path): sha256_file(path) for path in files}, "jepa_cache": cache_hashes,
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES}, "git_head": git_head(ROOT),
            "openblas_architectures": blas_architectures()}


def statement_table() -> pd.DataFrame:
    """SCP codes, abnormal diagnostic subclasses and diagnostic superclasses of every PTB-XL record."""
    meta = load_metadata()
    statements = load_statements()
    diagnostic = statements[statements["diagnostic"] == 1]
    subclass = diagnostic["diagnostic_subclass"].to_dict()
    superclass = diagnostic["diagnostic_class"].to_dict()
    codes = meta["scp_codes"].apply(set)
    return pd.DataFrame({
        "codes": codes,
        "subclasses": codes.apply(lambda found: {subclass[c] for c in found if c in subclass} - {"NORM"}),
        "superclasses": codes.apply(lambda found: {superclass[c] for c in found if c in superclass}),
    }, index=meta.index.astype(int))


def only(subclasses: set[str], name: str, other: str) -> bool:
    """Whether a record has one infarct location and neither the other nor lateral or posterior."""
    return name in subclasses and other not in subclasses and not subclasses & set(OTHER_MI)


def select_rows(smoke: bool) -> dict[str, Any]:
    """
    Select the Experiment 041 rows plus anterior-only and inferior-only infarcts.

    In smoke mode every scored set comes from training ECGs, so no development score exists before the run.

    Parameters
    ----------
    smoke : bool
        Use small training-only sets.

    Returns
    -------
    dict[str, Any]
        Frames keyed by set name, ``extract`` (fit set first, then ``scored``) and the statement table.

    Raises
    ------
    ValueError
        If a full-run count or the scored order differs from the protocol and Experiment 041.
    """
    groups = cohorts(ptb_table())
    statements = statement_table()
    common = set.intersection(*(set(np.load(path).astype(np.int64).tolist()) for path in CACHE_IDS))
    train, development = groups["train"], groups["development"]
    pool = train[train["standard"].notna() & train["ecg_id"].isin(common)]
    fit = pool[pool["standard"] == 0]
    pvc = set(statements.index[statements["codes"].map(lambda codes: "PVC" in codes)])
    benign = {i for i, codes, found in zip(statements.index, statements["codes"], statements["superclasses"],
                                           strict=True)
              if found == {"NORM"} and codes <= BENIGN_CODES and codes & {"SBRAD", "SARRH"}}
    anterior = set(statements.index[statements["subclasses"].map(lambda found: only(found, "AMI", "IMI"))])
    inferior = set(statements.index[statements["subclasses"].map(lambda found: only(found, "IMI", "AMI"))])

    def subset(frame: pd.DataFrame, ids: set[int], limit: int | None = None) -> pd.DataFrame:
        chosen = frame[frame["ecg_id"].isin(ids)]
        return chosen if limit is None else chosen.iloc[:limit]

    if smoke:
        fit = fit.iloc[:300]
        rest = pool.drop(fit.index)
        evaluation = pd.concat([rest[rest["standard"] == 0].iloc[:100],
                                rest[rest["standard"] == 1].iloc[:100]])
        others = train.drop(fit.index)
        sets = {"pvc": subset(others, pvc, 30), "benign": subset(others, benign, 20),
                "anterior": subset(others, anterior, 40), "inferior": subset(others, inferior, 40)}
        scored = pd.concat([evaluation, *sets.values()])
        scored = scored[~scored.index.duplicated()]
    else:
        original = development[development["original"]]
        evaluation = original[original["standard"].notna() & original["ecg_id"].isin(common)]
        sets = {"pvc": subset(development, pvc), "benign": subset(development, benign),
                "anterior": subset(development, anterior), "inferior": subset(development, inferior)}
        scored = development
        counts = {"fit": (len(fit), fit["patient_id"].nunique()),
                  "pool": (len(pool), int(pool["standard"].sum())),
                  "evaluation": (len(evaluation), int(evaluation["standard"].sum())),
                  "development": len(scored),
                  "pvc": (len(sets["pvc"]), sets["pvc"]["patient_id"].nunique()),
                  "benign": len(sets["benign"]),
                  "anterior": (len(sets["anterior"]), sets["anterior"]["patient_id"].nunique()),
                  "inferior": (len(sets["inferior"]), sets["inferior"]["patient_id"].nunique())}
        if counts != EXPECTED:
            raise ValueError(f"Row counts differ from the protocol: {counts}")
        with np.load(PRIOR041 / "section_scores.npz") as saved:
            if not np.array_equal(saved["ecg_ids"], scored["ecg_id"].to_numpy(np.int64)):
                raise ValueError("Scored rows differ from Experiment 041")
    if set(fit.index) & set(scored.index):
        raise ValueError("A fit ECG is also scored")
    return {"fit": fit, "pool": pool, "evaluation": evaluation, **sets, "scored": scored,
            "extract": pd.concat([fit, scored]), "statements": statements}


def reproduce026(rows: dict[str, Any]) -> tuple[dict[str, float], tuple]:
    """
    Reproduce Experiment 026's whole-ECG JEPA Mahalanobis and ``probe_all``, and check the streaming fit.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows`` (full run).

    Returns
    -------
    tuple[dict[str, float], tuple]
        Largest differences and the refitted head.

    Raises
    ------
    ValueError
        If a difference exceeds its tolerance.
    """
    evaluation = rows["evaluation"]
    saved = pd.read_csv(PRIOR026 / "development_scores.csv").set_index("ecg_id").loc[evaluation["ecg_id"]]
    x = cached(evaluation["ecg_id"].to_numpy())
    fit_features = cached(rows["fit"]["ecg_id"].to_numpy())
    with threadpool_limits(limits=PROBE_THREADS):
        whole = mahalanobis_scores(fit_mahalanobis(fit_features), x)
        head = fit_logistic(cached(rows["pool"]["ecg_id"].to_numpy()), rows["pool"]["standard"].to_numpy(int))
    streaming_reference = fit_token_reference(fit_features[:, None, :], None, clusters=False)
    streaming = token_scores(streaming_reference, x[:, None, :])[:, 0]
    differences = {
        "mahalanobis_relative": float(np.max(np.abs(whole - saved["jepa_mahalanobis"])
                                             / saved["jepa_mahalanobis"])),
        "probe_all_absolute": float(np.max(np.abs(predict(head, x) - saved["jepa_probe_all"]))),
        "streaming_relative": float(np.max(np.abs(streaming - whole) / whole)),
    }
    if (max(differences["mahalanobis_relative"], differences["probe_all_absolute"]) > REPRODUCTION_TOLERANCE
            or differences["streaming_relative"] > STREAMING_TOLERANCE):
        raise ValueError(f"Experiment 026 not reproduced: {differences}")
    return differences, head


def read_with(function: Any, stems: list[str]) -> list[Any]:
    """Apply a reader to PTB-XL records in parallel, in order."""
    with ThreadPoolExecutor(READER_THREADS) as pool:
        return list(pool.map(function, stems))


def extract(encoder: torch.nn.Module, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """
    ECG-JEPA tokens of every row, chunk by chunk, with read and model timings.

    Parameters
    ----------
    encoder : torch.nn.Module
        ECG-JEPA encoder on the GPU.
    frame : pd.DataFrame
        Rows with ``filename_hr``.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        Float32 ``[rows, 400, 768]`` tokens and the seconds spent reading and in the model.
    """
    tokens = np.empty((len(frame), 400, 768), dtype=np.float32)
    seconds = {"read": 0.0, "model": 0.0}
    stems = frame["filename_hr"].tolist()
    for start in range(0, len(stems), CHUNK):
        began = time.perf_counter()
        inputs = np.stack(read_with(ptb_jepa_input, stems[start:start + CHUNK]))
        seconds["read"] += time.perf_counter() - began
        began = time.perf_counter()
        tokens[start:start + len(inputs)] = jepa_tokens(encoder, inputs)
        seconds["model"] += time.perf_counter() - began
        LOG.info("extracted %d / %d", start + len(inputs), len(stems))
    return tokens, seconds


def check_tokens(frame: pd.DataFrame, tokens: np.ndarray) -> dict[str, float | int]:
    """
    Require the token mean to equal the cached pooled feature for every row in the cache.

    Parameters
    ----------
    frame : pd.DataFrame
        Extracted rows.
    tokens : np.ndarray
        Their tokens.

    Returns
    -------
    dict[str, float | int]
        Rows compared and the largest absolute difference.

    Raises
    ------
    ValueError
        If the difference exceeds ``FEATURE_TOLERANCE``.
    """
    _, ids, _ = opened_cache()
    present = frame["ecg_id"].isin(ids).to_numpy()
    worst = 0.0
    for start in range(0, int(present.sum()), 512):
        positions = np.flatnonzero(present)[start:start + 512]
        reference = cached(frame["ecg_id"].to_numpy()[positions])
        means = tokens[positions].mean(axis=1, dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(means - reference))))
    if worst > FEATURE_TOLERANCE:
        raise ValueError(f"Token mean differs from the cached feature by {worst}")
    return {"rows_compared": int(present.sum()), "max_abs_difference": worst}


def arm_j(rows: dict[str, Any], tokens: np.ndarray, head: tuple) -> tuple[dict[str, list[UnitMap]], dict]:
    """
    Fit one JEPA reference per lead on the fit tokens (Amendment 1) and map every scored ECG.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows``.
    tokens : np.ndarray
        Tokens of ``rows["extract"]``.
    head : tuple
        Experiment 026 ``jepa`` ``probe_all`` head.

    Returns
    -------
    tuple[dict[str, list[UnitMap]], dict]
        Unit maps per scored ECG for ``U_J``, ``U_J_kmeans`` and ``G_J``, and the dense score arrays.

    Raises
    ------
    ValueError
        If the mean contribution differs from the cached-feature logit.
    """
    fit, scored = tokens[:len(rows["fit"])], tokens[len(rows["fit"]):]
    with threadpool_limits(limits=ANALYSIS_THREADS):
        references = fit_lead_references(fit, SEED)
        dense = {"U_J": lead_token_scores(references, scored),
                 "U_J_kmeans": lead_token_scores(references, scored, kmeans=True)}
        contributions = np.concatenate([readout_contributions(head, scored[start:start + 128])
                                        for start in range(0, len(scored), 128)])
    dense["G_J"] = contributions
    position = pd.Series(np.arange(len(rows["scored"])), index=rows["scored"].index)
    evaluation = position.loc[rows["evaluation"].index].to_numpy()
    scaler, model = head
    logits = model.decision_function(scaler.transform(cached(rows["evaluation"]["ecg_id"].to_numpy())))
    difference = float(np.max(np.abs(contributions[evaluation].mean(axis=1) - logits)))
    if difference > LOGIT_TOLERANCE:
        raise ValueError(f"Mean JEPA contribution differs from the logit by {difference}")
    dense["logit_max_abs_difference"] = difference
    maps = {name: [jepa_unit_map(values) for values in dense[name]] for name in ("U_J", "U_J_kmeans", "G_J")}
    return maps, dense


def beats_of(stem: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Kept R times and wave pieces of one PTB-XL record."""
    return beat_pieces(read_ptb_float64(stem), FS)


def arm_b(rows: dict[str, Any]) -> tuple[dict[str, list[UnitMap | None]], dict[str, Any]]:
    """
    Cut beats, fit the wave references and the median-beat readout, and map every scored ECG.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows``.

    Returns
    -------
    tuple[dict[str, list[UnitMap | None]], dict[str, Any]]
        Unit maps per scored ECG (``None`` with fewer than two kept beats) for ``U_B``, ``U_B_median`` and
        ``G_B``, and counts and checks.

    Raises
    ------
    ValueError
        If the block shares do not sum to the logit.
    """
    pool, scored = rows["pool"], rows["scored"]
    everyone = pd.concat([pool, scored])
    everyone = everyone[~everyone.index.duplicated()]
    began = time.perf_counter()
    beats = dict(zip(everyone.index, read_with(beats_of, everyone["filename_hr"].tolist()), strict=True))
    read_seconds = time.perf_counter() - began
    usable = {index: len(found[0]) >= MINIMUM_BEATS for index, found in beats.items()}
    fit_index = [index for index in rows["fit"].index if usable[index]]
    with threadpool_limits(limits=ANALYSIS_THREADS):
        references = fit_wave_references({name: np.concatenate([beats[i][1][name] for i in fit_index])
                                          for name in WAVES})
        medians = {index: median_pieces(found[1]) for index, found in beats.items() if usable[index]}
        median_references = fit_wave_references({name: np.stack([medians[i][name] for i in fit_index])
                                                 for name in WAVES})
        pool_index = [index for index in pool.index if usable[index]]
        features = np.stack([median_features(medians[i]) for i in pool_index])
    with threadpool_limits(limits=PROBE_THREADS):
        head = fit_logistic(features, pool.loc[pool_index, "standard"].to_numpy(int))
    lengths = wave_lengths(FS)
    maps: dict[str, list[UnitMap | None]] = {"U_B": [], "U_B_median": [], "G_B": []}
    worst_share = 0.0
    for index in scored.index:
        if not usable[index]:
            for values in maps.values():
                values.append(None)
            continue
        r_times, pieces = beats[index]
        maps["U_B"].append(beat_unit_map(wave_scores(references, pieces), r_times))
        median = {name: values[None] for name, values in medians[index].items()}
        maps["U_B_median"].append(lead_wave_unit_map(wave_scores(median_references, median)[0]))
        vector = median_features(medians[index])[None]
        shares = block_contributions(head, vector, lengths)[0]
        logit = head[1].decision_function(head[0].transform(vector))[0]
        worst_share = max(worst_share, abs(float(shares.sum() - logit)))
        maps["G_B"].append(lead_wave_unit_map(shares))
    if worst_share > SHARE_TOLERANCE:
        raise ValueError(f"Block shares differ from the logit by {worst_share}")
    counts = {"read_seconds": read_seconds, "fit_beats": int(sum(len(beats[i][0]) for i in fit_index)),
              "fit_without_two_beats": len(rows["fit"]) - len(fit_index),
              "pool_without_two_beats": len(pool) - len(pool_index),
              "scored_without_two_beats": int(sum(not usable[i] for i in scored.index)),
              "share_max_abs_difference": worst_share}
    return maps, counts


def auroc_ap(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUROC and average precision."""
    return {"auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score))}


def prior041(rows: dict[str, Any], windows: dict[int, list[tuple[float, float]]]) -> dict[str, Any]:
    """
    Experiment 041 ``U`` per scored ECG: ECG score, any red, and hit - chance on premature beats.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows`` (full run).
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows of the included PVC rows, keyed by ECG ID.

    Returns
    -------
    dict[str, Any]
        Arrays keyed by ECG ID.
    """
    threshold = json.loads((PRIOR041 / "result.json").read_text())["thresholds"]["U"]
    with np.load(PRIOR041 / "section_scores.npz") as saved:
        ids, sections = saved["ecg_ids"], saved["U"]
    by_id = dict(zip(ids.tolist(), sections, strict=True))
    excess = {}
    for ecg_id, found in windows.items():
        hit, chance = pointing(by_id[ecg_id][None], section_overlap(found)[None])
        excess[ecg_id] = float(hit[0] - chance[0])
    return {"score": {i: float(v.max()) for i, v in by_id.items()},
            "red": {i: bool((v > threshold).any()) for i, v in by_id.items()}, "excess": excess}


def analyse(rows: dict[str, Any], maps: dict[str, list[UnitMap | None]],
            windows: dict[int, list[tuple[float, float]]], smoke: bool) -> dict[str, Any]:
    """
    Every prespecified metric, contrast and reading.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows``.
    maps : dict[str, list[UnitMap | None]]
        Unit maps in ``rows["scored"]`` order.
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows of the included PVC rows, keyed by ECG ID.
    smoke : bool
        Skip the Experiment 041 contrasts, which need development rows.

    Returns
    -------
    dict[str, Any]
        Thresholds, per-map metrics, contrasts and readings.
    """
    ids = rows["scored"]["ecg_id"].to_numpy()
    patients = dict(zip(ids, rows["scored"]["patient_id"], strict=True))
    evaluation = rows["evaluation"]
    labels = dict(zip(evaluation["ecg_id"], evaluation["standard"].astype(int), strict=True))
    normals = [i for i, y in labels.items() if y == 0]
    groups = {"normal": normals, "positive": [i for i, y in labels.items() if y == 1],
              "pvc": rows["pvc"]["ecg_id"].tolist(), "benign": rows["benign"]["ecg_id"].tolist()}
    previous = None if smoke else prior041(rows, windows)
    result: dict[str, Any] = {"maps": {}, "thresholds": {}, "pvc_included": len(windows)}
    for name, unit_maps in maps.items():
        by_id = {i: m for i, m in zip(ids, unit_maps, strict=True) if m is not None}
        threshold = float(np.quantile([ecg_score(by_id[i]) for i in normals if i in by_id], 0.95))
        result["thresholds"][name] = threshold
        scored_eval = [i for i in labels if i in by_id]
        y = np.array([labels[i] for i in scored_eval])
        score = np.array([ecg_score(by_id[i]) for i in scored_eval])
        red = {i: m.scores > threshold for i, m in by_id.items()}
        found = {
            "ecgs_with_map": len(by_id), "evaluation_ecgs": len(scored_eval), **auroc_ap(y, score),
            "any_red": {g: float(np.mean([red[i].any() for i in m if i in red])) for g, m in groups.items()},
            "mean_red_units": {g: float(np.mean([red[i].sum() for i in m if i in red]))
                               for g, m in groups.items()},
        }
        positives = [i for i in groups["positive"] if i in red]
        found["sensitivity_at_budget"] = bootstrap_mean(np.array([patients[i] for i in positives]),
                                                        np.array([float(red[i].any()) for i in positives]),
                                                        DRAWS, SEED)
        contrasts = {}
        for region, leads, first, second in (("anterior", ANTERIOR, "anterior", "inferior"),
                                             ("inferior", INFERIOR, "inferior", "anterior")):
            a = [i for i in rows[first]["ecg_id"] if i in by_id]
            b = [i for i in rows[second]["ecg_id"] if i in by_id]
            contrasts[region] = two_group_difference(
                np.array([patients[i] for i in a]), np.array([float(top_lead(by_id[i]) in leads) for i in a]),
                np.array([patients[i] for i in b]), np.array([float(top_lead(by_id[i]) in leads) for i in b]),
                DRAWS, SEED)
        found["lead_contrast"] = contrasts
        if name in TIMED:
            included = [i for i in windows if i in by_id]
            pairs = [premature_hit(by_id[i], windows[i]) for i in included]
            hit, chance = np.array(pairs).T
            pvc_patients = np.array([patients[i] for i in included])
            found.update({"pvc_ecgs": len(included), "hit_rate": float(hit.mean()),
                          "chance_rate": float(chance.mean()),
                          "hit_minus_chance": bootstrap_mean(pvc_patients, hit - chance, DRAWS, SEED)})
            if previous is not None:
                before = np.array([previous["excess"][i] for i in included])
                found["hit_minus_chance_minus_041"] = bootstrap_mean(pvc_patients, hit - chance - before,
                                                                     DRAWS, SEED)
        if previous is not None:
            found["auroc_minus_041"] = paired_auroc_difference(
                np.array([patients[i] for i in scored_eval]), y, score,
                np.array([previous["score"][i] for i in scored_eval]), DRAWS, SEED)
            benign = [i for i in groups["benign"] if i in red]
            found["benign_any_red_minus_041"] = bootstrap_mean(
                np.array([patients[i] for i in benign]),
                np.array([float(red[i].any()) - float(previous["red"][i]) for i in benign]), DRAWS, SEED)
        result["maps"][name] = found
    if previous is not None:
        result["reading"] = {}
        for arm, name in PRIMARY.items():
            found = result["maps"][name]
            keeps = found["hit_minus_chance_minus_041"]["ci_low"] > LOCALIZATION_MARGIN
            gains = {"lead": found["lead_contrast"]["anterior"]["ci_low"] > 0,
                     "detection": found["auroc_minus_041"]["ci_low"] > 0,
                     "benign": found["benign_any_red_minus_041"]["ci_high"] < 0}
            result["reading"][arm] = {"keeps_premature_localization": bool(keeps),
                                      "gains": {key: bool(value) for key, value in gains.items()},
                                      "improves_on_041": bool(keeps and any(gains.values()))}
        result["reading"]["notebook"] = any(result["reading"][arm]["improves_on_041"] for arm in PRIMARY)
    return result


def premature_targets(frame: pd.DataFrame) -> tuple[dict[int, list[tuple[float, float]]], dict[str, int]]:
    """Premature-beat windows of the PVC rows (041's rule), keyed by ECG ID, and exclusion counts."""
    windows, counts = {}, {"too_few_peaks": 0, "no_premature_beat": 0}
    for ecg_id, stem in zip(frame["ecg_id"], frame["filename_hr"], strict=True):
        peaks = r_peaks(read_ptb_float64(stem), FS)
        found = premature_windows(peaks, FS)
        if len(peaks) < 4:
            counts["too_few_peaks"] += 1
        elif not found:
            counts["no_premature_beat"] += 1
        else:
            windows[int(ecg_id)] = found
    return windows, counts


def draw_examples(rows: dict[str, Any], maps: dict[str, list[UnitMap | None]], thresholds: dict[str, float],
                  folder: Path, smoke: bool) -> dict[str, int]:
    """
    Save the 041 example figures with per-lead red marks of ``U_J``, ``U_B`` and ``G_J``.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows``.
    maps : dict[str, list[UnitMap | None]]
        Unit maps in ``rows["scored"]`` order.
    thresholds : dict[str, float]
        Red thresholds.
    folder : Path
        Destination folder.
    smoke : bool
        Use the first scored rows instead of the 041 examples.

    Returns
    -------
    dict[str, int]
        ECG ID of each example.
    """
    folder.mkdir(parents=True, exist_ok=True)
    scored = rows["scored"].reset_index()
    if smoke:
        examples = {f"smoke{k}": int(i) for k, i in enumerate(scored["ecg_id"].iloc[:3])}
    else:
        examples = json.loads((PRIOR041 / "result.json").read_text())["examples"]
    for name, ecg_id in examples.items():
        index = int(scored.index[scored["ecg_id"] == ecg_id][0])
        marks = {f"{key}": [] if maps[key][index] is None else red_marks(maps[key][index], thresholds[key])
                 for key in FIGURE_MAPS}
        figure = plot_lead_marks(read_ptb_float64(scored.at[index, "filename_hr"]), FS, marks,
                                 f"{name}: PTB-XL ECG {ecg_id}")
        figure.savefig(folder / f"{name}_{ecg_id}.png", dpi=110)
    return examples


def save_maps(path: Path, rows: dict[str, Any], dense: dict[str, np.ndarray],
              maps: dict[str, list[UnitMap | None]]) -> None:
    """Write dense arm-J scores and ragged arm-B units with per-ECG offsets."""
    arrays: dict[str, np.ndarray] = {"ecg_ids": rows["scored"]["ecg_id"].to_numpy(np.int64)}
    arrays.update({name: dense[name].astype(np.float32) for name in ("U_J", "U_J_kmeans", "G_J")})
    for name in ("U_B", "U_B_median", "G_B"):
        found = [m for m in maps[name] if m is not None]
        sizes = [0 if m is None else len(m.scores) for m in maps[name]]
        arrays[f"{name}_offsets"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        for field in ("scores", "leads", "starts", "ends"):
            arrays[f"{name}_{field}"] = np.concatenate([getattr(m, field) for m in found])
    write_npz_atomic(path, **arrays)


def run(smoke: bool, output: Path) -> None:
    """
    Run the experiment end to end and write its outputs.

    Parameters
    ----------
    smoke : bool
        Training-only smoke test.
    output : Path
        Final output folder; it must not exist.
    """
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(partial / "run.log")])
    run_identity = identity()
    rows = select_rows(smoke)
    LOG.info("rows: %s", {name: len(frame) for name, frame in rows.items()})
    if smoke:
        with threadpool_limits(limits=PROBE_THREADS):
            pool = rows["pool"]
            head = fit_logistic(cached(pool["ecg_id"].to_numpy()), pool["standard"].to_numpy(int))
        reproduction = {}
    else:
        reproduction, head = reproduce026(rows)
    LOG.info("Experiment 026 reproduction: %s", reproduction)

    frame = rows["extract"]
    with gpu_lock("cuda"):
        encoder = load_jepa()
        order_stems = rows["fit"]["filename_hr"].iloc[:ORDER_RECORDS].tolist()
        order_inputs = np.stack(read_with(ptb_jepa_input, order_stems))
        order = jepa_order_check(encoder, order_inputs, ORDER_CELLS)
        expected = {f"{lead},{patch}": [50 * lead + patch] * ORDER_RECORDS for lead, patch in ORDER_CELLS}
        if order != expected:
            raise ValueError(f"Tokens are not in lead and patch order: {order}")
        _, profile = extract(encoder, frame.iloc[:PROFILE_RECORDS])
        per_record = sum(profile.values()) / PROFILE_RECORDS
        projected = time.perf_counter() - started + per_record * len(frame) + ANALYSIS_RESERVE_SECONDS
        LOG.info("profile %.4f s per record, projected total %.1f s", per_record, projected)
        if projected > CEILING_SECONDS:
            raise RuntimeError(f"Projected {projected:.0f} s exceeds the {CEILING_SECONDS:.0f} s ceiling")
        tokens, extraction = extract(encoder, frame)
        del encoder
        torch.cuda.empty_cache()
    token_check = check_tokens(frame, tokens)
    LOG.info("token check: %s", token_check)
    maps, dense = arm_j(rows, tokens, head)
    del tokens
    LOG.info("arm J done at %.1f s", time.perf_counter() - started)
    b_maps, b_counts = arm_b(rows)
    maps.update(b_maps)
    LOG.info("arm B done at %.1f s: %s", time.perf_counter() - started, b_counts)

    windows, exclusions = premature_targets(rows["pvc"])
    result = analyse(rows, maps, windows, smoke)
    examples = draw_examples(rows, maps, result["thresholds"], partial / "figures", smoke)
    save_maps(partial / "unit_scores.npz", rows, dense, maps)
    write_json_atomic(partial / "result.json", {
        "status": "smoke" if smoke else "completed", "identity": run_identity,
        "counts": {name: len(rows[name]) for name in
                   ("fit", "pool", "evaluation", "pvc", "benign", "anterior", "inferior", "scored")},
        "reproduction026": reproduction, "order_check": order, "token_check": token_check,
        "jepa_logit_max_abs_difference": dense["logit_max_abs_difference"], "arm_b": b_counts,
        "pvc_exclusions": exclusions, "profile_seconds": profile, "extraction_seconds": extraction,
        "examples": examples, **result,
        "outputs_sha256": {"unit_scores.npz": sha256_file(partial / "unit_scores.npz")},
        "seed": SEED, "draws": DRAWS, "total_seconds": time.perf_counter() - started,
        "calibration_test_evaluated": False,
    })
    partial.rename(output)
    LOG.info("done in %.1f s: %s", time.perf_counter() - started, result.get("reading"))


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="training-only smoke test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    run(arguments.smoke, arguments.output)


if __name__ == "__main__":
    main()
