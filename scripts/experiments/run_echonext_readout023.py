"""Experiment 023 v3: frozen CPC, ECG-JEPA and xECG readout of echo-confirmed structural heart disease."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

from ecg_experiment.echonext import to_cpc_scale
from ecg_experiment.echonext_readout import (
    age_sex_inputs,
    cosine_summary,
    downsample_250,
    jepa_input_250,
    measured,
    pooled_lead_statistics,
    tabular_inputs,
    train_medians,
    xecg_input_250,
)
from ecg_experiment.eda.echonext import COMPONENTS, COMPOSITE
from ecg_experiment.external_encoders import (
    jepa_cache,
    jepa_features,
    load_jepa,
    load_xecg_backbone,
    ptb_jepa_input,
    ptb_xecg_input,
    read_ptb_float64,
    source_files,
    xecg_cache,
    xecg_features,
)
from ecg_experiment.external_readout import (
    ENCODER,
    ENCODER_HEADS,
    HEAD_TOLERANCE,
    Head,
    cache_positions,
    combine_features,
    evenly_spaced,
    fit_logistic_c,
    load_encoder,
    prior_cpc_features,
    prior_identity,
    ptb_heads,
    score,
)
from ecg_experiment.files import read_json, sha256_file, sha256_json, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, patient_bootstrap, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.ptb_cpc_features import NORMALIZATION, pooled_features

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/echonext_250hz_v1"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
OUTPUT = ROOT / "outputs/experiment023_echonext_v3"
SPLITS = ("train", "val")
CHUNK = 512
PROFILE_RECORDS = 512
CEILING_SECONDS = 7_200
STATISTICS_RECORDS = 1_000
DIAGNOSTIC_RECORDS = 32
INTEGRITY_TOLERANCE = 1e-4
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 23023
MIN_COMPONENT_POSITIVES = 50
ENCODER_NAMES = ("cpc", "jepa", "xecg")
FITTED_HEADS = {
    "age_sex": ("age_sex",), "tabular": ("tabular",), "cpc": ("cpc",), "cpc_tabular": ("cpc", "tabular"),
    "jepa": ("jepa",), "xecg": ("xecg",), "jepa_tabular": ("jepa", "tabular"),
    "xecg_tabular": ("xecg", "tabular"),
}
PTB_APPLIED = {"ptb_cpc_standard": ("cpc", "cpc_standard"), "ptb_jepa_standard": ("jepa", "jepa_standard"),
               "ptb_xecg_standard": ("xecg", "xecg_standard")}
COMPONENT_HEADS = ("cpc", "tabular", "jepa", "xecg")
CONTRASTS = {
    "cpc_minus_tabular": ("cpc", "tabular"),
    "cpc_tabular_minus_tabular": ("cpc_tabular", "tabular"),
    "cpc_minus_age_sex": ("cpc", "age_sex"),
    "jepa_minus_cpc": ("jepa", "cpc"),
    "xecg_minus_cpc": ("xecg", "cpc"),
    "jepa_tabular_minus_tabular": ("jepa_tabular", "tabular"),
    "xecg_tabular_minus_tabular": ("xecg_tabular", "tabular"),
}
ECHO_INPUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {"jepa": jepa_input_250, "xecg": xecg_input_250}
PTB_INPUTS: dict[str, Callable[[str], np.ndarray]] = {"jepa": ptb_jepa_input, "xecg": ptb_xecg_input}
EXTRACTORS: dict[str, Callable[[nn.Module, np.ndarray], np.ndarray]] = {
    "cpc": pooled_features, "jepa": jepa_features, "xecg": xecg_features}
SOURCES = (
    "ecg_experiment/echonext.py", "ecg_experiment/eda/echonext.py", "ecg_experiment/echonext_readout.py",
    "ecg_experiment/external_readout.py", "ecg_experiment/external_encoders.py",
    "ecg_experiment/full_development.py", "ecg_experiment/ptb_cpc_features.py", "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_input_audit.py", "ecg_experiment/waveforms.py", "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/xecg.py", "scripts/data/build_echonext_cache.py",
    "scripts/experiments/run_echonext_readout023.py", "pyproject.toml", "uv.lock",
    "docs/experiment-023-echonext-readout.md",
)

Caches = dict[str, tuple[np.ndarray, np.ndarray]]
Scales = dict[str, tuple[np.ndarray, np.ndarray]]


def echonext_rows() -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Read the EchoNext cache rows after checking the cache files and build sources against its receipt.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        Train and val rows indexed by ``ecg_key``, and the verified hashes.

    Raises
    ------
    ValueError
        If the receipt is incomplete, a file or build source changed, or the rows do not match the arrays.
    """
    metadata = read_json(CACHE / "metadata.json")
    hashes = {"metadata": sha256_file(CACHE / "metadata.json"), "rows": sha256_file(CACHE / "rows.csv"),
              "arrays": {f"{split}.npy": sha256_file(CACHE / f"{split}.npy") for split in SPLITS}}
    sources = {name: sha256_file(ROOT / name) for name in metadata["source_sha256"]}
    if not metadata["complete"] or hashes["rows"] != metadata["rows_sha256"] or \
            hashes["arrays"] != metadata["arrays_sha256"] or sources != metadata["source_sha256"]:
        raise ValueError("EchoNext cache differs from its receipt")
    rows = pd.read_csv(CACHE / "rows.csv", index_col="ecg_key", keep_default_na=False, na_values=[""])
    for split in SPLITS:
        part = rows[rows["split"] == split]
        if len(part) != metadata["records"][split] or int(part["use"].sum()) != metadata["usable"][split] or \
                not np.array_equal(part["row"].to_numpy(), np.arange(len(part))):
            raise ValueError(f"EchoNext {split} rows differ from the receipt")
    if rows[COMPOSITE].isna().any() or not rows[COMPOSITE].isin((0, 1)).all():
        raise ValueError("EchoNext composite label malformed")
    return rows, hashes


def echonext_scale() -> tuple[np.ndarray, np.ndarray]:
    """
    EchoNext usable-train per-lead mean and standard deviation from the cache receipt.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 mean and standard deviation, shape ``(12,)`` each.

    Raises
    ------
    ValueError
        If the statistics are not those of the usable training rows.
    """
    metadata = read_json(CACHE / "metadata.json")
    if metadata["lead_statistics_rows"] != "train rows passing the quality rules":
        raise ValueError("EchoNext lead statistics are not from the usable training rows")
    statistics = metadata["train_lead_statistics"]
    return np.asarray(statistics["mean"], dtype=np.float64), np.asarray(statistics["std"], dtype=np.float64)


def ptb_250hz_statistics(train: pd.DataFrame) -> dict[str, object]:
    """
    Per-lead mV statistics of 1,000 evenly spaced Experiment 020 training ECGs converted to 250 Hz.

    Parameters
    ----------
    train : pd.DataFrame
        ``full_development.cohorts`` training rows.

    Returns
    -------
    dict[str, object]
        Record count, hash of their ECG IDs, and the per-lead ``mean`` and ``std``.
    """
    sample = evenly_spaced(train, STATISTICS_RECORDS)
    signals = (downsample_250(read_ptb_float64(stem)) for stem in sample["filename_hr"])
    mean, std = pooled_lead_statistics(signals)
    return {"records": len(sample), "ecg_ids_sha256": sha256_json(sample["ecg_id"].astype(int).tolist()),
            "mean": mean.tolist(), "std": std.tolist()}


def encoder_scales(ptb_statistics: dict[str, object]) -> Scales:
    """
    Target per-lead mean and standard deviation of each encoder's input.

    Parameters
    ----------
    ptb_statistics : dict[str, object]
        Output of ``ptb_250hz_statistics``.

    Returns
    -------
    Scales
        Historical CPC normalization for ``cpc`` and the 250 Hz PTB-XL statistics for ``jepa`` and ``xecg``.
    """
    cpc = read_json(NORMALIZATION)
    ptb = (np.asarray(ptb_statistics["mean"]), np.asarray(ptb_statistics["std"]))
    return {"cpc": (np.asarray(cpc["mean"], dtype=np.float64), np.asarray(cpc["std"], dtype=np.float64)),
            "jepa": ptb, "xecg": ptb}


def open_caches() -> tuple[Caches, dict[str, dict[str, str]]]:
    """
    Open the ECG-JEPA and xECG PTB-XL feature caches.

    Returns
    -------
    tuple[Caches, dict[str, dict[str, str]]]
        Features and integer ECG IDs per encoder, and their verified hashes.
    """
    jepa, jepa_ids, jepa_hashes = jepa_cache()
    xecg, xecg_ids, xecg_hashes = xecg_cache()
    return {"jepa": (jepa, jepa_ids), "xecg": (xecg, xecg_ids)}, {"jepa": jepa_hashes, "xecg": xecg_hashes}


def prior022_identity() -> dict[str, str]:
    """
    Hash the Experiment 022 v3 artifacts the PTB-XL encoder heads are refitted from.

    Returns
    -------
    dict[str, str]
        SHA-256 of the saved PTB-XL features, development predictions, profile and result.

    Raises
    ------
    ValueError
        If the saved files differ from the hashes in the 022 receipts.
    """
    profile = read_json(PRIOR022 / "profile.json")
    result = read_json(PRIOR022 / "result.json")
    hashes = {name: sha256_file(PRIOR022 / name)
              for name in ("ptb_features.npz", "development_predictions.npz", "profile.json", "result.json")}
    if hashes["ptb_features.npz"] != profile["ptb_features_sha256"] or \
            hashes["development_predictions.npz"] != result["outputs_sha256"]["development_predictions.npz"]:
        raise ValueError("Experiment 022 artifacts differ from their receipts")
    return hashes


def ptb_encoder_heads(groups: dict[str, pd.DataFrame], caches: Caches) -> tuple[dict[str, Head],
                                                                                   dict[str, float]]:
    """
    Refit the Experiment 022 v3 ECG-JEPA and xECG PTB-XL heads and require its development probabilities.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.

    Returns
    -------
    tuple[dict[str, Head], dict[str, float]]
        Heads keyed by their 022 name (``jepa_standard``, ``xecg_standard``) and each head's largest
        absolute difference from Experiment 022.

    Raises
    ------
    ValueError
        If the development rows differ or a head differs from Experiment 022 by more than the tolerance.
    """
    train, development = groups["train"], groups["development"]
    ids = pd.concat([train, development])["ecg_id"].to_numpy(dtype=np.int64)
    standard = train["standard"].notna().to_numpy()
    y = train.loc[standard, "standard"].to_numpy(dtype=np.int64)
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    heads, differences = {}, {}
    with np.load(PRIOR022 / "development_predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], development.index.to_numpy(dtype=str)):
            raise ValueError("Development rows differ from Experiment 022")
        for name in ("jepa_standard", "xecg_standard"):
            encoder, c = ENCODER_HEADS[name]
            cache, cache_ids = caches[encoder]
            features = combine_features(ids, cache, cache_ids, extracted[f"{encoder}_ids"],
                                        extracted[encoder])
            heads[name] = fit_logistic_c(features[:len(train)][standard], y, c)
            differences[name] = float(np.abs(predict(heads[name], features[len(train):]) - saved[name]).max())
            if differences[name] > HEAD_TOLERANCE:
                raise ValueError(f"Refitted {name} differs from Experiment 022 by {differences[name]}")
    return heads, differences


def identity(rows: pd.DataFrame, echo_hashes: dict[str, object], cache_hashes: dict[str, dict[str, str]],
             ptb_statistics: dict[str, object]) -> dict[str, object]:
    """
    Hash every input of the readout.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``echonext_rows``.
    echo_hashes : dict[str, object]
        Verified EchoNext cache hashes.
    cache_hashes : dict[str, dict[str, str]]
        Verified PTB-XL feature cache hashes.
    ptb_statistics : dict[str, object]
        Output of ``ptb_250hz_statistics``.

    Returns
    -------
    dict[str, object]
        The run identity a profile and a run must share.
    """
    usable = rows[rows["use"]]
    counts = {split: {"records": int((rows["split"] == split).sum()),
                      "usable": int((usable["split"] == split).sum()),
                      "usable_positives": int(usable.loc[usable["split"] == split, COMPOSITE].sum()),
                      "usable_patients": int(usable.loc[usable["split"] == split, "patient_key"].nunique())}
              for split in SPLITS}
    encoder_sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files()}
    return {
        "echonext": echo_hashes, "counts": counts, "cpc_encoder": sha256_file(ENCODER),
        "cpc_normalization": sha256_file(NORMALIZATION), "feature_caches": cache_hashes,
        "experiment020": prior_identity(), "experiment022": prior022_identity(),
        "ptb_250hz_statistics": ptb_statistics,
        "sources": {name: sha256_file(ROOT / name) for name in SOURCES}, "encoder_sources": encoder_sources,
    }


def load_models() -> dict[str, nn.Module]:
    """
    Load the three frozen encoders on the GPU.

    Returns
    -------
    dict[str, nn.Module]
        Encoders keyed by ``cpc``, ``jepa`` and ``xecg``.
    """
    return {"cpc": load_encoder(), "jepa": load_jepa(), "xecg": load_xecg_backbone()}


def cached_sample(train: pd.DataFrame, cache: np.ndarray, cache_ids: np.ndarray) -> tuple[pd.DataFrame,
                                                                                         np.ndarray]:
    """
    Evenly spaced training rows present in a feature cache, with their cached features.

    Parameters
    ----------
    train : pd.DataFrame
        Experiment 020 training rows.
    cache : np.ndarray
        Cached features.
    cache_ids : np.ndarray
        Integer ECG ID of every cache row.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        The sampled rows and their float64 cached features.
    """
    positions = cache_positions(cache_ids, train["ecg_id"].to_numpy())
    sample = evenly_spaced(train[positions >= 0], DIAGNOSTIC_RECORDS)
    cached = np.asarray(cache[cache_positions(cache_ids, sample["ecg_id"].to_numpy())], dtype=np.float64)
    return sample, cached


def integrity(models: dict[str, nn.Module], train: pd.DataFrame, caches: Caches) -> dict[str, object]:
    """
    Recompute evenly spaced cached training features through the 500 Hz path and require the caches.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    train : pd.DataFrame
        Experiment 020 training rows.
    caches : Caches
        Output of ``open_caches``.

    Returns
    -------
    dict[str, object]
        Checked ECG IDs and the largest absolute difference per encoder.

    Raises
    ------
    ValueError
        If any recomputed feature differs from its cached value by more than ``INTEGRITY_TOLERANCE``.
    """
    result = {}
    for name, (cache, cache_ids) in caches.items():
        sample, cached = cached_sample(train, cache, cache_ids)
        inputs = np.stack([PTB_INPUTS[name](stem) for stem in sample["filename_hr"]])
        difference = float(np.abs(EXTRACTORS[name](models[name], inputs) - cached).max())
        result[name] = {"ecg_ids": sample["ecg_id"].astype(int).tolist(), "max_abs_difference": difference}
        if difference > INTEGRITY_TOLERANCE:
            raise ValueError(f"Recomputed {name} features differ from the cache by {difference}")
    return result


def diagnostic(models: dict[str, nn.Module], train: pd.DataFrame, caches: Caches) -> dict[str, object]:
    """
    Cosine similarity of 250 Hz-path features to the cached 500 Hz-path features (reported, not a gate).

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    train : pd.DataFrame
        Experiment 020 training rows.
    caches : Caches
        Output of ``open_caches``.

    Returns
    -------
    dict[str, object]
        Checked ECG IDs and the mean and minimum cosine per encoder.
    """
    result = {}
    for name, (cache, cache_ids) in caches.items():
        sample, cached = cached_sample(train, cache, cache_ids)
        signals = [downsample_250(read_ptb_float64(stem)) for stem in sample["filename_hr"]]
        features = EXTRACTORS[name](models[name], np.stack([ECHO_INPUTS[name](signal) for signal in signals]))
        result[name] = {"ecg_ids": sample["ecg_id"].astype(int).tolist(), **cosine_summary(features, cached)}
    return result


def encoder_inputs(name: str, signals: np.ndarray, echo: tuple[np.ndarray, np.ndarray],
                   scales: Scales) -> np.ndarray:
    """
    Map EchoNext waveforms onto one encoder's per-lead input scale and apply its 250 Hz preprocessing.

    Parameters
    ----------
    name : str
        ``cpc``, ``jepa`` or ``xecg``.
    signals : np.ndarray
        Float32 EchoNext waveforms ``(records, 12, 2500)``.
    echo : tuple[np.ndarray, np.ndarray]
        Output of ``echonext_scale``.
    scales : Scales
        Output of ``encoder_scales``.

    Returns
    -------
    np.ndarray
        Encoder inputs, one per record.
    """
    mapped = to_cpc_scale(signals, *echo, *scales[name])
    if name == "cpc":
        return mapped
    return np.stack([ECHO_INPUTS[name](signal) for signal in mapped])


def extract_echonext(models: dict[str, nn.Module], waveforms: np.ndarray, positions: np.ndarray,
                     echo: tuple[np.ndarray, np.ndarray], scales: Scales) -> tuple[dict[str, np.ndarray],
                                                                                   dict[str, float]]:
    """
    Read EchoNext waveforms chunkwise and extract every encoder's features.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    waveforms : np.ndarray
        Memory-mapped ``(N, 12, 2500)`` split array.
    positions : np.ndarray
        Rows of ``waveforms`` to featurize, in output order.
    echo : tuple[np.ndarray, np.ndarray]
        Output of ``echonext_scale``.
    scales : Scales
        Output of ``encoder_scales``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]
        Features per encoder, and seconds spent reading and per encoder (input conversion and model).
    """
    chunks = {name: [] for name in ENCODER_NAMES}
    seconds = dict.fromkeys(("read", *ENCODER_NAMES), 0.0)
    for start in range(0, len(positions), CHUNK):
        began = time.monotonic()
        signals = np.asarray(waveforms[positions[start:start + CHUNK]], dtype=np.float32)
        seconds["read"] += time.monotonic() - began
        for name in ENCODER_NAMES:
            began = time.monotonic()
            chunks[name].append(EXTRACTORS[name](models[name], encoder_inputs(name, signals, echo, scales)))
            torch.cuda.synchronize()
            seconds[name] += time.monotonic() - began
    return {name: np.concatenate(values) for name, values in chunks.items()}, seconds


def split_arrays(split: str) -> np.ndarray:
    """
    Memory-map one EchoNext split array.

    Parameters
    ----------
    split : str
        ``train`` or ``val``.

    Returns
    -------
    np.ndarray
        Read-only ``(N, 12, 2500)`` float32 array.
    """
    return np.load(CACHE / f"{split}.npy", mmap_mode="r")


def profile(rows: pd.DataFrame, groups: dict[str, pd.DataFrame], caches: Caches, scales: Scales,
            run_identity: dict[str, object], started: float) -> None:
    """
    Check feature integrity, run the 250 Hz diagnostic, time EchoNext extraction and gate the run.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``echonext_rows``.
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.
    scales : Scales
        Output of ``encoder_scales``.
    run_identity : dict[str, object]
        Output of ``identity`` with the checks added by ``main``.
    started : float
        Monotonic start time of the stage.
    """
    train = rows[(rows["split"] == "train") & rows["use"]]
    records = len(train) + int((rows["split"] == "val").sum())
    with gpu_lock("cuda", blocking=False):
        models = load_models()
        checks = integrity(models, groups["train"], caches)
        cosines = diagnostic(models, groups["train"], caches)
        began = time.monotonic()
        _, encoder_seconds = extract_echonext(models, split_arrays("train"),
                                              train["row"].to_numpy()[:PROFILE_RECORDS], echonext_scale(),
                                              scales)
        profile_seconds = time.monotonic() - began
    preflight_seconds = time.monotonic() - started - profile_seconds
    projected = 2 * preflight_seconds + 1.5 * math.ceil(records / PROFILE_RECORDS) * profile_seconds + 900
    receipt = {
        "identity": run_identity, "integrity": checks, "diagnostic_250hz_path": cosines,
        "profile_records": PROFILE_RECORDS, "profile_seconds": profile_seconds,
        "profile_seconds_by_part": encoder_seconds, "preflight_seconds": preflight_seconds,
        "run_records": records, "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "gate_passed": receipt["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def input_blocks(frame: pd.DataFrame, medians: dict[str, float],
                 features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Every input block of the heads for one set of ECGs.

    Parameters
    ----------
    frame : pd.DataFrame
        EchoNext rows.
    medians : dict[str, float]
        Output of ``train_medians``.
    features : dict[str, np.ndarray]
        Features per encoder in the row order of ``frame``.

    Returns
    -------
    dict[str, np.ndarray]
        ``age_sex``, ``tabular`` and the encoder features.
    """
    return {"age_sex": age_sex_inputs(frame), "tabular": tabular_inputs(frame, medians), **features}


def head_matrix(blocks: dict[str, np.ndarray], head: str) -> np.ndarray:
    """
    Input matrix of one fitted head.

    Parameters
    ----------
    blocks : dict[str, np.ndarray]
        Output of ``input_blocks``.
    head : str
        Key of ``FITTED_HEADS``.

    Returns
    -------
    np.ndarray
        Float64 inputs.
    """
    return np.hstack([np.asarray(blocks[part], dtype=np.float64) for part in FITTED_HEADS[head]])


def fit_heads(train: pd.DataFrame, blocks: dict[str, np.ndarray]) -> dict[str, Head]:
    """
    Fit every EchoNext head on the usable training rows and the composite label.

    Parameters
    ----------
    train : pd.DataFrame
        Usable training rows.
    blocks : dict[str, np.ndarray]
        Training ``input_blocks``.

    Returns
    -------
    dict[str, Head]
        Heads keyed as in ``FITTED_HEADS``.
    """
    y = train[COMPOSITE].to_numpy(dtype=np.int64)
    return {head: fit_logistic(head_matrix(blocks, head), y) for head in FITTED_HEADS}


def component_entry(flag: str, train: pd.DataFrame, val: pd.DataFrame, train_blocks: dict[str, np.ndarray],
                    val_blocks: dict[str, np.ndarray]) -> dict[str, object]:
    """
    One component label on the rows with a measured echo value.

    Parameters
    ----------
    flag : str
        Component flag, a key of ``COMPONENTS``.
    train, val : pd.DataFrame
        Usable training and validation rows.
    train_blocks, val_blocks : dict[str, np.ndarray]
        Their ``input_blocks``.

    Returns
    -------
    dict[str, object]
        Counts, eligibility and, when eligible, the scores of ``COMPONENT_HEADS``.
    """
    train_mask, val_mask = measured(train, flag), measured(val, flag)
    y_train = train.loc[train_mask, flag].to_numpy(dtype=np.int64)
    y_val = val.loc[val_mask, flag].to_numpy(dtype=np.int64)
    entry = {"train_records": len(y_train), "train_positives": int(y_train.sum()),
             "val_records": len(y_val), "val_positives": int(y_val.sum()),
             "eligible": int(y_val.sum()) >= MIN_COMPONENT_POSITIVES, "scores": {}}
    if not entry["eligible"]:
        return entry
    for head in COMPONENT_HEADS:
        fitted = fit_logistic(head_matrix(train_blocks, head)[train_mask], y_train)
        entry["scores"][head] = score(y_val, predict(fitted, head_matrix(val_blocks, head)[val_mask]))
    return entry


def contrasts(val: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Prespecified paired patient bootstrap contrasts on the composite label.

    Parameters
    ----------
    val : pd.DataFrame
        Usable validation rows.
    scores : dict[str, np.ndarray]
        Probabilities per head.

    Returns
    -------
    dict[str, object]
        Bootstrap result per contrast.
    """
    y = val[COMPOSITE].to_numpy(dtype=np.int64)
    patients = val["patient_key"].to_numpy()
    return {name: patient_bootstrap(patients, y, scores[first], scores[second], draws=BOOTSTRAP_DRAWS,
                                    seed=BOOTSTRAP_SEED)
            for name, (first, second) in CONTRASTS.items()}


def save_outputs(train: pd.DataFrame, val_all: pd.DataFrame, val: pd.DataFrame,
                 features: dict[str, dict[str, np.ndarray]], scores: dict[str, np.ndarray],
                 cpc_all: np.ndarray) -> dict[str, str]:
    """
    Write the local EchoNext features and predictions.

    Parameters
    ----------
    train, val_all, val : pd.DataFrame
        Usable training rows, all validation rows and usable validation rows.
    features : dict[str, dict[str, np.ndarray]]
        Features keyed by split (``train``, ``val_all``), then encoder.
    scores : dict[str, np.ndarray]
        Probabilities per head on the usable validation rows.
    cpc_all : np.ndarray
        ``cpc`` probabilities on all validation rows.

    Returns
    -------
    dict[str, str]
        SHA-256 of each written file.
    """
    arrays = {f"{split}_{name}": values for split, part in features.items() for name, values in part.items()}
    write_npz_atomic(OUTPUT / "features.npz", train_ecg_keys=train.index.to_numpy(),
                     val_all_ecg_keys=val_all.index.to_numpy(), **arrays)
    write_npz_atomic(OUTPUT / "predictions.npz", ecg_keys=val.index.to_numpy(),
                     patient_keys=val["patient_key"].to_numpy(), val_all_ecg_keys=val_all.index.to_numpy(),
                     cpc_all_val=cpc_all, **scores)
    return {name: sha256_file(OUTPUT / name) for name in ("features.npz", "predictions.npz")}


def require_profile(run_identity: dict[str, object]) -> str:
    """
    Require a passed profile with the same identity.

    Parameters
    ----------
    run_identity : dict[str, object]
        Output of ``identity`` with the checks added by ``main``.

    Returns
    -------
    str
        SHA-256 of the profile receipt.

    Raises
    ------
    ValueError
        If no matching passed profile exists.
    """
    receipt = read_json(OUTPUT / "profile.json")
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("A matching passed profile is required")
    return sha256_file(OUTPUT / "profile.json")


def run(rows: pd.DataFrame, groups: dict[str, pd.DataFrame], caches: Caches, scales: Scales,
        run_identity: dict[str, object], ptb: dict[str, Head]) -> None:
    """
    Extract EchoNext features, fit and score every head and write the aggregate result.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``echonext_rows``.
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.
    scales : Scales
        Output of ``encoder_scales``.
    run_identity : dict[str, object]
        Output of ``identity`` with the checks added by ``main``.
    ptb : dict[str, Head]
        Reproduced 020 ``cpc_standard`` and 022 ``jepa_standard`` and ``xecg_standard`` heads.
    """
    profile_sha256 = require_profile(run_identity)
    train = rows[(rows["split"] == "train") & rows["use"]]
    val_all = rows[rows["split"] == "val"]
    use = val_all["use"].to_numpy()
    val = val_all[use]
    echo = echonext_scale()
    with gpu_lock("cuda", blocking=False):
        models = load_models()
        checks = integrity(models, groups["train"], caches)
        began = time.monotonic()
        train_features, train_seconds = extract_echonext(models, split_arrays("train"),
                                                         train["row"].to_numpy(), echo, scales)
        val_features, val_seconds = extract_echonext(models, split_arrays("val"),
                                                     val_all["row"].to_numpy(), echo, scales)
        seconds = time.monotonic() - began
    del models
    medians = train_medians(train)
    train_blocks = input_blocks(train, medians, train_features)
    val_blocks = input_blocks(val, medians, {name: values[use] for name, values in val_features.items()})
    with threadpool_limits(limits=1):
        heads = fit_heads(train, train_blocks)
        scores = {head: predict(heads[head], head_matrix(val_blocks, head)) for head in FITTED_HEADS}
        scores.update({head: predict(ptb[source], val_blocks[encoder])
                       for head, (encoder, source) in PTB_APPLIED.items()})
        cpc_all = predict(heads["cpc"], val_features["cpc"])
        y = val[COMPOSITE].to_numpy(dtype=np.int64)
        results = {
            "by_head": {head: score(y, values) for head, values in scores.items()},
            "contrasts": contrasts(val, scores),
            "components": {flag: component_entry(flag, train, val, train_blocks, val_blocks)
                           for flag in COMPONENTS},
            "cpc_all_val_rows": score(val_all[COMPOSITE].to_numpy(dtype=np.int64), cpc_all),
        }
    outputs = save_outputs(train, val_all, val, {"train": train_features, "val_all": val_features}, scores,
                           cpc_all)
    result = {
        "status": "complete_development", "identity": run_identity, "profile_sha256": profile_sha256,
        "integrity": checks, "extraction_seconds": seconds,
        "extraction_seconds_by_part": {"train": train_seconds, "val": val_seconds},
        "tabular_train_medians": medians,
        "head_iterations": {head: int(model.n_iter_[0]) for head, (_, model) in heads.items()},
        "primary": results["by_head"]["cpc"], **results, "outputs_sha256": outputs, "test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete"}), flush=True)


def main() -> None:
    """Profile or run the frozen EchoNext readout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    rows, echo_hashes = echonext_rows()
    caches, cache_hashes = open_caches()
    groups = cohorts(ptb_table())
    ptb_statistics = ptb_250hz_statistics(groups["train"])
    run_identity = identity(rows, echo_hashes, cache_hashes, ptb_statistics)
    with threadpool_limits(limits=1):
        cpc_heads, _, cpc_differences = ptb_heads(groups["train"], groups["development"],
                                                  prior_cpc_features())
        encoder_heads, encoder_differences = ptb_encoder_heads(groups, caches)
    run_identity["experiment020_head_max_abs_difference"] = cpc_differences
    run_identity["experiment022_head_max_abs_difference"] = encoder_differences
    scales = encoder_scales(ptb_statistics)
    if args.stage == "profile":
        profile(rows, groups, caches, scales, run_identity, started)
    else:
        run(rows, groups, caches, scales, run_identity, {"cpc_standard": cpc_heads["cpc_standard"],
                                                          **encoder_heads})


if __name__ == "__main__":
    main()
