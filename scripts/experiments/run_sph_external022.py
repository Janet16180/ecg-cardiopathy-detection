"""Experiment 022 v3: external readout of the Experiment 020, ECG-JEPA and xECG heads on SPH."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

from ecg_experiment import sph
from ecg_experiment.cpc_input_audit import historical_resample
from ecg_experiment.external_encoders import (
    jepa_cache,
    jepa_features,
    jepa_input,
    load_jepa,
    load_xecg_backbone,
    ptb_jepa_input,
    ptb_xecg_input,
    source_files,
    xecg_cache,
    xecg_features,
    xecg_input,
)
from ecg_experiment.external_readout import (
    ALL_HEADS,
    ENCODER,
    Head,
    cache_positions,
    combine_features,
    encoder_heads,
    evenly_spaced,
    head_inputs,
    load_encoder,
    prior_cpc_features,
    prior_identity,
    ptb_heads,
    score,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, patient_bootstrap, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.ptb_cpc_features import NORMALIZATION, pooled_features

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/sph_clean_v1"
OUTPUT = ROOT / "outputs/experiment022_sph_external_v3"
CHUNK = 512
PROFILE_RECORDS = 512
CEILING_SECONDS = 3_600
LEAD_CHECK_RECORDS = 64
EINTHOVEN_TOLERANCE_MV = 0.02
LEAD_STD_RANGE_MV = (0.02, 2.0)
INTEGRITY_RECORDS = 32
INTEGRITY_TOLERANCE = 1e-4
BOOTSTRAP_SEED = 22022
EXPECTED_EVALUATION = {"records": 25577, "primary": {"0.0": 13818, "1.0": 7190, "nan": 4569}}
ENCODER_NAMES = ("cpc", "jepa", "xecg")
READOUT_HEADS = ("cpc_standard", "jepa_standard", "xecg_standard")
CONTRASTS = {
    "cpc_standard_minus_age_sex_standard": ("cpc_standard", "age_sex_standard"),
    "cpc_age_sex_minus_cpc_standard": ("cpc_age_sex_standard", "cpc_standard"),
    "cpc_standard_minus_cpc_project_full": ("cpc_standard", "cpc_project_full"),
    "jepa_standard_minus_cpc_standard": ("jepa_standard", "cpc_standard"),
    "xecg_standard_minus_cpc_standard": ("xecg_standard", "cpc_standard"),
}
SPH_INPUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "cpc": historical_resample, "jepa": jepa_input, "xecg": xecg_input}
PTB_INPUTS: dict[str, Callable[[str], np.ndarray]] = {"jepa": ptb_jepa_input, "xecg": ptb_xecg_input}
EXTRACTORS: dict[str, Callable[[nn.Module, np.ndarray], np.ndarray]] = {
    "cpc": pooled_features, "jepa": jepa_features, "xecg": xecg_features}
SOURCES = (
    "ecg_experiment/sph.py", "ecg_experiment/external_readout.py", "ecg_experiment/external_encoders.py",
    "ecg_experiment/full_development.py", "ecg_experiment/ptb_cpc_features.py", "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_input_audit.py", "ecg_experiment/waveforms.py", "ecg_experiment/eda/ptbxl.py",
    "scripts/data/build_sph_clean.py", "scripts/experiments/run_sph_external022.py", "pyproject.toml",
    "uv.lock", "docs/experiment-022-sph-external-readout.md",
)

Caches = dict[str, tuple[np.ndarray, np.ndarray]]


def manifest_rows() -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Read the clean SPH manifest after checking it and every record file against its receipt.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        All 25,770 manifest rows indexed by ``ecg_id``, and the verified hashes.

    Raises
    ------
    ValueError
        If the manifest, its label source, the SPH small files or any record file changed.
    """
    metadata = json.loads((MANIFEST / "metadata.json").read_text())
    hashes = {"rows": sha256_file(MANIFEST / "rows.csv"), "metadata": sha256_file(MANIFEST / "metadata.json"),
              "small_files_md5": sph.verify_small_files()}
    if hashes["rows"] != metadata["rows_sha256"] or hashes["small_files_md5"] != metadata["small_files_md5"]:
        raise ValueError("SPH manifest or small files differ from the manifest receipt")
    if sha256_file(ROOT / "ecg_experiment/sph.py") != metadata["source_sha256"]["ecg_experiment/sph.py"]:
        raise ValueError("SPH label mapping changed since the manifest was built")
    text = ["patient_id", "aha_code", "signal_sha256", "exclusion_reasons", "review_flags"]
    rows = pd.read_csv(MANIFEST / "rows.csv", dtype={"ecg_id": str, **dict.fromkeys(text, str)})
    rows[text] = rows[text].fillna("")
    rows = rows.set_index("ecg_id")
    hashes["records_sha256"] = sph.records_sha256(list(rows.index))
    if hashes["records_sha256"] != metadata["records_sha256"]:
        raise ValueError("SPH record files differ from the manifest receipt")
    return rows, hashes


def check_counts(frame: pd.DataFrame) -> dict[str, object]:
    """
    Require the evaluation rows to give the label counts frozen in addendum v2.

    Parameters
    ----------
    frame : pd.DataFrame
        The ``use_evaluation`` rows.

    Returns
    -------
    dict[str, object]
        Record, patient and label counts.

    Raises
    ------
    ValueError
        If the counts changed or an age is missing.
    """
    counts = {"records": len(frame),
              "primary": frame["primary"].value_counts(dropna=False).rename(str).to_dict()}
    if counts != EXPECTED_EVALUATION or frame["age"].isna().any():
        raise ValueError(f"SPH evaluation table changed: {counts}")
    return {**counts, "patients": int(frame["patient_id"].nunique()),
            "secondary": frame["secondary"].value_counts(dropna=False).rename(str).to_dict()}


def read_checked(ecg_id: str, signal_sha256: str) -> np.ndarray:
    """
    Read the first 10 s of one SPH record and require it to match the manifest's window hash.

    Parameters
    ----------
    ecg_id : str
        Record identifier.
    signal_sha256 : str
        Manifest hash of the float32 window.

    Returns
    -------
    np.ndarray
        Float32 ``[12, 5000]`` window in mV.

    Raises
    ------
    ValueError
        If the window differs from the manifest.
    """
    window = sph.read_window(ecg_id)
    if hashlib.sha256(window.tobytes()).hexdigest() != signal_sha256:
        raise ValueError(f"SPH window differs from the manifest: {ecg_id}")
    return window


def median_residual(signal: np.ndarray) -> float:
    """
    Larger of the median violations of III = II - I and aVR = -(I + II) / 2, in mV.

    A swapped or inverted limb lead violates an identity at most samples, while a single corrupted
    sample only moves the maximum, so the median tests lead order.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(12, samples)`` in canonical lead order.

    Returns
    -------
    float
        Larger median absolute residual.
    """
    lead_i, lead_ii, lead_iii, avr = signal[:4].astype(np.float64)
    return float(max(np.median(np.abs(lead_iii - (lead_ii - lead_i))),
                     np.median(np.abs(avr + (lead_i + lead_ii) / 2))))


def check_leads(frame: pd.DataFrame) -> dict[str, object]:
    """
    Check lead order and mV units on evenly spaced records.

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.

    Returns
    -------
    dict[str, object]
        Largest per-record median and maximum Einthoven residuals, the records whose maximum exceeds the
        tolerance, and the range of the per-record median lead standard deviation.

    Raises
    ------
    ValueError
        If a median residual or the amplitudes are outside the limits.
    """
    medians, maxima, spreads = {}, {}, []
    for ecg_id, row in evenly_spaced(frame, LEAD_CHECK_RECORDS).iterrows():
        window = read_checked(ecg_id, row["signal_sha256"])
        medians[ecg_id] = median_residual(window)
        maxima[ecg_id] = sph.einthoven_residual(window)
        spreads.append(float(np.median(window.std(axis=1))))
    low, high = LEAD_STD_RANGE_MV
    if max(medians.values()) > EINTHOVEN_TOLERANCE_MV or not low <= min(spreads) <= max(spreads) <= high:
        raise ValueError(f"SPH lead order or units unexpected: {max(medians.values())}, {min(spreads)}, "
                         f"{max(spreads)}")
    return {"records": len(medians), "max_median_einthoven_residual_mv": max(medians.values()),
            "max_einthoven_residual_mv": max(maxima.values()),
            "records_with_sample_residual_over_tolerance": {
                ecg_id: value for ecg_id, value in maxima.items() if value > EINTHOVEN_TOLERANCE_MV},
            "median_lead_std_min_mv": min(spreads), "median_lead_std_max_mv": max(spreads)}


def open_caches() -> tuple[Caches, dict[str, dict[str, str]]]:
    """
    Open the ECG-JEPA and xECG feature caches.

    Returns
    -------
    tuple[Caches, dict[str, dict[str, str]]]
        Features and integer ECG IDs per encoder, and their verified hashes.
    """
    jepa, jepa_ids, jepa_hashes = jepa_cache()
    xecg, xecg_ids, xecg_hashes = xecg_cache()
    return {"jepa": (jepa, jepa_ids), "xecg": (xecg, xecg_ids)}, {"jepa": jepa_hashes, "xecg": xecg_hashes}


def identity(rows: pd.DataFrame, frame: pd.DataFrame, source_hashes: dict[str, object],
             cache_hashes: dict[str, dict[str, str]]) -> dict[str, object]:
    """
    Hash every input of the readout.

    Parameters
    ----------
    rows : pd.DataFrame
        All manifest rows.
    frame : pd.DataFrame
        Evaluation rows.
    source_hashes : dict[str, object]
        Verified SPH hashes from ``manifest_rows``.
    cache_hashes : dict[str, dict[str, str]]
        Verified feature cache hashes from ``open_caches``.

    Returns
    -------
    dict[str, object]
        The run identity a profile and a run must share.
    """
    encoder_sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files()}
    return {
        "sph": source_hashes, "sph_manifest_records": len(rows), "counts": check_counts(frame),
        "cpc_encoder": sha256_file(ENCODER), "cpc_normalization": sha256_file(NORMALIZATION),
        "feature_caches": cache_hashes, "experiment020": prior_identity(),
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


def ptb_features(model: nn.Module, name: str, stems: list[str]) -> np.ndarray:
    """
    Extract features of raw PTB-XL records with one encoder's historical input path.

    Parameters
    ----------
    model : nn.Module
        Frozen encoder.
    name : str
        ``jepa`` or ``xecg``.
    stems : list[str]
        ``filename_hr`` of each record.

    Returns
    -------
    np.ndarray
        Float32 features, one row per record.
    """
    chunks = []
    for start in range(0, len(stems), CHUNK):
        inputs = np.stack([PTB_INPUTS[name](stem) for stem in stems[start:start + CHUNK]])
        chunks.append(EXTRACTORS[name](model, inputs))
    return np.concatenate(chunks)


def integrity(models: dict[str, nn.Module], train: pd.DataFrame, caches: Caches) -> dict[str, object]:
    """
    Recompute evenly spaced cached training features and require them to match the caches.

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
        positions = cache_positions(cache_ids, train["ecg_id"].to_numpy())
        sample = evenly_spaced(train[positions >= 0], INTEGRITY_RECORDS)
        cached = cache[cache_positions(cache_ids, sample["ecg_id"].to_numpy())]
        recomputed = ptb_features(models[name], name, list(sample["filename_hr"]))
        difference = float(np.abs(recomputed.astype(np.float64) - cached).max())
        result[name] = {"ecg_ids": sample["ecg_id"].astype(int).tolist(), "max_abs_difference": difference}
        if difference > INTEGRITY_TOLERANCE:
            raise ValueError(f"Recomputed {name} features differ from the cache by {difference}")
    return result


def ptb_rows(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Training then development rows, the only PTB-XL rows this readout reads.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.

    Returns
    -------
    pd.DataFrame
        Concatenated rows in that order.
    """
    return pd.concat([groups["train"], groups["development"]])


def extract_missing(models: dict[str, nn.Module], groups: dict[str, pd.DataFrame],
                    caches: Caches) -> dict[str, np.ndarray]:
    """
    Extract the training and development rows each cache lacks.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.

    Returns
    -------
    dict[str, np.ndarray]
        ``<encoder>_ids`` and ``<encoder>`` features per encoder.
    """
    rows = ptb_rows(groups)
    arrays = {}
    for name, (_, cache_ids) in caches.items():
        missing = rows[cache_positions(cache_ids, rows["ecg_id"].to_numpy()) < 0]
        arrays[f"{name}_ids"] = missing["ecg_id"].to_numpy(dtype=np.int64)
        arrays[name] = ptb_features(models[name], name, list(missing["filename_hr"]))
    return arrays


def ptb_encoder_features(groups: dict[str, pd.DataFrame], caches: Caches,
                         extracted: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """
    ECG-JEPA and xECG features of the training and development rows.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.
    extracted : dict[str, np.ndarray]
        Output of ``extract_missing``.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Features keyed by encoder, then by ``train`` and ``development``.
    """
    ids = ptb_rows(groups)["ecg_id"].to_numpy(dtype=np.int64)
    split = len(groups["train"])
    result = {}
    for name, (cache, cache_ids) in caches.items():
        combined = combine_features(ids, cache, cache_ids, extracted[f"{name}_ids"], extracted[name])
        result[name] = {"train": combined[:split], "development": combined[split:]}
    return result


def extract_sph(models: dict[str, nn.Module], frame: pd.DataFrame) -> tuple[dict[str, np.ndarray],
                                                                             dict[str, float]]:
    """
    Read SPH windows chunkwise and extract every encoder's features.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    frame : pd.DataFrame
        Rows to featurize, with ``signal_sha256``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]
        Features per encoder, and seconds spent reading and per encoder (input conversion and model).
    """
    chunks = {name: [] for name in ENCODER_NAMES}
    seconds = dict.fromkeys(("read", *ENCODER_NAMES), 0.0)
    for start in range(0, len(frame), CHUNK):
        began = time.monotonic()
        part = frame.iloc[start:start + CHUNK]
        windows = [read_checked(ecg_id, digest) for ecg_id, digest in part["signal_sha256"].items()]
        seconds["read"] += time.monotonic() - began
        for name in ENCODER_NAMES:
            began = time.monotonic()
            inputs = np.stack([SPH_INPUTS[name](window) for window in windows])
            chunks[name].append(EXTRACTORS[name](models[name], inputs))
            torch.cuda.synchronize()
            seconds[name] += time.monotonic() - began
    return {name: np.concatenate(values) for name, values in chunks.items()}, seconds


def score_heads(heads: dict[str, Head], inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Probabilities of every head.

    Parameters
    ----------
    heads : dict[str, Head]
        Fitted heads keyed by name.
    inputs : dict[str, np.ndarray]
        Output of ``head_inputs``.

    Returns
    -------
    dict[str, np.ndarray]
        Probabilities keyed by head name, in ``ALL_HEADS`` order.
    """
    return {name: predict(heads[name], inputs[name]) for name in ALL_HEADS}


def label_scores(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Every head on the primary and secondary SPH labels (analyses 1 and 3).

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    scores : dict[str, np.ndarray]
        Probabilities per head.

    Returns
    -------
    dict[str, object]
        Scores keyed by label, then head.
    """
    result = {}
    for label in ("primary", "secondary"):
        mask = frame[label].notna().to_numpy()
        y = frame.loc[mask, label].to_numpy(dtype=np.int64)
        result[label] = {name: score(y, values[mask]) for name, values in scores.items()}
    return result


def contrasts(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Prespecified paired whole-patient bootstrap contrasts on the primary label (analyses 2, 3 and v3).

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    scores : dict[str, np.ndarray]
        Probabilities per head.

    Returns
    -------
    dict[str, object]
        Bootstrap result per contrast.
    """
    defined = frame["primary"].notna().to_numpy()
    y = frame.loc[defined, "primary"].to_numpy(dtype=np.int64)
    patients = frame.loc[defined, "patient_id"].to_numpy()
    return {name: patient_bootstrap(patients, y, scores[first][defined], scores[second][defined],
                                    seed=BOOTSTRAP_SEED)
            for name, (first, second) in CONTRASTS.items()}


def superclass_scores(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Each superclass against the primary negatives (analysis 4).

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    scores : dict[str, np.ndarray]
        Probabilities per head.

    Returns
    -------
    dict[str, object]
        Scores keyed by head, then superclass.
    """
    result = {}
    for head in READOUT_HEADS:
        result[head] = {}
        for name in sph.SUPERCLASSES:
            mask = ((frame[name] == 1) | (frame["primary"] == 0)).to_numpy()
            result[head][name] = score(frame.loc[mask, name].to_numpy(dtype=np.int64), scores[head][mask])
    return result


def policy_scores(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Primary label without the records the quality policy excludes (analysis 5, manifest ``use_training``).

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    scores : dict[str, np.ndarray]
        Probabilities per head.

    Returns
    -------
    dict[str, object]
        Scores keyed by head.
    """
    kept = (frame["primary"].notna() & frame["use_training"]).to_numpy()
    y = frame.loc[kept, "primary"].to_numpy(dtype=np.int64)
    return {head: score(y, scores[head][kept]) for head in READOUT_HEADS}


def development_scores(development: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Every head on the labeled full-development ECGs, overall and split into original and added.

    Parameters
    ----------
    development : pd.DataFrame
        Experiment 020 development rows.
    scores : dict[str, np.ndarray]
        Probabilities per head on those rows.

    Returns
    -------
    dict[str, object]
        Scores keyed by head, then subset.
    """
    defined = development["standard"].notna().to_numpy()
    frame = development[defined]
    y = frame["standard"].to_numpy(dtype=np.int64)
    subsets = {"full": np.ones(len(frame), bool), "original": frame["original"].to_numpy(),
               "added": ~frame["original"].to_numpy()}
    return {name: {subset: score(y[mask], values[defined][mask]) for subset, mask in subsets.items()}
            for name, values in scores.items()}


def fit_encoder_heads(groups: dict[str, pd.DataFrame], caches: Caches, extracted: dict[str, np.ndarray],
                      ) -> tuple[dict[str, Head], dict[str, dict[str, np.ndarray]]]:
    """
    Assemble the PTB-XL encoder features and fit the ECG-JEPA and xECG heads.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.
    extracted : dict[str, np.ndarray]
        Output of ``extract_missing`` or its saved copy.

    Returns
    -------
    tuple[dict[str, Head], dict[str, dict[str, np.ndarray]]]
        Heads keyed by name and the features from ``ptb_encoder_features``.
    """
    features = ptb_encoder_features(groups, caches, extracted)
    with threadpool_limits(limits=1):
        heads = encoder_heads(groups["train"], {name: values["train"] for name, values in features.items()})
    return heads, features


def profile(frame: pd.DataFrame, groups: dict[str, pd.DataFrame], caches: Caches,
            run_identity: dict[str, object], started: float) -> None:
    """
    Check feature integrity, extract the missing PTB-XL rows, time SPH extraction and gate the run.

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.
    run_identity : dict[str, object]
        Output of ``identity`` with the checks added by ``main``.
    started : float
        Monotonic start time of the stage.
    """
    with gpu_lock("cuda", blocking=False):
        models = load_models()
        checks = integrity(models, groups["train"], caches)
        began = time.monotonic()
        extracted = extract_missing(models, groups, caches)
        ptb_seconds = time.monotonic() - began
        began = time.monotonic()
        _, encoder_seconds = extract_sph(models, frame.iloc[:PROFILE_RECORDS])
        sph_seconds = time.monotonic() - began
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(OUTPUT / "ptb_features.npz", **extracted)
    heads, _ = fit_encoder_heads(groups, caches, extracted)
    preflight_seconds = time.monotonic() - started - ptb_seconds - sph_seconds
    projected = 2 * preflight_seconds + 1.5 * math.ceil(len(frame) / PROFILE_RECORDS) * sph_seconds + 900
    receipt = {
        "identity": run_identity, "integrity": checks, "profile_records": PROFILE_RECORDS,
        "profile_seconds": sph_seconds, "profile_seconds_by_part": encoder_seconds,
        "preflight_seconds": preflight_seconds, "ptb_extraction_seconds": ptb_seconds,
        "ptb_extracted_records": {name: len(extracted[f"{name}_ids"]) for name in caches},
        "ptb_features_sha256": sha256_file(OUTPUT / "ptb_features.npz"),
        "encoder_head_iterations": {name: int(model.n_iter_[0]) for name, (_, model) in heads.items()},
        "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
    }
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "gate_passed": receipt["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def save_outputs(frame: pd.DataFrame, development: pd.DataFrame, features: dict[str, np.ndarray],
                 sph_scores: dict[str, np.ndarray], dev_scores: dict[str, np.ndarray]) -> dict[str, str]:
    """
    Write the local SPH features and predictions and the development predictions.

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    development : pd.DataFrame
        Experiment 020 development rows.
    features : dict[str, np.ndarray]
        SPH features per encoder.
    sph_scores, dev_scores : dict[str, np.ndarray]
        Probabilities per head.

    Returns
    -------
    dict[str, str]
        SHA-256 of each written file.
    """
    ids = frame.index.to_numpy(dtype=str)
    write_npz_atomic(OUTPUT / "features.npz", ecg_ids=ids, **features)
    patients = frame["patient_id"].to_numpy(dtype=str)
    write_npz_atomic(OUTPUT / "predictions.npz", ecg_ids=ids, patient_ids=patients, **sph_scores)
    records = development.index.to_numpy(dtype=str)
    write_npz_atomic(OUTPUT / "development_predictions.npz", record_ids=records, **dev_scores)
    return {name: sha256_file(OUTPUT / name)
            for name in ("features.npz", "predictions.npz", "development_predictions.npz")}


def quality_counts(frame: pd.DataFrame) -> dict[str, object]:
    """
    Manifest quality-policy exclusion reasons and review flags of the evaluation rows.

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.

    Returns
    -------
    dict[str, object]
        Counts of each reason and flag, and the number of policy-excluded records.
    """
    counts = {column: dict(Counter(item for text in frame[column] for item in text.split(";") if item))
              for column in ("exclusion_reasons", "review_flags")}
    return {**counts, "excluded_records": int((~frame["use_training"]).sum())}


def run(frame: pd.DataFrame, groups: dict[str, pd.DataFrame], caches: Caches, run_identity: dict[str, object],
        cpc_heads: dict[str, Head], median_age: float) -> None:
    """
    Extract SPH features, score every head and write the aggregate result.

    Parameters
    ----------
    frame : pd.DataFrame
        Evaluation rows.
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Caches
        Output of ``open_caches``.
    run_identity : dict[str, object]
        Output of ``identity`` with the checks added by ``main``.
    cpc_heads : dict[str, Head]
        Reproduced Experiment 020 heads.
    median_age : float
        PTB-XL training median age.

    Raises
    ------
    ValueError
        If no matching passed profile exists or its PTB-XL features changed.
    """
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("A matching passed profile is required")
    if sha256_file(OUTPUT / "ptb_features.npz") != receipt["ptb_features_sha256"]:
        raise ValueError("PTB-XL features differ from the profile")
    with np.load(OUTPUT / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    with gpu_lock("cuda", blocking=False):
        models = load_models()
        checks = integrity(models, groups["train"], caches)
        began = time.monotonic()
        features, encoder_seconds = extract_sph(models, frame)
        seconds = time.monotonic() - began
    new_heads, encoder_features = fit_encoder_heads(groups, caches, extracted)
    heads = {**cpc_heads, **new_heads}
    development = groups["development"]
    dev_features = {"cpc": prior_cpc_features()["development"],
                    **{name: values["development"] for name, values in encoder_features.items()}}
    with threadpool_limits(limits=1):
        sph_scores = score_heads(heads, head_inputs(frame, median_age, features))
        dev_scores = score_heads(heads, head_inputs(development, median_age, dev_features))
        results = {"by_label": label_scores(frame, sph_scores), "contrasts": contrasts(frame, sph_scores),
                   "superclasses": superclass_scores(frame, sph_scores),
                   "primary_without_policy_exclusions": policy_scores(frame, sph_scores),
                   "development_standard_label": development_scores(development, dev_scores)}
    outputs = save_outputs(frame, development, features, sph_scores, dev_scores)
    result = {
        "status": "complete_external", "identity": run_identity,
        "profile_sha256": sha256_file(OUTPUT / "profile.json"), "integrity": checks,
        "extraction_seconds": seconds, "extraction_seconds_by_part": encoder_seconds,
        "superclass_counts": {name: int(frame[name].sum()) for name in sph.SUPERCLASSES},
        "quality": quality_counts(frame), "outputs_sha256": outputs, "calibration_test_evaluated": False,
        **results,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete"}), flush=True)


def main() -> None:
    """Profile or run the frozen SPH external readout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    rows, source_hashes = manifest_rows()
    frame = rows[rows["use_evaluation"]]
    caches, cache_hashes = open_caches()
    run_identity = identity(rows, frame, source_hashes, cache_hashes)
    run_identity["lead_check"] = check_leads(frame)
    groups = cohorts(ptb_table())
    with threadpool_limits(limits=1):
        cpc_heads, median_age, differences = ptb_heads(groups["train"], groups["development"],
                                                       prior_cpc_features())
    run_identity["experiment020_head_max_abs_difference"] = differences
    if args.stage == "profile":
        profile(frame, groups, caches, run_identity, started)
    else:
        run(frame, groups, caches, run_identity, cpc_heads, median_age)


if __name__ == "__main__":
    main()
