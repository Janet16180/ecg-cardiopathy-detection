"""Experiment 027: Platt calibration and the at-least-95%-sensitivity threshold of the 022 heads, on SPH."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from ecg_experiment.cpc_pool import Pool
from ecg_experiment.external_encoders import load_jepa, load_xecg_backbone, source_files
from ecg_experiment.external_readout import (
    ENCODER,
    Head,
    cache_positions,
    evenly_spaced,
    load_encoder,
    prior_cpc_features,
    prior_identity,
    ptb_heads,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import CLEAN, cohorts, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.ptb_cpc_features import NORMALIZATION, POOL_DIR, pooled_features, ptb_signals
from ecg_experiment.screening_threshold import (
    bootstrap_summary,
    calibrate,
    fit_platt,
    head_logits,
    operating_point,
    screening_threshold,
)
from scripts.experiments.run_sph_external022 import fit_encoder_heads, integrity, open_caches, ptb_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment027_calibrated_threshold_v1"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
SPH_MANIFEST = ROOT / "data/processed/sph_clean_v1"
REFERENCES = ROOT / "data/processed/training_union_500hz_v1/heldout_references.csv"
HEADS = {"cpc": "cpc_standard", "jepa": "jepa_standard", "xecg": "xecg_standard"}
CONTRASTS = {"xecg_minus_cpc": ("xecg_standard", "cpc_standard"),
             "jepa_minus_cpc": ("jepa_standard", "cpc_standard"),
             "xecg_minus_jepa": ("xecg_standard", "jepa_standard")}
EXPECTED = {"calibration": (564, 504, 348), "development_full": (1572, 1413, 884),
            "development_original": (1306, 1173, 843), "sph": (21008, 20364, 7190)}
HEAD_TOLERANCE = 1e-9
INTEGRITY_RECORDS = 32
INTEGRITY_TOLERANCE = 1e-4
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 27027
TARGET_SENSITIVITY = 0.95
SOURCES = (
    "ecg_experiment/screening_threshold.py", "ecg_experiment/evaluation.py",
    "ecg_experiment/external_readout.py", "ecg_experiment/external_encoders.py",
    "ecg_experiment/full_development.py",
    "ecg_experiment/ptb_cpc_features.py", "ecg_experiment/cpc.py", "ecg_experiment/cpc_pool.py",
    "ecg_experiment/cpc_input_audit.py", "ecg_experiment/waveforms.py", "ecg_experiment/eda/ptbxl.py",
    "scripts/experiments/run_sph_external022.py", "scripts/experiments/run_calibrated_threshold027.py",
    "pyproject.toml", "uv.lock", "docs/experiment-027-calibrated-threshold.md",
)


def prior022_identity() -> dict[str, str]:
    """
    Hash the Experiment 022 v3 artifacts this readout reuses.

    Returns
    -------
    dict[str, str]
        SHA-256 of each artifact.

    Raises
    ------
    ValueError
        If an artifact differs from the hash recorded in 022's result or profile.
    """
    result = json.loads((PRIOR022 / "result.json").read_text())
    profile = json.loads((PRIOR022 / "profile.json").read_text())
    predictions = ("predictions.npz", "development_predictions.npz")
    names = (*predictions, "ptb_features.npz", "result.json", "profile.json")
    hashes = {name: sha256_file(PRIOR022 / name) for name in names}
    expected = {name: result["outputs_sha256"][name] for name in predictions}
    expected["ptb_features.npz"] = profile["ptb_features_sha256"]
    changed = any(hashes[name] != value for name, value in expected.items())
    if changed or result["status"] != "complete_external":
        raise ValueError("Experiment 022 artifacts differ from its receipts")
    return hashes


def counts(frame: pd.DataFrame, label: str) -> tuple[int, int, int]:
    """
    ECG, patient and positive counts.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``patient_id`` and a binary ``label`` column.
    label : str
        Label column.

    Returns
    -------
    tuple[int, int, int]
        ECGs, patients and positives.
    """
    return len(frame), int(frame["patient_id"].nunique()), int(frame[label].sum())


def calibration_rows(table: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    """
    PTB-XL calibration ECGs with a standard label.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``full_development.ptb_table``.
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Calibration rows of ``table`` and the hash of the references file.

    Raises
    ------
    ValueError
        If the references copies differ, a row is outside fold 9, a patient overlaps training or development,
        or the counts differ from the protocol.
    """
    digest = sha256_file(REFERENCES)
    if digest != sha256_file(CLEAN / "heldout_references.csv"):
        raise ValueError("The two heldout reference files differ")
    references = pd.read_csv(REFERENCES, dtype=str)
    frame = table.loc[references.loc[references["split"] == "calibration", "record_id"]]
    frame = frame[frame["standard"].notna()]
    if (frame["strat_fold"] != 9).any():
        raise ValueError("A calibration row is outside fold 9")
    others = set(groups["train"]["patient_id"]) | set(groups["development"]["patient_id"])
    if set(frame["patient_id"]) & others:
        raise ValueError("A calibration patient is also a training or development patient")
    if counts(frame, "standard") != EXPECTED["calibration"]:
        raise ValueError(f"Calibration rows changed: {counts(frame, 'standard')}")
    return frame, digest


def sph_rows(prior_result: dict[str, object]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    """
    SPH evaluation rows with a primary label and the saved 022 probabilities in the same order.

    Parameters
    ----------
    prior_result : dict[str, object]
        Experiment 022 ``result.json``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]
        Rows, probabilities per head and the manifest hashes.

    Raises
    ------
    ValueError
        If the manifest differs from its receipt or from 022, the saved IDs differ from the manifest, or the
        counts differ from the protocol.
    """
    metadata = json.loads((SPH_MANIFEST / "metadata.json").read_text())
    hashes = {"rows": sha256_file(SPH_MANIFEST / "rows.csv"),
              "metadata": sha256_file(SPH_MANIFEST / "metadata.json")}
    if hashes["rows"] != metadata["rows_sha256"] or \
            {name: prior_result["identity"]["sph"][name] for name in hashes} != hashes:
        raise ValueError("SPH manifest differs from its receipt or from Experiment 022")
    rows = pd.read_csv(SPH_MANIFEST / "rows.csv", dtype={"ecg_id": str, "patient_id": str})
    rows = rows[rows["use_evaluation"]]
    with np.load(PRIOR022 / "predictions.npz") as saved:
        if not (np.array_equal(saved["ecg_ids"], rows["ecg_id"].to_numpy(dtype=str))
                and np.array_equal(saved["patient_ids"], rows["patient_id"].to_numpy(dtype=str))):
            raise ValueError("Saved SPH predictions differ from the manifest rows")
        defined = rows["primary"].notna().to_numpy()
        scores = {name: saved[name][defined] for name in HEADS.values()}
    frame = rows[defined]
    if counts(frame, "primary") != EXPECTED["sph"]:
        raise ValueError(f"SPH rows changed: {counts(frame, 'primary')}")
    return frame, scores, hashes


def pool_identity(pool: Pool) -> dict[str, str]:
    """
    Hash the 250 Hz CPC pool files against its completion receipt.

    Parameters
    ----------
    pool : Pool
        The historical CPC pool.

    Returns
    -------
    dict[str, str]
        SHA-256 of the signals, rows and IDs.

    Raises
    ------
    ValueError
        If a file differs from the receipt.
    """
    files = {"signals_sha256": "signals.npy", "rows_sha256": "rows.csv", "ecg_ids_sha256": "ecg_ids.npy"}
    hashes = {key: sha256_file(POOL_DIR / name) for key, name in files.items()}
    if any(hashes[key] != pool.metadata[key] for key in files):
        raise ValueError("CPC pool differs from its receipt")
    return hashes


def reproduce_heads(groups: dict[str, pd.DataFrame], caches: dict[str, tuple[np.ndarray, np.ndarray]],
                    ) -> tuple[dict[str, Head], dict[str, float]]:
    """
    Refit the three 022 heads and require them to reproduce 022's development probabilities.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : dict[str, tuple[np.ndarray, np.ndarray]]
        Output of the 022 ``open_caches``.

    Returns
    -------
    tuple[dict[str, Head], dict[str, float]]
        Heads keyed by encoder and each head's largest absolute difference from 022.

    Raises
    ------
    ValueError
        If a head differs from 022 by more than ``HEAD_TOLERANCE``.
    """
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    with threadpool_limits(limits=1):
        cpc_heads, _, _ = ptb_heads(groups["train"], groups["development"], prior_cpc_features())
        encoder_heads, features = fit_encoder_heads(groups, caches, extracted)
    heads = {"cpc": cpc_heads["cpc_standard"], "jepa": encoder_heads["jepa_standard"],
             "xecg": encoder_heads["xecg_standard"]}
    development = {"cpc": prior_cpc_features()["development"], "jepa": features["jepa"]["development"],
                   "xecg": features["xecg"]["development"]}
    differences = {}
    with np.load(PRIOR022 / "development_predictions.npz") as saved:
        if not np.array_equal(saved["record_ids"], groups["development"].index.to_numpy(dtype=str)):
            raise ValueError("Development rows differ from Experiment 022")
        for name, head in heads.items():
            difference = np.abs(predict(head, development[name]) - saved[HEADS[name]])
            differences[HEADS[name]] = float(difference.max())
    if max(differences.values()) > HEAD_TOLERANCE:
        raise ValueError(f"Refitted heads differ from Experiment 022: {differences}")
    return heads, differences


def cpc_integrity(model: torch.nn.Module, pool: Pool, train: pd.DataFrame) -> dict[str, object]:
    """
    Recompute CPC features of evenly spaced training ECGs and compare them with Experiment 020's.

    Parameters
    ----------
    model : torch.nn.Module
        Unchanged starting CPC encoder.
    pool : Pool
        The historical CPC pool.
    train : pd.DataFrame
        Experiment 020 training rows, in the saved feature order.

    Returns
    -------
    dict[str, object]
        Checked ECG IDs and the largest absolute difference.

    Raises
    ------
    ValueError
        If the difference exceeds ``INTEGRITY_TOLERANCE``.
    """
    sample = evenly_spaced(train, INTEGRITY_RECORDS)
    cached = prior_cpc_features()["train"][train.index.get_indexer(sample.index)]
    recomputed = pooled_features(model, ptb_signals(pool, sample))
    difference = float(np.abs(recomputed.astype(np.float64) - cached).max())
    if difference > INTEGRITY_TOLERANCE:
        raise ValueError(f"Recomputed CPC features differ from Experiment 020 by {difference}")
    return {"ecg_ids": sample["ecg_id"].astype(int).tolist(), "max_abs_difference": difference}


def calibration_features(groups: dict[str, pd.DataFrame], calibration: pd.DataFrame, pool: Pool,
                         caches: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, np.ndarray],
                                                                                    dict[str, object]]:
    """
    Check feature integrity on the GPU, then extract CPC and xECG calibration features.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    calibration : pd.DataFrame
        Calibration rows.
    pool : Pool
        The historical CPC pool.
    caches : dict[str, tuple[np.ndarray, np.ndarray]]
        Output of the 022 ``open_caches``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, object]]
        Calibration features per encoder, and the integrity checks and timings.

    Raises
    ------
    ValueError
        If the ECG-JEPA cache lacks a calibration ECG.
    """
    jepa, jepa_ids = caches["jepa"]
    positions = cache_positions(jepa_ids, calibration["ecg_id"].to_numpy())
    if (positions < 0).any():
        raise ValueError("ECG-JEPA cache lacks a calibration ECG")
    features = {"jepa": np.asarray(jepa[positions], dtype=np.float32)}
    requested = time.monotonic()
    with gpu_lock("cuda", blocking=True):
        acquired = time.monotonic()
        models = {"cpc": load_encoder(), "jepa": load_jepa(), "xecg": load_xecg_backbone()}
        checks = {"cpc": cpc_integrity(models["cpc"], pool, groups["train"]),
                  **integrity(models, groups["train"], caches)}
        features["cpc"] = pooled_features(models["cpc"], ptb_signals(pool, calibration))
        features["xecg"] = ptb_features(models["xecg"], "xecg", list(calibration["filename_hr"]))
        torch.cuda.synchronize()
        released = time.monotonic()
    return features, {"integrity": checks, "gpu_lock_wait_seconds": acquired - requested,
                      "gpu_seconds": released - acquired, "gpu_name": torch.cuda.get_device_name(0)}


def evaluate(labels: dict[str, np.ndarray], patients: dict[str, np.ndarray],
             calibrated: dict[str, dict[str, np.ndarray]], thresholds: dict[str, float]) -> dict[str, object]:
    """
    Operating points of every head on every set, and the bootstrap contrasts on development and SPH.

    Parameters
    ----------
    labels, patients : dict[str, np.ndarray]
        Binary labels and patient IDs keyed by set.
    calibrated : dict[str, dict[str, np.ndarray]]
        Calibrated probabilities keyed by set, then head.
    thresholds : dict[str, float]
        Threshold per head.

    Returns
    -------
    dict[str, object]
        ``operating_points`` keyed by set then head, and ``bootstrap`` keyed by set.
    """
    points = {name: {head: operating_point(labels[name], values, thresholds[head])
                     for head, values in calibrated[name].items()}
              for name in labels}
    bootstrap = {}
    for name in ("development_full", "sph"):
        referred = {head: values >= thresholds[head] for head, values in calibrated[name].items()}
        bootstrap[name] = bootstrap_summary(patients[name], labels[name], referred, CONTRASTS,
                                            BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    return {"operating_points": points, "bootstrap": bootstrap}


def reading(bootstrap: dict[str, object]) -> dict[str, object]:
    """
    Apply the prespecified reading rules.

    Parameters
    ----------
    bootstrap : dict[str, object]
        ``bootstrap`` of ``evaluate``.

    Returns
    -------
    dict[str, object]
        Whether each head's threshold transfers to SPH, and the screening verdict of each contrast per set.
    """
    transfers = {}
    for head, rates in bootstrap["sph"]["heads"].items():
        sensitivity = rates["sensitivity"]
        transfers[head] = {"transfers": sensitivity["ci_low"] <= TARGET_SENSITIVITY <= sensitivity["ci_high"],
                           "point_at_or_above_target": sensitivity["value"] >= TARGET_SENSITIVITY}
    verdicts = {}
    for name in ("development_full", "sph"):
        verdicts[name] = {}
        for contrast, (first, second) in CONTRASTS.items():
            summary = bootstrap[name]["specificity_contrasts"][contrast]
            better = first if summary["ci_low"] > 0 else second if summary["ci_high"] < 0 else None
            verdicts[name][contrast] = {"better_for_screening": better}
    return {"threshold_transfers_to_sph": transfers, "specificity_contrasts": verdicts}


def development_sets(development: pd.DataFrame,
                     ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
    """
    Full and original development ECGs with a standard label, and their saved 022 probabilities.

    Parameters
    ----------
    development : pd.DataFrame
        Experiment 020 development rows, in the order of 022's saved predictions.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]
        Rows and probabilities per head, keyed by ``development_full`` and ``development_original``.

    Raises
    ------
    ValueError
        If the counts differ from the protocol.
    """
    labeled = development["standard"].notna().to_numpy()
    full = development[labeled]
    original = full["original"].to_numpy()
    with np.load(PRIOR022 / "development_predictions.npz") as saved:
        scores = {head: saved[head][labeled] for head in HEADS.values()}
    frames = {"development_full": full, "development_original": full[original]}
    for name, frame in frames.items():
        if counts(frame, "standard") != EXPECTED[name]:
            raise ValueError(f"{name} rows changed: {counts(frame, 'standard')}")
    return frames, {"development_full": scores,
                    "development_original": {head: values[original] for head, values in scores.items()}}


def fit_thresholds(y: np.ndarray, scores: dict[str, np.ndarray]) -> tuple[dict[str, LogisticRegression],
                                                                         dict[str, float]]:
    """
    Fit each head's Platt mapping and threshold on the calibration ECGs.

    Parameters
    ----------
    y : np.ndarray
        Calibration labels.
    scores : dict[str, np.ndarray]
        Calibration probabilities per head.

    Returns
    -------
    tuple[dict[str, LogisticRegression], dict[str, float]]
        Calibrator and threshold per head.
    """
    calibrators = {head: fit_platt(head_logits(values), y) for head, values in scores.items()}
    thresholds = {head: screening_threshold(y, calibrate(calibrators[head], head_logits(values)))
                  for head, values in scores.items()}
    return calibrators, thresholds


def main() -> None:
    """Fit the Platt mappings and thresholds on calibration ECGs and read them once on development and SPH."""
    torch.set_num_threads(1)
    started = time.monotonic()
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    pool = Pool(POOL_DIR)
    caches, cache_hashes = open_caches()
    if cache_hashes != prior_result["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    table = ptb_table()
    groups = cohorts(table)
    calibration, references_sha256 = calibration_rows(table, groups)
    sph, sph_scores, sph_hashes = sph_rows(prior_result)
    dev_frames, dev_scores = development_sets(groups["development"])
    identity = {
        "experiment022": prior022_identity(), "experiment020": prior_identity(), "sph_manifest": sph_hashes,
        "heldout_references": references_sha256, "cpc_pool": pool_identity(pool),
        "feature_caches": cache_hashes,
        "cpc_encoder": sha256_file(ENCODER), "cpc_normalization": sha256_file(NORMALIZATION),
        "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        "encoder_sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files()},
    }
    heads, head_differences = reproduce_heads(groups, caches)
    features, gpu = calibration_features(groups, calibration, pool, caches)
    with threadpool_limits(limits=1):
        cal_scores = {HEADS[name]: predict(head, features[name]) for name, head in heads.items()}

    frames = {"calibration": calibration, **dev_frames, "sph": sph}
    label_columns = {"calibration": "standard", "development_full": "standard",
                     "development_original": "standard", "sph": "primary"}
    labels = {name: frame[label_columns[name]].to_numpy(dtype=np.int64) for name, frame in frames.items()}
    patients = {name: frame["patient_id"].to_numpy(dtype=str) for name, frame in frames.items()}
    raw = {"calibration": cal_scores, **dev_scores, "sph": sph_scores}
    calibrators, thresholds = fit_thresholds(labels["calibration"], cal_scores)
    calibrated = {name: {head: calibrate(calibrators[head], head_logits(values))
                         for head, values in scores.items()}
                  for name, scores in raw.items()}
    results = evaluate(labels, patients, calibrated, thresholds)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(OUTPUT / "calibration_features.npz",
                     ecg_ids=calibration["ecg_id"].to_numpy(dtype=np.int64), **features)
    write_npz_atomic(OUTPUT / "calibrated_predictions.npz",
                     **{f"{name}_{head}": values for name, scores in calibrated.items()
                        for head, values in scores.items()},
                     **{f"{name}_labels": values for name, values in labels.items()},
                     **{f"{name}_patients": values for name, values in patients.items()})
    platt = {head: {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
             for head, model in calibrators.items()}
    result = {
        "status": "complete", "identity": identity, "head_max_abs_difference_from_022": head_differences,
        **gpu,
        "counts": {name: dict(zip(("records", "patients", "positives"), value, strict=True))
                   for name, value in EXPECTED.items()},
        "platt": platt, "thresholds": thresholds, "target_sensitivity": TARGET_SENSITIVITY, **results,
        "reading": reading(results["bootstrap"]), "ptbxl_test_read": False,
        "outputs_sha256": {name: sha256_file(OUTPUT / name)
                           for name in ("calibration_features.npz", "calibrated_predictions.npz")},
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
