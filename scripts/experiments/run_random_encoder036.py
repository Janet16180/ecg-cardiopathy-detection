"""Experiment 036: the normal reference and the supervised readout on random-weight and fine-tuned xECG."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from torch import nn

from ecg_experiment.challenge_features import (
    RAW_ROOTS,
    canonical_window,
    read_verified,
    skip_reasons,
    window_start,
)
from ecg_experiment.external_encoders import load_xecg_backbone, ptb_xecg_input, xecg_features, xecg_input
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.normal_manifold import fit_mahalanobis, mahalanobis_scores
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.random_encoder import (
    RANDOM_SEEDS,
    auroc_draws,
    build_random_xecg,
    decision,
    finetuned_head,
    interval_side,
    linear_contrast,
    load_finetuned_xecg,
    seed_mean_weights,
    state_digest,
)
from scripts.data.extract_challenge_features import load_checksums, ningbo_items
from scripts.experiments import run_normal_manifold026 as manifold026
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features, read_checked

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment036_random_encoder_v1"
PRIOR016 = ROOT / "outputs/experiment016_xecg_probe_finetune/random_head"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR026B = ROOT / "outputs/experiment026b_multisource_manifold_v1"
CHALLENGE_FEATURES = ROOT / "outputs/features_challenge_v1"
SETS = ("ptbxl_train", "ptbxl_development", "sph", "challenge_train")
RANDOM_ARMS = tuple(f"random_{seed}" for seed in RANDOM_SEEDS)
FINETUNED = "finetuned016"
EXTRACTED_ARMS = (*RANDOM_ARMS, FINETUNED)
ARMS = ("pretrained", *EXTRACTED_ARMS)
SCORES = ("mahalanobis", "readout")
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
EXPECTED_SETS = {"ptbxl_train": (17083, 9840), "ptbxl_development": (1572, 884), "sph": (21008, 7190),
                 "challenge_train": (22494, 17520)}
EXPECTED_FIT = 10846
EXPECTED_DEVELOPMENT = (1306, 843)
PROFILE_RECORDS = 128
CHECK_RECORDS = 512
CHUNK = 512
READER_THREADS = 2
ANALYSIS_WORKERS = 3
SCORING_WORKERS = 2
CEILING_SECONDS = 5400.0
DRAWS = 2000
SEED = 40040
MARGIN = 0.02
REPRODUCTION_TOLERANCE = 1e-9
LOGIT_TOLERANCE = 1e-4
SOURCES = (
    "ecg_experiment/random_encoder.py", "ecg_experiment/xecg.py", "ecg_experiment/external_encoders.py",
    "ecg_experiment/challenge_features.py", "ecg_experiment/normal_manifold.py",
    "ecg_experiment/multisource_readout.py", "ecg_experiment/external_readout.py",
    "ecg_experiment/full_development.py", "ecg_experiment/intervals.py", "ecg_experiment/sph.py",
    "scripts/experiments/run_random_encoder036.py", "scripts/experiments/run_sph_external022.py",
    "scripts/experiments/run_normal_manifold026.py", "scripts/data/extract_challenge_features.py",
    "scripts/experiments/run_multisource_calibration027b.py", "pyproject.toml", "uv.lock",
    "docs/experiment-036-random-encoder.md",
)
Reader = Callable[[dict[str, Any]], np.ndarray]
Task = tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[tuple[str, str]]]


def check_receipt(path: Path, expected: str, name: str) -> str:
    """
    Hash a file and require it to equal its receipt.

    Parameters
    ----------
    path : Path
        File to hash.
    expected : str
        SHA-256 recorded by the run that wrote it.
    name : str
        Name for the error message.

    Returns
    -------
    str
        The hash.

    Raises
    ------
    ValueError
        If the hash differs.
    """
    digest = sha256_file(path)
    if digest != expected:
        raise ValueError(f"{name} differs from its receipt")
    return digest


def ptb_sets(groups: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray],
                                                         dict[str, str]]:
    """
    PTB-XL training and development rows with a standard label, and their frozen xECG features.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, str]]
        Rows and frozen features per set, in 022's order, and the hashes checked.
    """
    prior = json.loads((PRIOR022 / "result.json").read_text())
    profile = json.loads((PRIOR022 / "profile.json").read_text())
    caches, cache_hashes = open_caches()
    if cache_hashes != prior["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    digest = check_receipt(PRIOR022 / "ptb_features.npz", profile["ptb_features_sha256"], "022 ptb_features")
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = ptb_encoder_features(groups, caches, extracted)["xecg"]
    rows, frozen = {}, {}
    for name, group, part in (("ptbxl_train", groups["train"], "train"),
                              ("ptbxl_development", groups["development"], "development")):
        labeled = group["standard"].notna().to_numpy()
        selected = group[labeled]
        rows[name] = pd.DataFrame({
            "key": selected["ecg_id"].astype(str).to_numpy(), "unit": selected["patient_id"].to_numpy(),
            "label": selected["standard"].to_numpy(dtype=np.int64),
            "filename_hr": selected["filename_hr"].to_numpy()})
        frozen[name] = np.asarray(encoded[part][labeled], dtype=np.float32)
    return rows, frozen, {"ptb_features.npz": digest, "caches": cache_hashes}


def sph_set() -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    """
    SPH evaluation rows in 026's order with their frozen Experiment 022 xECG features.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, dict[str, str]]
        Rows, frozen features and the hashes checked.
    """
    prior = json.loads((PRIOR022 / "result.json").read_text())
    hashes = {"features.npz": check_receipt(manifold026.INPUTS["sph_features"],
                                            prior["outputs_sha256"]["features.npz"], "022 SPH features"),
              "rows.csv": check_receipt(manifold026.SPH_ROWS, prior["identity"]["sph"]["rows"], "SPH rows")}
    sph, features = manifold026.load_sph()
    rows = pd.DataFrame({"key": sph["ecg_id"].to_numpy(), "unit": sph["patient_id"].to_numpy(),
                         "label": sph["primary"].to_numpy(dtype=np.int64),
                         "signal_sha256": sph["signal_sha256"].to_numpy(dtype=str),
                         **{name: sph[name].to_numpy() for name in SUPERCLASSES}})
    return rows, features["xecg"], hashes


def challenge_set() -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """
    022b's kept Challenge training rows with their frozen xECG features and Ningbo window hashes.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, dict[str, Any]]
        Rows in ``training_rows.csv`` order, frozen features and the hashes checked.

    Raises
    ------
    ValueError
        If a saved window start differs from 022b's.
    """
    receipt = json.loads((PRIOR022B / "result.json").read_text())["outputs_sha256"]
    hashes: dict[str, Any] = {"training_rows.csv": check_receipt(
        PRIOR022B / "training_rows.csv", receipt["training_rows.csv"], "022b training rows")}
    _, hashes["challenge_features"] = feature_identity()
    items, hashes["ningbo_manifest"] = ningbo_items()
    ningbo_windows = {record: window for record, _, window in items}
    table = pd.read_csv(PRIOR022B / "training_rows.csv", dtype={"record": str, "path": str})
    kept = table[table["kept"]].reset_index(drop=True)
    rows = pd.DataFrame({
        "key": (kept["source"] + ":" + kept["record"]).to_numpy(), "unit": kept["record"].to_numpy(),
        "label": kept["primary"].to_numpy(dtype=np.int64), "source": kept["source"].to_numpy(),
        "family": kept["family"].to_numpy(), "record": kept["record"].to_numpy(),
        "path": kept["path"].to_numpy(), "window_start": kept["window_start"].to_numpy(dtype=np.int64),
        "window_sha256": [ningbo_windows[record] if source == "ningbo" else ""
                          for source, record in zip(kept["source"], kept["record"], strict=True)]})
    frozen = np.zeros((len(rows), 1024), dtype=np.float32)
    for source, part in rows.groupby("source"):
        with np.load(CHALLENGE_FEATURES / f"{source}.npz") as saved:
            positions = pd.Index(saved["record"].astype(str)).get_indexer(part["record"])
            if (positions < 0).any() or not np.array_equal(saved["window_start"][positions],
                                                           part["window_start"].to_numpy()):
                raise ValueError(f"Challenge features differ from 022b's rows: {source}")
            frozen[part.index.to_numpy()] = saved["xecg"][positions]
    return rows, frozen, hashes


def load_sets() -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    """
    Every set's rows and frozen pretrained features, with counts checked.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]
        Rows and frozen features per set in ``SETS`` order, and the input hashes.

    Raises
    ------
    ValueError
        If a count differs from the protocol or a feature is nonfinite.
    """
    rows, frozen, ptb_hashes = ptb_sets(cohorts(ptb_table()))
    rows["sph"], frozen["sph"], sph_hashes = sph_set()
    rows["challenge_train"], frozen["challenge_train"], challenge_hashes = challenge_set()
    counts = {name: (len(rows[name]), int(rows[name]["label"].sum())) for name in SETS}
    if counts != EXPECTED_SETS:
        raise ValueError(f"Counts differ from the protocol: {counts}")
    for name in SETS:
        if not np.isfinite(frozen[name]).all() or frozen[name].shape != (len(rows[name]), 1024):
            raise ValueError(f"Invalid frozen features: {name}")
    hashes = {"ptbxl": ptb_hashes, "sph": sph_hashes, "challenge": challenge_hashes}
    return {name: rows[name] for name in SETS}, {name: frozen[name] for name in SETS}, hashes


def challenge_input(row: dict[str, Any], checksums: dict[str, dict[str, str]]) -> np.ndarray:
    """
    XECG input of one Challenge record at 022b's saved window.

    Parameters
    ----------
    row : dict[str, Any]
        Row of the ``challenge_train`` set.
    checksums : dict[str, dict[str, str]]
        Official checksums per source.

    Returns
    -------
    np.ndarray
        Float32 ``[1000, 12]`` input.

    Raises
    ------
    ValueError
        If the record is skipped, its window moved, or a Ningbo window differs from the manifest.
    """
    source = row["source"]
    signal, sampling_rate, names = read_verified(RAW_ROOTS[source], row["path"], checksums[source])
    start = window_start(len(signal))
    if skip_reasons(sampling_rate, signal) or start != row["window_start"]:
        raise ValueError(f"Challenge record differs from its feature extraction: {row['key']}")
    window = canonical_window(signal, names, start)
    if row["window_sha256"] and signal_sha256(window.astype(np.float32)) != row["window_sha256"]:
        raise ValueError(f"Ningbo window differs from the manifest: {row['key']}")
    return xecg_input(window)


def readers(checksums: dict[str, dict[str, str]]) -> dict[str, Reader]:
    """
    Build the input path of each set, identical to the frozen features'.

    Parameters
    ----------
    checksums : dict[str, dict[str, str]]
        Official Challenge checksums per source.

    Returns
    -------
    dict[str, Reader]
        One reader per set, taking a row dict and returning a float32 ``[1000, 12]`` input.
    """
    return {"ptbxl_train": lambda row: ptb_xecg_input(row["filename_hr"]),
            "ptbxl_development": lambda row: ptb_xecg_input(row["filename_hr"]),
            "sph": lambda row: xecg_input(read_checked(row["key"], row["signal_sha256"])),
            "challenge_train": partial(challenge_input, checksums=checksums)}


def finetuned_receipt() -> str:
    """
    Require the Experiment 016 fine-tuned model to match its completion receipt.

    Returns
    -------
    str
        The model's SHA-256.
    """
    receipt = json.loads((PRIOR016 / "complete.json").read_text())["sha256"]
    return check_receipt(PRIOR016 / "model.pt", receipt["model.pt"], "016 fine-tuned model")


def load_models(arms: tuple[str, ...]) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    """
    Load the requested arms on the GPU; each random model is built twice and must hash the same.

    Parameters
    ----------
    arms : tuple[str, ...]
        Names from ``ARMS``.

    Returns
    -------
    tuple[dict[str, nn.Module], dict[str, Any]]
        Models by arm and their state digests.

    Raises
    ------
    RuntimeError
        If a seeded construction is not reproducible.
    """
    models: dict[str, nn.Module] = {}
    info: dict[str, Any] = {"state_sha256": {}}
    for arm in arms:
        if arm == "pretrained":
            models[arm] = load_xecg_backbone()
        elif arm == FINETUNED:
            finetuned_receipt()
            models[arm] = load_finetuned_xecg(PRIOR016 / "model.pt").to("cuda")
        else:
            seed = int(arm.removeprefix("random_"))
            model = build_random_xecg(seed)
            digest = state_digest(model)
            if digest != state_digest(build_random_xecg(seed)):
                raise RuntimeError(f"Seeded xECG construction is not reproducible: {seed}")
            models[arm] = model.to("cuda")
        info["state_sha256"][arm] = state_digest(models[arm])
    return models, info


def extract(models: dict[str, nn.Module], reader: Reader, rows: list[dict[str, Any]]
            ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Pooled features of every arm for the given rows, read by ``READER_THREADS`` threads per chunk.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Models by arm, on the GPU.
    reader : Reader
        The set's input path.
    rows : list[dict[str, Any]]
        Rows to featurize.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]
        Float32 features per arm, and seconds spent reading and per arm.
    """
    chunks: dict[str, list[np.ndarray]] = {arm: [] for arm in models}
    seconds = dict.fromkeys(("read", *models), 0.0)
    with ThreadPoolExecutor(READER_THREADS) as pool:
        for start in range(0, len(rows), CHUNK):
            began = time.monotonic()
            inputs = np.stack(list(pool.map(reader, rows[start:start + CHUNK])))
            seconds["read"] += time.monotonic() - began
            for arm, model in models.items():
                began = time.monotonic()
                chunks[arm].append(xecg_features(model, inputs))
                torch.cuda.synchronize()
                seconds[arm] += time.monotonic() - began
    return {arm: np.concatenate(values) for arm, values in chunks.items()}, seconds


def max_difference(first: np.ndarray, second: np.ndarray) -> float:
    """
    Largest absolute difference between two feature arrays of the same shape.

    Parameters
    ----------
    first, second : np.ndarray
        Arrays to compare.

    Returns
    -------
    float
        The largest absolute difference.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    if first.shape != second.shape:
        raise ValueError(f"Shapes differ: {first.shape} and {second.shape}")
    return float(np.abs(first.astype(np.float64) - second.astype(np.float64)).max())


def identity(hashes: dict[str, Any]) -> dict[str, Any]:
    """
    Collect the identity a profile, an extraction and an analysis must share.

    Parameters
    ----------
    hashes : dict[str, Any]
        Input hashes from ``load_sets``.

    Returns
    -------
    dict[str, Any]
        Inputs, the fine-tuned model, seeds and source hashes.
    """
    return {"inputs": hashes, "finetuned_model_sha256": finetuned_receipt(),
            "random_seeds": list(RANDOM_SEEDS),
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES}}


def check_positions(rows: pd.DataFrame) -> np.ndarray:
    """
    Positions of the check sample: the first ``PROFILE_RECORDS`` and ``CHECK_RECORDS`` evenly spaced rows.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of one set.

    Returns
    -------
    np.ndarray
        Sorted unique positions.
    """
    spaced = np.linspace(0, len(rows) - 1, CHECK_RECORDS, dtype=np.int64)
    return np.unique(np.concatenate([np.arange(PROFILE_RECORDS), spaced]))


def profile(rows: dict[str, pd.DataFrame], frozen: dict[str, np.ndarray], run_identity: dict[str, Any]
            ) -> None:
    """
    Time the first ``PROFILE_RECORDS`` records of each set through every arm and gate the extraction.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Rows per set.
    frozen : dict[str, np.ndarray]
        Frozen pretrained features per set.
    run_identity : dict[str, Any]
        Output of ``identity``.
    """
    set_readers = readers(load_checksums())
    with gpu_lock("cuda", blocking=False):
        began = time.monotonic()
        models, info = load_models(ARMS)
        load_seconds = time.monotonic() - began
        timings, differences = {}, {}
        for name in SETS:
            part = rows[name].iloc[:PROFILE_RECORDS].to_dict("records")
            features, seconds = extract(models, set_readers[name], part)
            timings[name] = {key: value / len(part) for key, value in seconds.items()}
            differences[name] = max_difference(features["pretrained"], frozen[name][:PROFILE_RECORDS])
    projected = load_seconds
    for name in SETS:
        per = timings[name]
        extracted = sum(per[arm] for arm in EXTRACTED_ARMS)
        checked = len(check_positions(rows[name]))
        projected += len(rows[name]) * (per["read"] + extracted)
        projected += checked * (per["read"] + extracted + per["pretrained"])
    passed = projected <= CEILING_SECONDS and not any(differences.values())
    receipt = {"identity": run_identity, "profile_records_per_set": PROFILE_RECORDS,
               "seconds_per_record": timings, "model_load_seconds": load_seconds,
               "pretrained_max_abs_difference": differences, "state_sha256": info["state_sha256"],
               "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": passed}
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "projected_total_seconds": projected,
                      "pretrained_max_abs_difference": differences, "gate_passed": passed}), flush=True)


def completed_sets(run_identity: dict[str, Any]) -> dict[str, Any]:
    """
    Find the sets a previous extraction with the same identity finished, with unchanged feature files.

    Parameters
    ----------
    run_identity : dict[str, Any]
        Output of ``identity``.

    Returns
    -------
    dict[str, Any]
        Metadata per completed set.
    """
    path = OUTPUT / "extraction.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get("identity") != run_identity:
        return {}
    return {name: entry for name, entry in previous["sets"].items()
            if sha256_file(OUTPUT / f"features_{name}.npz") == entry["npz_sha256"]}


def load_features(name: str) -> dict[str, np.ndarray]:
    """
    Load the saved features of one set, by arm.

    Parameters
    ----------
    name : str
        Set name.

    Returns
    -------
    dict[str, np.ndarray]
        Features per extracted arm and the row ``keys``.
    """
    with np.load(OUTPUT / f"features_{name}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def check_pass(models: dict[str, nn.Module], set_readers: dict[str, Reader], rows: dict[str, pd.DataFrame],
               frozen: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Re-extract the check sample of every set with every arm and require identical features.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Every arm, on the GPU.
    set_readers : dict[str, Reader]
        Output of ``readers``.
    rows : dict[str, pd.DataFrame]
        Rows per set.
    frozen : dict[str, np.ndarray]
        Frozen pretrained features per set.

    Returns
    -------
    dict[str, Any]
        Checked records and the largest absolute difference per set and arm.

    Raises
    ------
    ValueError
        If any feature differs.
    """
    result = {}
    for name in SETS:
        positions = check_positions(rows[name])
        features, _ = extract(models, set_readers[name], rows[name].iloc[positions].to_dict("records"))
        saved = load_features(name)
        differences = {"pretrained": max_difference(features["pretrained"], frozen[name][positions]),
                       **{arm: max_difference(features[arm], saved[arm][positions])
                          for arm in EXTRACTED_ARMS}}
        result[name] = {"records": len(positions), "max_abs_difference": differences}
        print(json.dumps({"stage": f"check:{name}", **result[name]}), flush=True)
        if any(differences.values()):
            raise ValueError(f"Re-extracted features differ: {name} {differences}")
    return result


def run_extraction(rows: dict[str, pd.DataFrame], frozen: dict[str, np.ndarray],
                   run_identity: dict[str, Any]) -> None:
    """
    Extract every set with the random and fine-tuned arms, then run the check pass.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Rows per set.
    frozen : dict[str, np.ndarray]
        Frozen pretrained features per set.
    run_identity : dict[str, Any]
        Output of ``identity``.

    Raises
    ------
    ValueError
        If no matching passed profile exists.
    """
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("A matching passed profile is required")
    sets = completed_sets(run_identity)
    metadata = {"status": "running", "identity": run_identity,
                "profile_sha256": sha256_file(OUTPUT / "profile.json"), "sets": sets}
    set_readers = readers(load_checksums())
    began = time.monotonic()
    with gpu_lock("cuda", blocking=False):
        models, info = load_models(ARMS)
        extracted_models = {arm: models[arm] for arm in EXTRACTED_ARMS}
        for name in SETS:
            if name in sets:
                continue
            started = time.monotonic()
            features, seconds = extract(extracted_models, set_readers[name], rows[name].to_dict("records"))
            path = OUTPUT / f"features_{name}.npz"
            write_npz_atomic(path, keys=rows[name]["key"].to_numpy(dtype=str), **features)
            sets[name] = {"records": len(rows[name]), "seconds_by_part": seconds,
                          "seconds": time.monotonic() - started, "npz_sha256": sha256_file(path)}
            write_json_atomic(OUTPUT / "extraction.json", metadata)
            print(json.dumps({"stage": f"extract:{name}", "seconds": sets[name]["seconds"]}), flush=True)
        checks = check_pass(models, set_readers, rows, frozen)
    metadata.update({"status": "complete", "checks": checks, "state_sha256": info["state_sha256"],
                     "seconds_this_run": time.monotonic() - began})
    write_json_atomic(OUTPUT / "extraction.json", metadata)
    print(json.dumps({"stage": "extraction_complete", "seconds": metadata["seconds_this_run"]}), flush=True)


def score_arm(arm: str, features: dict[str, np.ndarray], fit: dict[str, np.ndarray],
              labels: np.ndarray, development: np.ndarray) -> dict[str, Any]:
    """
    Fit the normal reference and the supervised readout on one arm's features and score SPH and development.

    Parameters
    ----------
    arm : str
        Arm name, for the log.
    features : dict[str, np.ndarray]
        The arm's features per set.
    fit : dict[str, np.ndarray]
        Fit-set positions in ``ptbxl_train`` and ``challenge_train``, in 026b's order.
    labels : np.ndarray
        Standard labels of ``ptbxl_train`` followed by ``challenge_train``.
    development : np.ndarray
        Positions of 026's 1,306 development rows in ``ptbxl_development``.

    Returns
    -------
    dict[str, Any]
        Scores per score and set (``development_full`` for the 1,572 readout rows), fit details and seconds.
    """
    started = time.monotonic()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        fit_x = np.concatenate([features["ptbxl_train"][fit["ptbxl_train"]],
                                features["challenge_train"][fit["challenge_train"]]])
        reference = fit_mahalanobis(fit_x)
        scores = {"mahalanobis": {"sph": mahalanobis_scores(reference, features["sph"]),
                                  "development": mahalanobis_scores(reference,
                                                                    features["ptbxl_development"][development])}}
        details = {"pca_explained_variance": float(reference[1].explained_variance_ratio_.sum())}
        x = np.concatenate([features["ptbxl_train"], features["challenge_train"]])
        try:
            head = fit_readout(x, labels)
        except RuntimeError as error:
            details["readout"] = {"converged": False, "error": str(error)}
        else:
            full = predict(head, features["ptbxl_development"])
            scores["readout"] = {"sph": predict(head, features["sph"]), "development": full[development],
                                 "development_full": full}
            details["readout"] = {"converged": True, "iterations": int(head[1].n_iter_[0])}
    seconds = time.monotonic() - started
    print(json.dumps({"stage": f"scores:{arm}", **details, "seconds": seconds}), flush=True)
    return {"scores": scores, "details": details, "seconds": seconds}


def bootstrap_set(units: np.ndarray, y: np.ndarray, scores: dict[str, np.ndarray],
                  pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """
    AUROC draws of every score of one set, and the per-pair ``paired_auroc_difference`` contrasts.

    Parameters
    ----------
    units : np.ndarray
        Patient of each ECG.
    y : np.ndarray
        Labels.
    scores : dict[str, np.ndarray]
        Scores keyed ``<score>:<arm>``.
    pairs : list[tuple[str, str]]
        Keys compared with ``paired_auroc_difference``, first minus second.

    Returns
    -------
    dict[str, Any]
        Observed AUROCs, the draw matrix, skipped draws, average precision and the pair contrasts.
    """
    with threadpool_limits(limits=1):
        observed, matrix, skipped = auroc_draws(units, y, scores, DRAWS, SEED)
        pair_contrasts = {f"{first}_minus_{second}": paired_auroc_difference(
            units, y, scores[first], scores[second], DRAWS, SEED) for first, second in pairs}
        precision = {key: float(average_precision_score(y, values)) for key, values in scores.items()}
    return {"observed": observed, "matrix": matrix, "skipped_draws": skipped, "average_precision": precision,
            "pairs": pair_contrasts}


def fit_positions(rows: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """
    Positions of 026b's pooled fit set in ``ptbxl_train`` and ``challenge_train``, in its order.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Rows per set.

    Returns
    -------
    dict[str, np.ndarray]
        Positions per set.

    Raises
    ------
    ValueError
        If the fit set differs from its receipt, its count or its order (PTB-XL rows first).
    """
    receipt = json.loads((PRIOR026B / "result.json").read_text())["outputs_sha256"]["fit_sets.csv"]
    check_receipt(PRIOR026B / "fit_sets.csv", receipt, "026b fit sets")
    table = pd.read_csv(PRIOR026B / "fit_sets.csv", dtype={"record": str})
    pooled = table[table["pooled"].eq(True)]
    ptb = (pooled["family"] == "ptbxl").to_numpy()
    if len(pooled) != EXPECTED_FIT or not np.array_equal(ptb, np.sort(ptb)[::-1]):
        raise ValueError("The 026b pooled fit set differs from the protocol")
    ptb_index = pd.Index(rows["ptbxl_train"]["key"])
    challenge_index = pd.Index(rows["challenge_train"]["key"])
    positions = {"ptbxl_train": ptb_index.get_indexer(pooled.loc[ptb, "record"]),
                 "challenge_train": challenge_index.get_indexer(pooled.loc[~ptb, "source"] + ":"
                                                                + pooled.loc[~ptb, "record"])}
    if any((values < 0).any() for values in positions.values()):
        raise ValueError("A fit-set row has no features")
    return positions


def development_positions(rows: pd.DataFrame) -> np.ndarray:
    """
    Positions of 026's 1,306 development rows (026b's order) in ``ptbxl_development``.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of ``ptbxl_development``.

    Returns
    -------
    np.ndarray
        Positions.

    Raises
    ------
    ValueError
        If a row is missing or the count or positives differ from the protocol.
    """
    table = pd.read_csv(PRIOR026B / "development_scores.csv", dtype={"ecg_id": str})
    positions = pd.Index(rows["key"]).get_indexer(table["ecg_id"])
    labels = rows["label"].to_numpy()[positions]
    if (positions < 0).any() or (len(positions), int(labels.sum())) != EXPECTED_DEVELOPMENT:
        raise ValueError("026's development rows differ from the protocol")
    return positions


def finetuned_logits_check(features: np.ndarray, rows: pd.DataFrame) -> float:
    """
    Largest difference between 016's saved development logits and its head on the extracted features.

    Parameters
    ----------
    features : np.ndarray
        Fine-tuned arm features of ``ptbxl_development``.
    rows : pd.DataFrame
        Rows of ``ptbxl_development``.

    Returns
    -------
    float
        The largest absolute logit difference.
    """
    weight, bias = finetuned_head(PRIOR016 / "model.pt")
    with np.load(PRIOR016 / "development_logits.npz") as saved:
        positions = pd.Index(rows["key"]).get_indexer(saved["ecg_ids"].astype(str))
        if (positions < 0).any():
            raise ValueError("A 016 development logit has no extracted row")
        logits = features[positions].astype(np.float64) @ weight[0].astype(np.float64) + float(bias[0])
        return float(np.abs(logits - saved["logits"]).max())


def reproduce_pretrained(output: dict[str, Any], rows: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Require the pretrained arm's scores to reproduce 026b's pooled scores and 022b's pooled readout.

    Parameters
    ----------
    output : dict[str, Any]
        ``score_arm`` output of the pretrained arm.
    rows : dict[str, pd.DataFrame]
        Rows per set.

    Returns
    -------
    dict[str, float]
        Largest difference per comparison (relative for the Mahalanobis scores).

    Raises
    ------
    ValueError
        If a difference exceeds ``REPRODUCTION_TOLERANCE`` or a row order differs.
    """
    scores = output["scores"]
    sph = pd.read_csv(PRIOR026B / "sph_scores.csv", dtype={"ecg_id": str}, float_precision="round_trip")
    development = pd.read_csv(PRIOR026B / "development_scores.csv", dtype={"ecg_id": str},
                              float_precision="round_trip")
    if not np.array_equal(sph["ecg_id"].to_numpy(), rows["sph"]["key"].to_numpy()):
        raise ValueError("SPH order differs from 026b")
    differences = {}
    for name, table in (("sph", sph), ("development", development)):
        old = table["xecg_pooled"].to_numpy()
        differences[f"mahalanobis_{name}"] = float(np.max(np.abs(scores["mahalanobis"][name] - old)
                                                          / np.maximum(1.0, np.abs(old))))
    with np.load(PRIOR022B / "predictions.npz") as saved:
        if not (np.array_equal(saved["sph_ecg_ids"], rows["sph"]["key"].to_numpy(dtype=str))
                and np.array_equal(saved["development_record_ids"],
                                   ("ptbxl:" + rows["ptbxl_development"]["key"]).to_numpy(dtype=str))):
            raise ValueError("Row order differs from 022b")
        differences["readout_sph"] = float(np.abs(scores["readout"]["sph"] - saved["sph_xecg_pooled"]).max())
        differences["readout_development"] = float(np.abs(scores["readout"]["development_full"]
                                                          - saved["development_xecg_pooled"]).max())
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The pretrained arm does not reproduce 026b and 022b: {differences}")
    return differences


def evaluation_tasks(rows: dict[str, pd.DataFrame], development: np.ndarray, outputs: dict[str, Any]
                     ) -> dict[str, Task]:
    """
    Units, labels, scores and per-seed pairs of every evaluated set.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Rows per set.
    development : np.ndarray
        Output of ``development_positions``.
    outputs : dict[str, Any]
        ``score_arm`` output per arm.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[tuple[str, str]]]]
        Per set: units, labels, scores keyed ``<score>:<arm>`` and the pairs to contrast directly.
    """
    sph, dev = rows["sph"], rows["ptbxl_development"].iloc[development]

    def keyed(set_name: str, selection: np.ndarray | None = None) -> dict[str, np.ndarray]:
        found = {}
        for score in SCORES:
            for arm, output in outputs.items():
                if score in output["scores"]:
                    values = output["scores"][score][set_name]
                    found[f"{score}:{arm}"] = values if selection is None else values[selection]
        return found

    def pairs(scores: dict[str, np.ndarray]) -> list[tuple[str, str]]:
        chosen = []
        for score in SCORES:
            chosen += [(f"{score}:pretrained", f"{score}:{arm}") for arm in RANDOM_ARMS
                       if f"{score}:{arm}" in scores]
            if f"{score}:{FINETUNED}" in scores:
                chosen.append((f"{score}:{FINETUNED}", f"{score}:pretrained"))
        return chosen

    tasks = {}
    for name, frame in (("sph", sph), ("development", dev)):
        scores = keyed(name)
        tasks[name] = (frame["unit"].to_numpy(dtype=str), frame["label"].to_numpy(dtype=np.int64), scores,
                       pairs(scores))
    for condition in SUPERCLASSES:
        selected = ((sph[condition] == 1) | (sph["label"] == 0)).to_numpy()
        tasks[f"sph:{condition}"] = (sph.loc[selected, "unit"].to_numpy(dtype=str),
                                     sph.loc[selected, condition].to_numpy(dtype=np.int64),
                                     keyed("sph", selected), [])
    return tasks


def contrasts(evaluation: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the prespecified contrasts of one set from its draw matrix.

    Parameters
    ----------
    evaluation : dict[str, Any]
        Output of ``bootstrap_set``.

    Returns
    -------
    dict[str, Any]
        Pretrained minus the random mean, per-seed and fine-tuned contrasts per score, and the gap share.
    """
    observed, matrix = evaluation["observed"], evaluation["matrix"]
    result = {}
    for score in SCORES:
        randoms = tuple(f"{score}:{arm}" for arm in RANDOM_ARMS if f"{score}:{arm}" in observed)
        if f"{score}:pretrained" not in observed or len(randoms) != len(RANDOM_ARMS):
            continue
        result[f"{score}:pretrained_minus_random_mean"] = linear_contrast(
            observed, matrix, seed_mean_weights(f"{score}:pretrained", randoms))
        for key in randoms:
            result[f"{score}:pretrained_minus_{key.split(':')[1]}"] = linear_contrast(
                observed, matrix, {f"{score}:pretrained": 1.0, key: -1.0})
        if f"{score}:{FINETUNED}" in observed:
            result[f"{score}:{FINETUNED}_minus_pretrained"] = linear_contrast(
                observed, matrix, {f"{score}:{FINETUNED}": 1.0, f"{score}:pretrained": -1.0})
    if all(f"{score}:pretrained_minus_random_mean" in result for score in SCORES):
        weights = seed_mean_weights("mahalanobis:pretrained", tuple(f"mahalanobis:{a}" for a in RANDOM_ARMS))
        readout = tuple(f"readout:{arm}" for arm in RANDOM_ARMS)
        weights.update(seed_mean_weights("readout:pretrained", readout, -1.0))
        result["gap_mahalanobis_minus_gap_readout"] = linear_contrast(observed, matrix, weights)
    for value in result.values():
        value["side"] = interval_side(value)
    return result


def analyse(rows: dict[str, pd.DataFrame], frozen: dict[str, np.ndarray], run_identity: dict[str, Any]
            ) -> None:
    """
    Score every arm, check the pretrained reproduction, bootstrap and apply the decision rule.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Rows per set.
    frozen : dict[str, np.ndarray]
        Frozen pretrained features per set.
    run_identity : dict[str, Any]
        Output of ``identity``.

    Raises
    ------
    FileExistsError
        If the analysis has already run.
    ValueError
        If the extraction is incomplete or differs from its receipt.
    """
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 036 v1 has already been analysed")
    started = time.monotonic()
    extraction = json.loads((OUTPUT / "extraction.json").read_text())
    if extraction["status"] != "complete" or extraction["identity"] != run_identity:
        raise ValueError("A complete extraction with the same identity is required")
    features: dict[str, dict[str, np.ndarray]] = {"pretrained": frozen}
    for name in SETS:
        check_receipt(OUTPUT / f"features_{name}.npz", extraction["sets"][name]["npz_sha256"], name)
        saved = load_features(name)
        if not np.array_equal(saved["keys"], rows[name]["key"].to_numpy(dtype=str)):
            raise ValueError(f"Saved row order differs: {name}")
        for arm in EXTRACTED_ARMS:
            features.setdefault(arm, {})[name] = saved[arm]

    logit_difference = finetuned_logits_check(features[FINETUNED]["ptbxl_development"],
                                              rows["ptbxl_development"])
    arms = ARMS if logit_difference <= LOGIT_TOLERANCE else ARMS[:-1]
    fit = fit_positions(rows)
    development = development_positions(rows["ptbxl_development"])
    labels = np.concatenate([rows[name]["label"].to_numpy() for name in ("ptbxl_train", "challenge_train")])
    print(json.dumps({"stage": "inputs", "finetuned_logit_difference": logit_difference, "arms": list(arms)}),
          flush=True)

    with ProcessPoolExecutor(max_workers=SCORING_WORKERS) as executor:
        futures = {arm: executor.submit(score_arm, arm, features[arm], fit, labels, development)
                   for arm in arms}
        outputs = {arm: future.result() for arm, future in futures.items()}
    reproduction = reproduce_pretrained(outputs["pretrained"], rows)
    print(json.dumps({"stage": "reproduction", **reproduction}), flush=True)

    tasks = evaluation_tasks(rows, development, outputs)
    with ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
        futures = {name: executor.submit(bootstrap_set, *task) for name, task in tasks.items()}
        evaluations = {name: future.result() for name, future in futures.items()}
    readings = {}
    for name, evaluation in evaluations.items():
        readings[name] = {"records": len(tasks[name][1]), "positives": int(tasks[name][1].sum()),
                          "patients": len(np.unique(tasks[name][0])), "auroc": evaluation["observed"],
                          "average_precision": evaluation["average_precision"],
                          "skipped_draws": evaluation["skipped_draws"], "contrasts": contrasts(evaluation),
                          "paired_auroc_difference": evaluation["pairs"]}
        rounded = {key: round(value, 4) for key, value in evaluation["observed"].items()}
        print(json.dumps({"stage": f"auroc:{name}", **rounded}), flush=True)

    sph = readings["sph"]["contrasts"]
    primary = sph["mahalanobis:pretrained_minus_random_mean"]
    per_seed = [sph[f"mahalanobis:pretrained_minus_{arm}"] for arm in RANDOM_ARMS]
    direct = readings["sph"]["paired_auroc_difference"]
    consistency = max(abs(direct[f"mahalanobis:pretrained_minus_mahalanobis:{arm}"][bound]
                          - sph[f"mahalanobis:pretrained_minus_{arm}"][bound])
                      for arm in RANDOM_ARMS for bound in ("difference", "ci_low", "ci_high"))
    scores = {f"{set_name}_{score}_{arm}": values
              for arm, output in outputs.items() for score, by_set in output["scores"].items()
              for set_name, values in by_set.items()}
    write_npz_atomic(OUTPUT / "scores.npz", sph_keys=rows["sph"]["key"].to_numpy(dtype=str),
                     development_keys=rows["ptbxl_development"]["key"].to_numpy(dtype=str)[development],
                     development_full_keys=rows["ptbxl_development"]["key"].to_numpy(dtype=str), **scores)
    result = {
        "status": "complete", "identity": run_identity,
        "extraction_sha256": sha256_file(OUTPUT / "extraction.json"),
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
        "arms": list(arms), "random_seeds": list(RANDOM_SEEDS), "bootstrap_seed": SEED,
        "bootstrap_draws": DRAWS,
        "margin": MARGIN, "state_sha256": extraction["state_sha256"],
        "finetuned_logit_max_abs_difference": logit_difference,
        "pretrained_reproduction": reproduction,
        "per_seed_matrix_versus_paired_auroc_difference": consistency,
        "fits": {arm: {**output["details"], "seconds": output["seconds"]} for arm, output in outputs.items()},
        "sets": readings,
        "primary": {"contrast": primary, "per_seed": dict(zip(RANDOM_ARMS, per_seed, strict=True)),
                    "decision": decision(primary, per_seed, MARGIN)},
        "outputs_sha256": {"scores.npz": sha256_file(OUTPUT / "scores.npz")},
        "challenge_calibration_or_test_read": False, "ptbxl_calibration_or_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "primary": primary, "decision": result["primary"]["decision"]}),
          flush=True)


def main() -> None:
    """Profile, extract or analyse Experiment 036."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "extract", "analyse"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        rows, frozen, hashes = load_sets()
        run_identity = identity(hashes)
        if args.stage == "profile":
            profile(rows, frozen, run_identity)
        elif args.stage == "extract":
            run_extraction(rows, frozen, run_identity)
        else:
            analyse(rows, frozen, run_identity)


if __name__ == "__main__":
    main()
