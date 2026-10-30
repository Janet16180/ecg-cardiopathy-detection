"""Replay admitted predecessor outputs before fitting any corrective cohort model."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .age_cohort_inputs import canonical_integral_id, read_selected_csv
from .cpu_screening041 import RAW_FEATURES
from .files import sha256_file, write_json_atomic
from .full_development import predict
from .measurement_extractor042 import extract_array, read_ecg
from .pr74_screening_io import (
    CPU_PRIOR,
    MEASURE_PRIOR,
    PRIOR,
    binding,
    cohort_tables,
    completed,
    finish,
    read_json,
    stored,
    verify_hashes,
)
from .pr74_screening_metrics import budget_threshold, point_metrics

RAW_FEATURES_PATH = Path("outputs/avance2_fusion/caracteristicas_primarias_ee39c8f6f7.csv")


def raw_features(input_root: Path, ecg_ids: list[str]) -> pd.DataFrame:
    """Read exactly the requested cached PTB feature rows, aligned by canonical administrative ID."""
    ids = [canonical_integral_id(value) for value in ecg_ids]
    rows = read_selected_csv(input_root / RAW_FEATURES_PATH, "ecg_id", set(ids), ("ecg_id", *RAW_FEATURES))
    rows["ecg_id"] = rows["ecg_id"].map(canonical_integral_id)
    if rows["ecg_id"].duplicated().any() or set(rows["ecg_id"]) != set(ids):
        raise ValueError("PTB measurement-cache identities changed")
    features = rows.set_index("ecg_id").loc[ids, list(RAW_FEATURES)].replace("", np.nan).astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def site_data(input_root: Path, site: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Align known development probabilities to their exact predecessor identities and roles."""
    frame = pd.read_csv(input_root / PRIOR / f"qualify/{site}_rows.csv", dtype=str)
    if site == "ptb":
        frame = frame.loc[pd.to_numeric(frame["strat_fold"]).ge(8)].reset_index(drop=True)
        evaluation = pd.to_numeric(frame["strat_fold"]).eq(9).to_numpy()
    elif site == "sph":
        evaluation = frame["role"].eq("evaluation").to_numpy()
    else:
        raise ValueError("Only predeclared PTB/SPH development roles are allowed")
    with np.load(input_root / PRIOR / f"evaluate/{site}_probabilities.npz", allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ("ecg_id", "patient_id"):
        left, right = frame[key].astype(str), pd.Series(arrays[key]).astype(str)
        if site == "ptb":
            left, right = left.map(canonical_integral_id), right.map(canonical_integral_id)
        if left.tolist() != right.tolist():
            raise ValueError(f"Predecessor {site} {key} order changed")
    y = pd.to_numeric(frame["target"]).to_numpy(dtype=int)
    if not np.array_equal(y, arrays["target"]) or not np.array_equal(evaluation, arrays["evaluation"]):
        raise ValueError("Predecessor labels or evaluation roles changed")
    frame["target"] = y
    frame["evaluation"] = evaluation
    frame["patient_id"] = ("ptbxl:" + frame["patient_id"].map(canonical_integral_id) + ".0"
                           if site == "ptb" else "sph:" + frame["patient_id"])
    return frame, arrays


def verify_points(input_root: Path) -> dict[str, Any]:
    """Reproduce SCREEN-01 point metrics and the frozen equal-weight baseline on both known sites."""
    saved = read_json(input_root / PRIOR / "evaluate/results.json")
    summaries = {}
    for site in ("ptb", "sph"):
        frame, arrays = site_data(input_root, site)
        held = arrays["evaluation"]
        y = frame["target"].to_numpy()
        summaries[site] = {}
        for name in ("hgb", "logistic", "xecg", "ecg_jepa"):
            threshold = budget_threshold(y[~held], arrays[name][~held])
            expected_threshold = saved["thresholds"][site][name]["threshold"]
            if threshold["threshold"] != expected_threshold:
                raise ValueError("Predecessor strict normal-budget threshold changed")
            observed = point_metrics(y[held], arrays[name][held], threshold["threshold"])
            old = saved["sites"][site]["arms"][name]
            for key in ("auroc", "average_precision", "sensitivity", "specificity"):
                if abs(observed[key] - old["metrics"][key]["value"]) > 1e-12:
                    raise ValueError(f"Predecessor {site}/{name}/{key} failed replay")
            expected_counts = old["counts"]
            for key, full in (("tp", "true_positive"), ("tn", "true_negative"),
                              ("fp", "false_positive"), ("fn", "false_negative")):
                if observed[key] != expected_counts[full]:
                    raise ValueError("Predecessor integer confusion matrix changed")
            summaries[site][name] = {"threshold": threshold, "metrics": observed}
        blend = 0.5 * arrays["hgb"] + 0.5 * arrays["xecg"]
        threshold = budget_threshold(y[~held], blend[~held])
        summaries[site]["blend"] = {"threshold": threshold,
                                     "metrics": point_metrics(y[held], blend[held], threshold["threshold"])}
    expected = {"tp": 2460, "fn": 1124, "fp": 346, "tn": 6504}
    if any(summaries["sph"]["blend"]["metrics"][name] != value for name, value in expected.items()):
        raise ValueError("Known SCREEN-02 equal-mixture confusion matrix did not reproduce")
    return summaries


def measurement_replay(input_root: Path) -> dict[str, Any]:
    """Require exact historical six-significant-digit extraction on 32 admitted fitting records."""
    rows = pd.read_csv(input_root / PRIOR / "qualify/ptb_rows.csv", dtype=str)
    selected = rows.loc[pd.to_numeric(rows["strat_fold"]).le(7)].copy()
    selected["sort"] = pd.to_numeric(selected["ecg_id"])
    selected = selected.sort_values("sort").head(32)
    actual = pd.DataFrame([
        extract_array(read_ecg(input_root / "data/raw/ptb-xl/1.0.3" / stem))
        for stem in selected["filename_hr"]
    ]).reindex(columns=RAW_FEATURES)
    actual = pd.read_csv(io.StringIO(actual.to_csv(index=False, float_format="%.6g")))
    expected = raw_features(input_root, selected["ecg_id"].tolist())
    if not np.array_equal(actual.to_numpy(), expected.to_numpy(), equal_nan=True):
        raise ValueError("Frozen 130-feature measurement replay failed")
    return {"records": len(actual), "serialized_numeric_exact": True, "features": len(RAW_FEATURES)}


def evaluation_features(input_root: Path, site: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Align already admitted development raw measurements and xECG representations without new extraction."""
    frame, _ = site_data(input_root, site)
    if site == "ptb":
        all_rows = pd.read_csv(input_root / PRIOR / "qualify/ptb_rows.csv", dtype={"ecg_id": str})
        positions = pd.Index(all_rows["ecg_id"]).get_indexer(frame["ecg_id"])
        vectors = np.load(input_root / PRIOR / "qualify/ptb_xecg.npy", mmap_mode="r")[positions]
        raw = raw_features(input_root, frame["ecg_id"].tolist()).reset_index(drop=True)
    else:
        vectors = np.load(input_root / PRIOR / "extract/sph_xecg.npy", mmap_mode="r")
        measurements = pd.read_csv(input_root / MEASURE_PRIOR / "preparation/measurements.csv",
                                   dtype={"ecg_id": str})
        if measurements["ecg_id"].duplicated().any():
            raise ValueError("SPH measurement identities duplicated")
        raw = measurements.set_index("ecg_id").loc[frame["ecg_id"], list(RAW_FEATURES)]
        raw = raw.astype(float).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if vectors.shape != (len(frame), 1024):
        raise ValueError("Development vector/identity alignment failed")
    return frame, np.asarray(vectors), raw


def model_replay(input_root: Path) -> dict[str, Any]:
    """Require saved trusted predecessor models to reproduce their already known development scores."""
    hgb = joblib.load(input_root / CPU_PRIOR / "fitted_models.joblib")["models"]["hgb"]
    head = joblib.load(input_root / PRIOR / "evaluate/fitted_heads.joblib")["xecg"]
    result = {}
    for site in ("ptb", "sph"):
        _, vectors, raw = evaluation_features(input_root, site)
        _, old = site_data(input_root, site)
        actual = {"hgb": hgb.predict_proba(raw)[:, 1], "xecg": predict(head, vectors)}
        result[site] = {}
        for name, scores in actual.items():
            maximum = float(np.max(np.abs(scores - old[name])))
            if maximum > 1e-12:
                raise ValueError(f"Saved model {site}/{name} failed prediction replay")
            result[site][name] = {"maximum_absolute_error": maximum, "absolute_tolerance": 1e-12}
    return result


def qualify(input_root: Path, output: Path) -> dict[str, Any]:
    """Verify cohort/predecessor artifacts and replay known outputs before new training or extraction."""
    receipt = binding()
    started = time.perf_counter()
    directory = input_root / output / "qualify"
    directory.mkdir(parents=True, exist_ok=False)
    cohorts, hashes = cohort_tables(input_root)
    for stage in ("qualify", "extract", "evaluate"):
        prior_receipt, bound = completed(input_root, PRIOR / stage)
        hashes.update(bound)
        if stage == "qualify":
            hashes.update(prior_receipt["input_sha256"])
    verify_hashes(input_root, hashes)
    for stage in ("qualification", "preparation", "evaluation"):
        _, bound = completed(input_root, MEASURE_PRIOR / stage)
        hashes.update(bound)
    paths = (CPU_PRIOR / "result.json", CPU_PRIOR / "fitted_models.joblib", RAW_FEATURES_PATH)
    hashes.update({stored(path): sha256_file(input_root / path) for path in paths})
    result = {"predecessor": verify_points(input_root), "measurement_replay": measurement_replay(input_root),
              "saved_model_replay": model_replay(input_root),
              "cohort_records": {name: len(rows) for name, rows in cohorts.items()}}
    write_json_atomic(directory / "replay.json", result)
    return finish(input_root, output, "qualify", receipt, hashes,
                  {"wall_seconds": time.perf_counter() - started, "replay": result})
