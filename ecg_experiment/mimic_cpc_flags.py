"""Count fixed CPC classifier flags in the existing MIMIC waveform cache."""

from __future__ import annotations

import csv
import gzip
import json
import os
import time
from collections import Counter

import numpy as np
import torch

from . import ROOT
from .cpc import CPCClassifier
from .cpc_pool import Pool, loader
from .files import sha256_file, write_json_atomic
from .gpu import gpu_lock

CACHE = ROOT / "data/processed/cpc_pool_40k"
MODEL_DIR = ROOT / "outputs/experiment004_cpc_40k/cpc_fraction1_seed42"
NORMALIZATION = MODEL_DIR.parent / "normalization.json"
OUTPUT = ROOT / "outputs/mimic_cpc_flag_audit"
BATCH_SIZE = 128
PROFILE_RECORDS = 512
MAX_PROJECTED_SECONDS = 7200
SCORE_BANDS = (0.1, 0.25, 0.5, 0.75, 0.9)


def probability(logits: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Apply the frozen PTB-XL Platt mapping to classifier logits."""
    scaled = logits.astype(np.float64) * slope + intercept
    return (1 / (1 + np.exp(-scaled))).astype(np.float32)


def score(model: CPCClassifier, data: object, device: str) -> np.ndarray:
    """Score a batch iterator without retaining waveforms."""
    model.eval()
    parts = []
    with torch.inference_mode():
        for signals, _, _ in data:
            parts.append(model(signals.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(parts)


def replay_saved_test(
    pool: Pool, model: CPCClassifier, mean: np.ndarray, std: np.ndarray, device: str
) -> float:
    """Require current inference to reproduce historical saved test logits."""
    with (MODEL_DIR / "test_predictions.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))[:16]
    rows_by_id = {row["ecg_id"]: row for row in pool.rows}
    rows = [rows_by_id[row["ecg_id"]] for row in saved]
    data = loader(pool, rows, mean, std, BATCH_SIZE, False, None, device)
    actual = score(model, data, device)
    expected = np.array([float(row["raw_logit"]) for row in saved])
    maximum_error = float(np.max(np.abs(actual - expected)))
    if maximum_error > 1e-4:
        raise RuntimeError(f"Saved PTB-XL prediction replay failed: {maximum_error}")
    return maximum_error


def run(*, profile: bool = False) -> dict[str, object]:
    """Profile or count MIMIC flags with one GPU lock and no waveform output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with gpu_lock(device, blocking=False):
        pool = Pool(CACHE)
        mimic = [row for row in pool.rows if row["source"] == "mimic"]
        if len(mimic) != 39457:
            raise RuntimeError(f"Unexpected MIMIC pool size: {len(mimic)}")
        normalization = json.loads(NORMALIZATION.read_text())
        expected_hash = normalization["source"]["signals_sha256"]
        if pool.metadata["signals_sha256"] != expected_hash:
            raise RuntimeError("MIMIC cache identity differs from trained model")
        if sha256_file(CACHE / "rows.csv") != normalization["source"]["rows_sha256"]:
            raise RuntimeError("Cache row identity differs from trained model")
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
        checkpoint = torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=False)
        if checkpoint["fingerprint"] != json.loads((MODEL_DIR / "config.json").read_text())["fingerprint"]:
            raise RuntimeError("Checkpoint fingerprint differs from model config")
        model = CPCClassifier().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        replay_error = replay_saved_test(pool, model, mean, std, device)

        if not profile:
            gate = json.loads((OUTPUT / "profile.json").read_text())
            if gate["projected_seconds"] > MAX_PROJECTED_SECONDS:
                raise RuntimeError("Profile exceeded the cost gate")
        rows = mimic[:PROFILE_RECORDS] if profile else mimic
        data = loader(pool, rows, mean, std, BATCH_SIZE, False, None, device)
        start = time.monotonic()
        logits = score(model, data, device)
        elapsed = time.monotonic() - start
        projected = elapsed * len(mimic) / len(rows)
        if profile and projected > MAX_PROJECTED_SECONDS:
            raise RuntimeError(f"Projected inference exceeds cost gate: {projected:.1f}s")

        calibration = metrics["calibration"]
        probabilities = probability(logits, calibration["slope"], calibration["intercept"])
        flags = probabilities >= metrics["threshold"]
        flagged_patients = {row["patient_id"] for row, flag in zip(rows, flags, strict=True) if flag}
        score_bands = np.searchsorted(SCORE_BANDS, probabilities)
        bins = Counter(score_bands.tolist())
        report: dict[str, object] = {
            "stage": "profile" if profile else "complete",
            "source": "MIMIC-IV-ECG accepted records in cpc_pool_40k",
            "record_count": len(rows),
            "patient_count": len({row["patient_id"] for row in rows}),
            "flagged_records": int(flags.sum()),
            "flagged_patients": len(flagged_patients),
            "flagged_fraction": float(flags.mean()),
            "flagged_patient_fraction": len(flagged_patients) / len({row["patient_id"] for row in rows}),
            "threshold": metrics["threshold"],
            "calibration": calibration,
            "probability_bins_edges": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
            "probability_bin_counts": [bins[i] for i in range(6)],
            "seconds": elapsed,
            "projected_seconds": projected,
            "replay_max_logit_error": replay_error,
            "model_sha256": sha256_file(MODEL_DIR / "model.pt"),
            "rows_sha256": sha256_file(CACHE / "rows.csv"),
            "signals_sha256": expected_hash,
            "signals_hash_provenance": "completed cache receipt; full bytes not rehashed in this audit",
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if not profile:
            target = OUTPUT / "identified_scores.csv.gz"
            temporary = target.with_name(target.name + ".tmp")
            with gzip.open(temporary, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("ecg_id", "patient_id", "cpc_probability_ptb_calibrated",
                                 "flagged", "score_band_index"))
                writer.writerows(
                    (row["ecg_id"], row["patient_id"], f"{float(probability_value):.8f}",
                     int(flag), int(band))
                    for row, probability_value, flag, band in zip(
                        rows, probabilities, flags, score_bands, strict=True
                    )
                )
            os.replace(temporary, target)
            report["identified_scores"] = str(target.relative_to(ROOT))
            report["identified_scores_sha256"] = sha256_file(target)
        write_json_atomic(OUTPUT / ("profile.json" if profile else "report.json"), report)
        return report
