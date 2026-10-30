"""Profile and extract only admitted training features for corrective evaluation."""

from __future__ import annotations

import gc
import io
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb

from .cpu_screening041 import RAW_FEATURES
from .downloads import parse_checksums
from .external_encoders import xecg_features, xecg_input
from .files import sha256_file, write_json_atomic
from .gpu import gpu_lock
from .measurement_extractor042 import extract_array
from .pr74_screening_io import PRIOR, binding, completed, finish, relative, stored, training_pool
from .public_sources import load_view, signal_sha256
from .xecg import RELEASE_SHA256, load_xecg

CHECKPOINT = Path("third_party/checkpoints/xecg")
BATCH = 16


@lru_cache(maxsize=4)
def official_checksums(path: Path) -> dict[str, str]:
    """Read an immutable official release checksum list once per process."""
    return parse_checksums(path.read_text())


def raw_pair(input_root: Path, row: pd.Series) -> tuple[Path, str, dict[str, str]]:
    """Bind an admitted training pointer to a safe official raw release and its original pair checksums."""
    path = relative(row["waveform_path"])
    parts = path.parts
    if parts[:2] != ("data", "raw"):
        raise ValueError("Training waveform pointer must stay in data/raw")
    release = Path(*parts[:4])
    if row["source"] == "ptbxl":
        if release != Path("data/raw/ptb-xl/1.0.3") or row["backend"] != "ptbxl_wfdb":
            raise ValueError("PTB source and backend disagree")
        suffix = ".dat"
    else:
        allowed = {"ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra"}
        if row["source"] not in allowed or row["backend"] != "challenge_wfdb":
            raise ValueError("Unapproved supervised source")
        if parts[4:6] != ("training", row["source"]):
            raise ValueError("Challenge source and waveform path disagree")
        suffix = ".mat"
    stem = path.relative_to(release).as_posix()
    raw = input_root / release
    official = official_checksums(raw / "SHA256SUMS.txt")
    expected = {ext: official[stem + ext] for ext in (".hea", suffix)}
    return raw, stem, expected


def read_training_signal(input_root: Path, row: pd.Series) -> np.ndarray:
    """Verify raw bytes and the admitted canonical hash before returning the precise feature input."""
    if row["split"] != "train" or str(row["samples"]) != "5000" or str(row["window_start"]) != "0":
        raise ValueError("Only admitted original ten-second training windows may be extracted")
    raw, stem, expected = raw_pair(input_root, row)
    for suffix, value in expected.items():
        if sha256_file(raw / (stem + suffix)) != value:
            raise ValueError("Training raw checksum changed before extraction")
    canonical, start, samples, _ = load_view(raw, stem, "strict_10s")
    if start != 0 or samples != 5000 or signal_sha256(canonical) != row["signal_sha256"]:
        raise ValueError("Training canonical window changed")
    if row["source"] == "ptbxl":
        precise, fields = wfdb.rdsamp(str(raw / stem))
        names = [name.upper() for name in fields["sig_name"]]
        order = [names.index(name.upper()) for name in ("I", "II", "III", "aVR", "aVL", "aVF",
                                                       "V1", "V2", "V3", "V4", "V5", "V6")]
        signal = precise[:, order].T
    else:
        signal = canonical.astype(np.float64)
    for suffix, value in expected.items():
        if sha256_file(raw / (stem + suffix)) != value:
            raise ValueError("Training raw checksum changed during extraction")
    return np.asarray(signal, dtype=np.float64)


def measurements(signals: list[np.ndarray]) -> pd.DataFrame:
    """Preserve frozen feature arithmetic and historical six-significant-digit serialization."""
    rows = pd.DataFrame([extract_array(signal.T.copy()) for signal in signals]).reindex(columns=RAW_FEATURES)
    return pd.read_csv(io.StringIO(rows.to_csv(index=False, float_format="%.6g")))[list(RAW_FEATURES)]


def load_encoder(input_root: Path) -> torch.nn.Module:
    """Load the exact released frozen xECG backbone without modifying any parameters."""
    model = load_xecg(input_root / CHECKPOINT, backend="vanilla", device="cuda",
                     xlstm_dir=input_root / "third_party/xecg-deps").eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Frozen encoder contract failed")
    return model


def checkpoint_hashes(input_root: Path) -> dict[str, str]:
    """Bind official source/configuration/weights and reject a different model release."""
    result = {}
    for name, expected in RELEASE_SHA256.items():
        path = CHECKPOINT / name
        if sha256_file(input_root / path) != expected:
            raise ValueError("Released xECG checkpoint changed")
        result[stored(path)] = expected
    return result


def profile(input_root: Path, output: Path) -> dict[str, Any]:
    """Measure real training-path cost and reproduce admitted PTB vectors before full extraction."""
    receipt = binding()
    _, hashes = completed(input_root, output / "qualify")
    pool, bound = training_pool(input_root)
    hashes.update(bound)
    hashes.update(checkpoint_hashes(input_root))
    directory = input_root / output / "profile"
    directory.mkdir(parents=True, exist_ok=False)
    challenge = pool[pool["source"] != "ptbxl"]
    selected = challenge.groupby("source", sort=True).head(16)
    training = pool.loc[pool["source"] == "ptbxl"].head(32).set_index("record_id")
    all_ptb = pd.read_csv(input_root / PRIOR / "qualify/ptb_rows.csv", dtype={"ecg_id": str})
    positions = pd.Index("ptbxl:" + all_ptb["ecg_id"]).get_indexer(training.index)
    if len(training) != 32 or np.any(positions < 0):
        raise ValueError("Exactly 32 admitted cached PTB training vectors are required")
    expected = np.load(input_root / PRIOR / "qualify/ptb_xecg.npy", mmap_mode="r")[positions]
    with gpu_lock("cuda", blocking=False), torch.cuda.device(0):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB is available on GPU0")
        torch.cuda.reset_peak_memory_stats()
        model = load_encoder(input_root)
        replay = xecg_features(model, np.stack([
            xecg_input(read_training_signal(input_root, row)) for _, row in training.iterrows()
        ]))
        if not np.allclose(replay, expected, atol=1e-5, rtol=1e-5):
            raise ValueError("Admitted PTB frozen xECG cache replay failed")
        torch.cuda.synchronize()
        started = time.perf_counter()
        signals = [read_training_signal(input_root, row) for _, row in selected.iterrows()]
        measured = measurements(signals)
        vectors = xecg_features(model, np.stack([xecg_input(signal) for signal in signals]))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    forecast = 2 * elapsed / len(selected) * len(challenge) + 300
    footprint = len(challenge) * (1024 * 4 + 130 * 8)
    gates = {"free_gpu_at_least_4gib": free >= 4 * 1024**3,
             "peak_under_3gib": peak < 3 * 1024**3,
             "forecast_under_2hours": forecast < 7200, "output_under_1gib": footprint < 1024**3}
    details = {"profile_records": len(selected), "challenge_training_records": len(challenge),
               "fullpath_seconds": elapsed, "forecast_seconds": forecast, "output_forecast_bytes": footprint,
               "initial_free_gpu_bytes": free, "peak_gpu_bytes": peak, "resource_gates": gates,
               "ptb_cache_replay_records": len(training),
               "ptb_cache_max_absolute_error": float(np.max(np.abs(replay.astype(float) - expected)))}
    selected.to_csv(directory / "rows.csv", index=False)
    measured.to_csv(directory / "measurements.csv", index=False)
    np.save(directory / "xecg.npy", vectors)
    write_json_atomic(directory / "profile.json", details)
    if not all(gates.values()):
        raise RuntimeError("The predeclared feature resource profile failed")
    return finish(input_root, output, "profile", receipt, hashes, details)


def extract(input_root: Path, output: Path) -> dict[str, Any]:
    """Extract the admitted Challenge training union only after every measured profile gate passes."""
    receipt = binding()
    profiled, hashes = completed(input_root, output / "profile")
    if not all(profiled["resource_gates"].values()):
        raise ValueError("A completed passing profile is required")
    pool, bound = training_pool(input_root)
    hashes.update(bound)
    hashes.update(checkpoint_hashes(input_root))
    rows = pool[pool["source"] != "ptbxl"].reset_index(drop=True)
    directory = input_root / output / "extract"
    directory.mkdir(parents=True, exist_ok=False)
    rows.to_csv(directory / "rows.csv", index=False)
    vectors = np.lib.format.open_memmap(directory / "xecg.npy", mode="w+", dtype=np.float32,
                                        shape=(len(rows), 1024))
    features = np.full((len(rows), len(RAW_FEATURES)), np.nan)
    started = time.perf_counter()
    with gpu_lock("cuda", blocking=False), torch.cuda.device(0):
        torch.set_num_threads(2)
        model = load_encoder(input_root)
        for start in range(0, len(rows), BATCH):
            part = rows.iloc[start:start + BATCH]
            signals = [read_training_signal(input_root, row) for _, row in part.iterrows()]
            features[start:start + len(part)] = measurements(signals).to_numpy()
            vectors[start:start + len(part)] = xecg_features(
                model, np.stack([xecg_input(signal) for signal in signals]))
            if start % 512 == 0:
                vectors.flush()
                progress = {"records": start + len(part), "total": len(rows),
                            "wall_seconds": time.perf_counter() - started}
                write_json_atomic(directory / "progress.json", progress)
                print(f"Extracted {start + len(part)}/{len(rows)} admitted training ECGs", flush=True)
        vectors.flush()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite training encoder features")
    pd.DataFrame(features, columns=RAW_FEATURES).to_csv(directory / "measurements.csv", index=False)
    return finish(input_root, output, "extract", receipt, hashes,
                  {"records": len(rows), "wall_seconds": time.perf_counter() - started,
                   "protected_challenge_records_scored": 0, "encoder_fitted": False})
