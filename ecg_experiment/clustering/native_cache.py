"""Verified training-only native ECG-JEPA feature caching for CLUSTER-03."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_experiment import ROOT
from ecg_experiment.clustering01 import train_patient
from ecg_experiment.clustering02 import permitted_metadata
from ecg_experiment.eda.signals import LEADS, canonical_order
from ecg_experiment.files import sha256_file
from ecg_experiment.localization.jepa_native import SOURCE_REVISION, SOURCE_SHA256, encode_signal
from ecg_experiment.localization.native_study import automatic_regions, descriptors
from ecg_experiment.native_reference_ntt05 import pool_region
from ecg_experiment.paths import from_stored, to_stored

CLUSTER01 = ROOT / "outputs/cluster01_physical_morphology"
CLUSTER01_SHA = "74f0c82918f158a61368d308acdab5d5e98fe759852c966de76867e5c99bc8e7"
NTT06 = ROOT / "outputs/ntt06_native_encoder/result.json"
NTT06_SHA = "3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3"
DATABASE = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
SOURCE_DIR = ROOT / f"third_party/ntt_native/sources/ECG_JEPA-{SOURCE_REVISION}"
CHECKPOINT = ROOT / "third_party/ntt_native/checkpoints/ecg-jepa-multiblock-epoch100.pth"
PROTOCOL = "docs/research/clustering/cluster03-native-reference/protocol.md"
SOURCES = (
    PROTOCOL,
    "ecg_experiment/clustering/native_cache.py",
    "scripts/experiments/clustering/cache_native.py",
    "tests/clustering/test_native_cache.py",
    "ecg_experiment/localization/jepa_native.py",
    "ecg_experiment/localization/native_study.py",
    "ecg_experiment/native_reference_ntt05.py",
    "ecg_experiment/same_wave_ntt04.py",
    "ecg_experiment/clustering01.py",
    "ecg_experiment/clustering02.py",
    "ecg_experiment/visual_reference_ntt03.py",
    "ecg_experiment/reference_transfer_ntt02.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/localization/__init__.py",
    "ecg_experiment/clustering/__init__.py",
    "ecg_experiment/eda/avance2_features.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
)
IDENTITY_COLUMNS = ("ecg_id", "patient_id", "strat_fold", "filename_hr")


class QualityExclusionError(ValueError):
    """An explicitly permitted waveform-quality failure, never an integrity failure."""


def frozen_sources() -> dict[str, str]:
    """Require exact committed implementation and pinned external architecture bytes."""
    hashes = {}
    for name in SOURCES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted experiment source: {name}")
        hashes[name] = digest
    for name, expected in SOURCE_SHA256.items():
        path = SOURCE_DIR / name
        if sha256_file(path) != expected:
            raise ValueError(f"External architecture differs: {name}")
        hashes[to_stored(path)] = expected
    return hashes


def verify_hashes(expected: dict[str, str], hashes: dict[str, str]) -> None:
    """Verify receipt-bound file bytes while omitting mutable status snapshots."""
    for name, digest in expected.items():
        path = from_stored(name)
        if path.name == "status.json":
            continue
        if sha256_file(path) != digest:
            raise ValueError(f"Receipt-bound input differs: {name}")
        hashes[name] = digest


def verify_identity_roles(identities: pd.DataFrame, all_ids: pd.DataFrame, manifest: dict) -> None:
    """Quarantine whole closed-fold patients and require exact PR67 training ownership.

    Parameters
    ----------
    identities : pd.DataFrame
        Selected identifier-only rows, before consuming features or waveform files.
    all_ids : pd.DataFrame
        Complete identifier-only PTB-XL table for whole-patient fold exclusion.
    manifest : dict
        Hash-verified CLUSTER-01 reservation manifest.
    """
    if len(identities) != 1999 or not identities.ecg_id.is_unique or not identities.patient_id.is_unique:
        raise ValueError("CLUSTER-01 cohort identity denominator differs")
    closed = set(all_ids.loc[all_ids.strat_fold.isin([9, 10]), "patient_id"].astype(int))
    allowed = {int(row["ecg_id"]): row for row in manifest["records"] if row["role"] == "train"}
    indexed = all_ids.set_index("ecg_id")
    for row in identities.itertuples(index=False):
        ecg, patient = int(row.ecg_id), int(row.patient_id)
        bound = allowed.get(ecg, {})
        if int(bound.get("patient_id", -1)) != patient or not train_patient(patient) or patient in closed:
            raise ValueError("Patient outside verified whole-patient training role")
        if row.strat_fold not in range(1, 9) or ecg not in indexed.index:
            raise ValueError("Forbidden or absent source identity")
        original = indexed.loc[ecg]
        if any(getattr(row, name) != original[name] for name in IDENTITY_COLUMNS[1:]):
            raise ValueError("Selected identity differs from source metadata")


def verified_support_patients(hashes: dict[str, str]) -> set[int]:
    """Read only support identities after checking frozen NTT-06 predecessor bindings."""
    if sha256_file(NTT06) != NTT06_SHA:
        raise ValueError("NTT-06 result differs")
    hashes[to_stored(NTT06)] = NTT06_SHA
    result = json.loads(NTT06.read_text())
    verify_hashes(result["input_sha256"], hashes)
    verify_hashes(result["artifact_sha256"], hashes)
    selection_path = ROOT / "outputs/ntt04_same_wave/selection_local.json"
    manifest_path = ROOT / "outputs/ntt04_same_wave/reservation_manifest_local.json"
    names = ("outputs/ntt04_same_wave/result.json", to_stored(selection_path), to_stored(manifest_path))
    verify_hashes({name: result["input_sha256"][name] for name in names}, hashes)
    predecessor = json.loads((ROOT / names[0]).read_text())
    verify_hashes({name: predecessor["artifact_sha256"][name] for name in names[1:]}, hashes)
    selection = json.loads(selection_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    allowed = {
        int(row["ecg_id"]): int(row["patient_id"]) for row in manifest["records"] if row["role"] == "train"
    }
    rows = [row for draw in selection["support_draws"] for row in draw["supports"]]
    if any(allowed.get(int(row["ecg_id"])) != int(row["patient_id"]) for row in rows):
        raise ValueError("NTT support identity outside frozen training role")
    patients = {int(row["patient_id"]) for row in rows}
    if len(patients) != 27:
        raise ValueError("NTT support patient denominator differs")
    return patients


def verified_cohort() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Verify predecessors and patient roles before consuming label-free feature tables.

    Returns
    -------
    tuple
        Numerically ECG-sorted identities, physical features, acquisition/quality nuisance
        measurements and bound hashes for 1993 patients, excluding all 27 support patients.
        No diagnostic, age or sex fields are interpreted.
    """
    path = CLUSTER01 / "result.json"
    if sha256_file(path) != CLUSTER01_SHA:
        raise ValueError("CLUSTER-01 result differs")
    hashes = {to_stored(path): CLUSTER01_SHA}
    result = json.loads(path.read_text())
    verify_hashes(result["outputs_sha256"], hashes)
    verify_hashes(result["input_sha256"], hashes)
    identities = pd.read_csv(CLUSTER01 / "analyzed_training_patients.csv", usecols=IDENTITY_COLUMNS)
    all_ids = pd.read_csv(DATABASE, usecols=IDENTITY_COLUMNS)
    manifest = json.loads((CLUSTER01 / "reservation_manifest.json").read_text())
    verify_identity_roles(identities, all_ids, manifest)
    supports = verified_support_patients(hashes)
    selected = identities.loc[~identities.patient_id.isin(supports)].sort_values("ecg_id")
    if len(selected) != 1993:
        raise ValueError("Support-excluded candidate denominator differs")
    features = pd.read_csv(CLUSTER01 / "morphology_features.csv")
    if features.shape != (1999, 53) or not np.isfinite(features.to_numpy()).all():
        raise ValueError("CLUSTER-01 physical feature matrix differs")
    features = features.iloc[selected.index].reset_index(drop=True)
    measures = pd.read_csv(CLUSTER01 / "physical_measurements.csv").set_index("ecg_id")
    measures = measures.loc[selected.ecg_id].reset_index(drop=True)
    metadata = permitted_metadata(DATABASE, selected.ecg_id)
    nuisance = pd.concat([measures, metadata[["device", "site", "quality"]]], axis=1)
    return selected.reset_index(drop=True), features, nuisance, hashes


def read_waveform(row: dict[str, Any], hashes: dict[str, str]) -> np.ndarray:
    """Hash both local waveform files before physical waveform access.

    Parameters
    ----------
    row : dict
        Already role-verified identifier row.
    hashes : dict
        Receipt mapping updated with the individual header and signal digests.

    Returns
    -------
    np.ndarray
        Canonical finite samples-by-twelve-leads physical-mV waveform.

    Raises
    ------
    QualityExclusionError
        Only a prospective units, sample rate, shape or finite-value failure.
    OSError
        Missing or unreadable local files; this always aborts the stage.
    """
    relative = Path(row["filename_hr"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "records500":
        raise ValueError("Waveform path outside verified 500 Hz source")
    path = DATABASE.parent / relative
    for suffix in (".hea", ".dat"):
        part = path.with_suffix(suffix)
        digest = sha256_file(part)
        name = to_stored(part)
        if name in hashes and hashes[name] != digest:
            raise ValueError("Previously bound waveform changed")
        hashes[name] = digest
    signal, fields = wfdb.rdsamp(str(path))
    if fields["fs"] != 500 or set(fields["units"]) != {"mV"}:
        raise QualityExclusionError("Expected 500 Hz physical millivolts")
    if signal.shape != (5000, 12) or not np.isfinite(signal).all():
        raise QualityExclusionError("Expected finite ten-second twelve-lead waveform")
    if len(fields["sig_name"]) != 12 or set(fields["sig_name"]) != set(LEADS):
        raise ValueError("Canonical twelve-lead names unavailable")
    return canonical_order(signal, fields["sig_name"])


def record_features(encoder: torch.nn.Module, signal: np.ndarray, ecg_id: int) -> dict[str, np.ndarray]:
    """Preserve frozen patient pooling and every normalized automatic QRS beat vector.

    Parameters
    ----------
    encoder : torch.nn.Module
        Strictly verified, frozen ECG-JEPA encoder.
    signal : np.ndarray
        Canonical physical-mV waveform with shape [5000,12].
    ecg_id : int
        Already verified original record identifier.

    Returns
    -------
    dict
        Float64 qrs/global_pair patient vectors, normalized region vectors and second bounds.
    """
    try:
        qrs, late = automatic_regions(signal)
    except ValueError as error:
        if str(error) == "Fewer than three complete automatic beats":
            raise QualityExclusionError(str(error)) from error
        raise
    tokens = encode_signal(encoder, signal.T, f"ptbxl:{ecg_id}:original-10s")
    patient = descriptors(tokens, qrs, late)
    result = {
        "qrs": patient["native_qrs"],
        "global_pair": patient["native_global_pair"],
        "region_vectors": np.stack([pool_region(tokens, region) for region in qrs]),
        "qrs_bounds": np.array([[region.start, region.stop] for region in qrs], dtype=np.float64),
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Native encoder or pooling produced nonfinite features")
    return result


def pack_features(rows: list[dict[str, Any]], features: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pack patient vectors and ragged beat vectors without storing dense contextual tokens."""
    if not rows or len(rows) != len(features):
        raise ValueError("Missing or misaligned accepted features")
    offsets = np.concatenate([[0], np.cumsum([len(value["region_vectors"]) for value in features])])
    return {
        "ecg_ids": np.array([row["ecg_id"] for row in rows], dtype=np.int64),
        "patient_ids": np.array([row["patient_id"] for row in rows], dtype=np.int64),
        "qrs": np.stack([value["qrs"] for value in features]),
        "global_pair": np.stack([value["global_pair"] for value in features]),
        "region_vectors": np.concatenate([value["region_vectors"] for value in features]),
        "region_offsets": offsets.astype(np.int64),
        "qrs_bounds": np.concatenate([value["qrs_bounds"] for value in features]),
    }


def runtime_fingerprint() -> dict[str, Any]:
    """Bind installed numerical libraries, CPU threads and device environment."""
    import os
    import platform
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "timm", "wfdb", "pandas", "threadpoolctl")
        },
        "device": "cpu",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
