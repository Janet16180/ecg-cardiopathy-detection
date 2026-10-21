"""Identity-gated cached inputs and exact predecessor replay for contextual contributions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_inputs import NTT_SHA, read_bound
from ecg_experiment.clustering01 import train_patient
from ecg_experiment.files import sha256_file
from ecg_experiment.localization.jepa_native import CONTEXT_SCOPE, EMBEDDING_SPACE, NATIVE_LEADS
from ecg_experiment.localization.native_study import (
    descriptors,
    native_predictions,
    native_summary,
    predecessor_inputs,
    regions_from_saved,
    reproduce_predecessor,
)
from ecg_experiment.native_reference_ntt05 import NativeTokens, lead_patch_tokens
from ecg_experiment.paths import to_stored

IDENTITY_SHA = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"


def admitted_selection(directory: Path, inputs: dict) -> dict:
    """Check every selected patient against frozen roles and whole-patient fold exclusion."""
    selection = json.loads((directory / "selection_local.json").read_text())
    manifest_path = ROOT / "outputs/ntt04_same_wave/reservation_manifest_local.json"
    manifest = json.loads(manifest_path.read_text())
    inputs[to_stored(manifest_path)] = sha256_file(manifest_path)
    allowed = {
        int(row["ecg_id"]): int(row["patient_id"]) for row in manifest["records"] if row["role"] == "train"
    }
    metadata = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    if sha256_file(metadata) != IDENTITY_SHA:
        raise ValueError("Frozen PTB identity metadata changed before admission")
    inputs[to_stored(metadata)] = IDENTITY_SHA
    identities = pd.read_csv(
        metadata,
        usecols=["ecg_id", "patient_id", "strat_fold", "filename_hr"],
    )
    forbidden = set(identities.loc[identities.strat_fold.isin([9, 10]), "patient_id"].astype(int))
    indexed = identities.set_index("ecg_id")
    supports = [row for group in selection["support_draws"] for row in group["supports"]]
    rows = {row["ecg_id"]: row for row in [*selection["targets"], *supports]}
    targets = {row["patient_id"] for row in selection["targets"]}
    support_patients = {row["patient_id"] for row in supports}
    if (len(rows), len(targets), len(support_patients)) != (87, 60, 27) or targets & support_patients:
        raise ValueError("Frozen target/support separation or denominator changed")
    for ecg, row in rows.items():
        patient = int(row["patient_id"])
        source = indexed.loc[ecg]
        if (
            allowed.get(ecg) != patient
            or not train_patient(patient)
            or patient in forbidden
            or int(source.patient_id) != patient
            or int(source.strat_fold) not in range(1, 9)
            or source.filename_hr != row["filename_hr"]
        ):
            raise ValueError("Selected identity outside original whole-patient training role")
    return selection


def cached_tokens(values: np.ndarray, ecg: int) -> NativeTokens:
    """Restore the frozen 200 ms lead-major token geometry without reading a waveform."""
    if values.shape != (400, 768) or values.dtype != np.float32:
        raise ValueError("Native dense cache shape or precision differs")
    return lead_patch_tokens(
        values.reshape(8, 50, 768),
        NATIVE_LEADS,
        250,
        50,
        2500,
        EMBEDDING_SPACE,
        CONTEXT_SCOPE,
        f"ptbxl:{ecg}:original-10s",
    )


def replay_cached_inputs() -> dict:
    """Verify immutable inputs, gate identities and reproduce every original descriptor and score."""
    directory = ROOT / "outputs/ntt06_native_encoder"
    inputs: dict[str, str] = {}
    receipt = read_bound(directory / "result.json", NTT_SHA, inputs)
    selection = admitted_selection(directory, inputs)
    old_selection, old_features, previous, old_result, inherited = predecessor_inputs()
    inputs.update(inherited)
    if selection != old_selection:
        raise ValueError("NTT-06 selected identities or references differ from predecessor")
    historical = reproduce_predecessor(selection, old_features, previous, old_result)
    features = {
        int(key): {name: np.asarray(value, dtype=float) for name, value in row.items()}
        for key, row in json.loads((directory / "feature_vectors_local.json").read_text()).items()
    }
    selected_ids = sorted(features)
    expected_ids = sorted(
        {row["ecg_id"] for row in selection["targets"]}
        | {row["ecg_id"] for group in selection["support_draws"] for row in group["supports"]}
    )
    if selected_ids != expected_ids:
        raise ValueError("Feature identity set differs from admitted patients")
    coordinates = json.loads((directory / "automatic_coordinates_local.json").read_text())
    if sorted(map(int, coordinates)) != expected_ids:
        raise ValueError("Region identity set differs from admitted patients")
    with np.load(directory / "dense_tokens_local.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["ecg_ids"], expected_ids):
            raise ValueError("Dense cache identity/order differs before token access")
        values = archive["values"]
    if values.shape != (87, 400, 768):
        raise ValueError("Dense cache record denominator differs")
    maximum = 0.0
    for ecg, array in zip(expected_ids, values, strict=True):
        regions = coordinates[str(ecg)]
        rebuilt = descriptors(
            cached_tokens(array, ecg), regions_from_saved(regions["qrs"]), regions_from_saved(regions["late"])
        )
        for name, actual in rebuilt.items():
            maximum = max(maximum, float(np.max(np.abs(actual - features[ecg][name]))))
    if maximum > 1e-12:
        raise ValueError("Dense-token predecessor descriptor reconstruction differs")
    predictions, banks = native_predictions(features, selection, previous, EMBEDDING_SPACE)
    saved = json.loads((directory / "predictions_local.json").read_text())
    score_error = compare_predictions(predictions, saved)
    correct = sum(row["correct"]["native_qrs"] for row in predictions)
    if correct != 165 or native_summary(predictions) != receipt["summary"]:
        raise ValueError("Predecessor native score summary or correct count differs")
    return {
        "receipt": receipt,
        "inputs": inputs,
        "selection": selection,
        "features": features,
        "coordinates": coordinates,
        "tokens": dict(zip(expected_ids, values, strict=True)),
        "predictions": predictions,
        "banks": banks,
        "replay": {
            "historical": historical,
            "records_reconstructed": 87,
            "descriptors_per_record": 4,
            "maximum_descriptor_error": maximum,
            "maximum_score_error": score_error,
            "predictions": 180,
            "native_qrs_correct": correct,
            "complete_native_summary_exact": True,
            "whole_patient_training_gate_passed": True,
        },
    }


def compare_predictions(actual: list[dict], expected: list[dict]) -> float:
    """Require all predecessor choices and identities, with bounded float score differences."""
    if len(actual) != len(expected) or len(actual) != 180:
        raise ValueError("Predecessor prediction denominator differs")
    maximum = 0.0
    for row, saved in zip(actual, expected, strict=True):
        for key in ("patient_id", "ecg_id", "finding", "truth", "seed", "chosen", "correct"):
            if row[key] != saved[key]:
                raise ValueError(f"Predecessor prediction differs: {key}")
        for method, values in row["scores"].items():
            maximum = max(maximum, float(np.max(np.abs(np.asarray(values) - saved["scores"][method]))))
    if maximum > 1e-12:
        raise ValueError("Predecessor numerical scores differ")
    return maximum
