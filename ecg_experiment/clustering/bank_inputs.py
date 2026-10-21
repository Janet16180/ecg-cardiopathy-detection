"""Verified cached inputs and exact predecessor replay for reference-bank studies."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache import verified_cohort, verify_hashes
from ecg_experiment.clustering.reference_geometry import (
    SEED,
    affinity_views,
    fit_model,
    fit_view,
    single_reference_fit,
    support_prototypes,
)
from ecg_experiment.clustering02 import anchor_patient, silhouette
from ecg_experiment.files import sha256_file
from ecg_experiment.localization.native_study import predecessor_inputs
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import SEEDS

CLUSTER_SHA = "64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9"
CACHE_SHA = "6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3"
NTT_SHA = "3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3"


def read_bound(path: Path, expected: str, inputs: dict) -> dict:
    """Verify a receipt and every immutable input/source/artifact before loading payloads."""
    if sha256_file(path) != expected:
        raise ValueError(f"Predecessor receipt changed: {to_stored(path)}")
    inputs[to_stored(path)] = expected
    receipt = json.loads(path.read_text())
    for field in ("source_sha256", "input_sha256", "artifact_sha256"):
        verify_hashes(receipt[field], inputs)
    return receipt


def committed_sources(names: tuple[str, ...], inherited: dict) -> dict:
    """Require current bytes to match committed source and all predecessor source bindings."""
    result = dict(inherited)
    for name in names:
        path = ROOT / name
        digest = sha256_file(path)
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted source: {name}")
        result[name] = digest
    verify_hashes(result, {})
    return result


def cached_inputs() -> dict:
    """Gate patient identities before feature consumption and admit exactly the retained cohort."""
    ids, physical, nuisance, inputs = verified_cohort()
    cluster_dir = ROOT / "outputs/cluster03_native_reference"
    cluster = read_bound(cluster_dir / "result.json", CLUSTER_SHA, inputs)
    cache_dir = ROOT / "outputs/cluster03_native_cache_v2"
    cache_receipt = read_bound(cache_dir / "result.json", CACHE_SHA, inputs)
    ntt_dir = ROOT / "outputs/ntt06_native_encoder"
    ntt = read_bound(ntt_dir / "result.json", NTT_SHA, inputs)
    selection, _, _, _, previous_inputs = predecessor_inputs()
    inputs.update(previous_inputs)
    saved_selection = json.loads((ntt_dir / "selection_local.json").read_text())
    if selection != saved_selection:
        raise ValueError("NTT support/target selection changed")
    with np.load(cache_dir / "native_features_local.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["ecg_ids"], ids.ecg_id) or not np.array_equal(
            archive["patient_ids"], ids.patient_id
        ):
            raise ValueError("Native cache identities differ from verified training patients")
        cache = {key: archive[key] for key in archive.files}
    split = json.loads((cluster_dir / "patient_splits_local.json").read_text())
    anchors = np.array(split["anchors"], dtype=int)
    samples = [np.array(row, dtype=int) for row in split["samples"]]
    if (
        split["ecg_ids"] != ids.ecg_id.tolist()
        or not np.array_equal(anchors, np.flatnonzero(ids.patient_id.map(anchor_patient)))
        or not np.array_equal(samples[0], np.flatnonzero(~ids.patient_id.map(anchor_patient)))
        or (len(ids), len(samples[0]), len(anchors)) != (1993, 1582, 411)
    ):
        raise ValueError("Frozen fitting/anchor split differs")
    with np.load(cluster_dir / "representations_local.npz", allow_pickle=False) as archive:
        views = {key: archive[key] for key in archive.files}
    with np.load(cluster_dir / "assignments_local.npz", allow_pickle=False) as archive:
        assignments = {key: archive[key] for key in archive.files}
    for name, values in (
        ("native_qrs", cache["qrs"]),
        ("native_global", cache["global_pair"]),
        ("physical", physical.to_numpy()),
    ):
        if not np.array_equal(views[name], values):
            raise ValueError("Cached predecessor representation differs")
    return {
        "ids": ids,
        "nuisance": nuisance,
        "cache": cache,
        "views": views,
        "assignments": assignments,
        "cluster": cluster,
        "cache_receipt": cache_receipt,
        "ntt": ntt,
        "selection": selection,
        "support_features": json.loads((ntt_dir / "feature_vectors_local.json").read_text()),
        "ntt_predictions": json.loads((ntt_dir / "predictions_local.json").read_text()),
        "fitting": samples[0],
        "anchors": anchors,
        "inputs": inputs,
    }


def replay_predecessor(data: dict, deadline: float) -> dict:
    """Rebuild frozen reference fits, original bank fits and native classification predictions."""
    prototypes = support_prototypes(data["selection"], data["support_features"])
    record, regional, banks = affinity_views(data["cache"], prototypes)
    errors = [
        float(np.max(np.abs(record - data["views"]["ntt_affinity"]))),
        float(np.max(np.abs(regional - data["views"]["ntt_regions"]))),
    ]
    if max(errors) > 1e-12:
        raise ValueError("Original NTT coordinates failed replay")
    checks = {}
    for view, config in data["cluster"]["selected"].items():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Predecessor replay exceeded budget")
        _, values = fit_view(data["views"][view], data["fitting"], view)
        model, converged = fit_model(
            values[data["fitting"]], config["method"], config["covariance"], config["clusters"], SEED
        )
        actual = model.predict(values)
        reference = data["assignments"][config["key"] + "|all"]
        ari = float(adjusted_rand_score(actual, reference))
        error = abs(silhouette(values[data["anchors"]], actual[data["anchors"]]) - config["silhouette"])
        if ari != 1 or error > 1e-10 or converged != config["converged"][0]:
            raise ValueError(f"Selected predecessor fit failed replay: {view}")
        checks[view] = {"ari": ari, "silhouette_error": error}
    for i, values in enumerate(banks):
        state, labels = single_reference_fit(
            values, data["fitting"], data["anchors"], data["cluster"]["selected"]["ntt_regions"], deadline
        )
        if (
            adjusted_rand_score(labels, data["assignments"]["individual_banks"][i]) != 1
            or state != data["cluster"]["individual_bank_checks"][i]
        ):
            raise ValueError("Original individual bank failed replay")
    saved = data["ntt_predictions"]
    maximum, correct = 0.0, 0
    for row in saved:
        bank = SEEDS.index(row["seed"])
        vector = data["support_features"][str(row["ecg_id"])]["native_qrs"]
        scores = np.clip(np.asarray(vector) @ prototypes[bank].T, -1, 1)
        chosen = int(np.argmax(scores))
        maximum = max(maximum, float(np.max(np.abs(scores - row["scores"]["native_qrs"]))))
        if chosen != row["chosen"]["native_qrs"]:
            raise ValueError("Native predecessor classification failed replay")
        correct += int(chosen == row["truth"])
    if len(saved) != 180 or maximum > 1e-12 or correct != 165:
        raise ValueError("Native classification denominator or score differs")
    return {
        "selected_fits": checks,
        "original_banks_exact": True,
        "maximum_coordinate_error": max(errors),
        "native_qrs_predictions": len(saved),
        "native_qrs_correct": correct,
        "native_qrs_maximum_score_error": maximum,
    }
