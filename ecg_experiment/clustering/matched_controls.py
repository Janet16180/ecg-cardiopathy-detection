"""Fitting-only native and reference-conditioned dimension, pooling and scale controls."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache import verify_hashes
from ecg_experiment.clustering.native_cache_v2 import frozen_sources, runtime_fingerprint, verified_cohort
from ecg_experiment.clustering.reference_geometry import (
    COVARIANCES,
    SEED,
    affinity_views,
    candidate,
    check_budget,
    fit_model,
    fit_view,
    selection_key,
    single_reference_fit,
    support_prototypes,
)
from ecg_experiment.clustering01 import associations
from ecg_experiment.clustering02 import anchor_patient, balanced, permitted_metadata, silhouette
from ecg_experiment.eda.ptbxl import diagnostic_classes, superclass_flags
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import from_stored, to_stored

CLUSTER03_SHA = "64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9"
CACHE_SHA = "6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3"
BASES = ("native_region9", "ntt_region9", "native_patient3", "ntt_record3", "native_patient9")
VIEWS = tuple(f"{base}_{scale}" for base in BASES for scale in ("raw", "rms"))
NEW_SOURCES = (
    "docs/research/clustering/cluster05-matched-controls/protocol.md",
    "ecg_experiment/clustering/matched_controls.py",
    "tests/clustering/test_matched_controls.py",
    "scripts/experiments/clustering/run_matched_controls.py",
)


def bind_sources(previous: dict) -> dict[str, str]:
    """Require receipt-bound predecessor bytes and committed successor sources."""
    hashes = frozen_sources()
    for name, expected in previous["source_sha256"].items():
        path = from_stored(name)
        if sha256_file(path) != expected:
            raise ValueError(f"Predecessor source differs: {name}")
        hashes[name] = expected
    for name in NEW_SOURCES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted matched-control source: {name}")
        hashes[name] = digest
    return hashes


def verified_result(path: Path, expected: str, inputs: dict) -> dict:
    """Validate receipt and all bound artifacts before reading their payloads."""
    if sha256_file(path) != expected:
        raise ValueError(f"Predecessor receipt differs: {to_stored(path)}")
    result = json.loads(path.read_text())
    status = json.loads(path.with_name("status.json").read_text())
    if result["status"] != "completed" or status.get("status") != "completed":
        raise ValueError("Predecessor is incomplete")
    if status.get("result_sha256") != expected:
        raise ValueError("Predecessor completion binding differs")
    for key in ("input_sha256", "artifact_sha256"):
        verify_hashes(result[key], inputs)
    inputs[to_stored(path)] = expected
    return result


def load_verified() -> dict:
    """Admit exact role-checked cohort, cache, support banks and saved development splits."""
    old_dir = ROOT / "outputs/cluster03_native_reference"
    cache_dir = ROOT / "outputs/cluster03_native_cache_v2"
    inputs: dict[str, str] = {}
    previous = verified_result(old_dir / "result.json", CLUSTER03_SHA, inputs)
    sources = bind_sources(previous)
    cache_receipt = verified_result(cache_dir / "result.json", CACHE_SHA, inputs)
    if cache_receipt["source_sha256"] != frozen_sources():
        raise ValueError("Cache source identity differs")
    if cache_receipt["runtime"] != runtime_fingerprint():
        raise ValueError("Cache numerical runtime differs")
    if not cache_receipt["retention_gate_passed"] or cache_receipt["accepted_patients"] != 1993:
        raise ValueError("Cache retention differs")
    ids, physical, nuisance, verified_inputs = verified_cohort()
    inputs.update(verified_inputs)
    with np.load(cache_dir / "native_features_local.npz", allow_pickle=False) as saved:
        cache = {key: saved[key] for key in saved.files}
    if not np.array_equal(ids.ecg_id, cache["ecg_ids"]) or not np.array_equal(
        ids.patient_id, cache["patient_ids"]
    ):
        raise ValueError("Role-verified cohort and cached rows differ")
    original_ids = pd.read_csv(old_dir / "retained_identities_local.csv")
    if not ids.equals(original_ids):
        raise ValueError("Original retained identities differ")
    splits = json.loads((old_dir / "patient_splits_local.json").read_text())
    anchors, samples, sample_hashes = verify_splits(ids, cache, splits, previous)
    base = ROOT / "outputs/ntt06_native_encoder"
    selection = json.loads((base / "selection_local.json").read_text())
    features = json.loads((base / "feature_vectors_local.json").read_text())
    prototypes = support_prototypes(selection, features)
    affinity, regional, banks = affinity_views(cache, prototypes)
    views = {
        "native_qrs": cache["qrs"],
        "native_global": cache["global_pair"],
        "ntt_affinity": affinity,
        "ntt_regions": regional,
        "physical": physical.to_numpy(),
    }
    with np.load(old_dir / "representations_local.npz", allow_pickle=False) as saved:
        errors = {key: float(np.max(np.abs(views[key] - saved[key]))) for key in views}
    if max(errors.values()) > 1e-12:
        raise ValueError("Reconstructed raw predecessor representation differs")
    with np.load(old_dir / "assignments_local.npz", allow_pickle=False) as saved:
        assignments = {key: saved[key] for key in saved.files}
    return {
        "previous": previous,
        "sources": sources,
        "inputs": inputs,
        "cache": cache,
        "ids": ids,
        "nuisance": nuisance,
        "anchors": anchors,
        "samples": samples,
        "selection": selection,
        "features": features,
        "prototypes": prototypes,
        "ntt_record3": affinity,
        "ntt_region9": regional,
        "banks": banks,
        "original_views": views,
        "original_assignments": assignments,
        "raw_replay_errors": errors,
        "sample_hashes": sample_hashes,
    }


def verify_splits(ids: pd.DataFrame, cache: dict, splits: dict, previous: dict) -> tuple:
    """Require the saved samples to match identity-only deterministic patient partitions."""
    if splits["ecg_ids"] != cache["ecg_ids"].tolist():
        raise ValueError("Saved split row order differs")
    anchors = np.asarray(splits["anchors"], dtype=int)
    samples = [np.asarray(rows, dtype=int) for rows in splits["samples"]]
    is_anchor = ids.patient_id.map(anchor_patient).to_numpy()
    fitting = np.flatnonzero(~is_anchor)
    expected_samples = [fitting] + [
        np.sort(np.random.default_rng(SEED + 1 + i).choice(fitting, int(0.8 * len(fitting)), replace=False))
        for i in range(8)
    ]
    if not np.array_equal(anchors, np.flatnonzero(is_anchor)) or len(anchors) != 411 or len(fitting) != 1582:
        raise ValueError("Original anchor identities differ")
    if len(samples) != 9 or any(
        not np.array_equal(a, b) for a, b in zip(samples, expected_samples, strict=True)
    ):
        raise ValueError("Original patient subsamples differ")
    if any(np.intersect1d(rows, anchors).size for rows in samples):
        raise ValueError("Validation anchors entered fitting")
    sample_hashes = [
        hashlib.sha256(ids.patient_id.to_numpy()[rows].astype("<i8").tobytes()).hexdigest()
        for rows in samples
    ]
    if sample_hashes != previous["patient_sample_hashes"]:
        raise ValueError("Patient sample hashes differ")
    return anchors, samples, sample_hashes


def replay_predecessor(data: dict, deadline: float) -> dict:
    """Reproduce the five selected CLUSTER-03 fits and all original bank partitions."""
    checks = {}
    fitting, anchors = data["samples"][0], data["anchors"]
    for view, config in data["previous"]["selected"].items():
        check_budget(deadline)
        _, coordinates = fit_view(data["original_views"][view], fitting, view)
        model, converged = fit_model(
            coordinates[fitting], config["method"], config["covariance"], config["clusters"], SEED
        )
        labels = model.predict(coordinates)
        saved = data["original_assignments"][f"{config['key']}|all"]
        ari = float(adjusted_rand_score(labels, saved))
        anchor_ari = float(adjusted_rand_score(labels[anchors], saved[anchors]))
        separation = silhouette(coordinates[anchors], labels[anchors])
        error = abs(separation - config["silhouette"])
        if not converged or ari != 1 or anchor_ari != 1 or error > 1e-10:
            raise ValueError(f"Selected predecessor fit failed replay: {view}")
        checks[view] = {
            "all_assignment_ari": ari,
            "anchor_assignment_ari": anchor_ari,
            "silhouette_error": error,
        }
    labels, bank_checks = [], []
    config = data["previous"]["selected"]["ntt_regions"]
    for i, values in enumerate(data["banks"]):
        status, prediction = single_reference_fit(values, fitting, anchors, config, deadline)
        saved = data["original_assignments"]["individual_banks"][i]
        ari = float(adjusted_rand_score(prediction, saved))
        if ari != 1 or status != data["previous"]["individual_bank_checks"][i]:
            raise ValueError("Original single-bank fit failed replay")
        labels.append(prediction)
        bank_checks.append({"assignment_ari": ari, **status})
    bank_ari = {
        f"{i}:{j}": float(adjusted_rand_score(labels[i], labels[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    }
    original = data["previous"]["decision"]["individual_bank_pairwise_ari"]
    if any(abs(value - original[key]) > 1e-12 for key, value in bank_ari.items()):
        raise ValueError("Original bank agreement failed replay")
    return {
        "raw_representation_maximum_errors": data["raw_replay_errors"],
        "selected_fits": checks,
        "individual_banks": bank_checks,
        "bank_pairwise_ari": bank_ari,
    }


def orient_components(components: np.ndarray) -> np.ndarray:
    """Orient each PCA axis by its first largest-absolute loading before nonlinear pooling."""
    result = components.copy()
    pivots = np.argmax(np.abs(result), axis=1)
    signs = np.sign(result[np.arange(len(result)), pivots])
    result *= np.where(signs == 0, 1, signs)[:, None]
    return result


def pool_regions(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Pool each patient's projected regions with mean, maximum and population deviation."""
    if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 3):
        raise ValueError("Malformed patient region boundaries")
    return np.stack(
        [
            np.concatenate(
                (values[a:b].mean(axis=0), values[a:b].max(axis=0), values[a:b].std(axis=0, ddof=0))
            )
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    )


def region_variance(cache: dict, fitting: np.ndarray, components: np.ndarray) -> float:
    """Measure captured fitting-region variance with equal total weight per fitting patient."""
    offsets, regions = cache["region_offsets"], cache["region_vectors"]
    patient_means = np.stack([regions[offsets[i] : offsets[i + 1]].mean(axis=0) for i in fitting])
    center = patient_means.mean(axis=0)
    total, captured = 0.0, 0.0
    for i in fitting:
        centered = regions[offsets[i] : offsets[i + 1]] - center
        total += float(np.mean(np.sum(centered**2, axis=1)))
        captured += float(np.mean(np.sum((centered @ components.T) ** 2, axis=1)))
    if total <= 0:
        raise ValueError("Degenerate fitting-region variance")
    return captured / total


def center_scale(values: np.ndarray, fitting: np.ndarray, rms: bool) -> tuple[dict, np.ndarray]:
    """Center and optionally scale all axes by one fitting-only RMS coordinate deviation."""
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite matched representation")
    center = values[fitting].mean(axis=0)
    centered = values - center
    scalar = float(np.sqrt(np.mean(centered[fitting] ** 2)))
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError("Degenerate fitting RMS")
    divisor = scalar if rms else 1.0
    return {
        "dimensions": int(values.shape[1]),
        "fitting_rms": scalar,
        "divisor": divisor,
        "center": center.tolist(),
    }, centered / divisor


def matched_transforms(data: dict, deadline: float) -> dict:
    """Refit every PCA, patient center and scalar independently on the frozen samples."""
    result = {view: [] for view in VIEWS}
    qrs, regions, offsets = (data["cache"][key] for key in ("qrs", "region_vectors", "region_offsets"))
    for fitting in data["samples"]:
        check_budget(deadline)
        pca = PCA(n_components=9, svd_solver="full").fit(qrs[fitting])
        components = orient_components(pca.components_)
        patient = (qrs - pca.mean_) @ components.T
        region = (regions - pca.mean_) @ components[:3].T
        raw = {
            "native_region9": pool_regions(region, offsets),
            "native_patient3": patient[:, :3],
            "native_patient9": patient,
            "ntt_record3": data["ntt_record3"],
            "ntt_region9": data["ntt_region9"],
        }
        captured = region_variance(data["cache"], fitting, components[:3])
        for base in BASES:
            for scale in ("raw", "rms"):
                state, coordinates = center_scale(raw[base], fitting, scale == "rms")
                if base.startswith("native"):
                    dims = 9 if base == "native_patient9" else 3
                    state["patient_pca_explained_variance"] = float(
                        pca.explained_variance_ratio_[:dims].sum()
                    )
                    state["pca_components"] = components[:dims].tolist()
                    state["pca_center"] = pca.mean_.tolist()
                if base == "native_region9":
                    state["equal_patient_fitting_region_captured_variance"] = captured
                result[f"{base}_{scale}"].append({**state, "coordinates": coordinates})
    return result


def run_grid(
    transforms: dict, samples: list, anchors: np.ndarray, output: Path, deadline: float
) -> tuple[list, dict]:
    """Fit the complete predeclared grid with the frozen predecessor candidate function."""
    candidates, assignments = [], {}
    for view in VIEWS:
        for method, covariance in (("kmeans", "none"), *(("gmm", cov) for cov in COVARIANCES)):
            for clusters in (2, 3, 4, 8):
                row, labels = candidate(
                    view, method, covariance, clusters, transforms[view], samples, anchors, deadline
                )
                candidates.append(row)
                assignments[row["key"]] = labels
                write_json_atomic(output / "candidate_progress.json", candidates)
        print(f"{view}: 16 configurations completed", flush=True)
    return candidates, assignments


def decision(selected: dict, assignments: dict) -> dict:
    """Freeze matched geometric concordance without allowing secondary comparisons to rescue it."""
    pairs = {}
    for i, first in enumerate(VIEWS):
        for second in VIEWS[i + 1 :]:
            pairs[f"{first}:{second}"] = float(
                adjusted_rand_score(
                    assignments[selected[first]["key"]]["anchor"],
                    assignments[selected[second]["key"]]["anchor"],
                )
            )
    first, second = "native_region9_rms", "ntt_region9_rms"
    ari = pairs[f"{first}:{second}"]
    conditions = {
        "native_region9_rms_eligible": selected[first]["eligible"],
        "ntt_region9_rms_eligible": selected[second]["eligible"],
        "anchor_ari_at_least_0.50": ari >= 0.5,
    }
    return {
        "primary_matched_geometric_concordance_passed": all(conditions.values()),
        "conditions": conditions,
        "primary_anchor_ari": ari,
        "all_selected_pairwise_anchor_ari": pairs,
        "cluster03_negative_primary_unchanged": True,
    }


def selected_fit(
    values: np.ndarray, fitting: np.ndarray, anchors: np.ndarray, config: dict, deadline: float
) -> tuple[dict, np.ndarray]:
    """Refit the fixed scaled regional configuration to a support control."""
    check_budget(deadline)
    state, coordinates = center_scale(values, fitting, True)
    model, converged = fit_model(
        coordinates[fitting], config["method"], config["covariance"], config["clusters"], SEED
    )
    labels = model.predict(coordinates)
    return {
        "converged": converged,
        "fitting_balanced": balanced(labels[fitting], config["clusters"]),
        "anchor_balanced": balanced(labels[anchors], config["clusters"]),
        "silhouette": silhouette(coordinates[anchors], labels[anchors]),
        "iterations": int(model.n_iter_),
        "fitting_rms": state["fitting_rms"],
    }, labels[anchors]


def kmeans_invariance(assignments: dict) -> list[dict]:
    """Expose raw/scalar k-means agreement separately for every frozen fit and sample."""
    result = []
    for base in BASES:
        for k in (2, 3, 4, 8):
            raw = assignments[f"{base}_raw:kmeans:none:{k}"]
            scaled = assignments[f"{base}_rms:kmeans:none:{k}"]
            result.append(
                {
                    "base": base,
                    "clusters": k,
                    "all_ari": float(adjusted_rand_score(raw["all"], scaled["all"])),
                    "anchor_ari": float(adjusted_rand_score(raw["anchor"], scaled["anchor"])),
                    "refit_ari": [
                        float(adjusted_rand_score(a, b))
                        for a, b in zip(raw["refits"], scaled["refits"], strict=True)
                    ],
                }
            )
    return result


def interpretation(data: dict, selected: dict, assignments: dict, controls: list[np.ndarray]) -> dict:
    """Read existing annotations only after the label-free selection has been saved."""
    path = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    classes = diagnostic_classes(pd.read_csv(path.with_name("scp_statements.csv"), index_col=0))
    metadata = permitted_metadata(path, data["ids"].ecg_id, True)
    flags = (
        superclass_flags(metadata, classes)
        .drop(columns="abnormal")
        .iloc[data["anchors"]]
        .reset_index(drop=True)
    )
    nuisance = data["nuisance"].iloc[data["anchors"]].reset_index(drop=True)
    combinations = flags.astype(int).astype(str).agg("".join, axis=1).to_numpy()
    return {
        "existing_annotation_counts": flags.sum().astype(int).to_dict(),
        "selected_anchor_associations": {
            view: associations(assignments[config["key"]]["anchor"], flags, nuisance)
            for view, config in selected.items()
        },
        "support_control_annotation_ami": [
            float(adjusted_mutual_info_score(combinations, labels)) for labels in controls
        ],
        "support_control_sensitivity_not_confirmatory_p_value": True,
    }


def run(output: Path) -> dict:
    """Complete one bounded prospective matched-geometry study from verified local caches."""
    started = time.perf_counter()
    deadline = started + 900
    data = load_verified()
    replay = replay_predecessor(data, deadline)
    transforms = matched_transforms(data, deadline)
    candidates, assignments = run_grid(transforms, data["samples"], data["anchors"], output, deadline)
    selected = {
        view: max((row for row in candidates if row["view"] == view), key=selection_key) for view in VIEWS
    }
    primary = decision(selected, assignments)
    invariance = kmeans_invariance(assignments)
    fitting, anchors = data["samples"][0], data["anchors"]
    config = selected["ntt_region9_rms"]
    bank_checks, bank_labels = [], []
    for values in data["banks"]:
        check, labels = selected_fit(values, fitting, anchors, config, deadline)
        bank_checks.append(check)
        bank_labels.append(labels)
    support_checks, support_labels = [], []
    reference = assignments[config["key"]]["anchor"]
    for i in range(20):
        check_budget(deadline)
        _, values, _ = affinity_views(
            data["cache"], support_prototypes(data["selection"], data["features"], SEED + 1 + i)
        )
        check, labels = selected_fit(values, fitting, anchors, config, deadline)
        support_checks.append(
            {
                "seed": SEED + 1 + i,
                "anchor_ari_to_true_support_selected": float(adjusted_rand_score(reference, labels)),
                **check,
            }
        )
        support_labels.append(labels)
    unsupervised = {
        "selected": selected,
        "decision": primary,
        "kmeans_scalar_invariance": invariance,
        "individual_bank_checks": bank_checks,
        "individual_bank_pairwise_ari": {
            f"{i}:{j}": float(adjusted_rand_score(bank_labels[i], bank_labels[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        },
        "support_label_controls": support_checks,
    }
    selection_path = output / "unsupervised_selection.json"
    write_json_atomic(selection_path, unsupervised)
    selection_hash = sha256_file(selection_path)
    np.savez_compressed(
        output / "assignments_local.npz",
        **{f"{key}|{kind}": values for key, entry in assignments.items() for kind, values in entry.items()},
        individual_banks=np.stack(bank_labels),
        support_controls=np.stack(support_labels),
    )
    np.savez_compressed(
        output / "coordinates_local.npz",
        **{
            f"{view}|{draw}": state["coordinates"]
            for view, states in transforms.items()
            for draw, state in enumerate(states)
        },
    )
    write_json_atomic(
        output / "transforms_local.json",
        {
            view: [{key: value for key, value in state.items() if key != "coordinates"} for state in states]
            for view, states in transforms.items()
        },
    )
    data["ids"].to_csv(output / "identities_local.csv", index=False)
    write_json_atomic(
        output / "patient_splits_local.json",
        {
            "ecg_ids": data["cache"]["ecg_ids"].tolist(),
            "anchors": anchors.tolist(),
            "samples": [rows.tolist() for rows in data["samples"]],
        },
    )
    interpreted = interpretation(data, selected, assignments, support_labels)
    if sha256_file(selection_path) != selection_hash:
        raise ValueError("Selection changed during annotation interpretation")
    check_budget(deadline)
    result = {
        "study": "CLUSTER-05",
        "status": "completed",
        "source_sha256": data["sources"],
        "input_sha256": data["inputs"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": runtime_fingerprint(),
        "counts": {
            "patients": 1993,
            "fitting": 1582,
            "anchors": 411,
            "views": 10,
            "configurations": 160,
            "grid_model_fits": 1440,
            "predecessor_reference_fits": 8,
            "control_fits": 23,
        },
        "predecessor_replay": replay,
        "candidates": candidates,
        **unsupervised,
        "interpretation": interpreted,
        "patient_sample_hashes": data["sample_hashes"],
        "selection_frozen_before_target_annotation_parse": True,
        "unsupervised_selection_sha256": selection_hash,
        "seconds": time.perf_counter() - started,
        "waveform_reads": 0,
        "new_encoder_inference": False,
        "gpu_used": False,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in output.iterdir()
            if path.name not in ("status.json", "result.json", "failure.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result
