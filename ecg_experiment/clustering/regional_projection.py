"""Equal-patient regional covariance control with exact frozen predecessor replay."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ecg_experiment import ROOT
from ecg_experiment.clustering import matched_controls as matched
from ecg_experiment.clustering.native_cache_v2 import runtime_fingerprint
from ecg_experiment.clustering.reference_geometry import (
    COVARIANCES,
    SEED,
    affinity_views,
    candidate,
    check_budget,
    selection_key,
    support_prototypes,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

PREVIOUS_SHA = "cbe2d5af704e6a1ee86ba535ea03e2087bd667b33573ec676952d4e65275f885"
VIEWS = ("regional_covariance9_rms", "native_region9_rms", "ntt_region9_rms")
SOURCES = (
    "docs/research/clustering/cluster07-regional-projection/protocol.md",
    "ecg_experiment/clustering/regional_projection.py",
    "scripts/experiments/clustering/run_regional_projection.py",
    "tests/clustering/test_regional_projection.py",
)


def weighted_projection(cache: dict, fitting: np.ndarray) -> dict:
    """Fit three regional covariance axes with equal patient total weights.

    Parameters
    ----------
    cache : dict
        Region vectors and patient offsets; vectors are already unit normalized.
    fitting : numpy.ndarray
        Admitted fitting patient row indices.

    Returns
    -------
    dict
        Mean, signed components, spectrum and variance diagnostics.
    """
    offsets = np.asarray(cache["region_offsets"])
    regions = np.asarray(cache["region_vectors"], dtype=np.float64)
    if offsets[0] != 0 or offsets[-1] != len(regions) or np.any(np.diff(offsets) < 3):
        raise ValueError("Malformed region boundaries")
    if len(fitting) < 2 or len(np.unique(fitting)) != len(fitting):
        raise ValueError("Fitting patients must be distinct and nonempty")
    if np.any(fitting < 0) or np.any(fitting >= len(offsets) - 1):
        raise ValueError("Fitting patient index outside region boundaries")
    rows = [regions[offsets[i] : offsets[i + 1]] for i in fitting]
    values = np.concatenate(rows)
    weights = np.concatenate([np.full(len(row), 1.0 / (len(fitting) * len(row))) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite fitting regions")
    center = weights @ values
    centered = values - center
    weighted = centered * np.sqrt(weights[:, None])
    covariance = weighted.T @ weighted
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-14):
        raise ValueError("Invalid regional covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1]
    components = matched.orient_components(eigenvectors[:, ::-1].T[:3])
    if not np.isfinite(components).all() or not np.allclose(components @ components.T, np.eye(3), atol=1e-12):
        raise ValueError("Invalid regional covariance basis")
    total = float(np.trace(covariance))
    if total <= 0 or eigenvalues[2] <= 0:
        raise ValueError("Degenerate regional covariance")
    gap = float(eigenvalues[2] - eigenvalues[3])
    return {
        "center": center,
        "components": components,
        "eigenvalues": eigenvalues,
        "weighted_region_variance_retained": float(eigenvalues[:3].sum() / total),
        "eigenvalue3_minus4": gap,
        "relative_eigenvalue3_minus4": gap / float(eigenvalues[2]),
        "near_degenerate_boundary": bool(gap <= 1e-10 * eigenvalues[0]),
        "total_region_variance": total,
    }


def patient_variance(qrs: np.ndarray, fitting: np.ndarray, components: np.ndarray) -> float:
    """Measure fitting patient-vector variance captured by fixed orthonormal axes.

    Parameters
    ----------
    qrs : numpy.ndarray
        One native vector per patient.
    fitting : numpy.ndarray
        Fitting patient indices.
    components : numpy.ndarray
        Three orthonormal rows.

    Returns
    -------
    float
        Captured variance fraction centered on fitting patient vectors.
    """
    values = qrs[fitting] - qrs[fitting].mean(axis=0)
    total = float(np.sum(values**2))
    if total <= 0:
        raise ValueError("Degenerate patient-vector variance")
    return float(np.sum((values @ components.T) ** 2) / total)


def _compare(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Replay keys differ: {name}")
        for key, value in expected.items():
            _compare(actual[key], value, f"{name}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Replay list differs: {name}")
        for i, (first, second) in enumerate(zip(actual, expected, strict=True)):
            _compare(first, second, f"{name}/{i}")
        return
    if isinstance(expected, float):
        if actual is None or not np.isfinite(actual) or abs(actual - expected) > 1e-12:
            raise ValueError(f"Replay numeric value differs: {name}")
        return
    if actual != expected:
        raise ValueError(f"Replay value differs: {name}")


def _bind_sources(data: dict) -> dict:
    for name in SOURCES:
        content = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        actual = sha256_file(ROOT / name)
        if hashlib.sha256(content).hexdigest() != actual:
            raise ValueError(f"Uncommitted source: {name}")
        data["sources"][name] = actual
    return data["sources"]


def _replay_controls(data: dict, previous: dict, assignments: dict, original: Path, deadline: float) -> list:
    with np.load(original / "assignments_local.npz", allow_pickle=False) as saved:
        for key, entry in assignments.items():
            for kind, values in entry.items():
                if not np.array_equal(values, saved[f"{key}|{kind}"]):
                    raise ValueError("Predecessor grid labels differ")
        bank_checks, banks = [], []
        config = previous["selected"]["ntt_region9_rms"]
        fitting, anchors = data["samples"][0], data["anchors"]
        for i, values in enumerate(data["banks"]):
            check, labels = matched.selected_fit(values, fitting, anchors, config, deadline)
            if not np.array_equal(labels, saved["individual_banks"][i]):
                raise ValueError("Predecessor bank labels differ")
            bank_checks.append(check)
            banks.append(labels)
        _compare(bank_checks, previous["individual_bank_checks"], "banks")
        bank_aris = {
            f"{i}:{j}": float(adjusted_rand_score(banks[i], banks[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        }
        _compare(bank_aris, previous["individual_bank_pairwise_ari"], "bank ARIs")
        control_checks, controls = [], []
        for i in range(20):
            check_budget(deadline)
            _, values, _ = affinity_views(
                data["cache"], support_prototypes(data["selection"], data["features"], SEED + 1 + i)
            )
            check, labels = matched.selected_fit(values, fitting, anchors, config, deadline)
            if not np.array_equal(labels, saved["support_controls"][i]):
                raise ValueError("Predecessor support labels differ")
            control_checks.append(
                {
                    "seed": SEED + 1 + i,
                    "anchor_ari_to_true_support_selected": float(
                        adjusted_rand_score(assignments[config["key"]]["anchor"], labels)
                    ),
                    **check,
                }
            )
            controls.append(labels)
        _compare(control_checks, previous["support_label_controls"], "controls")
    return controls


def _replay(data: dict, previous: dict, output: Path, deadline: float) -> tuple:
    original = ROOT / "outputs/cluster05_matched_controls"
    reference = matched.replay_predecessor(data, deadline)
    _compare(reference, previous["predecessor_replay"], "cluster03 replay")
    transforms = matched.matched_transforms(data, deadline)
    states = json.loads((original / "transforms_local.json").read_text())
    maximum_error = 0.0
    with np.load(original / "coordinates_local.npz", allow_pickle=False) as saved:
        for view, draws in transforms.items():
            for i, state in enumerate(draws):
                error = float(np.max(np.abs(state["coordinates"] - saved[f"{view}|{i}"])))
                maximum_error = max(error, maximum_error)
                _compare({k: v for k, v in state.items() if k != "coordinates"}, states[view][i], view)
    if maximum_error > 1e-12:
        raise ValueError("Predecessor transform coordinates differ")
    replay_dir = output / "predecessor_replay"
    replay_dir.mkdir()
    candidates, assignments = matched.run_grid(
        transforms, data["samples"], data["anchors"], replay_dir, deadline
    )
    _compare(candidates, previous["candidates"], "candidates")
    selected = {
        view: max((c for c in candidates if c["view"] == view), key=selection_key) for view in matched.VIEWS
    }
    _compare(selected, previous["selected"], "selected")
    _compare(matched.decision(selected, assignments), previous["decision"], "decision")
    _compare(
        matched.kmeans_invariance(assignments), previous["kmeans_scalar_invariance"], "scalar invariance"
    )
    controls = _replay_controls(data, previous, assignments, original, deadline)
    return (
        transforms,
        assignments,
        controls,
        {
            "cluster03_reference": reference,
            "cluster05_coordinate_maximum_error": maximum_error,
            "all_1440_grid_fits_exact": True,
            "all_23_support_fits_exact": True,
            "all_candidates_selections_and_decision_exact": True,
        },
    )


def _transforms(data: dict, previous: dict, deadline: float) -> tuple:
    result = {view: [] for view in VIEWS}
    diagnostics = []
    cache = data["cache"]
    for draw, fitting in enumerate(data["samples"]):
        check_budget(deadline)
        started = time.perf_counter()
        basis = weighted_projection(cache, fitting)
        projected = (cache["region_vectors"] - basis["center"]) @ basis["components"].T
        raw = matched.pool_regions(projected, cache["region_offsets"])
        state, coordinates = matched.center_scale(raw, fitting, True)
        result[VIEWS[0]].append({**state, "coordinates": coordinates})
        for view in VIEWS[1:]:
            result[view].append(previous[view][draw])
        patient_components = np.asarray(previous["native_region9_rms"][draw]["pca_components"])
        diagnostics.append(
            {
                "draw": draw,
                "fitting_patients": int(len(fitting)),
                "regional_basis_region_variance_retained": basis["weighted_region_variance_retained"],
                "patient_basis_region_variance_retained": previous["native_region9_rms"][draw][
                    "equal_patient_fitting_region_captured_variance"
                ],
                "regional_basis_patient_variance_retained": patient_variance(
                    cache["qrs"], fitting, basis["components"]
                ),
                "patient_basis_patient_variance_retained": patient_variance(
                    cache["qrs"], fitting, patient_components
                ),
                "eigenvalues": basis["eigenvalues"].tolist(),
                "eigenvalue3_minus4": basis["eigenvalue3_minus4"],
                "relative_eigenvalue3_minus4": basis["relative_eigenvalue3_minus4"],
                "near_degenerate_boundary": basis["near_degenerate_boundary"],
                "seconds": time.perf_counter() - started,
                "basis_center_local": basis["center"].tolist(),
                "basis_components_local": basis["components"].tolist(),
            }
        )
    return result, diagnostics


def decision(selected: dict, assignments: dict) -> dict:
    """Apply the frozen primary gate without secondary rescue.

    Parameters
    ----------
    selected : dict
        Selected candidates including eligibility.
    assignments : dict
        Anchor assignments keyed by candidate.

    Returns
    -------
    dict
        All selected pairwise ARIs and the primary decision conditions.
    """
    pairs = {
        f"{first}:{second}": float(
            adjusted_rand_score(
                assignments[selected[first]["key"]]["anchor"], assignments[selected[second]["key"]]["anchor"]
            )
        )
        for i, first in enumerate(VIEWS)
        for second in VIEWS[i + 1 :]
    }
    ari = pairs[f"{VIEWS[0]}:{VIEWS[2]}"]
    conditions = {
        "regional_covariance9_rms_eligible": selected[VIEWS[0]]["eligible"],
        "ntt_region9_rms_eligible": selected[VIEWS[2]]["eligible"],
        "anchor_ari_at_least_0.50": ari >= 0.5,
    }
    return {
        "primary_regional_concordance_passed": all(conditions.values()),
        "conditions": conditions,
        "primary_anchor_ari": ari,
        "all_selected_pairwise_anchor_ari": pairs,
        "cluster03_and_cluster05_negative_primaries_unchanged": True,
    }


def run(output: Path) -> dict:
    """Execute the source-bound regional covariance study within 1,200 seconds.

    Parameters
    ----------
    output : pathlib.Path
        New local execution directory.

    Returns
    -------
    dict
        Completed scientific and provenance receipt.
    """
    started = time.perf_counter()
    deadline = started + 1200
    data = matched.load_verified()
    previous = matched.verified_result(
        ROOT / "outputs/cluster05_matched_controls/result.json", PREVIOUS_SHA, data["inputs"]
    )
    if previous["runtime"] != runtime_fingerprint():
        raise ValueError("Predecessor numerical runtime differs")
    for name, expected in previous["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Predecessor source differs: {name}")
        data["sources"][name] = expected
    _bind_sources(data)
    replay_started = time.perf_counter()
    previous_transforms, previous_assignments, previous_controls, replay = _replay(
        data, previous, output, deadline
    )
    replay["seconds"] = time.perf_counter() - replay_started
    transforms, diagnostics = _transforms(data, previous_transforms, deadline)
    candidates, assignments = [], {}
    for view in VIEWS:
        for method, covariance in (("kmeans", "none"), *(("gmm", cov) for cov in COVARIANCES)):
            for clusters in (2, 3, 4, 8):
                row, labels = candidate(
                    view,
                    method,
                    covariance,
                    clusters,
                    transforms[view],
                    data["samples"],
                    data["anchors"],
                    deadline,
                )
                candidates.append(row)
                assignments[row["key"]] = labels
                write_json_atomic(output / "candidate_progress.json", candidates)
        print(f"{view}: 16 configurations completed", flush=True)
    selected = {
        view: max((row for row in candidates if row["view"] == view), key=selection_key) for view in VIEWS
    }
    unsupervised = {"selected": selected, "decision": decision(selected, assignments)}
    selection_path = output / "unsupervised_selection.json"
    write_json_atomic(selection_path, unsupervised)
    selection_hash = sha256_file(selection_path)
    interpreted = matched.interpretation(data, selected, assignments, [])
    _compare(
        matched.interpretation(data, previous["selected"], previous_assignments, previous_controls),
        previous["interpretation"],
        "annotation replay",
    )
    replay["post_selection_annotation_replay_exact"] = True
    if sha256_file(selection_path) != selection_hash:
        raise ValueError("Selection changed after annotation parse")
    np.savez_compressed(
        output / "assignments_local.npz",
        **{f"{key}|{kind}": values for key, entry in assignments.items() for kind, values in entry.items()},
    )
    np.savez_compressed(
        output / "coordinates_local.npz",
        **{
            f"{view}|{i}": state["coordinates"]
            for view, states in transforms.items()
            for i, state in enumerate(states)
        },
    )
    write_json_atomic(
        output / "transforms_local.json",
        {
            view: [{k: v for k, v in state.items() if k != "coordinates"} for state in states]
            for view, states in transforms.items()
        },
    )
    write_json_atomic(output / "projection_diagnostics_local.json", diagnostics)
    data["ids"].to_csv(output / "identities_local.csv", index=False)
    write_json_atomic(
        output / "patient_splits_local.json",
        {
            "ecg_ids": data["cache"]["ecg_ids"].tolist(),
            "anchors": data["anchors"].tolist(),
            "samples": [rows.tolist() for rows in data["samples"]],
        },
    )
    check_budget(deadline)
    result = {
        "study": "CLUSTER-07",
        "status": "completed",
        "source_sha256": data["sources"],
        "input_sha256": data["inputs"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": runtime_fingerprint(),
        "counts": {
            "patients": 1993,
            "fitting": 1582,
            "anchors": 411,
            "views": 3,
            "configurations": 48,
            "new_grid_fits": 432,
            "predecessor_grid_fits": 1440,
            "predecessor_support_fits": 23,
            "cluster03_reference_fits": 8,
        },
        "predecessor_replay": replay,
        "candidates": candidates,
        **unsupervised,
        "interpretation": interpreted,
        "projection_diagnostics": [
            {k: v for k, v in row.items() if not k.endswith("_local")} for row in diagnostics
        ],
        "patient_sample_hashes": data["sample_hashes"],
        "selection_frozen_before_target_annotation_parse": True,
        "unsupervised_selection_sha256": selection_hash,
        "seconds": time.perf_counter() - started,
        "waveform_reads": 0,
        "new_encoder_inference": False,
        "gpu_used": False,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in output.rglob("*")
            if path.is_file() and path.name not in ("status.json", "result.json", "failure.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result
