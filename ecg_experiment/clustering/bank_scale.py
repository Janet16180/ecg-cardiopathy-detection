"""Fixed support-bank averaging and coordinate-equivalent covariance controls."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_inputs import (
    cached_inputs,
    committed_sources,
    read_bound,
    replay_predecessor,
)
from ecg_experiment.clustering.bank_robustness import (
    align_fitting,
    bank_affinities,
    bank_fit,
    bank_prototypes,
    distribution,
    sensitivity,
    support_draws,
)
from ecg_experiment.clustering.matched_controls import center_scale
from ecg_experiment.clustering.native_cache import runtime_fingerprint
from ecg_experiment.clustering.reference_geometry import SEED, check_budget
from ecg_experiment.clustering02 import balanced, silhouette
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

PREVIOUS = {
    "cluster04_bank_robustness": "c672be6b2fc9fe6d8c14d974179f04345381b068349df611d1656723e0427baa",
    "cluster05_matched_controls": "cbe2d5af704e6a1ee86ba535ea03e2087bd667b33573ec676952d4e65275f885",
}
CONFIGS = {
    "regional_tied2": ("regional", "tied", 2),
    "regional_diag2": ("regional", "diag", 2),
    "record_full3": ("record", "full", 3),
}
SCALES = ("raw", "rms", "equivalent")
SOURCES = (
    "ecg_experiment/clustering/bank_scale.py",
    "scripts/experiments/clustering/run_bank_scale.py",
    "tests/clustering/test_bank_scale.py",
    "docs/research/clustering/cluster06-bank-scale/protocol.md",
)


def fit_condition(
    values: np.ndarray,
    fitting: np.ndarray,
    anchors: np.ndarray,
    covariance: str,
    clusters: int,
    scale: str,
    deadline: float,
    seed: int = SEED,
) -> tuple:
    """Fit one frozen configuration with fitting-only scalar and covariance adjustment.

    Parameters
    ----------
    values, fitting, anchors : np.ndarray
        Raw coordinates and disjoint patient row indices.
    covariance, scale : str
        Frozen covariance geometry and coordinate condition.
    clusters : int
        Fixed component count.
    deadline : float
        Monotonic execution deadline.
    seed : int
        Frozen study seed, or the original sample-specific predecessor replay seed.

    Returns
    -------
    tuple
        Diagnostics, labels and coordinate-backtransformed model arrays.
    """
    check_budget(deadline)
    state, coordinates = center_scale(values, fitting, scale != "raw")
    divisor = state["fitting_rms"] if scale != "raw" else 1.0
    floor = 1e-3 / divisor**2 if scale == "equivalent" else 1e-3
    model = GaussianMixture(
        n_components=clusters,
        covariance_type=covariance,
        n_init=3,
        reg_covar=floor,
        max_iter=400,
        tol=1e-3,
        random_state=seed,
        init_params="kmeans",
    ).fit(coordinates[fitting])
    labels = model.predict(coordinates)
    if labels.shape != (len(values),) or not np.isfinite(labels).all():
        raise ValueError("Invalid finite fit labels")
    fit_ok, anchor_ok = balanced(labels[fitting], clusters), balanced(labels[anchors], clusters)
    diagnostics = {
        "converged": bool(model.converged_),
        "fit_balanced": bool(fit_ok),
        "anchor_balanced": bool(anchor_ok),
        "admissible": bool(model.converged_ and fit_ok and anchor_ok),
        "iterations": int(model.n_iter_),
        "fitting_rms": state["fitting_rms"],
        "divisor": float(divisor),
        "reg_covar": float(floor),
        "anchor_sizes": np.bincount(labels[anchors], minlength=clusters).tolist(),
        "fitting_sizes": np.bincount(labels[fitting], minlength=clusters).tolist(),
        "silhouette": silhouette(coordinates[anchors], labels[anchors]),
        "adjusted_lower_bound": float(model.lower_bound_ - values.shape[1] * np.log(divisor)),
    }
    parameters = {
        "means": model.means_ * divisor,
        "covariances": model.covariances_ * divisor**2,
        "weights": model.weights_,
        "responsibilities": model.predict_proba(coordinates),
    }
    if any(not np.isfinite(array).all() for array in parameters.values()):
        raise ValueError("Nonfinite model parameters")
    return diagnostics, labels, parameters


def equivalence(
    raw: tuple, scaled: tuple, fitting: np.ndarray, anchors: np.ndarray, covariance: str, clusters: int
) -> dict:
    """Compare equivalent coordinate fits after fitting-only permutation alignment.

    Parameters
    ----------
    raw, scaled : tuple
        Diagnostics, complete labels and model arrays for the matched fits.
    fitting, anchors : np.ndarray
        Frozen fitting and validation patient rows.
    covariance : str
        Fixed covariance structure.
    clusters : int
        Fixed component count.

    Returns
    -------
    dict
        Prespecified numerical errors, ARIs and pass conditions.
    """
    raw_check, raw_labels, raw_params = raw
    check, labels, params = scaled
    _, mapping = align_fitting(labels, raw_labels, fitting, clusters)
    order = np.argsort(mapping)
    errors = {}
    for key in ("means", "weights", "responsibilities", "covariances"):
        actual = params[key]
        if key == "responsibilities":
            actual = actual[:, order]
        elif key != "covariances" or covariance != "tied":
            actual = actual[order]
        errors[key] = float(np.max(np.abs(raw_params[key] - actual)))
    errors["lower_bound"] = abs(raw_check["adjusted_lower_bound"] - check["adjusted_lower_bound"])
    all_ari = float(adjusted_rand_score(raw_labels, labels))
    anchor_ari = float(adjusted_rand_score(raw_labels[anchors], labels[anchors]))
    conditions = {
        "all_ari_one": all_ari == 1,
        "anchor_ari_one": anchor_ari == 1,
        "convergence_equal": raw_check["converged"] == check["converged"],
        **{key + "_within_1e-6": value <= 1e-6 for key, value in errors.items()},
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "maximum_errors": errors,
        "all_ari": all_ari,
        "anchor_ari": anchor_ari,
    }


def group_agreement(
    assignments: np.ndarray, reference: np.ndarray, fitting: np.ndarray, anchors: np.ndarray, clusters: int
) -> dict:
    """Describe every frozen reference group without selecting a group by annotations.

    Parameters
    ----------
    assignments, reference : np.ndarray
        Bank partitions and the unchanged reference partition.
    fitting, anchors : np.ndarray
        Frozen fitting and validation patient rows.
    clusters : int
        Fixed component count.

    Returns
    -------
    dict
        Per-reference-group agreement, retaining empty groups explicitly.
    """
    aligned = np.stack([align_fitting(row, reference, fitting, clusters)[0] for row in assignments])
    agreement = np.mean(aligned[:, anchors] == reference[anchors], axis=0)
    return {
        str(k): {
            "anchors": int(np.sum(reference[anchors] == k)),
            "agreement": (
                distribution(agreement[reference[anchors] == k]) if np.any(reference[anchors] == k) else None
            ),
            "fraction_agreement_at_least_0.8": (
                float(np.mean(agreement[reference[anchors] == k] >= 0.8))
                if np.any(reference[anchors] == k)
                else None
            ),
        }
        for k in range(clusters)
    }


def averaging_decision(single: dict, ensemble: dict) -> dict:
    """Apply the frozen finite-design meaningful averaging-benefit criterion.

    Parameters
    ----------
    single, ensemble : dict
        Complete-bank sensitivity summaries with all finite fits retained.

    Returns
    -------
    dict
        Signed effect and all prospective admission conditions.
    """
    delta = ensemble["pairwise_anchor_ari"]["p10"] - single["pairwise_anchor_ari"]["p10"]
    conditions = {
        "p10_improvement_at_least_0.10": delta >= 0.10,
        "at_least_45_admissible_single_banks": single["admissible_banks"] >= 45,
        "at_least_9_admissible_ensembles": ensemble["admissible_banks"] >= 9,
    }
    return {
        "meaningful_conditional_averaging_benefit": all(conditions.values()),
        "p10_ari_difference": float(delta),
        "conditions": conditions,
    }


def scale_contrasts(assignments: dict, anchors: np.ndarray) -> dict:
    """Describe fixed-floor raw/RMS agreement without selecting a configuration.

    Parameters
    ----------
    assignments : dict
        Complete fixed-condition partition arrays.
    anchors : np.ndarray
        Frozen development-validation patient indices.

    Returns
    -------
    dict
        Reference and all-bank anchor ARIs for every configuration and bank count.
    """
    result = {}
    for name in CONFIGS:
        raw = assignments[name + ":reference:raw"]
        rms = assignments[name + ":reference:rms"]
        result[name + ":reference"] = {
            "anchor_ari": float(adjusted_rand_score(raw[anchors], rms[anchors])),
            "all_ari": float(adjusted_rand_score(raw, rms)),
        }
        for kind in ("single", "ensemble"):
            raw = assignments[name + ":" + kind + ":raw"][:, anchors]
            rms = assignments[name + ":" + kind + ":rms"][:, anchors]
            result[name + ":" + kind] = distribution(
                np.array([adjusted_rand_score(first, second) for first, second in zip(raw, rms, strict=True)])
            )
    return result


def replay04(data: dict, previous: dict, views: dict, deadline: float) -> dict:
    """Reproduce every CLUSTER-04 fixed bank/ensemble fit and sensitivity summary.

    Parameters
    ----------
    data, previous, views : dict
        Verified cohort, bound predecessor result and unchanged bank coordinates.
    deadline : float
        Monotonic execution deadline.

    Returns
    -------
    dict
        Exact partition/check replay and maximum summary error.
    """
    base = ROOT / "outputs/cluster04_bank_robustness"
    with np.load(base / "assignments_local.npz", allow_pickle=False) as saved:
        reference = {key: saved[key] for key in ("regional", "record", "ensembles")}
    comparisons, errors = 0, []
    for view in ("regional", "record", "five_bank_ensembles"):
        regional = view != "record"
        matrices = views["regional" if regional else "record"]
        matrices = (
            [np.mean(matrices[i : i + 5], axis=0) for i in range(0, 50, 5)]
            if view == "five_bank_ensembles"
            else matrices
        )
        config = data["cluster"]["selected"]["ntt_regions" if regional else "ntt_affinity"]
        checks, labels = [], []
        for i, values in enumerate(matrices):
            check, predicted = bank_fit(values, data["fitting"], data["anchors"], config, deadline)
            expected_check = (
                previous["ensemble_checks"][i]
                if view == "five_bank_ensembles"
                else previous["checks"][view][i]
            )
            saved_labels = reference["ensembles" if view == "five_bank_ensembles" else view][i]
            if check != expected_check or adjusted_rand_score(predicted, saved_labels) != 1:
                raise ValueError("CLUSTER-04 fit replay differs")
            checks.append(check)
            labels.append(predicted)
            comparisons += 1
        summary, _ = sensitivity(
            np.stack(labels),
            checks,
            data["assignments"][config["key"] + "|all"],
            data["fitting"],
            data["anchors"],
            config["clusters"],
        )
        for key, value in summary.items():
            expected = previous["summaries"][view][key]
            if isinstance(value, dict):
                errors.extend(abs(value[k] - expected[k]) for k in value)
            elif isinstance(value, (float, int)):
                errors.append(abs(value - expected))
        if max(errors) > 1e-12:
            raise ValueError("CLUSTER-04 sensitivity replay differs")
    return {
        "fits": comparisons,
        "all_partition_ari_one": True,
        "checks_exact": True,
        "maximum_summary_error": float(max(errors)),
    }


def replay05(data: dict, previous: dict, deadline: float) -> dict:
    """Reproduce six fixed CLUSTER-05 NTT configurations across all nine fitting samples.

    Parameters
    ----------
    data, previous : dict
        Verified cohort and source-bound predecessor result.
    deadline : float
        Monotonic execution deadline.

    Returns
    -------
    dict
        Replay counts and coordinate/partition diagnostics.
    """
    base = ROOT / "outputs/cluster05_matched_controls"
    split = json.loads((base / "patient_splits_local.json").read_text())
    if split["ecg_ids"] != data["ids"].ecg_id.tolist() or split["anchors"] != data["anchors"].tolist():
        raise ValueError("CLUSTER-05 identities differ")
    errors, fits = [], 0
    candidates = {row["key"]: row for row in previous["candidates"]}
    with (
        np.load(base / "coordinates_local.npz", allow_pickle=False) as coordinates,
        np.load(base / "assignments_local.npz", allow_pickle=False) as saved,
    ):
        for view, covariance, clusters in CONFIGS.values():
            name = "ntt_region9" if view == "regional" else "ntt_record3"
            values = data["views"]["ntt_regions" if view == "regional" else "ntt_affinity"]
            for scale in ("raw", "rms"):
                key = f"{name}_{scale}:gmm:{covariance}:{clusters}"
                for i, indices in enumerate(split["samples"]):
                    fitting = np.asarray(indices)
                    _, actual = center_scale(values, fitting, scale == "rms")
                    error = float(np.max(np.abs(actual - coordinates[f"{name}_{scale}|{i}"])))
                    check, labels, _ = fit_condition(
                        values, fitting, data["anchors"], covariance, clusters, scale, deadline, SEED + i
                    )
                    expected = saved[key + "|all"] if i == 0 else saved[key + "|refits"][i - 1]
                    observed = labels if i == 0 else labels[data["anchors"]]
                    if adjusted_rand_score(observed, expected) != 1 or error > 1e-12:
                        raise ValueError("CLUSTER-05 coordinates or partition replay differs")
                    if i == 0 and abs(check["silhouette"] - candidates[key]["silhouette"]) > 1e-12:
                        raise ValueError("CLUSTER-05 silhouette replay differs")
                    errors.append(error)
                    fits += 1
    return {
        "fits": fits,
        "all_partition_ari_one": True,
        "maximum_coordinate_error": max(errors),
        "reference_silhouettes_within_1e-12": True,
    }


def summarize(
    labels: np.ndarray,
    checks: list,
    reference: np.ndarray,
    data: dict,
    clusters: int,
    original: np.ndarray | None,
) -> dict:
    """Summarize all finite fits and reference groups with immutable denominators.

    Parameters
    ----------
    labels, reference : np.ndarray
        All bank assignments and fixed own-condition reference.
    checks : list
        Diagnostics for every finite fit.
    data : dict
        Verified fitting and anchor indices.
    clusters : int
        Fixed component count.
    original : np.ndarray or None
        Original regional partition for the additional persistence perspective.

    Returns
    -------
    dict
        Label-invariant sensitivity and per-group agreement.
    """
    result, _ = sensitivity(labels, checks, reference, data["fitting"], data["anchors"], clusters)
    result["own_reference_groups"] = group_agreement(
        labels, reference, data["fitting"], data["anchors"], clusters
    )
    if original is not None:
        result["original_regional_reference_groups"] = group_agreement(
            labels, original, data["fitting"], data["anchors"], clusters
        )
    return result


def store_fits(key: str, rows: list, checks: dict, assignments: dict, parameters: dict) -> None:
    """Store every finite fit without filtering on convergence or occupancy.

    Parameters
    ----------
    key : str
        Fixed configuration, bank count and coordinate-condition key.
    rows : list
        Complete finite fit diagnostics, partitions and model arrays.
    checks, assignments, parameters : dict
        Mutable local artifact accumulators.

    Returns
    -------
    None
    """
    checks[key] = [row[0] for row in rows]
    assignments[key] = np.stack([row[1] for row in rows])
    for parameter in rows[0][2]:
        parameters[key + ":" + parameter] = np.stack([row[2][parameter] for row in rows])


def run_configuration(
    name: str,
    configuration: tuple,
    views: dict,
    data: dict,
    original: np.ndarray,
    deadline: float,
    output: Path,
    checks: dict,
    assignments: dict,
    parameters: dict,
    summaries: dict,
    equivalences: dict,
) -> None:
    """Fit all three coordinate conditions for one prespecified configuration.

    Parameters
    ----------
    name : str
        Frozen configuration name.
    configuration : tuple
        View, covariance structure and component count.
    views, data : dict
        Frozen bank coordinates and verified cohort inputs.
    original : np.ndarray
        Original regional reference partition.
    deadline : float
        Monotonic execution deadline.
    output : Path
        Local progress destination.
    checks, assignments, parameters, summaries, equivalences : dict
        Complete artifact accumulators, including inadmissible fits.

    Returns
    -------
    None
    """
    view, covariance, clusters = configuration
    references = {}
    for scale in SCALES:
        values = data["views"]["ntt_regions" if view == "regional" else "ntt_affinity"]
        reference = fit_condition(
            values, data["fitting"], data["anchors"], covariance, clusters, scale, deadline
        )
        references[scale] = reference
        key = name + ":reference:" + scale
        checks[key] = reference[0]
        assignments[key] = reference[1]
        for parameter, array in reference[2].items():
            parameters[key + ":" + parameter] = array
    equivalences[name + ":reference"] = [
        equivalence(
            references["raw"],
            references["equivalent"],
            data["fitting"],
            data["anchors"],
            covariance,
            clusters,
        )
    ]
    for kind in ("single", "ensemble"):
        matrices = (
            views[view]
            if kind == "single"
            else [np.mean(views[view][i : i + 5], axis=0) for i in range(0, 50, 5)]
        )
        entries = {scale: [] for scale in SCALES}
        equivalents = []
        for i, values in enumerate(matrices):
            for scale in SCALES:
                entries[scale].append(
                    fit_condition(
                        values, data["fitting"], data["anchors"], covariance, clusters, scale, deadline
                    )
                )
            equivalents.append(
                equivalence(
                    entries["raw"][-1],
                    entries["equivalent"][-1],
                    data["fitting"],
                    data["anchors"],
                    covariance,
                    clusters,
                )
            )
            write_json_atomic(
                output / "progress.json", {"configuration": name, "kind": kind, "completed": i + 1}
            )
        equivalences[name + ":" + kind] = equivalents
        for scale, rows in entries.items():
            key = name + ":" + kind + ":" + scale
            store_fits(key, rows, checks, assignments, parameters)
            summaries[key] = summarize(
                assignments[key],
                checks[key],
                references[scale][1],
                data,
                clusters,
                original if view == "regional" else None,
            )
        print(f"{name} {kind}: {len(matrices)} banks complete", flush=True)


def run(output: Path) -> dict:
    """Execute the prospective source-bound factorial and preserve complete local artifacts.

    Parameters
    ----------
    output : Path
        Previously created empty local execution directory.

    Returns
    -------
    dict
        Source-bound completed result with all finite fit denominators.
    """
    started = time.perf_counter()
    deadline = started + 1200
    data = cached_inputs()
    predecessors = {
        name: read_bound(ROOT / "outputs" / name / "result.json", digest, data["inputs"])
        for name, digest in PREVIOUS.items()
    }
    for name, receipt in predecessors.items():
        status = json.loads((ROOT / "outputs" / name / "status.json").read_text())
        if (
            receipt["status"] != "completed"
            or status.get("status") != "completed"
            or status.get("result_sha256") != PREVIOUS[name]
        ):
            raise ValueError("Predecessor completion differs")
    inherited = {**data["cluster"]["source_sha256"]}
    for receipt in predecessors.values():
        inherited.update(receipt["source_sha256"])
    sources = committed_sources(SOURCES, inherited)
    replay = {"cluster03": replay_predecessor(data, deadline)}
    manifest = json.loads((ROOT / "outputs/cluster04_bank_robustness/bank_manifest_local.json").read_text())
    if manifest["banks"] != support_draws(data["selection"]):
        raise ValueError("Frozen support draws differ")
    write_json_atomic(output / "bank_manifest_local.json", manifest)
    views = {"regional": [], "record": []}
    for draw in manifest["banks"]:
        record, regional = bank_affinities(data["cache"], bank_prototypes(draw, data["support_features"]))
        views["record"].append(record)
        views["regional"].append(regional)
    replay["cluster04"] = replay04(data, predecessors["cluster04_bank_robustness"], views, deadline)
    replay["cluster05"] = replay05(data, predecessors["cluster05_matched_controls"], deadline)
    checks, assignments, parameters, summaries, equivalences = {}, {}, {}, {}, {}
    original = data["assignments"][data["cluster"]["selected"]["ntt_regions"]["key"] + "|all"]
    for name, configuration in CONFIGS.items():
        run_configuration(
            name,
            configuration,
            views,
            data,
            original,
            deadline,
            output,
            checks,
            assignments,
            parameters,
            summaries,
            equivalences,
        )
    decision = averaging_decision(
        summaries["regional_tied2:single:rms"], summaries["regional_tied2:ensemble:rms"]
    )
    np.savez_compressed(output / "assignments_local.npz", **assignments)
    np.savez_compressed(output / "model_parameters_local.npz", **parameters)
    np.savez_compressed(
        output / "coordinates_local.npz", **{key: np.stack(value) for key, value in views.items()}
    )
    data["ids"].to_csv(output / "identities_local.csv", index=False)
    write_json_atomic(
        output / "patient_splits_local.json",
        {"fitting": data["fitting"].tolist(), "anchors": data["anchors"].tolist()},
    )
    check_budget(deadline)
    result = {
        "study": "CLUSTER-06",
        "status": "completed",
        "decision": decision,
        "equivalence_passed": all(row["passed"] for rows in equivalences.values() for row in rows),
        "equivalence_checks": equivalences,
        "summaries": summaries,
        "checks": checks,
        "predecessor_replay": replay,
        "fixed_floor_raw_rms_ari": scale_contrasts(assignments, data["anchors"]),
        "source_sha256": sources,
        "input_sha256": data["inputs"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "counts": {
            "patients": 1993,
            "fitting": 1582,
            "anchors": 411,
            "single_banks": 50,
            "ensembles": 10,
            "factorial_fits": 540,
            "own_reference_fits": 9,
            "exception_failed_fits": 0,
            "nonconverged_factorial_fits": sum(
                not row["converged"] for value in checks.values() if isinstance(value, list) for row in value
            ),
        },
        "runtime": runtime_fingerprint(),
        "seconds": time.perf_counter() - started,
        "waveform_reads": 0,
        "new_encoder_inference": 0,
        "gpu_used": False,
        "target_annotations_parsed": False,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in output.iterdir()
            if path.name not in ("result.json", "status.json", "failure.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result
