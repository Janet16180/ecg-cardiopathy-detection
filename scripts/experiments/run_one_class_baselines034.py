"""Experiment 034: one-class baselines against the Mahalanobis normal reference on 026b's pooled fit set."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment import one_class_baselines as baselines
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.intervals import metric_intervals, paired_auroc_difference
from ecg_experiment.normal_manifold import fit_knn, fit_mahalanobis, knn_scores, mahalanobis_scores
from ecg_experiment.referral_budget import budget_threshold
from scripts.experiments import run_label_efficiency025 as label_efficiency
from scripts.experiments import run_multisource_manifold026b as manifold026b
from scripts.experiments import run_normal_manifold026 as manifold026
from scripts.experiments.run_multisource_calibration027b import feature_identity

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment034_one_class_baselines_v1"
PRIOR026B = manifold026b.OUTPUT
ENCODERS = ("xecg", "jepa", "cpc")
PRIMARY_ENCODER = "xecg"
REFERENCE = "mahalanobis"
METHODS = (REFERENCE, "knn_kth", "knn_mean", "knn026", "ocsvm", "iforest", "gmm", *baselines.NETWORKS)
FAMILIES = manifold026b.CHALLENGE_FAMILIES
SEED = baselines.SEED
DRAWS = 2000
WALL_CAP_SECONDS = 1800.0
REPRODUCTION_TOLERANCE = 1e-9
DEVELOPMENT_MARGIN = 0.01
BUDGET_PER_MILLE = 50
Method = Callable[[float], tuple[dict[str, np.ndarray], dict[str, Any]]]
EXPECTED = {"fit": 10846, "sph": (21008, 7190), "development": (1306, 843),
            "family:chapman_ningbo": (4432, 3254), "family:georgia": (1718, 1372),
            "family:cpsc": (1462, 1279), "source_normals": 1707}
SOURCES = (
    "ecg_experiment/one_class_baselines.py", "ecg_experiment/normal_manifold.py",
    "ecg_experiment/intervals.py", "ecg_experiment/referral_budget.py",
    "ecg_experiment/multisource_manifold.py", "ecg_experiment/full_development.py",
    "scripts/experiments/run_one_class_baselines034.py",
    "scripts/experiments/run_multisource_manifold026b.py",
    "scripts/experiments/run_normal_manifold026.py", "scripts/experiments/run_label_efficiency025.py",
    "scripts/experiments/run_multisource_calibration027b.py", "pyproject.toml", "uv.lock",
    "docs/experiment-034-one-class-baselines.md",
)


def prior_026b() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """
    Load 026b's result and its saved tables, each checked against its receipt.

    Returns
    -------
    tuple[dict[str, Any], dict[str, pd.DataFrame]]
        026b's ``result.json`` and its fit-set, SPH, development and Challenge tables.

    Raises
    ------
    ValueError
        If a table differs from 026b's recorded hash.
    """
    result = json.loads((PRIOR026B / "result.json").read_text())
    tables = {}
    for name in ("fit_sets", "sph_scores", "development_scores", "challenge_scores"):
        path = PRIOR026B / f"{name}.csv"
        if sha256_file(path) != result["outputs_sha256"][f"{name}.csv"]:
            raise ValueError(f"Experiment 026b {name}.csv differs from its receipt")
        tables[name] = pd.read_csv(path, dtype={"record": str, "ecg_id": str}, float_precision="round_trip")
    return result, tables


def fit_candidates(pool: pd.DataFrame, rows: pd.DataFrame, saved: pd.DataFrame
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Select 026b's pooled fit set among the PTB-XL and Challenge training normals, with tuning units.

    Parameters
    ----------
    pool : pd.DataFrame
        Experiment 025's PTB-XL training pool.
    rows : pd.DataFrame
        Challenge split rows with features, in 026b's order.
    saved : pd.DataFrame
        026b's ``fit_sets.csv``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean pooled membership over the candidates (PTB-XL normals, then Challenge training normals)
        and the tuning unit of each candidate.

    Raises
    ------
    ValueError
        If the candidates differ from 026b's, in order.
    """
    ptb = pool[pool["standard"] == 0]
    normals = rows[(rows["split"] == "train") & (rows["primary"] == 0)]
    candidates = pd.concat([
        pd.DataFrame({"family": "ptbxl", "source": "ptbxl", "record": ptb["ecg_id"].astype(str).to_numpy(),
                      "unit": ("ptbxl:" + ptb["patient_id"].astype(str)).to_numpy()}),
        pd.DataFrame({"family": normals["family"].to_numpy(), "source": normals["source"].to_numpy(),
                      "record": normals["record"].to_numpy(),
                      "unit": ("challenge:" + normals["source"] + ":" + normals["record"]).to_numpy()}),
    ], ignore_index=True)
    columns = ["family", "source", "record"]
    if not candidates[columns].equals(saved[columns]):
        raise ValueError("Fit-set candidates differ from Experiment 026b")
    pooled = saved["pooled"].eq(True).to_numpy()
    return pooled, candidates["unit"].to_numpy()


def run_method(name: str, compute: Method) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """
    Run one method under the wall-time cap.

    Parameters
    ----------
    name : str
        Method name, for the log.
    compute : Method
        Takes the ``time.monotonic()`` deadline and returns scores per set and tuning details.

    Returns
    -------
    tuple[dict[str, np.ndarray] | None, dict[str, Any]]
        Scores per set (None when dropped) and the method's status, seconds and details.
    """
    start = time.monotonic()
    scores, details, status = None, {}, "complete"
    try:
        scores, details = compute(start + WALL_CAP_SECONDS)
    except baselines.WallTimeExceededError:
        status = "dropped_wall_time"
    seconds = time.monotonic() - start
    if seconds > WALL_CAP_SECONDS:
        scores, status = None, "dropped_wall_time"
    print(json.dumps({"stage": f"method:{name}", "status": status, "seconds": round(seconds, 1)}), flush=True)
    return scores, {"status": status, "seconds": seconds, **details}


def method_table(fit_x: np.ndarray, units: np.ndarray, inputs: dict[str, np.ndarray]
                 ) -> tuple[dict[str, Method], dict[str, int]]:
    """
    Build the protocol's methods as callables of a deadline.

    The reference is fitted here, since its scaler and PCA give the whitened space of the other methods.

    Parameters
    ----------
    fit_x : np.ndarray
        Raw embeddings of the fit set.
    units : np.ndarray
        Tuning unit of each fit-set row.
    inputs : dict[str, np.ndarray]
        Raw embeddings to score, by set.

    Returns
    -------
    tuple[dict[str, Method], dict[str, int]]
        One callable per method, in ``METHODS`` order, and the size of the tuning slice.
    """
    reference = fit_mahalanobis(fit_x)
    whitened = baselines.whiten(reference, fit_x)
    queries = {name: baselines.whiten(reference, x) for name, x in inputs.items()}
    held = baselines.holdout_mask(units)
    explained = float(reference[1].explained_variance_ratio_.sum())

    def neighbours(position: int) -> dict[str, np.ndarray]:
        return {name: baselines.knn_distances(fit_x, x)[position] for name, x in inputs.items()}

    def knn026(_: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        model = fit_knn(fit_x)
        return {name: knn_scores(model, x) for name, x in inputs.items()}, {"k": 25}

    def gmm(_: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        chosen, likelihood = baselines.choose_gmm_components(whitened[~held], whitened[held])
        details = {"components": chosen, "held_out_log_likelihood": likelihood}
        return baselines.gmm_scores(whitened, queries, chosen), details

    table: dict[str, Method] = {
        REFERENCE: lambda _: ({name: mahalanobis_scores(reference, x) for name, x in inputs.items()},
                              {"pca_explained_variance": explained}),
        "knn_kth": lambda _: (neighbours(0), {"k": baselines.KNN_K}),
        "knn_mean": lambda _: (neighbours(1), {"k": baselines.KNN_K}),
        "knn026": knn026,
        "ocsvm": lambda _: (baselines.ocsvm_scores(whitened, queries), {}),
        "iforest": lambda _: (baselines.iforest_scores(whitened, queries), {}),
        "gmm": gmm,
    }
    for kind in baselines.NETWORKS:
        table[kind] = lambda deadline, kind=kind: baselines.network_scores(
            kind, whitened, held, queries, deadline)
    tuning = {"held_out_rows": int(held.sum()), "held_out_units": int(len(np.unique(units[held]))),
              "units": int(len(np.unique(units)))}
    return table, tuning


def evaluate_set(y: np.ndarray, units: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    AUROC and average precision of every method, and each method minus the reference with its interval.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    units : np.ndarray
        Bootstrap unit of each ECG (patient or record).
    scores : dict[str, np.ndarray]
        Score per method.

    Returns
    -------
    dict[str, Any]
        Counts, ``metrics`` per method and ``contrasts`` per method against the reference.
    """
    return {
        "records": len(y), "positives": int(y.sum()), "units": len(np.unique(units)),
        "metrics": {name: {"auroc": float(roc_auc_score(y, values)),
                           "average_precision": float(average_precision_score(y, values))}
                    for name, values in scores.items()},
        "contrasts": {f"{name}_minus_{REFERENCE}": paired_auroc_difference(
                          units, y, values, scores[REFERENCE], DRAWS, SEED)
                      for name, values in scores.items() if name != REFERENCE},
    }


def referral(sph_y: np.ndarray, sph_patients: np.ndarray, sph_scores: np.ndarray,
             source_normals: np.ndarray) -> dict[str, Any]:
    """
    SPH sensitivity and false-referral rate at a 5% budget set on the Challenge calibration normals.

    Parameters
    ----------
    sph_y : np.ndarray
        SPH standard labels.
    sph_patients : np.ndarray
        SPH patient IDs.
    sph_scores : np.ndarray
        SPH scores of one method.
    source_normals : np.ndarray
        The method's scores of the Challenge calibration normals.

    Returns
    -------
    dict[str, Any]
        Threshold, sensitivity and false-referral rate with whole-patient intervals, and the sensitivity at
        the budget threshold set on the SPH normals themselves.
    """
    threshold = budget_threshold(source_normals, BUDGET_PER_MILLE)
    # metric_intervals refers scores >= its threshold; the next float up gives 030's strict rule.
    strict = float(np.nextafter(threshold, np.inf))
    summary = metric_intervals(sph_y, sph_scores, sph_patients, strict, DRAWS, SEED)
    sensitivity, specificity = summary["metrics"]["sensitivity"], summary["metrics"]["specificity"]
    local = budget_threshold(sph_scores[sph_y == 0], BUDGET_PER_MILLE)
    return {"threshold": threshold, "sensitivity": sensitivity,
            "false_referral_rate": {"value": 1 - specificity["value"], "ci_low": 1 - specificity["ci_high"],
                                    "ci_high": 1 - specificity["ci_low"]},
            "skipped_draws": summary["skipped_draws"],
            "sensitivity_at_sph_normal_quantile": float(np.mean(sph_scores[sph_y == 1] > local))}


def evaluate_encoder(name: str, fit_x: np.ndarray, units: np.ndarray, inputs: dict[str, np.ndarray],
                     sets: dict[str, dict[str, Any]], source_normals: np.ndarray) -> dict[str, Any]:
    """
    Fit every method on one encoder's fit set and evaluate it on every set.

    Parameters
    ----------
    name : str
        Encoder name.
    fit_x : np.ndarray
        Raw embeddings of the fit set.
    units : np.ndarray
        Tuning unit of each fit-set row.
    inputs : dict[str, np.ndarray]
        Raw embeddings of ``sph``, ``development`` and ``challenge``.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.
    source_normals : np.ndarray
        True for the Challenge calibration normals, which set the referral threshold.

    Returns
    -------
    dict[str, Any]
        Scores, method details, evaluations and the referral readout.
    """
    start = time.monotonic()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        table, tuning = method_table(fit_x, units, inputs)
        scores, details = {}, {}
        for method, compute in table.items():
            result, details[method] = run_method(f"{name}:{method}", compute)
            if result is not None:
                scores[method] = result
        evaluations = {}
        for set_name, spec in sets.items():
            chosen = {method: values[spec["scores_of"]][spec["selection"]]
                      for method, values in scores.items()}
            evaluations[set_name] = evaluate_set(spec["y"], spec["units"], chosen)
            aurocs = {method: round(value["auroc"], 4)
                      for method, value in evaluations[set_name]["metrics"].items()}
            print(json.dumps({"stage": f"{name}:{set_name}", "auroc": aurocs}), flush=True)
        sph = sets["sph"]
        budget = {method: referral(sph["y"], sph["units"], values["sph"], values["challenge"][source_normals])
                  for method, values in scores.items()}
    return {"scores": scores, "methods": details, "tuning_slice": tuning, "evaluations": evaluations,
            "referral_5pct": budget, "seconds": time.monotonic() - start}


def evaluation_sets(evaluation: pd.DataFrame, sph: pd.DataFrame, challenge: pd.DataFrame
                    ) -> dict[str, dict[str, Any]]:
    """
    Labels, bootstrap units and row selections of every evaluation set.

    Parameters
    ----------
    evaluation : pd.DataFrame
        026's development rows.
    sph : pd.DataFrame
        SPH evaluation rows.
    challenge : pd.DataFrame
        Challenge calibration rows with features.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per set: the scored set it reads (``scores_of``), its ``selection``, ``y`` and ``units``.
    """
    sets = {"sph": {"scores_of": "sph", "selection": np.ones(len(sph), dtype=bool),
                    "y": sph["primary"].to_numpy(dtype=np.int64),
                    "units": sph["patient_id"].to_numpy(dtype=str)},
            "development": {"scores_of": "development", "selection": np.ones(len(evaluation), dtype=bool),
                            "y": evaluation["standard"].to_numpy(dtype=np.int64),
                            "units": evaluation["patient_id"].to_numpy(dtype=str)}}
    family = challenge["family"].to_numpy()
    for name in FAMILIES:
        selected = family == name
        sets[f"family:{name}"] = {"scores_of": "challenge", "selection": selected,
                                  "y": challenge.loc[selected, "primary"].to_numpy(dtype=np.int64),
                                  "units": challenge.loc[selected, "record"].to_numpy(dtype=str)}
    return sets


def check_counts(fit_rows: int, sets: dict[str, dict[str, Any]], source_normals: np.ndarray
                 ) -> dict[str, Any]:
    """
    Require the fit-set and evaluation counts of the protocol.

    Parameters
    ----------
    fit_rows : int
        Fit-set ECGs.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.
    source_normals : np.ndarray
        True for the Challenge calibration normals.

    Returns
    -------
    dict[str, Any]
        The counts found.

    Raises
    ------
    ValueError
        If a count differs from the protocol.
    """
    found = {"fit": fit_rows, "source_normals": int(source_normals.sum()),
             **{name: (len(spec["y"]), int(spec["y"].sum())) for name, spec in sets.items()}}
    if found != EXPECTED:
        raise ValueError(f"Counts differ from the protocol: {found}")
    return found


def reproduce_026b(results: dict[str, Any], tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Require the Mahalanobis reference to reproduce 026b's saved pooled scores.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.
    tables : dict[str, pd.DataFrame]
        026b's saved tables, in the same row order as this run's sets.

    Returns
    -------
    dict[str, float]
        Largest relative difference per encoder over SPH, development and the Challenge calibration rows.

    Raises
    ------
    ValueError
        If any difference exceeds ``REPRODUCTION_TOLERANCE``.
    """
    differences = {}
    for name, output in results.items():
        worst = 0.0
        for set_name in ("sph", "development", "challenge"):
            new = output["scores"][REFERENCE][set_name]
            old = tables[f"{set_name}_scores"][f"{name}_pooled"].to_numpy()
            worst = max(worst, float(np.max(np.abs(new - old) / np.maximum(1.0, np.abs(old)))))
        differences[name] = worst
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The reference does not reproduce Experiment 026b: {differences}")
    return differences


def reading(results: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the prespecified decision rule and describe every contrast by its interval.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.

    Returns
    -------
    dict[str, Any]
        Per method on the primary encoder whether it qualifies, the decision, and the side of every contrast.
    """
    primary = results[PRIMARY_ENCODER]["evaluations"]
    qualifies = {}
    for key, contrast in primary["sph"]["contrasts"].items():
        development = primary["development"]["contrasts"][key]["difference"]
        qualifies[key.removesuffix(f"_minus_{REFERENCE}")] = bool(
            contrast["ci_low"] > 0 and development >= -DEVELOPMENT_MARGIN)
    winners = [method for method, passed in qualifies.items() if passed]
    candidate = max(winners, key=lambda method: primary["sph"]["contrasts"][f"{method}_minus_{REFERENCE}"]
                    ["difference"], default=None)
    sides = {name: {set_name: {key: manifold026b.interval_side(contrast)
                               for key, contrast in evaluation["contrasts"].items()}
                    for set_name, evaluation in output["evaluations"].items()}
             for name, output in results.items()}
    decision = "mahalanobis_stays" if candidate is None else "candidate_replacement"
    return {"qualifies": qualifies, "decision": decision, "candidate": candidate, "interval_sides": sides}


def load_inputs() -> dict[str, Any]:
    """
    Load and check every input: 026b's receipts and tables, the fit set and the evaluation rows.

    Returns
    -------
    dict[str, Any]
        Identities, 026b tables, fit features and units per encoder, scored inputs, sets and counts.
    """
    identity026 = manifold026.identity()
    features_metadata, feature_hashes = feature_identity()
    _, tables = prior_026b()
    groups = cohorts(ptb_table())
    pool, evaluation, _ = label_efficiency.select_rows(groups["train"], groups["development"])
    ptb_features = label_efficiency.load_features(groups["train"], groups["development"], pool, evaluation)
    sph, sph_features = manifold026.load_sph()
    development_ids = evaluation["ecg_id"].astype(str).to_numpy()
    if not (np.array_equal(tables["development_scores"]["ecg_id"].to_numpy(), development_ids)
            and np.array_equal(tables["sph_scores"]["ecg_id"].to_numpy(), sph["ecg_id"].to_numpy())):
        raise ValueError("Row order differs from Experiment 026b")
    rows, challenge_x = manifold026b.challenge_features(manifold026b.split_table())
    pooled, units = fit_candidates(pool, rows, tables["fit_sets"])
    normal = ((rows["split"] == "train") & (rows["primary"] == 0)).to_numpy()
    calibration = (rows["split"] == "calibration").to_numpy()
    challenge = rows[calibration].reset_index(drop=True)
    if not challenge["record"].equals(tables["challenge_scores"]["record"]):
        raise ValueError("Challenge row order differs from Experiment 026b")
    ptb_fit = (pool["standard"] == 0).to_numpy()
    fit_x = {name: np.concatenate([ptb_features[name][0][ptb_fit], challenge_x[name][normal]])[pooled]
             for name in ENCODERS}
    inputs = {name: {"sph": sph_features[name], "development": ptb_features[name][1],
                     "challenge": challenge_x[name][calibration]} for name in ENCODERS}
    sets = evaluation_sets(evaluation, sph, challenge)
    source_normals = (challenge["primary"] == 0).to_numpy()
    counts = check_counts(int(pooled.sum()), sets, source_normals)
    frames = {"development": evaluation[["ecg_id", "patient_id", "standard"]].reset_index(drop=True),
              "sph": sph[["ecg_id", "patient_id", "primary"]],
              "challenge": challenge[["source", "family", "record", "primary"]]}
    return {"identity026": identity026, "feature_metadata": features_metadata,
            "feature_hashes": feature_hashes,
            "tables": tables, "fit_x": fit_x, "units": units[pooled], "inputs": inputs, "sets": sets,
            "source_normals": source_normals, "counts": counts, "frames": frames}


def write_scores(frames: dict[str, pd.DataFrame], results: dict[str, Any]) -> list[str]:
    """
    Write the per-row scores of every encoder and method, one table per scored set.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Identifying columns per scored set.
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.

    Returns
    -------
    list[str]
        Names of the written files.
    """
    written = []
    for set_name, frame in frames.items():
        table = frame.copy()
        for name, output in results.items():
            for method, values in output["scores"].items():
                table[f"{name}_{method}"] = values[set_name]
        table.to_csv(OUTPUT / f"{set_name}_scores.csv", index=False)
        written.append(f"{set_name}_scores.csv")
    return written


def main() -> None:
    """Fit every one-class method once per encoder and score SPH, PTB-XL development and the families."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 034 v1 has already run")
    started = time.monotonic()
    loaded = load_inputs()
    print(json.dumps({"stage": "counts", **loaded["counts"]}), flush=True)
    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {name: executor.submit(evaluate_encoder, name, loaded["fit_x"][name], loaded["units"],
                                         loaded["inputs"][name], loaded["sets"], loaded["source_normals"])
                   for name in ENCODERS}
        results = {name: future.result() for name, future in futures.items()}
    reproduction = reproduce_026b(results, loaded["tables"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    written = write_scores(loaded["frames"], results)
    result = {
        "status": "complete",
        "identity": {"experiment026": loaded["identity026"],
                     "experiment026b_result_sha256": sha256_file(PRIOR026B / "result.json"),
                     "challenge_features": loaded["feature_hashes"],
                     "challenge_features_identity": loaded["feature_metadata"]["identity"],
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "counts": loaded["counts"], "methods": list(METHODS), "seed": SEED, "bootstrap_draws": DRAWS,
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "reference_max_relative_difference_from_026b": reproduction,
        "encoders": {name: {key: value for key, value in output.items() if key != "scores"}
                     for name, output in results.items()},
        "reading": reading(results),
        "outputs_sha256": {name: sha256_file(OUTPUT / name) for name in written},
        "challenge_test_read": False, "ptbxl_calibration_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": result["reading"]["decision"],
                      "candidate": result["reading"]["candidate"]}), flush=True)


if __name__ == "__main__":
    main()
