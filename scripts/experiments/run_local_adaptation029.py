"""Experiment 029: local recalibration and a local normal reference at a simulated new site (SPH)."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.local_adaptation import (
    OPTIONS,
    adapt,
    draw_records,
    local_components,
    recentered_scores,
    screening_rates,
    split_site,
)
from ecg_experiment.multisource_calibration import weighted_rates
from ecg_experiment.normal_manifold import fit_mahalanobis, mahalanobis_scores, metrics
from ecg_experiment.screening_threshold import (
    calibrate,
    calibration_fit,
    fit_platt,
    head_logits,
    screening_threshold,
)
from scripts.experiments import run_label_efficiency025 as label_efficiency
from scripts.experiments import run_multisource_manifold026b as manifold026b
from scripts.experiments import run_normal_manifold026 as manifold026
from scripts.experiments.run_multisource_calibration027b import feature_identity

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment029_local_adaptation_v1"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR026B = ROOT / "outputs/experiment026b_multisource_manifold_v1"
SPH_ROWS = ROOT / "data/processed/sph_clean_v1/rows.csv"
ENCODERS = ("xecg", "jepa", "cpc")
READOUTS = ("pooled", "ptbxl")
PRIMARY_ENCODER = "xecg"
PRIMARY_READOUT = "pooled"
PRIMARY_OPTION = "platt_threshold"
SEED = 29029
SIZES = (50, 100, 200, 500, 1000)
LOW_SIZES = (50, 100, 200, 500, 1000, 2000, 5000)
LOW_PREVALENCE = 0.05
DRAWS_A = 200
NORMAL_SIZES = (50, 100, 200, 500, 1000)
DRAWS_B = 50
NORMAL_OPTIONS = ("local_only", "local_plus_pooled", "recentered")
TARGET = 0.95
SHARE = 0.90
NEGLIGIBLE_AUROC = 0.005
TOLERANCE = 1e-9
EXPECTED_SPLIT = {"evaluation": {"ecgs": 10479, "patients": 10181, "positives": 3584},
                  "local": {"ecgs": 10529, "patients": 10183, "positives": 3606}}
EXPECTED_POOLED_NORMALS = 10846
SOURCES = (
    "ecg_experiment/local_adaptation.py", "ecg_experiment/screening_threshold.py",
    "ecg_experiment/multisource_calibration.py", "ecg_experiment/normal_manifold.py",
    "ecg_experiment/evaluation.py", "scripts/experiments/run_local_adaptation029.py",
    "scripts/experiments/run_multisource_manifold026b.py", "scripts/experiments/run_normal_manifold026.py",
    "scripts/experiments/run_label_efficiency025.py", "pyproject.toml", "uv.lock",
    "docs/experiment-029-local-adaptation.md",
)


def prior_readouts() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Load 022b's result and saved predictions, checked against its receipt and the SPH manifest.

    Returns
    -------
    tuple[dict[str, Any], dict[str, np.ndarray]]
        022b's ``result.json`` and the arrays of its ``predictions.npz``.

    Raises
    ------
    ValueError
        If the predictions differ from their receipt or from the SPH manifest rows.
    """
    result = json.loads((PRIOR022B / "result.json").read_text())
    if sha256_file(PRIOR022B / "predictions.npz") != result["outputs_sha256"]["predictions.npz"]:
        raise ValueError("022b predictions differ from their receipt")
    with np.load(PRIOR022B / "predictions.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    rows = pd.read_csv(SPH_ROWS, dtype={"ecg_id": str, "patient_id": str})
    rows = rows[rows["use_evaluation"].astype(str).eq("True") & rows["primary"].notna()]
    if not (np.array_equal(rows["ecg_id"].to_numpy(), arrays["sph_ecg_ids"])
            and np.array_equal(rows["patient_id"].to_numpy(), arrays["sph_patient_ids"])
            and np.array_equal(rows["primary"].to_numpy(dtype=np.int64), arrays["sph_labels"])):
        raise ValueError("022b SPH rows differ from the SPH manifest")
    return result, arrays


def site_split(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Split SPH by patient into a local pool and an evaluation half, as the protocol fixes.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Output of ``prior_readouts``.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Evaluation mask per SPH ECG and the counts of each half.

    Raises
    ------
    ValueError
        If the counts differ from the protocol or a patient is in both halves.
    """
    patients, y = arrays["sph_patient_ids"], arrays["sph_labels"]
    evaluation = split_site(patients, y, SEED)
    counts = {name: {"ecgs": int(mask.sum()), "patients": len(np.unique(patients[mask])),
                     "positives": int(y[mask].sum())}
              for name, mask in (("evaluation", evaluation), ("local", ~evaluation))}
    if counts != EXPECTED_SPLIT or set(patients[evaluation]) & set(patients[~evaluation]):
        raise ValueError(f"Split differs from the protocol: {counts}")
    return evaluation, counts


def source_points(result: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], float]:
    """
    Refit each readout's source Platt mapping and threshold and require 022b's values.

    Parameters
    ----------
    result : dict[str, Any]
        022b's ``result.json``.
    arrays : dict[str, np.ndarray]
        Output of ``prior_readouts``.

    Returns
    -------
    tuple[dict[str, Any], float]
        Per ``readout:encoder``: calibration logits and labels and the (calibrator, threshold) pair; and the
        largest difference from 022b.

    Raises
    ------
    ValueError
        If a coefficient, threshold or SPH calibrated probability differs by more than ``TOLERANCE``.
    """
    families = arrays["calibration_families"]
    columns = list(arrays["sph_calibrated_columns"])
    members = {"pooled": np.ones(len(families), dtype=bool), "ptbxl": families == "ptbxl"}
    points, worst = {}, 0.0
    for readout in READOUTS:
        for name in ENCODERS:
            selected = members[readout]
            logits = head_logits(arrays[f"calibration_{name}_{readout}"][selected])
            y = arrays["calibration_labels"][selected]
            calibrator = fit_platt(logits, y)
            threshold = screening_threshold(y, calibrate(calibrator, logits))
            saved = result["platt"][readout][name]
            sph = calibrate(calibrator, head_logits(arrays[f"sph_{name}_{readout}"]))
            differences = (abs(calibrator.coef_[0, 0] - saved["slope"]),
                           abs(calibrator.intercept_[0] - saved["intercept"]),
                           abs(threshold - result["thresholds"][readout][name]),
                           float(np.abs(sph - arrays["sph_calibrated"][:, columns.index(f"{readout}:{name}")])
                                 .max()))
            worst = max(worst, *differences)
            points[f"{readout}:{name}"] = {"logits": logits, "y": y, "point": (calibrator, threshold)}
    if worst > TOLERANCE:
        raise ValueError(f"Source operating points do not reproduce 022b: {worst}")
    return points, worst


def draw_plans(local_y: np.ndarray) -> dict[str, dict[int, list[np.ndarray]]]:
    """
    Local-pool positions of every Part A draw, at the pool's prevalence and at 5%.

    Parameters
    ----------
    local_y : np.ndarray
        Labels of the local pool.

    Returns
    -------
    dict[str, dict[int, list[np.ndarray]]]
        ``site`` and ``low`` plans keyed by size, one position array per draw.
    """
    site = {n: [draw_records(local_y, n, np.random.default_rng([SEED, 1, n, draw]))
                for draw in range(DRAWS_A)] for n in SIZES}
    low = {n: [draw_records(local_y, n, np.random.default_rng([SEED, 2, n, draw]),
                            prevalence=LOW_PREVALENCE)
               for draw in range(DRAWS_A)] for n in LOW_SIZES}
    return {"site": site, "low": low}


def probability_quality(y: np.ndarray, calibrated: np.ndarray) -> dict[str, float]:
    """
    Brier score, 10-bin ECE and calibration intercept and slope of calibrated probabilities.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    calibrated : np.ndarray
        Calibrated probabilities.

    Returns
    -------
    dict[str, float]
        ``brier``, ``ece_10_bins``, ``intercept`` and ``slope``; the last two are NaN when a probability is
        exactly 0 or 1, where the calibration logit is undefined.
    """
    rates = weighted_rates(np.ones(len(y)), y, calibrated[:, None], np.array([0.5]))
    quality = {"brier": float(rates["brier"][0]), "ece_10_bins": float(rates["ece_10_bins"][0]),
               "intercept": np.nan, "slope": np.nan}
    if ((calibrated > 0) & (calibrated < 1)).all():
        quality.update(calibration_fit(y, calibrated))
    return quality


def part_a_encoder(name: str, logits: dict[str, np.ndarray], local_y: np.ndarray, evaluation_y: np.ndarray,
                   points: dict[str, Any], plans: dict[str, dict[int, list[np.ndarray]]]
                   ) -> list[dict[str, Any]]:
    """
    Score every Part A draw and option of one encoder's two readouts on the evaluation half.

    Parameters
    ----------
    name : str
        Encoder name.
    logits : dict[str, np.ndarray]
        SPH head logits per readout, keyed ``local`` and ``evaluation`` inside, as ``readout:half``.
    local_y, evaluation_y : np.ndarray
        Labels of the local pool and of the evaluation half.
    points : dict[str, Any]
        Output of ``source_points`` for this encoder's heads.
    plans : dict[str, dict[int, list[np.ndarray]]]
        Output of ``draw_plans``.

    Returns
    -------
    list[dict[str, Any]]
        One row per readout, plan, size, draw and option (size 0 is the source point).
    """
    rows = []
    with threadpool_limits(limits=1):
        for readout in READOUTS:
            head = points[f"{readout}:{name}"]
            source_logits, source_y, source = head["logits"], head["y"], head["point"]
            local_logits, evaluation_logits = logits[f"{readout}:local"], logits[f"{readout}:evaluation"]
            source_calibrated = calibrate(source[0], evaluation_logits)
            base = {"readout": readout, "encoder": name}
            referred = source_calibrated >= source[1]
            for plan in plans:
                prevalence = LOW_PREVALENCE if plan == "low" else None
                rows.append({**base, "plan": plan, "n": 0, "draw": 0, "option": "source", "positives": 0,
                             "fallback": False, "confident": False, "differs_from_threshold_only": False,
                             **screening_rates(evaluation_y, referred, LOW_PREVALENCE),
                             **probability_quality(evaluation_y, source_calibrated)})
                for n, draws in plans[plan].items():
                    for draw, positions in enumerate(draws):
                        decisions = {}
                        for option in OPTIONS:
                            calibrator, threshold, fallback, confident = adapt(
                                option, local_logits[positions], local_y[positions], source_logits, source_y,
                                source)
                            calibrated = calibrate(calibrator, evaluation_logits)
                            decisions[option] = calibrated >= threshold
                            quality = {"brier": np.nan, "ece_10_bins": np.nan, "intercept": np.nan,
                                       "slope": np.nan}
                            if prevalence is None and option in ("platt_threshold", "blend") and not fallback:
                                quality = probability_quality(evaluation_y, calibrated)
                            rows.append({
                                **base, "plan": plan, "n": n, "draw": draw, "option": option,
                                "positives": int(local_y[positions].sum()), "fallback": fallback,
                                "confident": confident,
                                "differs_from_threshold_only": bool(
                                    (decisions[option] != decisions["threshold_only"]).any()),
                                **screening_rates(evaluation_y, decisions[option], LOW_PREVALENCE),
                                **quality})
            print(json.dumps({"stage": f"part_a:{readout}:{name}", "rows": len(rows)}), flush=True)
    return rows


def summarize_a(draws: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Summary of the Part A draws per plan, readout, encoder, size and option.

    Parameters
    ----------
    draws : pd.DataFrame
        Every Part A row.

    Returns
    -------
    list[dict[str, Any]]
        Shares, means and percentiles of each group.
    """
    summary = []
    keys = ["plan", "readout", "encoder", "n", "option"]
    for key, group in draws.groupby(keys, sort=False):
        sensitivity, specificity = group["sensitivity"], group["specificity"]
        summary.append({
            **dict(zip(keys, key, strict=True)), "draws": len(group),
            "share_sensitivity_at_least_095": float((sensitivity >= TARGET - 1e-12).mean()),
            "sensitivity_mean": float(sensitivity.mean()),
            "sensitivity_p5": float(sensitivity.quantile(0.05)),
            "specificity_mean": float(specificity.mean()),
            "specificity_p5": float(specificity.quantile(0.05)),
            "specificity_p95": float(specificity.quantile(0.95)),
            "referrals_per_1000_mean": float(group["referrals_per_1000"].mean()),
            "referrals_per_1000_at_5pct_mean": float(group["referrals_per_1000_assumed"].mean()),
            "positives_mean": float(group["positives"].mean()),
            "positives_p5": float(group["positives"].quantile(0.05)),
            "positives_p95": float(group["positives"].quantile(0.95)),
            "fallbacks": int(group["fallback"].sum()), "confident_share": float(group["confident"].mean()),
            "draws_differing_from_threshold_only": int(group["differs_from_threshold_only"].sum()),
            **{f"{metric}_mean": float(group[metric].mean()) if group[metric].notna().any() else None
               for metric in ("brier", "ece_10_bins", "intercept", "slope")},
            "calibration_undefined_draws": int(group["intercept"].isna().sum()),
        })
    return summary


def smallest_size(summary: list[dict[str, Any]], select: dict[str, Any], share_key: str,
                  size_key: str) -> dict[str, Any]:
    """
    Smallest size at which the share of draws meeting a criterion reaches 90%.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Summary rows.
    select : dict[str, Any]
        Values the rows must match.
    share_key, size_key : str
        Names of the share and the size fields.

    Returns
    -------
    dict[str, Any]
        ``smallest`` (``None`` when never reached), the ``largest_share`` and the share at every size.
    """
    rows = [row for row in summary if all(row[key] == value for key, value in select.items())
            and row[size_key] > 0]
    shares = {int(row[size_key]): row[share_key] for row in sorted(rows, key=lambda row: row[size_key])}
    reached = [size for size, share in shares.items() if share >= SHARE - 1e-12]
    return {"smallest": reached[0] if reached else None, "largest_share": max(shares.values()),
            "shares": shares}


def pooled_normals(sph_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                                                 pd.DataFrame, dict[str, Any]]:
    """
    Rebuild 026b's pooled normal fit set and load the SPH features.

    Parameters
    ----------
    sph_ids : np.ndarray
        SPH ECG IDs in 022b's order.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]
        Fit-set features per encoder, SPH features per encoder, 026b's saved SPH scores and the identity of
        the inputs.

    Raises
    ------
    ValueError
        If a file differs from its receipt, the fit set differs from 026b's, or the row orders differ.
    """
    result026b = json.loads((PRIOR026B / "result.json").read_text())
    for name in ("fit_sets.csv", "sph_scores.csv"):
        if sha256_file(PRIOR026B / name) != result026b["outputs_sha256"][name]:
            raise ValueError(f"026b {name} differs from its receipt")
    identity026 = manifold026.identity()
    _, feature_hashes = feature_identity()
    groups = cohorts(ptb_table())
    pool, evaluation, _ = label_efficiency.select_rows(groups["train"], groups["development"])
    ptb_features = label_efficiency.load_features(groups["train"], groups["development"], pool, evaluation)
    sph, sph_features = manifold026.load_sph()
    if not np.array_equal(sph["ecg_id"].to_numpy(), sph_ids):
        raise ValueError("SPH feature order differs from 022b")
    rows, challenge_x = manifold026b.challenge_features(manifold026b.split_table())
    normal = ((rows["split"] == "train") & (rows["primary"] == 0)).to_numpy()
    ptb_fit = (pool["standard"] == 0).to_numpy()
    candidates = pd.concat([pd.DataFrame({"source": "ptbxl", "record": pool["ecg_id"][ptb_fit].astype(str)}),
                            rows.loc[normal, ["source", "record"]].astype(str)], ignore_index=True)
    fit_sets = pd.read_csv(PRIOR026B / "fit_sets.csv", dtype={"record": str, "source": str})
    if not candidates.equals(fit_sets[["source", "record"]]):
        raise ValueError("Fit-set candidates differ from 026b's")
    pooled = (fit_sets["kept"] & fit_sets["pooled"].eq(True)).to_numpy()
    if pooled.sum() != EXPECTED_POOLED_NORMALS:
        raise ValueError(f"Pooled fit set has {pooled.sum()} normals")
    fit_x = {name: np.concatenate([ptb_features[name][0][ptb_fit], challenge_x[name][normal]])[pooled]
             for name in ENCODERS}
    sph_scores = pd.read_csv(PRIOR026B / "sph_scores.csv", dtype={"ecg_id": str},
                             float_precision="round_trip")
    if not np.array_equal(sph_scores["ecg_id"].to_numpy(), sph_ids):
        raise ValueError("026b SPH score order differs from 022b")
    identity = {"experiment026": identity026, "challenge_features": feature_hashes,
                "experiment026b_result_sha256": sha256_file(PRIOR026B / "result.json")}
    return fit_x, sph_features, sph_scores, identity


def part_b_encoder(name: str, fit_x: np.ndarray, x: np.ndarray, evaluation: np.ndarray,
                   local_normal: np.ndarray, y: np.ndarray, saved_pooled: np.ndarray,
                   plans: dict[int, list[np.ndarray]]
                   ) -> dict[str, Any]:
    """
    Fit every Part B reference of one encoder and score the evaluation half.

    Parameters
    ----------
    name : str
        Encoder name.
    fit_x : np.ndarray
        Features of 026b's pooled normals.
    x : np.ndarray
        SPH features.
    evaluation : np.ndarray
        Evaluation mask per SPH ECG.
    local_normal : np.ndarray
        SPH positions of the local pool's normals.
    y : np.ndarray
        SPH labels.
    saved_pooled : np.ndarray
        026b's saved SPH scores of the pooled reference.
    plans : dict[int, list[np.ndarray]]
        Positions into ``local_normal`` per size and draw.

    Returns
    -------
    dict[str, Any]
        The reproduction difference, the non-local AUROC and AP on the evaluation half, and one row per
        size, draw and option.
    """
    start = time.monotonic()
    with threadpool_limits(limits=1):
        reference = fit_mahalanobis(fit_x)
        scores = mahalanobis_scores(reference, x)
        difference = float(np.max(np.abs(scores - saved_pooled) / np.maximum(1.0, np.abs(saved_pooled))))
        evaluation_x, evaluation_y = x[evaluation], y[evaluation]
        nonlocal_metrics = metrics(evaluation_y, scores[evaluation])
        rows = []
        for m, draws in plans.items():
            for draw, positions in enumerate(draws):
                local = x[local_normal[positions]].astype(np.float64)
                local_model = fit_mahalanobis(local, local_components(m))
                combined_model = fit_mahalanobis(np.concatenate([fit_x, local]))
                fitted = {
                    "local_only": mahalanobis_scores(local_model, evaluation_x),
                    "local_plus_pooled": mahalanobis_scores(combined_model, evaluation_x),
                    "recentered": recentered_scores(reference, evaluation_x, local.mean(axis=0)),
                }
                for option, values in fitted.items():
                    rows.append({"encoder": name, "m": m, "draw": draw, "option": option,
                                 **metrics(evaluation_y, values)})
            print(json.dumps({"stage": f"part_b:{name}:{m}", "seconds": round(time.monotonic() - start)}),
                  flush=True)
    return {"reproduction_difference": difference, "nonlocal": nonlocal_metrics, "rows": rows,
            "seconds": time.monotonic() - start}


def summarize_b(draws: pd.DataFrame, nonlocal_auroc: dict[str, float]) -> list[dict[str, Any]]:
    """
    Summary of the Part B draws per encoder, size and option.

    Parameters
    ----------
    draws : pd.DataFrame
        Every Part B row.
    nonlocal_auroc : dict[str, float]
        Evaluation-half AUROC of the non-local reference per encoder.

    Returns
    -------
    list[dict[str, Any]]
        Means, percentiles and the share of draws beating the non-local reference.
    """
    summary = []
    for (name, m, option), group in draws.groupby(["encoder", "m", "option"], sort=False):
        gain = group["auroc"] - nonlocal_auroc[name]
        summary.append({"encoder": name, "m": int(m), "option": option, "draws": len(group),
                        "auroc_mean": float(group["auroc"].mean()),
                        "auroc_p5": float(group["auroc"].quantile(0.05)),
                        "auroc_p95": float(group["auroc"].quantile(0.95)),
                        "average_precision_mean": float(group["average_precision"].mean()),
                        "gain_mean": float(gain.mean()), "share_beating_nonlocal": float((gain > 0).mean())})
    return summary


def main() -> None:
    """Run Part A and Part B once and write the aggregate result."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 029 v1 has already run")
    started = time.monotonic()
    result022b, arrays = prior_readouts()
    evaluation, split_counts = site_split(arrays)
    points, reproduction_a = source_points(result022b, arrays)
    y = arrays["sph_labels"]
    local_y, evaluation_y = y[~evaluation], y[evaluation]
    plans = draw_plans(local_y)
    print(json.dumps({"stage": "part_a_start", "split": split_counts, "source_difference": reproduction_a}),
          flush=True)

    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = []
        for name in ENCODERS:
            logits = {}
            for readout in READOUTS:
                values = head_logits(arrays[f"sph_{name}_{readout}"])
                logits[f"{readout}:local"] = values[~evaluation]
                logits[f"{readout}:evaluation"] = values[evaluation]
            heads = {key: value for key, value in points.items() if key.endswith(f":{name}")}
            futures.append(executor.submit(part_a_encoder, name, logits, local_y, evaluation_y, heads, plans))
        draws_a = pd.DataFrame([row for future in futures for row in future.result()])
    summary_a = summarize_a(draws_a)

    fit_x, sph_features, sph_scores, identity_b = pooled_normals(arrays["sph_ecg_ids"])
    local_normal = np.flatnonzero(~evaluation & (y == 0))
    normal_plans = {m: [np.sort(np.random.default_rng([SEED, 3, m, draw]).choice(len(local_normal), m,
                                                                                  replace=False))
                        for draw in range(DRAWS_B)] for m in NORMAL_SIZES}
    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {name: executor.submit(part_b_encoder, name, fit_x[name], sph_features[name], evaluation,
                                         local_normal, y, sph_scores[f"{name}_pooled"].to_numpy(),
                                         normal_plans)
                   for name in ENCODERS}
        outputs_b = {name: future.result() for name, future in futures.items()}
    reproduction_b = {name: output["reproduction_difference"] for name, output in outputs_b.items()}
    if max(reproduction_b.values()) > TOLERANCE:
        raise ValueError(f"The pooled reference does not reproduce 026b: {reproduction_b}")
    draws_b = pd.DataFrame([row for output in outputs_b.values() for row in output["rows"]])
    nonlocal_auroc = {name: output["nonlocal"]["auroc"] for name, output in outputs_b.items()}
    summary_b = summarize_b(draws_b, nonlocal_auroc)
    references_b = {name: {
        "nonlocal_pooled": outputs_b[name]["nonlocal"],
        "ptbxl_026": metrics(evaluation_y, sph_scores[f"{name}_ptbxl"].to_numpy()[evaluation]),
        **{f"supervised_{readout}": metrics(evaluation_y, arrays[f"sph_{name}_{readout}"][evaluation])
           for readout in READOUTS}} for name in ENCODERS}

    primary_select = {"plan": "site", "readout": PRIMARY_READOUT, "encoder": PRIMARY_ENCODER,
                      "option": PRIMARY_OPTION}
    primary_a = smallest_size(summary_a, primary_select, "share_sensitivity_at_least_095", "n")
    primary_b = smallest_size(summary_b, {"encoder": PRIMARY_ENCODER, "option": "local_plus_pooled"},
                              "share_beating_nonlocal", "m")
    at_b = [row for row in summary_b
            if row["encoder"] == PRIMARY_ENCODER and row["option"] == "local_plus_pooled"
            and row["m"] == primary_b["smallest"]]
    secondary_a = [
        {"plan": plan, "readout": readout, "encoder": name, "option": option,
         **smallest_size(summary_a, {"plan": plan, "readout": readout, "encoder": name, "option": option},
                         "share_sensitivity_at_least_095", "n")}
        for plan in ("site", "low") for readout in READOUTS for name in ENCODERS for option in OPTIONS]
    secondary_b = [{"encoder": name, "option": option,
                    **smallest_size(summary_b, {"encoder": name, "option": option},
                                    "share_beating_nonlocal", "m")}
                   for name in ENCODERS for option in NORMAL_OPTIONS]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ecg_id": arrays["sph_ecg_ids"], "patient_id": arrays["sph_patient_ids"], "primary": y,
                  "evaluation": evaluation}).to_csv(OUTPUT / "split.csv", index=False)
    draws_a.to_csv(OUTPUT / "part_a_draws.csv", index=False)
    draws_b.to_csv(OUTPUT / "part_b_draws.csv", index=False)
    written = ("split.csv", "part_a_draws.csv", "part_b_draws.csv")
    result = {
        "status": "complete",
        "identity": {"experiment022b_result_sha256": sha256_file(PRIOR022B / "result.json"),
                     "experiment022b_predictions_sha256": sha256_file(PRIOR022B / "predictions.npz"),
                     "sph_rows_sha256": sha256_file(SPH_ROWS), **identity_b,
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "seed": SEED, "split": split_counts,
        "part_a": {"sizes": list(SIZES), "low_sizes": list(LOW_SIZES), "low_prevalence": LOW_PREVALENCE,
                   "draws": DRAWS_A, "source_max_difference_from_022b": reproduction_a, "summary": summary_a},
        "part_b": {"sizes": list(NORMAL_SIZES), "draws": DRAWS_B,
                   "local_components": {m: local_components(m) for m in NORMAL_SIZES},
                   "pooled_reference_max_relative_difference_from_026b": reproduction_b,
                   "references": references_b, "summary": summary_b,
                   "seconds": {name: output["seconds"] for name, output in outputs_b.items()}},
        "reading": {"primary_a": primary_a, "primary_b": primary_b,
                    "primary_b_gain_negligible": (at_b[0]["gain_mean"] < NEGLIGIBLE_AUROC) if at_b else None,
                    "secondary_a": secondary_a, "secondary_b": secondary_b},
        "outputs_sha256": {name: sha256_file(OUTPUT / name) for name in written},
        "challenge_test_read": False, "ptbxl_calibration_as_test": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "primary_a": primary_a, "primary_b": primary_b}), flush=True)


if __name__ == "__main__":
    main()
