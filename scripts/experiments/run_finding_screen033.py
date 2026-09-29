"""Experiment 033: a screen that adds the PVC and pre-excitation heads to the binary readout, one budget."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.finding_screen import (
    clipped_logits,
    extend_split,
    finding_z,
    referred_matrix,
    split_thresholds,
)
from ecg_experiment.referral_budget import bootstrap_counts, group_sums
from ecg_experiment.rhythm_findings import (
    GROUPS,
    budget_sensitivity,
    clear_normal,
    frequent_pvc,
    label_table,
)
from ecg_experiment.sph import base_codes
from scripts.experiments.run_referral_budget030 import checked_csv, load_site

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment033_finding_heads_screen_v1"
PRIOR030 = ROOT / "outputs/experiment030_referral_budget_v1"
PRIOR032 = ROOT / "outputs/experiment032_rhythm_findings_v1"
SPH_ROWS = ROOT / "data/processed/sph_clean_v1/rows.csv"
ENCODER = "xecg"
BUDGETS = (20, 50, 100)
SIZES = (200, 1000)
PRIMARY_BUDGET, PRIMARY_SIZE = 50, 200
DRAWS = 200
DRAW_SEED = 30030
SPLIT_SEED = 33033
BOOTSTRAP_SEED = 37037
RESAMPLES = 2000
MAX_BINARY_COST = 0.010
MAX_RATE_INCREASE = 0.0025
TOLERANCE = 1e-12
FINDINGS = ("ventricular_ectopy", "preexcitation", "af_flutter", "high_grade_av_block", "long_qt")
RULES: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "binary": ((), ()),
    "combined_50": (("combined",), (50,)),
    "combined_100": (("combined",), (100,)),
    "combined_200": (("combined",), (200,)),
    "combined_300": (("combined",), (300,)),
    "separate_50_50": (("pvc", "wpw"), (50, 50)),
    "separate_100_100": (("pvc", "wpw"), (100, 100)),
    "separate_200_100": (("pvc", "wpw"), (200, 100)),
}
CANDIDATES = tuple(name for name in RULES if name != "binary")
OUTCOMES = ("normal", "binary_positive", "composite", "pvc", "frequent_pvc", "wpw", "af_flutter",
            "high_grade_av_block", "long_qt", "finding_only", "other")
CONTRASTED = ("normal", "binary_positive", "composite", "pvc", "frequent_pvc", "wpw", "finding_only", "other")
LOCAL_OUTCOMES = ("binary_positive", "composite")
EXPECTED: dict[str, Any] = {
    "evaluation": {"ecgs": 12759, "patients": 12320, "outside_030": 2280, "normal": 6895,
                   "binary_positive": 3584, "composite": 4052, "pvc": 531, "frequent_pvc": 183, "wpw": 15,
                   "af_flutter": 370, "high_grade_av_block": 12, "long_qt": 10, "finding_only": 468,
                   "other": 1812},
    "local": {"ecgs": 12818, "patients": 12322, "outside_030": 2289, "normal": 6923, "binary_positive": 3606,
              "composite": 4090, "pvc": 527, "frequent_pvc": 189, "wpw": 12, "af_flutter": 392,
              "high_grade_av_block": 15, "long_qt": 14, "finding_only": 484, "other": 1805},
    "outside_030": {"ecgs": 4569, "joined_patient": 204, "new_patients": 4278},
}
SOURCES = (
    "ecg_experiment/finding_screen.py", "ecg_experiment/hybrid_score.py", "ecg_experiment/referral_budget.py",
    "ecg_experiment/rhythm_findings.py", "ecg_experiment/local_adaptation.py", "ecg_experiment/sph.py",
    "scripts/experiments/run_finding_screen033.py", "scripts/experiments/run_referral_budget030.py",
    "scripts/experiments/run_local_adaptation029.py", "pyproject.toml", "uv.lock",
    "docs/experiment-033-finding-heads-screen.md",
)


def load_rows() -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    """
    SPH rows with finding labels, 030's split extended to every SPH evaluation ECG, and 030's inputs.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any], np.ndarray]
        The 25,577 rows with ``in030``, ``evaluation`` and one boolean column per outcome; 030's arrays,
        evaluation mask and label masks (``site``); and the local normals as row positions, in 030's order.

    Raises
    ------
    ValueError
        If a row order or hash differs from its receipt, or a patient is in both halves.
    """
    arrays, evaluation030, masks030, _ = load_site()
    receipt = json.loads((PRIOR032 / "result.json").read_text())
    if sha256_file(SPH_ROWS) != receipt["identity"]["sph_manifest"]["rows"]:
        raise ValueError("SPH manifest differs from Experiment 032's")
    rows = pd.read_csv(SPH_ROWS, dtype={"ecg_id": str, "patient_id": str, "aha_code": str})
    rows = rows[rows["use_evaluation"].astype(str).eq("True")].reset_index(drop=True)
    rows = pd.concat([rows, label_table([base_codes(text) for text in rows["aha_code"]], "sph")], axis=1)
    in030 = rows["primary"].notna().to_numpy()
    if not (np.array_equal(rows.loc[in030, "ecg_id"].to_numpy(), arrays["sph_ecg_ids"])
            and np.array_equal(clear_normal(rows["primary"].to_numpy(), rows[list(GROUPS)]),
                               (rows["primary"] == 0).to_numpy())):
        raise ValueError("SPH rows or normals differ from Experiment 030's")
    primary = rows["primary"].to_numpy()
    finding = {group: (rows[group] == 1).to_numpy() for group in FINDINGS}
    composite = (primary == 1) | np.any(list(finding.values()), axis=0)
    normal = primary == 0
    known_evaluation = np.zeros(len(rows), dtype=bool)
    known_evaluation[in030] = evaluation030
    evaluation = extend_split(rows["patient_id"].to_numpy(), in030, known_evaluation, composite, SPLIT_SEED)
    patients = rows["patient_id"].to_numpy()
    if set(patients[evaluation]) & set(patients[~evaluation]):
        raise ValueError("A patient is in both halves")
    frequent = rows["aha_code"].map(frequent_pvc).to_numpy() & finding["ventricular_ectopy"]
    outcomes = {"normal": normal, "binary_positive": primary == 1, "composite": composite,
                "pvc": finding["ventricular_ectopy"], "frequent_pvc": frequent,
                "wpw": finding["preexcitation"],
                "af_flutter": finding["af_flutter"], "high_grade_av_block": finding["high_grade_av_block"],
                "long_qt": finding["long_qt"], "finding_only": composite & (primary != 1),
                "other": ~composite & ~normal}
    rows = rows.assign(in030=in030, evaluation=evaluation, **outcomes)
    local030 = np.flatnonzero(~evaluation030 & (arrays["sph_labels"] == 0))
    local_normal = np.flatnonzero(in030)[local030]
    if not np.array_equal(np.sort(local_normal), np.flatnonzero(~evaluation & normal)):
        raise ValueError("Local normals differ from Experiment 030's")
    return rows, {"arrays": arrays, "evaluation": evaluation030, "masks": masks030}, local_normal


def counts_of(rows: pd.DataFrame) -> dict[str, Any]:
    """
    ECG and patient counts of each half and outcome.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``load_rows``.

    Returns
    -------
    dict[str, Any]
        Counts per half, and how the ECGs outside 030's split were assigned.
    """
    counts: dict[str, Any] = {}
    for half, mask in (("evaluation", rows["evaluation"]), ("local", ~rows["evaluation"])):
        part = rows[mask]
        counts[half] = {"ecgs": len(part), "patients": int(part["patient_id"].nunique()),
                        "outside_030": int((~part["in030"]).sum()),
                        **{name: int(part[name].sum()) for name in OUTCOMES}}
    outside = rows[~rows["in030"]]
    known_patients = set(rows.loc[rows["in030"], "patient_id"])
    joined = outside["patient_id"].isin(known_patients)
    counts["outside_030"] = {"ecgs": len(outside), "joined_patient": int(joined.sum()),
                             "new_patients": int(outside.loc[~joined, "patient_id"].nunique())}
    return counts


def load_scores(rows: pd.DataFrame, site: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Binary readout and finding z-scores of every SPH row from 032's saved heads, with the reproduction checks.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``load_rows``.
    site : dict[str, Any]
        030's arrays, evaluation mask and label masks.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        ``binary``, ``pvc``, ``wpw`` and ``combined`` scores; and the check values.

    Raises
    ------
    ValueError
        If a file differs from its receipt or a reproduction fails.
    """
    receipt = json.loads((PRIOR032 / "result.json").read_text())
    if sha256_file(PRIOR032 / "predictions.npz") != receipt["outputs_sha256"]["predictions.npz"]:
        raise ValueError("032 predictions differ from their receipt")
    with np.load(PRIOR032 / "predictions.npz") as saved:
        heads = {name: saved[name] for name in saved.files}
    calibration = checked_csv(PRIOR032, "calibration_rows.csv", dtype={"record": str})
    if not (np.array_equal(heads["sph_ecg_ids"], rows["ecg_id"].to_numpy(dtype=str))
            and np.array_equal(heads["sph_patient_ids"], rows["patient_id"].to_numpy(dtype=str))
            and np.array_equal(heads["calibration_records"],
                               (calibration["source"] + ":" + calibration["record"]).to_numpy(dtype=str))):
        raise ValueError("032 row orders differ")
    binary = heads[f"sph_{ENCODER}_binary"]
    pooled = site["arrays"][f"sph_{ENCODER}_pooled"]
    difference = float(np.abs(binary[rows["in030"].to_numpy()] - pooled).max())
    if difference > TOLERANCE:
        raise ValueError(f"032's binary readout differs from 022b's by {difference}")
    standard = np.where(calibration["evaluable"].astype(bool), calibration["primary"], np.nan)
    source_normal = clear_normal(standard, calibration[list(GROUPS)])
    z = {}
    clipped = {}
    for name, group in (("pvc", "ventricular_ectopy"), ("wpw", "preexcitation")):
        sph = heads[f"sph_{ENCODER}_{group}"]
        source = heads[f"calibration_{ENCODER}_{group}"][source_normal]
        z[name] = finding_z(sph, source)
        clipped[name] = {"sph": clipped_logits(sph)[1], "source": clipped_logits(source)[1]}
    scores = {"binary": binary, **z, "combined": np.maximum(z["pvc"], z["wpw"])}
    reproduced = {}
    normal = rows["normal"].to_numpy()
    for group, key in (("ventricular_ectopy", "ventricular_ectopy"), ("ventricular_ectopy", "binary"),
                       ("preexcitation", "preexcitation"), ("preexcitation", "binary")):
        defined = rows[group].notna().to_numpy()
        values = heads[f"sph_{ENCODER}_{key}"][defined]
        found = budget_sensitivity(values, rows.loc[defined, group].to_numpy(), normal[defined], 50)
        saved = receipt["evaluations"][f"{group}@sph"]["scores"][f"{key}:{ENCODER}"]
        saved = saved["sensitivity_at_50"]["value"]
        if abs(found - saved) > TOLERANCE:
            raise ValueError(f"032's {key} sensitivity for {group} does not reproduce")
        reproduced[f"{key}@{group}"] = found
    checks = {"binary_difference_from_022b": difference, "source_normals": int(source_normal.sum()),
              "clipped": clipped, "sensitivity_at_50_032": reproduced}
    return scores, checks


def run_rules(scores: dict[str, np.ndarray], rows: pd.DataFrame, local_normal: np.ndarray
              ) -> tuple[pd.DataFrame, dict[tuple[str, int, int], np.ndarray]]:
    """
    Thresholds and outcomes of every rule, size, budget and draw.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Output of ``load_scores``.
    rows : pd.DataFrame
        Output of ``load_rows``.
    local_normal : np.ndarray
        Row positions of the local normals, in 030's order.

    Returns
    -------
    tuple[pd.DataFrame, dict[tuple[str, int, int], np.ndarray]]
        One row per rule, size, budget and draw; and the referral share over draws of each evaluation ECG
        per (rule, size, budget).
    """
    evaluation = rows["evaluation"].to_numpy()
    evaluation_masks = {name: rows.loc[evaluation, name].to_numpy() for name in OUTCOMES}
    local_masks = {name: rows.loc[~evaluation, name].to_numpy() for name in LOCAL_OUTCOMES}
    plans = {m: [local_normal[np.sort(np.random.default_rng([DRAW_SEED, m, draw])
                                      .choice(len(local_normal), m, replace=False))]
                 for draw in range(DRAWS)] for m in SIZES}
    records, shares = [], {}
    for rule, (columns, rule_shares) in RULES.items():
        matrix = np.column_stack([scores["binary"], *(scores[column] for column in columns)])
        for m, draws in plans.items():
            for budget in BUDGETS:
                thresholds = np.array([split_thresholds(matrix[positions], budget, rule_shares)
                                       for positions in draws])
                referred = referred_matrix(matrix[evaluation], thresholds)
                local = referred_matrix(matrix[~evaluation], thresholds)
                shares[(rule, m, budget)] = referred.mean(axis=1)
                found = {name: referred[mask].mean(axis=0) for name, mask in evaluation_masks.items()}
                found_local = {name: local[mask].mean(axis=0) for name, mask in local_masks.items()}
                for draw in range(DRAWS):
                    records.append({
                        "rule": rule, "m": m, "budget": budget / 1000, "draw": draw,
                        **{f"threshold_{index}": float(value)
                           for index, value in enumerate(thresholds[draw])},
                        **{name: float(values[draw]) for name, values in found.items()},
                        **{f"local_{name}": float(values[draw]) for name, values in found_local.items()}})
    return pd.DataFrame(records), shares


def check_030(draws: pd.DataFrame) -> float:
    """
    Require the binary rule's draws to equal 030's pooled xECG draws.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_rules``.

    Returns
    -------
    float
        Largest absolute difference of threshold, rate or sensitivity.

    Raises
    ------
    ValueError
        If any difference exceeds the tolerance.
    """
    prior = checked_csv(PRIOR030, "draws.csv", float_precision="round_trip")
    prior = prior[(prior["score"] == "pooled") & (prior["encoder"] == ENCODER) & prior["m"].isin(SIZES)
                  & prior["budget"].isin([budget / 1000 for budget in BUDGETS])]
    ours = draws[draws["rule"] == "binary"]
    merged = ours.merge(prior, on=["m", "budget", "draw"], validate="one_to_one")
    if len(merged) != len(ours) or len(ours) != len(SIZES) * len(BUDGETS) * DRAWS:
        raise ValueError("030's draws do not match")
    largest = float(max((merged["threshold_0"] - merged["threshold"]).abs().max(),
                        (merged["normal"] - merged["rate"]).abs().max(),
                        (merged["binary_positive"] - merged["sensitivity"]).abs().max()))
    if largest > TOLERANCE:
        raise ValueError(f"The binary rule differs from 030 by {largest}")
    return largest


def bootstrap(shares: dict[tuple[str, int, int], np.ndarray], rows: pd.DataFrame
              ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    Patient-bootstrap distribution of the draw-averaged referral share of every outcome and key.

    Parameters
    ----------
    shares : dict[tuple[str, int, int], np.ndarray]
        Referral share of each evaluation ECG per key.
    rows : pd.DataFrame
        Output of ``load_rows``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, int]]
        Per outcome a ``(RESAMPLES, keys)`` array (NaN in a resample without that outcome), and the number
        of such resamples per outcome.
    """
    part = rows[rows["evaluation"]]
    unique, codes = np.unique(part["patient_id"].to_numpy(), return_inverse=True)
    counts = bootstrap_counts(len(unique), RESAMPLES, BOOTSTRAP_SEED)
    matrix = np.column_stack(list(shares.values()))
    result, empty = {}, {}
    for name in OUTCOMES:
        mask = part[name].to_numpy()
        denominator = counts @ group_sums(codes, mask[:, None].astype(np.float64), len(unique))
        numerator = counts @ group_sums(codes, matrix * mask[:, None], len(unique))
        with np.errstate(invalid="ignore", divide="ignore"):
            result[name] = np.where(denominator > 0, numerator / denominator, np.nan)
        empty[name] = int((denominator[:, 0] == 0).sum())
    return result, empty


def interval(values: np.ndarray) -> list[float]:
    """
    Take the 2.5th and 97.5th percentiles over the resamples that hold the outcome.

    Parameters
    ----------
    values : np.ndarray
        One value per resample, NaN where undefined.

    Returns
    -------
    list[float]
        Lower and upper bounds.
    """
    return [float(value) for value in np.nanpercentile(values, [2.5, 97.5])]


def summarize(draws: pd.DataFrame, boot: dict[str, np.ndarray], keys: list[tuple[str, int, int]]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Per-key summaries and the paired contrasts of every rule against the binary readout alone.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_rules``.
    boot : dict[str, np.ndarray]
        Output of ``bootstrap``.
    keys : list[tuple[str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Summary rows and contrast rows.
    """
    groups = draws.groupby(["rule", "m", "budget"], sort=False)
    column = {key: index for index, key in enumerate(keys)}
    summary, contrasts = [], []
    for rule, m, budget in keys:
        group = groups.get_group((rule, m, budget / 1000))
        row: dict[str, Any] = {"rule": rule, "m": m, "budget": budget / 1000}
        here, reference = column[(rule, m, budget)], column[("binary", m, budget)]
        for name in OUTCOMES:
            values = group[name]
            row[name] = {"mean": float(values.mean()), "p5": float(values.quantile(0.05)),
                         "p95": float(values.quantile(0.95)), "ci": interval(boot[name][:, here])}
        for name in LOCAL_OUTCOMES:
            row[f"local_{name}"] = float(group[f"local_{name}"].mean())
        summary.append(row)
        if rule == "binary":
            continue
        base = groups.get_group(("binary", m, budget / 1000))
        for name in CONTRASTED:
            difference = boot[name][:, here] - boot[name][:, reference]
            contrasts.append({"rule": rule, "m": m, "budget": budget / 1000, "outcome": name,
                              "difference": float(group[name].mean() - base[name].mean()),
                              "ci": interval(difference)})
    return summary, contrasts


def decide(summary: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Select the candidate on the local pool and apply the pre-registered decision rule on the evaluation half.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Output of ``summarize``.
    contrasts : list[dict[str, Any]]
        Output of ``summarize``.

    Returns
    -------
    dict[str, Any]
        The local selection table, the selected rule, its primary contrasts and the decision.
    """
    at = {row["rule"]: row for row in summary
          if row["m"] == PRIMARY_SIZE and row["budget"] == PRIMARY_BUDGET / 1000}
    base = at["binary"]["local_binary_positive"]
    table = [{"rule": rule, "local_composite": at[rule]["local_composite"],
              "local_binary_cost": base - at[rule]["local_binary_positive"]} for rule in CANDIDATES]
    eligible = [row for row in table if row["local_binary_cost"] <= MAX_BINARY_COST + TOLERANCE]
    if eligible:
        best = max(row["local_composite"] for row in eligible)
        selected = next(row["rule"] for row in eligible if row["local_composite"] >= best - TOLERANCE)
    else:
        cheapest = min(row["local_binary_cost"] for row in table)
        selected = next(row["rule"] for row in table if row["local_binary_cost"] <= cheapest + TOLERANCE)
    primary = {row["outcome"]: row for row in contrasts
               if row["rule"] == selected and row["m"] == PRIMARY_SIZE
               and row["budget"] == PRIMARY_BUDGET / 1000}
    gain = primary["composite"]
    cost = -primary["binary_positive"]["difference"]
    rate = primary["normal"]["difference"]
    adopted = bool(gain["ci"][0] > 0 and cost <= MAX_BINARY_COST and rate <= MAX_RATE_INCREASE)
    return {"local_selection": table, "selected": selected, "selected_by_local_cost": bool(eligible),
            "composite_gain": gain["difference"], "composite_gain_ci": gain["ci"],
            "binary_cost": cost, "binary_cost_ci": [-primary["binary_positive"]["ci"][1],
                                                    -primary["binary_positive"]["ci"][0]],
            "rate_difference": rate, "rate_difference_ci": primary["normal"]["ci"],
            "decision": "adopt" if adopted else "not adopted"}


def main() -> None:
    """Run the experiment once and write the aggregate result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", action="store_true", help="print the counts and stop before any score")
    arguments = parser.parse_args()
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 033 v1 has already run")
    started = time.monotonic()
    rows, site, local_normal = load_rows()
    counts = counts_of(rows)
    print(json.dumps({"stage": "counts", **counts, "local_normals": len(local_normal)}), flush=True)
    if arguments.counts:
        return
    if json.loads(json.dumps(counts)) != EXPECTED:
        raise ValueError("Counts differ from the protocol")

    with threadpool_limits(limits=3):
        scores, checks = load_scores(rows, site)
        print(json.dumps({"stage": "scores", **checks}), flush=True)
        draws, shares = run_rules(scores, rows, local_normal)
        checks["largest_difference_from_030_draws"] = check_030(draws)
        keys = list(shares)
        boot, empty = bootstrap(shares, rows)
        print(json.dumps({"stage": "bootstrap", "seconds": round(time.monotonic() - started),
                          "empty_resamples": empty}), flush=True)
    evaluation = rows["evaluation"].to_numpy()
    for (rule, m, budget), values in shares.items():
        mean = draws.loc[(draws["rule"] == rule) & (draws["m"] == m) & (draws["budget"] == budget / 1000),
                         "composite"].mean()
        if abs(values[rows.loc[evaluation, "composite"].to_numpy()].mean() - mean) > 1e-9:
            raise ValueError(f"Referral shares do not reproduce the draw mean for {rule}, {m}, {budget}")
    summary, contrasts = summarize(draws, boot, keys)
    reading = decide(summary, contrasts)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUTPUT / "draws.csv", index=False)
    result = {
        "status": "complete",
        "identity": {"experiment030_result_sha256": sha256_file(PRIOR030 / "result.json"),
                     "experiment032_result_sha256": sha256_file(PRIOR032 / "result.json"),
                     "experiment032_predictions_sha256": sha256_file(PRIOR032 / "predictions.npz"),
                     "sph_rows_sha256": sha256_file(SPH_ROWS),
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "encoder": ENCODER, "rules": {name: {"columns": list(columns), "shares_per_mille": list(values)}
                                      for name, (columns, values) in RULES.items()},
        "draw_seed": DRAW_SEED, "split_seed": SPLIT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "resamples": RESAMPLES, "draws": DRAWS, "sizes": list(SIZES),
        "budgets": [budget / 1000 for budget in BUDGETS], "counts": counts, "checks": checks,
        "empty_resamples": empty, "summary": summary, "contrasts": contrasts, "reading": reading,
        "outputs_sha256": {"draws.csv": sha256_file(OUTPUT / "draws.csv")},
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"], "reading": reading}),
          flush=True)


if __name__ == "__main__":
    main()
