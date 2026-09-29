"""Experiment 030: an operating point set by a referral budget on local normal ECGs (SPH as the new site)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.referral_budget import (
    bootstrap_counts,
    budget_threshold,
    count_above,
    group_sums,
    referral_share,
    referrals_per_1000,
    wilson_interval,
)
from scripts.experiments.run_local_adaptation029 import prior_readouts, site_split

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment030_referral_budget_v1"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR026B = ROOT / "outputs/experiment026b_multisource_manifold_v1"
PRIOR029 = ROOT / "outputs/experiment029_local_adaptation_v1"
SPH_ROWS = ROOT / "data/processed/sph_clean_v1/rows.csv"
ENCODERS = ("xecg", "jepa", "cpc")
READOUTS = ("pooled", "ptbxl")
SCORES = (*READOUTS, "normal_ref")
BUDGETS = (10, 20, 50, 100)
SIZES = (50, 100, 200, 500, 1000)
EXTRA_SIZES = (2000,)
DRAWS = 200
SEED = 30030
BOOTSTRAP_SEED = 34034
RESAMPLES = 2000
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
GROUPS = ("positive", "normal", *SUPERCLASSES)
PREVALENCES = (0.02, 0.05)
BAND = 0.01
NARROW_BAND = 0.005
SHARE = 0.90
PRIMARY_SCORE, PRIMARY_ENCODER, PRIMARY_BUDGET, PRIMARY_SIZE = "pooled", "xecg", 50, 200
SECONDARY_BUDGETS = (20, 100)
FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
EXPECTED_SOURCE_NORMALS = {"supervised": 1923, "normal_ref": 1707}
EXPECTED_EVALUATION = {"positive": 3584, "normal": 6895, "MI": 122, "STTC": 2507, "CD": 1184, "HYP": 117}
SOURCES = (
    "ecg_experiment/referral_budget.py", "ecg_experiment/local_adaptation.py",
    "scripts/experiments/run_referral_budget030.py", "scripts/experiments/run_local_adaptation029.py",
    "pyproject.toml", "uv.lock", "docs/experiment-030-referral-budget.md",
)


def checked_csv(directory: Path, name: str, **options: Any) -> pd.DataFrame:
    """
    Read an output table of an earlier experiment after checking it against that run's receipt.

    Parameters
    ----------
    directory : Path
        Output directory of the earlier run, with a ``result.json``.
    name : str
        File name of the table.
    **options : Any
        Passed to ``pandas.read_csv``.

    Returns
    -------
    pd.DataFrame
        The table.

    Raises
    ------
    ValueError
        If the file's hash differs from the receipt.
    """
    receipt = json.loads((directory / "result.json").read_text())
    if sha256_file(directory / name) != receipt["outputs_sha256"][name]:
        raise ValueError(f"{directory.name}/{name} differs from its receipt")
    return pd.read_csv(directory / name, **options)


def load_site() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """
    Load 022b's predictions, 029's split and SPH's superclass flags, with every check of the protocol.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, Any]]
        022b's arrays, the evaluation mask, the label masks of the evaluation half (``GROUPS``) and the
        split counts.

    Raises
    ------
    ValueError
        If the split differs from 029's or a count differs from the protocol.
    """
    _, arrays = prior_readouts()
    evaluation, counts = site_split(arrays)
    split = checked_csv(PRIOR029, "split.csv", dtype={"ecg_id": str})
    if not (np.array_equal(split["ecg_id"].to_numpy(), arrays["sph_ecg_ids"])
            and np.array_equal(split["evaluation"].to_numpy(dtype=bool), evaluation)):
        raise ValueError("Split differs from 029's split.csv")
    rows = pd.read_csv(SPH_ROWS, dtype={"ecg_id": str, "patient_id": str})
    rows = rows[rows["use_evaluation"].astype(str).eq("True") & rows["primary"].notna()]
    if not np.array_equal(rows["ecg_id"].to_numpy(), arrays["sph_ecg_ids"]):
        raise ValueError("SPH rows differ from 022b's order")
    y = arrays["sph_labels"][evaluation]
    masks = {"positive": y == 1, "normal": y == 0,
             **{name: rows[name].to_numpy()[evaluation] == 1 for name in SUPERCLASSES}}
    found = {name: int(mask.sum()) for name, mask in masks.items()}
    if found != EXPECTED_EVALUATION:
        raise ValueError(f"Evaluation counts differ from the protocol: {found}")
    return arrays, evaluation, masks, counts


def load_heads(arrays: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """
    SPH scores and source normal scores of every score and encoder.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        022b's arrays.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], pd.DataFrame]
        Per ``score:encoder``: ``sph`` scores and ``source`` normal scores; and 026b's Challenge scores.

    Raises
    ------
    ValueError
        If a row order or a source-normal count differs from the protocol.
    """
    sph_scores = checked_csv(PRIOR026B, "sph_scores.csv", dtype={"ecg_id": str}, float_precision="round_trip")
    challenge = checked_csv(PRIOR026B, "challenge_scores.csv", dtype={"record": str},
                            float_precision="round_trip")
    outside_ptbxl = arrays["calibration_families"] != "ptbxl"
    if not (np.array_equal(sph_scores["ecg_id"].to_numpy(), arrays["sph_ecg_ids"])
            and np.array_equal(challenge["record"].to_numpy(), arrays["calibration_records"][outside_ptbxl])
            and np.array_equal(challenge["primary"].to_numpy(dtype=np.int64),
                               arrays["calibration_labels"][outside_ptbxl])):
        raise ValueError("026b score orders differ from 022b")
    normal = arrays["calibration_labels"] == 0
    heads = {}
    for name in ENCODERS:
        for readout in READOUTS:
            heads[f"{readout}:{name}"] = {"sph": arrays[f"sph_{name}_{readout}"],
                                          "source": arrays[f"calibration_{name}_{readout}"][normal]}
        heads[f"normal_ref:{name}"] = {"sph": sph_scores[f"{name}_pooled"].to_numpy(),
                                       "source": challenge[f"{name}_pooled"].to_numpy()[
                                           challenge["primary"].to_numpy() == 0]}
    counts = {"supervised": len(heads["pooled:xecg"]["source"]),
              "normal_ref": len(heads["normal_ref:xecg"]["source"])}
    if counts != EXPECTED_SOURCE_NORMALS:
        raise ValueError(f"Source normal counts differ from the protocol: {counts}")
    return heads, challenge


def score_head(key: str, head: dict[str, Any], evaluation: np.ndarray, local_normal: np.ndarray,
               masks: dict[str, np.ndarray], plans: dict[int, list[np.ndarray]]
               ) -> tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]:
    """
    Thresholds, achieved rates and sensitivities of one score for every size, draw and budget.

    Parameters
    ----------
    key : str
        ``score:encoder``.
    head : dict[str, Any]
        SPH and source normal scores.
    evaluation : np.ndarray
        Evaluation mask per SPH ECG.
    local_normal : np.ndarray
        SPH positions of the local pool's normals.
    masks : dict[str, np.ndarray]
        Label masks of the evaluation half.
    plans : dict[int, list[np.ndarray]]
        Positions into ``local_normal`` per size and draw.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]
        One row per size, draw and budget (size 0 is the source quantile), and the referral share over
        draws of each evaluation ECG per (size, budget).
    """
    score, name = key.split(":")
    evaluation_scores = head["sph"][evaluation]
    local_scores = head["sph"][local_normal]
    ordered = {group: np.sort(evaluation_scores[mask]) for group, mask in masks.items()}
    samples = {0: [head["source"]], **{m: [local_scores[positions] for positions in draws]
                                       for m, draws in plans.items()}}
    rows, shares = [], {}
    for m, draws in samples.items():
        for budget in BUDGETS:
            thresholds = np.array([budget_threshold(normals, budget) for normals in draws])
            rates = {group: count_above(values, thresholds) / len(values)
                     for group, values in ordered.items()}
            shares[(m, budget)] = referral_share(evaluation_scores, thresholds)
            for draw, threshold in enumerate(thresholds):
                rows.append({"score": score, "encoder": name, "m": m, "draw": draw, "budget": budget / 1000,
                             "threshold": float(threshold), "rate": float(rates["normal"][draw]),
                             "sensitivity": float(rates["positive"][draw]),
                             **{f"sensitivity_{group}": float(rates[group][draw]) for group in SUPERCLASSES}})
    return rows, shares


def bootstrap(shares: dict[tuple[str, int, int], np.ndarray], masks: dict[str, np.ndarray],
              patients: np.ndarray) -> dict[str, np.ndarray]:
    """
    Patient-bootstrap distribution of the draw-averaged rate of every group and key.

    Parameters
    ----------
    shares : dict[tuple[str, int, int], np.ndarray]
        Referral share of each evaluation ECG per (``score:encoder``, size, budget).
    masks : dict[str, np.ndarray]
        Label masks of the evaluation half.
    patients : np.ndarray
        Patient ID of each evaluation ECG.

    Returns
    -------
    dict[str, np.ndarray]
        Per group, a ``(RESAMPLES, keys)`` array in the order of ``shares``.

    Raises
    ------
    ValueError
        If a resample holds no ECG of a group.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    counts = bootstrap_counts(len(unique), RESAMPLES, BOOTSTRAP_SEED)
    matrix = np.column_stack(list(shares.values()))
    result = {}
    for group, mask in masks.items():
        denominator = counts @ group_sums(codes, mask[:, None].astype(np.float64), len(unique))
        if (denominator == 0).any():
            raise ValueError(f"A resample holds no {group} ECG")
        result[group] = counts @ group_sums(codes, matrix * mask[:, None], len(unique)) / denominator
    return result


def summarize(draws: pd.DataFrame, boot: dict[str, np.ndarray], keys: list[tuple[str, int, int]]
              ) -> list[dict[str, Any]]:
    """
    Summary per score, encoder, size and budget: rates over draws with patient-bootstrap intervals.

    Parameters
    ----------
    draws : pd.DataFrame
        Every draw row.
    boot : dict[str, np.ndarray]
        Output of ``bootstrap``.
    keys : list[tuple[str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    list[dict[str, Any]]
        One summary row per key.
    """
    columns = {"normal": "rate", "positive": "sensitivity",
               **{group: f"sensitivity_{group}" for group in SUPERCLASSES}}
    groups = draws.groupby(["score", "encoder", "m", "budget"], sort=False)
    summary = []
    for index, (key, m, budget) in enumerate(keys):
        score, name = key.split(":")
        group = groups.get_group((score, name, m, budget / 1000))
        target = budget / 1000
        distance = (group["rate"] - target).abs()
        row = {"score": score, "encoder": name, "m": m, "budget": target, "draws": len(group),
               "threshold_mean": float(group["threshold"].mean()),
               "share_rate_within_1pp": float((distance <= BAND + 1e-12).mean()),
               "share_rate_within_05pp": float((distance <= NARROW_BAND + 1e-12).mean())}
        for label, column in columns.items():
            values = group[column]
            low, high = np.percentile(boot[label][:, index], [2.5, 97.5])
            row.update({f"{column}_mean": float(values.mean()), f"{column}_p5": float(values.quantile(0.05)),
                        f"{column}_p95": float(values.quantile(0.95)),
                        f"{column}_ci": [float(low), float(high)]})
        for prevalence in PREVALENCES:
            per_1000 = referrals_per_1000(group["sensitivity"], group["rate"], prevalence)
            tag = f"{round(prevalence * 100)}pct"
            caught = 1000 * prevalence * group["sensitivity"].mean()
            row.update({f"referrals_per_1000_at_{tag}": float(per_1000.mean()),
                        f"caught_per_1000_at_{tag}": float(caught),
                        f"missed_per_1000_at_{tag}": float(1000 * prevalence - caught)})
        summary.append(row)
    return summary


def find(summary: list[dict[str, Any]], **select: Any) -> dict[str, Any]:
    """
    Return the one summary row matching every selected field.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Summary rows.
    **select : Any
        Field values to match.

    Returns
    -------
    dict[str, Any]
        The row.
    """
    (row,) = [row for row in summary if all(row[key] == value for key, value in select.items())]
    return row


def point(summary: list[dict[str, Any]], key: str, m: int, budget: int) -> dict[str, Any]:
    """
    Return the summary row of one ``score:encoder``, size and budget.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Summary rows.
    key : str
        ``score:encoder``.
    m : int
        Number of local normals (0 for the source quantile).
    budget : int
        Budget in thousandths.

    Returns
    -------
    dict[str, Any]
        The row.
    """
    score, name = key.split(":")
    return find(summary, score=score, encoder=name, m=m, budget=budget / 1000)


def reading(summary: list[dict[str, Any]], boot: dict[str, np.ndarray], keys: list[tuple[str, int, int]]
            ) -> dict[str, Any]:
    """
    Answers to the pre-registered questions and the paired secondary contrasts.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Output of ``summarize``.
    boot : dict[str, np.ndarray]
        Output of ``bootstrap``.
    keys : list[tuple[str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    dict[str, Any]
        ``smallest_m`` and ``sensitivity`` per budget and score, and the contrasts.
    """
    answers = {}
    for score in SCORES:
        for name in ENCODERS:
            for budget in BUDGETS:
                select = {"score": score, "encoder": name, "budget": budget / 1000}
                shares = {m: find(summary, m=m, **select)["share_rate_within_1pp"]
                          for m in SIZES + EXTRA_SIZES}
                reached = [m for m in SIZES if shares[m] >= SHARE - 1e-12]
                at = find(summary, m=PRIMARY_SIZE, **select)
                answers[f"{score}:{name}:{budget}"] = {
                    "smallest_m": reached[0] if reached else None,
                    "largest_share_to_1000": max(shares[m] for m in SIZES), "shares": shares,
                    "sensitivity_at_200": {key: at[f"sensitivity_{key}"]
                                           for key in ("mean", "p5", "p95", "ci")},
                    "rate_at_200": at["rate_mean"]}
    columns = {key: index for index, key in enumerate(keys)}
    contrasts = []
    pairs = (("m200_minus_source", ("pooled:xecg", 200), ("pooled:xecg", 0)),
             ("pooled_minus_ptbxl", ("pooled:xecg", 200), ("ptbxl:xecg", 200)),
             ("pooled_minus_normal_ref", ("pooled:xecg", 200), ("normal_ref:xecg", 200)),
             ("xecg_minus_jepa", ("pooled:xecg", 200), ("pooled:jepa", 200)),
             ("xecg_minus_cpc", ("pooled:xecg", 200), ("pooled:cpc", 200)))
    for budget in BUDGETS:
        for label, (key_a, m_a), (key_b, m_b) in pairs:
            first, second = columns[(key_a, m_a, budget)], columns[(key_b, m_b, budget)]
            for group in ("positive", "normal"):
                difference = boot[group][:, first] - boot[group][:, second]
                column = "sensitivity_mean" if group == "positive" else "rate_mean"
                contrasts.append({"contrast": label, "budget": budget / 1000,
                                  "outcome": "sensitivity" if group == "positive" else "rate",
                                  "difference": (point(summary, key_a, m_a, budget)[column]
                                                 - point(summary, key_b, m_b, budget)[column]),
                                  "ci": [float(value) for value in np.percentile(difference, [2.5, 97.5])]})
    prefix = f"{PRIMARY_SCORE}:{PRIMARY_ENCODER}"
    primary = answers[f"{prefix}:{PRIMARY_BUDGET}"]
    return {"A": {key: primary[key] for key in ("smallest_m", "largest_share_to_1000", "shares")},
            "B": primary["sensitivity_at_200"],
            "C": {budget: answers[f"{prefix}:{budget}"] for budget in SECONDARY_BUDGETS},
            "all": answers, "contrasts": contrasts}


def family_check(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """
    Sensitivity and specificity in each Challenge family's calibration group with a source quantile.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        022b's arrays.

    Returns
    -------
    list[dict[str, Any]]
        One row per family, encoder, setting and budget, with Wilson intervals.
    """
    families, y = arrays["calibration_families"], arrays["calibration_labels"]
    rows = []
    for family in FAMILIES:
        members = families == family
        outside = (families != family) & (y == 0)
        for name in ENCODERS:
            pooled = arrays[f"calibration_{name}_pooled"]
            held_out = arrays[f"calibration_{name}_loso_{family}"]
            settings = {"pooled_all_source": (pooled, pooled[y == 0]),
                        "pooled_other_sites": (pooled, pooled[outside]),
                        "loso_other_sites": (held_out, held_out[outside])}
            for setting, (scores, normals) in settings.items():
                for budget in BUDGETS:
                    referred = scores[members] > budget_threshold(normals, budget)
                    positive = y[members] == 1
                    positives, normals_here = int(positive.sum()), int((~positive).sum())
                    caught = int(referred[positive].sum())
                    passed = normals_here - int(referred[~positive].sum())
                    rows.append({"family": family, "encoder": name, "setting": setting,
                                 "budget": budget / 1000, "source_normals": len(normals),
                                 "positives": positives, "normals": normals_here,
                                 "sensitivity": caught / positives,
                                 "sensitivity_ci": wilson_interval(caught, positives),
                                 "specificity": passed / normals_here,
                                 "specificity_ci": wilson_interval(passed, normals_here)})
    return rows


def main() -> None:
    """Run the experiment once and write the aggregate result."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 030 v1 has already run")
    started = time.monotonic()
    arrays, evaluation, masks, split_counts = load_site()
    heads, _ = load_heads(arrays)
    y = arrays["sph_labels"]
    local_normal = np.flatnonzero(~evaluation & (y == 0))
    plans = {m: [np.sort(np.random.default_rng([SEED, m, draw]).choice(len(local_normal), m, replace=False))
                 for draw in range(DRAWS)] for m in SIZES + EXTRA_SIZES}
    print(json.dumps({"stage": "loaded", "split": split_counts, "local_normals": len(local_normal)}),
          flush=True)

    with threadpool_limits(limits=4):
        rows, shares = [], {}
        for key, head in heads.items():
            head_rows, head_shares = score_head(key, head, evaluation, local_normal, masks, plans)
            rows.extend(head_rows)
            shares.update({(key, m, budget): values for (m, budget), values in head_shares.items()})
            print(json.dumps({"stage": f"scored:{key}", "seconds": round(time.monotonic() - started)}),
                  flush=True)
        draws = pd.DataFrame(rows)
        keys = list(shares)
        boot = bootstrap(shares, masks, arrays["sph_patient_ids"][evaluation])
        print(json.dumps({"stage": "bootstrap", "seconds": round(time.monotonic() - started)}), flush=True)
    summary = summarize(draws, boot, keys)
    for row, (key, m, budget) in zip(summary, keys, strict=True):
        point = shares[(key, m, budget)][masks["positive"]].mean()
        if abs(point - row["sensitivity_mean"]) > 1e-9:
            raise ValueError(f"Referral shares do not reproduce the draw mean for {key}, {m}, {budget}")
    answers = reading(summary, boot, keys)
    families = family_check(arrays)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUTPUT / "draws.csv", index=False)
    pd.DataFrame(families).to_csv(OUTPUT / "families.csv", index=False)
    written = ("draws.csv", "families.csv")
    result = {
        "status": "complete",
        "identity": {"experiment022b_result_sha256": sha256_file(PRIOR022B / "result.json"),
                     "experiment022b_predictions_sha256": sha256_file(PRIOR022B / "predictions.npz"),
                     "experiment026b_result_sha256": sha256_file(PRIOR026B / "result.json"),
                     "experiment029_result_sha256": sha256_file(PRIOR029 / "result.json"),
                     "sph_rows_sha256": sha256_file(SPH_ROWS),
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "draws": DRAWS,
        "sizes": list(SIZES), "extra_sizes": list(EXTRA_SIZES), "budgets": [b / 1000 for b in BUDGETS],
        "split": split_counts, "evaluation_counts": EXPECTED_EVALUATION,
        "source_normals": EXPECTED_SOURCE_NORMALS, "local_normals": len(local_normal),
        "summary": summary, "reading": answers, "families": families,
        "outputs_sha256": {name: sha256_file(OUTPUT / name) for name in written},
        "challenge_test_read": False, "ptbxl_calibration_as_test": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"], "A": answers["A"],
                      "B": answers["B"]}), flush=True)


if __name__ == "__main__":
    main()
