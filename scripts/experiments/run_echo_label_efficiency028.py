"""Experiment 028: label efficiency of frozen encoders for echo-confirmed structural heart disease."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.echo_label_efficiency import (
    Plan,
    budget_summary,
    check_keys,
    draw_plans,
    gain_target,
    smallest_budget_by_draws,
    split_rows,
    wide_metric,
)
from ecg_experiment.echonext_readout import age_sex_inputs
from ecg_experiment.eda.echonext import COMPOSITE
from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.full_development import fit_logistic, patient_bootstrap, predict
from ecg_experiment.label_efficiency import fit_logistic_c, select_c, smallest_budget

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment028_echo_label_efficiency_v1"
PRIOR = ROOT / "outputs/experiment023_echonext_v3"
ROWS = ROOT / "data/processed/echonext_250hz_v1/rows.csv"
ENCODERS = ("xecg", "jepa", "cpc", "age_sex")
BUDGETS = (100, 250, 500, 1000, 2000, 4000, 8000, 16000)
READ_BUDGETS = ("250", "1000")
READOUTS = ("primary", "secondary")
DRAWS = 20
SEED = 28028
BOOTSTRAP_DRAWS = 2000
WORKERS = 4
PROBABILITY_TOLERANCE = 1e-8
AUROC_TOLERANCE = 1e-12
NEAR_CEILING = 0.95
ROW_COLUMNS = ["patient_key", COMPOSITE, "age_at_ecg", "sex"]
CONTRASTS = (("xecg", "cpc"), ("jepa", "cpc"), ("xecg", "jepa"), ("cpc", "age_sex"), ("jepa", "age_sex"),
             ("xecg", "age_sex"))
SOURCES = (
    "ecg_experiment/echo_label_efficiency.py", "ecg_experiment/label_efficiency.py",
    "ecg_experiment/echonext_readout.py", "ecg_experiment/eda/echonext.py",
    "ecg_experiment/full_development.py", "ecg_experiment/files.py",
    "scripts/experiments/run_echo_label_efficiency028.py", "pyproject.toml", "uv.lock",
    "docs/experiment-028-echo-label-efficiency.md",
)
INPUTS = {"features": PRIOR / "features.npz", "predictions": PRIOR / "predictions.npz",
          "prior_result": PRIOR / "result.json", "rows": ROWS}


def identity() -> dict[str, Any]:
    """
    Hash every input and require the cached files to match the Experiment 023 receipt.

    Returns
    -------
    dict[str, Any]
        Input and source hashes.

    Raises
    ------
    ValueError
        If the features, predictions or rows differ from what Experiment 023 recorded.
    """
    hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    prior = read_json(PRIOR / "result.json")
    expected = {"features": prior["outputs_sha256"]["features.npz"],
                "predictions": prior["outputs_sha256"]["predictions.npz"],
                "rows": prior["identity"]["echonext"]["rows"]}
    for name, digest in expected.items():
        if hashes[name] != digest:
            raise ValueError(f"Input differs from the Experiment 023 receipt: {name}")
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    return {"inputs": hashes, "sources": sources, "verified_against_experiment023": sorted(expected)}


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Training pool, all validation rows and usable validation rows, checked against the feature keys.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Output of ``split_rows``.
    """
    rows = pd.read_csv(ROWS, index_col="ecg_key", keep_default_na=False, na_values=[""])
    train, val_all, val = split_rows(rows)
    with np.load(PRIOR / "features.npz") as saved:
        check_keys(saved["train_ecg_keys"], saved["val_all_ecg_keys"], train, val_all)
    return train, val_all, val


def inputs(name: str, train: pd.DataFrame, val: pd.DataFrame,
           val_use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool and evaluation inputs of one encoder.

    Parameters
    ----------
    name : str
        Encoder name.
    train, val : pd.DataFrame
        Training pool and usable validation rows.
    val_use : np.ndarray
        Usable mask over all validation rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pool and evaluation inputs.
    """
    if name == "age_sex":
        return age_sex_inputs(train), age_sex_inputs(val)
    with np.load(PRIOR / "features.npz") as saved:
        return saved[f"train_{name}"], saved[f"val_all_{name}"][val_use]


def fit_readout(readout: str, x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray, seed: int,
                x_eval: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
    """
    Fit one readout on a drawn subset and score the evaluation rows.

    Parameters
    ----------
    readout : str
        ``primary`` (C = 0.01) or ``secondary`` (C by grouped cross-validation).
    x_train, y_train : np.ndarray
        Training inputs and labels.
    groups : np.ndarray
        Patient ID of each training row.
    seed : int
        Fold seed of the secondary readout.
    x_eval : np.ndarray
        Evaluation inputs.

    Returns
    -------
    tuple[np.ndarray, float, list[float]]
        Probabilities, the C used and the cross-validated AUROC per grid C (empty for primary).
    """
    if readout == "primary":
        return predict(fit_logistic(x_train, y_train), x_eval), 0.01, []
    chosen, means = select_c(x_train, y_train, groups, seed)
    return predict(fit_logistic_c(x_train, y_train, chosen), x_eval), chosen, means


def evaluate(name: str, plans: list[Plan], train: pd.DataFrame, val: pd.DataFrame, val_use: np.ndarray,
             readouts: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], float]:
    """
    Fit the requested readouts of one encoder on every given draw and score the evaluation rows.

    Parameters
    ----------
    name : str
        Encoder name.
    plans : list[Plan]
        Draws to fit.
    train, val : pd.DataFrame
        Training pool and usable validation rows.
    val_use : np.ndarray
        Usable mask over all validation rows.
    readouts : tuple[str, ...]
        Readouts to fit.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, np.ndarray], float]
        Per-draw rows, the probabilities of the last draw per readout, and the seconds taken.
    """
    start = time.monotonic()
    x_pool, x_eval = inputs(name, train, val, val_use)
    y_pool = train[COMPOSITE].to_numpy(dtype=np.int64)
    patients = train["patient_key"].to_numpy()
    y_eval = val[COMPOSITE].to_numpy(dtype=np.int64)
    rows, last = [], {}
    with threadpool_limits(limits=1):
        for budget, draw, seed, positions in plans:
            y_train = y_pool[positions]
            subset_hash = hashlib.sha256("\n".join(map(str, train.index[positions])).encode()).hexdigest()
            for readout in readouts:
                probability, c, cv_means = fit_readout(readout, x_pool[positions], y_train,
                                                       patients[positions], seed, x_eval)
                last[readout] = probability
                rows.append({
                    "encoder": name, "readout": readout, "budget": budget, "draw": draw, "seed": seed,
                    "records": len(positions), "positives": int(y_train.sum()),
                    "patients": len(np.unique(patients[positions])), "c": c,
                    "cv_auroc": json.dumps(cv_means) if cv_means else "",
                    "auroc": float(roc_auc_score(y_eval, probability)),
                    "average_precision": float(average_precision_score(y_eval, probability)),
                    "subset_sha256": subset_hash,
                })
    return rows, last, time.monotonic() - start


def reproduction(finals: dict[str, np.ndarray], val: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Require the N = all primary heads to reproduce Experiment 023.

    Parameters
    ----------
    finals : dict[str, np.ndarray]
        Primary probabilities at N = all per encoder.
    val : pd.DataFrame
        Usable validation rows.

    Returns
    -------
    dict[str, dict[str, float]]
        Maximum probability difference, AUROC and 023 AUROC per encoder.

    Raises
    ------
    ValueError
        If the validation order, a probability or an AUROC differs from Experiment 023.
    """
    prior = read_json(PRIOR / "result.json")["by_head"]
    y = val[COMPOSITE].to_numpy(dtype=np.int64)
    checks = {}
    with np.load(PRIOR / "predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["ecg_keys"], val.index.to_numpy()):
            raise ValueError("Experiment 023 validation order changed")
        for name, probability in finals.items():
            difference = float(np.abs(probability - saved[name]).max())
            auroc = float(roc_auc_score(y, probability))
            checks[name] = {"max_abs_difference": difference, "auroc": auroc,
                            "experiment023_auroc": prior[name]["auroc"]}
            if difference > PROBABILITY_TOLERANCE or abs(auroc - prior[name]["auroc"]) > AUROC_TOLERANCE:
                raise ValueError(f"{name} does not reproduce Experiment 023: {checks[name]}")
    return checks


def budget_targets(table: pd.DataFrame, per_budget: dict[str, dict], full: dict[str, dict],
                   tabular: float) -> dict[str, dict[str, float | int | None]]:
    """
    Smallest budgets reaching the prespecified targets, per encoder.

    Parameters
    ----------
    table : pd.DataFrame
        Per-draw rows of one readout.
    per_budget : dict[str, dict]
        ``budget_summary`` per budget label.
    full : dict[str, dict]
        N = all metrics per encoder.
    tabular : float
        Experiment 023 ``tabular`` AUROC.

    Returns
    -------
    dict[str, dict[str, float | int | None]]
        Targets and the smallest budget reaching each, per encoder.
    """
    labels = [str(size) for size in BUDGETS]
    targets = {}
    for name in ENCODERS:
        means = {int(b): per_budget[b]["encoders"][name]["auroc"]["mean"] for b in labels}
        draws = {int(b): wide_metric(table, b, "auroc")[name].to_numpy() for b in labels}
        share = NEAR_CEILING * full[name]["auroc"]
        gain = gain_target(full[name]["auroc"], NEAR_CEILING)
        targets[name] = {
            "share_target": share, "smallest_budget_share": smallest_budget(means, share),
            "gain_target": gain, "smallest_budget_gain": smallest_budget(means, gain),
            "smallest_budget_mean_tabular": smallest_budget(means, tabular),
            "smallest_budget_draws_tabular": smallest_budget_by_draws(draws, tabular),
        }
    return targets


def analyse(table: pd.DataFrame, finals: dict[str, np.ndarray], val: pd.DataFrame,
            tabular: float) -> dict[str, Any]:
    """
    Budget summaries, contrasts, bootstrap at N = all and the prespecified reading of one readout.

    Parameters
    ----------
    table : pd.DataFrame
        Per-draw rows of one readout.
    finals : dict[str, np.ndarray]
        Probabilities at N = all per encoder.
    val : pd.DataFrame
        Usable validation rows.
    tabular : float
        Experiment 023 ``tabular`` AUROC.

    Returns
    -------
    dict[str, Any]
        The readout's aggregate results.
    """
    per_budget = {str(size): budget_summary(table, str(size), ENCODERS, CONTRASTS) for size in BUDGETS}
    counts = table.groupby(["budget", "encoder", "c"]).size()
    chosen_c = {budget: {name: {str(c): int(n) for c, n in counts[budget][name].items()} for name in ENCODERS}
                for budget in [*per_budget, "all"]}
    at_all = table[table["budget"] == "all"].set_index("encoder")
    full = {name: {"auroc": float(at_all.loc[name, "auroc"]),
                   "average_precision": float(at_all.loc[name, "average_precision"]),
                   "c": float(at_all.loc[name, "c"])} for name in ENCODERS}
    y = val[COMPOSITE].to_numpy(dtype=np.int64)
    patients = val["patient_key"].to_numpy()
    bootstrap = {f"{first}_minus_{second}": patient_bootstrap(
        patients, y, finals[first], finals[second], draws=BOOTSTRAP_DRAWS, seed=SEED)
        for first, second in CONTRASTS}
    return {"per_budget": per_budget, "chosen_c": chosen_c, "all": full, "all_bootstrap": bootstrap,
            "reading": {budget: per_budget[budget]["reading"] for budget in READ_BUDGETS},
            "targets": budget_targets(table, per_budget, full, tabular)}


def counts(train: pd.DataFrame, val_all: pd.DataFrame, val: pd.DataFrame) -> dict[str, int | float]:
    """
    Row, patient and positive counts of the pool and the evaluation set.

    Parameters
    ----------
    train, val_all, val : pd.DataFrame
        Training pool, all validation rows and usable validation rows.

    Returns
    -------
    dict[str, int | float]
        Aggregate counts and the pool prevalence.
    """
    return {"pool_records": len(train), "pool_patients": int(train["patient_key"].nunique()),
            "pool_positives": int(train[COMPOSITE].sum()), "pool_prevalence": float(train[COMPOSITE].mean()),
            "val_records": len(val_all), "evaluation_records": len(val),
            "evaluation_patients": int(val["patient_key"].nunique()),
            "evaluation_positives": int(val[COMPOSITE].sum())}


def phase_two(executor: ProcessPoolExecutor, by_budget: dict[str, list[Plan]], train: pd.DataFrame,
              val: pd.DataFrame, val_use: np.ndarray) -> dict[tuple[str, str], tuple]:
    """
    Fit every remaining draw and readout, largest budgets first.

    Parameters
    ----------
    executor : ProcessPoolExecutor
        Worker pool.
    by_budget : dict[str, list[Plan]]
        Draws per budget label.
    train, val : pd.DataFrame
        Training pool and usable validation rows.
    val_use : np.ndarray
        Usable mask over all validation rows.

    Returns
    -------
    dict[tuple[str, str], tuple]
        ``evaluate`` output per encoder and budget.
    """
    order = ["all", *[str(size) for size in reversed(BUDGETS)]]
    futures = {(name, budget): executor.submit(evaluate, name, by_budget[budget], train, val, val_use,
                                               ("secondary",) if budget == "all" else READOUTS)
               for budget in order for name in ENCODERS}
    return {key: future.result() for key, future in futures.items()}


def main() -> None:
    """Run the frozen validation-only EchoNext label-efficiency study once."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 028 v1 has already run")
    start = time.monotonic()
    run_identity = identity()
    train_rows, val_all, val_rows = load_rows()
    train, val = train_rows[ROW_COLUMNS], val_rows[ROW_COLUMNS]
    val_use = val_all["use"].to_numpy()
    plans = draw_plans(train["patient_key"].to_numpy(), train[COMPOSITE].to_numpy(dtype=np.int64), BUDGETS,
                       DRAWS, SEED)
    by_budget = {label: [plan for plan in plans if plan[0] == label] for label in [*map(str, BUDGETS), "all"]}
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        first = {name: executor.submit(evaluate, name, by_budget["all"], train, val, val_use, ("primary",))
                 for name in ENCODERS}
        reproduced = {name: future.result() for name, future in first.items()}
        checks = reproduction({name: last["primary"] for name, (_, last, _) in reproduced.items()}, val)
        print(json.dumps({"stage": "reproduced", "checks": checks, "seconds": time.monotonic() - start}),
              flush=True)
        remaining = phase_two(executor, by_budget, train, val, val_use)
    outputs = {(name, "all", "primary"): result for name, result in reproduced.items()}
    outputs.update({(name, budget, "rest"): result for (name, budget), result in remaining.items()})
    table = pd.DataFrame([row for rows, _, _ in outputs.values() for row in rows])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "draws.csv", index=False)
    finals = {"primary": {name: reproduced[name][1]["primary"] for name in ENCODERS},
              "secondary": {name: remaining[(name, "all")][1]["secondary"] for name in ENCODERS}}
    tabular = float(read_json(PRIOR / "result.json")["by_head"]["tabular"]["auroc"])
    result = {
        "status": "complete_validation_only", "identity": run_identity, "reproduction": checks,
        "counts": counts(train_rows, val_all, val_rows), "budgets": list(BUDGETS), "draws": DRAWS,
        "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "experiment023_tabular_auroc": tabular,
        "readouts": {readout: analyse(table[table["readout"] == readout], finals[readout], val, tabular)
                     for readout in READOUTS},
        "task_seconds": {"/".join(key): seconds for key, (_, _, seconds) in outputs.items()},
        "draws_sha256": sha256_file(OUTPUT / "draws.csv"), "total_seconds": time.monotonic() - start,
        "test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "total_seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
