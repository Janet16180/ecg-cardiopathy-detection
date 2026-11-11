"""
Stage 3 evaluation of MLC-02 from saved test predictions only.

Every metric comes from the tested scoring library (``mlc02_evaluation``, ``mlc02_bootstrap``); this module
only arranges the calls. For PTB-XL fold 10 and the SPH evaluation half:

- per candidate, raw probabilities: per-condition and macro AUROC, average precision with prevalence,
  sensitivity and specificity at the validation thresholds, calibration slope, calibration in the large and
  the joint-fit intercept, Brier score, the imprecise flag (< 100 positives) and the usability rule, with
  2,000 patient-cluster bootstrap intervals shared by every model of the set (``interval_report``);
- the same after the validation-fitted Platt calibration (no thresholds, which were fixed on raw scores);
- paired comparisons of every candidate against P0 (macro AUROC difference, per-condition differences with
  Holm over the entering conditions). Only the selected model against P0 on the SPH evaluation half is
  confirmatory;
- audits with point estimates: by sex, by flat or missing lead, one condition against several, and the
  derived multiclass view (top-1 and top-2 under both readings, exact set) on raw probabilities, with
  top-k after Platt beside it.

PTB-XL fold 10 also gets the likelihood sensitivity analysis (statements with likelihood below 50 undefined),
with its own resamples. The SPH calibration half is used only for the named secondary analysis: thresholds and
Platt refitted there and applied to the SPH evaluation half, compared with the validation-fitted ones on the
same resamples.

``standin_masks`` builds validation stand-ins for a dry run of the whole path; nothing it produces is a test
result.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import read_json, sha256_file, write_csv_atomic, write_json_atomic
from ecg_experiment.mlc02_bootstrap import (
    CALIBRATION,
    DRAWS,
    cluster_resamples,
    interval_report,
    paired_comparison,
    percentile_interval,
)
from ecg_experiment.mlc02_calibration import calibration_report, fit_platt_damped
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST, load_manifest, load_predictions, role_data, subset
from ecg_experiment.mlc02_evaluation import (
    apply_platt,
    cooccurrence_tables,
    lead_masks,
    multiclass_view,
    sex_masks,
    specificity_thresholds,
    subgroup_tables,
    top_k_accuracy,
)
from ecg_experiment.mlc02_stage3 import (
    STAGE3,
    TEST_RECEIPT_NAME,
    TEST_ROLES,
    code_state,
    json_ready,
    read_operating_points,
    read_record,
    receipt,
    stored,
    validation_data,
    validation_probabilities,
)
from ecg_experiment.paths import from_stored

PRIMARY_SETS = ("test_same_source", "test_external")
CONFIRMATORY_SET = "test_external"
SECONDARY_FIT_SET = "sph_calibration"
REFERENCE = "P0"
LIKELIHOOD_SET = "test_same_source"
WORKERS = 8
EVALUATION = "evaluation"
TABLE_KINDS = ("raw", "platt", "raw_calibration_damped", "platt_calibration_damped")
NOT_IMPLEMENTED = {
    "space_a_beside_key04": (
        "The candidates output the 14 space-B conditions only; KEY-04 scored space A (MI, STTC, CD, HYP). "
        "Space-A metrics need either a rule mapping the 14 outputs to the four groups (for example the "
        "maximum probability over each group's member conditions), fixed in a protocol addendum before "
        "test-predict, or space-A heads trained in Stage 2. With either, the call is "
        "interval_report(labels_a, scores, cluster_resamples(groups, labels_a), conditions=GROUPS_A) beside "
        "KEY-04's 0.9608."
    ),
    "sph_window_mean_longer_records": (
        "The waveform cache holds the first 10 s only. The analysis needs the full SPH recordings of the "
        "evaluation half read from the raw files, every consecutive 10 s window passed through each "
        "candidate's "
        "input pipeline (the three encoders and the measurements for the feature candidates, S-R and F on "
        "waveforms), and the window probabilities averaged per record, then metrics_table on the averages."
    ),
    "source_identification": "Validation-feature audit, done in Stage 2 (source_identification/).",
    "leave_one_family_out": "Train and validation audit, done in Stage 2 (lofo/); no test input.",
}


def load_test_predictions(
    output: Path,
    record: dict[str, Any],
    record_sha256: str,
    operating_sha256: str,
    sets: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """
    Load the saved test predictions of every candidate, checked against the test-predict receipt.

    Parameters
    ----------
    output : Path
        Stage 3 directory holding ``test_predictions_receipt.json``.
    record : dict[str, Any]
        Selection record.
    record_sha256, operating_sha256 : str
        SHA-256 of the record and of the operating points, which must be those the receipt pinned.
    sets : dict[str, dict[str, np.ndarray]]
        Manifest arrays of each test role.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]
        Candidate id to role to probabilities, and the receipt.

    Raises
    ------
    ValueError
        If the record, the operating points or a prediction file differ from the receipt.
    """
    pinned = read_json(Path(output) / TEST_RECEIPT_NAME)
    if (
        pinned["selection_record_sha256"] != record_sha256
        or pinned["operating_points_sha256"] != operating_sha256
    ):
        raise ValueError("the selection record or operating points changed after test-predict")
    result: dict[str, dict[str, np.ndarray]] = {}
    for row in record["candidates"]:
        result[row["id"]] = {}
        for role in TEST_ROLES:
            entry = pinned["files"][row["id"]][role]
            path = from_stored(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{path} differs from the test-predict receipt")
            result[row["id"]][role] = load_predictions(path, sets[role]["record_key"])["probability"]
    return result, pinned


def set_summary(data: dict[str, np.ndarray], resamples: dict[str, Any]) -> dict[str, Any]:
    """
    Size and resampling identity of one evaluation set.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Manifest arrays of the set.
    resamples : dict[str, Any]
        Output of ``cluster_resamples``.

    Returns
    -------
    dict[str, Any]
        Records, groups, sources, draws, redraws, seed, fingerprint, label SHA-256 and entering conditions.
    """
    return {
        "records": len(data["record_key"]),
        "sources": sorted(set(data["source"])),
        "groups": resamples["groups"],
        "draws": resamples["draws"],
        "redraws": resamples["redraws"],
        "seed": resamples["seed"],
        "fingerprint": resamples["fingerprint"],
        "labels_sha256": resamples["labels_sha256"],
        "entering": [name for name, flag in zip(CONDITIONS, resamples["entering"], strict=True) if flag],
    }


def sample_differences(
    first: dict[str, Any], second: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, Any]:
    """
    Per-condition differences of resampled metrics, first minus second, with percentile intervals.

    Parameters
    ----------
    first, second : dict[str, Any]
        ``bootstrap_metrics`` outputs on the same resamples.
    metrics : tuple[str, ...]
        Metric names present in both.

    Returns
    -------
    dict[str, Any]
        Metric to condition to ``ci_low`` and ``ci_high`` of the difference.

    Raises
    ------
    ValueError
        If the resamples differ.
    """
    if first["fingerprint"] != second["fingerprint"]:
        raise ValueError("differences need the same resamples")
    result = {}
    for metric in metrics:
        low, high, _ = percentile_interval(first["samples"][metric] - second["samples"][metric])
        result[metric] = {
            name: {"ci_low": float(low[c]), "ci_high": float(high[c])}
            for c, name in enumerate(CONDITIONS)
            if first["entering"][c]
        }
    return result


def damped_differences(
    first: dict[str, Any],
    second: dict[str, Any],
    first_samples: np.ndarray,
    second_samples: np.ndarray,
    entering: np.ndarray,
) -> dict[str, Any]:
    """
    Differences of damped-Newton calibration fits, first minus second, observed and bootstrapped.

    Parameters
    ----------
    first, second : dict[str, Any]
        ``calibration_report`` reports of the same set.
    first_samples, second_samples : np.ndarray
        Their resampled values ``[draws, conditions, 2]`` on the same resamples.
    entering : np.ndarray
        Conditions that enter the set's macro.

    Returns
    -------
    dict[str, Any]
        Metric to condition to ``difference``, ``ci_low`` and ``ci_high``.
    """
    result = {}
    for index, metric in enumerate(("calibration_intercept", "calibration_slope")):
        low, high, _ = percentile_interval(first_samples[:, :, index] - second_samples[:, :, index])
        result[metric] = {
            name: {
                "difference": first["conditions"][c][metric] - second["conditions"][c][metric],
                "ci_low": float(low[c]),
                "ci_high": float(high[c]),
            }
            for c, name in enumerate(CONDITIONS)
            if entering[c]
        }
    return result


def library_failures(report: dict[str, Any]) -> dict[str, Any]:
    """
    Count where the library's undamped calibration fit gave NaN, for the conditions entering the macro.

    Parameters
    ----------
    report : dict[str, Any]
        One candidate's output of ``primary_set``.

    Returns
    -------
    dict[str, Any]
        Per probability kind (``raw``, ``platt``): conditions with a NaN point slope, and the summed NaN
        resamples of the slope.
    """
    result = {}
    for kind in ("raw", "platt"):
        rows = [row for row in report[kind]["conditions"] if row["enters_macro"]]
        result[kind] = {
            "point_slope_nan": [
                row["condition"] for row in rows if not np.isfinite(row["calibration_slope"])
            ],
            "slope_nan_draws": int(sum(row["calibration_slope_nan_draws"] or 0 for row in rows)),
        }
    return result


def table_difference(
    first: dict[str, Any], second: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, Any]:
    """
    Observed per-condition differences of two ``interval_report`` tables, first minus second.

    Parameters
    ----------
    first, second : dict[str, Any]
        ``interval_report`` tables of the same set.
    metrics : tuple[str, ...]
        Metric names present in both.

    Returns
    -------
    dict[str, Any]
        Metric to condition to the difference.
    """
    return {
        metric: {
            a["condition"]: a[metric] - b[metric]
            for a, b in zip(first["conditions"], second["conditions"], strict=True)
        }
        for metric in metrics
    }


def primary_set(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    operating: dict[str, dict[str, np.ndarray]],
    resamples: dict[str, Any],
    workers: int,
) -> dict[str, dict[str, Any]]:
    """
    Raw and validation-Platt interval reports of every candidate on one set.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states of the set.
    predictions : dict[str, np.ndarray]
        Candidate id to raw probabilities.
    operating : dict[str, dict[str, np.ndarray]]
        Candidate id to validation ``thresholds`` and ``platt``.
    resamples : dict[str, Any]
        Output of ``cluster_resamples`` for the set.
    workers : int
        Threads.

    Returns
    -------
    dict[str, dict[str, Any]]
        Candidate id to ``raw`` and ``platt`` tables, their damped-Newton calibration reports
        ``raw_calibration_damped`` and ``platt_calibration_damped``, the bootstrap outputs ``raw_boot`` and
        ``platt_boot``, and ``platt_damped_samples``.
    """

    def job(name: str) -> dict[str, Any]:
        calibrated = apply_platt(predictions[name], operating[name]["platt"])
        raw, raw_boot = interval_report(labels, predictions[name], resamples, operating[name]["thresholds"])
        platt, platt_boot = interval_report(labels, calibrated, resamples, None)
        raw_damped, raw_damped_samples = calibration_report(labels, predictions[name], resamples)
        platt_damped, platt_damped_samples = calibration_report(labels, calibrated, resamples)
        return {
            "raw": raw,
            "platt": platt,
            "raw_calibration_damped": raw_damped,
            "platt_calibration_damped": platt_damped,
            "raw_boot": raw_boot,
            "platt_boot": platt_boot,
            "platt_damped_samples": platt_damped_samples,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(predictions, pool.map(job, predictions), strict=True))


def audits(
    data: dict[str, np.ndarray], probability: np.ndarray, operating: dict[str, np.ndarray]
) -> dict[str, Any]:
    """
    Point-estimate audits of one candidate on one set.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Manifest arrays of the set.
    probability : np.ndarray
        Raw probabilities.
    operating : dict[str, np.ndarray]
        Validation ``thresholds`` and ``platt``.

    Returns
    -------
    dict[str, Any]
        ``sex``, ``flat_leads``, ``cooccurrence``, ``multiclass`` (raw) and ``top_k_platt``.
    """
    labels, thresholds = data["labels"], operating["thresholds"]
    calibrated = apply_platt(probability, operating["platt"])
    return {
        "sex": subgroup_tables(labels, probability, sex_masks(data["sex"]), thresholds),
        "flat_leads": subgroup_tables(labels, probability, lead_masks(data["lead_flag"]), thresholds),
        "cooccurrence": cooccurrence_tables(labels, probability, thresholds),
        "multiclass": multiclass_view(labels, probability, thresholds),
        "top_k_platt": {
            f"top_{k}": {
                reading: top_k_accuracy(labels, calibrated, k, require)
                for reading, require in (("all_defined", True), ("undefined_allowed", False))
            }
            for k in (1, 2)
        },
    }


def secondary_analysis(
    fit: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    fit_predictions: dict[str, np.ndarray],
    target_predictions: dict[str, np.ndarray],
    primary: dict[str, dict[str, Any]],
    resamples: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    """
    Thresholds and Platt refitted on the SPH calibration half, applied to the SPH evaluation half.

    Parameters
    ----------
    fit : dict[str, np.ndarray]
        Manifest arrays of the SPH calibration half.
    target : dict[str, np.ndarray]
        Manifest arrays of the SPH evaluation half.
    fit_predictions, target_predictions : dict[str, np.ndarray]
        Candidate id to raw probabilities on each half.
    primary : dict[str, dict[str, Any]]
        Output of ``primary_set`` on the evaluation half (validation-fitted reports and bootstraps).
    resamples : dict[str, Any]
        Resamples of the evaluation half.
    workers : int
        Threads.

    Returns
    -------
    dict[str, Any]
        Candidate id to ``local_thresholds``, ``local_platt`` (damped Newton), ``at_local_thresholds``,
        ``after_local_platt`` and ``after_local_platt_calibration_damped`` tables, and
        ``local_minus_validation`` (observed differences and their bootstrap intervals for sensitivity,
        specificity and the calibration fits, library and damped).
    """

    def job(name: str) -> dict[str, Any]:
        thresholds = specificity_thresholds(fit["labels"], fit_predictions[name])
        platt = fit_platt_damped(fit["labels"], fit_predictions[name])
        calibrated = apply_platt(target_predictions[name], platt)
        at_threshold, threshold_boot = interval_report(
            target["labels"], target_predictions[name], resamples, thresholds, calibration=False
        )
        after_platt, platt_boot = interval_report(target["labels"], calibrated, resamples, None)
        damped, damped_samples = calibration_report(target["labels"], calibrated, resamples)
        rates = ("sensitivity", "specificity")
        return {
            "local_thresholds": dict(zip(CONDITIONS, thresholds.tolist(), strict=True)),
            "local_platt": {
                c: {"intercept": float(a), "slope": float(b)}
                for c, (a, b) in zip(CONDITIONS, platt, strict=True)
            },
            "at_local_thresholds": at_threshold,
            "after_local_platt": after_platt,
            "after_local_platt_calibration_damped": damped,
            "local_minus_validation": {
                "observed": table_difference(at_threshold, primary[name]["raw"], rates)
                | table_difference(after_platt, primary[name]["platt"], CALIBRATION),
                "interval": sample_differences(threshold_boot, primary[name]["raw_boot"], rates)
                | sample_differences(platt_boot, primary[name]["platt_boot"], CALIBRATION),
                "calibration_damped": damped_differences(
                    damped,
                    primary[name]["platt_calibration_damped"],
                    damped_samples,
                    primary[name]["platt_damped_samples"],
                    resamples["entering"],
                ),
            },
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(target_predictions, pool.map(job, target_predictions), strict=True))


def evaluate_sets(
    sets: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    operating: dict[str, dict[str, np.ndarray]],
    selected: str,
    draws: int = DRAWS,
    workers: int = WORKERS,
) -> dict[str, Any]:
    """
    Every Stage 3 quantity that depends on test predictions.

    Parameters
    ----------
    sets : dict[str, dict[str, np.ndarray]]
        Manifest arrays of ``test_same_source``, ``test_external`` and ``sph_calibration``.
    predictions : dict[str, dict[str, np.ndarray]]
        Candidate id to role to raw probabilities; must include ``REFERENCE`` and ``selected``.
    operating : dict[str, dict[str, np.ndarray]]
        Candidate id to validation ``thresholds`` and ``platt``.
    selected : str
        Selected model of the record.
    draws : int
        Bootstrap resamples per set.
    workers : int
        Threads.

    Returns
    -------
    dict[str, Any]
        ``sets``, ``candidates`` (set, candidate, ``raw`` and ``platt`` tables), ``comparisons`` (set,
        candidate minus P0), ``confirmatory``, ``secondary_sph_calibration``, ``likelihood_50`` (PTB-XL fold
        10), ``audits`` and ``not_implemented``.
    """
    from threadpoolctl import threadpool_limits

    names = list(predictions)
    result: dict[str, Any] = {
        "selected": selected,
        "reference": REFERENCE,
        "sets": {},
        "candidates": {},
        "comparisons": {},
        "audits": {},
    }
    reports, resamples = {}, {}
    with threadpool_limits(1):
        for role in PRIMARY_SETS:
            data = sets[role]
            resamples[role] = cluster_resamples(data["group"], data["labels"], draws)
            result["sets"][role] = set_summary(data, resamples[role])
            reports[role] = primary_set(
                data["labels"],
                {name: predictions[name][role] for name in names},
                operating,
                resamples[role],
                workers,
            )
            result["candidates"][role] = {
                name: {kind: reports[role][name][kind] for kind in TABLE_KINDS} for name in names
            }
            result["library_calibration_failures"] = result.get("library_calibration_failures", {}) | {
                role: {name: library_failures(reports[role][name]) for name in names}
            }
            result["comparisons"][role] = {
                name: paired_comparison(reports[role][name]["raw_boot"], reports[role][REFERENCE]["raw_boot"])
                for name in names
                if name != REFERENCE
            }
            result["audits"][role] = {
                name: audits(data, predictions[name][role], operating[name]) for name in names
            }
        result["secondary_sph_calibration"] = {
            "fit_set": SECONDARY_FIT_SET,
            "applied_to": CONFIRMATORY_SET,
            "fit_records": len(sets[SECONDARY_FIT_SET]["record_key"]),
            "candidates": secondary_analysis(
                sets[SECONDARY_FIT_SET],
                sets[CONFIRMATORY_SET],
                {name: predictions[name][SECONDARY_FIT_SET] for name in names},
                {name: predictions[name][CONFIRMATORY_SET] for name in names},
                reports[CONFIRMATORY_SET],
                resamples[CONFIRMATORY_SET],
                workers,
            ),
        }
        likelihood = sets[LIKELIHOOD_SET]
        changed = int((np.isnan(likelihood["labels_lik50"]) != np.isnan(likelihood["labels"])).sum())
        lik_resamples = cluster_resamples(likelihood["group"], likelihood["labels_lik50"], draws)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tables = list(
                pool.map(
                    lambda name: interval_report(
                        likelihood["labels_lik50"],
                        predictions[name][LIKELIHOOD_SET],
                        lik_resamples,
                        operating[name]["thresholds"],
                    )[0],
                    names,
                )
            )
        result["likelihood_50"] = {
            "set": LIKELIHOOD_SET,
            "entries_made_undefined": changed,
            "resampling": set_summary(likelihood, lik_resamples),
            "candidates": dict(zip(names, tables, strict=True)),
        }
    if selected == REFERENCE:
        result["confirmatory"] = {
            "set": CONFIRMATORY_SET,
            "selected": selected,
            "reference": REFERENCE,
            "selected_is_reference": True,
            "better": False,
        }
    else:
        result["confirmatory"] = {
            "set": CONFIRMATORY_SET,
            "selected": selected,
            "reference": REFERENCE,
            "selected_is_reference": False,
            "better": result["comparisons"][CONFIRMATORY_SET][selected]["macro"]["better"],
            **result["comparisons"][CONFIRMATORY_SET][selected],
        }
    result["not_implemented"] = NOT_IMPLEMENTED
    return result


def summary_rows(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Flat rows of the per-condition tables, the macro AUROCs and the comparisons.

    Parameters
    ----------
    result : dict[str, Any]
        Output of ``evaluate_sets``.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        Condition rows, macro rows and comparison rows.
    """
    conditions, macros, comparisons = [], [], []
    for role, by_name in result["candidates"].items():
        for name, tables in by_name.items():
            for kind, table in tables.items():
                if "macro_auroc" in table:
                    macros.append(
                        {
                            "set": role,
                            "candidate": name,
                            "probabilities": kind,
                            "macro_auroc": table["macro_auroc"],
                            "ci_low": table["macro_auroc_ci_low"],
                            "ci_high": table["macro_auroc_ci_high"],
                            "entering": len(table["entering"]),
                        }
                    )
                conditions += [
                    {"set": role, "candidate": name, "probabilities": kind, **row}
                    for row in table["conditions"]
                ]
    for role, by_name in result["comparisons"].items():
        for name, comparison in by_name.items():
            confirmatory = role == CONFIRMATORY_SET and name == result["selected"]
            comparisons.append(
                {
                    "set": role,
                    "candidate": name,
                    "reference": REFERENCE,
                    "condition": "macro",
                    "confirmatory": confirmatory,
                    **comparison["macro"],
                }
            )
            comparisons += [
                {"set": role, "candidate": name, "reference": REFERENCE, "confirmatory": False, **row}
                for row in comparison["conditions"]
            ]
    return conditions, macros, comparisons


def write_evaluation(directory: Path, prefix: str, result: dict[str, Any]) -> dict[str, str]:
    """
    Write the evaluation JSON and its CSV summaries.

    Parameters
    ----------
    directory : Path
        Destination directory.
    prefix : str
        File name prefix (``DRY_RUN_validation_standin_`` for dry runs).
    result : dict[str, Any]
        Output of ``evaluate_sets`` with provenance fields.

    Returns
    -------
    dict[str, str]
        File name to SHA-256.
    """
    directory = Path(directory)
    write_json_atomic(directory / f"{prefix}evaluation.json", json_ready(result))
    for name, rows in zip(("conditions", "macro", "comparisons"), summary_rows(result), strict=True):
        rows = json_ready(rows)
        write_csv_atomic(
            directory / f"{prefix}{name}.csv", rows, list(dict.fromkeys(k for r in rows for k in r))
        )
    return {path.name: sha256_file(path) for path in sorted(directory.glob(f"{prefix}*"))}


def run_evaluate(
    output: Path = STAGE3,
    manifest: Path = MANIFEST,
    draws: int = DRAWS,
    workers: int = WORKERS,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate the saved test predictions and write ``evaluation/``.

    Parameters
    ----------
    output : Path
        Stage 3 directory with the record, the operating points, the test-predict receipt and the predictions.
    manifest : Path
        MLC-01 ``manifest.csv``.
    draws : int
        Bootstrap resamples per set.
    workers : int
        Threads.
    command : list[str] | None
        Command line for the receipt.

    Returns
    -------
    dict[str, Any]
        The confirmatory comparison and the written files.
    """
    started = time.time()
    output = Path(output)
    record, record_sha = read_record(output)
    operating, operating_sha = read_operating_points(output, record_sha)
    data = load_manifest(manifest)
    sets = {role: role_data(data, role) for role in TEST_ROLES}
    predictions, pinned = load_test_predictions(output, record, record_sha, operating_sha, sets)
    result = evaluate_sets(sets, predictions, operating, record["selected"], draws, workers)
    code = code_state(require_committed=True)
    result["provenance"] = receipt(
        "evaluate",
        command or [],
        started,
        code,
        {
            "selection_record_sha256": record_sha,
            "operating_points_sha256": operating_sha,
            "test_predictions_receipt_sha256": sha256_file(output / TEST_RECEIPT_NAME),
            "manifest": stored(manifest),
            "manifest_sha256": sha256_file(manifest),
            "dry_run": False,
        },
    )
    files = write_evaluation(output / EVALUATION, "", result)
    return {"confirmatory": result["confirmatory"], "files": files}


STANDIN = {
    "test_same_source": "validation PTB-XL fold 9",
    "test_external": "validation Ningbo, even key hash",
    "sph_calibration": "validation Ningbo, odd key hash",
}


def standin_masks(validation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Mark validation stand-ins for the three test roles, for a dry run only.

    ``test_same_source`` is validation PTB-XL fold 9; ``test_external`` and ``sph_calibration`` are the two
    halves of validation Ningbo, split by the parity of the SHA-256 of the record key. Every condition is
    defined in both sources, as at SPH.

    Parameters
    ----------
    validation : dict[str, np.ndarray]
        Validation records.

    Returns
    -------
    dict[str, np.ndarray]
        Stand-in role to a mask over the validation records.
    """
    parity = np.array(
        [int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 for key in validation["record_key"]]
    )
    ningbo = validation["source"] == "ningbo"
    return {
        "test_same_source": validation["source"] == "ptbxl",
        "test_external": ningbo & (parity == 0),
        "sph_calibration": ningbo & (parity == 1),
    }


def run_evaluate_dry(
    scratch: Path,
    output: Path = STAGE3,
    manifest: Path = MANIFEST,
    draws: int = DRAWS,
    workers: int = WORKERS,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """
    Dry run of ``evaluate_sets`` on validation stand-ins, written to a scratch directory.

    The validation predictions of the record stand in for test predictions. The operating points were fitted
    on the same validation records, so nothing here is a result; it only shows that the path runs.

    Parameters
    ----------
    scratch : Path
        Scratch directory; files are prefixed ``DRY_RUN_validation_standin_``.
    output : Path
        Stage 3 directory holding the record and operating points to follow (provisional accepted).
    manifest : Path
        MLC-01 ``manifest.csv``.
    draws : int
        Bootstrap resamples per set.
    workers : int
        Threads.
    command : list[str] | None
        Command line for the provenance.

    Returns
    -------
    dict[str, Any]
        Timing, the stand-in sizes and the written files.
    """
    started = time.time()
    record, record_sha = read_record(output)
    operating, operating_sha = read_operating_points(output, record_sha)
    validation = validation_data(manifest)
    saved = validation_probabilities(record, validation)
    masks = standin_masks(validation)
    sets = {role: subset(validation, mask) for role, mask in masks.items()}
    predictions = {
        name: {role: probability[masks[role]] for role in TEST_ROLES} for name, probability in saved.items()
    }
    result = evaluate_sets(sets, predictions, operating, record["selected"], draws, workers)
    code = code_state(require_committed=False)
    result["provenance"] = receipt(
        "evaluate dry run",
        command or [],
        started,
        code,
        {
            "dry_run": True,
            "standin": STANDIN,
            "selection_record_sha256": record_sha,
            "operating_points_sha256": operating_sha,
            "selection_record_provisional": record["provisional"],
            "test_roles_opened": False,
        },
    )
    prefix = "DRY_RUN_validation_standin_"
    files = write_evaluation(Path(scratch), prefix, result)
    return {
        "seconds": time.time() - started,
        "records": {role: len(sets[role]["record_key"]) for role in TEST_ROLES},
        "files": files,
    }
