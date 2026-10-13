"""Forward-only historical map inputs and metrics factored from frozen Experiment 042.

The seven historical helpers preserve numerical operations and seeds verbatim. New runners import
this library rather than importing scripts. ROOT comes from the shared package; PRIMARY contains
only the map whose integrity is required here. Historical 042 comparisons remain descriptive.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment import ROOT
from ecg_experiment.ann_heads import ragged_unit_maps
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file
from ecg_experiment.fragment_localization import (
    bootstrap_mean,
    pointing,
    premature_windows,
    r_peaks,
    section_overlap,
)
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    UnitMap,
    ecg_score,
    premature_hit,
    top_lead,
    two_group_difference,
)

PRIOR041 = ROOT / "outputs/experiment041_fragment_localization_v1"
PRIOR042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
CACHE_IDS = (
    ROOT / "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
    ROOT / "outputs/experiment016_xecg_probe_finetune/features/ecg_ids.npy",
    ROOT / "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
)
EXPECTED = {
    "fit": (5872, 5537),
    "pool": (15359, 9487),
    "evaluation": (1306, 843),
    "development": 1604,
    "pvc": (84, 81),
    "benign": 52,
    "anterior": (146, 133),
    "inferior": (149, 139),
}
BENIGN_CODES = {"NORM", "SR", "SBRAD", "SARRH"}
OTHER_MI = ("LMI", "PMI")
PRIMARY = {"B": "U_B"}
TIMED = ("U_J", "U_J_kmeans", "G_J", "U_B", "raw_residual", "U_B_fixed")
FS = 500
SEED = 42042
DRAWS = 2000
LOCALIZATION_MARGIN = -0.10


def statement_table() -> pd.DataFrame:
    """Return historical statement sets per ECG.

    Returns
    -------
    pd.DataFrame
        SCP codes and diagnostic classes/subclasses indexed by ECG ID.
    """
    meta = load_metadata()
    statements = load_statements()
    diagnostic = statements[statements["diagnostic"] == 1]
    subclass = diagnostic["diagnostic_subclass"].to_dict()
    superclass = diagnostic["diagnostic_class"].to_dict()
    codes = meta["scp_codes"].apply(set)
    return pd.DataFrame(
        {
            "codes": codes,
            "subclasses": codes.apply(lambda found: {subclass[c] for c in found if c in subclass} - {"NORM"}),
            "superclasses": codes.apply(lambda found: {superclass[c] for c in found if c in superclass}),
        },
        index=meta.index.astype(int),
    )


def only(subclasses: set[str], name: str, other: str) -> bool:
    """Check the historical exclusive infarct-location grouping.

    Parameters
    ----------
    subclasses : set[str]
        Diagnostic subclasses of one ECG.
    name, other : str
        Included and excluded infarct locations.

    Returns
    -------
    bool
        Whether the record meets the original grouping rule.
    """
    return name in subclasses and other not in subclasses and not subclasses & set(OTHER_MI)


def select_rows(smoke: bool) -> dict[str, Any]:
    """
    Select the Experiment 041 rows plus anterior-only and inferior-only infarcts.

    In smoke mode every scored set comes from training ECGs, so no development score exists before the run.

    Parameters
    ----------
    smoke : bool
        Use small training-only sets.

    Returns
    -------
    dict[str, Any]
        Frames keyed by set name, ``extract`` (fit set first, then ``scored``) and the statement table.

    Raises
    ------
    ValueError
        If a full-run count or the scored order differs from the protocol and Experiment 041.
    """
    groups = cohorts(ptb_table())
    statements = statement_table()
    common = set.intersection(*(set(np.load(path).astype(np.int64).tolist()) for path in CACHE_IDS))
    train, development = groups["train"], groups["development"]
    pool = train[train["standard"].notna() & train["ecg_id"].isin(common)]
    fit = pool[pool["standard"] == 0]
    pvc = set(statements.index[statements["codes"].map(lambda codes: "PVC" in codes)])
    benign = {
        i
        for i, codes, found in zip(
            statements.index, statements["codes"], statements["superclasses"], strict=True
        )
        if found == {"NORM"} and codes <= BENIGN_CODES and codes & {"SBRAD", "SARRH"}
    }
    anterior = set(statements.index[statements["subclasses"].map(lambda found: only(found, "AMI", "IMI"))])
    inferior = set(statements.index[statements["subclasses"].map(lambda found: only(found, "IMI", "AMI"))])

    def subset(frame: pd.DataFrame, ids: set[int], limit: int | None = None) -> pd.DataFrame:
        chosen = frame[frame["ecg_id"].isin(ids)]
        return chosen if limit is None else chosen.iloc[:limit]

    if smoke:
        fit = fit.iloc[:300]
        rest = pool.drop(fit.index)
        evaluation = pd.concat(
            [rest[rest["standard"] == 0].iloc[:100], rest[rest["standard"] == 1].iloc[:100]]
        )
        others = train.drop(fit.index)
        sets = {
            "pvc": subset(others, pvc, 30),
            "benign": subset(others, benign, 20),
            "anterior": subset(others, anterior, 40),
            "inferior": subset(others, inferior, 40),
        }
        scored = pd.concat([evaluation, *sets.values()])
        scored = scored[~scored.index.duplicated()]
    else:
        original = development[development["original"]]
        evaluation = original[original["standard"].notna() & original["ecg_id"].isin(common)]
        sets = {
            "pvc": subset(development, pvc),
            "benign": subset(development, benign),
            "anterior": subset(development, anterior),
            "inferior": subset(development, inferior),
        }
        scored = development
        counts = {
            "fit": (len(fit), fit["patient_id"].nunique()),
            "pool": (len(pool), int(pool["standard"].sum())),
            "evaluation": (len(evaluation), int(evaluation["standard"].sum())),
            "development": len(scored),
            "pvc": (len(sets["pvc"]), sets["pvc"]["patient_id"].nunique()),
            "benign": len(sets["benign"]),
            "anterior": (len(sets["anterior"]), sets["anterior"]["patient_id"].nunique()),
            "inferior": (len(sets["inferior"]), sets["inferior"]["patient_id"].nunique()),
        }
        if counts != EXPECTED:
            raise ValueError(f"Row counts differ from the protocol: {counts}")
        with np.load(PRIOR041 / "section_scores.npz") as saved:
            if not np.array_equal(saved["ecg_ids"], scored["ecg_id"].to_numpy(np.int64)):
                raise ValueError("Scored rows differ from Experiment 041")
    if set(fit.index) & set(scored.index):
        raise ValueError("A fit ECG is also scored")
    return {
        "fit": fit,
        "pool": pool,
        "evaluation": evaluation,
        **sets,
        "scored": scored,
        "extract": pd.concat([fit, scored]),
        "statements": statements,
    }


def auroc_ap(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Return historical discrimination point metrics.

    Parameters
    ----------
    y, score : np.ndarray
        Binary labels and aligned anomaly scores.

    Returns
    -------
    dict[str, float]
        AUROC and average precision.
    """
    return {
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def prior041(rows: dict[str, Any], windows: dict[int, list[tuple[float, float]]]) -> dict[str, Any]:
    """
    Experiment 041 ``U`` per scored ECG: ECG score, any red, and hit - chance on premature beats.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows`` (full run).
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows of the included PVC rows, keyed by ECG ID.

    Returns
    -------
    dict[str, Any]
        Arrays keyed by ECG ID.
    """
    threshold = json.loads((PRIOR041 / "result.json").read_text())["thresholds"]["U"]
    with np.load(PRIOR041 / "section_scores.npz") as saved:
        ids, sections = saved["ecg_ids"], saved["U"]
    by_id = dict(zip(ids.tolist(), sections, strict=True))
    excess = {}
    for ecg_id, found in windows.items():
        hit, chance = pointing(by_id[ecg_id][None], section_overlap(found)[None])
        excess[ecg_id] = float(hit[0] - chance[0])
    return {
        "score": {i: float(v.max()) for i, v in by_id.items()},
        "red": {i: bool((v > threshold).any()) for i, v in by_id.items()},
        "excess": excess,
    }


def analyse(
    rows: dict[str, Any],
    maps: dict[str, list[UnitMap | None]],
    windows: dict[int, list[tuple[float, float]]],
    smoke: bool,
) -> dict[str, Any]:
    """
    Every prespecified metric, contrast and reading.

    Parameters
    ----------
    rows : dict[str, Any]
        Output of ``select_rows``.
    maps : dict[str, list[UnitMap | None]]
        Unit maps in ``rows["scored"]`` order.
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows of the included PVC rows, keyed by ECG ID.
    smoke : bool
        Skip the Experiment 041 contrasts, which need development rows.

    Returns
    -------
    dict[str, Any]
        Thresholds, per-map metrics, contrasts and readings.
    """
    ids = rows["scored"]["ecg_id"].to_numpy()
    patients = dict(zip(ids, rows["scored"]["patient_id"], strict=True))
    evaluation = rows["evaluation"]
    labels = dict(zip(evaluation["ecg_id"], evaluation["standard"].astype(int), strict=True))
    normals = [i for i, y in labels.items() if y == 0]
    groups = {
        "normal": normals,
        "positive": [i for i, y in labels.items() if y == 1],
        "pvc": rows["pvc"]["ecg_id"].tolist(),
        "benign": rows["benign"]["ecg_id"].tolist(),
    }
    previous = None if smoke else prior041(rows, windows)
    result: dict[str, Any] = {"maps": {}, "thresholds": {}, "pvc_included": len(windows)}
    for name, unit_maps in maps.items():
        by_id = {i: m for i, m in zip(ids, unit_maps, strict=True) if m is not None}
        threshold = float(np.quantile([ecg_score(by_id[i]) for i in normals if i in by_id], 0.95))
        result["thresholds"][name] = threshold
        scored_eval = [i for i in labels if i in by_id]
        y = np.array([labels[i] for i in scored_eval])
        score = np.array([ecg_score(by_id[i]) for i in scored_eval])
        red = {i: m.scores > threshold for i, m in by_id.items()}
        found = {
            "ecgs_with_map": len(by_id),
            "evaluation_ecgs": len(scored_eval),
            **auroc_ap(y, score),
            "any_red": {g: float(np.mean([red[i].any() for i in m if i in red])) for g, m in groups.items()},
            "mean_red_units": {
                g: float(np.mean([red[i].sum() for i in m if i in red])) for g, m in groups.items()
            },
        }
        positives = [i for i in groups["positive"] if i in red]
        found["sensitivity_at_budget"] = bootstrap_mean(
            np.array([patients[i] for i in positives]),
            np.array([float(red[i].any()) for i in positives]),
            DRAWS,
            SEED,
        )
        contrasts = {}
        for region, leads, first, second in (
            ("anterior", ANTERIOR, "anterior", "inferior"),
            ("inferior", INFERIOR, "inferior", "anterior"),
        ):
            a = [i for i in rows[first]["ecg_id"] if i in by_id]
            b = [i for i in rows[second]["ecg_id"] if i in by_id]
            contrasts[region] = two_group_difference(
                np.array([patients[i] for i in a]),
                np.array([float(top_lead(by_id[i]) in leads) for i in a]),
                np.array([patients[i] for i in b]),
                np.array([float(top_lead(by_id[i]) in leads) for i in b]),
                DRAWS,
                SEED,
            )
        found["lead_contrast"] = contrasts
        if name in TIMED:
            included = [i for i in windows if i in by_id]
            pairs = [premature_hit(by_id[i], windows[i]) for i in included]
            hit, chance = np.array(pairs).T
            pvc_patients = np.array([patients[i] for i in included])
            found.update(
                {
                    "pvc_ecgs": len(included),
                    "hit_rate": float(hit.mean()),
                    "chance_rate": float(chance.mean()),
                    "hit_minus_chance": bootstrap_mean(pvc_patients, hit - chance, DRAWS, SEED),
                }
            )
            if previous is not None:
                before = np.array([previous["excess"][i] for i in included])
                found["hit_minus_chance_minus_041"] = bootstrap_mean(
                    pvc_patients, hit - chance - before, DRAWS, SEED
                )
        if previous is not None:
            found["auroc_minus_041"] = paired_auroc_difference(
                np.array([patients[i] for i in scored_eval]),
                y,
                score,
                np.array([previous["score"][i] for i in scored_eval]),
                DRAWS,
                SEED,
            )
            benign = [i for i in groups["benign"] if i in red]
            found["benign_any_red_minus_041"] = bootstrap_mean(
                np.array([patients[i] for i in benign]),
                np.array([float(red[i].any()) - float(previous["red"][i]) for i in benign]),
                DRAWS,
                SEED,
            )
        result["maps"][name] = found
    if previous is not None:
        result["reading"] = {}
        for arm, name in PRIMARY.items():
            found = result["maps"][name]
            keeps = found["hit_minus_chance_minus_041"]["ci_low"] > LOCALIZATION_MARGIN
            gains = {
                "lead": found["lead_contrast"]["anterior"]["ci_low"] > 0,
                "detection": found["auroc_minus_041"]["ci_low"] > 0,
                "benign": found["benign_any_red_minus_041"]["ci_high"] < 0,
            }
            result["reading"][arm] = {
                "keeps_premature_localization": bool(keeps),
                "gains": {key: bool(value) for key, value in gains.items()},
                "improves_on_041": bool(keeps and any(gains.values())),
            }
        result["reading"]["notebook"] = any(result["reading"][arm]["improves_on_041"] for arm in PRIMARY)
    return result


def premature_targets(frame: pd.DataFrame) -> tuple[dict[int, list[tuple[float, float]]], dict[str, int]]:
    """Rebuild historical automatic premature-beat windows.

    Parameters
    ----------
    frame : pd.DataFrame
        Historical PVC metadata rows with waveform stems.

    Returns
    -------
    tuple
        Windows keyed by ECG ID and exclusion counts.
    """
    windows, counts = {}, {"too_few_peaks": 0, "no_premature_beat": 0}
    for ecg_id, stem in zip(frame["ecg_id"], frame["filename_hr"], strict=True):
        peaks = r_peaks(read_ptb_float64(stem), FS)
        found = premature_windows(peaks, FS)
        if len(peaks) < 4:
            counts["too_few_peaks"] += 1
        elif not found:
            counts["no_premature_beat"] += 1
        else:
            windows[int(ecg_id)] = found
    return windows, counts


def load_baseline() -> tuple[
    dict[str, Any], list[UnitMap], dict[int, list[tuple[float, float]]], dict[str, Any]
]:
    """Load, receipt-check and reproduce the complete historical U_B map metrics.

    Returns
    -------
    tuple
        Historical rows, maps, premature windows, and integrity receipt.
    """
    prior = json.loads((PRIOR042 / "result.json").read_text())
    checked = {}
    expected = {**prior["identity"]["inputs"], **prior["identity"]["sources"]}
    relevant = (
        "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
        "data/raw/ptb-xl/1.0.3/scp_statements.csv",
        "ecg_experiment/lead_wave_maps.py",
        "ecg_experiment/fragment_localization.py",
        "ecg_experiment/full_development.py",
        "ecg_experiment/intervals.py",
        "ecg_experiment/eda/ptbxl.py",
        "ecg_experiment/external_encoders.py",
        "scripts/experiments/run_lead_wave_maps042.py",
        "outputs/experiment041_fragment_localization_v1/section_scores.npz",
        "outputs/experiment041_fragment_localization_v1/result.json",
        "pyproject.toml",
        "uv.lock",
    )
    for name in relevant:
        checked[name] = sha256_file(ROOT / name)
        if checked[name] != expected[name]:
            raise ValueError(f"Historical receipt mismatch: {name}")
    later = json.loads((ROOT / "outputs/experiment051_beat_sum_v1/result.json").read_text())
    name = "outputs/experiment042_lead_wave_maps_v1/result.json"
    checked[name] = sha256_file(ROOT / name)
    if checked[name] != later["identity"]["inputs"][name]:
        raise ValueError("042 receipt changed since 051")
    path = PRIOR042 / "unit_scores.npz"
    digest = sha256_file(path)
    if digest != prior["outputs_sha256"]["unit_scores.npz"]:
        raise ValueError("042 units differ from their receipt")
    rows = select_rows(False)
    with np.load(path) as saved:
        if not np.array_equal(saved["ecg_ids"], rows["scored"]["ecg_id"].to_numpy(np.int64)):
            raise ValueError("042 unit rows differ")
        maps = ragged_unit_maps({key: saved[key] for key in saved.files}, "U_B")
    if any(found is None for found in maps):
        raise ValueError("Missing U_B map")
    windows, exclusions = premature_targets(rows["pvc"])
    if len(windows) != 73 or exclusions != {"too_few_peaks": 0, "no_premature_beat": 11}:
        raise ValueError("Historical premature windows differ")
    metrics = analyse(rows, {"U_B": maps}, windows, False)
    found = json.loads(json.dumps(metrics["maps"]["U_B"]))
    if found != prior["maps"]["U_B"] or metrics["thresholds"]["U_B"] != prior["thresholds"]["U_B"]:
        raise ValueError("Full U_B metric reproduction failed")
    return (
        rows,
        maps,
        windows,
        {
            "full_U_B_metrics_equal": True,
            "checked_historical_hashes": checked,
            "exclusions": exclusions,
            "unit_scores_sha256": digest,
            "metrics": found,
        },
    )
