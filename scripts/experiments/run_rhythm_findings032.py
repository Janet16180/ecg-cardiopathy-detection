"""Experiment 032: frozen-encoder readouts for the rhythm findings the athlete criteria call abnormal."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from ecg_experiment.eda.ptbxl import load_metadata
from ecg_experiment.external_readout import prior_cpc_features, prior_identity
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, predict, ptb_table
from ecg_experiment.rhythm_findings import (
    GROUPS,
    clear_normal,
    frequent_pvc,
    label_table,
    set_statistics,
    usability,
)
from ecg_experiment.sph import base_codes
from scripts.experiments.run_calibrated_threshold027 import prior022_identity
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_multisource_manifold026b import (
    SPLIT_ROWS_SHA256,
    challenge_features,
    quality_reasons,
)
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment032_rhythm_findings_v1"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
SPLITS = ROOT / "data/processed/challenge_splits_v1"
NINGBO = ROOT / "data/processed/ningbo_clean_v1"
SPH_MANIFEST = ROOT / "data/processed/sph_clean_v1"
HEADERS = ROOT / "outputs/eda/features/challenge_headers.parquet"
ENCODERS = ("xecg", "jepa", "cpc")
PRIMARY_ENCODER = "xecg"
SEED = 36036
BOOTSTRAP_DRAWS = 2000
BUDGETS = (20, 50, 100)
PRIMARY_BUDGET = 50
MIN_TRAINING_POSITIVES = 50
MIN_EVALUATION_POSITIVES = 20
REPRODUCTION_TOLERANCE = 1e-9
USABLE_AUROC = 0.90
USABLE_LOWER = 0.85
WITHOUT_NINGBO = "af_flutter_without_ningbo"
READOUTS = (*GROUPS, WITHOUT_NINGBO)
FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
EXPECTED: dict[str, Any] = {'positives': {'ventricular_ectopy': {'ptbxl:train': [915, 17370],
                                          'ptbxl:development': [84, 1600],
                                          'sph:evaluation': [1058, 25577],
                                          'chapman_shaoxing:train': [177, 6126],
                                          'cpsc_2018:train': [360, 3765],
                                          'cpsc_2018_extra:train': [83, 1797],
                                          'georgia:train': [239, 6102],
                                          'ningbo:train': [663, 19750],
                                          'chapman_shaoxing:calibration': [64, 2046],
                                          'cpsc_2018:calibration': [114, 1265],
                                          'cpsc_2018_extra:calibration': [23, 603],
                                          'georgia:calibration': [66, 2041],
                                          'ningbo:calibration': [229, 6924]},
                   'preexcitation': {'ptbxl:train': [64, 17417],
                                     'ptbxl:development': [6, 1604],
                                     'sph:evaluation': [27, 25566],
                                     'chapman_shaoxing:train': [11, 6127],
                                     'cpsc_2018:train': [0, 0],
                                     'cpsc_2018_extra:train': [4, 1796],
                                     'georgia:train': [3, 6101],
                                     'ningbo:train': [42, 19788],
                                     'chapman_shaoxing:calibration': [3, 2046],
                                     'cpsc_2018:calibration': [0, 0],
                                     'cpsc_2018_extra:calibration': [0, 602],
                                     'georgia:calibration': [0, 2041],
                                     'ningbo:calibration': [11, 6941]},
                   'af_flutter': {'ptbxl:train': [1257, 17417],
                                  'ptbxl:development': [115, 1604],
                                  'sph:evaluation': [762, 25577],
                                  'chapman_shaoxing:train': [1339, 6127],
                                  'cpsc_2018:train': [584, 3765],
                                  'cpsc_2018_extra:train': [75, 1797],
                                  'georgia:train': [439, 6102],
                                  'ningbo:train': [4445, 19803],
                                  'chapman_shaoxing:calibration': [458, 2046],
                                  'cpsc_2018:calibration': [215, 1265],
                                  'cpsc_2018_extra:calibration': [31, 603],
                                  'georgia:calibration': [133, 2041],
                                  'ningbo:calibration': [1531, 6946]},
                   'svt': {'ptbxl:train': [32, 17291],
                           'ptbxl:development': [4, 1592],
                           'sph:evaluation': [13, 25577],
                           'chapman_shaoxing:train': [435, 6127],
                           'cpsc_2018:train': [0, 0],
                           'cpsc_2018_extra:train': [8, 1797],
                           'georgia:train': [33, 6102],
                           'ningbo:train': [167, 19803],
                           'chapman_shaoxing:calibration': [146, 2046],
                           'cpsc_2018:calibration': [0, 0],
                           'cpsc_2018_extra:calibration': [4, 603],
                           'georgia:calibration': [15, 2041],
                           'ningbo:calibration': [70, 6946]},
                   'high_grade_av_block': {'ptbxl:train': [12, 17405],
                                           'ptbxl:development': [1, 1603],
                                           'sph:evaluation': [27, 25497],
                                           'chapman_shaoxing:train': [0, 6023],
                                           'cpsc_2018:train': [0, 0],
                                           'cpsc_2018_extra:train': [12, 1787],
                                           'georgia:train': [5, 6044],
                                           'ningbo:train': [43, 19688],
                                           'chapman_shaoxing:calibration': [0, 2005],
                                           'cpsc_2018:calibration': [0, 0],
                                           'cpsc_2018_extra:calibration': [4, 599],
                                           'georgia:calibration': [1, 2020],
                                           'ningbo:calibration': [24, 6910]},
                   'long_qt': {'ptbxl:train': [94, 17417],
                               'ptbxl:development': [5, 1604],
                               'sph:evaluation': [24, 25577],
                               'chapman_shaoxing:train': [34, 6127],
                               'cpsc_2018:train': [0, 0],
                               'cpsc_2018_extra:train': [2, 1797],
                               'georgia:train': [814, 6102],
                               'ningbo:train': [191, 19803],
                               'chapman_shaoxing:calibration': [11, 2046],
                               'cpsc_2018:calibration': [0, 0],
                               'cpsc_2018_extra:calibration': [1, 603],
                               'georgia:calibration': [304, 2041],
                               'ningbo:calibration': [64, 6946]}},
     'readouts': {'binary': [39577, 27360],
                  'ventricular_ectopy': [54910, 2437],
                  'preexcitation': [51229, 124],
                  'af_flutter': [55011, 8139],
                  'svt': [51120, 675],
                  'high_grade_av_block': [50947, 72],
                  'long_qt': [51246, 1135],
                  'af_flutter_without_ningbo': [35208, 3694]},
     'sph_frequent_pvc': 372,
     'excluded': {'chapman_shaoxing': 10,
                  'cpsc_2018': 31,
                  'cpsc_2018_extra': 18,
                  'georgia': 12,
                  'ningbo': 1031}}
SOURCES = (
    "ecg_experiment/rhythm_findings.py", "ecg_experiment/referral_budget.py",
    "ecg_experiment/normal_manifold.py", "ecg_experiment/full_development.py",
    "ecg_experiment/external_readout.py", "ecg_experiment/challenge_features.py",
    "ecg_experiment/multisource_manifold.py", "ecg_experiment/sph.py",
    "scripts/experiments/run_rhythm_findings032.py", "scripts/experiments/run_sph_external022.py",
    "scripts/experiments/run_calibrated_threshold027.py",
    "scripts/experiments/run_multisource_calibration027b.py",
    "scripts/experiments/run_multisource_manifold026b.py", "pyproject.toml", "uv.lock",
    "docs/experiment-032-rhythm-findings.md",
)


def challenge_codes(rows: pd.DataFrame) -> list[set[str]]:
    """
    SNOMED codes of each Challenge row: Ningbo from its clean manifest, the others from the EDA header cache.

    Parameters
    ----------
    rows : pd.DataFrame
        Split rows with ``source``, ``record`` and Ningbo's ``snomed_codes``.

    Returns
    -------
    list[set[str]]
        Codes per row.

    Raises
    ------
    ValueError
        If the header cache differs from the split's input or a record lacks a header.
    """
    metadata = json.loads((SPLITS / "metadata.json").read_text())
    if sha256_file(HEADERS) != metadata["inputs_sha256"]["challenge_headers_parquet"]:
        raise ValueError("Challenge header cache differs from the split's input")
    headers = pd.read_parquet(HEADERS, columns=["dx_codes"])["dx_codes"]
    codes = []
    for row in rows.itertuples(index=False):
        if row.source == "ningbo":
            codes.append(set(str(row.snomed_codes).split(";")))
            continue
        codes.append({str(code) for code in headers.loc[f"{row.source}:{row.record}"]})
    return codes


def challenge_table() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Train and calibration rows of the frozen Challenge split with features, finding labels and codes.

    Every row that is not a dropped duplicate is kept, whatever its binary label, because a record whose
    only findings are rhythm statements has an undefined binary label.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        Rows in source then record order, and features per encoder.

    Raises
    ------
    ValueError
        If the split or the Ningbo manifest differs from its receipt.
    """
    metadata = json.loads((SPLITS / "metadata.json").read_text())
    digest = sha256_file(SPLITS / "rows.csv")
    if digest != metadata["rows_sha256"] or digest != SPLIT_ROWS_SHA256:
        raise ValueError("Challenge split differs from its receipt")
    if sha256_file(NINGBO / "rows.csv") != metadata["inputs_sha256"]["ningbo_rows_csv"]:
        raise ValueError("Ningbo manifest differs from the split's input")
    rows = pd.read_csv(SPLITS / "rows.csv", dtype={"record": str})
    keep = rows["split"].isin(["train", "calibration"]) & rows["duplicate_status"].isin(["unique", "kept"])
    ningbo = pd.read_csv(NINGBO / "rows.csv", dtype={"record": str, "snomed_codes": str},
                         usecols=["record", "zero_leads", "use_training", "snomed_codes"])
    rows = rows[keep].merge(ningbo, on="record", how="left")
    rows["zero_lead"] = rows["zero_leads"].fillna("") != ""
    rows, features = challenge_features(rows.drop(columns="zero_leads"))
    labels = label_table(challenge_codes(rows), "snomed", rows["source"])
    rows = pd.concat([rows.drop(columns="snomed_codes"), labels.set_axis(rows.index)], axis=1)
    return rows, features


def ptb_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """
    PTB-XL training and full-development rows with finding labels and every encoder's features.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]], dict[str, Any]]
        Rows keyed by ``train`` and ``development`` (with ``standard`` and the group labels), features keyed
        by encoder then part, and the input hashes.

    Raises
    ------
    ValueError
        If the feature caches differ from Experiment 022's.
    """
    prior = json.loads((PRIOR022 / "result.json").read_text())
    hashes022 = prior022_identity()
    caches, cache_hashes = open_caches()
    if cache_hashes != prior["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    table = ptb_table()
    groups = cohorts(table)
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = ptb_encoder_features(groups, caches, extracted)
    cpc = prior_cpc_features()
    features = {"xecg": encoded["xecg"], "jepa": encoded["jepa"], "cpc": cpc}
    scp = load_metadata()["scp_codes"]
    rows = {}
    for part in ("train", "development"):
        frame = groups[part]
        codes = [set(scp.loc[ecg_id]) for ecg_id in frame["ecg_id"].astype(int)]
        labels = label_table(codes, "ptbxl").set_axis(frame.index)
        rows[part] = pd.concat([frame, labels], axis=1)
        if any(len(features[name][part]) != len(frame) for name in ENCODERS):
            raise ValueError(f"PTB-XL {part} features do not match the rows")
    identity = {"experiment022": hashes022, "experiment020": prior_identity(), "feature_caches": cache_hashes}
    return rows, features, identity


def sph_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    """
    Every SPH evaluation ECG with finding labels and Experiment 022's saved features.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]
        Rows (with ``primary``, the group labels and ``frequent_pvc``), features per encoder, and hashes.

    Raises
    ------
    ValueError
        If the manifest or the features differ from their receipts or from each other.
    """
    prior = json.loads((PRIOR022 / "result.json").read_text())
    metadata = json.loads((SPH_MANIFEST / "metadata.json").read_text())
    hashes = {"rows": sha256_file(SPH_MANIFEST / "rows.csv"),
              "metadata": sha256_file(SPH_MANIFEST / "metadata.json"),
              "features.npz": sha256_file(PRIOR022 / "features.npz")}
    if hashes["rows"] != metadata["rows_sha256"] or hashes["rows"] != prior["identity"]["sph"]["rows"]:
        raise ValueError("SPH manifest differs from its receipt or from Experiment 022")
    if hashes["features.npz"] != prior["outputs_sha256"]["features.npz"]:
        raise ValueError("Experiment 022 SPH features differ from its receipt")
    rows = pd.read_csv(SPH_MANIFEST / "rows.csv", dtype={"ecg_id": str, "patient_id": str, "aha_code": str})
    rows = rows[rows["use_evaluation"]].reset_index(drop=True)
    with np.load(PRIOR022 / "features.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], rows["ecg_id"].to_numpy(dtype=str)):
            raise ValueError("Experiment 022 SPH features differ from the manifest rows")
        features = {name: saved[name] for name in ENCODERS}
    labels = label_table([base_codes(text) for text in rows["aha_code"]], "sph")
    rows = pd.concat([rows, labels], axis=1)
    rows["frequent_pvc"] = rows["aha_code"].map(frequent_pvc)
    return rows, features, hashes


def check_binary_rows(train: pd.DataFrame) -> None:
    """
    Require the Challenge rows of the binary readout to be 022b's kept training rows, in its order.

    Parameters
    ----------
    train : pd.DataFrame
        Challenge training rows with ``evaluable`` and ``reasons``.

    Raises
    ------
    ValueError
        If the rows or their quality reasons differ from 022b's ``training_rows.csv``.
    """
    result = json.loads((PRIOR022B / "result.json").read_text())
    if sha256_file(PRIOR022B / "training_rows.csv") != result["outputs_sha256"]["training_rows.csv"]:
        raise ValueError("022b training rows differ from its receipt")
    prior = pd.read_csv(PRIOR022B / "training_rows.csv", dtype={"record": str}, keep_default_na=False)
    mine = train[train["evaluable"]]
    same_rows = np.array_equal(prior["record"].to_numpy(dtype=str), mine["record"].to_numpy(dtype=str))
    if not same_rows or not np.array_equal(prior["reasons"].to_numpy(dtype=str),
                                           mine["reasons"].to_numpy(dtype=str)):
        raise ValueError("Challenge training rows or quality reasons differ from 022b")


def training_sets(ptb: pd.DataFrame, train: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Rows and targets of every readout within the stacked training matrix (PTB-XL first).

    Parameters
    ----------
    ptb : pd.DataFrame
        PTB-XL training rows with ``standard`` and the group labels.
    train : pd.DataFrame
        Challenge training rows passing the quality policy, with ``evaluable``, ``primary``, ``source``
        and the group labels.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per readout (``binary`` and ``READOUTS``): ``rows`` (positions) and ``y``.
    """
    standard = np.concatenate([ptb["standard"].to_numpy(dtype=np.float64),
                               np.where(train["evaluable"], train["primary"], np.nan)])
    sets = {"binary": standard}
    from_ningbo = np.concatenate([np.zeros(len(ptb), dtype=bool), (train["source"] == "ningbo").to_numpy()])
    for group in GROUPS:
        sets[group] = np.concatenate([ptb[group].to_numpy(dtype=np.float64),
                                      train[group].to_numpy(dtype=np.float64)])
    sets[WITHOUT_NINGBO] = np.where(from_ningbo, np.nan, sets["af_flutter"])
    return {name: {"rows": np.flatnonzero(~np.isnan(y)), "y": y[~np.isnan(y)].astype(np.int64)}
            for name, y in sets.items()}


def fit_encoder(name: str, x: np.ndarray, readouts: dict[str, dict[str, np.ndarray]],
                inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Fit the binary readout and every finding readout on one encoder and score every set.

    Parameters
    ----------
    name : str
        Encoder name.
    x : np.ndarray
        Stacked training features.
    readouts : dict[str, dict[str, np.ndarray]]
        Output of ``training_sets``, restricted to the readouts to fit.
    inputs : dict[str, np.ndarray]
        Features of each scored set.

    Returns
    -------
    dict[str, Any]
        ``scores[set][readout]``, fit diagnostics and seconds.
    """
    started = time.monotonic()
    scores: dict[str, dict[str, np.ndarray]] = {set_name: {} for set_name in inputs}
    fits = {}
    with threadpool_limits(limits=1):
        for readout, spec in readouts.items():
            began = time.monotonic()
            head = fit_logistic(x[spec["rows"]], spec["y"])
            model: LogisticRegression = head[1]
            fits[readout] = {"seconds": time.monotonic() - began, "iterations": int(model.n_iter_[0]),
                             "rows": len(spec["y"]), "positives": int(spec["y"].sum())}
            print(json.dumps({"stage": f"{name}:fit:{readout}", **fits[readout]}), flush=True)
            for set_name, values in inputs.items():
                scores[set_name][readout] = predict(head, values)
    return {"scores": scores, "fits": fits, "seconds": time.monotonic() - started}


def reproduce_022b(results: dict[str, Any], sph: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, Any]:
    """
    Require the refitted binary readout to reproduce 022b's ``pooled`` SPH and Challenge probabilities.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``fit_encoder`` per encoder.
    sph, calibration : pd.DataFrame
        SPH rows and Challenge calibration rows in the scored order.

    Returns
    -------
    dict[str, Any]
        Largest absolute differences per encoder and set.

    Raises
    ------
    ValueError
        If a probability differs by more than ``REPRODUCTION_TOLERANCE`` or 022b's file differs from its
        receipt.
    """
    result = json.loads((PRIOR022B / "result.json").read_text())
    if sha256_file(PRIOR022B / "predictions.npz") != result["outputs_sha256"]["predictions.npz"]:
        raise ValueError("022b predictions differ from its receipt")
    differences = {}
    with np.load(PRIOR022B / "predictions.npz") as saved:
        sph_positions = pd.Index(sph["ecg_id"]).get_indexer(saved["sph_ecg_ids"])
        challenge = saved["calibration_families"] != "ptbxl"
        evaluable = calibration["evaluable"].to_numpy(dtype=bool)
        same = np.array_equal(calibration.loc[evaluable, "record"].to_numpy(dtype=str),
                              saved["calibration_records"][challenge])
        if (sph_positions < 0).any() or not same:
            raise ValueError("022b rows are missing from this experiment's sets")
        for name, output in results.items():
            differences[name] = {
                "sph": float(np.abs(output["scores"]["sph"]["binary"][sph_positions]
                                    - saved[f"sph_{name}_pooled"]).max()),
                "challenge_calibration": float(np.abs(
                    output["scores"]["calibration"]["binary"][evaluable]
                    - saved[f"calibration_{name}_pooled"][challenge]).max())}
    if max(max(value.values()) for value in differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The binary readout does not reproduce 022b: {differences}")
    return differences


def evaluation_sets(sph: pd.DataFrame, development: pd.DataFrame, calibration: pd.DataFrame
                    ) -> dict[str, dict[str, Any]]:
    """
    Every (group, set) evaluation: the scored set, its row selection, labels, units and normals.

    Parameters
    ----------
    sph, development, calibration : pd.DataFrame
        SPH, PTB-XL development and Challenge calibration rows with ``standard`` and the group labels.

    Returns
    -------
    dict[str, dict[str, Any]]
        Keyed by ``<group>@<set>``, with ``group``, ``scores_of``, ``selection``, ``y``, ``units``, ``normal``
        and ``standard`` (the binary label of the selected rows).
    """
    sources = {"sph": (sph, "patient_id"), "development": (development, "patient_id"),
               "calibration": (calibration, "record")}
    normals = {name: clear_normal(frame["standard"].to_numpy(), frame[list(GROUPS)])
               for name, (frame, _) in sources.items()}
    sets = {}
    for group in GROUPS:
        choices = {"sph": ("sph", sph[group].notna().to_numpy()),
                   "development": ("development", development[group].notna().to_numpy())}
        for family in FAMILIES:
            choices[f"family:{family}"] = ("calibration", ((calibration["family"] == family)
                                                           & calibration[group].notna()).to_numpy())
        if group == "ventricular_ectopy":
            frequent = (sph[group] == 0) | sph["frequent_pvc"]
            choices["sph_frequent_pvc"] = ("sph", (frequent & sph[group].notna()).to_numpy())
        for set_name, (scored, selected) in choices.items():
            frame, unit = sources[scored]
            y = frame.loc[selected, group].to_numpy(dtype=np.int64)
            if set_name == "sph_frequent_pvc":
                y = frame.loc[selected, "frequent_pvc"].to_numpy(dtype=np.int64)
            sets[f"{group}@{set_name}"] = {
                "group": group, "set": set_name, "scores_of": scored, "selection": selected, "y": y,
                "units": frame.loc[selected, unit].to_numpy(dtype=str), "normal": normals[scored][selected],
                "standard": frame.loc[selected, "standard"].to_numpy(dtype=np.float64)}
    return sets


def score_names(group: str) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Scores compared on a group's sets and the pre-registered contrasts.

    Parameters
    ----------
    group : str
        Group name.

    Returns
    -------
    tuple[list[str], list[tuple[str, str]]]
        Score names (``<readout>:<encoder>``) and (first, second) pairs.
    """
    readouts = [group, "binary", *([WITHOUT_NINGBO] if group == "af_flutter" else [])]
    names = [f"{readout}:{name}" for readout in readouts for name in ENCODERS]
    pairs = [(f"{group}:{name}", f"binary:{name}") for name in ENCODERS]
    pairs += [(f"{group}:xecg", f"{group}:jepa"), (f"{group}:xecg", f"{group}:cpc")]
    if group == "af_flutter":
        pairs += [(f"{group}:{name}", f"{WITHOUT_NINGBO}:{name}") for name in ENCODERS]
    return names, pairs


def evaluate_set(key: str, spec: dict[str, Any], scores: dict[str, np.ndarray]) -> tuple[str, dict[str, Any]]:
    """
    Statistics of one (group, set) evaluation, run in a worker with one BLAS thread.

    Parameters
    ----------
    key : str
        ``<group>@<set>``.
    spec : dict[str, Any]
        One entry of ``evaluation_sets``.
    scores : dict[str, np.ndarray]
        Scores of the selected rows by name.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The key and ``set_statistics`` output plus the share of positives the binary label already marks.
    """
    started = time.monotonic()
    _, pairs = score_names(spec["group"])
    with threadpool_limits(limits=1):
        found = set_statistics(spec["y"], spec["units"], spec["normal"], scores, pairs, BUDGETS,
                               BOOTSTRAP_DRAWS, SEED)
    positive = spec["y"] == 1
    found["binary_label_positive_share"] = float(np.mean(spec["standard"][positive] == 1))
    found["binary_label_undefined_share"] = float(np.mean(np.isnan(spec["standard"][positive])))
    found["seconds"] = time.monotonic() - started
    print(json.dumps({"stage": f"evaluate:{key}", "positives": found["positives"],
                      "seconds": round(found["seconds"], 1)}), flush=True)
    return key, found


def count_table(ptb: dict[str, pd.DataFrame], train: pd.DataFrame, calibration: pd.DataFrame,
                sph: pd.DataFrame, readouts: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """
    Positives and defined rows per group, source and split, and the training size of every readout.

    Parameters
    ----------
    ptb : dict[str, pd.DataFrame]
        PTB-XL ``train`` and ``development`` rows.
    train : pd.DataFrame
        Challenge training rows passing the quality policy.
    calibration : pd.DataFrame
        Challenge calibration rows.
    sph : pd.DataFrame
        SPH rows.
    readouts : dict[str, dict[str, np.ndarray]]
        Output of ``training_sets``.

    Returns
    -------
    dict[str, Any]
        ``positives[group][source:split]`` as (positives, defined rows), ``readouts[name]`` as (rows,
        positives), ``excluded`` quality-policy exclusions per source, and SPH frequent-PVC positives.
    """
    parts = {"ptbxl:train": ptb["train"], "ptbxl:development": ptb["development"], "sph:evaluation": sph}
    for split, frame in (("train", train), ("calibration", calibration)):
        for source, part in frame.groupby("source", sort=True):
            parts[f"{source}:{split}"] = part
    positives = {group: {name: (int((frame[group] == 1).sum()), int(frame[group].notna().sum()))
                         for name, frame in parts.items()}
                 for group in GROUPS}
    return {"positives": positives,
            "readouts": {name: (len(spec["y"]), int(spec["y"].sum())) for name, spec in readouts.items()},
            "sph_frequent_pvc": int(sph["frequent_pvc"].sum())}


def reading(evaluations: dict[str, Any], fitted: set[str]) -> dict[str, Any]:
    """
    Apply the pre-registered decision rule to the primary xECG SPH AUROC of every group.

    Parameters
    ----------
    evaluations : dict[str, Any]
        Statistics keyed by ``<group>@<set>``.
    fitted : set[str]
        Groups whose readout was fitted.

    Returns
    -------
    dict[str, Any]
        Per group: the reading (``usable``, ``not_usable``, ``undetermined``, ``too_few_sph_positives`` or
        ``not_fitted``) and the AUROC with its interval.
    """
    result = {}
    for group in GROUPS:
        entry: dict[str, Any] = {"reading": "not_fitted"}
        found = evaluations.get(f"{group}@sph")
        if group in fitted and found is None:
            entry = {"reading": "too_few_sph_positives"}
        elif found is not None:
            auroc = found["scores"][f"{group}:{PRIMARY_ENCODER}"]["auroc"]
            entry = {"reading": usability(auroc, USABLE_AUROC, USABLE_LOWER), "auroc": auroc}
        result[group] = entry
    return result


def load_all() -> dict[str, Any]:
    """
    Load and check every input, apply the training quality policy, and build the training sets.

    Returns
    -------
    dict[str, Any]
        Rows, features, readout training sets, counts and identity hashes.
    """
    feature_metadata, feature_hashes = feature_identity()
    ptb, ptb_x, ptb_identity = ptb_inputs()
    sph, sph_x, sph_hashes = sph_inputs()
    rows, challenge_x = challenge_table()
    in_train = (rows["split"] == "train").to_numpy()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    train = rows[in_train].copy()
    began = time.monotonic()
    train["reasons"] = quality_reasons(train)
    quality_seconds = time.monotonic() - began
    check_binary_rows(train)
    kept = (train["reasons"] == "").to_numpy()
    sph["standard"] = sph["primary"]
    calibration = rows[in_calibration].reset_index(drop=True)
    calibration["standard"] = np.where(calibration["evaluable"], calibration["primary"], np.nan)
    readouts = training_sets(ptb["train"], train[kept])
    counts = count_table(ptb, train[kept], calibration, sph, readouts)
    counts["excluded"] = {source: int((part["reasons"] != "").sum())
                          for source, part in train.groupby("source")}
    x = {name: np.concatenate([ptb_x[name]["train"], challenge_x[name][in_train][kept]]) for name in ENCODERS}
    inputs = {name: {"sph": sph_x[name], "development": ptb_x[name]["development"],
                     "calibration": challenge_x[name][in_calibration]} for name in ENCODERS}
    identity = {**ptb_identity, "sph_manifest": sph_hashes, "challenge_features": feature_hashes,
                "challenge_features_identity": feature_metadata["identity"],
                "challenge_split": {"rows.csv": sha256_file(SPLITS / "rows.csv"),
                                    "metadata.json": sha256_file(SPLITS / "metadata.json")},
                "challenge_headers": sha256_file(HEADERS),
                "ningbo_rows_sha256": sha256_file(NINGBO / "rows.csv"),
                "experiment022b": {name: sha256_file(PRIOR022B / name)
                                   for name in ("result.json", "predictions.npz", "training_rows.csv")}}
    return {"ptb": ptb, "sph": sph, "calibration": calibration, "train": train.assign(kept=kept),
            "readouts": readouts, "counts": counts, "x": x, "inputs": inputs, "identity": identity,
            "quality_seconds": quality_seconds}


def main() -> None:
    """Fit every readout once, reproduce the binary readout of 022b, and evaluate every group and set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", action="store_true", help="print the counts and stop before any fit")
    arguments = parser.parse_args()
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 032 v1 has already run")
    started = time.monotonic()
    loaded = load_all()
    counts = loaded["counts"]
    print(json.dumps({"stage": "counts", "quality_seconds": loaded["quality_seconds"], **counts}), flush=True)
    if arguments.counts:
        return
    if json.loads(json.dumps(counts)) != json.loads(json.dumps(EXPECTED)):
        raise ValueError("Counts differ from the protocol")
    readouts = loaded["readouts"]
    fitted = {name for name in READOUTS if readouts[name]["y"].sum() >= MIN_TRAINING_POSITIVES}
    to_fit = {name: spec for name, spec in readouts.items() if name == "binary" or name in fitted}

    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {name: executor.submit(fit_encoder, name, loaded["x"][name], to_fit, loaded["inputs"][name])
                   for name in ENCODERS}
        results = {name: future.result() for name, future in futures.items()}
    sph, development, calibration = loaded["sph"], loaded["ptb"]["development"], loaded["calibration"]
    reproduction = reproduce_022b(results, sph, calibration)

    sets = evaluation_sets(sph, development, calibration)
    tasks = {}
    for key, spec in sets.items():
        names, _ = score_names(spec["group"])
        if spec["group"] not in fitted or spec["y"].sum() < MIN_EVALUATION_POSITIVES:
            continue
        tasks[key] = {name: results[name.split(":")[1]]["scores"][spec["scores_of"]][name.split(":")[0]]
                      [spec["selection"]] for name in names}
    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = [executor.submit(evaluate_set, key, sets[key], scores) for key, scores in tasks.items()]
        evaluations = dict(future.result() for future in futures)
    skipped = {key: int(spec["y"].sum()) for key, spec in sets.items() if key not in evaluations}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(
        OUTPUT / "predictions.npz",
        sph_ecg_ids=sph["ecg_id"].to_numpy(dtype=str), sph_patient_ids=sph["patient_id"].to_numpy(dtype=str),
        development_record_ids=development.index.to_numpy(dtype=str),
        calibration_records=(calibration["source"] + ":" + calibration["record"]).to_numpy(dtype=str),
        **{f"{set_name}_{name}_{readout}": values for name, output in results.items()
           for set_name, by_readout in output["scores"].items() for readout, values in by_readout.items()})
    label_columns = ["source", "family", "record", "split", "evaluable", "primary", *GROUPS]
    loaded["train"][[*label_columns, "reasons", "kept"]].to_csv(OUTPUT / "training_rows.csv", index=False)
    calibration[label_columns].to_csv(OUTPUT / "calibration_rows.csv", index=False)
    result = {
        "status": "complete",
        "identity": {**loaded["identity"], "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "counts": counts, "groups": {group: {kind: {scheme: sorted(codes) for scheme, codes in lists.items()}
                                             for kind, lists in definition.items()}
                                     for group, definition in GROUPS.items()},
        "fitted": sorted(fitted), "encoders": list(ENCODERS), "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "budgets_per_mille": list(BUDGETS), "binary_difference_from_022b": reproduction,
        "fits": {name: output["fits"] for name, output in results.items()},
        "evaluations": evaluations, "skipped_sets_positives": skipped,
        "reading": reading(evaluations, fitted),
        "outputs_sha256": {name: sha256_file(OUTPUT / name)
                           for name in ("predictions.npz", "training_rows.csv", "calibration_rows.csv")},
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
        "quality_seconds": loaded["quality_seconds"], "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "reading": result["reading"]}), flush=True)


if __name__ == "__main__":
    main()
