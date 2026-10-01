"""Experiment 045: a two-layer explanation map and a detection ensemble, from saved 042 and 043 scores.

Layer 1 is Experiment 042's beat-aligned wave map ``U_B``; layer 2 is Experiment 043's ``attention_jepa``
token contributions. Both red thresholds are the same quantile of each layer's worst-unit scores on the
NORM-only normals, chosen so that 5% of them have either layer red. The ensemble is the mean of the
``logistic_concat`` and ``attention_jepa`` logits. Row, metric and statistic helpers are imported from the
042 and 043 runners, as the protocol documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from ecg_experiment.ann_heads import detection_reading
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    UnitMap,
    ecg_score,
    jepa_unit_map,
    plot_lead_marks,
    premature_hit,
    red_marks,
    top_lead,
    two_group_difference,
)
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.two_layer_map import choose_quantile, explain, mean_logit, quantile_grid, red_unit_count
from ecg_experiment.waveforms import LEADS
from scripts.experiments.run_ann_heads043 import (
    detection_statistics,
    evaluation_sets,
    full_length,
    map_contrasts,
    map_inputs,
    map_metrics,
    paired_contrast,
    ptb_inputs,
    public_map_metrics,
    receipt_checked,
    sph_inputs,
)
from scripts.experiments.run_lead_wave_maps042 import blas_architectures

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment045_two_layer_map_v1"
PRIOR042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PRIOR_STAGE1 = ROOT / "outputs/experiment043_ann_heads_v1/stage1"
PRIOR_STAGE2 = ROOT / "outputs/experiment043_ann_heads_v1/stage2"
FIGURES_COPY = ROOT / "docs/figures/experiment-045"
PROTOCOL = "docs/experiment-045-two-layer-map.md"
SOURCES = (
    PROTOCOL,
    "scripts/experiments/run_two_layer_map045.py",
    "ecg_experiment/two_layer_map.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/ann_heads.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/intervals.py",
    "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_lead_wave_maps042.py",
    "pyproject.toml",
    "uv.lock",
)
SEED = 45045
SEED042 = 42042
SEED043 = 43043
DRAWS = 2000
TARGET_SHARE = 0.05
SINGLE_QUANTILE = 0.95
LOCALIZATION_MARGIN = -0.10
EXPECTED_PVC = 73
EXPECTED_EXCLUSIONS = {"too_few_peaks": 0, "no_premature_beat": 11}
FS = 500
THREADS = 4
U_B_FIELDS = ("auroc", "average_precision", "any_red", "mean_red_units", "sensitivity_at_budget",
              "lead_contrast", "pvc_ecgs", "hit_rate", "chance_rate", "hit_minus_chance")
DETECTION_ARMS = {"logistic_concat": PRIOR_STAGE1, "attention_jepa": PRIOR_STAGE2}
LOG = logging.getLogger("experiment045")


def plain(value: Any) -> Any:
    """Return a JSON round trip of a value, so that tuples, NumPy scalars and floats compare as stored."""
    return json.loads(json.dumps(value))


def load_inputs() -> dict[str, Any]:
    """
    Load and receipt-check the saved 042 and 043 maps and predictions, and Experiment 042's rows.

    Returns
    -------
    dict[str, Any]
        ``rows``, ``windows``, ``exclusions``, ``U_B`` and ``attention_jepa`` maps in scored order, the
        prediction arrays per part, the detection ``sets``, and ``receipts``.

    Raises
    ------
    ValueError
        If a hash, a row order or a premature-beat count differs.
    """
    inputs = map_inputs()
    receipts = {to_stored(PRIOR042 / name): digest for name, digest in inputs["receipt"].items()}
    stage1 = receipt_checked(PRIOR_STAGE1, ("predictions.npz",))
    stage2 = receipt_checked(PRIOR_STAGE2, ("predictions.npz", "token_maps.npz"))
    receipts.update({to_stored(PRIOR_STAGE1 / name): digest for name, digest in stage1.items()})
    receipts.update({to_stored(PRIOR_STAGE2 / name): digest for name, digest in stage2.items()})
    recorded = json.loads((PRIOR_STAGE2 / "result.json").read_text())["inputs_042"]
    if recorded != inputs["receipt"]:
        raise ValueError("042's outputs differ from the hashes recorded by 043 Stage 2")
    scored = inputs["rows"]["scored"]["ecg_id"].to_numpy(np.int64)
    with np.load(PRIOR_STAGE2 / "token_maps.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], scored):
            raise ValueError("043 token maps differ from 042's scored rows")
        attention = [jepa_unit_map(values) for values in saved["attention_jepa"]]
    if len(inputs["windows"]) != EXPECTED_PVC or inputs["exclusions"] != EXPECTED_EXCLUSIONS:
        raise ValueError(f"Premature-beat windows differ from 042: {inputs['exclusions']}")
    if any(m is None for m in inputs["maps"]["U_B"]):
        raise ValueError("An ECG has no U_B units")

    ptb, _, _ = ptb_inputs()
    sph, _, _ = sph_inputs()
    sets = evaluation_sets({"development": ptb["development"], "sph": sph})
    with (np.load(PRIOR_STAGE1 / "predictions.npz") as first,
          np.load(PRIOR_STAGE2 / "predictions.npz") as second):
        record_ids = ptb["development"].index.to_numpy(dtype=str)
        sph_ids = sph["ecg_id"].to_numpy(dtype=str)
        positions = sets["sph"]["positions"]
        aligned = {
            "stage1_development": np.array_equal(first["development_record_ids"], record_ids),
            "stage2_development": np.array_equal(second["development_record_ids"], record_ids),
            "stage1_sph": np.array_equal(first["sph_ecg_ids"], sph_ids),
            "stage2_sph": np.array_equal(second["sph_ecg_ids"], sph_ids[positions]),
            "development_ecg_ids": np.array_equal(ptb["development"]["ecg_id"].to_numpy(np.int64), scored),
        }
        if not all(aligned.values()):
            raise ValueError(f"Prediction rows differ: {aligned}")
        parts = ("development", "sph")
        scores = {"R": {part: first[f"{part}_R"] for part in parts},
                  "logistic_concat": {part: first[f"{part}_logistic_concat"] for part in parts},
                  "attention_jepa": {"development": second["development_attention_jepa"],
                                     "sph": full_length(second["sph_attention_jepa"], positions, len(sph))}}
    return {**inputs, "U_B": inputs["maps"]["U_B"], "attention_jepa": attention, "scores": scores,
            "sets": sets, "receipts": receipts, "aligned": aligned,
            "record_ids": record_ids, "sph_ids": sph_ids[positions]}


def groups_of(rows: dict[str, Any]) -> dict[str, Any]:
    """
    Return the map test groups of Experiment 042's rows.

    Parameters
    ----------
    rows : dict[str, Any]
        042 ``select_rows`` output.

    Returns
    -------
    dict[str, Any]
        ``patients`` by ECG ID, evaluation ``labels`` and the ``normal``, ``positive``, ``pvc``, ``benign``,
        ``anterior`` and ``inferior`` ECG IDs, in 042's order.
    """
    scored = rows["scored"]
    labels = dict(zip(rows["evaluation"]["ecg_id"], rows["evaluation"]["standard"].astype(int), strict=True))
    return {"patients": dict(zip(scored["ecg_id"], scored["patient_id"], strict=True)), "labels": labels,
            "normal": [i for i, y in labels.items() if y == 0],
            "positive": [i for i, y in labels.items() if y == 1],
            **{name: rows[name]["ecg_id"].tolist() for name in ("pvc", "benign", "anterior", "inferior")}}


def single_sensitivity(rows: dict[str, Any], maps: list[UnitMap], threshold: float, seed: int
                       ) -> tuple[dict[str, float], dict[int, bool]]:
    """
    Return a single layer's any-red sensitivity at its threshold on the positives, as Experiment 042 does.

    Parameters
    ----------
    rows : dict[str, Any]
        042 ``select_rows`` output.
    maps : list[UnitMap]
        The layer's maps in scored order.
    threshold : float
        Red threshold.
    seed : int
        Bootstrap seed.

    Returns
    -------
    tuple[dict[str, float], dict[int, bool]]
        The bootstrap mean of any red on positives, and any red per positive ECG.
    """
    groups = groups_of(rows)
    by_id = dict(zip(rows["scored"]["ecg_id"], maps, strict=True))
    red = {i: bool(ecg_score(by_id[i]) > threshold) for i in groups["positive"]}
    found = bootstrap_mean(np.array([groups["patients"][i] for i in groups["positive"]]),
                           np.array([float(red[i]) for i in groups["positive"]]), DRAWS, seed)
    return found, red


def check_042(inputs: dict[str, Any]) -> dict[str, bool]:
    """
    Require ``U_B`` recomputed with seed 42042 to equal Experiment 042's reported metrics.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.

    Returns
    -------
    dict[str, bool]
        Per field, whether it equals 042's.

    Raises
    ------
    ValueError
        If any differs.
    """
    prior = json.loads((PRIOR042 / "result.json").read_text())
    metrics = map_metrics(inputs["rows"], {"U_B": inputs["U_B"]}, inputs["windows"], DRAWS, SEED042)
    found = dict(metrics["maps"]["U_B"])
    found["sensitivity_at_budget"], _ = single_sensitivity(inputs["rows"], inputs["U_B"],
                                                           metrics["thresholds"]["U_B"], SEED042)
    checks = {key: plain(found[key]) == prior["maps"]["U_B"][key] for key in U_B_FIELDS}
    checks["threshold"] = metrics["thresholds"]["U_B"] == prior["thresholds"]["U_B"]
    if not all(checks.values()):
        raise ValueError(f"U_B does not reproduce 042: {checks}")
    return checks


def check_043_maps(inputs: dict[str, Any]) -> dict[str, bool]:
    """
    Require 043's map metrics and contrasts, recomputed with seed 43043, to equal Stage 2's.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.

    Returns
    -------
    dict[str, bool]
        Whether the ``maps`` and ``attention_jepa_minus_U_B`` blocks are equal.

    Raises
    ------
    ValueError
        If either differs.
    """
    prior = json.loads((PRIOR_STAGE2 / "result.json").read_text())["maps"]
    metrics = map_metrics(inputs["rows"], {"U_B": inputs["U_B"], "attention_jepa": inputs["attention_jepa"]},
                          inputs["windows"], DRAWS, SEED043)
    contrasts = map_contrasts(metrics, "attention_jepa", "U_B", DRAWS, SEED043)
    public = public_map_metrics(metrics)
    checks = {"thresholds": plain(public["thresholds"]) == prior["thresholds"],
              "maps": plain(public["maps"]) == prior["maps"],
              "attention_jepa_minus_U_B": plain(contrasts) == prior["attention_jepa_minus_U_B"]}
    if not all(checks.values()):
        raise ValueError(f"043 maps do not reproduce: {checks}")
    return checks


def check_043_detection(inputs: dict[str, Any]) -> dict[str, bool]:
    """
    Require the saved logits to reproduce 043's AUROC, AP and differences from R, with seed 43043.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.

    Returns
    -------
    dict[str, bool]
        Per arm, whether its detection block is equal, and whether R's is.

    Raises
    ------
    ValueError
        If any differs.
    """
    scores = inputs["scores"]
    checks = {}
    for arm, folder in DETECTION_ARMS.items():
        prior = json.loads((folder / "result.json").read_text())["detection"]
        found = detection_statistics(inputs["sets"], {arm: scores[arm]}, scores["R"], DRAWS, SEED043)
        checks[arm] = plain(found[arm]) == prior[arm]
        checks[f"R_via_{arm}"] = plain(found["R"]) == prior["R"]
    if not all(checks.values()):
        raise ValueError(f"043 detection does not reproduce: {checks}")
    return checks


def worst_scores(rows: dict[str, Any], maps: list[UnitMap], ids: list[int]) -> np.ndarray:
    """Return the worst-unit score of each listed ECG."""
    by_id = dict(zip(rows["scored"]["ecg_id"], maps, strict=True))
    return np.array([ecg_score(by_id[i]) for i in ids])


def combined_metrics(inputs: dict[str, Any], choice: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Score the two-layer map with Experiment 042's tests, and keep the per-ECG values for the contrasts.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.
    choice : dict[str, Any]
        Output of ``choose_quantile``.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``metrics`` (public), ``per_ecg`` values and the ``explanations`` in scored order.
    """
    rows, windows = inputs["rows"], inputs["windows"]
    groups = groups_of(rows)
    patients = groups["patients"]
    ids = rows["scored"]["ecg_id"].tolist()
    first_t, second_t = choice["first_threshold"], choice["second_threshold"]
    explanations = [explain(a, b, first_t, second_t)
                    for a, b in zip(inputs["U_B"], inputs["attention_jepa"], strict=True)]
    by_id = dict(zip(ids, explanations, strict=True))
    pairs = dict(zip(ids, zip(inputs["U_B"], inputs["attention_jepa"], strict=True), strict=True))
    red = {i: e.first_red or e.second_red for i, e in by_id.items()}
    names = ("normal", "positive", "pvc", "benign")
    found: dict[str, Any] = {
        "ecgs_with_map": len(ids),
        "any_red": {g: float(np.mean([red[i] for i in groups[g]])) for g in names},
        "mean_red_units": {g: float(np.mean([red_unit_count(*pairs[i], first_t, second_t)
                                             for i in groups[g]])) for g in names},
        "explained_by_layer1": {g: float(np.mean([by_id[i].layer == 1 for i in groups[g]]))
                                for g in (*names, "anterior", "inferior")},
        "layer_red": {f"layer{k}": {g: float(np.mean([getattr(by_id[i], f"{which}_red") for i in groups[g]]))
                                    for g in names}
                      for k, which in ((1, "first"), (2, "second"))},
    }
    found["sensitivity_at_budget"] = bootstrap_mean(np.array([patients[i] for i in groups["positive"]]),
                                                    np.array([float(red[i]) for i in groups["positive"]]),
                                                    DRAWS, seed)
    contrasts = {}
    for region, leads, first, second in (("anterior", ANTERIOR, "anterior", "inferior"),
                                         ("inferior", INFERIOR, "inferior", "anterior")):
        a, b = groups[first], groups[second]
        hits_a = np.array([float(top_lead(by_id[i].units) in leads) for i in a])
        hits_b = np.array([float(top_lead(by_id[i].units) in leads) for i in b])
        contrasts[region] = two_group_difference(np.array([patients[i] for i in a]), hits_a,
                                                 np.array([patients[i] for i in b]), hits_b, DRAWS, seed)
    found["lead_contrast"] = contrasts
    included = list(windows)
    hit, chance = np.array([premature_hit(by_id[i].units, windows[i]) for i in included]).T
    found.update({"pvc_ecgs": len(included), "hit_rate": float(hit.mean()),
                  "chance_rate": float(chance.mean()),
                  "hit_minus_chance": bootstrap_mean(np.array([patients[i] for i in included]), hit - chance,
                                                     DRAWS, seed)})
    per_ecg = {"excess": dict(zip(included, hit - chance, strict=True)),
               "benign_red": {i: red[i] for i in groups["benign"]},
               "positive_red": {i: red[i] for i in groups["positive"]}}
    return {"metrics": found, "per_ecg": per_ecg, "explanations": explanations}


def single_layers(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Score ``U_B`` and ``attention_jepa`` alone at their own 95th-percentile thresholds, with this run's seed.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``map_metrics`` output with ``sensitivity_at_budget`` added per map, and positive red per ECG.
    """
    maps = {"U_B": inputs["U_B"], "attention_jepa": inputs["attention_jepa"]}
    metrics = map_metrics(inputs["rows"], maps, inputs["windows"], DRAWS, seed)
    for name, unit_maps in maps.items():
        sensitivity, red = single_sensitivity(inputs["rows"], unit_maps, metrics["thresholds"][name], seed)
        metrics["maps"][name]["sensitivity_at_budget"] = sensitivity
        metrics["per_ecg"][name]["positive_red"] = red
    return metrics


def paired_against_u_b(combined: dict[str, Any], single: dict[str, Any], patients: dict[int, Any],
                       seed: int) -> dict[str, Any]:
    """
    Return the per-ECG paired contrasts of the combined map minus ``U_B`` alone, and the reading.

    Parameters
    ----------
    combined : dict[str, Any]
        Output of ``combined_metrics``.
    single : dict[str, Any]
        Output of ``single_layers``.
    patients : dict[int, Any]
        Patient ID by ECG ID.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``hit_minus_chance``, ``benign_any_red`` and ``positive_any_red`` differences, and ``reading``.
    """
    mine, theirs = combined["per_ecg"], single["per_ecg"]["U_B"]
    found = {}
    for key, field in (("hit_minus_chance", "excess"), ("benign_any_red", "benign_red"),
                       ("positive_any_red", "positive_red")):
        ids = list(mine[field])
        if set(ids) != set(theirs[field]):
            raise ValueError(f"The two maps cover different ECGs for {key}")
        found[key] = bootstrap_mean(np.array([patients[i] for i in ids]),
                                    np.array([float(mine[field][i]) - float(theirs[field][i]) for i in ids]),
                                    DRAWS, seed)
    keeps = found["hit_minus_chance"]["ci_low"] > LOCALIZATION_MARGIN
    gains = {"lead": combined["metrics"]["lead_contrast"]["anterior"]["ci_low"] > 0,
             "benign": found["benign_any_red"]["ci_high"] < 0,
             "positive": found["positive_any_red"]["ci_low"] > 0}
    found["reading"] = {"keeps_premature_localization": bool(keeps),
                        "gains": {key: bool(value) for key, value in gains.items()},
                        "improves_on_U_B": bool(keeps and any(gains.values()))}
    return found


def ensemble(inputs: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Score the mean-logit ensemble E against ``logistic_concat`` and R.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``scores`` of E per part, ``detection`` statistics, the primary ``E_minus_logistic_concat`` and
        ``readings``.
    """
    scores, sets = inputs["scores"], inputs["sets"]
    combined = {part: mean_logit(scores["logistic_concat"][part], scores["attention_jepa"][part])
                for part in ("development", "sph")}
    arms = {"E": combined, "logistic_concat": scores["logistic_concat"],
            "attention_jepa": scores["attention_jepa"]}
    statistics = detection_statistics(sets, arms, scores["R"], DRAWS, seed)
    primary = paired_contrast(sets, combined, scores["logistic_concat"], DRAWS, seed)
    readings = {"E_vs_logistic_concat": detection_reading(primary["sph"]["ci_low"],
                                                          primary["full"]["difference"]),
                "E_vs_R": detection_reading(statistics["E"]["minus_R"]["sph"]["ci_low"],
                                            statistics["E"]["minus_R"]["full"]["difference"])}
    return {"scores": combined, "detection": statistics, "E_minus_logistic_concat": primary,
            "readings": readings}


def draw_examples(inputs: dict[str, Any], choice: dict[str, Any], explanations: list[Any], folder: Path
                  ) -> dict[str, int]:
    """
    Save the 041 example figures with both layers' red marks per lead.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of ``load_inputs``.
    choice : dict[str, Any]
        Output of ``choose_quantile``.
    explanations : list[Any]
        ``combined_metrics`` explanations in scored order.
    folder : Path
        Destination folder.

    Returns
    -------
    dict[str, int]
        ECG ID of each example.
    """
    folder.mkdir(parents=True, exist_ok=True)
    scored = inputs["rows"]["scored"].reset_index()
    examples = json.loads((PRIOR042 / "result.json").read_text())["examples"]
    for name, ecg_id in examples.items():
        index = int(scored.index[scored["ecg_id"] == ecg_id][0])
        found = explanations[index]
        top = int(found.units.scores.argmax())
        marks = {"Layer 1, U_B": red_marks(inputs["U_B"][index], choice["first_threshold"]),
                 "Layer 2, attention_jepa": red_marks(inputs["attention_jepa"][index],
                                                      choice["second_threshold"])}
        title = (f"{name}: PTB-XL ECG {ecg_id}; explained by layer {found.layer}, "
                 f"{LEADS[int(found.units.leads[top])]} at {found.units.starts[top]:.2f} s")
        figure = plot_lead_marks(read_ptb_float64(scored.at[index, "filename_hr"]), FS, marks, title)
        figure.savefig(folder / f"{name}_{ecg_id}.png", dpi=110)
    return examples


def save_arrays(folder: Path, inputs: dict[str, Any], explanations: list[Any], scores: dict[str, np.ndarray]
                ) -> None:
    """Write the per-ECG combined map and the ensemble logits."""
    leads, starts, ends = [], [], []
    for found in explanations:
        top = int(found.units.scores.argmax())
        leads.append(int(found.units.leads[top]))
        starts.append(float(found.units.starts[top]))
        ends.append(float(found.units.ends[top]))
    write_npz_atomic(folder / "combined_map.npz",
                     ecg_ids=inputs["rows"]["scored"]["ecg_id"].to_numpy(np.int64),
                     layer1_worst=np.array([ecg_score(m) for m in inputs["U_B"]]),
                     layer2_worst=np.array([ecg_score(m) for m in inputs["attention_jepa"]]),
                     layer1_red=np.array([e.first_red for e in explanations]),
                     layer2_red=np.array([e.second_red for e in explanations]),
                     explanation_layer=np.array([e.layer for e in explanations], dtype=np.int64),
                     explanation_lead=np.array(leads, dtype=np.int64), explanation_start=np.array(starts),
                     explanation_end=np.array(ends))
    write_npz_atomic(folder / "ensemble.npz", development_record_ids=inputs["record_ids"],
                     development_E=scores["development"], sph_ecg_ids=inputs["sph_ids"],
                     sph_E=scores["sph"][inputs["sets"]["sph"]["positions"]])


def identity(receipts: dict[str, str]) -> dict[str, Any]:
    """Return the input receipts, source hashes and run environment."""
    return {"inputs": receipts, "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
            "git_head": git_head(ROOT), "openblas_architectures": blas_architectures()}


def run(output: Path, integrity_only: bool) -> None:
    """
    Run the experiment end to end and write its outputs.

    Parameters
    ----------
    output : Path
        Final output folder; it must not exist.
    integrity_only : bool
        Stop after the integrity checks, before any score of this experiment, and remove the partial folder.
    """
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(partial / "run.log")])
    inputs = load_inputs()
    run_identity = identity(inputs["receipts"])
    integrity = {"aligned": inputs["aligned"], "pvc_exclusions": inputs["exclusions"],
                 "pvc_included": len(inputs["windows"])}
    integrity["U_B_vs_042"] = check_042(inputs)
    integrity["maps_vs_043"] = check_043_maps(inputs)
    integrity["detection_vs_043"] = check_043_detection(inputs)
    LOG.info("integrity passed at %.1f s: %s", time.perf_counter() - started, integrity)
    if integrity_only:
        logging.shutdown()
        shutil.rmtree(partial)
        return

    groups = groups_of(inputs["rows"])
    first_normal = worst_scores(inputs["rows"], inputs["U_B"], groups["normal"])
    second_normal = worst_scores(inputs["rows"], inputs["attention_jepa"], groups["normal"])
    choice = choose_quantile(first_normal, second_normal, TARGET_SHARE)
    LOG.info("quantile choice: %s", choice)
    single = single_layers(inputs, SEED)
    combined = combined_metrics(inputs, choice, SEED)
    contrasts = paired_against_u_b(combined, single, groups["patients"], SEED)
    LOG.info("combined map: %s", contrasts)
    detection = ensemble(inputs, SEED)
    LOG.info("ensemble: %s %s", detection["E_minus_logistic_concat"], detection["readings"])

    examples = None
    if contrasts["reading"]["improves_on_U_B"]:
        examples = draw_examples(inputs, choice, combined["explanations"], partial / "figures")
    save_arrays(partial, inputs, combined["explanations"], detection["scores"])
    outputs = {name: sha256_file(partial / name) for name in ("combined_map.npz", "ensemble.npz")}
    if examples is not None:
        outputs.update({f"figures/{path.name}": sha256_file(path)
                        for path in sorted((partial / "figures").glob("*.png"))})
    write_json_atomic(partial / "result.json", {
        "status": "completed", "identity": run_identity, "integrity": integrity,
        "counts": {name: len(groups[name]) for name in
                   ("normal", "positive", "pvc", "benign", "anterior", "inferior")},
        "quantile": {**choice, "grid": [float(quantile_grid()[0]), float(quantile_grid()[-1]),
                                        len(quantile_grid())], "target_share": TARGET_SHARE},
        "maps": {"thresholds": {**single["thresholds"], "combined_layer1": choice["first_threshold"],
                                "combined_layer2": choice["second_threshold"]},
                 "U_B": single["maps"]["U_B"], "attention_jepa": single["maps"]["attention_jepa"],
                 "combined": combined["metrics"]},
        "combined_minus_U_B": contrasts, "map_reading": contrasts["reading"],
        "ensemble": {key: detection[key] for key in ("detection", "E_minus_logistic_concat", "readings")},
        "examples": examples, "outputs_sha256": outputs, "seed": SEED, "draws": DRAWS,
        "total_seconds": time.perf_counter() - started, "calibration_test_evaluated": False,
        "challenge_test_read": False, "ptbxl_test_read": False,
    })
    logging.shutdown()
    partial.rename(output)
    if examples is not None:
        FIGURES_COPY.mkdir(parents=True, exist_ok=True)
        for path in sorted((output / "figures").glob("*.png")):
            shutil.copy2(path, FIGURES_COPY / path.name)
    LOG.info("done in %.1f s", time.perf_counter() - started)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--integrity-only", action="store_true",
                        help="stop after the integrity checks, before any score of this experiment")
    arguments = parser.parse_args()
    for folder in (arguments.output, arguments.output.with_name(arguments.output.name + ".partial")):
        if folder.exists():
            raise FileExistsError(f"Refusing to overwrite {folder}")
    with threadpool_limits(limits=THREADS):
        run(arguments.output, arguments.integrity_only)


if __name__ == "__main__":
    main()
