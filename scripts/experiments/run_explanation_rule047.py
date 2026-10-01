"""Experiment 047: an explanation rule keyed to the referral decision, from saved 042, 043 and 045 scores.

An ECG is referred when pipeline v3's readout logit (043's ``logistic_concat``) is above the 95th
percentile of the NORM-only normals' logits. A referred ECG is explained by 042's ``U_B`` top unit if
``U_B`` is red at its own threshold, otherwise by 043's ``attention_jepa`` top token. The comparators are
each map alone on the same referred ECGs. Loading, receipt and reproduction checks are imported from the 045
runner, as the protocol documents.
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

from ecg_experiment.explanation_rule import (
    explanation_metrics,
    group_rates,
    layer_shares,
    paired_difference,
    referral_threshold,
    referred_ids,
    rule_marks,
    rule_reading,
)
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.lead_wave_maps import ecg_score, plot_lead_marks, red_marks
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.two_layer_map import explain, mean_logit
from ecg_experiment.waveforms import LEADS
from scripts.experiments.run_lead_wave_maps042 import blas_architectures
from scripts.experiments.run_two_layer_map045 import (
    check_042,
    check_043_detection,
    check_043_maps,
    groups_of,
    load_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment047_explanation_rule_v1"
PRIOR042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PRIOR_STAGE2 = ROOT / "outputs/experiment043_ann_heads_v1/stage2"
PRIOR045 = ROOT / "outputs/experiment045_two_layer_map_v1"
FIGURES_COPY = ROOT / "docs/figures/experiment-047"
PROTOCOL = "docs/experiment-047-explanation-rule.md"
SOURCES = (
    PROTOCOL,
    "scripts/experiments/run_explanation_rule047.py",
    "ecg_experiment/explanation_rule.py",
    "ecg_experiment/two_layer_map.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/intervals.py",
    "scripts/experiments/run_two_layer_map045.py",
    "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_lead_wave_maps042.py",
    "pyproject.toml",
    "uv.lock",
)
SEED = 47047
DRAWS = 2000
QUANTILE = 0.95
FS = 500
THREADS = 4
ARMS = ("rule", "U_B", "attention_jepa")
COMPARATORS = ("U_B", "attention_jepa")
LOG = logging.getLogger("experiment047")


def check_045(inputs: dict[str, Any]) -> tuple[dict[str, bool], dict[str, str]]:
    """
    Require 045's saved ensemble to match its receipt and to equal E recomputed from the saved logits.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.

    Returns
    -------
    tuple[dict[str, bool], dict[str, str]]
        The checks, and the receipt of ``ensemble.npz``.

    Raises
    ------
    ValueError
        If any check fails.
    """
    path = PRIOR045 / "ensemble.npz"
    digest = sha256_file(path)
    recorded = json.loads((PRIOR045 / "result.json").read_text())["outputs_sha256"]["ensemble.npz"]
    scores = inputs["scores"]
    found = mean_logit(scores["logistic_concat"]["development"], scores["attention_jepa"]["development"])
    with np.load(path) as saved:
        checks = {"receipt": digest == recorded,
                  "record_ids": np.array_equal(saved["development_record_ids"], inputs["record_ids"]),
                  "development_E": np.array_equal(saved["development_E"], found)}
    if not all(checks.values()):
        raise ValueError(f"045's ensemble does not reproduce: {checks}")
    return checks, {to_stored(path): digest}


def single_thresholds(inputs: dict[str, Any], groups: dict[str, Any]) -> dict[str, float]:
    """
    Return each map's own red threshold, required to equal 042's (``U_B``) and 043's (``attention_jepa``).

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.

    Returns
    -------
    dict[str, float]
        Threshold per map.

    Raises
    ------
    ValueError
        If a threshold differs from the reported one.
    """
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    found = {}
    for name in COMPARATORS:
        by_id = dict(zip(ids, inputs[name], strict=True))
        found[name] = float(np.quantile([ecg_score(by_id[i]) for i in groups["normal"]], QUANTILE))
    prior = {"U_B": json.loads((PRIOR042 / "result.json").read_text())["thresholds"]["U_B"],
             "attention_jepa": json.loads((PRIOR_STAGE2 / "result.json").read_text())["maps"]["thresholds"][
                 "attention_jepa"]}
    if found != prior:
        raise ValueError(f"Map thresholds differ: {found} against {prior}")
    return found


def readout_logits(inputs: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Return the development logit by ECG ID of pipeline v3's readout and of E."""
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    scores = inputs["scores"]
    first, second = scores["logistic_concat"]["development"], scores["attention_jepa"]["development"]
    arrays = {"logistic_concat": first, "E": mean_logit(first, second)}
    return {name: dict(zip(ids, values.tolist(), strict=True)) for name, values in arrays.items()}


def analyse(inputs: dict[str, Any], groups: dict[str, Any], logits: dict[int, float],
            thresholds: dict[str, float], seed: int) -> dict[str, Any]:
    """
    Refer by one readout and score the rule and both single maps on the referred ECGs.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    logits : dict[int, float]
        Readout logit by ECG ID.
    thresholds : dict[str, float]
        Each map's own red threshold.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        Public ``result`` and the per-ECG ``explanations`` of the referred ECGs.
    """
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    patients, windows = groups["patients"], inputs["windows"]
    cutoff = referral_threshold(np.array([logits[i] for i in groups["normal"]]))
    referred = referred_ids(ids, np.array([logits[i] for i in ids]), cutoff)
    chosen = set(referred)
    maps = {name: dict(zip(ids, inputs[name], strict=True)) for name in COMPARATORS}
    explanations = {i: explain(maps["U_B"][i], maps["attention_jepa"][i], thresholds["U_B"],
                               thresholds["attention_jepa"]) for i in referred}
    units = {"rule": {i: e.units for i, e in explanations.items()},
             **{name: {i: maps[name][i] for i in referred} for name in COMPARATORS}}
    named = {"normal": groups["normal"], "positive": groups["positive"], "pvc": groups["pvc"],
             "pvc_window": list(windows), "benign": groups["benign"], "anterior": groups["anterior"],
             "inferior": groups["inferior"]}
    positive = groups["positive"]
    sensitivity = bootstrap_mean(np.array([patients[i] for i in positive]),
                                 np.array([float(i in chosen) for i in positive]), DRAWS, seed)
    scored = {name: explanation_metrics(units[name], patients, windows, groups["anterior"],
                                        groups["inferior"], DRAWS, seed) for name in ARMS}
    paired = {f"rule_minus_{name}": paired_difference(scored["rule"], scored[name], patients, DRAWS, seed)
              for name in COMPARATORS}
    anterior_low = scored["rule"]["metrics"]["lead_contrast"]["anterior"]["ci_low"]
    readings = {name: rule_reading(paired[f"rule_minus_{name}"]["hit_minus_chance"]["ci_low"], anterior_low)
                for name in COMPARATORS}
    layers = {i: e.layer for i, e in explanations.items()}
    result = {"threshold": cutoff, "referred": len(referred),
              "referred_counts": {name: int(sum(i in chosen for i in members))
                                  for name, members in named.items()},
              "group_sizes": {name: len(members) for name, members in named.items()},
              "referral_rate": group_rates(chosen, named), "positive_referral": sensitivity,
              "explained_by_layer": layer_shares(layers, {"referred": referred, **named}),
              "attention_red_when_supplying": float(np.mean([e.second_red for e in explanations.values()
                                                             if e.layer == 2])),
              "maps": {name: scored[name]["metrics"] for name in ARMS}, "paired": paired,
              "readings": {"vs_U_B": readings["U_B"], "vs_attention_jepa": readings["attention_jepa"]}}
    return {"result": result, "explanations": explanations}


def draw_examples(inputs: dict[str, Any], explanations: dict[int, Any], thresholds: dict[str, float],
                  folder: Path) -> dict[str, int]:
    """
    Save the 041 example figures: the rule's marks, and each map's red units at its own threshold.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    explanations : dict[int, Any]
        Explanation of each referred ECG under the primary readout.
    thresholds : dict[str, float]
        Each map's own red threshold.
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
    layer_names = {1: "U_B", 2: "attention_jepa"}
    for name, ecg_id in examples.items():
        index = int(scored.index[scored["ecg_id"] == ecg_id][0])
        found = explanations.get(ecg_id)
        if found is None:
            rule, title = [], f"{name}: PTB-XL ECG {ecg_id}; not referred"
        else:
            source = layer_names[found.layer]
            rule = rule_marks(found, thresholds[source])
            top = int(found.units.scores.argmax())
            title = (f"{name}: PTB-XL ECG {ecg_id}; referred, explained by {source}, "
                     f"{LEADS[int(found.units.leads[top])]} at {found.units.starts[top]:.2f} s")
        marks = {"Rule (referred ECGs only)": rule,
                 "U_B alone": red_marks(inputs["U_B"][index], thresholds["U_B"]),
                 "attention_jepa alone": red_marks(inputs["attention_jepa"][index],
                                                   thresholds["attention_jepa"])}
        figure = plot_lead_marks(read_ptb_float64(scored.at[index, "filename_hr"]), FS, marks, title)
        figure.savefig(folder / f"{name}_{ecg_id}.png", dpi=110)
    return examples


def explanation_arrays(ids: list[int], explanations: dict[int, Any]) -> dict[str, np.ndarray]:
    """Return per-ECG referral, supplying layer (0 if not referred) and explanation unit."""
    layer, lead = np.zeros(len(ids), dtype=np.int64), np.full(len(ids), -1, dtype=np.int64)
    start, end = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    for row, i in enumerate(ids):
        if i in explanations:
            units = explanations[i].units
            top = int(units.scores.argmax())
            layer[row], lead[row] = explanations[i].layer, int(units.leads[top])
            start[row], end[row] = float(units.starts[top]), float(units.ends[top])
    return {"referred": layer > 0, "layer": layer, "lead": lead, "start": start, "end": end}


def save_arrays(folder: Path, inputs: dict[str, Any], logits: dict[str, dict[int, float]],
                found: dict[str, dict[str, Any]]) -> None:
    """Write the per-ECG readout logits, referrals and explanations of both readouts."""
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    arrays: dict[str, np.ndarray] = {"ecg_ids": np.array(ids, dtype=np.int64),
                                     "U_B_worst": np.array([ecg_score(m) for m in inputs["U_B"]]),
                                     "attention_jepa_worst": np.array([ecg_score(m)
                                                                       for m in inputs["attention_jepa"]])}
    for readout, values in logits.items():
        arrays[f"{readout}_logit"] = np.array([values[i] for i in ids])
        for key, value in explanation_arrays(ids, found[readout]["explanations"]).items():
            arrays[f"{readout}_{key}"] = value
    write_npz_atomic(folder / "explanations.npz", **arrays)


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
    log_path = partial / "run.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)])
    inputs = load_inputs()
    groups = groups_of(inputs["rows"])
    integrity: dict[str, Any] = {"aligned": inputs["aligned"], "pvc_exclusions": inputs["exclusions"],
                                 "pvc_included": len(inputs["windows"])}
    integrity["U_B_vs_042"] = check_042(inputs)
    integrity["maps_vs_043"] = check_043_maps(inputs)
    integrity["detection_vs_043"] = check_043_detection(inputs)
    integrity["E_vs_045"], receipt045 = check_045(inputs)
    thresholds = single_thresholds(inputs, groups)
    integrity["map_thresholds_equal_reported"] = True
    receipts = {**inputs["receipts"], **receipt045}
    run_identity = identity(receipts)
    LOG.info("integrity passed at %.1f s: %s", time.perf_counter() - started, integrity)
    if integrity_only:
        logging.shutdown()
        shutil.rmtree(partial)
        return

    logits = readout_logits(inputs)
    found = {readout: analyse(inputs, groups, values, thresholds, SEED) for readout, values in logits.items()}
    for readout, value in found.items():
        LOG.info("%s: referred %d, readings %s", readout, value["result"]["referred"],
                 value["result"]["readings"])

    examples = None
    if found["logistic_concat"]["result"]["readings"]["vs_U_B"]["improves"]:
        examples = draw_examples(inputs, found["logistic_concat"]["explanations"], thresholds,
                                 partial / "figures")
    save_arrays(partial, inputs, logits, found)
    outputs = {"explanations.npz": sha256_file(partial / "explanations.npz")}
    if examples is not None:
        outputs.update({f"figures/{path.name}": sha256_file(path)
                        for path in sorted((partial / "figures").glob("*.png"))})
    write_json_atomic(partial / "result.json", {
        "status": "completed", "identity": run_identity, "integrity": integrity,
        "counts": {name: len(groups[name]) for name in
                   ("normal", "positive", "pvc", "benign", "anterior", "inferior")},
        "map_thresholds": thresholds, "primary_readout": "logistic_concat",
        "readouts": {readout: value["result"] for readout, value in found.items()},
        "primary_reading": found["logistic_concat"]["result"]["readings"]["vs_U_B"],
        "examples": examples, "outputs_sha256": outputs, "seed": SEED, "draws": DRAWS, "quantile": QUANTILE,
        "total_seconds": time.perf_counter() - started, "calibration_test_evaluated": False,
        "challenge_test_read": False, "ptbxl_test_read": False,
    })
    LOG.info("done in %.1f s", time.perf_counter() - started)
    logging.shutdown()
    partial.rename(output)
    if examples is not None:
        FIGURES_COPY.mkdir(parents=True, exist_ok=True)
        for path in sorted((output / "figures").glob("*.png")):
            shutil.copy2(path, FIGURES_COPY / path.name)


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
