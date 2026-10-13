"""Experiment 048: switch the explanation on pipeline v3's PVC finding head, from saved scores.

Referral is Experiment 047's: pipeline v3's readout logit above the 95th percentile of the NORM-only normals'.
A referred ECG is explained by 042's ``U_B`` top unit when the xECG PVC head's z-score is above a quantile
of the normals' z-scores, otherwise by 043's ``attention_jepa`` top token. Loading and integrity checks are
imported from the 045 and 047 runners, as the protocol documents.
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
from ecg_experiment.lead_wave_maps import red_marks
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.pvc_switch import LAYER_NAMES, plot_layer_marks, pvc_z, switch_explain, switch_threshold
from ecg_experiment.two_layer_map import explain
from ecg_experiment.waveforms import LEADS
from scripts.experiments.run_ann_heads043 import ptb_inputs, sph_inputs
from scripts.experiments.run_explanation_rule047 import analyse as analyse047
from scripts.experiments.run_explanation_rule047 import check_045, readout_logits, single_thresholds
from scripts.experiments.run_lead_wave_maps042 import blas_architectures
from scripts.experiments.run_two_layer_map045 import (
    check_042,
    check_043_detection,
    check_043_maps,
    groups_of,
    load_inputs,
    plain,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment048_pvc_switch_v1"
PRIOR042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PRIOR044 = ROOT / "outputs/experiment044_pipeline_v3_v1"
PRIOR047 = ROOT / "outputs/experiment047_explanation_rule_v1"
PIPELINE_V3 = ROOT / "docs/pipeline-v3.md"
FIGURES_COPY = ROOT / "docs/figures/experiment-048"
PROTOCOL = "docs/experiment-048-pvc-switch.md"
SOURCES = (
    PROTOCOL,
    "scripts/experiments/run_pvc_switch048.py",
    "ecg_experiment/pvc_switch.py",
    "ecg_experiment/explanation_rule.py",
    "ecg_experiment/two_layer_map.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/finding_screen.py",
    "ecg_experiment/hybrid_score.py",
    "ecg_experiment/pipeline_v2.py",
    "ecg_experiment/intervals.py",
    "scripts/experiments/run_explanation_rule047.py",
    "scripts/experiments/run_two_layer_map045.py",
    "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_lead_wave_maps042.py",
    "scripts/experiments/run_rhythm_findings032.py",
    "pyproject.toml",
    "uv.lock",
)
SEED = 48048
SEED047 = 47047
DRAWS = 2000
PRIMARY_QUANTILE = 0.975
QUANTILES = (0.95, 0.975, 0.99)
Z_TOLERANCE = 1e-10
FS = 500
THREADS = 4
ARMS = ("rule", "U_B", "attention_jepa", "rule047")
COMPARATORS = ("U_B", "attention_jepa")
LOG = logging.getLogger("experiment048")


def check_047(inputs: dict[str, Any], groups: dict[str, Any], thresholds: dict[str, float]
              ) -> tuple[dict[str, bool], dict[str, str]]:
    """
    Require 047's saved explanations to match their receipt and its analysis to reproduce exactly.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    thresholds : dict[str, float]
        Each map's own red threshold.

    Returns
    -------
    tuple[dict[str, bool], dict[str, str]]
        The checks, and the receipt of 047's ``explanations.npz``.

    Raises
    ------
    ValueError
        If any check fails.
    """
    prior = json.loads((PRIOR047 / "result.json").read_text())
    digest = sha256_file(PRIOR047 / "explanations.npz")
    checks = {"receipt": digest == prior["outputs_sha256"]["explanations.npz"]}
    for readout, logits in readout_logits(inputs).items():
        found = analyse047(inputs, groups, logits, thresholds, SEED047)["result"]
        checks[readout] = plain(found) == prior["readouts"][readout]
    if not all(checks.values()):
        raise ValueError(f"047 does not reproduce: {checks}")
    return checks, {to_stored(PRIOR047 / "explanations.npz"): digest}


def pvc_scores(inputs: dict[str, Any]) -> tuple[dict[int, float], dict[str, Any], dict[str, str]]:
    """
    Score the xECG PVC head of pipeline v3 on the development ECGs, checked against 044 on SPH.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.

    Returns
    -------
    tuple[dict[int, float], dict[str, Any], dict[str, str]]
        z_PVC by development ECG ID, the checks, and the receipts of 044's files.

    Raises
    ------
    ValueError
        If a hash, a row order or the SPH reproduction differs.
    """
    recorded = json.loads((PRIOR044 / "result.json").read_text())["outputs_sha256"]
    receipts = {name: sha256_file(PRIOR044 / name) for name in ("pipeline_v3_heads.npz", "predictions.npz")}
    checks: dict[str, Any] = {f"{name}_receipt": digest == recorded[name]
                              for name, digest in receipts.items()}
    checks["heads_in_pipeline_v3_doc"] = receipts["pipeline_v3_heads.npz"] in PIPELINE_V3.read_text()
    ptb, ptb_x, _ = ptb_inputs()
    sph, sph_x, _ = sph_inputs()
    with (np.load(PRIOR044 / "pipeline_v3_heads.npz") as saved,
          np.load(PRIOR044 / "predictions.npz") as predictions):
        parameters = {name: saved[name] for name in saved.files}
        checks["sph_ids"] = bool(np.array_equal(predictions["sph_ecg_ids"],
                                                sph["ecg_id"].to_numpy(dtype=str)))
        checks["development_ids"] = bool(np.array_equal(predictions["development_record_ids"],
                                                        inputs["record_ids"]))
        checks["development_rows"] = bool(np.array_equal(ptb["development"].index.to_numpy(dtype=str),
                                                         inputs["record_ids"]))
        sph_z = pvc_z(parameters, sph_x["xecg"])
        checks["sph_z_max_difference"] = float(np.abs(sph_z - predictions["sph_v2_z_pvc"]).max())
    checks["sph_z_reproduces"] = checks["sph_z_max_difference"] <= Z_TOLERANCE
    if not all(value for key, value in checks.items() if key != "sph_z_max_difference"):
        raise ValueError(f"044's PVC head does not reproduce: {checks}")
    development = pvc_z(parameters, ptb_x["xecg"]["development"])
    ids = ptb["development"]["ecg_id"].astype(int).tolist()
    return (dict(zip(ids, development.tolist(), strict=True)), checks,
            {to_stored(PRIOR044 / name): digest for name, digest in receipts.items()})


def analyse(inputs: dict[str, Any], groups: dict[str, Any], logits: dict[int, float], z: dict[int, float],
            switch: float, thresholds: dict[str, float], seed: int) -> dict[str, Any]:
    """
    Refer by one readout, switch on z_PVC, and score the rule, both single maps and 047's rule.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    logits : dict[int, float]
        Readout logit by ECG ID.
    z : dict[int, float]
        z_PVC by ECG ID.
    switch : float
        Switch threshold; ``U_B`` explains when z_PVC is strictly above it.
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
    first_t, second_t = thresholds["U_B"], thresholds["attention_jepa"]
    explanations = {i: switch_explain(maps["U_B"][i], maps["attention_jepa"][i], z[i] > switch, first_t,
                                      second_t) for i in referred}
    units = {"rule": {i: e.units for i, e in explanations.items()},
             **{name: {i: maps[name][i] for i in referred} for name in COMPARATORS},
             "rule047": {i: explain(maps["U_B"][i], maps["attention_jepa"][i], first_t, second_t).units
                         for i in referred}}
    named = {"normal": groups["normal"], "positive": groups["positive"], "pvc": groups["pvc"],
             "pvc_window": list(windows), "benign": groups["benign"], "anterior": groups["anterior"],
             "inferior": groups["inferior"]}
    positive = groups["positive"]
    sensitivity = bootstrap_mean(np.array([patients[i] for i in positive]),
                                 np.array([float(i in chosen) for i in positive]), DRAWS, seed)
    scored = {name: explanation_metrics(units[name], patients, windows, groups["anterior"],
                                        groups["inferior"], DRAWS, seed) for name in ARMS}
    paired = {f"rule_minus_{name}": paired_difference(scored["rule"], scored[name], patients, DRAWS, seed)
              for name in (*COMPARATORS, "rule047")}
    anterior_low = scored["rule"]["metrics"]["lead_contrast"]["anterior"]["ci_low"]
    readings = {f"vs_{name}": rule_reading(paired[f"rule_minus_{name}"]["hit_minus_chance"]["ci_low"],
                                           anterior_low) for name in COMPARATORS}
    layers = {i: e.layer for i, e in explanations.items()}
    to_u_b = {name: int(sum(layers[i] == 1 for i in named[name] if i in chosen))
              for name in ("anterior", "inferior", "pvc_window")}
    result = {"threshold": cutoff, "switch_threshold": switch, "referred": len(referred),
              "referred_counts": {name: int(sum(i in chosen for i in members))
                                  for name, members in named.items()},
              "group_sizes": {name: len(members) for name, members in named.items()},
              "referral_rate": group_rates(chosen, named), "positive_referral": sensitivity,
              "explained_by_layer": layer_shares(layers, {"referred": referred, **named}),
              "sent_to_U_B": to_u_b,
              "switch_on_all": {name: float(np.mean([z[i] > switch for i in members]))
                                for name, members in named.items()},
              "U_B_red_when_supplying": float(np.mean([e.first_red for e in explanations.values()
                                                       if e.layer == 1])),
              "attention_red_when_supplying": float(np.mean([e.second_red for e in explanations.values()
                                                             if e.layer == 2])),
              "maps": {name: scored[name]["metrics"] for name in ARMS}, "paired": paired,
              "readings": readings}
    return {"result": result, "explanations": explanations}


def draw_examples(inputs: dict[str, Any], explanations: dict[int, Any], thresholds: dict[str, float],
                  folder: Path) -> dict[str, int]:
    """
    Save the 041 example figures: the rule's marks coloured by layer, and each map's own red units.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    explanations : dict[int, Any]
        Explanation of each referred ECG in the primary analysis.
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
    for name, ecg_id in examples.items():
        index = int(scored.index[scored["ecg_id"] == ecg_id][0])
        found = explanations.get(ecg_id)
        if found is None:
            rule, title = [], f"{name}: PTB-XL ECG {ecg_id}; not referred"
        else:
            source = LAYER_NAMES[found.layer]
            rule = [(*mark, found.layer) for mark in rule_marks(found, thresholds[source])]
            top = int(found.units.scores.argmax())
            title = (f"{name}: PTB-XL ECG {ecg_id}; referred, explained by {source}, "
                     f"{LEADS[int(found.units.leads[top])]} at {found.units.starts[top]:.2f} s")
        marks = {"Rule (referred ECGs only)": rule,
                 "U_B alone": [(*m, 1) for m in red_marks(inputs["U_B"][index], thresholds["U_B"])],
                 "attention_jepa alone": [(*m, 2) for m in red_marks(inputs["attention_jepa"][index],
                                                                     thresholds["attention_jepa"])]}
        figure = plot_layer_marks(read_ptb_float64(scored.at[index, "filename_hr"]), FS, marks, title)
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


def case_name(readout: str, quantile: float) -> str:
    """Return the name of one analysis, such as ``logistic_concat_q0.975``."""
    return f"{readout}_q{quantile}"


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
    groups = groups_of(inputs["rows"])
    integrity: dict[str, Any] = {"aligned": inputs["aligned"], "pvc_exclusions": inputs["exclusions"],
                                 "pvc_included": len(inputs["windows"])}
    integrity["U_B_vs_042"] = check_042(inputs)
    integrity["maps_vs_043"] = check_043_maps(inputs)
    integrity["detection_vs_043"] = check_043_detection(inputs)
    integrity["E_vs_045"], receipt045 = check_045(inputs)
    thresholds = single_thresholds(inputs, groups)
    integrity["map_thresholds_equal_reported"] = True
    integrity["vs_047"], receipt047 = check_047(inputs, groups, thresholds)
    z, integrity["pvc_head_vs_044"], receipt044 = pvc_scores(inputs)
    receipts = {**inputs["receipts"], **receipt045, **receipt047, **receipt044}
    run_identity = {"inputs": receipts, "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
                    "git_head": git_head(ROOT), "openblas_architectures": blas_architectures()}
    LOG.info("integrity passed at %.1f s: %s", time.perf_counter() - started, integrity)
    if integrity_only:
        logging.shutdown()
        shutil.rmtree(partial)
        return

    switches = {q: switch_threshold(np.array([z[i] for i in groups["normal"]]), q) for q in QUANTILES}
    logits = readout_logits(inputs)
    cases = [("logistic_concat", q) for q in QUANTILES] + [("E", PRIMARY_QUANTILE)]
    found = {case_name(readout, q): analyse(inputs, groups, logits[readout], z, switches[q], thresholds, SEED)
             for readout, q in cases}
    for name, value in found.items():
        LOG.info("%s: referred %d, sent to U_B %s, readings %s", name, value["result"]["referred"],
                 value["result"]["sent_to_U_B"], value["result"]["readings"])
    primary = case_name("logistic_concat", PRIMARY_QUANTILE)

    examples = None
    if found[primary]["result"]["readings"]["vs_U_B"]["improves"]:
        examples = draw_examples(inputs, found[primary]["explanations"], thresholds, partial / "figures")
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    arrays: dict[str, np.ndarray] = {"ecg_ids": np.array(ids, dtype=np.int64),
                                     "z_pvc": np.array([z[i] for i in ids])}
    for name, value in found.items():
        for key, array in explanation_arrays(ids, value["explanations"]).items():
            arrays[f"{name}_{key}"] = array
    write_npz_atomic(partial / "explanations.npz", **arrays)
    outputs = {"explanations.npz": sha256_file(partial / "explanations.npz")}
    if examples is not None:
        outputs.update({f"figures/{path.name}": sha256_file(path)
                        for path in sorted((partial / "figures").glob("*.png"))})
    write_json_atomic(partial / "result.json", {
        "status": "completed", "identity": run_identity, "integrity": integrity,
        "counts": {name: len(groups[name]) for name in
                   ("normal", "positive", "pvc", "benign", "anterior", "inferior")},
        "map_thresholds": thresholds, "switch_thresholds": {str(q): value for q, value in switches.items()},
        "primary": primary, "cases": {name: value["result"] for name, value in found.items()},
        "primary_reading": found[primary]["result"]["readings"]["vs_U_B"],
        "examples": examples, "outputs_sha256": outputs, "seed": SEED, "draws": DRAWS,
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
