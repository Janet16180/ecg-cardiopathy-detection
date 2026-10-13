"""Experiment 049: a label-free focal-versus-diffuse switch for the explanation, from saved scores.

A referred ECG is explained by 042's ``U_B`` top unit when its worst beat stands out from its typical beat
(focal ratio above a quantile of the normals'), otherwise by 043's ``attention_jepa`` top token. The
comparators are each map alone and Experiment 048's PVC-head switch. Loading, receipt and reproduction checks
are imported from the 045, 047 and 048 runners, as the protocol documents.
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
    rule_reading,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.focal_switch import beat_layout_ok, combined_referral, focal_ratio
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.pvc_switch import pvc_z, switch_explain, switch_threshold
from scripts.experiments.run_ann_heads043 import ptb_inputs, sph_inputs
from scripts.experiments.run_explanation_rule047 import check_045, readout_logits, single_thresholds
from scripts.experiments.run_lead_wave_maps042 import blas_architectures
from scripts.experiments.run_pvc_switch048 import QUANTILES as QUANTILES048
from scripts.experiments.run_pvc_switch048 import analyse as analyse048
from scripts.experiments.run_pvc_switch048 import (
    case_name,
    check_047,
    draw_examples,
    explanation_arrays,
    pvc_scores,
)
from scripts.experiments.run_two_layer_map045 import (
    check_042,
    check_043_detection,
    check_043_maps,
    groups_of,
    load_inputs,
    plain,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment049_focal_switch_v1"
PRIOR044 = ROOT / "outputs/experiment044_pipeline_v3_v1"
PRIOR048 = ROOT / "outputs/experiment048_pvc_switch_v1"
FIGURES_COPY = ROOT / "docs/figures/experiment-049"
PROTOCOL = "docs/experiment-049-focal-switch.md"
SOURCES = (
    PROTOCOL,
    "scripts/experiments/run_focal_switch049.py",
    "ecg_experiment/focal_switch.py",
    "ecg_experiment/pvc_switch.py",
    "ecg_experiment/explanation_rule.py",
    "ecg_experiment/two_layer_map.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/finding_screen.py",
    "ecg_experiment/hybrid_score.py",
    "ecg_experiment/pipeline_v2.py",
    "ecg_experiment/intervals.py",
    "scripts/experiments/run_pvc_switch048.py",
    "scripts/experiments/run_explanation_rule047.py",
    "scripts/experiments/run_two_layer_map045.py",
    "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_lead_wave_maps042.py",
    "scripts/experiments/run_rhythm_findings032.py",
    "pyproject.toml",
    "uv.lock",
)
SEED = 49049
SEED048 = 48048
DRAWS = 2000
PRIMARY_QUANTILE = 0.975
QUANTILES = (0.95, 0.975, 0.99)
PVC_QUANTILE = 0.975
Z_TOLERANCE = 1e-10
THREADS = 4
SWITCH_ARMS = ("focal_switch", "focal_and_pvc", "focal_or_pvc", "pvc_switch")
FIXED_ARMS = ("U_B", "attention_jepa")
CANDIDATES = ("focal_switch", "focal_and_pvc", "focal_or_pvc")
CASES = (("logistic_concat", 0.975), ("logistic_concat", 0.95), ("logistic_concat", 0.99), ("E", 0.975),
         ("combined_50", 0.975))
PRIMARY = "logistic_concat_q0.975"
FIGURE_CASE = "combined_50_q0.975"
COST_GROUPS = ("anterior", "inferior", "pvc_window", "positive", "normal", "benign")
LOG = logging.getLogger("experiment049")


def wpw_scores(inputs: dict[str, Any], z_pvc: dict[int, float]) -> tuple[dict[int, float], dict[str, Any]]:
    """
    Score the xECG WPW head of pipeline v3 on the development ECGs, checked against 044 on SPH.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    z_pvc : dict[int, float]
        Development z_PVC by ECG ID (048's ``pvc_scores``).

    Returns
    -------
    tuple[dict[int, float], dict[str, Any]]
        z_WPW by development ECG ID, and the checks.

    Raises
    ------
    ValueError
        If a row order or the SPH reproduction of z_WPW or F differs.
    """
    ptb, ptb_x, _ = ptb_inputs()
    sph, sph_x, _ = sph_inputs()
    with (np.load(PRIOR044 / "pipeline_v3_heads.npz") as saved,
          np.load(PRIOR044 / "predictions.npz") as predictions):
        parameters = {name: saved[name] for name in saved.files}
        sph_pvc, sph_wpw = pvc_z(parameters, sph_x["xecg"]), pvc_z(parameters, sph_x["xecg"], "wpw")
        checks: dict[str, Any] = {
            "sph_ids": bool(np.array_equal(predictions["sph_ecg_ids"], sph["ecg_id"].to_numpy(dtype=str))),
            "development_rows": bool(np.array_equal(ptb["development"].index.to_numpy(dtype=str),
                                                    inputs["record_ids"])),
            "sph_z_wpw_max_difference": float(np.abs(sph_wpw - predictions["sph_v2_z_wpw"]).max()),
            "sph_F_max_difference": float(np.abs(np.maximum(sph_pvc, sph_wpw)
                                                 - predictions["sph_v2_z_combined"]).max())}
    checks["reproduces"] = (checks["sph_z_wpw_max_difference"] <= Z_TOLERANCE
                            and checks["sph_F_max_difference"] <= Z_TOLERANCE)
    ids = ptb["development"]["ecg_id"].astype(int).tolist()
    development_pvc = pvc_z(parameters, ptb_x["xecg"]["development"])
    checks["development_z_pvc_equals_048"] = bool(np.array_equal(development_pvc, [z_pvc[i] for i in ids]))
    if not (checks["sph_ids"] and checks["development_rows"] and checks["reproduces"]
            and checks["development_z_pvc_equals_048"]):
        raise ValueError(f"044's WPW head does not reproduce: {checks}")
    development = pvc_z(parameters, ptb_x["xecg"]["development"], "wpw")
    return dict(zip(ids, development.tolist(), strict=True)), checks


def check_048(inputs: dict[str, Any], groups: dict[str, Any], z_pvc: dict[int, float],
              thresholds: dict[str, float]) -> tuple[dict[str, bool], dict[str, str]]:
    """
    Require 048's saved explanations to match their receipt and its analysis to reproduce exactly.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    z_pvc : dict[int, float]
        Development z_PVC by ECG ID.
    thresholds : dict[str, float]
        Each map's own red threshold.

    Returns
    -------
    tuple[dict[str, bool], dict[str, str]]
        The checks, and the receipt of 048's ``explanations.npz``.

    Raises
    ------
    ValueError
        If any check fails.
    """
    prior = json.loads((PRIOR048 / "result.json").read_text())
    path = PRIOR048 / "explanations.npz"
    digest = sha256_file(path)
    checks = {"receipt": digest == prior["outputs_sha256"]["explanations.npz"]}
    with np.load(path) as saved:
        ids = saved["ecg_ids"].tolist()
        checks["z_pvc"] = bool(np.array_equal(saved["z_pvc"], [z_pvc[i] for i in ids]))
    normal_z = np.array([z_pvc[i] for i in groups["normal"]])
    switches = {q: switch_threshold(normal_z, q) for q in QUANTILES048}
    logits = readout_logits(inputs)
    for name, found in prior["cases"].items():
        readout, quantile = name.rsplit("_q", 1)
        mine = analyse048(inputs, groups, logits[readout], z_pvc, switches[float(quantile)], thresholds,
                          SEED048)["result"]
        checks[name] = plain(mine) == found and case_name(readout, float(quantile)) == name
    if not all(checks.values()):
        raise ValueError(f"048 does not reproduce: {checks}")
    return checks, {to_stored(path): digest}


def check_layout(inputs: dict[str, Any]) -> dict[str, int]:
    """Require every ``U_B`` map to follow the beat-lead-wave unit order, and count the beats."""
    bad = [i for i, m in enumerate(inputs["U_B"]) if not beat_layout_ok(m)]
    if bad:
        raise ValueError(f"{len(bad)} U_B maps do not follow the unit layout")
    beats = np.array([len(m.scores) // 48 for m in inputs["U_B"]])
    return {"ecgs": len(beats), "min_beats": int(beats.min()), "median_beats": float(np.median(beats)),
            "max_beats": int(beats.max())}


def named_groups(inputs: dict[str, Any], groups: dict[str, Any]) -> dict[str, list[int]]:
    """Return the reporting groups, including the PVC ECGs with a premature-beat window."""
    return {"normal": groups["normal"], "positive": groups["positive"], "pvc": groups["pvc"],
            "pvc_window": list(inputs["windows"]), "benign": groups["benign"], "anterior": groups["anterior"],
            "inferior": groups["inferior"]}


def referrals(inputs: dict[str, Any], groups: dict[str, Any], z_pvc: dict[int, float], z_wpw: dict[int, float]
              ) -> dict[str, dict[str, Any]]:
    """
    Return the referred ECG IDs and thresholds of each referral variant.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    z_pvc, z_wpw : dict[int, float]
        Development z-scores by ECG ID.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per variant: ``ids`` (scored order) and ``thresholds``.
    """
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    logits = readout_logits(inputs)
    found = {}
    for readout, values in logits.items():
        cutoff = referral_threshold(np.array([values[i] for i in groups["normal"]]))
        found[readout] = {"ids": [i for i in ids if values[i] > cutoff], "thresholds": {"binary": cutoff}}
    normal = set(groups["normal"])
    binary = np.array([logits["logistic_concat"][i] for i in ids])
    finding = np.array([max(z_pvc[i], z_wpw[i]) for i in ids])
    referred, thresholds = combined_referral(binary, finding, np.array([i in normal for i in ids]))
    found["combined_50"] = {"ids": [i for i, r in zip(ids, referred, strict=True) if r],
                            "thresholds": {"binary": float(thresholds[0]), "F": float(thresholds[1])}}
    return found


def focal_distribution(focal: dict[int, float], named: dict[str, list[int]], switch: dict[float, float]
                       ) -> dict[str, dict[str, float]]:
    """Return per group the focal ratio's median and 97.5th percentile, and the share above each threshold."""
    found = {}
    for name, members in named.items():
        values = np.array([focal[i] for i in members])
        found[name] = {"n": len(values), "median": float(np.median(values)),
                       "q975": float(np.quantile(values, 0.975)),
                       **{f"on_q{q}": float(np.mean(values > t)) for q, t in switch.items()}}
    return found


def analyse(inputs: dict[str, Any], groups: dict[str, Any], referred: list[int],
            switches: dict[str, set[int]], thresholds: dict[str, float], seed: int) -> dict[str, Any]:
    """
    Score every switch arm and both single maps on one set of referred ECGs.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    referred : list[int]
        Referred ECG IDs in scored order.
    switches : dict[str, set[int]]
        Per switch arm, the ECG IDs with the switch on.
    thresholds : dict[str, float]
        Each map's own red threshold.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        Public ``result`` and per-arm ``explanations`` of the referred ECGs.
    """
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    patients, windows = groups["patients"], inputs["windows"]
    maps = {name: dict(zip(ids, inputs[name], strict=True)) for name in FIXED_ARMS}
    chosen = set(referred)
    first_t, second_t = thresholds["U_B"], thresholds["attention_jepa"]
    explanations = {arm: {i: switch_explain(maps["U_B"][i], maps["attention_jepa"][i], i in on, first_t,
                                            second_t) for i in referred}
                    for arm, on in switches.items()}
    units = {**{arm: {i: e.units for i, e in found.items()} for arm, found in explanations.items()},
             **{name: {i: maps[name][i] for i in referred} for name in FIXED_ARMS}}
    named = named_groups(inputs, groups)
    positive = groups["positive"]
    scored = {name: explanation_metrics(units[name], patients, windows, groups["anterior"],
                                        groups["inferior"], DRAWS, seed) for name in units}
    paired, readings = {}, {}
    for arm in SWITCH_ARMS:
        for baseline in ("U_B", "attention_jepa", "pvc_switch"):
            if arm == baseline:
                continue
            difference = paired_difference(scored[arm], scored[baseline], patients, DRAWS, seed)
            paired[f"{arm}_minus_{baseline}"] = difference
            source = difference if baseline == "pvc_switch" else scored[arm]["metrics"]
            anterior = source["lead_contrast"]["anterior"]["ci_low"]
            readings[f"{arm}_vs_{baseline}"] = rule_reading(difference["hit_minus_chance"]["ci_low"],
                                                            anterior)
    layers = {arm: {i: e.layer for i, e in found.items()} for arm, found in explanations.items()}
    result = {
        "referred": len(referred),
        "referred_counts": {name: int(sum(i in chosen for i in members)) for name, members in named.items()},
        "referral_rate": group_rates(chosen, named),
        "positive_referral": bootstrap_mean(np.array([patients[i] for i in positive]),
                                            np.array([float(i in chosen) for i in positive]), DRAWS, seed),
        "sent_to_U_B": {arm: {name: int(sum(found[i] == 1 for i in named[name] if i in chosen))
                              for name in COST_GROUPS} for arm, found in layers.items()},
        "explained_by_layer": {arm: layer_shares(found, {"referred": referred, **named})
                               for arm, found in layers.items()},
        "maps": {name: value["metrics"] for name, value in scored.items()}, "paired": paired,
        "readings": readings,
    }
    return {"result": result, "explanations": explanations}


def run_cases(inputs: dict[str, Any], groups: dict[str, Any], referral: dict[str, dict[str, Any]],
              focal: dict[int, float], focal_cut: dict[float, float], pvc_on: set[int],
              thresholds: dict[str, float]) -> dict[str, dict[str, Any]]:
    """
    Run ``analyse`` for every prespecified referral and focal quantile.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    referral : dict[str, dict[str, Any]]
        Output of ``referrals``.
    focal : dict[int, float]
        Focal ratio by ECG ID.
    focal_cut : dict[float, float]
        Focal threshold per quantile.
    pvc_on : set[int]
        ECG IDs with 048's PVC switch on.
    thresholds : dict[str, float]
        Each map's own red threshold.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``analyse`` output per case name.
    """
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    found = {}
    for readout, q in CASES:
        focal_on = {i for i in ids if focal[i] > focal_cut[q]}
        switches = {"focal_switch": focal_on, "focal_and_pvc": focal_on & pvc_on,
                    "focal_or_pvc": focal_on | pvc_on, "pvc_switch": pvc_on}
        name = case_name(readout, q)
        found[name] = analyse(inputs, groups, referral[readout]["ids"], switches, thresholds, SEED)
        found[name]["result"].update({"referral": readout, "focal_quantile": q,
                                      "focal_threshold": focal_cut[q],
                                      "referral_thresholds": referral[readout]["thresholds"]})
        LOG.info("%s: referred %d, sent to U_B %s, readings %s", name, found[name]["result"]["referred"],
                 found[name]["result"]["sent_to_U_B"], found[name]["result"]["readings"])
    return found


def output_arrays(ids: list[int], found: dict[str, dict[str, Any]], focal: dict[int, float],
                  z_pvc: dict[int, float], z_wpw: dict[int, float]) -> dict[str, np.ndarray]:
    """Return the per-ECG scores, and each case's referrals, layers and ``focal_switch`` explanation units."""
    arrays: dict[str, np.ndarray] = {"ecg_ids": np.array(ids, dtype=np.int64),
                                     "focal_ratio": np.array([focal[i] for i in ids]),
                                     "z_pvc": np.array([z_pvc[i] for i in ids]),
                                     "z_wpw": np.array([z_wpw[i] for i in ids])}
    for name, value in found.items():
        for arm, explained in value["explanations"].items():
            for key, array in explanation_arrays(ids, explained).items():
                if arm == "focal_switch" or key in ("referred", "layer"):
                    arrays[f"{name}_{arm}_{key}"] = array
    return arrays


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
    z_pvc, integrity["pvc_head_vs_044"], receipt044 = pvc_scores(inputs)
    z_wpw, integrity["wpw_head_vs_044"] = wpw_scores(inputs, z_pvc)
    integrity["vs_048"], receipt048 = check_048(inputs, groups, z_pvc, thresholds)
    integrity["U_B_layout"] = check_layout(inputs)
    receipts = {**inputs["receipts"], **receipt045, **receipt047, **receipt044, **receipt048}
    run_identity = {"inputs": receipts, "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
                    "git_head": git_head(ROOT), "openblas_architectures": blas_architectures()}
    LOG.info("integrity passed at %.1f s: %s", time.perf_counter() - started, integrity)
    if integrity_only:
        logging.shutdown()
        shutil.rmtree(partial)
        return

    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    focal = dict(zip(ids, [focal_ratio(m) for m in inputs["U_B"]], strict=True))
    named = named_groups(inputs, groups)
    focal_cut = {q: switch_threshold(np.array([focal[i] for i in groups["normal"]]), q) for q in QUANTILES}
    pvc_cut = switch_threshold(np.array([z_pvc[i] for i in groups["normal"]]), PVC_QUANTILE)
    distribution = focal_distribution(focal, named, focal_cut)
    LOG.info("focal thresholds %s, PVC threshold %.4f, distribution %s", focal_cut, pvc_cut, distribution)
    referral = referrals(inputs, groups, z_pvc, z_wpw)
    found = run_cases(inputs, groups, referral, focal, focal_cut, {i for i in ids if z_pvc[i] > pvc_cut},
                      thresholds)

    improves = found[PRIMARY]["result"]["readings"]["focal_switch_vs_pvc_switch"]["improves"]
    examples = None
    if improves:
        examples = draw_examples(inputs, found[FIGURE_CASE]["explanations"]["focal_switch"], thresholds,
                                 partial / "figures")
    arrays = output_arrays(ids, found, focal, z_pvc, z_wpw)
    write_npz_atomic(partial / "explanations.npz", **arrays)
    outputs = {"explanations.npz": sha256_file(partial / "explanations.npz")}
    if examples is not None:
        outputs.update({f"figures/{path.name}": sha256_file(path)
                        for path in sorted((partial / "figures").glob("*.png"))})
    write_json_atomic(partial / "result.json", {
        "status": "completed", "identity": run_identity, "integrity": integrity,
        "counts": {name: len(members) for name, members in named.items()},
        "map_thresholds": thresholds, "focal_thresholds": {str(q): t for q, t in focal_cut.items()},
        "pvc_threshold": pvc_cut, "focal_distribution": distribution,
        "referral": {name: {"referred": len(value["ids"]), "thresholds": value["thresholds"]}
                     for name, value in referral.items()},
        "primary": PRIMARY, "cases": {name: value["result"] for name, value in found.items()},
        "primary_reading": found[PRIMARY]["result"]["readings"]["focal_switch_vs_pvc_switch"],
        "figure_case": FIGURE_CASE, "examples": examples, "outputs_sha256": outputs, "seed": SEED,
        "draws": DRAWS, "total_seconds": time.perf_counter() - started, "calibration_test_evaluated": False,
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
