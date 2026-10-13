"""Experiment 051: a beat-sum wave map from Experiment 042's saved beat-aligned units.

Each beat is scored by the sum of its 48 lead-and-wave scores (``beat_sum``), or by that sum relative to the
ECG's median beat (``beat_sum_relative``), and compared with ``U_B`` (042's maximum over pieces) by 042's map
rule. A secondary analysis puts ``beat_sum`` in place of ``U_B`` as the rhythm layer of 048's explanation with
049's ``combined_50`` referral. Loading and integrity checks are imported from the 042-049 runners, as the
protocol documents.
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

from ecg_experiment.beat_sum_map import (
    beat_r_times,
    beat_sum_map,
    plot_beat_marks,
    r_time_hit,
    top_pieces,
    unit_beat,
)
from ecg_experiment.explanation_rule import explanation_metrics, paired_difference, rule_reading
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.focal_switch import focal_ratio
from ecg_experiment.fragment_localization import bootstrap_mean
from ecg_experiment.lead_wave_maps import WAVES, UnitMap, ecg_score, red_marks
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.pvc_switch import switch_threshold
from ecg_experiment.waveforms import LEADS
from scripts.experiments.run_ann_heads043 import map_contrasts, map_metrics, public_map_metrics
from scripts.experiments.run_explanation_rule047 import check_045, single_thresholds
from scripts.experiments.run_focal_switch049 import (
    PVC_QUANTILE,
    QUANTILES,
    check_048,
    check_layout,
    referrals,
    run_cases,
    wpw_scores,
)
from scripts.experiments.run_lead_wave_maps042 import blas_architectures
from scripts.experiments.run_pvc_switch048 import check_047, pvc_scores
from scripts.experiments.run_two_layer_map045 import (
    check_042,
    check_043_detection,
    check_043_maps,
    groups_of,
    load_inputs,
    plain,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment051_beat_sum_v1"
PRIOR042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PRIOR049 = ROOT / "outputs/experiment049_focal_switch_v1"
FIGURES_COPY = ROOT / "docs/figures/experiment-051"
PROTOCOL = "docs/experiment-051-beat-sum-map.md"
SOURCES = (
    PROTOCOL,
    "scripts/experiments/run_beat_sum051.py",
    "ecg_experiment/beat_sum_map.py",
    "ecg_experiment/focal_switch.py",
    "ecg_experiment/pvc_switch.py",
    "ecg_experiment/explanation_rule.py",
    "ecg_experiment/two_layer_map.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/ann_heads.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/finding_screen.py",
    "ecg_experiment/intervals.py",
    "scripts/experiments/run_focal_switch049.py",
    "scripts/experiments/run_pvc_switch048.py",
    "scripts/experiments/run_explanation_rule047.py",
    "scripts/experiments/run_two_layer_map045.py",
    "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_lead_wave_maps042.py",
    "pyproject.toml",
    "uv.lock",
)
SEED = 51051
DRAWS = 2000
FS = 500
THREADS = 4
MAPS = ("beat_sum", "beat_sum_relative")
EXAMPLE_LBBB = 287
REFERRAL = "combined_50"
LOG = logging.getLogger("experiment051")


def check_049(inputs: dict[str, Any], groups: dict[str, Any], z_pvc: dict[int, float],
              z_wpw: dict[int, float], thresholds: dict[str, float]
              ) -> tuple[dict[str, bool], dict[str, str], dict[str, Any]]:
    """
    Require 049's saved explanations to match their receipt and its five cases to reproduce exactly.

    Parameters
    ----------
    inputs : dict[str, Any]
        Output of 045's ``load_inputs``.
    groups : dict[str, Any]
        Output of 045's ``groups_of``.
    z_pvc, z_wpw : dict[int, float]
        Development z-scores by ECG ID.
    thresholds : dict[str, float]
        Each map's own red threshold.

    Returns
    -------
    tuple[dict[str, bool], dict[str, str], dict[str, Any]]
        The checks, the receipt of 049's ``explanations.npz``, and 049's referrals and PVC switch threshold.

    Raises
    ------
    ValueError
        If any check fails.
    """
    prior = json.loads((PRIOR049 / "result.json").read_text())
    path = PRIOR049 / "explanations.npz"
    digest = sha256_file(path)
    checks = {"receipt": digest == prior["outputs_sha256"]["explanations.npz"]}
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    focal = dict(zip(ids, [focal_ratio(m) for m in inputs["U_B"]], strict=True))
    focal_cut = {q: switch_threshold(np.array([focal[i] for i in groups["normal"]]), q) for q in QUANTILES}
    pvc_cut = switch_threshold(np.array([z_pvc[i] for i in groups["normal"]]), PVC_QUANTILE)
    referral = referrals(inputs, groups, z_pvc, z_wpw)
    found = run_cases(inputs, groups, referral, focal, focal_cut, {i for i in ids if z_pvc[i] > pvc_cut},
                      thresholds)
    for name, value in found.items():
        checks[name] = plain(value["result"]) == prior["cases"][name]
    checks["pvc_threshold"] = pvc_cut == prior["pvc_threshold"]
    if not all(checks.values()):
        raise ValueError(f"049 does not reproduce: {checks}")
    return checks, {to_stored(path): digest}, {"referral": referral, "pvc_cut": pvc_cut}


def strict_hits(maps: dict[str, list[UnitMap]], ids: list[int], windows: dict[int, list[tuple[float, float]]],
                patients: dict[int, Any], seed: int) -> dict[str, Any]:
    """
    Score the strict premature-beat hit: the R time of the top unit's beat inside a window.

    Parameters
    ----------
    maps : dict[str, list[UnitMap]]
        ``U_B`` and the beat maps in scored order.
    ids : list[int]
        Scored ECG IDs.
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows by ECG ID.
    patients : dict[int, Any]
        Patient ID by ECG ID.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        Per map hit, chance and hit - chance; per beat map the paired difference from ``U_B``; and the per-ECG
        top beats.
    """
    index = {i: k for k, i in enumerate(ids)}
    pvc = list(windows)
    found: dict[str, Any] = {"maps": {}, "minus_U_B": {}, "top_beat": {}}
    excess = {}
    for name, values in maps.items():
        rows = []
        for i in pvc:
            unit_map = maps["U_B"][index[i]]
            top = int(values[index[i]].scores.argmax())
            beat = unit_beat(top) if name == "U_B" else top
            found["top_beat"].setdefault(name, {})[i] = beat
            rows.append(r_time_hit(beat_r_times(unit_map), beat, windows[i]))
        hit, chance = np.array(rows).T
        excess[name] = hit - chance
        found["maps"][name] = {"hit_rate": float(hit.mean()), "chance_rate": float(chance.mean()),
                               "hit_minus_chance": bootstrap_mean(np.array([patients[i] for i in pvc]),
                                                                  hit - chance, DRAWS, seed)}
    for name in MAPS:
        found["minus_U_B"][name] = bootstrap_mean(np.array([patients[i] for i in pvc]),
                                                  excess[name] - excess["U_B"], DRAWS, seed)
    return found


def disagreement(maps: dict[str, list[UnitMap]], ids: list[int], windows: dict[int, list[tuple[float, float]]]
                 ) -> dict[str, Any]:
    """Count ECGs whose ``U_B`` top unit and ``beat_sum`` top beat differ, and who is right on PVC ECGs."""
    u_b = [unit_beat(int(m.scores.argmax())) for m in maps["U_B"]]
    beat = [int(m.scores.argmax()) for m in maps["beat_sum"]]
    index = {i: k for k, i in enumerate(ids)}
    differ = [i for i, a, b in zip(ids, u_b, beat, strict=True) if a != b]
    pvc_differ = [i for i in windows if i in set(differ)]
    on_beat_sum = on_u_b = 0
    for i in pvc_differ:
        r_times = beat_r_times(maps["U_B"][index[i]])
        on_beat_sum += int(r_time_hit(r_times, beat[index[i]], windows[i])[0])
        on_u_b += int(r_time_hit(r_times, u_b[index[i]], windows[i])[0])
    return {"ecgs": len(ids), "differ": len(differ), "pvc_ecgs": len(windows), "pvc_differ": len(pvc_differ),
            "pvc_differ_premature_is_beat_sum_top": on_beat_sum, "pvc_differ_premature_is_U_B_top": on_u_b}


def example_lbbb(maps: dict[str, list[UnitMap]], ids: list[int]) -> dict[str, Any]:
    """Return ECG 287's beat sums and both maps' top units."""
    k = ids.index(EXAMPLE_LBBB)
    u_b, beats = maps["U_B"][k], maps["beat_sum"][k]
    top = int(u_b.scores.argmax())
    r_times = beat_r_times(u_b)
    return {"ecg_id": EXAMPLE_LBBB, "r_times": r_times.round(3).tolist(), "beat_sums": beats.scores.tolist(),
            "beat_sum_top_r_time": float(r_times[int(beats.scores.argmax())]),
            "beat_sum_relative_top": float(maps["beat_sum_relative"][k].scores.max()),
            "U_B_top": {"score": float(u_b.scores[top]), "lead": LEADS[int(u_b.leads[top])],
                        "wave": list(WAVES)[top % 4], "beat_r_time": float(r_times[unit_beat(top)])}}


def positive_red(metrics: dict[str, Any], maps: dict[str, list[UnitMap]], groups: dict[str, Any], seed: int
                 ) -> dict[str, Any]:
    """Return each beat map's positive any-red share minus ``U_B``'s, per positive ECG."""
    ids = groups["patients"]
    order = list(ids)
    found = {}
    red = {name: {i: ecg_score(m) > metrics["thresholds"][name] for i, m in zip(order, values, strict=True)}
           for name, values in maps.items()}
    patients = np.array([ids[i] for i in groups["positive"]])
    for name in MAPS:
        found[name] = bootstrap_mean(patients, np.array([float(red[name][i]) - float(red["U_B"][i])
                                                         for i in groups["positive"]]), DRAWS, seed)
    return found


def explanation_with(inputs: dict[str, Any], groups: dict[str, Any], rhythm: list[UnitMap],
                     referred: list[int], pvc_on: set[int], seed: int) -> dict[str, Any]:
    """Score 048's PVC-switch explanation on the referred ECGs with a given rhythm layer."""
    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    rhythm_by, attention_by = dict(zip(ids, rhythm, strict=True)), dict(zip(ids, inputs["attention_jepa"],
                                                                            strict=True))
    units = {i: rhythm_by[i] if i in pvc_on else attention_by[i] for i in referred}
    return explanation_metrics(units, groups["patients"], inputs["windows"], groups["anterior"],
                               groups["inferior"], DRAWS, seed)


def draw_examples(inputs: dict[str, Any], maps: dict[str, list[UnitMap]], thresholds: dict[str, float],
                  folder: Path) -> dict[str, int]:
    """Save the 041 example figures: ``beat_sum``'s top beat and pieces, and ``U_B``'s red units."""
    folder.mkdir(parents=True, exist_ok=True)
    scored = inputs["rows"]["scored"].reset_index()
    examples = json.loads((PRIOR042 / "result.json").read_text())["examples"]
    for name, ecg_id in examples.items():
        index = int(scored.index[scored["ecg_id"] == ecg_id][0])
        u_b, beats = maps["U_B"][index], maps["beat_sum"][index]
        beat = int(beats.scores.argmax())
        red = bool(beats.scores[beat] > thresholds["beat_sum"])
        top = int(u_b.scores.argmax())
        u_b_marks = red_marks(u_b, thresholds["U_B"])
        top_mark = (int(u_b.leads[top]), float(u_b.starts[top]), float(u_b.ends[top]))
        columns = {
            f"beat_sum: top beat {'red' if red else 'not red'} ({beats.scores[beat]:,.0f})": {
                "beat": (float(beats.starts[beat]), float(beats.ends[beat]), "red" if red else "grey"),
                "pieces": top_pieces(u_b, beat)},
            f"U_B: red units (top {LEADS[top_mark[0]]} {u_b.scores[top]:,.0f})": {
                "beat": None, "pieces": u_b_marks if top_mark in u_b_marks else [*u_b_marks, top_mark]},
        }
        r_time = float(beat_r_times(u_b)[beat])
        title = f"{name}: PTB-XL ECG {ecg_id}; beat_sum top beat at R = {r_time:.2f} s"
        figure = plot_beat_marks(read_ptb_float64(scored.at[index, "filename_hr"]), FS, columns, title)
        figure.savefig(folder / f"{name}_{ecg_id}.png", dpi=110)
    return examples


def integrity(inputs: dict[str, Any], groups: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict]:
    """Run 049's integrity checks and the 049 reproduction; return checks, receipts and the shared state."""
    checks: dict[str, Any] = {"aligned": inputs["aligned"], "pvc_exclusions": inputs["exclusions"],
                              "pvc_included": len(inputs["windows"])}
    checks["U_B_vs_042"] = check_042(inputs)
    checks["maps_vs_043"] = check_043_maps(inputs)
    checks["detection_vs_043"] = check_043_detection(inputs)
    checks["E_vs_045"], receipt045 = check_045(inputs)
    thresholds = single_thresholds(inputs, groups)
    checks["vs_047"], receipt047 = check_047(inputs, groups, thresholds)
    z_pvc, checks["pvc_head_vs_044"], receipt044 = pvc_scores(inputs)
    z_wpw, checks["wpw_head_vs_044"] = wpw_scores(inputs, z_pvc)
    checks["vs_048"], receipt048 = check_048(inputs, groups, z_pvc, thresholds)
    checks["U_B_layout"] = check_layout(inputs)
    checks["vs_049"], receipt049, state = check_049(inputs, groups, z_pvc, z_wpw, thresholds)
    receipts = {**inputs["receipts"], **receipt045, **receipt047, **receipt044, **receipt048, **receipt049}
    return checks, receipts, {**state, "z_pvc": z_pvc, "thresholds": thresholds}


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
    checks, receipts, state = integrity(inputs, groups)
    run_identity = {"inputs": receipts, "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
                    "git_head": git_head(ROOT), "openblas_architectures": blas_architectures()}
    LOG.info("integrity passed at %.1f s: %s", time.perf_counter() - started, checks)
    if integrity_only:
        logging.shutdown()
        shutil.rmtree(partial)
        return

    ids = inputs["rows"]["scored"]["ecg_id"].tolist()
    maps = {"U_B": inputs["U_B"], "beat_sum": [beat_sum_map(m) for m in inputs["U_B"]],
            "beat_sum_relative": [beat_sum_map(m, relative=True) for m in inputs["U_B"]]}
    metrics = map_metrics(inputs["rows"], maps, inputs["windows"], DRAWS, SEED)
    contrasts = {name: map_contrasts(metrics, name, "U_B", DRAWS, SEED) for name in MAPS}
    positive = positive_red(metrics, maps, groups, SEED)
    strict = strict_hits(maps, ids, inputs["windows"], groups["patients"], SEED)
    differ = disagreement(maps, ids, inputs["windows"])
    lbbb = example_lbbb(maps, ids)
    LOG.info("contrasts %s; strict %s; disagreement %s; 287 %s",
             {k: v["reading"] for k, v in contrasts.items()}, strict["minus_U_B"], differ, lbbb)

    referred = state["referral"][REFERRAL]["ids"]
    pvc_on = {i for i in ids if state["z_pvc"][i] > state["pvc_cut"]}
    explained = {name: explanation_with(inputs, groups, maps[name], referred, pvc_on, SEED)
                 for name in ("U_B", "beat_sum")}
    paired = paired_difference(explained["beat_sum"], explained["U_B"], groups["patients"], DRAWS, SEED)
    secondary = {"referral": REFERRAL, "referred": len(referred), "pvc_threshold": state["pvc_cut"],
                 "maps": {name: value["metrics"] for name, value in explained.items()},
                 "beat_sum_minus_U_B": paired,
                 "reading": rule_reading(paired["hit_minus_chance"]["ci_low"],
                                         explained["beat_sum"]["metrics"]["lead_contrast"]["anterior"]["ci_low"])}
    LOG.info("secondary %s", secondary["reading"])

    examples = None
    if contrasts["beat_sum"]["reading"]["improves_on_U_B"]:
        examples = draw_examples(inputs, maps, metrics["thresholds"], partial / "figures")
    offsets = np.concatenate([[0], np.cumsum([len(m.scores) for m in maps["beat_sum"]])])
    write_npz_atomic(partial / "beat_maps.npz", ecg_ids=np.array(ids, dtype=np.int64), beat_offsets=offsets,
                     beat_sums=np.concatenate([m.scores for m in maps["beat_sum"]]),
                     beat_r_times=np.concatenate([beat_r_times(m) for m in maps["U_B"]]),
                     beat_sum_top=np.array([int(m.scores.argmax()) for m in maps["beat_sum"]]),
                     U_B_top_beat=np.array([unit_beat(int(m.scores.argmax())) for m in maps["U_B"]]))
    outputs = {"beat_maps.npz": sha256_file(partial / "beat_maps.npz")}
    if examples is not None:
        outputs.update({f"figures/{path.name}": sha256_file(path)
                        for path in sorted((partial / "figures").glob("*.png"))})
    strict_public = {key: strict[key] for key in ("maps", "minus_U_B")}
    write_json_atomic(partial / "result.json", {
        "status": "completed", "identity": run_identity, "integrity": checks,
        **plain(public_map_metrics(metrics)), "minus_U_B": contrasts, "positive_any_red_minus_U_B": positive,
        "readings": {name: value["reading"] for name, value in contrasts.items()},
        "strict_premature_hit": strict_public, "disagreement": differ, "example_287": lbbb,
        "secondary_explanation": secondary, "examples": examples, "outputs_sha256": outputs, "seed": SEED,
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
