"""Experiment 041: per-section anomaly maps of xECG tokens, unsupervised and label-guided."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.external_encoders import (
    XECG_CACHE,
    load_xecg_backbone,
    ptb_xecg_input,
    read_ptb_float64,
    xecg_cache,
)
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import (
    SECTIONS,
    fit_section_reference,
    gated,
    order_check,
    plot_sections,
    pointing,
    premature_windows,
    r_peaks,
    readout_contributions,
    red_threshold,
    section_kmeans,
    section_mahalanobis,
    section_overlap,
    xecg_tokens,
)
from ecg_experiment.fragment_localization import (
    bootstrap_mean as bootstrap,
)
from ecg_experiment.full_development import cohorts, fit_logistic, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.normal_manifold import fit_mahalanobis, mahalanobis_scores
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment041_fragment_localization_v1"
PRIOR026 = ROOT / "outputs/experiment026_normal_manifold_v1"
CACHE_IDS = {
    "jepa": ROOT / "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
    "xecg": XECG_CACHE / "ecg_ids.npy",
    "released_cpc": ROOT / "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
}
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
PTB_FILES = (PTB_RAW / "ptbxl_database.csv", PTB_RAW / "scp_statements.csv")
PROTOCOL = "docs/experiment-041-fragment-localization.md"
SOURCES = (
    "ecg_experiment/fragment_localization.py", "ecg_experiment/external_encoders.py",
    "ecg_experiment/xecg.py", "ecg_experiment/normal_manifold.py", "ecg_experiment/full_development.py",
    "ecg_experiment/intervals.py", "ecg_experiment/eda/ptbxl.py",
    "scripts/experiments/run_fragment_localization041.py", "pyproject.toml", "uv.lock", PROTOCOL,
)
MAPS = ("U", "U_kmeans", "U_uncentred", "G", "G_gated")
EXPECTED = {"fit": (5872, 5537), "pool": (15359, 9487), "evaluation": (1306, 843), "development": 1604,
            "pvc": (84, 81), "benign": 52}
BENIGN_CODES = {"NORM", "SR", "SBRAD", "SARRH"}
EXAMPLES = ("NORM", "PVC", "IMI", "AMI", "CLBBB", "LVH", "benign")
FS = 500
SEED = 41041
DRAWS = 2000
ORDER_RECORDS = 8
PROFILE_RECORDS = 128
CHUNK = 256
READER_THREADS = 4
PROBE_THREADS = 1
ANALYSIS_RESERVE_SECONDS = 600.0
CEILING_SECONDS = 1800.0
FEATURE_TOLERANCE = 1e-4
REPRODUCTION_TOLERANCE = 1e-8
LOGIT_TOLERANCE = 1e-3
DETECTION_MARGIN = 0.05
LOG = logging.getLogger("experiment041")


def identity() -> dict[str, Any]:
    """
    Hash every input and source; the xECG cache and the Experiment 026 scores must match their receipts.

    Returns
    -------
    dict[str, Any]
        Input and source hashes, keyed by repository-relative path.

    Raises
    ------
    ValueError
        If an input differs from its receipt.
    """
    _, _, cache_hashes = opened_cache()
    prior = json.loads((PRIOR026 / "result.json").read_text())
    scores = sha256_file(PRIOR026 / "development_scores.csv")
    if scores != prior["development_scores_sha256"]:
        raise ValueError("Experiment 026 development scores differ from their receipt")
    inputs = {to_stored(path): sha256_file(path) for path in (*CACHE_IDS.values(), *PTB_FILES)}
    inputs[to_stored(PRIOR026 / "development_scores.csv")] = scores
    inputs[to_stored(PRIOR026 / "result.json")] = sha256_file(PRIOR026 / "result.json")
    return {"inputs": inputs, "xecg_cache": cache_hashes,
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES}, "git_head": git_head(ROOT)}


def statement_table() -> pd.DataFrame:
    """
    SCP codes, abnormal diagnostic subclasses and diagnostic superclasses of every PTB-XL record.

    Returns
    -------
    pd.DataFrame
        Indexed by ECG ID, with ``codes``, ``subclasses`` and ``superclasses`` sets.
    """
    meta = load_metadata()
    statements = load_statements()
    diagnostic = statements[statements["diagnostic"] == 1]
    subclass = diagnostic["diagnostic_subclass"].to_dict()
    superclass = diagnostic["diagnostic_class"].to_dict()
    codes = meta["scp_codes"].apply(set)
    return pd.DataFrame({
        "codes": codes,
        "subclasses": codes.apply(lambda found: {subclass[c] for c in found if c in subclass} - {"NORM"}),
        "superclasses": codes.apply(lambda found: {superclass[c] for c in found if c in superclass}),
    }, index=meta.index.astype(int))


def is_benign(codes: set[str], superclasses: set[str]) -> bool:
    """NORM-only diagnosis, statements within NORM, SR, SBRAD and SARRH, and SBRAD or SARRH present."""
    return superclasses == {"NORM"} and codes <= BENIGN_CODES and bool(codes & {"SBRAD", "SARRH"})


def select_rows(smoke: bool) -> dict[str, pd.DataFrame]:
    """
    Fit, readout pool, binary evaluation, premature-beat, benign-variant and extraction rows.

    In smoke mode, every scored set is drawn from training ECGs so that no development score exists
    before the full run.

    Parameters
    ----------
    smoke : bool
        Use small training-only sets.

    Returns
    -------
    dict[str, pd.DataFrame]
        Frames keyed by set name; ``extract`` holds every row whose tokens are needed, fit set first.

    Raises
    ------
    ValueError
        If a full-run count differs from the protocol.
    """
    groups = cohorts(ptb_table())
    statements = statement_table()
    common = set.intersection(*(set(np.load(path).astype(np.int64).tolist()) for path in CACHE_IDS.values()))
    train, development = groups["train"], groups["development"]
    pool = train[train["standard"].notna() & train["ecg_id"].isin(common)]
    fit = pool[pool["standard"] == 0]
    pvc = statements.index[statements["codes"].map(lambda codes: "PVC" in codes)]
    benign = statements.index[[is_benign(codes, found) for codes, found
                               in zip(statements["codes"], statements["superclasses"], strict=True)]]

    def pvc_of(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["ecg_id"].isin(pvc)]

    def benign_of(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["ecg_id"].isin(benign)]
    if smoke:
        fit = fit.iloc[:300]
        rest = pool.drop(fit.index)
        evaluation = pd.concat([rest[rest["standard"] == 0].iloc[:100],
                                rest[rest["standard"] == 1].iloc[:100]])
        scored = train.drop(fit.index)
        pvc, benign = pvc_of(scored).iloc[:30], benign_of(scored).iloc[:20]
        scored = pd.concat([evaluation, pvc, benign])
        scored = scored[~scored.index.duplicated()]
    else:
        original = development[development["original"]]
        evaluation = original[original["standard"].notna() & original["ecg_id"].isin(common)]
        pvc, benign, scored = pvc_of(development), benign_of(development), development
        counts = {"fit": (len(fit), fit["patient_id"].nunique()),
                  "pool": (len(pool), int(pool["standard"].sum())),
                  "evaluation": (len(evaluation), int(evaluation["standard"].sum())),
                  "development": len(development), "pvc": (len(pvc), pvc["patient_id"].nunique()),
                  "benign": len(benign)}
        if counts != EXPECTED:
            raise ValueError(f"Row counts differ from the protocol: {counts}")
    if set(fit.index) & set(scored.index):
        raise ValueError("A fit ECG is also scored")
    return {"fit": fit, "pool": pool, "evaluation": evaluation, "pvc": pvc, "benign": benign,
            "scored": scored, "extract": pd.concat([fit, scored]), "statements": statements}


@cache
def opened_cache() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Open the xECG cache, checked against its receipt once per process."""
    return xecg_cache()


def cached(ecg_ids: np.ndarray) -> np.ndarray:
    """Return the cached pooled xECG features of the given ECGs, in order."""
    features, ids, _ = opened_cache()
    position = pd.Series(np.arange(len(ids)), index=ids)
    return np.asarray(features[position.loc[ecg_ids].to_numpy()], dtype=np.float64)


def reproduce026(rows: dict[str, pd.DataFrame]) -> tuple[dict[str, float], Any, np.ndarray]:
    """
    Refit Experiment 026's whole-ECG xECG Mahalanobis and ``probe_all`` and reproduce its saved scores.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Output of ``select_rows`` (full run).

    Returns
    -------
    tuple[dict[str, float], Any, np.ndarray]
        Largest differences, the refitted head and the whole-ECG Mahalanobis score of the evaluation rows.

    Raises
    ------
    ValueError
        If either score differs by more than ``REPRODUCTION_TOLERANCE``.
    """
    evaluation = rows["evaluation"]
    saved = pd.read_csv(PRIOR026 / "development_scores.csv").set_index("ecg_id").loc[evaluation["ecg_id"]]
    x = cached(evaluation["ecg_id"].to_numpy())
    with threadpool_limits(limits=PROBE_THREADS):
        whole = mahalanobis_scores(fit_mahalanobis(cached(rows["fit"]["ecg_id"].to_numpy())), x)
        head = fit_logistic(cached(rows["pool"]["ecg_id"].to_numpy()), rows["pool"]["standard"].to_numpy(int))
    probability = predict(head, x)
    differences = {
        "mahalanobis_relative": float(np.max(np.abs(whole - saved["xecg_mahalanobis"])
                                             / saved["xecg_mahalanobis"])),
        "probe_all_absolute": float(np.max(np.abs(probability - saved["xecg_probe_all"]))),
    }
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"Experiment 026 scores not reproduced: {differences}")
    return differences, head, whole


def read_inputs(stems: list[str]) -> np.ndarray:
    """Read the xECG inputs of PTB-XL records in parallel."""
    with ThreadPoolExecutor(READER_THREADS) as pool:
        return np.stack(list(pool.map(ptb_xecg_input, stems)))


def extract(model: torch.nn.Module, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """
    Tokens of every row, chunk by chunk, with read and model timings.

    Parameters
    ----------
    model : torch.nn.Module
        xECG backbone on the GPU.
    frame : pd.DataFrame
        Rows with ``filename_hr``.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        Float32 ``[rows, 40, 1024]`` tokens and the seconds spent reading and in the model.
    """
    tokens = np.empty((len(frame), SECTIONS, 1024), dtype=np.float32)
    seconds = {"read": 0.0, "model": 0.0}
    stems = frame["filename_hr"].tolist()
    for start in range(0, len(stems), CHUNK):
        began = time.perf_counter()
        inputs = read_inputs(stems[start:start + CHUNK])
        seconds["read"] += time.perf_counter() - began
        began = time.perf_counter()
        tokens[start:start + len(inputs)] = xecg_tokens(model, inputs)
        seconds["model"] += time.perf_counter() - began
        LOG.info("extracted %d / %d", start + len(inputs), len(stems))
    return tokens, seconds


def check_tokens(frame: pd.DataFrame, tokens: np.ndarray) -> dict[str, float | int]:
    """
    Require the token mean to equal the cached pooled feature for every row in the cache.

    Parameters
    ----------
    frame : pd.DataFrame
        Extracted rows.
    tokens : np.ndarray
        Their tokens.

    Returns
    -------
    dict[str, float | int]
        Rows compared and the largest absolute difference.

    Raises
    ------
    ValueError
        If the difference exceeds ``FEATURE_TOLERANCE``.
    """
    _, ids, _ = opened_cache()
    present = frame["ecg_id"].isin(ids).to_numpy()
    reference = cached(frame["ecg_id"].to_numpy()[present])
    difference = float(np.max(np.abs(tokens[present].mean(axis=1) - reference)))
    if difference > FEATURE_TOLERANCE:
        raise ValueError(f"Token mean differs from the cached feature by {difference}")
    return {"rows_compared": int(present.sum()), "max_abs_difference": difference}


def premature_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Sections overlapping a rule-detected premature beat, for the premature-beat set.

    Parameters
    ----------
    frame : pd.DataFrame
        Premature-beat rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, int]]
        Mask of included rows, boolean ``[included, 40]`` targets and exclusion counts.
    """
    included, targets = [], []
    counts = {"too_few_peaks": 0, "no_premature_beat": 0}
    for stem in frame["filename_hr"]:
        peaks = r_peaks(read_ptb_float64(stem), FS)
        windows = premature_windows(peaks, FS)
        if len(peaks) < 4:
            counts["too_few_peaks"] += 1
        elif not windows:
            counts["no_premature_beat"] += 1
        included.append(bool(windows))
        if windows:
            targets.append(section_overlap(windows))
    return np.array(included), np.array(targets).reshape(-1, SECTIONS), counts


def auroc_ap(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUROC and average precision."""
    return {"auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score))}


def analyse(rows: dict[str, pd.DataFrame], maps: dict[str, np.ndarray], whole: np.ndarray,
            targets: tuple[np.ndarray, np.ndarray, dict[str, int]]) -> dict[str, Any]:
    """
    Every prespecified metric, interval and reading.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Output of ``select_rows``.
    maps : dict[str, np.ndarray]
        ``[scored rows, 40]`` maps keyed by name, in ``rows["scored"]`` order.
    whole : np.ndarray
        Experiment 026 whole-ECG Mahalanobis of the evaluation rows (NaN in smoke mode).
    targets : tuple[np.ndarray, np.ndarray, dict[str, int]]
        Output of ``premature_targets``.

    Returns
    -------
    dict[str, Any]
        Thresholds, per-map metrics, contrasts and readings.
    """
    scored = rows["scored"]
    position = pd.Series(np.arange(len(scored)), index=scored.index)
    take = {name: position.loc[rows[name].index].to_numpy() for name in ("evaluation", "pvc", "benign")}
    evaluation = rows["evaluation"]
    y = evaluation["standard"].to_numpy(int)
    patients = evaluation["patient_id"].to_numpy()
    normal = take["evaluation"][y == 0]
    included, overlap, exclusions = targets
    pvc_rows = take["pvc"][included]
    pvc_patients = rows["pvc"]["patient_id"].to_numpy()[included]
    sets = {"normal": normal, "positive": take["evaluation"][y == 1], "pvc": take["pvc"],
            "benign": take["benign"]}
    thresholds = {name: red_threshold(values[normal]) for name, values in maps.items()}
    red = {name: values > thresholds[name] for name, values in maps.items()}
    result: dict[str, Any] = {"thresholds": thresholds, "pvc_included": int(included.sum()),
                              "pvc_exclusions": exclusions, "maps": {}}
    for name, values in maps.items():
        hit, chance = pointing(values[pvc_rows], overlap)
        ecg_score = values[take["evaluation"]].max(axis=1)
        result["maps"][name] = {
            **auroc_ap(y, ecg_score),
            "hit_rate": float(hit.mean()), "chance_rate": float(chance.mean()),
            "hit_minus_chance": bootstrap(pvc_patients, hit - chance, DRAWS, SEED),
            "any_red": {group: float(red[name][index].any(axis=1).mean()) for group, index in sets.items()},
            "mean_red_sections": {group: float(red[name][index].sum(axis=1).mean())
                                  for group, index in sets.items()},
            "sensitivity_at_budget": bootstrap(patients[y == 1], red[name][sets["positive"]].any(axis=1)
                                               .astype(float), DRAWS, SEED),
            "normal_position_mean": values[normal].mean(axis=0).tolist(),
        }
        if name != "U":
            result["maps"][name]["auroc_minus_U"] = paired_auroc_difference(
                patients, y, ecg_score, maps["U"][take["evaluation"]].max(axis=1), DRAWS, SEED)
    benign_patients = rows["benign"]["patient_id"].to_numpy()
    benign_difference = (red["G"][take["benign"]].any(axis=1).astype(float)
                         - red["U"][take["benign"]].any(axis=1).astype(float))
    result["benign_any_red_G_minus_U"] = bootstrap(benign_patients, benign_difference, DRAWS, SEED)
    u_score = maps["U"][take["evaluation"]].max(axis=1)
    if np.isfinite(whole).all():
        result["whole_ecg_026"] = auroc_ap(y, whole)
        result["U_minus_whole_ecg_026"] = paired_auroc_difference(patients, y, u_score, whole, DRAWS, SEED)
        detection = result["maps"]["U"]["auroc"] >= result["whole_ecg_026"]["auroc"] - DETECTION_MARGIN
    else:
        detection = None
    part1 = result["maps"]["U"]["hit_minus_chance"]["ci_low"] > 0
    result["reading"] = {
        "part1_localizes": bool(part1), "part1_keeps_detection": detection,
        "part2_improves_detection": bool(result["maps"]["G"]["auroc_minus_U"]["ci_low"] > 0),
        "part2_fewer_benign_marks": bool(result["benign_any_red_G_minus_U"]["ci_high"] < 0),
        "jepa_fallback_triggered": None if detection is None else bool(not (part1 and detection)),
    }
    return result


def example_rows(rows: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Lowest ECG ID of each example group among the scored rows."""
    scored, statements = rows["scored"], rows["statements"]
    groups = {
        "NORM": rows["evaluation"][rows["evaluation"]["standard"] == 0],
        "PVC": rows["pvc"], "benign": rows["benign"],
        **{name: scored[scored["ecg_id"].map(lambda i, n=name: n in statements.at[i, "subclasses"])
                        .to_numpy()] for name in ("IMI", "AMI", "CLBBB", "LVH")},
    }
    return {name: int(groups[name]["ecg_id"].min()) for name in EXAMPLES if len(groups[name])}


def draw_examples(rows: dict[str, pd.DataFrame], maps: dict[str, np.ndarray], thresholds: dict[str, float],
                  folder: Path) -> dict[str, int]:
    """
    Save the rule-chosen example figures with the red sections of ``U`` and ``G``.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Output of ``select_rows``.
    maps : dict[str, np.ndarray]
        Maps in ``rows["scored"]`` order.
    thresholds : dict[str, float]
        Red thresholds of each map.
    folder : Path
        Destination folder.

    Returns
    -------
    dict[str, int]
        ECG ID of each example.
    """
    folder.mkdir(parents=True, exist_ok=True)
    scored = rows["scored"].reset_index()
    examples = example_rows(rows)
    for name, ecg_id in examples.items():
        index = int(scored.index[scored["ecg_id"] == ecg_id][0])
        red = {"U (unsupervised)": maps["U"][index] > thresholds["U"],
               "G (label-guided)": maps["G"][index] > thresholds["G"]}
        figure = plot_sections(read_ptb_float64(scored.at[index, "filename_hr"]), FS, red,
                               f"{name}: PTB-XL ECG {ecg_id}")
        figure.savefig(folder / f"{name}_{ecg_id}.png", dpi=110)
    return examples


def run(smoke: bool, output: Path) -> None:
    """
    Run the experiment end to end and write its outputs.

    Parameters
    ----------
    smoke : bool
        Training-only smoke test (no development row is scored).
    output : Path
        Final output folder; it must not exist.
    """
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(partial / "run.log")])
    run_identity = identity()
    rows = select_rows(smoke)
    LOG.info("rows: %s", {name: len(frame) for name, frame in rows.items()})
    if smoke:
        pool = rows["pool"]
        with threadpool_limits(limits=PROBE_THREADS):
            head = fit_logistic(cached(pool["ecg_id"].to_numpy()), pool["standard"].to_numpy(int))
        reproduction, whole = {}, np.full(len(rows["evaluation"]), np.nan)
    else:
        reproduction, head, whole = reproduce026(rows)
    LOG.info("Experiment 026 reproduction: %s", reproduction)

    frame = rows["extract"]
    with gpu_lock("cuda"):
        model = load_xecg_backbone()
        order = order_check(model, read_inputs(rows["fit"]["filename_hr"].iloc[:ORDER_RECORDS].tolist()))
        if order != {"first": [0] * ORDER_RECORDS, "last": [SECTIONS - 1] * ORDER_RECORDS}:
            raise ValueError(f"Sections are not in time order: {order}")
        _, profile = extract(model, frame.iloc[:PROFILE_RECORDS])
        per_record = sum(profile.values()) / PROFILE_RECORDS
        projected = time.perf_counter() - started + per_record * len(frame) + ANALYSIS_RESERVE_SECONDS
        LOG.info("profile %s s per record, projected total %.1f s", per_record, projected)
        if projected > CEILING_SECONDS:
            raise RuntimeError(f"Projected {projected:.0f} s exceeds the {CEILING_SECONDS:.0f} s ceiling")
        tokens, extraction = extract(model, frame)
        del model
        torch.cuda.empty_cache()
    feature_check = check_tokens(frame, tokens)
    LOG.info("token check: %s", feature_check)

    fit_tokens, scored_tokens = tokens[:len(rows["fit"])], tokens[len(rows["fit"]):]
    with threadpool_limits(limits=4):
        reference = fit_section_reference(fit_tokens, SEED)
        maps = {"U": section_mahalanobis(reference, scored_tokens),
                "U_kmeans": section_kmeans(reference, scored_tokens),
                "U_uncentred": section_mahalanobis(reference, scored_tokens, centred=False),
                "G": readout_contributions(head, scored_tokens)}
    maps["G_gated"] = gated(maps["U"], maps["G"])
    position = pd.Series(np.arange(len(rows["scored"])), index=rows["scored"].index)
    evaluation_rows = position.loc[rows["evaluation"].index].to_numpy()
    scaler, model = head
    logits = model.decision_function(scaler.transform(cached(rows["evaluation"]["ecg_id"].to_numpy())))
    logit_difference = float(np.max(np.abs(maps["G"][evaluation_rows].mean(axis=1) - logits)))
    if logit_difference > LOGIT_TOLERANCE:
        raise ValueError(f"Mean section contribution differs from the logit by {logit_difference}")
    del tokens, fit_tokens, scored_tokens

    targets = premature_targets(rows["pvc"])
    result = analyse(rows, maps, whole, targets)
    examples = draw_examples(rows, maps, result["thresholds"], partial / "figures")
    write_npz_atomic(partial / "section_scores.npz", ecg_ids=rows["scored"]["ecg_id"].to_numpy(np.int64),
                     **maps)
    (partial / "reference.pkl").write_bytes(pickle.dumps(
        {"reference": reference, "head": head, "thresholds": result["thresholds"]}, protocol=5))
    write_json_atomic(partial / "result.json", {
        "status": "smoke" if smoke else "completed", "identity": run_identity,
        "counts": {name: len(rows[name])
                   for name in ("fit", "pool", "evaluation", "pvc", "benign", "scored")},
        "reproduction026": reproduction, "order_check": order, "token_check": feature_check,
        "logit_max_abs_difference": logit_difference, "profile_seconds": profile,
        "extraction_seconds": extraction, "examples": examples, **result,
        "outputs_sha256": {name: sha256_file(partial / name)
                           for name in ("section_scores.npz", "reference.pkl")},
        "seed": SEED, "draws": DRAWS, "total_seconds": time.perf_counter() - started,
        "calibration_test_evaluated": False,
    })
    partial.rename(output)
    LOG.info("done in %.1f s: %s", time.perf_counter() - started, result["reading"])


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="training-only smoke test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    run(arguments.smoke, arguments.output)


if __name__ == "__main__":
    main()
