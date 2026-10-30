"""HEADS-01 stages: qualify, smoke, profile, extract, evaluate and aggregate export."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import ROOT
from .age_cohort_inputs import canonical_integral_id, parse_boolean
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .full_development import fit_logistic, predict
from .heads01_features import pooled_parity, run_blocks, superclass_table
from .heads01_heads import (
    DRAWS,
    SEED,
    SUPERCLASSES,
    fit_hgb,
    fit_superclass,
    gates,
    mixture_score,
    paired_sensitivity,
)
from .intervals import paired_auroc_difference
from .pr74_fresh_baseline import stage_inputs
from .pr74_raw_study import cohorts, evaluation_features, feature_matrix
from .pr74_screening_features import checkpoint_hashes
from .pr74_screening_io import read_json, relative, stored, verify_hashes
from .pr74_screening_metrics import budget_threshold, point_metrics, threshold_sweep
from .pr74_screening_study import assert_evaluation_independence, primary_comparison, score_models

OUTPUT = Path("outputs/heads01_token_superclass_20261028")
RAW = Path("outputs/pr74_from_raw_validation_20261028")
BASELINE = Path("outputs/pr74_fresh_baseline_exact_20261028")
COHORTS = Path("outputs/cohort25_recovery_20261028/cold_start/cohorts_25k")
STRICT = Path("outputs/cohort25_recovery_20261028/cold_start/strict_matched")
PROTOCOL = Path("docs/heads01-token-superclass-protocol-2026-10-28.md")
REPORT = Path("reports/heads01-token-superclass-2026-10-28/aggregate.json")
FROZEN = ("pr74_fresh_baseline", "pr74_raw_study", "pr74_screening_features", "pr74_screening_io",
          "pr74_screening_metrics", "pr74_screening_study", "cpu_screening041", "full_development",
          "intervals", "challenge_labels", "eda/ptbxl", "external_encoders", "xecg", "gpu", "files", "paths")
PARTS = {"training": 28568, "evaluation": 21677}
CURRENT, DEFAULT = "all_age_blend", "ptb_baseline_blend"
PRIMARY = ("token_pool_blend", "superclass_blend")


def binding() -> dict[str, Any]:
    """
    Require committed HEADS-01 source and protocol, then capture every scientific source hash.

    Returns
    -------
    dict[str, Any]
        Source and protocol hashes and the git revision.

    Raises
    ------
    ValueError
        If an own source or the protocol differs from ``HEAD``.
    """
    own = sorted((ROOT / "ecg_experiment").glob("heads01_*.py"))
    own += [ROOT / "scripts/experiments/run_heads01.py", ROOT / PROTOCOL]
    own += sorted((ROOT / "tests").glob("test_heads01*.py"))
    for path in own:
        blob = subprocess.check_output(["git", "show", "HEAD:" + stored(path.relative_to(ROOT))], cwd=ROOT)
        if blob != path.read_bytes():
            raise ValueError(f"Scientific source must be committed before execution: {path.name}")
    sources = own + [ROOT / f"ecg_experiment/{name}.py" for name in FROZEN]
    return {"source_sha256": {stored(path.relative_to(ROOT)): sha256_file(path) for path in sources},
            "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(ROOT / PROTOCOL)},
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                    text=True).strip(),
            "scope": "reused_development", "threads": 2}


def finish(input_root: Path, output: Path, stage: str, receipt: dict[str, Any], inputs: dict[str, str],
           details: dict[str, Any]) -> dict[str, Any]:
    """Recheck inputs and sources, then bind every stage artifact in a completion receipt."""
    verify_hashes(input_root, inputs)
    verify_hashes(ROOT, receipt["source_sha256"])
    directory = input_root / relative(output) / stage
    artifacts = {stored(output / stage / path.name): sha256_file(path)
                 for path in sorted(directory.iterdir()) if path.is_file() and path.name != "completion.json"}
    value = {**receipt, **details, "status": "completed", "stage": stage, "input_sha256": inputs,
             "artifacts": artifacts}
    write_json_atomic(directory / "completion.json", value)
    return value


def require(input_root: Path, output: Path, stage: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify a completed HEADS-01 stage, its inputs, artifacts and source."""
    path = relative(output) / stage / "completion.json"
    receipt = read_json(input_root / path)
    if receipt.get("status") != "completed":
        raise ValueError(f"Stage {stage} is not complete")
    hashes = {stored(path): sha256_file(input_root / path), **receipt["artifacts"]}
    verify_hashes(input_root, {**hashes, **receipt["input_sha256"]})
    verify_hashes(ROOT, receipt["source_sha256"])
    return receipt, hashes


def predecessor(input_root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    """Verify every completed predecessor stage and the cohort receipts the study reuses."""
    hashes = {}
    for directory, stage in ((RAW, "extract"), (RAW, "evaluate"), (BASELINE, "features"), (BASELINE, "fit")):
        _, bound = stage_inputs(input_root, directory, stage)
        hashes.update(bound)
    pool, arms, bound = cohorts(input_root, COHORTS, STRICT)
    hashes.update(bound)
    return pool, arms, hashes


def site_point(y: np.ndarray, held: np.ndarray, scores: np.ndarray, site: str) -> dict[str, Any]:
    """Calibrate on one site's normal calibration rows and score its evaluation rows."""
    calibrated = budget_threshold(y[~held], scores[~held])
    threshold = calibrated["threshold"]
    return {"pool": site, "label": site.upper() + " reused development", "threshold": threshold,
            "comparison": ">", "calibration": calibrated,
            "metrics": point_metrics(y[held], scores[held], threshold),
            "sweep": threshold_sweep(y[held], scores[held], threshold)}


def replay_saved(input_root: Path) -> dict[str, Any]:
    """Recompute all 30 saved predecessor points and its primary interval exactly from saved scores."""
    saved = read_json(input_root / RAW / "evaluate/screening_evaluation.json")
    expected = {(model["id"], point["pool"]): point for model in saved["models"]
                for point in model["evaluations"]}
    checked = 0
    for site in ("sph", "ptb"):
        data = np.load(input_root / RAW / f"evaluate/{site}_predictions.npz")
        y, held = data["target"].astype(int), data["evaluation"].astype(bool)
        records, arrays = [], {}
        for model_id in [key for key in data.files if key.endswith(("_hgb", "_xecg", "_blend"))]:
            point = site_point(y, held, data[model_id], site)
            if point != expected[(model_id, site)]:
                raise ValueError(f"Saved predecessor point does not reproduce: {model_id}/{site}")
            records.append({"model_id": model_id, "evaluation": point})
            arrays[model_id] = data[model_id]
            checked += 1
        if site == "sph":
            frame = pd.DataFrame({"evaluation": held, "patient_id": data["patient_id"], "target": y})
            if primary_comparison(records, arrays, frame) != saved["primary"]:
                raise ValueError("Saved predecessor primary interval does not reproduce")
    if checked != len(expected) or checked != 30:
        raise ValueError("Predecessor point inventory differs")
    return {"points_reproduced_exactly": checked, "primary_interval_reproduced_exactly": True}


def refit_reference(input_root: Path, pool: pd.DataFrame, arms: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Refit the all-age and age-preferred binary heads and require the saved scores to 1e-12."""
    raw, vectors = feature_matrix(input_root, RAW, BASELINE, pool)
    result = {}
    for name in ("all_age", "younger"):
        positions = pd.Index(pool["record_id"]).get_indexer(arms[name]["record_id"])
        y = arms[name]["target"].to_numpy(dtype=int)
        model = {"hgb": fit_hgb(raw.iloc[positions], y), "xecg": fit_logistic(vectors[positions], y)}
        for site in ("sph", "ptb"):
            frame, site_vectors, site_raw = evaluation_features(input_root, BASELINE, site)
            data = np.load(input_root / RAW / f"evaluate/{site}_predictions.npz")
            if not np.array_equal(data["ecg_id"], frame["ecg_id"].to_numpy(dtype=str)):
                raise ValueError("Saved predictions and evaluation rows are misaligned")
            for kind, scores in score_models(model, site_vectors, site_raw).items():
                worst = float(np.max(np.abs(scores - data[f"{name}_{kind}"])))
                if worst > 1e-12:
                    raise ValueError(f"Refit {name}_{kind} at {site} differs by {worst}")
                result[f"{name}_{kind}/{site}"] = worst
    return {"max_absolute_score_difference": result}


def qualify(input_root: Path, output: Path) -> dict[str, Any]:
    """Run the predecessor integrity check and reconstruct the training superclass labels."""
    receipt = binding()
    started = time.perf_counter()
    pool, arms, hashes = predecessor(input_root)
    directory = input_root / relative(output) / "qualify"
    directory.mkdir(parents=True, exist_ok=False)
    replay = replay_saved(input_root)
    refit = refit_reference(input_root, pool, arms)
    labels = superclass_table(input_root, pool)
    labels.to_csv(directory / "labels.csv", index=False)
    cohort = labels.loc[labels["record_id"].isin(arms["all_age"]["record_id"])]
    counts = {name: int(cohort[name].sum()) for name in SUPERCLASSES}
    details = {**replay, **refit, "pool_records": len(labels), "labels_match_target": True,
               "all_age_superclass_positives": counts, "all_age_negatives": int(cohort["target"].eq(0).sum()),
               "wall_seconds": time.perf_counter() - started}
    write_json_atomic(directory / "qualification.json", details)
    return finish(input_root, output, "qualify", receipt, hashes, details)


def baseline_rows(input_root: Path, site: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one site's fresh baseline rows and saved pooled features in file order."""
    source = input_root / BASELINE / "features"
    rows = pd.read_csv(source / f"{site}_rows.csv", dtype=str, keep_default_na=False)
    return rows, np.load(source / f"{site}_xecg.npy", mmap_mode="r")


def part_rows(input_root: Path, part: str, pool: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Rows of one extraction part in fixed block order with their saved predecessor pooled features.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    part : str
        ``"training"`` or ``"evaluation"``.
    pool : pd.DataFrame
        Verified admitted training union.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Rows with ``block`` and ``record_id``, and the aligned saved ``[rows, 1024]`` features.
    """
    ptb, ptb_saved = baseline_rows(input_root, "ptb")
    ptb_ids = "ptbxl:" + ptb["ecg_id"].map(canonical_integral_id)
    if part == "training":
        challenge = pd.read_csv(input_root / RAW / "extract/rows.csv", dtype=str, keep_default_na=False)
        first = challenge.assign(block="challenge")
        keep = ptb_ids.isin(pool.loc[pool["source"].eq("ptbxl"), "record_id"]).to_numpy()
        second = ptb.loc[keep].assign(block="ptb_train", record_id=ptb_ids[keep])
        saved = np.concatenate([np.load(input_root / RAW / "extract/xecg.npy"), ptb_saved[keep]])
    else:
        sph, sph_saved = baseline_rows(input_root, "sph")
        first = sph.assign(block="sph")
        keep = ~parse_boolean(ptb["fit"]).to_numpy()
        second = ptb.loc[keep].assign(block="ptb_eval", record_id=ptb_ids[keep])
        saved = np.concatenate([sph_saved, ptb_saved[keep]])
    rows = pd.concat([first, second], ignore_index=True)
    if len(rows) != PARTS[part] or rows["record_id"].duplicated().any() or len(saved) != len(rows):
        raise ValueError(f"Unexpected {part} extraction rows")
    return rows, saved


def profile(input_root: Path, output: Path, part: str, device: int) -> dict[str, Any]:
    """Measure the real raw-to-token path of one part on one GPU before full extraction."""
    receipt = binding()
    _, hashes = require(input_root, output, "qualify")
    pool, _, bound = predecessor(input_root)
    hashes.update(bound)
    hashes.update(checkpoint_hashes(input_root))
    rows, saved = part_rows(input_root, part, pool)
    selected = rows.groupby("block", sort=False).head(24)
    stage = f"profile_{part}"
    directory = input_root / relative(output) / stage
    directory.mkdir(parents=True, exist_ok=False)
    measured = run_blocks(input_root, selected.reset_index(drop=True), directory, device, progress=False)
    parity = pooled_parity(np.load(directory / "pooled.npy"), saved[selected.index.to_numpy()])
    forecast = 2 * measured["wall_seconds"] / len(selected) * len(rows) + 300
    total = sum(PARTS.values()) * (40 * 1024 * 2 + 2 * 1024 * 4)
    resource = {"free_at_least_4gib": measured["initial_free_gpu_bytes"] >= 4 * 1024**3,
                "peak_under_3gib": measured["peak_gpu_bytes"] < 3 * 1024**3,
                "forecast_under_2hours": forecast < 7200, "outputs_under_6gib": total < 6 * 1024**3}
    selected.to_csv(directory / "rows.csv", index=False)
    details = {**measured, "part": part, "device": device, "part_records": len(rows),
               "forecast_seconds": forecast, "output_bytes_forecast": total, "resource_gates": resource,
               "pooled_parity": parity}
    write_json_atomic(directory / "profile.json", details)
    if not all(resource.values()):
        raise RuntimeError(f"HEADS-01 profile gate failed: {resource}")
    return finish(input_root, output, stage, receipt, hashes, details)


def extract(input_root: Path, output: Path, part: str, device: int) -> dict[str, Any]:
    """Extract every row of one part after its passing profile on the same GPU."""
    receipt = binding()
    profiled, hashes = require(input_root, output, f"profile_{part}")
    if not all(profiled["resource_gates"].values()) or profiled["device"] != device:
        raise ValueError("A passing profile on the same GPU is required")
    pool, _, bound = predecessor(input_root)
    hashes.update(bound)
    hashes.update(checkpoint_hashes(input_root))
    rows, saved = part_rows(input_root, part, pool)
    stage = f"extract_{part}"
    directory = input_root / relative(output) / stage
    directory.mkdir(parents=True, exist_ok=False)
    rows[["record_id", "block"]].to_csv(directory / "rows.csv", index=False)
    measured = run_blocks(input_root, rows, directory, device)
    parity = {block: pooled_parity(np.load(directory / "pooled.npy", mmap_mode="r")[index.to_numpy()],
                                   saved[index.to_numpy()])
              for block, index in rows.groupby("block", sort=False).groups.items()}
    (directory / "progress.json").unlink(missing_ok=True)
    return finish(input_root, output, stage, receipt, hashes,
                  {**measured, "part": part, "device": device, "pooled_parity": parity,
                   "encoder_fitted": False})


def synthetic_run(seed: int = 0) -> dict[str, Any]:
    """Exercise fitting, scoring, calibration, contrasts and export shapes on synthetic data only."""
    rng = np.random.default_rng(seed)
    n, width = 900, 32
    flags = (rng.random((n, len(SUPERCLASSES))) < 0.3).astype(int)
    y = flags.max(axis=1)
    raw = pd.DataFrame(rng.normal(size=(n, 130)) + y[:, None] * 0.3)
    raw.iloc[::7, 3] = np.nan
    directions = rng.normal(size=(len(SUPERCLASSES), width))
    mean = (rng.normal(size=(n, width)) + flags @ directions).astype(np.float32)
    maximum = (mean + rng.normal(size=(n, width))).astype(np.float32)
    train = np.arange(n) < 500
    fitted, _ = fit_arms(raw[train], mean[train], maximum[train], y[train], flags[train])
    scores = score_arms(fitted, raw[~train], mean[~train], maximum[~train])
    count = int((~train).sum())
    scores["ptb_baseline_blend"] = rng.random(count)
    frame = pd.DataFrame({"target": y[~train], "patient_id": [f"p{i // 2}" for i in range(count)]})
    frame["evaluation"] = (frame["target"].eq(1) | (np.arange(len(frame)) % 2 == 0)).to_numpy()
    for column, name in enumerate(SUPERCLASSES):
        frame[name] = flags[~train, column]
    held, target = frame["evaluation"].to_numpy(), frame["target"].to_numpy()
    points = {name: site_point(target, held, values, "synthetic") for name, values in scores.items()}
    result = comparisons(points, scores, frame, draws=50)
    breakdown = by_superclass(frame, scores, points)
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Synthetic smoke produced nonfinite scores")
    return {"models": len(points), "primary": list(result["primary"]), "superclasses": list(breakdown)}


def smoke(input_root: Path, output: Path, device: int) -> dict[str, Any]:
    """Short smoke test: real token parity on 32 training records and the synthetic pipeline."""
    receipt = binding()
    _, hashes = require(input_root, output, "qualify")
    pool, _, bound = predecessor(input_root)
    hashes.update(bound)
    hashes.update(checkpoint_hashes(input_root))
    rows, saved = part_rows(input_root, "training", pool)
    selected = rows.groupby("block", sort=False).head(16)
    directory = input_root / relative(output) / "smoke"
    directory.mkdir(parents=True, exist_ok=False)
    measured = run_blocks(input_root, selected.reset_index(drop=True), directory, device, progress=False)
    parity = pooled_parity(np.load(directory / "pooled.npy"), saved[selected.index.to_numpy()])
    details = {"not_a_result": True, "token_records": len(selected), "pooled_parity": parity,
               "token_seconds": measured["wall_seconds"], "synthetic": synthetic_run()}
    write_json_atomic(directory / "smoke.json", details)
    return finish(input_root, output, "smoke", receipt, hashes, details)


def fit_arms(raw: pd.DataFrame, mean: np.ndarray, maximum: np.ndarray, y: np.ndarray,
             flags: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Fit every HEADS-01 head on one training cohort before any evaluation score.

    Parameters
    ----------
    raw : pd.DataFrame
        Training measurements.
    mean, maximum : np.ndarray
        Saved pooled and new token-maximum xECG features, aligned with ``raw``.
    y : np.ndarray
        Binary targets.
    flags : np.ndarray
        ``[records, 4]`` superclass flags.

    Returns
    -------
    tuple[dict[str, Any], dict[str, float]]
        Fitted heads and fit seconds.
    """
    both = np.hstack([mean, maximum])
    recipes = {
        "hgb": lambda: fit_hgb(raw, y),
        "xecg": lambda: fit_logistic(mean, y),
        "token_xecg": lambda: fit_logistic(both, y),
        "superclass_hgb": lambda: fit_superclass(raw, y, flags, "hgb"),
        "superclass_xecg": lambda: fit_superclass(mean, y, flags, "logistic"),
        "token_superclass_xecg": lambda: fit_superclass(both, y, flags, "logistic"),
    }
    fitted, seconds = {}, {}
    for name, recipe in recipes.items():
        started = time.perf_counter()
        fitted[name] = recipe()
        seconds[name] = time.perf_counter() - started
        print(f"Fitted {name} in {seconds[name]:.1f} s", flush=True)
    return fitted, seconds


def score_arms(fitted: dict[str, Any], raw: pd.DataFrame, mean: np.ndarray,
               maximum: np.ndarray) -> dict[str, np.ndarray]:
    """Score every fixed component and 50:50 blend of the fitted HEADS-01 heads."""
    both = np.hstack([mean, maximum])
    scores = {"all_age_hgb": fitted["hgb"].predict_proba(raw)[:, 1],
              "all_age_xecg": predict(fitted["xecg"], mean),
              "token_pool_xecg": predict(fitted["token_xecg"], both),
              "superclass_hgb": mixture_score(fitted["superclass_hgb"], raw),
              "superclass_xecg": mixture_score(fitted["superclass_xecg"], mean),
              "token_superclass_xecg": mixture_score(fitted["token_superclass_xecg"], both)}
    scores["all_age_blend"] = 0.5 * scores["all_age_hgb"] + 0.5 * scores["all_age_xecg"]
    scores["token_pool_blend"] = 0.5 * scores["all_age_hgb"] + 0.5 * scores["token_pool_xecg"]
    scores["superclass_blend"] = 0.5 * scores["superclass_hgb"] + 0.5 * scores["superclass_xecg"]
    scores["token_superclass_blend"] = 0.5 * scores["superclass_hgb"] + 0.5 * scores["token_superclass_xecg"]
    return scores


def contrast(points: dict[str, dict], scores: dict[str, np.ndarray], frame: pd.DataFrame, first: str,
             second: str, levels: tuple[float, ...], draws: int = DRAWS) -> dict[str, Any]:
    """Paired sensitivity, specificity and AUROC contrast of two frozen models on evaluation rows."""
    held = frame["evaluation"].to_numpy()
    y = frame.loc[held, "target"].to_numpy(dtype=int)
    patients = frame.loc[held, "patient_id"].to_numpy(dtype=str)
    a, b = scores[first][held], scores[second][held]
    sensitivity = paired_sensitivity(patients, y, a > points[first]["threshold"],
                                     b > points[second]["threshold"], levels, draws, SEED)
    return {"first": first, "second": second, "sensitivity": sensitivity,
            "specificity_change": points[first]["metrics"]["specificity"]
            - points[second]["metrics"]["specificity"],
            "auroc": paired_auroc_difference(patients, y, a, b, draws, SEED)}


def comparisons(points: dict[str, dict], scores: dict[str, np.ndarray], frame: pd.DataFrame,
                draws: int = DRAWS) -> dict[str, Any]:
    """Prespecified primary, default-context and descriptive SPH contrasts with the decision."""
    primary = {}
    for name in PRIMARY:
        result = contrast(points, scores, frame, name, CURRENT, (0.95, 0.975), draws)
        result["gates"] = gates(result["sensitivity"], result["specificity_change"])
        primary[name] = result
    default = {name: contrast(points, scores, frame, name, DEFAULT, (0.95,), draws) for name in PRIMARY}
    pairs = (("token_superclass_blend", CURRENT), ("token_pool_xecg", "all_age_xecg"),
             ("superclass_hgb", "all_age_hgb"), ("superclass_xecg", "all_age_xecg"))
    secondary = {f"{a}-{b}": contrast(points, scores, frame, a, b, (0.95,), draws) for a, b in pairs}
    established = [name for name in PRIMARY if primary[name]["gates"]["established"]]
    return {"primary": primary, "versus_default": default, "secondary": secondary,
            "established_improvements": established, "pipeline_promoted": False}


def by_superclass(frame: pd.DataFrame, scores: dict[str, np.ndarray], points: dict[str, dict]) -> dict:
    """Descriptive SPH sensitivity within each superclass flag, without intervals."""
    held = frame["evaluation"].to_numpy()
    result = {}
    for name in SUPERCLASSES:
        mask = held & frame["target"].eq(1).to_numpy() & pd.to_numeric(frame[name]).eq(1).to_numpy()
        result[name] = {"positives": int(mask.sum()), **{
            model: float(np.mean(scores[model][mask] > points[model]["threshold"]))
            for model in ("ptb_baseline_blend", CURRENT, *PRIMARY, "token_superclass_blend")}}
    return result


def site_scores(input_root: Path, output: Path, site: str, fitted: dict[str, Any],
                cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score one site with the fitted heads and attach the saved default and reference predictions."""
    frame, mean, raw = evaluation_features(input_root, BASELINE, site)
    assert_evaluation_independence({"all_age": cohort}, site, frame)
    rows = pd.read_csv(input_root / output / "extract_evaluation/rows.csv", dtype=str)
    block = rows["block"].eq("sph" if site == "sph" else "ptb_eval").to_numpy()
    prefix = "sph:" if site == "sph" else "ptbxl:"
    ids = frame["ecg_id"].map(canonical_integral_id) if site == "ptb" else frame["ecg_id"]
    if rows.loc[block, "record_id"].tolist() != (prefix + ids).tolist():
        raise ValueError(f"Token rows and {site} evaluation rows are misaligned")
    maximum = np.load(input_root / output / "extract_evaluation/max.npy", mmap_mode="r")[block]
    scores = score_arms(fitted, raw, mean, np.asarray(maximum))
    saved = np.load(input_root / RAW / f"evaluate/{site}_predictions.npz")
    if not np.array_equal(saved["ecg_id"], frame["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("Saved predictions and evaluation rows are misaligned")
    for kind in ("hgb", "xecg", "blend"):
        worst = float(np.max(np.abs(scores[f"all_age_{kind}"] - saved[f"all_age_{kind}"])))
        if worst > 1e-12:
            raise ValueError(f"Refitted reference all_age_{kind} differs at {site} by {worst}")
        scores[f"ptb_baseline_{kind}"] = saved[f"ptb_baseline_{kind}"]
    return frame, scores


def training_inputs(input_root: Path, output: Path, pool: pd.DataFrame,
                    cohort: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Align saved training features, new token maxima and verified labels on the cohort's records."""
    raw, vectors = feature_matrix(input_root, RAW, BASELINE, pool)
    rows = pd.read_csv(input_root / output / "extract_training/rows.csv", dtype=str)
    maxima = np.load(input_root / output / "extract_training/max.npy", mmap_mode="r")
    labels = pd.read_csv(input_root / output / "qualify/labels.csv", dtype={"record_id": str})
    if labels["record_id"].tolist() != pool["record_id"].tolist():
        raise ValueError("Superclass labels are not aligned with the pool")
    positions = pd.Index(pool["record_id"]).get_indexer(cohort["record_id"])
    token_positions = pd.Index(rows["record_id"]).get_indexer(cohort["record_id"])
    if (positions < 0).any() or (token_positions < 0).any():
        raise ValueError("A cohort record lacks features")
    if not np.array_equal(labels["target"].to_numpy()[positions], cohort["target"].to_numpy(dtype=int)):
        raise ValueError("Cohort targets and labels disagree")
    flags = labels[list(SUPERCLASSES)].to_numpy(dtype=int)[positions]
    return raw.iloc[positions], vectors[positions], np.asarray(maxima[token_positions]), flags


def evaluate(input_root: Path, output: Path) -> dict[str, Any]:
    """Fit all arms on the all-age cohort, then score SPH and PTB and apply the decision rule."""
    receipt = binding()
    started = time.perf_counter()
    _, hashes = require(input_root, output, "qualify")
    for part in PARTS:
        _, bound = require(input_root, output, f"extract_{part}")
        hashes.update(bound)
    pool, arms, bound = predecessor(input_root)
    hashes.update(bound)
    cohort = arms["all_age"]
    raw, mean, maximum, flags = training_inputs(input_root, output, pool, cohort)
    fitted, seconds = fit_arms(raw, mean, maximum, cohort["target"].to_numpy(dtype=int), flags)
    directory = input_root / relative(output) / "evaluate"
    directory.mkdir(parents=True, exist_ok=False)
    models, decision, breakdown = {}, None, None
    for site in ("sph", "ptb"):
        frame, scores = site_scores(input_root, output, site, fitted, cohort)
        y, held = frame["target"].to_numpy(dtype=int), frame["evaluation"].to_numpy()
        points = {name: site_point(y, held, values, site) for name, values in scores.items()}
        for name, point in points.items():
            models.setdefault(name, []).append(point)
        if site == "sph":
            decision = comparisons(points, scores, frame)
            breakdown = by_superclass(frame, scores, points)
        write_npz_atomic(directory / f"{site}_predictions.npz", **scores,
                         ecg_id=frame["ecg_id"].to_numpy(dtype=str),
                         patient_id=frame["patient_id"].to_numpy(dtype=str), target=y, evaluation=held)
    joblib.dump({"schema_version": 1, "fitted": fitted}, directory / "model_bundle.joblib")
    result = {"schema_version": 1, "complete": True, "scope": "reused_development",
              "score_interpretation": "uncalibrated", "default_model_id": DEFAULT,
              "operating_point": {"normal_budget": 0.05, "comparison": ">",
                                  "selection": "calibration_normals_only"},
              "training": {"cohort": "all_age", "records": len(cohort),
                           "negatives": int(cohort["target"].eq(0).sum()),
                           "superclass_positives": fitted["superclass_hgb"]["counts"]},
              "models": [{"id": name, "evaluations": points} for name, points in models.items()],
              "decision": decision, "sph_sensitivity_by_superclass": breakdown, "fit_seconds": seconds}
    write_json_atomic(directory / "evaluation.json", result)
    return finish(input_root, output, "evaluate", receipt, hashes,
                  {"decision": decision, "fit_seconds": seconds,
                   "wall_seconds": time.perf_counter() - started,
                   "model_scores_computed_on_closed_challenge_or_echonext_test": 0})


def export(input_root: Path, output: Path) -> dict[str, Any]:
    """Write the aggregate-only report artifact from the completed evaluation and extraction receipts."""
    evaluated, _ = require(input_root, output, "evaluate")
    qualified, _ = require(input_root, output, "qualify")
    result = read_json(input_root / output / "evaluate/evaluation.json")
    result["receipt_sha256"] = {stage: sha256_file(input_root / output / stage / "completion.json")
                                for stage in ("qualify", "profile_training", "profile_evaluation",
                                              "extract_training", "extract_evaluation", "evaluate")}
    result["integrity"] = {key: qualified[key] for key in
                           ("points_reproduced_exactly", "primary_interval_reproduced_exactly",
                            "max_absolute_score_difference", "labels_match_target")}
    result["extraction"] = {}
    for part in PARTS:
        receipt = read_json(input_root / output / f"extract_{part}/completion.json")
        result["extraction"][part] = {key: receipt[key] for key in
                                      ("records", "wall_seconds", "peak_gpu_bytes", "device",
                                       "pooled_parity")}
    result["evaluate_wall_seconds"] = evaluated["wall_seconds"]
    write_json_atomic(ROOT / REPORT, result)
    return {"report": str(REPORT), "sha256": sha256_file(ROOT / REPORT)}
