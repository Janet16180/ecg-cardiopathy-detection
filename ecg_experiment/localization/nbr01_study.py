"""NBR-01 stages: counterpart, profile, select, prepare, encode, score, analyze and export."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import ROOT
from ..files import read_json, sha256_file, write_json_atomic, write_npz_atomic
from ..paths import to_stored
from ..pr74_screening_io import verify_hashes
from ..pr74_screening_metrics import budget_threshold
from . import nbr01_analysis as na
from . import nbr01_counterpart as nc
from . import nbr01_features as nf
from . import nbr01_selection as ns

OUTPUT = Path("outputs/nbr01_normal_beat_replacement_20261028")
WORK = Path("/tmp/nbr01-work")
PROTOCOL = Path("docs/nbr01-normal-beat-replacement-protocol-2026-10-28.md")
CAP = 300
NORMALS = 300
PROFILE_SEED = 20261029
REPRODUCE_070 = 100
COUNTERPART_TOLERANCE_MV = 1e-6
SCORE_TOLERANCE = 1e-12
MINIMUM_TERRITORY_PATIENTS = 40
BROUGHT = ("normative_map062", "morphology_boundaries", "fragment_localization", "combination066", "mil065",
           "locbench", "locbench_data", "locbench_cache", "ecgad", "replacement070")
FROZEN = ("pr74_fresh_baseline", "pr74_screening_features", "pr74_screening_metrics", "pr74_screening_io",
          "external_encoders", "xecg", "measurement_extractor042", "cpu_screening041", "full_development",
          "intervals", "gpu", "files", "paths", "heads01_features")


def binding(require_committed: bool = True) -> dict[str, Any]:
    """
    Hash the study's sources, the 062/066/070 machinery and the frozen pipeline it calls.

    Parameters
    ----------
    require_committed : bool
        Require every own source and the protocol to equal ``HEAD``.

    Returns
    -------
    dict[str, Any]
        Source hashes, protocol hash and git revision.

    Raises
    ------
    ValueError
        If an own source or the protocol differs from ``HEAD``.
    """
    own = sorted((ROOT / "ecg_experiment/localization").glob("*.py"))
    own += [ROOT / "scripts/experiments/run_nbr01.py", ROOT / "docs/locbench-partition-v1.json"]
    own += sorted((ROOT / "tests/localization").glob("test_nbr01*.py"))
    own += [ROOT / f"ecg_experiment/{name}.py" for name in BROUGHT]
    if require_committed:
        for path in [*own, ROOT / PROTOCOL]:
            blob = subprocess.check_output(["git", "show", "HEAD:" + to_stored(path)], cwd=ROOT)
            if blob != path.read_bytes():
                raise ValueError(f"Source must be committed before execution: {path.name}")
    sources = own + [ROOT / f"ecg_experiment/{name}.py" for name in FROZEN]
    protocol = ROOT / PROTOCOL
    return {"source_sha256": {to_stored(path): sha256_file(path) for path in sources},
            "protocol": {"path": str(PROTOCOL),
                         "sha256": sha256_file(protocol) if protocol.exists() else None},
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                    text=True).strip()}


def stage_dir(input_root: Path, stage: str) -> Path:
    """Create and return one stage's output directory."""
    directory = input_root / OUTPUT / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def complete(input_root: Path, stage: str, receipt: dict[str, Any],
             details: dict[str, Any]) -> dict[str, Any]:
    """Bind a stage's artifacts and sources in its completion receipt."""
    directory = input_root / OUTPUT / stage
    verify_hashes(ROOT, receipt["source_sha256"])
    artifacts = {to_stored(path): sha256_file(path) for path in sorted(directory.iterdir())
                 if path.is_file() and path.name != "completion.json"}
    value = {**receipt, **details, "status": "completed", "stage": stage, "artifacts": artifacts}
    write_json_atomic(directory / "completion.json", value)
    return value


def require(input_root: Path, stage: str) -> dict[str, Any]:
    """Verify a completed stage's artifacts and return its receipt."""
    receipt = read_json(input_root / OUTPUT / stage / "completion.json")
    if receipt.get("status") != "completed":
        raise ValueError(f"Stage {stage} is not complete")
    verify_hashes(input_root, receipt["artifacts"])
    return receipt


def counterpart(input_root: Path, workers: int) -> dict[str, Any]:
    """Refit the 062 rank-0 model and reproduce saved Experiment 070 counterparts."""
    receipt = binding(require_committed=False)
    directory = stage_dir(input_root, "counterpart")
    model, summary = nc.fit_rank0(workers)
    nf.save_model(directory / "rank0.npz", model)
    reproduction = nc.reproduce_070(model, REPRODUCE_070, workers)
    worst = max(value["max_abs_difference_mv"] for value in reproduction.values())
    details = {"fit": summary, "reproduction_070": reproduction, "tolerance_mv": COUNTERPART_TOLERANCE_MV,
               "passed": worst <= COUNTERPART_TOLERANCE_MV}
    write_json_atomic(directory / "counterpart.json", details)
    if not details["passed"]:
        raise ValueError(f"070 counterpart reproduction failed: {worst} mV")
    return complete(input_root, "counterpart", receipt, details)


def model_of(input_root: Path) -> Any:
    """Load the refitted rank-0 model of the completed counterpart stage."""
    require(input_root, "counterpart")
    return nf.load_model(input_root / OUTPUT / "counterpart/rank0.npz")


def profile(input_root: Path, device: int, workers: int) -> dict[str, Any]:
    """
    Time preparation and encoding on 20 ECGs and compare original-variant features, without scoring.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    device : int
        CUDA device.
    workers : int
        CPU processes.

    Returns
    -------
    dict[str, Any]
        Completion receipt with timings, forecasts and feature parity.
    """
    receipt = binding(require_committed=False)
    model = model_of(input_root)
    rows = ns.draw_sample(ns.evaluation_rows(input_root), 3, 5, PROFILE_SEED)
    work = WORK / "profile"
    started = time.perf_counter()
    prepared = nf.prepare_all(input_root, rows, model, work, workers)
    prepare_wall = time.perf_counter() - started
    kept = rows.loc[[status == "prepared" for _, status, _ in prepared]]
    encoded = nf.encode_records(input_root, kept, work, device)
    raw = pd.read_csv(input_root / "outputs/pr74_fresh_baseline_exact_20261028/features/sph_measurements.csv")
    vectors = np.load(input_root / "outputs/pr74_fresh_baseline_exact_20261028/features/sph_xecg.npy",
                      mmap_mode="r")
    measurement_gap, vector_gap, identical = 0.0, 0.0, 0
    for _, row in kept.iterrows():
        plan = nf.load_plan(work / "plans" / f"{row['ecg_id']}.npz")
        pooled = np.load(work / "pooled" / f"{row['ecg_id']}.npy")
        position = int(row["position"])
        measurement_gap = max(measurement_gap, float(np.nanmax(np.abs(
            plan["measurements"][0] - raw.iloc[position].to_numpy()))))
        vector_gap = max(vector_gap, float(np.max(np.abs(pooled[0].astype(float) - vectors[position]))))
        identical += int(np.array_equal(pooled[0], vectors[position]))
    seconds = [value for _, status, value in prepared if status == "prepared"]
    details = {"records": len(rows), "prepared": len(kept), "prepare_wall_seconds": prepare_wall,
               "prepare_cpu_seconds_per_record": float(np.mean(seconds)), "workers": workers,
               "encode": encoded,
               "encode_seconds_per_record": encoded["wall_seconds"] / max(encoded["records"], 1),
               "variants": len(nc.variants()), "original_measurement_max_abs_difference": measurement_gap,
               "original_xecg_max_abs_difference": vector_gap,
               "original_xecg_bit_identical_records": identical}
    write_json_atomic(stage_dir(input_root, "profile") / "profile.json", details)
    return complete(input_root, "profile", receipt, details)


def select(input_root: Path) -> dict[str, Any]:
    """Draw the analysis sample from labels only and write it with the group counts."""
    receipt = binding()
    rows = ns.evaluation_rows(input_root)
    sample = ns.draw_sample(rows, CAP, NORMALS)
    directory = stage_dir(input_root, "select")
    columns = ["ecg_id", "patient_key", "group", "position", "age", "male", "aha_codes", "target"]
    sample[columns].to_csv(directory / "selection.csv", index=False)
    details = {"eligible": ns.group_counts(rows), "sample": sample.groupby("group").size().to_dict(),
               "cap": CAP, "normals": NORMALS}
    write_json_atomic(directory / "selection.json", details)
    return complete(input_root, "select", receipt, details)


def selection_rows(input_root: Path) -> pd.DataFrame:
    """Full baseline rows of the selected ECGs, in draw order."""
    require(input_root, "select")
    chosen = pd.read_csv(input_root / OUTPUT / "select/selection.csv", dtype=str, keep_default_na=False)
    rows = ns.evaluation_rows(input_root).set_index("ecg_id").loc[chosen["ecg_id"]].reset_index()
    if not rows["group"].equals(chosen["group"]):
        raise ValueError("Selection groups changed")
    return rows


def prepare(input_root: Path, workers: int) -> dict[str, Any]:
    """Prepare plans, energies and measurements of every selected ECG."""
    receipt = binding()
    started = time.perf_counter()
    statuses = nf.prepare_all(input_root, selection_rows(input_root), model_of(input_root), WORK / "full",
                              workers)
    frame = pd.DataFrame(statuses, columns=["ecg_id", "status", "seconds"])
    frame.to_csv(stage_dir(input_root, "prepare") / "prepare.csv", index=False)
    rows = selection_rows(input_root).set_index("ecg_id").loc[frame["ecg_id"]]
    kept = np.array([str(np.load(WORK / "full/plans" / f"{ecg_id}.npz")["status"]) == "prepared"
                     for ecg_id in frame["ecg_id"]])
    territory = territory_patients(rows["group"].to_numpy()[kept], rows["patient_key"].to_numpy()[kept])
    primary = "territory" if min(territory.values()) >= MINIMUM_TERRITORY_PATIENTS else "wave"
    details = {"records": len(frame), "status": frame["status"].value_counts().to_dict(),
               "wall_seconds": time.perf_counter() - started, "workers": workers,
               "territory_patients": territory, "primary_endpoint": primary}
    return complete(input_root, "prepare", receipt, details)


def territory_patients(groups: np.ndarray, patients: np.ndarray) -> dict[str, int]:
    """Count analysed patients of the two territory groups SPH can label."""
    return {name: int(np.unique(patients[groups == name]).size) for name in ("inferior", "anterior")}


def prepared_rows(input_root: Path) -> pd.DataFrame:
    """Return the selected rows whose plan was prepared, in draw order."""
    require(input_root, "prepare")
    status = pd.read_csv(input_root / OUTPUT / "prepare/prepare.csv", dtype=str)
    rows = selection_rows(input_root)
    keep = rows["ecg_id"].isin(status.loc[status["status"].isin(["prepared", "done"]), "ecg_id"])
    keep &= [(WORK / "full/plans" / f"{ecg_id}.npz").exists() for ecg_id in rows["ecg_id"]]
    prepared = [str(np.load(WORK / "full/plans" / f"{ecg_id}.npz")["status"]) == "prepared"
                for ecg_id in rows["ecg_id"]]
    return rows.loc[keep & np.array(prepared)].reset_index(drop=True)


def encode(input_root: Path, device: int, shard: int, shards: int) -> dict[str, Any]:
    """Encode one shard of the prepared ECGs on one GPU."""
    receipt = binding()
    rows = prepared_rows(input_root).iloc[shard::shards]
    result = nf.encode_records(input_root, rows, WORK / "full", device)
    directory = stage_dir(input_root, f"encode_{shard}")
    write_json_atomic(directory / "encode.json", result)
    return complete(input_root, f"encode_{shard}", receipt, {**result, "shard": shard, "shards": shards})


def score(input_root: Path) -> dict[str, Any]:
    """Score all variants, then require the unreplaced scores to reproduce the saved SPH scores."""
    receipt = binding()
    rows = prepared_rows(input_root)
    scored = nf.score_records(input_root, rows["ecg_id"].tolist(), WORK / "full")
    saved = np.load(input_root / nf.PREDICTIONS)
    positions = rows["position"].astype(int).to_numpy()
    if not np.array_equal(saved["ecg_id"][positions], rows["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("Saved predictions and selected rows are misaligned")
    gaps = {f"{model}_{kind}": float(np.max(np.abs(scored["scores"][:, 0, m, k]
                                                   - saved[f"{model}_{kind}"][positions])))
            for m, model in enumerate(nf.MODELS) for k, kind in enumerate(nf.KINDS)}
    limits = nf.thresholds(input_root)
    calibration = ~saved["evaluation"].astype(bool)
    normals = saved["target"][calibration].astype(int)
    replayed = {key: budget_threshold(normals, saved[key][calibration])["threshold"] for key in limits}
    if replayed != limits:
        raise ValueError("Bundle thresholds differ from the saved calibration-normal replay")
    decisions = {key: bool(np.array_equal(scored["scores"][:, 0, m, 2] > limits[key],
                                          saved[key][positions] > limits[key]))
                 for m, key in enumerate(f"{model}_blend" for model in nf.MODELS)}
    directory = stage_dir(input_root, "score")
    write_npz_atomic(directory / "scores.npz", scores=scored["scores"], energy=scored["energy"],
                     ecg_id=rows["ecg_id"].to_numpy(dtype=str), group=rows["group"].to_numpy(dtype=str),
                     patient=rows["patient_key"].to_numpy(dtype=str))
    integrity = {"max_abs_difference": gaps, "tolerance": SCORE_TOLERANCE, "decisions_identical": decisions,
                 "thresholds_replayed_exactly": True,
                 "passed": max(gaps.values()) <= SCORE_TOLERANCE and all(decisions.values()),
                 "thresholds": limits, "records": len(rows)}
    write_json_atomic(directory / "integrity.json", integrity)
    if not integrity["passed"]:
        raise ValueError(f"Unreplaced scores do not reproduce the saved SPH scores: {gaps}")
    return complete(input_root, "score", receipt, integrity)


def analyze(input_root: Path) -> dict[str, Any]:
    """Compute every endpoint with whole-patient bootstrap intervals."""
    receipt = binding()
    require(input_root, "score")
    data = np.load(input_root / OUTPUT / "score/scores.npz")
    primary = require(input_root, "prepare")["primary_endpoint"]
    territory = territory_patients(data["group"], data["patient"])
    if territory != require(input_root, "prepare")["territory_patients"]:
        raise ValueError("Scored territory patients differ from the prepared ones")
    table = na.record_table(data["scores"], data["energy"], nf.thresholds(input_root))
    started = time.perf_counter()
    results = na.bootstrap(table, data["group"], data["patient"])
    directory = stage_dir(input_root, "analyze")
    counts = na.counts(data["group"], table)
    counts.to_csv(directory / "counts.csv")
    details = {"draws": na.DRAWS, "seed": na.SEED, "bootstrap_seconds": time.perf_counter() - started,
               "records": len(data["group"]), "territory_patients": territory, "primary_endpoint": primary}
    write_json_atomic(directory / "results.json", {"details": details, "statistics": results})
    return complete(input_root, "analyze", receipt, details)
