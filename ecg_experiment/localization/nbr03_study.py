"""NBR-03 stages: checks, profile, select, prepare, encode, parity, score, analyze (PTB-XL fold 9)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..files import read_json, write_json_atomic, write_npz_atomic
from ..intervals import patient_groups, patient_resample
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_metrics import budget_threshold
from . import nbr01_analysis as na
from . import nbr01_counterpart as nc
from . import nbr01_features as nf
from . import nbr01_selection as ns
from . import nbr01_study as study01
from . import nbr03_ptbxl as ptb
from . import stages

OUTPUT = Path("outputs/nbr03_ptbxl_replication_20261028")
WORK = Path("/tmp/nbr03-work")
PROTOCOL = Path("docs/nbr03-ptbxl-replication-protocol-2026-10-28.md")
CAP = 300
NORMALS = 300
MINIMUM_TERRITORY_PATIENTS = 40
SCORE_TOLERANCE = 1e-12
NBR01_PRIMARY = 0.3056275517940945
SPH_REBUILD = 100
PROFILE_RECORDS = 20
TERRITORY_NAMES = (
    "territory",
    "territory_minus_energy",
    "swap_territory",
    "top1_hit",
    "top1_hit_minus_energy",
    "top3_hit",
    "top3_hit_minus_energy",
)
SCORES = tuple(f"{model}_{kind}" for model in nf.MODELS for kind in nf.KINDS)


def binding(require_committed: bool = True) -> dict[str, Any]:
    """Bind NBR-03's sources and protocol."""
    return stages.binding("nbr03", PROTOCOL, require_committed)


def complete(
    input_root: Path, stage: str, receipt: dict[str, Any], details: dict[str, Any]
) -> dict[str, Any]:
    """Complete one NBR-03 stage."""
    return stages.complete(input_root, OUTPUT, stage, receipt, details)


def require(input_root: Path, stage: str) -> dict[str, Any]:
    """Verify one completed NBR-03 stage."""
    return stages.require(input_root, OUTPUT, stage)


def p0_model(input_root: Path) -> Any:
    """Load NBR-01's verified P0 (rank-0) model."""
    study01.require(input_root, "counterpart")
    return nf.load_model(input_root / study01.OUTPUT / "counterpart/rank0.npz")


def eligible_rows(input_root: Path) -> pd.DataFrame:
    """Return fold-9 rows with a group; ``checks`` proves that no fold-9 patient needs excluding."""
    rows = ptb.fold_rows(input_root)
    return rows.loc[rows["group"].notna()].reset_index(drop=True)


def checks(input_root: Path, workers: int) -> dict[str, Any]:
    """Run the data rules, count groups from labels and rebuild 100 SPH P0 counterparts."""
    receipt = binding(require_committed=False)
    rows = ptb.fold_rows(input_root)
    rules = ptb.data_rules(input_root, rows)
    if not rules["passed"]:
        raise ValueError(f"PTB-XL fold-9 data rules failed: {rules}")
    counts = {
        group: {
            "ecgs": int(rows["group"].eq(group).sum()),
            "patients": int(rows.loc[rows["group"].eq(group), "patient_key"].nunique()),
        }
        for group in ns.GROUPS
    }
    model = p0_model(input_root)
    sph = study01.prepared_rows(input_root).head(SPH_REBUILD)
    worst = 0.0
    for _, row in sph.iterrows():
        plan = nc.record_plan(read_signal(input_root, row), *nf.covariates(row), model)
        saved = nf.load_plan(study01.WORK / "full/plans" / f"{row['ecg_id']}.npz")
        worst = max(worst, float(np.max(np.abs(plan["difference"].astype(float) - saved["difference"]))))
    details = {
        "data_rules": rules,
        "eligible": counts,
        "sph_p0_rebuilt": len(sph),
        "sph_p0_max_abs_difference_mv": worst,
    }
    write_json_atomic(stages.stage_dir(input_root, OUTPUT, "checks") / "checks.json", details)
    if worst != 0.0:
        raise ValueError(f"Rebuilt SPH P0 counterparts differ from NBR-01 by {worst} mV")
    return complete(input_root, "checks", receipt, details)


def profile(input_root: Path, device: int, workers: int) -> dict[str, Any]:
    """Time preparation and encoding on 20 fold-9 ECGs, without scoring."""
    receipt = binding(require_committed=False)
    require(input_root, "checks")
    rows = ns.draw_sample(eligible_rows(input_root), 3, 2, seed=20261029).head(PROFILE_RECORDS)
    work = WORK / "profile"
    started = time.perf_counter()
    statuses = ptb.prepare_all(input_root, rows, p0_model(input_root), work, workers)
    prepare_wall = time.perf_counter() - started
    kept = rows.loc[[status == "prepared" for _, status, _ in statuses]]
    encoded = nf.encode_records(input_root, kept, work, device)
    details = {
        "records": len(rows),
        "prepared": len(kept),
        "prepare_wall_seconds": prepare_wall,
        "prepare_cpu_seconds_per_record": float(np.mean([value for _, _, value in statuses])),
        "encode": encoded,
        "encode_seconds_per_record": encoded["wall_seconds"] / max(encoded["records"], 1),
    }
    write_json_atomic(stages.stage_dir(input_root, OUTPUT, "profile") / "profile.json", details)
    return complete(input_root, "profile", receipt, details)


def select(input_root: Path) -> dict[str, Any]:
    """Draw one ECG per patient, NBR-01's caps and seed, from labels only."""
    receipt = binding()
    require(input_root, "checks")
    sample = ns.draw_sample(eligible_rows(input_root), CAP, NORMALS)
    directory = stages.stage_dir(input_root, OUTPUT, "select")
    sample[["ecg_id", "patient_key", "group", "position", "strat_fold", "target"]].to_csv(
        directory / "selection.csv", index=False
    )
    details = {"sample": sample.groupby("group").size().to_dict(), "cap": CAP, "normals": NORMALS}
    write_json_atomic(directory / "selection.json", details)
    return complete(input_root, "select", receipt, details)


def selection_rows(input_root: Path) -> pd.DataFrame:
    """Return the full rows of the selected ECGs in draw order."""
    require(input_root, "select")
    chosen = pd.read_csv(input_root / OUTPUT / "select/selection.csv", dtype=str, keep_default_na=False)
    rows = eligible_rows(input_root).set_index("ecg_id").loc[chosen["ecg_id"]].reset_index()
    if not rows["group"].equals(chosen["group"]) or not rows["strat_fold"].eq(ptb.FOLD).all():
        raise ValueError("Selection changed or left fold 9")
    return rows


def prepare(input_root: Path, workers: int) -> dict[str, Any]:
    """Prepare every selected ECG and fix the primary territories from the analysed patient counts."""
    receipt = binding()
    rows = selection_rows(input_root)
    started = time.perf_counter()
    statuses = ptb.prepare_all(input_root, rows, p0_model(input_root), WORK / "full", workers)
    frame = pd.DataFrame(statuses, columns=["ecg_id", "result", "seconds"])
    frame.to_csv(stages.stage_dir(input_root, OUTPUT, "prepare") / "prepare.csv", index=False)
    kept = rows.loc[[ptb.prepared(WORK / "full/plans" / f"{e}.npz") for e in rows["ecg_id"]]]
    territory = {
        name: int(kept.loc[kept["group"].eq(name), "patient_key"].nunique())
        for name in ("inferior", "anterior", "lateral")
    }
    primary = [name for name, count in territory.items() if count >= MINIMUM_TERRITORY_PATIENTS]
    details = {
        "records": len(frame),
        "prepared": len(kept),
        "excluded": int(len(rows) - len(kept)),
        "wall_seconds": time.perf_counter() - started,
        "territory_patients": territory,
        "primary_territories": primary,
    }
    if len(primary) < 2:
        raise ValueError("Fewer than two territories have 40 patients; the primary is undefined")
    return complete(input_root, "prepare", receipt, details)


def prepared_rows(input_root: Path) -> pd.DataFrame:
    """Return the selected rows whose plan was prepared, in draw order."""
    require(input_root, "prepare")
    rows = selection_rows(input_root)
    keep = np.array([ptb.prepared(WORK / "full/plans" / f"{e}.npz") for e in rows["ecg_id"]])
    return rows.loc[keep].reset_index(drop=True)


def encode(input_root: Path, device: int, shard: int, shards: int) -> dict[str, Any]:
    """Encode one shard of the prepared ECGs on one GPU."""
    receipt = binding()
    rows = prepared_rows(input_root).iloc[shard::shards]
    result = nf.encode_records(input_root, rows, WORK / "full", device)
    write_json_atomic(stages.stage_dir(input_root, OUTPUT, f"encode_{shard}") / "encode.json", result)
    return complete(input_root, f"encode_{shard}", receipt, {**result, "shard": shard, "shards": shards})


def parity(input_root: Path, device: int) -> dict[str, Any]:
    """Re-encode records with non-identical unreplaced features in their historical batch (pre-registered)."""
    receipt = binding()
    result = ptb.historical_parity(input_root, prepared_rows(input_root), WORK / "full", device)
    write_json_atomic(stages.stage_dir(input_root, OUTPUT, "parity") / "parity.json", result)
    return complete(input_root, "parity", receipt, result)


def score(input_root: Path) -> dict[str, Any]:
    """Score all variants and require the unreplaced scores to reproduce the saved fold-9 scores."""
    receipt = binding()
    checked = require(input_root, "parity")
    rows = prepared_rows(input_root)
    scored = nf.score_records(input_root, rows["ecg_id"].tolist(), WORK / "full")
    saved = np.load(input_root / ptb.PREDICTIONS)
    index = {str(value): position for position, value in enumerate(saved["ecg_id"])}
    positions = np.array([index[ecg_id] for ecg_id in rows["ecg_id"]])
    if not saved["evaluation"][positions].astype(bool).all():
        raise ValueError("A selected ECG is not a saved fold-9 evaluation row")
    limits = ptb.thresholds(input_root)
    calibration = ~saved["evaluation"].astype(bool)
    replayed = {
        key: budget_threshold(saved["target"][calibration].astype(int), saved[key][calibration])["threshold"]
        for key in limits
    }
    worst = np.zeros(len(rows))
    gaps = {}
    for m, model in enumerate(nf.MODELS):
        for k, kind in enumerate(nf.KINDS):
            difference = np.abs(scored["scores"][:, 0, m, k] - saved[f"{model}_{kind}"][positions])
            gaps[f"{model}_{kind}"] = float(difference.max())
            worst = np.maximum(worst, difference)
    decisions = {
        f"{model}_blend": bool(
            np.array_equal(
                scored["scores"][:, 0, m, 2] > limits[f"{model}_blend"],
                saved[f"{model}_blend"][positions] > limits[f"{model}_blend"],
            )
        )
        for m, model in enumerate(nf.MODELS)
    }
    over = sorted(rows.loc[worst > SCORE_TOLERANCE, "ecg_id"])
    explained = {item["ecg_id"] for item in checked["differing"] if item["historical_batch_bit_identical"]}
    registered = not over and all(decisions.values())
    integrity = {
        "max_abs_difference": gaps,
        "tolerance": SCORE_TOLERANCE,
        "decisions_identical": decisions,
        "thresholds_replayed_exactly": replayed == limits,
        "thresholds": limits,
        "registered_check_passed": registered,
        "records_over_tolerance": over,
        "amendment_1": None
        if registered
        else {
            "passed": bool(checked["passed"]) and set(over) <= explained,
            "parity_records": sorted(explained),
        },
        "records": len(rows),
    }
    integrity["passed"] = (
        integrity["thresholds_replayed_exactly"]
        and all(decisions.values())
        and (registered or integrity["amendment_1"]["passed"])
    )
    directory = stages.stage_dir(input_root, OUTPUT, "score")
    write_npz_atomic(
        directory / "scores.npz",
        scores=scored["scores"],
        energy=scored["energy"],
        ecg_id=rows["ecg_id"].to_numpy(dtype=str),
        group=rows["group"].to_numpy(dtype=str),
        patient=rows["patient_key"].to_numpy(dtype=str),
    )
    write_json_atomic(directory / "integrity.json", integrity)
    if not integrity["passed"]:
        raise ValueError(f"Unreplaced fold-9 scores do not reproduce: {gaps}")
    return complete(input_root, "score", receipt, integrity)


def primary_groups(groups: np.ndarray, primary: list[str]) -> np.ndarray:
    """Rename territories below 40 patients so that they leave the primary contrast."""
    renamed = groups.astype(object).copy()
    for name in ("inferior", "anterior", "lateral"):
        if name not in primary:
            renamed[renamed == name] = f"{name}_secondary"
    return renamed.astype(str)


def territory_parts(lead_shares: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Each present territory's expected-lead share contrast against the other present territories."""
    present = [name for name in na.TERRITORIES if (groups == name).any()]
    return {
        name: na.group_contrast(
            lead_shares[:, na.lead_mask(ns.TERRITORY_LEADS[name])].sum(axis=1),
            groups,
            (name,),
            tuple(other for other in present if other != name),
        )
        for name in present
        if len(present) > 1
    }


def lateral_statistics(table: dict[str, Any], groups: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    """Compute the lateral group's flagged share, changes and flips, which NBR-01 does not cover."""
    mask = groups[rows] == "lateral"
    if not mask.any():
        return {}
    result = {}
    for key in SCORES:
        entry = table[key]
        flagged = entry["flagged"][rows][mask]
        result[f"{key}/lateral/flagged"] = float(flagged.mean())
        result[f"{key}/lateral/full_drop"] = float(entry["full_drop"][rows][mask].mean())
        result[f"{key}/lateral/lead_abs"] = float(entry["lead_abs"][rows][mask].mean())
        for family in ("grid", "lead", "component", "full"):
            flips = entry[f"{family}_flip"][rows][mask]
            result[f"{key}/lateral/{family}_flip"] = (
                float(flips[flagged].mean()) if flagged.any() else float("nan")
            )
    return result


def statistics(
    table: dict[str, Any], groups: np.ndarray, primary: np.ndarray, rows: np.ndarray
) -> dict[str, float]:
    """
    NBR-01 statistics with the primary territories, the all-territory variants and per-territory parts.

    Parameters
    ----------
    table : dict[str, Any]
        ``nbr01_analysis.record_table`` output.
    groups, primary : np.ndarray
        All groups, and the groups with non-primary territories renamed.
    rows : np.ndarray
        Record positions of the sample or a bootstrap draw.

    Returns
    -------
    dict[str, float]
        Unprefixed primary statistics, ``all_territories/`` variants, ``parts/`` and lateral statistics.
    """
    result = na.statistics(table, primary, rows)
    every = na.statistics(table, groups, rows)
    for key, value in every.items():
        if key.split("/")[-1] in TERRITORY_NAMES:
            result[f"all_territories/{key}"] = value
    maps = {"energy": table["energy_lead"], **{key: table[key]["lead_shares"] for key in SCORES}}
    for name, shares in maps.items():
        for part, value in territory_parts(shares[rows], groups[rows]).items():
            result[f"parts/{name}/{part}"] = value
    result.update(lateral_statistics(table, groups, rows))
    for key in SCORES:
        result[f"{key}/selectivity"] = result[f"{key}/abnormal/lead_abs"] / result[f"{key}/normal/lead_abs"]
    return result


def bootstrap(
    table: dict[str, Any], groups: np.ndarray, primary: np.ndarray, patients: np.ndarray
) -> dict[str, dict[str, float]]:
    """Point values and 95% whole-patient bootstrap intervals, every statistic on the same draws."""
    point = statistics(table, groups, primary, np.arange(len(groups)))
    rng = np.random.default_rng(na.SEED)
    resample = patient_groups(np.asarray(patients))
    values: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(na.DRAWS):
        drawn = statistics(table, groups, primary, patient_resample(resample, rng))
        for key in point:
            value = drawn.get(key, float("nan"))
            if np.isfinite(value):
                values[key].append(value)
    result = {}
    for key, value in point.items():
        low, high = np.percentile(values[key], [2.5, 97.5]) if values[key] else (np.nan, np.nan)
        result[key] = {
            "point": value,
            "low": float(low),
            "high": float(high),
            "valid_draws": len(values[key]),
        }
    return result


def analyze(input_root: Path) -> dict[str, Any]:
    """Reproduce NBR-01's SPH primary with this code, then compute every fold-9 endpoint."""
    receipt = binding()
    require(input_root, "score")
    sph = dict(np.load(input_root / study01.OUTPUT / "score/scores.npz"))
    sph_table = na.record_table(sph["scores"], sph["energy"], nf.thresholds(input_root))
    sph_groups = sph["group"]
    reproduced = statistics(sph_table, sph_groups, sph_groups, np.arange(len(sph_groups)))[
        "all_age_blend/territory"
    ]
    if reproduced != NBR01_PRIMARY:
        raise ValueError(f"NBR-01 primary does not reproduce: {reproduced}")
    data = dict(np.load(input_root / OUTPUT / "score/scores.npz"))
    primary_names = require(input_root, "prepare")["primary_territories"]
    table = na.record_table(data["scores"], data["energy"], ptb.thresholds(input_root))
    primary = primary_groups(data["group"], primary_names)
    started = time.perf_counter()
    results = bootstrap(table, data["group"], primary, data["patient"])
    statistic = results["all_age_blend/territory"]
    qualifier = results["all_age_blend/territory_minus_energy"]
    decision = {
        "primary_territories": primary_names,
        "territory": statistic,
        "territory_minus_energy": qualifier,
        "localization_replicates": statistic["low"] > 0,
        "beyond_energy": qualifier["low"] > 0,
    }
    directory = stages.stage_dir(input_root, OUTPUT, "analyze")
    counts = na.counts(data["group"], table)
    counts.to_csv(directory / "counts.csv")
    write_json_atomic(
        directory / "results.json",
        {"statistics": results, "decision": decision, "nbr01_primary_reproduced": reproduced},
    )
    details = {
        "draws": na.DRAWS,
        "seed": na.SEED,
        "bootstrap_seconds": time.perf_counter() - started,
        "records": len(data["group"]),
        "nbr01_primary_reproduced": reproduced,
        "decision": decision,
    }
    return complete(input_root, "analyze", receipt, details)


def load_results(input_root: Path) -> dict[str, Any]:
    """Read the analyze stage's results after verifying it."""
    require(input_root, "analyze")
    return read_json(input_root / OUTPUT / "analyze/results.json")
