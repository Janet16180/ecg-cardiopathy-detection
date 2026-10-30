"""NBR-02 stages: fitsets, profile, prepare, encode, score, analyze and export."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import ROOT
from .. import normative_map062 as nm
from ..files import read_json, sha256_file, write_json_atomic, write_npz_atomic
from ..paths import to_stored
from ..pr74_screening_io import verify_hashes
from . import nbr01_features as nf
from . import nbr01_study as study01
from . import nbr02_analysis as na2
from . import nbr02_pipeline as pipeline
from . import nbr02_templates as templates

OUTPUT = Path("outputs/nbr02_personal_counterpart_20261028")
WORK = Path("/tmp/nbr02-work")
PROTOCOL = Path("docs/nbr02-personal-counterpart-protocol-2026-10-28.md")
NBR01_SCORES = study01.OUTPUT / "score/scores.npz"
NBR01_RESULTS = study01.OUTPUT / "analyze/results.json"
NBR01_PRIMARY = 0.3056275517940945
PROFILE_RECORDS = 20
FROZEN = ("normative_map062", "replacement070", "combination066", "pr74_fresh_baseline",
          "pr74_screening_features", "pr74_screening_io", "external_encoders", "xecg",
          "measurement_extractor042", "cpu_screening041", "full_development", "intervals", "gpu", "files",
          "paths", "heads01_features", "locbench_cache", "locbench_data")


def binding(require_committed: bool = True) -> dict[str, Any]:
    """
    Hash NBR-02's sources, the NBR-01 code it reuses and the frozen pipeline.

    Parameters
    ----------
    require_committed : bool
        Require NBR-02's own sources and protocol to equal ``HEAD``.

    Returns
    -------
    dict[str, Any]
        Source hashes, protocol hash and git revision.

    Raises
    ------
    ValueError
        If an own source or the protocol differs from ``HEAD``.
    """
    own = sorted((ROOT / "ecg_experiment/localization").glob("nbr02_*.py"))
    own += [ROOT / "scripts/experiments/run_nbr02.py"]
    own += sorted((ROOT / "tests/localization").glob("test_nbr02*.py"))
    if require_committed:
        for path in [*own, ROOT / PROTOCOL]:
            blob = subprocess.check_output(["git", "show", "HEAD:" + to_stored(path)], cwd=ROOT)
            if blob != path.read_bytes():
                raise ValueError(f"Source must be committed before execution: {path.name}")
    sources = own + sorted((ROOT / "ecg_experiment/localization").glob("nbr01_*.py"))
    sources += [ROOT / f"ecg_experiment/{name}.py" for name in FROZEN]
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
    value = {**receipt, "details": details, "status": "completed", "stage": stage, "artifacts": artifacts}
    write_json_atomic(directory / "completion.json", value)
    return value


def require(input_root: Path, stage: str) -> dict[str, Any]:
    """Verify a completed stage's artifacts and return its details."""
    receipt = read_json(input_root / OUTPUT / stage / "completion.json")
    if receipt.get("status") != "completed":
        raise ValueError(f"Stage {stage} is not complete")
    verify_hashes(input_root, receipt["artifacts"])
    return receipt["details"]


def fitsets(input_root: Path, workers: int) -> dict[str, Any]:
    """Check the cohort fit set, then fit the rank-0 and rank-6 models of both fit sets."""
    receipt = binding(require_committed=False)
    started = time.perf_counter()
    directory = stage_dir(input_root, "fitsets")
    rows = templates.cohort_rows(input_root)
    checks = templates.cohort_checks(input_root, rows)
    write_json_atomic(directory / "cohort_checks.json", checks)
    if not checks["passed"]:
        raise ValueError("A cohort normal overlaps closed or SPH data")
    sets = {"P": templates.locbench_fit_set(workers),
            "C": templates.cohort_fit_set(input_root, rows, workers)}
    models = {}
    for name, fit in sets.items():
        for rank, model in templates.fit_models(fit).items():
            models[f"{name}{rank}"] = model
            nf.save_model(directory / f"{name}{rank}.npz", model)
    saved = nf.load_model(input_root / study01.OUTPUT / "counterpart/rank0.npz")
    identical = {field: bool(np.array_equal(getattr(saved, field), getattr(models["P0"], field)))
                 for field in ("coefficients", "scale", "basis", "eigenvalues", "noise", "off_scale")}
    details = {"cohort_checks": checks, "fit_sets": templates.fit_summary(started, **sets),
               "p0_equals_nbr01_rank0": identical,
               "eigenvalues": {arm: models[arm].eigenvalues.tolist() for arm in ("P6", "C6")}}
    write_json_atomic(directory / "fitsets.json", details)
    if not all(identical.values()):
        raise ValueError("Refitted P0 differs from NBR-01's rank-0 model")
    return complete(input_root, "fitsets", receipt, details)


def models_of(input_root: Path) -> dict[str, nm.PopulationModel]:
    """Load the four arm models of the completed fitsets stage."""
    require(input_root, "fitsets")
    return {arm: nf.load_model(input_root / OUTPUT / f"fitsets/{arm}.npz") for arm in templates.ARMS}


def sample_rows(input_root: Path) -> pd.DataFrame:
    """NBR-01's 1,037 analysed rows, required to match its saved scores in order."""
    study01.require(input_root, "score")
    rows = study01.prepared_rows(input_root)
    saved = np.load(input_root / NBR01_SCORES)
    if not np.array_equal(saved["ecg_id"], rows["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("NBR-01 rows and saved scores are misaligned")
    return rows


def profile(input_root: Path, device: int, workers: int) -> dict[str, Any]:
    """Time preparation and encoding of the first 20 analysed ECGs, without scoring."""
    receipt = binding(require_committed=False)
    rows = sample_rows(input_root).head(PROFILE_RECORDS)
    work = WORK / "profile"
    started = time.perf_counter()
    timings = pipeline.prepare_all(input_root, rows, models_of(input_root), work, workers)
    prepare_wall = time.perf_counter() - started
    encoded = pipeline.encode_records(input_root, rows, work, device)
    gaps = [float(np.load(work / "plans" / f"{ecg_id}.npz")["p0_max_abs_difference_mv"])
            for ecg_id in rows["ecg_id"]]
    details = {"records": len(rows), "prepare_wall_seconds": prepare_wall, "workers": workers,
               "prepare_cpu_seconds_per_record": float(np.mean([value for _, value in timings])),
               "encode": encoded,
               "encode_seconds_per_record": encoded["wall_seconds"] / max(encoded["records"], 1),
               "variants_per_record": 3 * len(pipeline.counterpart_variants()),
               "p0_max_abs_difference_mv": max(gaps)}
    write_json_atomic(stage_dir(input_root, "profile") / "profile.json", details)
    return complete(input_root, "profile", receipt, details)


def prepare(input_root: Path, workers: int) -> dict[str, Any]:
    """Prepare all analysed ECGs and require the rebuilt P0 counterparts to equal NBR-01's."""
    receipt = binding()
    rows = sample_rows(input_root)
    started = time.perf_counter()
    timings = pipeline.prepare_all(input_root, rows, models_of(input_root), WORK / "full", workers)
    frame = pd.DataFrame(timings, columns=["ecg_id", "seconds"])
    frame["p0_max_abs_difference_mv"] = [
        float(np.load(WORK / "full/plans" / f"{ecg_id}.npz")["p0_max_abs_difference_mv"])
        for ecg_id in frame.ecg_id]
    frame.to_csv(stage_dir(input_root, "prepare") / "prepare.csv", index=False)
    worst = float(frame["p0_max_abs_difference_mv"].max())
    details = {"records": len(frame), "wall_seconds": time.perf_counter() - started, "workers": workers,
               "p0_records_checked": len(frame), "p0_max_abs_difference_mv": worst}
    if worst != 0.0:
        raise ValueError(f"Rebuilt P0 counterparts differ from NBR-01 by {worst} mV")
    return complete(input_root, "prepare", receipt, details)


def encode(input_root: Path, device: int, shard: int, shards: int) -> dict[str, Any]:
    """Encode one shard of the analysed ECGs on one GPU."""
    receipt = binding()
    require(input_root, "prepare")
    rows = sample_rows(input_root).iloc[shard::shards]
    result = pipeline.encode_records(input_root, rows, WORK / "full", device)
    write_json_atomic(stage_dir(input_root, f"encode_{shard}") / "encode.json", result)
    return complete(input_root, f"encode_{shard}", receipt, {**result, "shard": shard, "shards": shards})


def score(input_root: Path) -> dict[str, Any]:
    """Score the three new arms with both frozen models."""
    receipt = binding()
    require(input_root, "prepare")
    rows = sample_rows(input_root)
    scored = pipeline.score_records(input_root, rows["ecg_id"].tolist(), WORK / "full")
    write_npz_atomic(stage_dir(input_root, "score") / "scores.npz",
                     ecg_id=rows["ecg_id"].to_numpy(dtype=str), **scored)
    return complete(input_root, "score", receipt, {"records": len(rows), "arms": list(templates.NEW_ARMS)})


def analyze(input_root: Path) -> dict[str, Any]:
    """Reproduce NBR-01 from its saved scores, then compute every arm statistic and the decision."""
    receipt = binding()
    require(input_root, "score")
    base = dict(np.load(input_root / NBR01_SCORES))
    new = dict(np.load(input_root / OUTPUT / "score/scores.npz"))
    if not np.array_equal(base["ecg_id"], new["ecg_id"]):
        raise ValueError("NBR-01 and NBR-02 scores are misaligned")
    tables = na2.arm_tables(base, new, nf.thresholds(input_root))
    groups, patients = base["group"], base["patient"]
    diagnostics = {key: value for key, value in new.items()
                   if key.startswith("template_") or key.endswith("_clipped")}
    point = na2.arm_statistics(tables, groups, diagnostics, np.arange(len(groups)))
    saved = read_json(input_root / NBR01_RESULTS)["statistics"]
    mismatched = [key for key, value in saved.items()
                  if not (point[f"P0/{key}"] == value["point"]
                          or (np.isnan(point[f"P0/{key}"]) and np.isnan(value["point"])))]
    reproduction = {"primary": point["P0/all_age_blend/territory"], "expected": NBR01_PRIMARY,
                    "primary_exact": point["P0/all_age_blend/territory"] == NBR01_PRIMARY,
                    "nbr01_statistics_checked": len(saved), "nbr01_statistics_mismatched": mismatched}
    if not reproduction["primary_exact"] or mismatched:
        raise ValueError(f"NBR-01 statistics do not reproduce: {reproduction}")
    started = time.perf_counter()
    statistics = na2.bootstrap(tables, groups, patients, diagnostics)
    decided = na2.decision(statistics)
    directory = stage_dir(input_root, "analyze")
    write_json_atomic(directory / "results.json", {"statistics": statistics, "decision": decided,
                                                   "reproduction": reproduction})
    details = {"draws": na2.na.DRAWS, "seed": na2.na.SEED, "bootstrap_seconds": time.perf_counter() - started,
               "records": len(groups), "reproduction": reproduction, "decision": decided}
    return complete(input_root, "analyze", receipt, details)
