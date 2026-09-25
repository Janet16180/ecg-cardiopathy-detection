#!/usr/bin/env python3
"""Verified second-order-seed replication of the xECG encoder-motion test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Never

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state, flat_rng_state
from ecg_experiment.xecg_encoder_motion_replication_v10 import OUT, run_arm, v9
from ecg_experiment.xecg_encoder_motion_v9 import (
    ARMS,
    DEVELOPMENT_RECORDS,
    EFFECTIVE_BATCH,
    TRAIN_RECORDS,
    UPDATES,
    build_model,
    encoder_equal,
    encoder_state,
    infer_features,
    model_identity,
    train_update,
)
from ecg_experiment.xecg_encoder_motion_v10_analysis import (
    CEILING_SECONDS,
    H_SECONDS,
    P_SECONDS,
    cost_projection,
    paired_seed_bootstrap,
    run_arm_compatibility,
)
from ecg_experiment.xecg_rescue_analysis import development_metrics
from ecg_experiment.xecg_rescue_training import compare_replay_snapshots, cpu_nested, restore_checkpoint

ROOT = Path(__file__).resolve().parents[2]
V9_PROFILE = ROOT / "outputs/experiment_queue_016_encoder_motion_v9_profile"
V9_FULL = ROOT / "outputs/experiment_queue_016_encoder_motion_v9_full"
V9_OUT = ROOT / "outputs/experiment016_encoder_motion_v9"
V9_PROFILE_MANIFEST_SHA = "9d6a2b033db352c9974adfbbf77a1b2d3cada5c5bb4107c3f5095fefddb1a676"
V9_PROFILE_SOURCE_SHA = "9cb55fc92d1930fb3d10539a157213609c17190078900b27d96d068d6eff2a09"
V9_FULL_MANIFEST_SHA = "0b3b2456e94ea1a21a6bae24d720d893460aa78f1d2351406d2c47659396d45f"
V9_FULL_SOURCE_SHA = "f52010d13e4f3b03a06605a17d5ba12721e59e35df2f2e9d81d8fa64e310b405"
V9_REPORT_SHA = "938bc656167a01afa5cf38100a1f042a98ad9d2a843390fcb2f1f8c7f7eaf50b"
V10_SOURCES = (
    "docs/experiment-016-encoder-motion-replication-v10.md",
    "ecg_experiment/xecg_encoder_motion_replication_v10.py",
    "ecg_experiment/xecg_encoder_motion_v10_analysis.py",
    "scripts/experiments/run_xecg_encoder_motion_replication016_v10.py",
    "tests/test_xecg_encoder_motion_replication_v10.py",
)


def _verify_map(directory: Path, manifest_sha: str, source_sha: str) -> dict:
    """Rehash an immutable predecessor manifest, all sources and all outputs."""
    queue, source = directory / "queue.json", directory / "sources.json"
    if sha256_file(queue) != manifest_sha or sha256_file(source) != source_sha:
        raise ValueError(f"Frozen predecessor manifest/source map changed: {directory}")
    mapped = json.loads(source.read_text())
    changed = [name for name, digest in mapped.items() if sha256_file(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Frozen predecessor mapped bytes changed: {changed}")
    status = json.loads((directory / "status.json").read_text())
    completion = directory / "job/priority_queue_completion.json"
    receipt = json.loads(completion.read_text())
    if (
        status["state"] != "complete"
        or status["returncode"] != 0
        or receipt["queue_manifest_sha256"] != manifest_sha
    ):
        raise ValueError("Frozen predecessor did not complete under its manifest")
    changed = [name for name, digest in receipt["artifacts"].items() if sha256_file(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Frozen predecessor output bytes changed: {changed}")
    return {
        "manifest_sha256": manifest_sha,
        "sources_sha256": source_sha,
        "source_entries": len(mapped),
        "completion_sha256": sha256_file(completion),
        "completion_artifacts": len(receipt["artifacts"]),
    }


def inputs() -> dict:
    """Bind the exact V9 cohort, release, code, profile and seed-46 result."""
    old = {
        "profile": _verify_map(V9_PROFILE, V9_PROFILE_MANIFEST_SHA, V9_PROFILE_SOURCE_SHA),
        "production": _verify_map(V9_FULL, V9_FULL_MANIFEST_SHA, V9_FULL_SOURCE_SHA),
    }
    if sha256_file(V9_OUT / "report.json") != V9_REPORT_SHA:
        raise ValueError("Frozen V9 development report changed")
    profile = json.loads((V9_OUT / "profile.json").read_text())
    gate = json.loads((V9_OUT / "cost_gate.json").read_text())
    if (
        not gate["passed"]
        or gate["P_seconds"] != P_SECONDS
        or gate["measured_preparation_plus_both_profile_pipelines_seconds"] != H_SECONDS
        or not all(profile["correctness"].values())
    ):
        raise ValueError("V9 full-path profile evidence differs from frozen V10 cost premise")
    data = v9.inputs()
    data["fingerprint"] = {
        "v9": data["fingerprint"],
        "predecessors": old,
        "v9_report_sha256": V9_REPORT_SHA,
        "v10_sources": {name: sha256_file(ROOT / name) for name in V10_SOURCES},
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "torch_cuda": torch.version.cuda,
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
    }
    if data["fingerprint"]["environment"]["torch"] != "2.6.0+cu124":
        raise ValueError("V10 PyTorch differs from verified V9 environment")
    return data


def _driver_receipt() -> dict:
    """Record present V100 and CUDA settings against V9's device-class evidence."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if torch.cuda.get_device_count() != 1 or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V10 GPU device differs from V9's single V100")
    return {
        "nvidia_smi": query,
        "torch_device": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "historical_device_class": "V100 16 GB",
        "historical_torch": "2.6.0+cu124",
        "historical_driver_version": "not recorded in frozen V9 receipts",
    }


def check() -> dict:
    """Verify executable inheritance and fresh seed-47 start on real clean ECGs."""
    started = time.monotonic()
    data = inputs()
    compatibility = run_arm_compatibility()
    starts = {}
    _, _, diagnostic, released, _ = v9.datasets(data)
    four = torch.utils.data.Subset(diagnostic, list(range(4)))
    with np.load(v9.v6.V5 / "probe.npz") as probe:
        weight, bias = probe["raw_weight"], float(probe["raw_bias"])
    expected_order46 = torch.randperm(TRAIN_RECORDS, generator=torch.Generator().manual_seed(46))[:64]
    expected_order47 = torch.randperm(TRAIN_RECORDS, generator=torch.Generator().manual_seed(47))[:64]
    if torch.equal(expected_order46, expected_order47):
        raise RuntimeError("Fresh order unexpectedly equals inspected seed 46")
    reference = json.loads((V9_OUT / "production/M/complete.json").read_text())
    for arm in ARMS:
        model, optimizer, scheduler, permutation = build_model(
            v9.v5.RELEASE,
            v9.v6.V5 / "probe.npz",
            arm,
            47,
            "cpu",
        )
        start = {
            "model_sha256": model_identity(model),
            "optimizer_state_empty": optimizer.state_dict()["state"] == {},
            "group_names": [group["name"] for group in optimizer.param_groups],
            "scheduler_base_lrs": scheduler.base_lrs,
            "first64_sha256": hashlib.sha256(
                torch.randperm(
                    TRAIN_RECORDS,
                    generator=permutation,
                )[:64]
                .numpy()
                .tobytes()
            ).hexdigest(),
            "encoder_requires_grad": all(
                parameter.requires_grad for parameter in model.backbone.parameters()
            ),
        }
        features, logits = infer_features(model, four, "cpu")
        if not np.allclose(features, released[:4], atol=2e-4, rtol=2e-5):
            raise RuntimeError("V10 initial released feature path differs")
        if not np.allclose(logits, released[:4].astype(np.float64) @ weight + bias, atol=2e-4, rtol=2e-5):
            raise RuntimeError("V10 initial probe-head logits differ")
        starts[arm] = start
        del model, optimizer, scheduler, permutation
        gc.collect()
    if (
        starts["M"]["model_sha256"] != reference["initial_model_sha256"]
        or starts["M"]["model_sha256"] != starts["F"]["model_sha256"]
    ):
        raise RuntimeError("V10 released tensor initialization differs from V9")
    common_order = hashlib.sha256(expected_order47.numpy().tobytes()).hexdigest()
    if starts["M"]["first64_sha256"] != common_order or starts["F"]["first64_sha256"] != common_order:
        raise RuntimeError("V10 M/F seed-47 record order differs")
    if not all(row["optimizer_state_empty"] and row["encoder_requires_grad"] for row in starts.values()):
        raise RuntimeError("V10 full-gradient or fresh-moment initialization differs")
    if (
        any(rate != 0 for rate in starts["F"]["scheduler_base_lrs"][:-1])
        or starts["F"]["scheduler_base_lrs"][-1] != 0.001
    ):
        raise RuntimeError("V10 F encoder or head scheduler base rate differs")
    receipt = {
        "fingerprint": data["fingerprint"],
        "compatibility": compatibility,
        "starts": starts,
        "seed46_first64_sha256": hashlib.sha256(expected_order46.numpy().tobytes()).hexdigest(),
        "seed47_first64_sha256": common_order,
        "checked_real_training_ecgs": 4,
        "seconds": time.monotonic() - started,
        "status": "passed_no_training",
    }
    write_json_atomic(
        OUT / "compatibility.json",
        {
            "fingerprint": data["fingerprint"],
            "compatibility": compatibility,
            "predecessor_profile_complete_full_paths": True,
        },
    )
    write_json_atomic(OUT / "check.json", receipt)
    return receipt


def _bridge_snapshot(path: Path, source: dict, arm: str, data: Dataset, ecg_ids: np.ndarray) -> dict:
    """Restore an update-three bridge checkpoint and execute update four."""
    model, optimizer, scheduler, permutation = build_model(
        v9.v5.RELEASE,
        v9.v6.V5 / "probe.npz",
        arm,
        47,
        "cuda",
    )
    mask = model.backbone.core.drop_path
    saved = restore_checkpoint(path, source, model, mask, optimizer, scheduler, permutation)
    if saved["updates"] != 3 or saved["next_index"] != 3 * EFFECTIVE_BATCH:
        raise RuntimeError("Bridge checkpoint is not the three-update seed-47 state")
    group = saved["order"][3 * EFFECTIVE_BATCH : 4 * EFFECTIVE_BATCH]
    row = train_update(model, optimizer, scheduler, data, group, ecg_ids, "cuda", 3)
    snapshot = {
        "model": cpu_state(model),
        "optimizer": cpu_nested(optimizer.state_dict()),
        "scheduler": cpu_nested(scheduler.state_dict()),
        "mask_rng": mask.get_rng_state(),
        "permutation_rng": permutation.get_state(),
        "global_rng": cpu_nested(flat_rng_state()),
        "batch_ids_sha256": row["batch_ids_sha256"],
        "updates": 4,
    }
    del model, optimizer, scheduler, permutation, mask, saved
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() > 256 * 1024**2:
        raise RuntimeError("Bridge replay retained an active CUDA model")
    return snapshot


def _bridge_arm(data: dict, arm: str, train: Dataset, ecg_ids: np.ndarray) -> dict:
    """Run three full real seed-47 updates and exact checkpoint/replay checks."""
    directory = OUT / "bridge" / arm
    directory.mkdir(parents=True, exist_ok=True)
    source = {"fingerprint": data["fingerprint"], "arm": arm, "seed": 47, "mode": "bridge"}
    torch.cuda.reset_peak_memory_stats()
    model, optimizer, scheduler, permutation = build_model(
        v9.v5.RELEASE,
        v9.v6.V5 / "probe.npz",
        arm,
        47,
        "cuda",
    )
    initial = model_identity(model)
    baseline = encoder_state(model)
    order = torch.randperm(TRAIN_RECORDS, generator=permutation)
    rows = []
    for update in range(3):
        group = order[update * EFFECTIVE_BATCH : (update + 1) * EFFECTIVE_BATCH]
        row = train_update(model, optimizer, scheduler, train, group, ecg_ids, "cuda", update)
        rows.append(row)
        if arm == "F" and not encoder_equal(model, baseline):
            raise RuntimeError("Frozen bridge encoder changed")
        if update == 0 and row["head_update_norm"] != 0:
            raise RuntimeError("First zero-rate bridge update changed head")
        if update == 1 and (row["head_update_norm"] <= 0 or (arm == "M" and encoder_equal(model, baseline))):
            raise RuntimeError("First nonzero-rate bridge update failed")
    checkpoint = directory / "resume.pt"
    v9._verify_checkpoint(
        checkpoint, source, model, optimizer, scheduler, permutation, order, 3 * EFFECTIVE_BATCH, 3
    )
    write_json_atomic(directory / "updates.json", rows)
    changed = not encoder_equal(model, baseline)
    del model, optimizer, scheduler, permutation, baseline
    gc.collect()
    torch.cuda.empty_cache()
    first = _bridge_snapshot(checkpoint, source, arm, train, ecg_ids)
    second = _bridge_snapshot(checkpoint, source, arm, train, ecg_ids)
    replay = compare_replay_snapshots(first, second)
    if replay["atol"] != 1e-8 or replay["rtol"] != 1e-6 or not replay["exact_fields"]:
        raise RuntimeError("Bridge same-device replay criterion changed")
    receipt = {
        "source": source,
        "initial_model_sha256": initial,
        "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        "updates": 3,
        "record_exposures": 3 * EFFECTIVE_BATCH,
        "encoder_bitwise_changed": changed,
        "rows": rows,
        "replay": replay,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "sha256": {name: sha256_file(directory / name) for name in ("resume.pt", "updates.json")},
    }
    write_json_atomic(directory / "complete.json", receipt)
    return receipt


def bridge(data: dict, preparation_seconds: float) -> dict:
    """Bound both three-update V100 arms and prove compatibility without dev metrics."""
    checked = json.loads((OUT / "check.json").read_text())
    if (
        checked["fingerprint"] != data["fingerprint"]
        or not checked["compatibility"][
            "scientific_training_loop_ast_identical_after_identity_and_gate_guards"
        ]
    ):
        raise ValueError("Bridge differs from passed CPU/executable compatibility check")
    train, _, _, _, _ = v9.datasets(data)
    ecg_ids = np.asarray([int(row["ecg_id"]) for row in data["train"]], dtype=np.int64)
    device = _driver_receipt()
    started = time.monotonic()

    def timed_out(_signum: int, _frame: FrameType | None) -> Never:
        raise TimeoutError("V10 bridge exceeded 300 GPU-stage wall seconds")

    prior_handler = signal.signal(signal.SIGALRM, timed_out)
    receipts = {}
    try:
        with gpu_lock("cuda", blocking=False):
            signal.setitimer(signal.ITIMER_REAL, 300)
            for arm in ARMS:
                receipts[arm] = _bridge_arm(data, arm, train, ecg_ids)
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
    elapsed = time.monotonic() - started
    if elapsed > 300:
        raise TimeoutError("V10 bridge exceeded the frozen 300-second wall bound")
    v9.compare_profile_arms(
        {
            arm: {
                "initial_model_sha256": row["initial_model_sha256"],
                "initial_order_sha256": row["order_sha256"],
                "first_backward": row["rows"][0],
                "first_nonzero_update": row["rows"][1],
                "final_encoder_bitwise_unchanged": not row["encoder_bitwise_changed"],
            }
            for arm, row in receipts.items()
        }
    )
    result = {
        "fingerprint": data["fingerprint"],
        "device": device,
        "preparation_seconds": preparation_seconds,
        "gpu_stage_seconds": elapsed,
        "bounded_gpu_stage_seconds": 300,
        "arms": {arm: sha256_file(OUT / "bridge" / arm / "complete.json") for arm in ARMS},
        "no_development_metrics": True,
        "passed": True,
    }
    write_json_atomic(OUT / "bridge.json", result)
    return result


def train(data: dict, preparation_seconds: float) -> dict:
    """Run only a fresh seed-47 production pair after the inherited gate passes."""
    gate = json.loads((OUT / "cost_gate.json").read_text())
    if gate["fingerprint"] != data["fingerprint"] or not gate["passed"]:
        raise ValueError("V10 production lacks passing verified inherited profile gate")
    base_v = gate["new_preparation_V_seconds"] + preparation_seconds
    before = cost_projection(base_v, [])
    if not before["passed"]:
        raise RuntimeError("V10 two-hour gate failed after fresh production preparation")
    started = time.monotonic()

    def cost_guard() -> None:
        if H_SECONDS + base_v + time.monotonic() - started > CEILING_SECONDS:
            raise RuntimeError("V10 actual charged wall ceiling exhausted at checkpoint")

    receipts = {}
    with gpu_lock("cuda", blocking=False):
        for arm in ARMS:
            receipts[arm] = run_arm(data, arm, 47, "cuda", "production", cost_guard)
            completed = [row["complete_pipeline_seconds"] for row in receipts.values()]
            runtime = cost_projection(base_v, completed)
            runtime["fingerprint"] = data["fingerprint"]
            runtime["completed_arms"] = list(receipts)
            runtime["actual_charged_elapsed_seconds"] = H_SECONDS + base_v + time.monotonic() - started
            write_json_atomic(OUT / "runtime_gate.json", runtime)
            if not runtime["passed"] or runtime["actual_charged_elapsed_seconds"] > CEILING_SECONDS:
                raise RuntimeError(f"V10 runtime cost gate failed after {arm}")
            gc.collect()
            torch.cuda.empty_cache()
    if (
        receipts["M"]["initial_model_sha256"] != receipts["F"]["initial_model_sha256"]
        or receipts["M"]["initial_order_sha256"] != receipts["F"]["initial_order_sha256"]
    ):
        raise RuntimeError("V10 production M/F initial tensors or order differ")
    result = {
        "fingerprint": data["fingerprint"],
        "seed": 47,
        "arms": {arm: sha256_file(OUT / "production" / arm / "complete.json") for arm in ARMS},
    }
    write_json_atomic(OUT / "production.json", result)
    return result


def report(data: dict) -> dict:  # noqa: C901 - two-seed identity guards remain explicit
    """Analyze both fixed seeds with common patient draws without changing V9."""
    production = json.loads((OUT / "production.json").read_text())
    runtime = json.loads((OUT / "runtime_gate.json").read_text())
    if (
        production["fingerprint"] != data["fingerprint"]
        or not runtime["passed"]
        or set(production["arms"]) != set(ARMS)
    ):
        raise ValueError("Incomplete seed-47 production pair or failed cost gate")
    labels = np.asarray([int(row["target"]) for row in data["development"]])
    patients = np.asarray([row["patient_id"] for row in data["development"]])
    logits = {}
    for seed, root in (("46", V9_OUT), ("47", OUT)):
        logits[seed] = {}
        for arm in ARMS:
            directory = root / "production" / arm
            receipt = json.loads((directory / "complete.json").read_text())
            if (
                receipt["seed"] != int(seed)
                or receipt["updates"] != UPDATES
                or receipt["record_exposures"] != TRAIN_RECORDS
            ):
                raise ValueError(f"Incomplete {seed}/{arm} endpoint")
            if seed == "47" and receipt["source"] != {
                "fingerprint": data["fingerprint"],
                "arm": arm,
                "seed": 47,
                "mode": "production",
            }:
                raise ValueError("Seed-47 arm belongs to a different V10 source")
            for name, digest in receipt["sha256"].items():
                if sha256_file(directory / name) != digest:
                    raise ValueError(f"Changed {seed}/{arm} artifact: {name}")
            logits[seed][f"{arm}_joint"] = np.load(directory / "joint_logits.npy")
            logits[seed][f"{arm}_refit"] = np.load(directory / "refit_logits.npy")
    if any(len(value) != DEVELOPMENT_RECORDS for seed in logits.values() for value in seed.values()):
        raise ValueError("Development readout row count differs")
    bootstrap = paired_seed_bootstrap(labels, patients, logits)
    prior = json.loads((V9_OUT / "report.json").read_text())
    d46 = prior["bootstrap"]["contrasts"]["F_joint_minus_M_joint"]["observed"]
    if not np.isclose(d46, bootstrap["contrasts"]["46"]["D"]["observed"], atol=1e-12):
        raise RuntimeError("Recomputed seed-46 primary point differs from immutable V9")
    metrics = {
        seed: {name: development_metrics(labels, patients, value) for name, value in rows.items()}
        for seed, rows in logits.items()
    }
    released = np.load(V9_OUT / "../experiment016_droppath_rescue_v6/probe_logits.npy")
    metrics["released_probe"] = development_metrics(labels, patients, released)
    d47 = bootstrap["contrasts"]["47"]["D"]["observed"]
    result = {
        "fingerprint": data["fingerprint"],
        "scope": "two fixed order seeds on repeatedly inspected development patients",
        "train_records": TRAIN_RECORDS,
        "development_records": DEVELOPMENT_RECORDS,
        "development_patients": len(np.unique(patients)),
        "optimization_seeds": [46, 47],
        "updates_per_arm": UPDATES,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "v9_original_report_sha256": V9_REPORT_SHA,
        "v9_original_primary_interval_95": prior["bootstrap"]["contrasts"]["F_joint_minus_M_joint"][
            "interval_95"
        ],
        "seed47_primary_D": d47,
        "seed47_harm_screen_replicated": bootstrap["seed47_decision"]["joint_harm_replicated"],
        "seed47_readout_gap_pattern_replicated": bootstrap["seed47_decision"][
            "readout_gap_pattern_replicated"
        ],
        "seed47_primary_interval_excludes_zero": bootstrap["seed47_decision"][
            "primary_interval_excludes_zero"
        ],
        "two_seed_mean_D": bootstrap["contrasts"]["mean"]["D"]["observed"],
        "two_seed_D_range": [min(d46, d47), max(d46, d47)],
        "calibration_test_evaluated": False,
        "runtime_gate": runtime,
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 v10: encoder-motion seed-47 replication",
        "",
        "Development only; one released encoder and one patient cohort; calibration/test closed.",
        "",
        "| Seed | Readout | AUROC | AP | BCE | Fold sensitivity | Fold specificity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in ("released_probe", "46", "47"):
        for name, row in (
            metrics[seed].items() if seed != "released_probe" else [("released_probe", metrics[seed])]
        ):
            lines.append(
                f"| {seed} | {name} | {row['auroc']:.6f} | {row['average_precision']:.6f} | "
                f"{row['bce']:.6f} | {row['mean_fold_sensitivity']:.6f} | "
                f"{row['mean_fold_specificity']:.6f} |"
            )
    for name in ("D", "R", "G"):
        lines.append(
            f"{name}: seed46 {bootstrap['contrasts']['46'][name]['observed']:+.6f}, "
            f"seed47 {bootstrap['contrasts']['47'][name]['observed']:+.6f}, "
            f"mean {bootstrap['contrasts']['mean'][name]['observed']:+.6f}."
        )
    lines += [
        "",
        f"Seed47 harm screen: {result['seed47_harm_screen_replicated']}; "
        f"readout-gap screen: {result['seed47_readout_gap_pattern_replicated']}.",
        f"Common paired patient draws: {bootstrap['valid']}/{bootstrap['requested']} valid.",
        "The seed46 published V9 intervals remain unchanged; new common-draw intervals are in report.json.",
        "Patient resampling does not estimate seed-population uncertainty or provide independent validation.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Run one frozen stage and never auto-launch a successor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "bridge", "train", "report"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.stage == "check":
        result = check()
    else:
        data = inputs()
        preparation_seconds = time.monotonic() - started
        if args.stage == "bridge":
            result = bridge(data, preparation_seconds)
        elif args.stage == "train":
            result = train(data, preparation_seconds)
        else:
            result = report(data)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": "complete",
                "passed": result.get("passed"),
                "seed47_primary_D": result.get("seed47_primary_D"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
