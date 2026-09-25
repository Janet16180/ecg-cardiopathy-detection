#!/usr/bin/env python3
"""Verified second-order-seed replication of the xECG encoder-motion test."""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import gc
import hashlib
import inspect
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
from ecg_experiment.xecg_encoder_motion_replication_v14 import OUT, run_arm, v9
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
    H_SECONDS,
    paired_seed_bootstrap,
    run_arm_compatibility,
)
from ecg_experiment.xecg_encoder_motion_v14_cost import (
    CEILING,
    P_HISTORICAL,
    REPORT_RESERVE,
    STOP_RESERVE,
    projection,
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
V14_SOURCES = (
    "docs/experiment-016-encoder-motion-replication-v14.md",
    "ecg_experiment/xecg_encoder_motion_replication_v14.py",
    "ecg_experiment/xecg_encoder_motion_v14_cost.py",
    "scripts/experiments/run_xecg_encoder_motion_replication016_v14.py",
    "tests/test_xecg_encoder_motion_replication_v14.py",
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
        or gate["P_seconds"] != P_HISTORICAL
        or gate["measured_preparation_plus_both_profile_pipelines_seconds"] != H_SECONDS
        or not all(profile["correctness"].values())
    ):
        raise ValueError("V9 full-path profile evidence differs from frozen V14 cost premise")
    data = v9.inputs()
    data["fingerprint"] = {
        "v9": data["fingerprint"],
        "predecessors": old,
        "v9_report_sha256": V9_REPORT_SHA,
        "v14_sources": {name: sha256_file(ROOT / name) for name in V14_SOURCES},
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "torch_cuda": torch.version.cuda,
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
    }
    if data["fingerprint"]["environment"]["torch"] != "2.6.0+cu124":
        raise ValueError("V14 PyTorch differs from verified V9 environment")
    return data


def _driver_receipt() -> dict:
    """Record present V100 and CUDA settings against V9's device-class evidence."""
    _cheap_api_check(torch.cuda)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if torch.cuda.device_count() != 1 or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V14 GPU device differs from V9's single V100")
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


def _cheap_api_check(cuda: object) -> int:
    """Exercise installed CUDA API names before any large input rehash."""
    names = (
        "device_count", "get_device_name", "reset_peak_memory_stats",
        "max_memory_allocated", "max_memory_reserved", "empty_cache", "memory_allocated",
    )
    if any(not callable(getattr(cuda, name, None)) for name in names):
        raise RuntimeError("Required installed CUDA API is missing")
    count = cuda.device_count()
    if not isinstance(count, int) or count < 0:
        raise RuntimeError("Malformed CUDA device count")
    return count


def _compatibility_map() -> dict:
    """Authenticate unchanged V9/V10 scientific callables and show V14 runner changes."""
    prior = run_arm_compatibility()
    inherited = {
        "datasets": v9.datasets,
        "checkpoint_cpu_roundtrip": v9._verify_checkpoint,
        "sequential_final_replay": v9._replay,
        "fixed_final_probe": v9._fit_final_probe,
        "released_model": build_model,
        "full_gradient_train_update": train_update,
        "fixed_training_diagnostic": v9.fixed_diagnostic,
        "full_feature_extraction": infer_features,
    }
    old = inspect.getsource(v9.run_arm)
    from ecg_experiment.xecg_encoder_motion_replication_v10 import run_arm as v10_run_arm

    middle = inspect.getsource(v10_run_arm)
    new = inspect.getsource(run_arm)

    def scientific_loop(source: str) -> str:
        tree = ast.parse(source)
        loops = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
            and node.target.id == "start"
        ]
        if len(loops) != 1:
            raise RuntimeError("Expected exactly one full-data training loop")
        loop = copy.deepcopy(loops[0])
        for node in ast.walk(loop):
            if hasattr(node, "body") and isinstance(node.body, list):
                node.body = [
                    statement for statement in node.body
                    if not (
                        isinstance(statement, ast.If)
                        and ast.unparse(statement.test) == "cost_guard is not None"
                    )
                ]
        return ast.dump(loop, include_attributes=False)

    if scientific_loop(middle) != scientific_loop(new):
        raise RuntimeError("V14 scientific training loop differs from authenticated V10")
    return {
        "v9_to_v10": prior,
        "v14_seed": 47,
        "v14_run_arm_ast_sha256": hashlib.sha256(
            ast.dump(ast.parse(new), include_attributes=False).encode()
        ).hexdigest(),
        "v10_v14_scientific_training_loop_ast_identical_after_cost_hooks": True,
        "inherited_callables": {
            name: {
                "binding": f"{value.__module__}.{value.__name__}",
                "ast_sha256": hashlib.sha256(
                    ast.dump(ast.parse(inspect.getsource(value)), include_attributes=False).encode()
                ).hexdigest(),
                "source_sha256": sha256_file(inspect.getsourcefile(value)),
            }
            for name, value in inherited.items()
        },
        "v10_to_v14_explicit_diff": "".join(difflib.unified_diff(
            middle.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="v10.run_arm", tofile="v14.run_arm",
        )),
        "v9_run_arm_ast_sha256": hashlib.sha256(
            ast.dump(ast.parse(old), include_attributes=False).encode()
        ).hexdigest(),
        "unchanged_science_bindings": True,
    }


def check() -> dict:
    """Verify executable inheritance and fresh seed-47 start on real clean ECGs."""
    started = time.monotonic()
    _cheap_api_check(torch.cuda)
    data = inputs()
    compatibility = _compatibility_map()
    starts = {}
    _, _, diagnostic, released, _ = v9.datasets(data)
    four = torch.utils.data.Subset(diagnostic, list(range(4)))
    with np.load(v9.v6.V5 / "probe.npz") as probe:
        weight, bias = probe["raw_weight"], float(probe["raw_bias"])
    expected_order46 = torch.randperm(TRAIN_RECORDS, generator=torch.Generator().manual_seed(46))
    expected_order47 = torch.randperm(TRAIN_RECORDS, generator=torch.Generator().manual_seed(47))
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
            "warmup_multipliers": [scheduler.lr_lambdas[-1](step) for step in range(UPDATES)],
            "optimizer_betas": list(optimizer.defaults["betas"]),
            "optimizer_epsilon": optimizer.defaults["eps"],
            "group_weight_decay": [group["weight_decay"] for group in optimizer.param_groups],
            "full_order_sha256": hashlib.sha256(
                torch.randperm(
                    TRAIN_RECORDS,
                    generator=permutation,
                )
                .numpy()
                .tobytes()
            ).hexdigest(),
            "encoder_requires_grad": all(
                parameter.requires_grad for parameter in model.backbone.parameters()
            ),
        }
        features, logits = infer_features(model, four, "cpu")
        if not np.allclose(features, released[:4], atol=2e-4, rtol=2e-5):
            raise RuntimeError("V14 initial released feature path differs")
        if not np.allclose(logits, released[:4].astype(np.float64) @ weight + bias, atol=2e-4, rtol=2e-5):
            raise RuntimeError("V14 initial probe-head logits differ")
        starts[arm] = start
        del model, optimizer, scheduler, permutation
        gc.collect()
    if (
        starts["M"]["model_sha256"] != reference["initial_model_sha256"]
        or starts["M"]["model_sha256"] != starts["F"]["model_sha256"]
    ):
        raise RuntimeError("V14 released tensor initialization differs from V9")
    common_order = hashlib.sha256(expected_order47.numpy().tobytes()).hexdigest()
    if starts["M"]["full_order_sha256"] != common_order or starts["F"]["full_order_sha256"] != common_order:
        raise RuntimeError("V14 M/F seed-47 record order differs")
    if not all(row["optimizer_state_empty"] and row["encoder_requires_grad"] for row in starts.values()):
        raise RuntimeError("V14 full-gradient or fresh-moment initialization differs")
    if (
        any(rate != 0 for rate in starts["F"]["scheduler_base_lrs"][:-1])
        or starts["F"]["scheduler_base_lrs"][-1] != 0.001
    ):
        raise RuntimeError("V14 F encoder or head scheduler base rate differs")
    warmup = [step / UPDATES for step in range(UPDATES)]
    if (
        len(warmup) != 240 or warmup[0] != 0 or warmup[-1] != 239 / 240
        or any(row["warmup_multipliers"] != warmup for row in starts.values())
        or any(row["optimizer_betas"] != [0.9, 0.999] for row in starts.values())
        or any(row["optimizer_epsilon"] != 1e-8 for row in starts.values())
        or any(any(decay != 0.1 for decay in row["group_weight_decay"]) for row in starts.values())
    ):
        raise RuntimeError("V14 warmup schedule differs")
    receipt = {
        "fingerprint": data["fingerprint"],
        "compatibility": compatibility,
        "starts": starts,
        "seed46_full_order_sha256": hashlib.sha256(expected_order46.numpy().tobytes()).hexdigest(),
        "seed47_full_order_sha256": common_order,
        "all_240_warmup_multipliers": warmup,
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
        or not checked["compatibility"]["v9_to_v10"][
            "scientific_training_loop_ast_identical_after_identity_and_gate_guards"
        ]
        or not checked["compatibility"]["unchanged_science_bindings"]
    ):
        raise ValueError("Bridge differs from passed CPU/executable compatibility check")
    train, _, _, _, _ = v9.datasets(data)
    ecg_ids = np.asarray([int(row["ecg_id"]) for row in data["train"]], dtype=np.int64)
    device = _driver_receipt()
    started = time.monotonic()

    def timed_out(_signum: int, _frame: FrameType | None) -> Never:
        raise TimeoutError("V14 bridge exceeded 300 GPU-stage wall seconds")

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
        raise TimeoutError("V14 bridge exceeded the frozen 300-second wall bound")
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


def train(data: dict, preparation_seconds: float, prior_elapsed: float) -> dict:  # noqa: C901
    """Run one fresh M/F pair with checkpoint forecasts and a wall watchdog."""
    bridge_receipt = json.loads((OUT / "bridge.json").read_text())
    if bridge_receipt["fingerprint"] != data["fingerprint"] or not bridge_receipt["passed"]:
        raise ValueError("V14 production lacks a passing bridge")
    for arm, digest in bridge_receipt["arms"].items():
        if sha256_file(OUT / "bridge" / arm / "complete.json") != digest:
            raise ValueError("Bridge completion artifact changed")
        bridge_arm = json.loads((OUT / "bridge" / arm / "complete.json").read_text())
        if bridge_arm["updates"] != 3 or bridge_arm["source"]["seed"] != 47:
            raise ValueError("Bridge checkpoint identity changed")
        for name, expected in bridge_arm["sha256"].items():
            if sha256_file(OUT / "bridge" / arm / name) != expected:
                raise ValueError("Bridge checkpoint or update evidence changed")
    stage_started = time.monotonic() - preparation_seconds
    completed: list[float] = []
    receipts = {}
    block_rates: list[float] = []

    def gate(
        moment: str, unstarted: int, active_update: int | None = None,
        *, tail_complete: bool = False, tail_seconds: float = 0.0,
    ) -> dict:
        receipt = projection(
            prior_elapsed + time.monotonic() - stage_started,
            unstarted,
            tuple(completed),
            active_update=active_update,
            slowest_block_seconds_per_update=max(block_rates) if block_rates else None,
            active_tail_complete=tail_complete,
            observed_tail_seconds=tail_seconds,
            q_remaining=300.0 if len(completed) < 2 else 0.0,
            report_remaining=REPORT_RESERVE,
            stop_reserve=STOP_RESERVE,
        )
        receipt["moment"] = moment
        receipt["fingerprint"] = data["fingerprint"]
        write_json_atomic(OUT / "runtime_gate.json", receipt)
        if not receipt["passed"]:
            raise RuntimeError(f"V14 incremental cost gate failed at {moment}")
        return receipt

    gate("after_production_input_verification", 2)

    def timed_out(_signum: int, _frame: FrameType | None) -> Never:
        raise TimeoutError("V14 production wall watchdog reached safe-stop reserve")

    prior_handler = signal.signal(signal.SIGALRM, timed_out)
    remaining = CEILING - STOP_RESERVE - prior_elapsed - (time.monotonic() - stage_started)
    if remaining <= 0:
        raise TimeoutError("V14 production has no safe-stop wall allowance")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        with gpu_lock("cuda", blocking=False):
            for arm in ARMS:
                unstarted = 1 if arm == "M" else 0
                gate(f"before_{arm}", unstarted + 1)
                block_count = 0

                def cost_guard(
                    moment: str, update: int | None, seconds: float | None,
                    arm_name: str = arm, still_unstarted: int = unstarted,
                ) -> None:
                    nonlocal block_count
                    if moment == "checkpoint":
                        block_count += 1
                        if update != 40 * block_count or seconds is None:
                            raise RuntimeError("V14 checkpoint block sequence changed")
                        block_rates.append(seconds / 40)
                    gate(
                        f"{arm_name}_{moment}_{update}", still_unstarted, update,
                        tail_complete=moment == "tail_complete",
                        tail_seconds=seconds if moment == "tail_complete" and seconds is not None else 0.0,
                    )

                receipts[arm] = run_arm(data, arm, 47, "cuda", "production", cost_guard)
                completed.append(receipts[arm]["complete_pipeline_seconds"])
                gate(f"after_{arm}", unstarted)
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
    if (
        receipts["M"]["initial_model_sha256"] != receipts["F"]["initial_model_sha256"]
        or receipts["M"]["initial_order_sha256"] != receipts["F"]["initial_order_sha256"]
    ):
        raise RuntimeError("V14 production M/F initial tensors or order differ")
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
                raise ValueError("Seed-47 arm belongs to a different V14 source")
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
    if d47 >= 0.005:
        directional_label = "practical_point_screen_passed"
    elif d47 > 0:
        directional_label = "directional_agreement_below_threshold"
    else:
        directional_label = "predicted_direction_not_replicated"
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
        "seed47_directional_label": directional_label,
        "seed47_harm_screen_replicated": bootstrap["seed47_decision"]["joint_harm_replicated"],
        "seed47_readout_gap_pattern_replicated": bootstrap["seed47_decision"][
            "readout_gap_pattern_replicated"
        ],
        "seed47_primary_interval_excludes_zero": bootstrap["seed47_decision"][
            "primary_interval_excludes_zero"
        ],
        "two_seed_mean_D": bootstrap["contrasts"]["mean"]["D"]["observed"],
        "two_seed_D_range": [min(d46, d47), max(d46, d47)],
        "historical_v9_through_v13_lower_approximate_seconds_separate": 6406.508688546029,
        "v14_actual_elapsed_location": (
            "outputs/experiment016_encoder_motion_replication_v14/cost_ledger.json"
        ),
        "calibration_test_evaluated": False,
        "runtime_gate": runtime,
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 v14: encoder-motion seed-47 replication",
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
        seed46_value = bootstrap["contrasts"]["46"][name]["observed"]
        seed47_value = bootstrap["contrasts"]["47"][name]["observed"]
        lines.append(
            f"{name}: seed46 {seed46_value:+.6f}, "
            f"seed47 {seed47_value:+.6f}, "
            f"range [{min(seed46_value, seed47_value):+.6f}, "
            f"{max(seed46_value, seed47_value):+.6f}], "
            f"mean {bootstrap['contrasts']['mean'][name]['observed']:+.6f}; "
            f"seed47 common-draw 95% interval {bootstrap['contrasts']['47'][name]['interval_95']}."
        )
    lines += [
        "",
        f"Seed47 harm screen: {result['seed47_harm_screen_replicated']}; "
        f"readout-gap screen: {result['seed47_readout_gap_pattern_replicated']}.",
        f"Common paired patient draws: {bootstrap['valid']}/{bootstrap['requested']} valid.",
        f"Seed47 decision: {directional_label}; primary interval excludes zero: "
        f"{result['seed47_primary_interval_excludes_zero']}.",
        "Historical v9-v13 spending is at least approximately 6406.509 seconds; "
        "the separate v14 actual elapsed ledger is cost_ledger.json.",
        "The seed46 published V9 intervals remain unchanged; new common-draw intervals are in report.json.",
        "Patient resampling does not estimate seed-population uncertainty or provide independent validation.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:  # noqa: C901 - explicit one-attempt stage accounting
    """Run one frozen stage and never auto-launch a successor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "bridge", "production"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _cheap_api_check(torch.cuda)
    ledger_path = OUT / "cost_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "stages": [], "external_preparation_seconds": 0.0,
    }
    if any(row["stage"] == args.stage for row in ledger["stages"]):
        raise RuntimeError(f"V14 {args.stage} is single invocation; retry forbidden")
    prior_elapsed = ledger["external_preparation_seconds"] + sum(
        row["seconds"] for row in ledger["stages"]
    )
    attempt = OUT / f"attempt_{args.stage}.json"
    if attempt.exists():
        raise RuntimeError(f"V14 {args.stage} attempt already exists")
    write_json_atomic(attempt, {"stage": args.stage, "state": "started"})
    state = "failed"
    try:
        if args.stage == "check":
            result = check()
            after_check = projection(
                prior_elapsed + time.monotonic() - started, 2, (),
                q_remaining=1800.0, report_remaining=REPORT_RESERVE,
                stop_reserve=STOP_RESERVE,
            )
            write_json_atomic(OUT / "cost_gate.json", after_check)
            if not after_check["passed"]:
                raise RuntimeError("V14 post-check cost gate failed")
        else:
            data = inputs()
            preparation_seconds = time.monotonic() - started
            if args.stage == "bridge":
                before = projection(
                    prior_elapsed + preparation_seconds, 2, (),
                    q_remaining=1200.0, report_remaining=REPORT_RESERVE,
                    stop_reserve=STOP_RESERVE,
                )
                write_json_atomic(OUT / "cost_gate.json", before)
                if not before["passed"]:
                    raise RuntimeError("V14 pre-bridge cost gate failed")
                result = bridge(data, preparation_seconds)
                after_bridge = projection(
                    prior_elapsed + time.monotonic() - started, 2, (),
                    q_remaining=900.0, report_remaining=REPORT_RESERVE,
                    stop_reserve=STOP_RESERVE,
                )
                write_json_atomic(OUT / "cost_gate.json", after_bridge)
                if not after_bridge["passed"]:
                    raise RuntimeError("V14 post-bridge cost gate failed")
            else:
                train(data, preparation_seconds, prior_elapsed)
                interim = projection(
                    prior_elapsed + time.monotonic() - started, 0,
                    tuple(json.loads((OUT / "production" / arm / "complete.json").read_text())[
                        "complete_pipeline_seconds"
                    ] for arm in ARMS),
                    q_remaining=0.0, report_remaining=REPORT_RESERVE,
                    stop_reserve=STOP_RESERVE,
                )
                write_json_atomic(OUT / "cost_gate.json", interim)
                if not interim["passed"]:
                    raise RuntimeError("V14 pre-report cost gate failed")
                result = report(data)
        state = "complete"
    finally:
        seconds = time.monotonic() - started
        ledger["stages"].append({"stage": args.stage, "state": state, "seconds": seconds})
        ledger["elapsed_new_v14_seconds"] = ledger["external_preparation_seconds"] + sum(
            row["seconds"] for row in ledger["stages"]
        )
        write_json_atomic(ledger_path, ledger)
        write_json_atomic(attempt, {"stage": args.stage, "state": state, "seconds": seconds})
    if args.stage == "production":
        final = projection(ledger["elapsed_new_v14_seconds"], 0, (
            json.loads((OUT / "production/M/complete.json").read_text())["complete_pipeline_seconds"],
            json.loads((OUT / "production/F/complete.json").read_text())["complete_pipeline_seconds"],
        ), q_remaining=0, report_remaining=0, stop_reserve=0)
        write_json_atomic(OUT / "final_cost_gate.json", final)
        if not final["passed"]:
            raise RuntimeError("V14 final incremental ceiling exceeded")
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
