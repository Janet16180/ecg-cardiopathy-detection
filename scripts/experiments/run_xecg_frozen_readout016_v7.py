#!/usr/bin/env python3
"""Frozen-backbone xECG readout audit on clean training/development ECGs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.xecg_readout_v7 import (
    DEVELOPMENT_RECORDS,
    FEATURE_WIDTH,
    TOTAL_RECORDS,
    TRAIN_RECORDS,
    affine_logits,
    decisions,
    fixed_probe,
    paired_bootstrap,
    selected_indices,
)
from ecg_experiment.xecg_rescue_analysis import development_metrics
from ecg_experiment.xecg_rescue_v6 import build_model_seed43
from scripts.experiments import run_xecg_droppath_rescue016 as v5
from scripts.experiments import run_xecg_droppath_rescue016_v6 as v6
from scripts.experiments.run_xecg_probe_finetune016 import CachedECGs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiment016_frozen_readout_audit_v7"
V6 = ROOT / "outputs/experiment016_droppath_rescue_v6"
V6_MAP = ROOT / "outputs/experiment_queue_016_rescue_v6_audit_v2/sources.json"
V6_MAP_SHA = "49f142ed4c5459b71724b314fcf34583c1a2884878a344baaa785f03f96a2a88"
ARMS = ("off", "residual", "legacy")
V7_SOURCES = (
    "docs/experiment-016-frozen-readout-audit-v7.md",
    "ecg_experiment/xecg_readout_v7.py",
    "scripts/experiments/run_xecg_frozen_readout016_v7.py",
    "tests/test_xecg_frozen_readout_v7.py",
)


def verified_inputs() -> dict:
    """Bind clean rows and all three final frozen states to prior receipts."""
    if sha256_file(V6_MAP) != V6_MAP_SHA:
        raise ValueError("Frozen v6 source map changed")
    prior_sources = json.loads(V6_MAP.read_text())
    changed = [name for name, expected in prior_sources.items() if sha256_file(ROOT / name) != expected]
    if changed:
        raise ValueError(f"Frozen v6 source/input bytes changed: {changed}")
    inputs = v6.evidence_and_inputs()
    v6_report = json.loads((V6 / "report.json").read_text())
    v6_audit = json.loads((V6 / "gradient_audit.json").read_text())
    if v6_report["fingerprint"] != inputs["fingerprint"] or v6_audit["fingerprint"] != inputs["fingerprint"]:
        raise ValueError("V6 result/gradient audit belongs to other inputs")
    if v6_report["records"] != {
        "train": TRAIN_RECORDS,
        "development": DEVELOPMENT_RECORDS,
        "development_patients": 1173,
    }:
        raise ValueError("V6 clean cohort differs")
    checkpoints = {}
    for arm in ARMS:
        directory = V6 / arm
        receipt = json.loads((directory / "complete.json").read_text())
        if receipt["fingerprint"] != {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}:
            raise ValueError(f"{arm} checkpoint input identity differs")
        if (receipt["optimizer_updates"], receipt["record_exposures"], receipt["final_epoch"]) != (
            240,
            15359,
            1,
        ):
            raise ValueError(f"{arm} is not final seed-43 one-epoch state")
        checkpoint = directory / "resume.pt"
        digest = sha256_file(checkpoint)
        if digest != receipt["sha256"]["resume.pt"] or digest != v6_audit["arms"][arm]["checkpoint_sha256"]:
            raise ValueError(f"{arm} checkpoint digest differs from receipts")
        if sha256_file(directory / "complete.json") != v6_audit["arms"][arm]["complete_receipt_sha256"]:
            raise ValueError(f"{arm} completion receipt changed after audit")
        checkpoints[arm] = digest
    inputs["v6_fingerprint"] = inputs["fingerprint"]
    inputs["fingerprint"] = {
        "v6": inputs["v6_fingerprint"],
        "v6_source_map_sha256": V6_MAP_SHA,
        "v6_report_sha256": sha256_file(V6 / "report.json"),
        "v6_audit_sha256": sha256_file(V6 / "gradient_audit.json"),
        "v6_checkpoints_sha256": checkpoints,
        "v7_sources": {name: sha256_file(ROOT / name) for name in V7_SOURCES},
        "readout_c": 0.01,
        "train_records": TRAIN_RECORDS,
        "development_records": DEVELOPMENT_RECORDS,
    }
    return inputs


def frozen_model(arm: str, device: str, inputs: dict) -> tuple[torch.nn.Module, dict | None]:
    """Strictly load an adapted state, discard optimizer, and freeze/eval it."""
    model, mask, optimizer, scheduler, permutation = build_model_seed43(
        v5.RELEASE, v6.V5 / "probe.npz", arm if arm in ARMS else "off", device, TRAIN_RECORDS
    )
    del mask, optimizer, scheduler, permutation
    saved = None
    if arm != "released":
        saved = torch.load(V6 / arm / "resume.pt", map_location="cpu", weights_only=True)
        if (
            saved["fingerprint"] != {"source": inputs["v6_fingerprint"], "arm": arm, "stage": "train"}
            or saved["epoch"] != 0
            or saved["updates"] != 240
            or saved["next_index"] != TRAIN_RECORDS
        ):
            raise ValueError(f"{arm} checkpoint is not at the exact frozen v6 endpoint")
        model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Readout audit encoder is not frozen/eval")
    return model, saved


def assert_immutable(model: torch.nn.Module, saved_state: dict[str, torch.Tensor]) -> None:
    """Require every extracted model tensor to match its initial frozen state."""
    state = model.state_dict()
    if state.keys() != saved_state.keys() or any(
        not torch.equal(value.detach().cpu(), saved_state[name]) for name, value in state.items()
    ):
        raise RuntimeError("Frozen model tensor changed during feature extraction")


def check() -> dict:
    """Validate cohort/checkpoints and released cache on a fixed train sample."""
    started = time.monotonic()
    inputs = verified_inputs()
    ids = np.load(v5.HISTORICAL / "ecg_ids.npy")
    requested = [int(row["ecg_id"]) for row in inputs["train"] + inputs["development"]]
    indices = selected_indices(ids, requested)
    cached = np.load(v5.HISTORICAL / "features.npy", mmap_mode="r")
    if not np.array_equal(np.asarray(cached[indices[:TRAIN_RECORDS]]), inputs["train_features"]):
        raise ValueError("Clean train feature join differs from v6")
    if not np.array_equal(np.asarray(cached[indices[TRAIN_RECORDS:]]), inputs["development_features"]):
        raise ValueError("Development feature join differs from v6")
    rows = inputs["train"][:16]
    sample = CachedECGs(inputs["views"], inputs["index"], rows)
    model, _ = frozen_model("released", "cpu", inputs)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    logits, features = v5.diagnostic_pass(model, sample, "cpu")
    assert_immutable(model, before)
    expected_features = inputs["train_features"][:16]
    difference = float(np.max(np.abs(features - expected_features)))
    if not np.allclose(features, expected_features, atol=2e-4, rtol=2e-5):
        raise RuntimeError(f"Native released features differ from verified cache: {difference}")
    with np.load(v6.V5 / "probe.npz") as probe:
        expected_logits = affine_logits(expected_features, probe["raw_weight"], float(probe["raw_bias"]))
    logit_difference = float(np.max(np.abs(logits - expected_logits)))
    if not np.allclose(logits, expected_logits, atol=2e-4, rtol=2e-5):
        raise RuntimeError("Native released head differs from cached probe on fixed sample")
    del model, before
    gc.collect()
    strict = {}
    for arm in ARMS:
        model, saved = frozen_model(arm, "cpu", inputs)
        if saved is None:
            raise RuntimeError("Adapted checkpoint did not load")
        assert_immutable(model, saved["model"])
        strict[arm] = {
            "checkpoint_sha256": inputs["fingerprint"]["v6_checkpoints_sha256"][arm],
            "updates": saved["updates"],
            "model_tensors": len(saved["model"]),
            "strict_state_load": True,
        }
        del model, saved
        gc.collect()
    receipt = {
        "fingerprint": inputs["fingerprint"],
        "fixed_sample_ecg_ids": [row["ecg_id"] for row in rows],
        "release_feature_cache_max_abs_difference": difference,
        "release_native_head_max_abs_difference": logit_difference,
        "strict_checkpoints": strict,
        "seconds": time.monotonic() - started,
        "status": "passed_no_outcome_calculation",
    }
    write_json_atomic(OUT / "check.json", receipt)
    return receipt


def feature_paths(arm: str) -> tuple[Path, Path, Path]:
    """Return this arm's feature, chunk-progress and completion paths."""
    return OUT / f"{arm}_features.npy", OUT / f"{arm}_progress.json", OUT / f"{arm}_extract.json"


def extract_arm(inputs: dict, arm: str, device: str) -> dict:  # noqa: C901 - resumable chunk invariants
    """Extract all 16,665 eval-mode features into resumable hashed chunks."""
    if device != "cuda" or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen feature extraction requires the profiled V100")
    feature_path, progress_path, receipt_path = feature_paths(arm)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != inputs["fingerprint"] or receipt["feature_sha256"] != sha256_file(
            feature_path
        ):
            raise ValueError(f"Existing {arm} feature receipt differs")
        return receipt
    started = time.monotonic()
    rows = inputs["train"] + inputs["development"]
    data = CachedECGs(inputs["views"], inputs["index"], rows)
    ids = [int(row["ecg_id"]) for row in rows]
    ids_sha = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != inputs["fingerprint"] or progress["ecg_ids_sha256"] != ids_sha:
            raise ValueError("Feature-chunk resume identity differs")
        features = np.lib.format.open_memmap(
            feature_path, mode="r+", dtype=np.float32, shape=(TOTAL_RECORDS, FEATURE_WIDTH)
        )
        expected_start = 0
        for chunk in progress["chunks"]:
            if chunk["start"] != expected_start or not chunk["start"] < chunk["end"] <= TOTAL_RECORDS:
                raise ValueError("Feature chunks are not contiguous clean rows")
            actual = hashlib.sha256(features[chunk["start"] : chunk["end"]].tobytes()).hexdigest()
            if actual != chunk["sha256"]:
                raise ValueError("A completed feature chunk changed before resume")
            expected_start = chunk["end"]
        position = progress["next_index"]
        if position != expected_start:
            raise ValueError("Feature resume pointer differs from verified chunks")
        earlier_seconds = float(progress.get("elapsed_seconds", 0.0))
    else:
        if feature_path.exists():
            raise FileExistsError("Unreceipted feature file exists")
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(TOTAL_RECORDS, FEATURE_WIDTH)
        )
        progress = {
            "fingerprint": inputs["fingerprint"],
            "ecg_ids_sha256": ids_sha,
            "next_index": 0,
            "chunks": [],
        }
        position = 0
        earlier_seconds = 0.0
    model, saved = frozen_model(arm, device, inputs)
    baseline = (
        saved["model"]
        if saved is not None
        else {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    )
    with torch.inference_mode():
        while position < TOTAL_RECORDS:
            end = min(position + 256, TOTAL_RECORDS)
            loader = DataLoader(
                Subset(data, range(position, end)), batch_size=16, shuffle=False, num_workers=0
            )
            cursor = position
            for signal, _ in loader:
                pooled, _ = model.backbone(signal.to(device))
                values = pooled.float().cpu().numpy()
                if values.shape[1] != FEATURE_WIDTH or not np.isfinite(values).all():
                    raise RuntimeError("Invalid extracted pooled feature dimensions/values")
                features[cursor : cursor + len(values)] = values
                cursor += len(values)
            if cursor != end:
                raise RuntimeError("Feature chunk coverage differs")
            features.flush()
            progress["chunks"].append(
                {
                    "start": position,
                    "end": end,
                    "sha256": hashlib.sha256(features[position:end].tobytes()).hexdigest(),
                }
            )
            position = end
            progress["next_index"] = position
            progress["elapsed_seconds"] = earlier_seconds + time.monotonic() - started
            write_json_atomic(progress_path, progress)
    assert_immutable(model, baseline)
    del model, saved, baseline
    gc.collect()
    torch.cuda.empty_cache()
    if position != TOTAL_RECORDS:
        raise RuntimeError("Feature extraction stopped before all clean train/development rows")
    receipt = {
        "fingerprint": inputs["fingerprint"],
        "arm": arm,
        "feature_sha256": sha256_file(feature_path),
        "progress_sha256": sha256_file(progress_path),
        "ecg_ids_sha256": ids_sha,
        "shape": [TOTAL_RECORDS, FEATURE_WIDTH],
        "train_records": TRAIN_RECORDS,
        "development_records": DEVELOPMENT_RECORDS,
        "chunks": len(progress["chunks"]),
        "encoder_tensors_unchanged": True,
        "encoder_gradients_disabled": True,
        "encoder_eval_mode": True,
        "complete_pass_seconds": earlier_seconds + time.monotonic() - started,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def fit_release_control(inputs: dict) -> dict:
    """Time and verify a fresh fixed-C probe on the released cached features."""
    receipt_path = OUT / "released_control.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != inputs["fingerprint"]:
            raise ValueError("Existing released control belongs to other inputs")
        for name, digest in receipt["sha256"].items():
            if sha256_file(OUT / name) != digest:
                raise ValueError(f"Released control artifact changed: {name}")
        return receipt
    started = time.monotonic()
    labels = np.array([int(row["target"]) for row in inputs["train"]], dtype=np.int64)
    probe = fixed_probe(inputs["train_features"], labels, inputs["development_features"])
    reference = np.load(V6 / "probe_logits.npy")
    difference = float(np.max(np.abs(probe["development_logits"] - reference)))
    if difference > 1e-8:
        raise RuntimeError(f"Fresh released probe differs from recorded reference logits: {difference}")
    development_labels = np.array([int(row["target"]) for row in inputs["development"]])
    auc = float(roc_auc_score(development_labels, probe["development_logits"]))
    if abs(auc - 0.9619404113151374) > 1e-12:
        raise RuntimeError("Fresh released probe does not reproduce 0.9619404 AUROC")
    np.save(OUT / "released_refit_logits.npy", probe["development_logits"])
    write_npz_atomic(
        OUT / "released_refit_coefficients.npz",
        weight=probe["weight"],
        bias=np.array(probe["bias"]),
        mean=probe["mean"],
        scale=probe["scale"],
        coefficient=probe["coefficient"],
        intercept=np.array(probe["intercept"]),
    )
    receipt = {
        "fingerprint": inputs["fingerprint"],
        "C": 0.01,
        "solver": "lbfgs",
        "max_iter": 3000,
        "iterations": probe["iterations"],
        "development_auroc_identity_check": auc,
        "reference_logit_max_abs_difference": difference,
        "train_only_scaler": True,
        "seconds": time.monotonic() - started,
        "sha256": {
            name: sha256_file(OUT / name)
            for name in ("released_refit_logits.npy", "released_refit_coefficients.npz")
        },
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def cost_projection(
    base_seconds: float,
    off_seconds: float,
    control_fit_seconds: float,
    completed_extractions: list[float],
    completed_probes: list[float],
) -> dict:
    """Apply the frozen inherited floor and update remaining work estimates."""
    if len(completed_extractions) > 2 or len(completed_probes) > 3:
        raise ValueError("Too many completed readout stages")
    t = max(190.0, off_seconds)
    p = max(60.0, control_fit_seconds)
    measured_extra = sum(completed_extractions) + sum(completed_probes)
    remaining = (2 - len(completed_extractions)) * t + (3 - len(completed_probes)) * p + 300
    projected = base_seconds + measured_extra + 1.25 * remaining
    return {
        "actual_preparation_off_control_seconds": base_seconds,
        "off_complete_extraction_seconds": off_seconds,
        "released_control_fit_seconds": control_fit_seconds,
        "T_seconds": t,
        "P_seconds": p,
        "completed_additional_extraction_seconds": completed_extractions,
        "completed_adapted_probe_seconds": completed_probes,
        "remaining_plus_report_allowance_seconds": remaining,
        "margin": 1.25,
        "projected_total_seconds": projected,
        "ceiling_seconds": 7200,
        "passed": projected <= 7200,
    }


def initial_cost_gate(inputs: dict, off_stage_seconds: float) -> dict:
    """Freeze the measured Off/control real-path gate before any other arm."""
    check_receipt = json.loads((OUT / "check.json").read_text())
    off_receipt = json.loads(feature_paths("off")[2].read_text())
    control = json.loads((OUT / "released_control.json").read_text())
    if any(
        receipt["fingerprint"] != inputs["fingerprint"] for receipt in (check_receipt, off_receipt, control)
    ):
        raise ValueError("Off cost evidence source identities differ")
    gate = cost_projection(
        check_receipt["seconds"] + off_stage_seconds,
        off_receipt["complete_pass_seconds"],
        control["seconds"],
        [],
        [],
    )
    gate["off_stage_seconds"] = off_stage_seconds
    gate["off_feature_sha256"] = off_receipt["feature_sha256"]
    gate["control_receipt_sha256"] = sha256_file(OUT / "released_control.json")
    write_json_atomic(OUT / "cost_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"V7 Off full-path gate exceeds two hours: {gate['projected_total_seconds']:.1f}s")
    return gate


def verified_cost_gate(inputs: dict) -> dict:
    """Require passing Off evidence before extracting or fitting controls."""
    gate = json.loads((OUT / "cost_gate.json").read_text())
    if not gate["passed"] or gate["off_feature_sha256"] != sha256_file(feature_paths("off")[0]):
        raise ValueError("Off feature gate is absent, failed or changed")
    if gate["control_receipt_sha256"] != sha256_file(OUT / "released_control.json"):
        raise ValueError("Released control changed after gate")
    check_receipt = json.loads((OUT / "check.json").read_text())
    if check_receipt["fingerprint"] != inputs["fingerprint"]:
        raise ValueError("CPU check belongs to other inputs")
    return gate


def update_runtime_gate(inputs: dict, stage_seconds: float, *, kind: str, arm: str) -> dict:
    """Replace one remaining estimate with its measured stage cost."""
    initial = verified_cost_gate(inputs)
    path = OUT / "runtime_gate.json"
    if path.exists():
        previous = json.loads(path.read_text())
        extractions = previous["completed_additional_extraction_seconds"]
        probes = previous["completed_adapted_probe_seconds"]
        extraction_arms = previous["completed_extraction_arms"]
        probe_arms = previous["completed_probe_arms"]
    else:
        extractions, probes = [], []
        extraction_arms, probe_arms = [], []
    if kind == "extract":
        if arm not in extraction_arms:
            extractions = [*extractions, stage_seconds]
            extraction_arms = [*extraction_arms, arm]
    elif kind == "probe":
        if arm not in probe_arms:
            probes = [*probes, stage_seconds]
            probe_arms = [*probe_arms, arm]
    else:
        raise ValueError("Unknown runtime-gate stage")
    gate = cost_projection(
        initial["actual_preparation_off_control_seconds"],
        initial["off_complete_extraction_seconds"],
        initial["released_control_fit_seconds"],
        extractions,
        probes,
    )
    gate["completed_extraction_arms"] = extraction_arms
    gate["completed_probe_arms"] = probe_arms
    write_json_atomic(path, gate)
    if not gate["passed"]:
        raise RuntimeError(
            f"V7 measured runtime gate exceeded two hours: {gate['projected_total_seconds']:.1f}s"
        )
    return gate


def adapted_probe(inputs: dict, arm: str) -> dict:
    """Fit one fixed train-only probe and evaluate two saved affine heads."""
    receipt_path = OUT / f"{arm}_probe.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != inputs["fingerprint"]:
            raise ValueError(f"Existing {arm} probe belongs to other inputs")
        for name, digest in receipt["sha256"].items():
            if sha256_file(OUT / name) != digest:
                raise ValueError(f"{arm} probe artifact changed: {name}")
        return receipt
    started = time.monotonic()
    feature_path, _, extract_path = feature_paths(arm)
    extraction = json.loads(extract_path.read_text())
    if (
        extraction["fingerprint"] != inputs["fingerprint"]
        or sha256_file(feature_path) != extraction["feature_sha256"]
    ):
        raise ValueError(f"{arm} extraction changed before fixed probe")
    features = np.load(feature_path, mmap_mode="r")
    if features.shape != (TOTAL_RECORDS, FEATURE_WIDTH):
        raise ValueError("Adapted feature shape differs")
    train_labels = np.array([int(row["target"]) for row in inputs["train"]], dtype=np.int64)
    probe = fixed_probe(features[:TRAIN_RECORDS], train_labels, features[TRAIN_RECORDS:])
    dev = features[TRAIN_RECORDS:]
    with np.load(v6.V5 / "probe.npz") as original:
        original_logits = affine_logits(dev, original["raw_weight"], float(original["raw_bias"]))
    saved = torch.load(V6 / arm / "resume.pt", map_location="cpu", weights_only=True)
    if saved["fingerprint"] != {"source": inputs["v6_fingerprint"], "arm": arm, "stage": "train"}:
        raise ValueError("Saved joint head belongs to another frozen model")
    weight = saved["model"]["head.weight"].numpy().reshape(-1)
    bias = float(saved["model"]["head.bias"].item())
    joint_logits = affine_logits(dev, weight, bias)
    recorded = np.load(V6 / arm / "development_logits_epoch1.npy")
    difference = float(np.max(np.abs(joint_logits - recorded)))
    if not np.allclose(joint_logits, recorded, atol=2e-4, rtol=2e-5):
        raise RuntimeError(f"{arm} saved joint head fails v6 development-logit reproduction: {difference}")
    names = {
        "refit": f"{arm}_refit_logits.npy",
        "original_head": f"{arm}_original_head_logits.npy",
        "joint": f"{arm}_joint_logits.npy",
        "coefficients": f"{arm}_refit_coefficients.npz",
    }
    np.save(OUT / names["refit"], probe["development_logits"])
    np.save(OUT / names["original_head"], original_logits)
    np.save(OUT / names["joint"], joint_logits)
    write_npz_atomic(
        OUT / names["coefficients"],
        weight=probe["weight"],
        bias=np.array(probe["bias"]),
        mean=probe["mean"],
        scale=probe["scale"],
        coefficient=probe["coefficient"],
        intercept=np.array(probe["intercept"]),
    )
    receipt = {
        "fingerprint": inputs["fingerprint"],
        "arm": arm,
        "feature_sha256": extraction["feature_sha256"],
        "C": 0.01,
        "solver": "lbfgs",
        "max_iter": 3000,
        "iterations": probe["iterations"],
        "train_only_scaler": True,
        "saved_joint_head_max_abs_logit_difference": difference,
        "logits": names,
        "sha256": {name: sha256_file(OUT / name) for name in names.values()},
        "seconds": time.monotonic() - started,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def report(inputs: dict, preparation_seconds: float = 0.0) -> dict:  # noqa: C901 - analysis gates
    """Analyze frozen readouts, paired patients and prespecified Off gates."""
    started = time.monotonic()
    gate = verified_cost_gate(inputs)
    runtime = json.loads((OUT / "runtime_gate.json").read_text())
    if not runtime["passed"] or set(runtime["completed_extraction_arms"]) != {"residual", "legacy"}:
        raise RuntimeError("Both secondary feature extractions must finish within cost gate")
    if set(runtime["completed_probe_arms"]) != set(ARMS):
        raise RuntimeError("All three fixed adapted probes must finish before outcomes")
    labels = np.array([int(row["target"]) for row in inputs["development"]], dtype=np.int64)
    patients = np.array([row["patient_id"] for row in inputs["development"]])
    logits = {"released_refit": np.load(OUT / "released_refit_logits.npy")}
    probe_receipts = {}
    for arm in ARMS:
        receipt = json.loads((OUT / f"{arm}_probe.json").read_text())
        if receipt["fingerprint"] != inputs["fingerprint"]:
            raise ValueError(f"{arm} fixed probe belongs to other source")
        for name, digest in receipt["sha256"].items():
            if sha256_file(OUT / name) != digest:
                raise ValueError(f"{arm} readout artifact changed: {name}")
        for name in ("refit", "original_head", "joint"):
            logits[f"{arm}_{name}"] = np.load(OUT / receipt["logits"][name])
        probe_receipts[arm] = sha256_file(OUT / f"{arm}_probe.json")
    if any(len(values) != DEVELOPMENT_RECORDS or not np.isfinite(values).all() for values in logits.values()):
        raise ValueError("Development readout logits are incomplete or nonfinite")
    metrics = {name: development_metrics(labels, patients, values) for name, values in logits.items()}
    contrasts = {
        "off_refit_minus_off_joint": ("off_refit", "off_joint"),
        "off_refit_minus_released_probe": ("off_refit", "released_refit"),
        "residual_refit_minus_residual_joint": ("residual_refit", "residual_joint"),
        "legacy_refit_minus_legacy_joint": ("legacy_refit", "legacy_joint"),
        "off_original_head_minus_off_joint": ("off_original_head", "off_joint"),
        "residual_original_head_minus_residual_joint": ("residual_original_head", "residual_joint"),
        "legacy_original_head_minus_legacy_joint": ("legacy_original_head", "legacy_joint"),
    }
    bootstrap = paired_bootstrap(labels, patients, logits, contrasts)
    decision = decisions(metrics, bootstrap)
    elapsed = preparation_seconds + time.monotonic() - started
    if elapsed > 300:
        raise RuntimeError("V7 report exceeded the frozen 300-second allowance")
    measured = (
        gate["actual_preparation_off_control_seconds"]
        + sum(runtime["completed_additional_extraction_seconds"])
        + sum(runtime["completed_adapted_probe_seconds"])
        + elapsed
    )
    if measured > 7200:
        raise RuntimeError("V7 actual total exceeded two-hour ceiling")
    result = {
        "fingerprint": inputs["fingerprint"],
        "scope": "postmortem fixed readouts on inspected development annotation proxy; no calibration/test",
        "records": {
            "train": TRAIN_RECORDS,
            "development": DEVELOPMENT_RECORDS,
            "development_patients": len(np.unique(patients)),
        },
        "readout_metrics": metrics,
        "paired_patient_bootstrap": bootstrap,
        "decision": decision,
        "probe_receipt_sha256": probe_receipts,
        "initial_cost_gate": gate,
        "final_runtime_gate": runtime,
        "measured_total_seconds_including_report": measured,
        "report_seconds": elapsed,
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 v7: frozen-backbone linear readout audit",
        "",
        "Development-only, postmortem use of one seed-43 adapted checkpoint per arm; no encoder updates.",
        "",
        "| Backbone / head | AUROC | AP | BCE | Patient-fold sensitivity | Patient-fold specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "released_refit",
        *(f"{arm}_{head}" for arm in ARMS for head in ("joint", "original_head", "refit")),
    ):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['auroc']:.5f} | {row['average_precision']:.5f} | {row['bce']:.5f} | "
            f"{row['mean_fold_sensitivity']:.5f} | {row['mean_fold_specificity']:.5f} |"
        )
    lines += [
        "",
        f"Primary Off refit − Off joint AUROC: {decision['off_refit_minus_joint']:+.5f}; "
        f"paired patient interval {bootstrap['contrasts']['off_refit_minus_off_joint']['interval_95']}.",
        f"Off refit − released probe: {decision['off_refit_minus_released_probe']:+.5f}.",
        f"Recoverable readout point screen: {decision['recoverable_readout_point_screen']}; "
        f"near-complete recovery: {decision['near_complete_practical_recovery']}; "
        f"potential useful adaptation: {decision['potential_useful_adaptation']}.",
        f"Paired patient draws: {bootstrap['valid']} valid / {bootstrap['requested']} requested.",
        "Intervals condition on frozen trained models and this repeatedly inspected development cohort.",
        "A refit changes head optimization/regularization and does not isolate "
        "representation information loss.",
        "No calibration/test or follow-up fitting was performed.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Run one frozen check/extract/probe/report stage without encoder updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "extract", "probe", "report"), required=True)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        if args.device != "cpu":
            raise ValueError("V7 preflight must run on CPU")
        print(json.dumps(check()), flush=True)
        return
    started = time.monotonic()
    inputs = verified_inputs()
    if args.stage == "extract":
        if args.arm is None:
            raise ValueError("Feature extraction requires one explicit arm")
        if args.arm != "off":
            verified_cost_gate(inputs)
        with gpu_lock(args.device, blocking=False):
            receipt = extract_arm(inputs, args.arm, args.device)
        if args.arm == "off":
            fit_release_control(inputs)
            gate = initial_cost_gate(inputs, time.monotonic() - started)
        else:
            measured = max(time.monotonic() - started, receipt["complete_pass_seconds"])
            gate = update_runtime_gate(inputs, measured, kind="extract", arm=args.arm)
        print(json.dumps({"stage": "extract", "arm": args.arm, "gate": gate}), flush=True)
    elif args.stage == "probe":
        verified_cost_gate(inputs)
        for arm in ARMS:
            arm_start = started if arm == ARMS[0] else time.monotonic()
            receipt = adapted_probe(inputs, arm)
            measured = max(time.monotonic() - arm_start, receipt["seconds"])
            gate = update_runtime_gate(inputs, measured, kind="probe", arm=arm)
            print(json.dumps({"stage": "probe", "arm": arm, "gate": gate}), flush=True)
    elif args.stage == "report":
        result = report(inputs, time.monotonic() - started)
        print(json.dumps({"stage": "report", "decision": result["decision"]}), flush=True)
    else:
        raise ValueError("Unknown v7 stage")


if __name__ == "__main__":
    main()
