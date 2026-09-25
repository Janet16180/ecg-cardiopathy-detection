#!/usr/bin/env python3
"""Audit fixed-set evaluation gradients after the frozen 016 v6 training."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.xecg_rescue_training import restore_checkpoint
from ecg_experiment.xecg_rescue_v6 import UPDATES, build_model_seed43
from scripts.experiments import run_xecg_droppath_rescue016 as v5
from scripts.experiments import run_xecg_droppath_rescue016_v6 as v6
from scripts.experiments.run_xecg_probe_finetune016 import CachedECGs

ARMS = ("legacy", "residual", "off")


def fixed_set_gradient(model: nn.Module, data: CachedECGs, device: str) -> dict:
    """Measure the unclipped gradient on the frozen 128 training ECGs in eval mode."""
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_sum = 0.0
    count = 0
    for signal, labels in DataLoader(data, batch_size=16, shuffle=False, num_workers=0):
        signal, labels = signal.to(device), labels.to(device)
        loss = nn.functional.binary_cross_entropy_with_logits(model(signal), labels, reduction="sum")
        (loss / len(data)).backward()
        loss_sum += float(loss.detach())
        count += len(labels)
    if count != 128:
        raise RuntimeError("Fixed gradient diagnostic did not cover all 128 ECGs")
    squared = sum(float(parameter.grad.detach().float().square().sum()) for parameter in model.parameters())
    norm = math.sqrt(squared)
    if not math.isfinite(norm) or not math.isfinite(loss_sum):
        raise RuntimeError("Nonfinite fixed-set evaluation gradient")
    model.zero_grad(set_to_none=True)
    return {
        "dataset": "v5 fixed 64-positive/64-negative clean training ECGs",
        "mode": "evaluation",
        "record_exposures": count,
        "mean_bce": loss_sum / count,
        "gradient_global_l2_before_hypothetical_clip": norm,
        "hypothetical_clip_limit": 3.0,
        "would_clip_at_3": norm > 3.0,
        "actual_training_batch_clip": "not measured by this fixed-set audit",
    }


def audit(inputs: dict) -> dict:  # noqa: C901 - audit all three frozen phases/arms
    """Bind gradient points to checked arm checkpoints and refresh receipts."""
    started = time.monotonic()
    v6.verified_gate(inputs)
    gate = json.loads((v6.OUT / "runtime_gate.json").read_text())
    if not gate["passed"] or gate["remaining_profiled_passes"] != 0:
        raise RuntimeError("Three complete arms and a passing runtime gate are required")
    fixed_ids = json.loads((v6.V5 / "diagnostic.json").read_text())["ecg_ids"]
    rows = v5.diagnostic_rows(inputs)
    if [row["ecg_id"] for row in rows] != fixed_ids:
        raise ValueError("Fixed diagnostic ECG identities changed")
    data = CachedECGs(inputs["views"], inputs["index"], rows)
    result = {
        "fingerprint": inputs["fingerprint"],
        "audit_source_sha256": sha256_file(__file__),
        "inherited_diagnostic_sha256": sha256_file(v6.V5 / "diagnostic.json"),
        "arms": {},
    }
    for arm in ARMS:
        directory = v6.OUT / arm
        receipt_path = directory / "complete.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}:
            raise ValueError(f"{arm} completion fingerprint differs")
        for name, digest in receipt["sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"{arm}/{name} changed before gradient audit")
        trajectory_path = directory / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        if [point["phase"] for point in trajectory] != ["initial", "first_update", "epoch_1"]:
            raise RuntimeError(f"{arm} trajectory phases differ")
        if trajectory[1].get("parameters_identical_to_initial") is not True:
            raise RuntimeError("Zero-LR first update was not proven to preserve initial tensors")
        model, mask, optimizer, scheduler, permutation = build_model_seed43(
            v5.RELEASE, v6.V5 / "probe.npz", arm, "cuda", 15359
        )
        initial_gradient = fixed_set_gradient(model, data, "cuda")
        identity = {"source": inputs["fingerprint"], "arm": arm, "stage": "train"}
        saved = restore_checkpoint(
            directory / "resume.pt", identity, model, mask, optimizer, scheduler, permutation
        )
        if saved["updates"] != UPDATES or saved["next_index"] != 15359 or saved["epoch"] != 0:
            raise RuntimeError("Final checkpoint is not at the one-epoch endpoint")
        final_gradient = fixed_set_gradient(model, data, "cuda")
        trajectory[0]["evaluation_mode_gradient_diagnostic"] = initial_gradient
        trajectory[1]["evaluation_mode_gradient_diagnostic"] = {
            **initial_gradient,
            "same_tensors_as_initial": True,
        }
        trajectory[2]["evaluation_mode_gradient_diagnostic"] = final_gradient
        write_json_atomic(trajectory_path, trajectory)
        receipt["sha256"]["trajectory.json"] = sha256_file(trajectory_path)
        receipt["gradient_audit_source_sha256"] = result["audit_source_sha256"]
        write_json_atomic(receipt_path, receipt)
        result["arms"][arm] = {
            "checkpoint_sha256": receipt["sha256"]["resume.pt"],
            "complete_receipt_sha256": sha256_file(receipt_path),
            "initial": initial_gradient,
            "after_update_1": trajectory[1]["evaluation_mode_gradient_diagnostic"],
            "after_update_240": final_gradient,
            "actual_training_first_update_gradient_before_clip": trajectory[1]["gradient_norm_before_clip"],
        }
        del model, mask, optimizer, scheduler, permutation, saved
        gc.collect()
        torch.cuda.empty_cache()
    result["audit_seconds"] = time.monotonic() - started
    if result["audit_seconds"] >= 300:
        raise RuntimeError("Gradient audit exhausted the 300-second report allowance")
    write_json_atomic(v6.OUT / "gradient_audit.json", result)
    return result


def report(inputs: dict) -> dict:
    """Attach audited gradients to the prespecified v6 report and cost check."""
    audited = json.loads((v6.OUT / "gradient_audit.json").read_text())
    if audited["fingerprint"] != inputs["fingerprint"]:
        raise ValueError("Gradient audit belongs to another v6 source")
    for arm in ARMS:
        if sha256_file(v6.OUT / arm / "complete.json") != audited["arms"][arm]["complete_receipt_sha256"]:
            raise ValueError(f"{arm} receipt changed after gradient audit")
    started = time.monotonic()
    result = v6.report(inputs)
    report_seconds = time.monotonic() - started
    if audited["audit_seconds"] + report_seconds > 300:
        raise RuntimeError("Audit plus report exceeded frozen 300-second allowance")
    result["gradient_audit_sha256"] = sha256_file(v6.OUT / "gradient_audit.json")
    result["report_seconds"] = report_seconds
    inherited_gate = v6.verified_gate(inputs)
    result["actual_cost_components_seconds"] = {
        "historical_preparation_and_profile": inherited_gate["historical_preparation_and_profile_seconds"],
        "new_verification": inherited_gate["new_verification_seconds"],
        "three_arm_passes": [
            json.loads((v6.OUT / arm / "complete.json").read_text())["complete_pass_seconds"] for arm in ARMS
        ],
        "gradient_audit": audited["audit_seconds"],
        "report": report_seconds,
    }
    write_json_atomic(v6.OUT / "report.json", result)
    with (v6.OUT / "report.md").open("a") as stream:
        stream.write(
            "\nFixed 128-training-ECG evaluation-mode gradient norms and hypothetical 3.0 clipping "
            "eligibility are in `gradient_audit.json` and each trajectory. They differ from the "
            "actual training-batch clipping; only update 1's actual preclip norm was captured.\n"
        )
    return result


def main() -> None:
    """Run the post-training audit or final report under the verified source map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "report"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    inputs = v6.evidence_and_inputs()
    if args.stage == "audit":
        with gpu_lock("cuda", blocking=False):
            print(json.dumps(audit(inputs)), flush=True)
    else:
        result = report(inputs)
        print(json.dumps({"stage": "016_v6_report", "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
