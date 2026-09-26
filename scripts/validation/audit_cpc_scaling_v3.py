"""Audit completed CPC scaling checkpoints without evaluating held-out patients."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment018_cpc_data_scaling_v3"
BASE = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
BATCHES = 902
RECORDS = 115359


def audit_arm(arm: str, identity: dict[str, Any], initial: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Check one arm's final state, artifact hash, and actual encoder updates."""
    directory = OUTPUT / arm
    completion = json.loads((directory / "complete.json").read_text())
    checkpoint_path = directory / "latest.pt"
    digest = sha256_file(checkpoint_path)
    if completion["checkpoint_sha256"] != digest:
        raise ValueError(f"{arm} checkpoint hash mismatch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (completion["identity"] != identity or state["identity"] != identity
            or completion["arm"] != arm or state["arm"] != arm):
        raise ValueError(f"{arm} identity mismatch")
    if (completion["records"] != RECORDS or completion["batches"] != BATCHES
            or state["completed_batches"] != BATCHES):
        raise ValueError(f"{arm} exposure or update count mismatch")
    if not math.isfinite(completion["mean_training_loss"]) or not state.get("rng"):
        raise ValueError(f"{arm} missing finite loss or resumable RNG")

    encoder = {name.removeprefix("encoder."): tensor for name, tensor in state["model"].items()
               if name.startswith("encoder.")}
    if encoder.keys() != initial.keys():
        raise ValueError(f"{arm} encoder structure mismatch")
    changed = sum(not torch.equal(value, initial[name]) for name, value in encoder.items())
    if changed == 0 or any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError(f"{arm} encoder did not update or has nonfinite parameters")
    steps = {int(value["step"]) for value in state["optimizer"]["state"].values()
             if "step" in value}
    if steps != {BATCHES}:
        raise ValueError(f"{arm} optimizer step count mismatch: {steps}")
    return {
        "checkpoint_sha256": digest,
        "records": completion["records"],
        "optimizer_updates": state["completed_batches"],
        "mean_training_loss": completion["mean_training_loss"],
        "encoder_tensors_changed": changed,
        "encoder_tensor_count": len(encoder),
    }


def main() -> None:
    """Write an aggregate training-only artifact audit."""
    profile = json.loads((OUTPUT / "profile.json").read_text())
    if not profile["gate_passed"]:
        raise ValueError("Training profile did not pass")
    initial = torch.load(BASE, map_location="cpu", weights_only=True)["encoder"]
    identity = profile["identity"]
    arms = {arm: audit_arm(arm, identity, initial) for arm in ("old", "new")}
    old_state = torch.load(OUTPUT / "old/latest.pt", map_location="cpu", weights_only=False)
    new_state = torch.load(OUTPUT / "new/latest.pt", map_location="cpu", weights_only=False)
    if all(torch.equal(value, new_state["model"][name])
           for name, value in old_state["model"].items()):
        raise ValueError("Matched arms ended with identical model weights")
    write_json_atomic(OUTPUT / "training_audit.json", {
        "status": "passed_training_only", "profile_sha256": sha256_file(OUTPUT / "profile.json"),
        "arms": arms, "calibration_test_evaluated": False,
    })
    print(json.dumps(arms), flush=True)


if __name__ == "__main__":
    main()
