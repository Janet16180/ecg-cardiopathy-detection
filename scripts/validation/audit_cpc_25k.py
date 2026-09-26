"""Audit the completed 25k CPC checkpoint and training receipt."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment019_cpc_25k_v1"
BASE = ROOT / "outputs/experiment004_cpc_40k"


def check(condition: bool, message: str) -> None:
    """Stop on a failed training artifact contract."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Verify hashes, update count, finite weights and encoder movement."""
    completion_path = OUTPUT / "complete.json"
    completion = json.loads(completion_path.read_text())
    profile = json.loads((OUTPUT / "profile.json").read_text())
    checkpoint_path = OUTPUT / "latest.pt"
    check(profile["gate_passed"], "Cost gate did not pass")
    check(completion["identity"] == profile["identity"], "Training/profile identity differs")
    check(sha256_file(checkpoint_path) == completion["checkpoint_sha256"],
          "Checkpoint hash mismatch")
    check(sha256_file(OUTPUT / "profile.json") == completion["profile_sha256"],
          "Profile hash mismatch")
    check(completion["subset_size"] == 25_000, "Unexpected subset size")
    check(completion["record_exposures"] == 115_359, "Unexpected exposure count")
    check(completion["completed_updates"] == math.ceil(115_359 / 128),
          "Unexpected optimizer update count")
    check(sum(profile["source_counts"].values()) == 25_000, "Source quotas do not sum to 25k")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    initial = torch.load(BASE / "cpc_ssl/encoder.pt", map_location="cpu", weights_only=True)
    check(saved["identity"] == completion["identity"], "Checkpoint identity differs")
    check(saved["completed_batches"] == completion["completed_updates"],
          "Checkpoint update count differs")
    check(math.isfinite(saved["loss_sum"]), "Training loss is nonfinite")
    moved = 0
    for name, tensor in initial["encoder"].items():
        trained = saved["model"][f"encoder.{name}"]
        check(bool(torch.isfinite(trained).all()), f"Nonfinite encoder tensor: {name}")
        moved += int(not torch.equal(tensor, trained))
    check(moved == len(initial["encoder"]), "Some encoder tensors did not change")
    check("rng" in saved and "optimizer" in saved, "Missing resume state")

    audit = {
        "status": "passed_training_only",
        "completion_sha256": sha256_file(completion_path),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "completed_updates": saved["completed_batches"],
        "changed_encoder_tensors": moved,
        "total_encoder_tensors": len(initial["encoder"]),
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "training_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
