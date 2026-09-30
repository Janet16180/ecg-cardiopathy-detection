"""Read-only CPU audit of completed Experiments 011–013 artifacts.

Only local training receipts and, when present, development output are opened.
The script never reads calibration/test files or runs a model forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
SELECTED_SHA256 = "43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791"
EXPOSURES = 115_359
UPDATES = 902
ARMS = {
    "011": ("gru", "kda", "ckda"),
    "012": ("gru", "mixed", "local"),
    "013": ("gru", "mamba2", "mamba3"),
}
OUTPUTS = {
    "011": "outputs/experiment011_delta_memory_25k_v1",
    "012": "outputs/experiment012_temporal_hybrid",
    "013": "outputs/experiment013_mamba3_25k_v1",
}


def require(condition: bool, message: str) -> None:
    """Raise a useful error for a failed artifact contract."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    """Load an existing JSON receipt."""
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def expected_identity(experiment: str, identity: dict[str, Any]) -> None:
    """Check the frozen common selection, seeds, and training budget."""
    fields = {
        "selected_indices_sha256": SELECTED_SHA256,
        "subset_size": 25_000,
        "selection_seed": 18046,
        "order_seed": 18047,
        "model_seed": {"011": 9001, "012": 12012, "013": 13042}[experiment],
        "weight_decay": 0.01,
    }
    if experiment == "012":
        fields.update(record_exposures=EXPOSURES, updates=UPDATES, batch=128,
                      learning_rate=1e-4)
    else:
        fields.update(record_exposures_per_arm=EXPOSURES, batch_size=128,
                      learning_rate=1e-3, arms=list(ARMS[experiment]))
        require(len(identity["exposure_order_sha256"]) == 64,
                "Missing exposure-order SHA-256")
        require(sum(identity["source_counts"].values()) == 25_000,
                "Source counts do not total 25,000")
    for key, value in fields.items():
        require(identity.get(key) == value, f"Unexpected {experiment} identity {key}")


def fresh_models(experiment: str, seed: int) -> dict[str, torch.nn.Module]:
    """Recreate the runner's seeded, CPU-only initial model states."""
    if experiment == "011":
        from ecg_experiment.cpc_delta_memory import matched_initial_models
    elif experiment == "012":
        from ecg_experiment.cpc_temporal_hybrid import matched_initial_models
    else:
        from ecg_experiment.cpc_mamba013 import matched_initial_models
    return matched_initial_models(seed)


def check_model_source(experiment: str, root: Path, identity: dict[str, Any]) -> None:
    """Require the model implementation used for fresh-state comparison."""
    model_sources = {
        "011": ("ecg_experiment/cpc.py", "ecg_experiment/cpc_delta_memory.py"),
        "012": ("ecg_experiment/cpc.py", "ecg_experiment/cpc_temporal_hybrid.py"),
        "013": ("ecg_experiment/cpc.py", "ecg_experiment/cpc_delta_memory.py",
                "ecg_experiment/cpc_mamba013.py"),
    }[experiment]
    hashes = identity["source_sha256"] if experiment == "012" else identity["hashes"]
    for relative in model_sources:
        require(hashes[relative] == sha256_file(root / relative),
                f"Model source changed since the frozen profile: {relative}")


def check_rng(rng: Any, arm: str) -> None:
    """Check that a checkpoint contains usable Python, NumPy, torch and CUDA RNG states."""
    require(isinstance(rng, dict), f"{arm}: missing RNG mapping")
    require({"python", "numpy", "torch", "cuda"} <= rng.keys(),
            f"{arm}: incomplete RNG state")
    require(isinstance(rng["torch"], torch.Tensor) and rng["torch"].numel() > 0,
            f"{arm}: empty torch RNG state")
    require(isinstance(rng["cuda"], list) and len(rng["cuda"]) > 0,
            f"{arm}: missing CUDA RNG state")
    require(all(isinstance(state, torch.Tensor) and state.numel() > 0
                for state in rng["cuda"]), f"{arm}: invalid CUDA RNG state")
    require(isinstance(rng["python"], tuple) and isinstance(rng["numpy"], tuple),
            f"{arm}: invalid Python/NumPy RNG state")


def check_optimizer(optimizer: Any, arm: str) -> int:
    """Check AdamW state is populated, finite, and reached the final update."""
    require(isinstance(optimizer, dict) and optimizer.get("param_groups"),
            f"{arm}: missing optimizer groups")
    states = optimizer.get("state")
    require(isinstance(states, dict) and states, f"{arm}: empty optimizer state")
    steps = []
    for state in states.values():
        require({"step", "exp_avg", "exp_avg_sq"} <= state.keys(),
                f"{arm}: incomplete AdamW state")
        for value in state.values():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), f"{arm}: nonfinite optimizer state")
        steps.append(int(state["step"].item()))
    require(max(steps) == UPDATES and min(steps) > 0,
            f"{arm}: optimizer update counter differs from completion")
    return len(states)


def audit_arm(experiment: str, output: Path, arm: str, identity: dict[str, Any],
              profile_hash: str, initial: torch.nn.Module) -> dict[str, Any]:
    """Check one completed training receipt and its checkpoint on CPU."""
    arm_dir = output / arm
    complete_path = arm_dir / "complete.json"
    checkpoint = arm_dir / "latest.pt"
    complete = read_json(complete_path)
    require(complete["identity"] == identity and complete["arm"] == arm,
            f"{arm}: completion identity differs from profile")
    require(complete["completed_updates"] == UPDATES
            and complete["record_exposures"] == EXPOSURES,
            f"{arm}: wrong updates or exposures")
    require(complete["profile_sha256"] == profile_hash,
            f"{arm}: profile hash mismatch")
    checkpoint_hash = sha256_file(checkpoint)
    require(complete["checkpoint_sha256"] == checkpoint_hash,
            f"{arm}: checkpoint SHA-256 mismatch")
    # The hash check precedes pickle loading. These are trusted local experiment files.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(saved["identity"] == identity and saved["arm"] == arm,
            f"{arm}: checkpoint identity differs")
    require(saved["completed_updates"] == UPDATES,
            f"{arm}: checkpoint update count differs")
    require(math.isfinite(float(saved["loss_sum"])), f"{arm}: nonfinite loss sum")
    require(math.isclose(float(complete["mean_cpc_loss"]),
                         float(saved["loss_sum"]) / UPDATES, rel_tol=1e-10),
            f"{arm}: mean CPC loss differs")
    state = saved["model"]
    original = initial.state_dict()
    require(state.keys() == original.keys(), f"{arm}: model state keys differ")
    changed = 0
    changed_encoder = 0
    for name, tensor in state.items():
        require(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()),
                f"{arm}: invalid model tensor {name}")
        require(tensor.shape == original[name].shape, f"{arm}: shape mismatch {name}")
        if not torch.equal(tensor, original[name]):
            changed += 1
            changed_encoder += int(name.startswith("encoder."))
    require(changed > 0 and changed_encoder > 0,
            f"{arm}: no trained encoder movement from fresh initialization")
    optimizer_states = check_optimizer(saved["optimizer"], arm)
    check_rng(saved["rng"], arm)
    return {"status": "passed", "checkpoint_sha256": checkpoint_hash,
            "changed_tensors": changed, "changed_encoder_tensors": changed_encoder,
            "total_tensors": len(state), "optimizer_states": optimizer_states}


def audit_readout(experiment: str, root: Path, output: Path,
                  identity: dict[str, Any], profile_hash: str,
                  checkpoints: dict[str, str]) -> dict[str, Any]:
    """Check development artifact hashes and internal prediction identities."""
    readout = (root / "outputs/experiment012_temporal_hybrid_readout"
               if experiment == "012" else output)
    result_path = readout / "result.json"
    if not result_path.exists():
        return {"status": "pending"}
    result = read_json(result_path)
    require(result["status"] == "complete_development_only"
            and result["calibration_test_evaluated"] is False,
            "Development-only readout status changed")
    require(result["profile_sha256"] == sha256_file(readout / "profile.json"),
            "Readout profile SHA-256 mismatch")
    readout_profile = read_json(readout / "profile.json")
    require(readout_profile["gate_passed"] and readout_profile["identity"] == result["identity"],
            "Readout profile gate or identity mismatch")
    if experiment == "012":
        for arm in ARMS[experiment]:
            require(result["identity"]["sha256"][f"{arm}_checkpoint"]
                    == checkpoints[arm], f"Readout checkpoint identity mismatch: {arm}")
        require(result["identity"]["sha256"]["training_profile"] == profile_hash,
                "Readout training profile identity mismatch")
    else:
        require(result["identity"] == identity, "Readout training identity mismatch")
        require(result["checkpoint_sha256"] == checkpoints,
                "Readout checkpoint SHA-256 mapping mismatch")
    for arm in ARMS[experiment]:
        require(result["feature_sha256"][arm]
                == sha256_file(readout / f"{arm}_features.npy"),
                f"Readout feature SHA-256 mismatch: {arm}")
    if "head_sha256" in result:
        for name, expected in result["head_sha256"].items():
            require(expected == sha256_file(readout / f"{name}_head.npz"),
                    f"Readout head SHA-256 mismatch: {name}")
    predictions_path = readout / "development_predictions.npz"
    require(result["predictions_sha256"] == sha256_file(predictions_path),
            "Development predictions SHA-256 mismatch")
    with np.load(predictions_path, allow_pickle=False) as predictions:
        expected_keys = {"record_ids", "patient_ids", "targets"} | {
            f"{budget}_{arm}" for budget in ("full", "limited")
            for arm in ARMS[experiment]
        }
        require(set(predictions.files) == expected_keys,
                "Development prediction keys differ")
        count = result["development_records"]
        require(count == 1_306 and result["development_patients"] == 1_173,
                "Development cohort size differs")
        record_ids = predictions["record_ids"]
        patient_ids = predictions["patient_ids"]
        targets = predictions["targets"]
        require(len(record_ids) == count and len(set(record_ids.tolist())) == count,
                "Development record identity is incomplete or duplicated")
        require(len(patient_ids) == count and len(set(patient_ids.tolist())) == 1_173,
                "Development patient identity differs")
        require(targets.shape == (count,) and np.isin(targets, (0, 1)).all(),
                "Development target vector is invalid")
        for key in expected_keys - {"record_ids", "patient_ids", "targets"}:
            probability = predictions[key]
            require(probability.shape == (count,) and np.isfinite(probability).all()
                    and ((probability >= 0) & (probability <= 1)).all(),
                    f"Invalid development probabilities: {key}")
    return {"status": "passed", "predictions_sha256": result["predictions_sha256"]}


def audit(experiment: str, root: Path = ROOT) -> dict[str, Any]:
    """Audit all completed arms and the optional development readout."""
    output = root / OUTPUTS[experiment]
    profile_path = output / "profile.json"
    if not profile_path.exists():
        return {"experiment": experiment, "status": "pending_profile"}
    profile = read_json(profile_path)
    identity = profile["identity"]
    expected_identity(experiment, identity)
    require(profile["gate_passed"] is True, "Training cost gate did not pass")
    profile_hash = sha256_file(profile_path)
    completed = [arm for arm in ARMS[experiment]
                 if (output / arm / "complete.json").exists()]
    if completed:
        check_model_source(experiment, root, identity)
    initial = fresh_models(experiment, identity["model_seed"]) if completed else {}
    arms = {}
    for arm in ARMS[experiment]:
        if (output / arm / "complete.json").exists():
            arms[arm] = audit_arm(experiment, output, arm, identity, profile_hash,
                                  initial[arm])
        else:
            arms[arm] = {"status": "pending"}
    checkpoints = {arm: report["checkpoint_sha256"] for arm, report in arms.items()
                   if report["status"] == "passed"}
    readout_path = (root / "outputs/experiment012_temporal_hybrid_readout/result.json"
                    if experiment == "012" else output / "result.json")
    if readout_path.exists():
        require(len(checkpoints) == len(ARMS[experiment]),
                "Readout exists before all training arms completed")
    readout = (audit_readout(experiment, root, output, identity, profile_hash, checkpoints)
               if len(checkpoints) == len(ARMS[experiment]) else {"status": "pending"})
    return {"experiment": experiment, "status": "passed" if len(checkpoints)
            == len(ARMS[experiment]) else "pending_training",
            "profile_sha256": profile_hash, "arms": arms, "readout": readout,
            "calibration_test_opened": False}


def main() -> None:
    """Print a JSON audit, optionally preserving one immutable local receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    selected = ARMS if args.experiment == "all" else (args.experiment,)
    result = {name: audit(name) for name in selected}
    if args.receipt is not None:
        receipt = (ROOT / args.receipt).resolve()
        if not receipt.is_relative_to(ROOT / "outputs") or receipt.exists():
            raise ValueError("Receipt must be a new path under repository outputs/")
        write_json_atomic(receipt, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
