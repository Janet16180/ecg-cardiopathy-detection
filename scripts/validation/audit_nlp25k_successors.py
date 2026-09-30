"""Read-only CPU audit of the versioned 011–013 training and development artifacts.

The shared cache seal checks file identity and bounded blocks. This command does
not hash full waveform caches, run a forward pass, or open held-out waveforms.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment.cache_session_seal import validate_seal
from ecg_experiment.cpc_subset25 import exposure_order, select_indices
from ecg_experiment.files import read_csv, sha256_file, sha256_json, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
SEAL = "outputs/cache_sessions/nlp25k_v1/seal.json"
COHORT = "data/processed/sampled_100k_plus_labels_v1"
CACHE = "data/processed/sampled_100k_cpc_cache_v1"
PTB = "data/processed/cpc_pool_40k"
CLEAN = "outputs/data_quality/clean_rerun_preflight_v1"
NORMALIZATION = "outputs/experiment004_cpc_40k/normalization.json"
SELECTED_SHA256 = "43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791"
EXPOSURES = 115_359
UPDATES = 902
ARMS = {"011": ("gru", "kda", "ckda"), "012": ("gru", "mixed", "local"),
        "013": ("gru", "mamba2", "mamba3")}
OUTPUTS = {"011": "outputs/experiment011_delta_memory_25k_v2",
           "012": "outputs/experiment012_temporal_hybrid_25k_v3",
           "013": "outputs/experiment013_mamba3_25k_v2"}
READOUT_012 = "outputs/experiment012_temporal_hybrid_25k_v3_readout"
COMMON_INPUTS = {
    "cohort_manifest": f"{COHORT}/train_manifest.csv",
    "cohort_metadata": f"{COHORT}/metadata.json",
    "cache_receipt": f"{CACHE}/complete.json",
    "cache_verification": "outputs/experiment018_cpc_data_scaling_v2/cache_verification.json",
    "ptb_receipt": f"{PTB}/complete.json",
    "ptb_rows": f"{PTB}/rows.csv",
    "ptb_ids": f"{PTB}/ecg_ids.npy",
    "normalization": NORMALIZATION,
    "clean_full_labels": f"{CLEAN}/labels_fraction1.csv",
    "clean_limited_labels": f"{CLEAN}/labels_fraction0.1.csv",
    "heldout_references": f"{CLEAN}/heldout_references.csv",
}
READOUT_INPUTS_012 = {
    "pool_complete": f"{PTB}/complete.json", "pool_rows": f"{PTB}/rows.csv",
    "pool_ids": f"{PTB}/ecg_ids.npy", "normalization": NORMALIZATION,
    "full_labels": f"{CLEAN}/labels_fraction1.csv",
    "limited_labels": f"{CLEAN}/labels_fraction0.1.csv",
    "heldout_references": f"{CLEAN}/heldout_references.csv",
    "model_source": "ecg_experiment/cpc_temporal_hybrid.py",
    "cpc_source": "ecg_experiment/cpc.py", "pool_source": "ecg_experiment/cpc_pool.py",
    "readout_source": "ecg_experiment/cpc_scaling_readout.py",
    "head_source": "ecg_experiment/cpc_local_readout.py",
    "runner_source": "scripts/experiments/run_cpc_temporal_hybrid012_readout_v3.py",
    "protocol": "docs/experiment-012-temporal-hybrid-v3.md",
}


def require(condition: bool, message: str) -> None:
    """Reject an artifact contract failure with a specific error."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from an existing receipt."""
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def check_hashes(root: Path, hashes: dict[str, str], paths: dict[str, str]) -> None:
    """Hash exactly the named small inputs and executable sources."""
    require(set(hashes) == set(paths), "Unexpected or missing pinned input hashes")
    for name, relative in paths.items():
        require(hashes[name] == sha256_file(root / relative), f"Changed input: {name}")


def check_seal(root: Path, identity: dict[str, Any], experiment: str) -> dict[str, Any]:
    """Validate the pinned bounded cache seal without a full waveform hash."""
    seal_path = root / SEAL
    file_hash = sha256_file(seal_path)
    seal = validate_seal(seal_path)
    require(sha256_file(seal_path) == file_hash, "Cache seal changed during audit")
    require(seal["status"] == "passed"
            and seal["verification_mode"] == "cached_stat_and_bounded_blocks"
            and seal["full_sha256_recomputed"] is False
            and seal["creation_verification"] in (
                "fresh_full_sha256", "external_completed_full_sha256_profile"),
            "Cache seal verification mode or creation evidence changed")
    entries = read_json(seal_path)["files"]
    expected = {"training_signals": (root / CACHE / "signals.npy").resolve(),
                "ptb_signals": (root / PTB / "signals.npy").resolve()}
    require(set(entries) == set(expected) and all(
        Path(entries[name]["path"]) == path for name, path in expected.items()
    ), "Cache seal covers different waveform paths")
    key = "cache_seal_content_sha256" if experiment == "011" else "cache_session_seal_sha256"
    require(identity.get(key) == seal["seal_sha256"], "Cache seal content SHA-256 mismatch")
    if experiment == "011":
        require(identity.get("cache_seal_sha256") == file_hash
                and identity.get("cache_verification_mode") == seal["verification_mode"]
                and identity.get("cache_seal_creation_verification") == seal["creation_verification"],
                "011 cache seal identity differs")
    return seal


def check_identity(experiment: str, root: Path, identity: dict[str, Any],
                   seal: dict[str, Any]) -> None:
    """Reconstruct the frozen 25k selection and verify small pinned inputs."""
    expected = {"selected_indices_sha256": SELECTED_SHA256, "subset_size": 25_000,
                "selection_seed": 18046, "order_seed": 18047,
                "model_seed": {"011": 9001, "012": 12012, "013": 13042}[experiment],
                "weight_decay": 0.01}
    if experiment == "012":
        expected.update(record_exposures=EXPOSURES, updates=UPDATES,
                        batch=128, learning_rate=1e-4)
    else:
        expected.update(record_exposures_per_arm=EXPOSURES, batch_size=128,
                        learning_rate=1e-3, arms=list(ARMS[experiment]))
    for key, value in expected.items():
        require(identity.get(key) == value, f"Unexpected {experiment} identity {key}")
    rows = read_csv(root / COMMON_INPUTS["cohort_manifest"])
    require(len(rows) == EXPOSURES and all(row["split"] == "train" for row in rows)
            and len({row["record_id"] for row in rows}) == EXPOSURES,
            "Parent training cohort changed")
    sources = [row["source"] for row in rows]
    selected = select_indices(sources, 25_000, 18046)
    require(sha256_json(selected.tolist()) == SELECTED_SHA256,
            "Selected training rows differ")
    if experiment != "012":
        order = exposure_order(selected, EXPOSURES, 18047)
        require(identity.get("exposure_order_sha256") == sha256_json(order.tolist()),
                "Exposure order SHA-256 mismatch")
        counts = dict(sorted(Counter(sources[index] for index in selected).items()))
        require(identity.get("source_counts") == counts, "Source-stratified counts differ")
        hashes = identity["hashes"]
        paths = dict(COMMON_INPUTS)
        paths.update({key: key for key in hashes if "/" in key})
        small_hashes = {key: value for key, value in hashes.items()
                        if key not in ("cache_signals", "ptb_signals")}
        check_hashes(root, small_hashes, paths)
        require(hashes["cache_signals"] == seal["full_sha256_from_creation"]["training_signals"]
                and hashes["ptb_signals"] == seal["full_sha256_from_creation"]["ptb_signals"],
                "Waveform cache SHA-256 differs from seal creation evidence")
    else:
        paths = {key: COMMON_INPUTS[key] for key in (
            "cohort_manifest", "cohort_metadata", "cache_receipt", "cache_verification",
            "normalization", "ptb_receipt", "ptb_rows", "ptb_ids",
            "clean_full_labels", "clean_limited_labels", "heldout_references")}
        paths["experiment019_profile"] = "outputs/experiment019_cpc_25k_v1/profile.json"
        # The 012 receipt names its PTB inputs identically to the 011/013 map.
        check_hashes(root, {key: value for key, value in identity["input_sha256"].items()
                            if key != "ptb_signals"}, paths)
        check_hashes(root, identity["source_sha256"],
                     {key: key for key in identity["source_sha256"]})
        require(identity["cached_signals_sha256"]
                == seal["full_sha256_from_creation"]["training_signals"]
                and identity["input_sha256"]["ptb_signals"]
                == seal["full_sha256_from_creation"]["ptb_signals"],
                "012 waveform cache SHA-256 differs from seal")
    cache = read_json(root / COMMON_INPUTS["cache_receipt"])
    ptb = read_json(root / COMMON_INPUTS["ptb_receipt"])
    hashes = identity["input_sha256"] if experiment == "012" else identity["hashes"]
    require(cache["signals_sha256"] == seal["full_sha256_from_creation"]["training_signals"]
            and ptb["signals_sha256"] == seal["full_sha256_from_creation"]["ptb_signals"]
            and ptb["rows_sha256"] == hashes["ptb_rows"]
            and ptb["ecg_ids_sha256"] == hashes["ptb_ids"],
            "Waveform source receipts differ from sealed identities")


def initial_models(experiment: str, seed: int) -> dict[str, torch.nn.Module]:
    """Create each seeded initial CPU model after source hashes pass."""
    if experiment == "011":
        from ecg_experiment.cpc_delta_memory import matched_initial_models
    elif experiment == "012":
        from ecg_experiment.cpc_temporal_hybrid import matched_initial_models
    else:
        from ecg_experiment.cpc_mamba013 import matched_initial_models
    return matched_initial_models(seed)


def check_optimizer_rng(saved: dict[str, Any], arm: str) -> int:
    """Verify a populated final AdamW state and resumable RNG streams."""
    optimizer = saved["optimizer"]
    states = optimizer.get("state")
    require(optimizer.get("param_groups") and isinstance(states, dict) and states,
            f"{arm}: missing optimizer state")
    steps = []
    for state in states.values():
        require({"step", "exp_avg", "exp_avg_sq"} <= state.keys(),
                f"{arm}: incomplete AdamW state")
        for value in state.values():
            if isinstance(value, torch.Tensor):
                require(bool(torch.isfinite(value).all()), f"{arm}: nonfinite optimizer state")
        steps.append(int(state["step"].item()))
    require(min(steps) > 0 and max(steps) == UPDATES, f"{arm}: wrong optimizer update")
    rng = saved["rng"]
    require(isinstance(rng, dict) and {"python", "numpy", "torch", "cuda"} <= rng.keys()
            and isinstance(rng["python"], tuple) and isinstance(rng["numpy"], tuple)
            and isinstance(rng["torch"], torch.Tensor) and rng["torch"].numel() > 0
            and isinstance(rng["cuda"], list) and len(rng["cuda"]) > 0
            and all(isinstance(item, torch.Tensor) and item.numel() > 0
                    for item in rng["cuda"]), f"{arm}: incomplete RNG state")
    return len(states)


def check_arm(output: Path, arm: str, identity: dict[str, Any], profile_hash: str,
              initial: torch.nn.Module) -> dict[str, Any]:
    """Verify final checkpoint integrity, identity, optimizer and encoder movement."""
    directory = output / arm
    complete = read_json(directory / "complete.json")
    checkpoint = directory / "latest.pt"
    digest = sha256_file(checkpoint)
    require(complete["identity"] == identity and complete["arm"] == arm
            and complete["completed_updates"] == UPDATES
            and complete["record_exposures"] == EXPOSURES
            and complete["profile_sha256"] == profile_hash
            and complete["checkpoint_sha256"] == digest,
            f"{arm}: completion receipt identity or SHA-256 mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(saved["identity"] == identity and saved["arm"] == arm
            and saved["completed_updates"] == UPDATES
            and math.isfinite(float(saved["loss_sum"]))
            and math.isclose(complete["mean_cpc_loss"], saved["loss_sum"] / UPDATES,
                             rel_tol=1e-10), f"{arm}: final checkpoint contents differ")
    model = saved["model"]
    fresh = initial.state_dict()
    require(model.keys() == fresh.keys(), f"{arm}: model parameter keys differ")
    changed = 0
    encoder_changed = 0
    for name, tensor in model.items():
        require(isinstance(tensor, torch.Tensor) and tensor.shape == fresh[name].shape
                and bool(torch.isfinite(tensor).all()), f"{arm}: invalid model tensor {name}")
        if not torch.equal(tensor, fresh[name]):
            changed += 1
            encoder_changed += int(name.startswith("encoder."))
    require(changed > 0 and encoder_changed > 0, f"{arm}: encoder did not move")
    states = check_optimizer_rng(saved, arm)
    return {"status": "passed", "checkpoint_sha256": digest,
            "changed_tensors": changed, "changed_encoder_tensors": encoder_changed,
            "optimizer_states": states}


def development_rows(root: Path) -> tuple[list[tuple[str, str, int]], int]:
    """Reconstruct development ID order from small manifests, without waveform reads."""
    full = read_csv(root / COMMON_INPUTS["clean_full_labels"])
    limited = read_csv(root / COMMON_INPUTS["clean_limited_labels"])
    require(len(full) == 15_359 and len(limited) == 1_518,
            "PTB training label budgets differ")
    references = [row for row in read_csv(root / COMMON_INPUTS["heldout_references"])
                  if row["split"] == "development"]
    require(len(references) == 1_306, "PTB development count differs")
    ids = np.load(root / COMMON_INPUTS["ptb_ids"], allow_pickle=False)
    index = {str(value): position for position, value in enumerate(ids)}
    require(len(index) == len(ids), "Duplicate PTB cache IDs")
    ordered = sorted(references, key=lambda row: index[row["record_id"].removeprefix("ptbxl:")])
    expected = [(row["record_id"], row["patient_id"], int(row["target"]))
                for row in ordered]
    require(len({item[0] for item in expected}) == 1_306
            and len({item[1] for item in expected}) == 1_173
            and not ({row["patient_id"] for row in full} & {item[1] for item in expected}),
            "PTB development identities or patient separation differ")
    return expected, len(full)


def check_readout(experiment: str, root: Path, output: Path,
                  identity: dict[str, Any], profile_hash: str,
                  checkpoints: dict[str, str]) -> dict[str, Any]:
    """Verify development-only receipt, hashes, IDs, heads and probabilities."""
    directory = root / READOUT_012 if experiment == "012" else output
    result_path = directory / "result.json"
    if not result_path.exists():
        return {"status": "pending"}
    result = read_json(result_path)
    require(result["status"] == "complete_development_only"
            and result["calibration_test_evaluated"] is False,
            "Development-only readout status differs")
    if experiment == "012":
        readout_identity = result["identity"]
        require(readout_identity["cache_session_seal_sha256"]
                == identity["cache_session_seal_sha256"], "012 readout seal differs")
        paths = dict(READOUT_INPUTS_012)
        paths["training_profile"] = str((output / "profile.json").relative_to(root))
        for arm in ARMS[experiment]:
            paths[f"{arm}_completion"] = str((output / arm / "complete.json").relative_to(root))
            paths[f"{arm}_checkpoint"] = str((output / arm / "latest.pt").relative_to(root))
        sha = readout_identity["sha256"]
        require(sha.get("pool_signals") == identity["input_sha256"]["ptb_signals"],
                "012 readout PTB cache identity differs")
        check_hashes(root, {key: value for key, value in sha.items() if key != "pool_signals"}, paths)
        require(sha["training_profile"] == profile_hash
                and all(sha[f"{arm}_checkpoint"] == checkpoints[arm] for arm in ARMS[experiment]),
                "012 readout checkpoint identity differs")
        for key, expected in {"feature_width": 512, "bootstrap_seed": 12045,
                              "bootstrap_draws": 2000, "full_labels": 15_359,
                              "limited_labels": 1_518, "development_records": 1_306,
                              "development_patients": 1_173}.items():
            require(readout_identity.get(key) == expected, f"012 readout identity {key} differs")
        require(readout_identity.get("classifier") == (
            "StandardScaler plus L2 logistic C=0.01 lbfgs max_iter=5000 tol=1e-8 seed=42"
        ), "012 readout classifier identity differs")
    else:
        require(result["identity"] == identity
                and result["checkpoint_sha256"] == checkpoints
                and result["profile_sha256"] == profile_hash
                and result["training_labels_full"] == 15_359
                and result["training_labels_limited"] == 1_518,
                "Readout training identity or label budget differs")
    readout_profile = read_json(directory / "profile.json")
    require(readout_profile["gate_passed"] is True
            and readout_profile["identity"] == result["identity"]
            and result["profile_sha256"] == sha256_file(directory / "profile.json"),
            "Readout profile gate, identity or SHA-256 differs")
    expected, training_count = development_rows(root)
    require(result["development_records"] == 1_306
            and result["development_patients"] == 1_173,
            "Readout development cohort count differs")
    require(set(result["feature_sha256"]) == set(ARMS[experiment]),
            "Readout feature arm mapping differs")
    for arm in ARMS[experiment]:
        path = directory / f"{arm}_features.npy"
        require(result["feature_sha256"][arm] == sha256_file(path),
                f"{arm}: feature SHA-256 mismatch")
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        require(features.shape == (training_count + 1_306, 512)
                and np.issubdtype(features.dtype, np.floating)
                and bool(np.isfinite(features).all()), f"{arm}: invalid feature matrix")
    prediction_path = directory / "development_predictions.npz"
    require(result["predictions_sha256"] == sha256_file(prediction_path),
            "Development prediction SHA-256 mismatch")
    expected_keys = {"record_ids", "patient_ids", "targets"} | {
        f"{budget}_{arm}" for budget in ("full", "limited") for arm in ARMS[experiment]}
    with np.load(prediction_path, allow_pickle=False) as values:
        require(set(values.files) == expected_keys, "Development prediction keys differ")
        actual = list(zip(values["record_ids"].tolist(), values["patient_ids"].tolist(),
                          values["targets"].tolist(), strict=True))
        require(actual == expected, "Development record, patient or target IDs differ")
        for budget in ("full", "limited"):
            for arm in ARMS[experiment]:
                name = f"{budget}_{arm}"
                probability = values[name]
                require(probability.shape == (1_306,) and bool(np.isfinite(probability).all())
                        and bool(((probability >= 0) & (probability <= 1)).all()),
                        f"{name}: invalid development probabilities")
                score = result["scores"][budget][arm]
                require(score["training_labels"] == (15_359 if budget == "full" else 1_518)
                        and math.isclose(score["auroc"], roc_auc_score(values["targets"], probability),
                                         abs_tol=1e-10)
                        and math.isclose(score["average_precision"],
                                         average_precision_score(values["targets"], probability),
                                         abs_tol=1e-10), f"{name}: score receipt differs")
                if experiment == "013":
                    head_path = directory / f"{name}_head.npz"
                    require(result["head_sha256"][name] == sha256_file(head_path),
                            f"{name}: head SHA-256 mismatch")
                    with np.load(head_path, allow_pickle=False) as head:
                        x = np.load(directory / f"{arm}_features.npy", mmap_mode="r",
                                    allow_pickle=False)[training_count:].astype(np.float64)
                        replay = expit(((x - head["mean"]) / head["scale"])
                                       @ head["coef"].reshape(-1) + head["intercept"].item())
                        require(bool(np.allclose(replay, probability, rtol=0, atol=1e-10)),
                                f"{name}: saved head does not replay predictions")
    if experiment == "013":
        require(set(result["head_sha256"]) == expected_keys - {
            "record_ids", "patient_ids", "targets"}, "Readout head mapping differs")
    return {"status": "passed", "predictions_sha256": result["predictions_sha256"]}


def audit(experiment: str, root: Path = ROOT) -> dict[str, Any]:
    """Audit one versioned successor, allowing pending stages without GPU work."""
    output = root / OUTPUTS[experiment]
    profile_path = output / "profile.json"
    if not profile_path.exists():
        return {"experiment": experiment, "status": "pending_profile"}
    profile = read_json(profile_path)
    require(profile["gate_passed"] is True, "Training cost gate did not pass")
    identity = profile["identity"]
    seal = check_seal(root, identity, experiment)
    check_identity(experiment, root, identity, seal)
    profile_hash = sha256_file(profile_path)
    completed = [arm for arm in ARMS[experiment]
                 if (output / arm / "complete.json").exists()]
    initial = initial_models(experiment, identity["model_seed"]) if completed else {}
    arms = {arm: check_arm(output, arm, identity, profile_hash, initial[arm])
            if arm in completed else {"status": "pending"} for arm in ARMS[experiment]}
    checkpoints = {arm: report["checkpoint_sha256"] for arm, report in arms.items()
                   if report["status"] == "passed"}
    readout_dir = root / READOUT_012 if experiment == "012" else output
    if (readout_dir / "result.json").exists():
        require(len(checkpoints) == len(ARMS[experiment]),
                "Readout exists before completed training artifacts")
    readout = (check_readout(experiment, root, output, identity, profile_hash, checkpoints)
               if len(checkpoints) == len(ARMS[experiment]) else {"status": "pending"})
    return {"experiment": experiment, "status": "passed" if len(checkpoints)
            == len(ARMS[experiment]) else "pending_training",
            "profile_sha256": profile_hash, "arms": arms, "readout": readout,
            "calibration_test_opened": False}


def main() -> None:
    """Print a JSON report and optionally save one new receipt under outputs/."""
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
