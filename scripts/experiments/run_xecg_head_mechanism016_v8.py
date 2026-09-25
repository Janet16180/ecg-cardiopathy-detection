#!/usr/bin/env python3
"""CPU-only matched-objective head optimization on frozen v7 Off features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from ecg_experiment.evaluation import partition_validation
from ecg_experiment.files import read_csv, sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.xecg_head_mechanism_v8 import (
    ARMS,
    BATCH,
    EPOCHS,
    LAMBDA,
    N_DEV,
    N_TRAIN,
    UPDATES_PER_EPOCH,
    WIDTH,
    HeadState,
    clean_cohort_join,
    diagnostics,
    fit_reference,
    initial_state,
    objective_standardized,
    paired_bootstrap,
    step,
    to_raw,
    to_standardized,
    train_scaler,
)
from ecg_experiment.xecg_rescue_analysis import development_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiment016_head_mechanism_v8"
V7 = ROOT / "outputs/experiment016_frozen_readout_audit_v7"
V6 = ROOT / "outputs/experiment016_droppath_rescue_v6"
V7_MAP = ROOT / "outputs/experiment_queue_016_readout_v7_full/sources.json"
V7_MAP_SHA = "fefa248cb82f54fc812814c7f08ea42a9cb24b4cc367eca8f34cc7219e995c66"
V7_COMPLETION = ROOT / "outputs/experiment_queue_016_readout_v7_full/job/priority_queue_completion.json"
V7_MANIFEST_SHA = "e10b5d404c57f791214f36e2bb5533053252fee6413c60d06f3d9bd2bc0c2921"
TRAIN_CSV = ROOT / "data/processed/ptbxl/seed42_fraction1/labeled_train.csv"
DEV_CSV = ROOT / "data/processed/ptbxl/seed42_fraction1/validation.csv"
CLEAN = ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction1.csv"
V8_SOURCES = (
    "docs/experiment-016-head-mechanism-v8.md",
    "ecg_experiment/xecg_head_mechanism_v8.py",
    "scripts/experiments/run_xecg_head_mechanism016_v8.py",
    "tests/test_xecg_head_mechanism_v8.py",
)


def fingerprint() -> dict:
    """Verify frozen v7 source/results and bind the new implementation bytes."""
    if sha256_file(V7_MAP) != V7_MAP_SHA:
        raise ValueError("V7 frozen source map changed")
    for name, expected in json.loads(V7_MAP.read_text()).items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"V7 source/input changed: {name}")
    completion = json.loads(V7_COMPLETION.read_text())
    if completion["queue_manifest_sha256"] != V7_MANIFEST_SHA:
        raise ValueError("V7 completion belongs to another manifest")
    for name, expected in completion["artifacts"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"V7 completion artifact changed: {name}")
    return {
        "v7_source_map_sha256": V7_MAP_SHA,
        "v7_completion_sha256": sha256_file(V7_COMPLETION),
        "v7_off_feature_sha256": sha256_file(V7 / "off_features.npy"),
        "v7_off_probe_sha256": sha256_file(V7 / "off_probe.json"),
        "v7_report_sha256": sha256_file(V7 / "report.json"),
        "v6_off_checkpoint_sha256": sha256_file(V6 / "off/resume.pt"),
        "v8_sources": {name: sha256_file(ROOT / name) for name in V8_SOURCES},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": str(torch.__version__),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
            "sklearn_logistic_source_sha256": sha256_file(
                Path(sklearn.__file__).parent / "linear_model/_logistic.py"
            ),
        },
    }


def load_data(*, verify: bool = True) -> dict:  # noqa: C901 - one frozen-data identity boundary
    """Load exactly the frozen clean rows, Off features and saved readouts."""
    identity = fingerprint() if verify else json.loads((OUT / "check.json").read_text())["fingerprint"]
    full = read_csv(TRAIN_CSV)
    overlay = read_csv(CLEAN)
    dev, calibration = partition_validation(read_csv(DEV_CSV))
    if len(calibration) != 564:
        raise ValueError("Frozen calibration partition size differs")
    off_receipt = json.loads((V7 / "off_extract.json").read_text())
    if len(dev) != N_DEV:
        raise ValueError("Development row count differs")
    train = clean_cohort_join(full, overlay, dev, off_receipt["ecg_ids_sha256"])
    ids_sha = off_receipt["ecg_ids_sha256"]
    if identity["v7_off_feature_sha256"] != off_receipt["feature_sha256"]:
        raise ValueError("Off feature rows/checkpoint do not match frozen identities")
    features = np.load(V7 / "off_features.npy", mmap_mode="r")
    if features.shape != (N_TRAIN + N_DEV, WIDTH) or features.dtype != np.float32:
        raise ValueError("Off feature shape/dtype changed")
    x = np.asarray(features[:N_TRAIN], dtype=np.float64)
    dev_x = np.asarray(features[N_TRAIN:], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(dev_x).all():
        raise ValueError("Nonfinite Off features")
    y = np.asarray([int(row["target"]) for row in train], dtype=np.float64)
    dev_y = np.asarray([int(row["target"]) for row in dev], dtype=np.int64)
    patients = np.asarray([row["patient_id"] for row in dev])
    mu, scale = train_scaler(x)
    with np.load(V7 / "off_refit_coefficients.npz") as saved:
        if not np.allclose(mu, saved["mean"], rtol=0, atol=1e-12) or not np.allclose(
            scale, saved["scale"], rtol=0, atol=1e-12
        ):
            raise ValueError("Train-only scaler differs from v7")
        refit = np.concatenate((saved["weight"], [float(saved["bias"])]))
    check = json.loads((OUT / "check.json").read_text()) if (OUT / "check.json").exists() else None
    if check is not None:
        if check["fingerprint"] != identity:
            raise ValueError("V8 CPU check belongs to other source")
        with np.load(OUT / "initial_head.npz") as head:
            raw_head = head["raw_head"]
    else:
        saved = torch.load(V6 / "off/resume.pt", map_location="cpu", weights_only=True)
        if saved["updates"] != 240 or saved["next_index"] != N_TRAIN:
            raise ValueError("V6 Off head is not final 240-update checkpoint")
        raw_head = np.concatenate(
            (
                saved["model"]["head.weight"].numpy().reshape(-1).astype(np.float64),
                [float(saved["model"]["head.bias"].item())],
            )
        )
        del saved
    joint = dev_x @ raw_head[:-1] + raw_head[-1]
    refit_logits = dev_x @ refit[:-1] + refit[-1]
    if not np.allclose(joint, np.load(V7 / "off_joint_logits.npy"), atol=1e-10, rtol=1e-11):
        raise ValueError("V6 Off joint head fails v7 logit reproduction")
    if not np.allclose(refit_logits, np.load(V7 / "off_refit_logits.npy"), atol=1e-9, rtol=1e-10):
        raise ValueError("V7 Off refit logit identity differs")
    return {
        "fingerprint": identity,
        "x": x,
        "z": (x - mu) / scale,
        "dev_x": dev_x,
        "y": y,
        "dev_y": dev_y,
        "patients": patients,
        "mu": mu,
        "scale": scale,
        "raw_head": raw_head,
        "refit": refit,
        "ecg_ids_sha256": ids_sha,
        "train_labels_sha256": sha256_file(TRAIN_CSV),
        "development_labels_sha256": sha256_file(DEV_CSV),
        "clean_overlay_sha256": sha256_file(CLEAN),
    }


def check() -> dict:
    """Validate immutable data/source identities, coordinates and solver scaling."""
    started = time.monotonic()
    data = load_data()
    standardized = to_standardized(data["raw_head"], data["mu"], data["scale"])
    raw_logits = data["dev_x"] @ data["raw_head"][:-1] + data["raw_head"][-1]
    z_dev = (data["dev_x"] - data["mu"]) / data["scale"]
    if not np.allclose(raw_logits, z_dev @ standardized[:-1] + standardized[-1], atol=1e-10):
        raise RuntimeError("Raw and standardized initial logits differ")
    sklearn_source = (Path(sklearn.__file__).parent / "linear_model/_logistic.py").read_text()
    if "l2_reg_strength = 1.0 / (C * sw_sum)" not in sklearn_source:
        raise RuntimeError("Pinned sklearn logistic penalty scaling changed")
    # Synthetic optimum checks the installed numerical solver, not just its source text.
    from sklearn.linear_model import LogisticRegression

    synth = np.array([[-2.0, 0.5], [-1.0, -0.3], [0.0, 0.7], [0.4, -0.8], [1.2, 1.1], [2.0, 0.2]])
    sy = np.array([0, 0, 0, 1, 1, 1])
    c = 0.7
    reference = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, tol=1e-12).fit(synth, sy)
    theta = np.concatenate((reference.coef_[0], reference.intercept_))
    _, gradient, _, _ = objective_standardized(theta, synth, sy, 1.0 / (len(sy) * c))
    if np.max(np.abs(gradient)) > 1e-6:
        raise RuntimeError("Installed sklearn optimum contradicts 1/(N*C) normalization")
    write_npz_atomic(
        OUT / "initial_head.npz", raw_head=data["raw_head"], mean=data["mu"], scale=data["scale"]
    )
    receipt = {
        "fingerprint": data["fingerprint"],
        "records": {
            "train": N_TRAIN,
            "development": N_DEV,
            "development_patients": len(np.unique(data["patients"])),
        },
        "ecg_ids_sha256": data["ecg_ids_sha256"],
        "train_labels_sha256": data["train_labels_sha256"],
        "development_labels_sha256": data["development_labels_sha256"],
        "clean_overlay_sha256": data["clean_overlay_sha256"],
        "lambda": LAMBDA,
        "sklearn_synthetic_gradient_inf_norm": float(np.max(np.abs(gradient))),
        "initial_head_sha256": sha256_file(OUT / "initial_head.npz"),
        "seconds": time.monotonic() - started,
        "status": "passed_no_new_outcomes",
    }
    write_json_atomic(OUT / "check.json", receipt)
    return receipt


def state_file(directory: Path) -> Path:
    """Return an arm checkpoint file path."""
    return directory / "head_state.npz"


def save_state(path: Path, state: HeadState) -> str:
    """Atomically preserve numerical state, exact permutation and RNG."""
    metadata = {
        "updates": state.updates,
        "epoch": state.epoch,
        "next_index": state.next_index,
        "rng_state": state.rng_state,
        "clip_events": state.clip_events,
        "clip_factors": state.clip_factors,
    }
    write_npz_atomic(
        path,
        theta=state.theta,
        first=state.first,
        second=state.second,
        order=state.order,
        metadata=np.array(json.dumps(metadata)),
    )
    return sha256_file(path)


def load_state(path: Path) -> HeadState:
    """Restore a checkpoint without pickle, retaining RNG/minibatch position."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        return HeadState(
            saved["theta"].copy(),
            saved["first"].copy(),
            saved["second"].copy(),
            metadata["updates"],
            metadata["epoch"],
            metadata["next_index"],
            saved["order"].copy(),
            metadata["rng_state"],
            metadata["clip_events"],
            metadata["clip_factors"],
        )


def replay_check(state: HeadState, path: Path, data: dict, arm: str) -> dict:
    """Require checkpoint reload to reproduce the next CPU update exactly."""
    continuing = state.copy()
    restored = load_state(path)
    index_a, factor_a = step(continuing, data["x"], data["z"], data["y"], arm, data["mu"], data["scale"])
    index_b, factor_b = step(restored, data["x"], data["z"], data["y"], arm, data["mu"], data["scale"])
    exact = (
        np.array_equal(index_a, index_b)
        and factor_a == factor_b
        and np.array_equal(continuing.theta, restored.theta)
        and np.array_equal(continuing.first, restored.first)
        and np.array_equal(continuing.second, restored.second)
        and np.array_equal(continuing.order, restored.order)
        and continuing.rng_state == restored.rng_state
        and continuing.updates == restored.updates
        and continuing.next_index == restored.next_index
    )
    if not exact:
        raise RuntimeError("CPU saved next-update state differs after reload")
    return {
        "exact": True,
        "next_batch_sha256": hashlib.sha256(index_a.tobytes()).hexdigest(),
        "next_batch_records": len(index_a),
        "next_clipping_factor": factor_a,
    }


def reference_fit(data: dict, directory: Path) -> dict:
    """Fit both fixed-start D references and require train-only convergence."""
    started = time.monotonic()
    joint = to_standardized(data["raw_head"], data["mu"], data["scale"])
    joint_theta, joint_result = fit_reference(data["z"], data["y"], joint)
    zero_theta, zero_result = fit_reference(data["z"], data["y"], np.zeros_like(joint))
    difference = abs(joint_result["J"] - zero_result["J"])
    verified = (
        difference <= 1e-7
        and max(joint_result["standardized_gradient_inf_norm"], zero_result["standardized_gradient_inf_norm"])
        <= 1e-6
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(directory / "reference.npz", joint=joint_theta, zero=zero_theta)
    receipt = {
        "fingerprint": data["fingerprint"],
        "from_joint": joint_result,
        "from_zero": zero_result,
        "objective_difference": difference,
        "numerical_reference_verified": verified,
        "seconds": time.monotonic() - started,
        "reference_sha256": sha256_file(directory / "reference.npz"),
    }
    write_json_atomic(directory / "reference.json", receipt)
    if not verified:
        raise RuntimeError("D train-only optimum cross-check failed; no nonconvergence claims allowed")
    return receipt


def run_arm(data: dict, arm: str, seed: int, epochs: int, directory: Path, *, profile: bool) -> dict:
    """Execute exact complete epochs and store the mandatory 240/2400 endpoints."""
    complete_started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    state = initial_state(data["raw_head"], data["mu"], data["scale"], arm, seed, N_TRAIN)
    first_order_hash = hashlib.sha256(state.order.tobytes()).hexdigest()
    records = {
        "initial": diagnostics(
            state,
            arm,
            data["x"],
            data["z"],
            data["y"],
            data["mu"],
            data["scale"],
            data["raw_head"],
            data["dev_x"],
        )
    }
    epoch_seconds = []
    for epoch in range(epochs):
        started = time.monotonic()
        seen = []
        for _ in range(UPDATES_PER_EPOCH):
            index, _ = step(state, data["x"], data["z"], data["y"], arm, data["mu"], data["scale"])
            seen.append(index)
        if not np.array_equal(np.sort(np.concatenate(seen)), np.arange(N_TRAIN)):
            raise RuntimeError("Epoch did not cover each clean training row once")
        if state.next_index != N_TRAIN or state.updates != (epoch + 1) * UPDATES_PER_EPOCH:
            raise RuntimeError("Fixed 240-update epoch endpoint differs")
        if epoch == 0 or epoch == epochs - 1:
            label = "update240" if epoch == 0 else "update2400"
            records[label] = diagnostics(
                state,
                arm,
                data["x"],
                data["z"],
                data["y"],
                data["mu"],
                data["scale"],
                data["raw_head"],
                data["dev_x"],
            )
            raw = to_raw(state.theta, data["mu"], data["scale"]) if arm == "C" else state.theta
            endpoint_logits = data["dev_x"] @ raw[:-1] + raw[-1]
            np.save(directory / f"{label}_logits.npy", endpoint_logits)
            if profile:
                records[label]["nonselected_development_metrics"] = development_metrics(
                    data["dev_y"], data["patients"], endpoint_logits
                )
            write_npz_atomic(
                directory / f"{label}_head.npz",
                raw=raw,
                standardized=to_standardized(raw, data["mu"], data["scale"]),
            )
            checkpoint_hash = save_state(
                state_file(directory) if profile else directory / f"{label}_state.npz", state
            )
            records[label]["checkpoint_sha256"] = checkpoint_hash
        epoch_seconds.append(time.monotonic() - started)
    if profile:
        replay = replay_check(state, state_file(directory), data, arm)
    else:
        replay = replay_check(state, directory / "update2400_state.npz", data, arm)
    receipt = {
        "fingerprint": data["fingerprint"],
        "arm": arm,
        "seed": seed,
        "profile_only": profile,
        "epochs": epochs,
        "updates": state.updates,
        "exposures": epochs * N_TRAIN,
        "batch_size": BATCH,
        "first_epoch_order_sha256": first_order_hash,
        "epoch_seconds": epoch_seconds,
        "complete_seconds": time.monotonic() - complete_started,
        "replay": replay,
        "diagnostics": records,
        "artifacts_sha256": {path.name: sha256_file(path) for path in directory.iterdir() if path.is_file()},
    }
    write_json_atomic(directory / "receipt.json", receipt)
    return receipt


def profile() -> dict:
    """Time all three complete CPU epochs plus two-start D and freeze the gate."""
    started = time.monotonic()
    data = load_data()
    checked = json.loads((OUT / "check.json").read_text())
    if checked["fingerprint"] != data["fingerprint"]:
        raise ValueError("Profile source differs from CPU preflight")
    prep = time.monotonic() - started
    reference = reference_fit(data, OUT / "profile/reference")
    profile_arms = {}
    for arm in ARMS:
        profile_arms[arm] = run_arm(data, arm, 16080, 1, OUT / "profile" / arm, profile=True)
    p = max(row["complete_seconds"] for row in profile_arms.values())
    q = reference["seconds"]
    measured = time.monotonic() - started + checked["seconds"]
    projected = measured + 1.25 * (60 * p + 2 * q + 300)
    gate = {
        "fingerprint": data["fingerprint"],
        "measured_preparation_profile_seconds": measured,
        "slowest_full_epoch_P_seconds": p,
        "D_two_start_Q_seconds": q,
        "projected_total_seconds": projected,
        "ceiling_seconds": 7200,
        "passed": projected <= 7200,
        "formula": "measured + 1.25*(60*P + 2*Q + 300)",
    }
    write_json_atomic(
        OUT / "profile.json",
        {
            "fingerprint": data["fingerprint"],
            "seed": 16080,
            "nonselected_profile_only": True,
            "preparation_seconds": prep,
            "arms": {
                arm: {
                    "complete_seconds": row["complete_seconds"],
                    "replay": row["replay"],
                    "receipt_sha256": sha256_file(OUT / "profile" / arm / "receipt.json"),
                }
                for arm, row in profile_arms.items()
            },
            "D": reference,
        },
    )
    write_json_atomic(OUT / "cost_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"V8 full CPU cost gate failed: {projected:.1f}s >7200s")
    return gate


def runtime_gate(data: dict, completed: list[dict], d_seconds: float) -> dict:
    """Recheck the remaining prescribed work after each complete production arm."""
    initial = json.loads((OUT / "cost_gate.json").read_text())
    if initial["fingerprint"] != data["fingerprint"] or not initial["passed"]:
        raise ValueError("V8 profile gate missing or changed")
    p = initial["slowest_full_epoch_P_seconds"]
    measured = (
        initial["measured_preparation_profile_seconds"]
        + d_seconds
        + sum(row["complete_seconds"] for row in completed)
    )
    projected = measured + 1.25 * ((6 - len(completed)) * 10 * p + 300)
    gate = {
        "completed_arm_count": len(completed),
        "measured_seconds": measured,
        "projected_total_seconds": projected,
        "ceiling_seconds": 7200,
        "passed": projected <= 7200,
    }
    write_json_atomic(OUT / "runtime_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("V8 measured remaining-cost gate failed")
    return gate


def train() -> dict:
    """Run fresh D and all six fixed arm/seed combinations after the profile gate."""
    data = load_data()
    gate = json.loads((OUT / "cost_gate.json").read_text())
    if gate["fingerprint"] != data["fingerprint"] or not gate["passed"]:
        raise ValueError("V8 production requires a verified passing profile gate")
    reference = reference_fit(data, OUT / "production/reference")
    completed = []
    runtime_gate(data, completed, reference["seconds"])
    for seed in (44, 45):
        for arm in ARMS:
            receipt = run_arm(
                data, arm, seed, EPOCHS, OUT / "production" / f"seed{seed}" / arm, profile=False
            )
            completed.append(receipt)
            runtime_gate(data, completed, reference["seconds"])
            print(
                json.dumps({"seed": seed, "arm": arm, "complete_seconds": receipt["complete_seconds"]}),
                flush=True,
            )
    result = {
        "fingerprint": data["fingerprint"],
        "D": reference,
        "arms": {
            f"seed{row['seed']}_{row['arm']}": sha256_file(
                OUT / "production" / f"seed{row['seed']}" / row["arm"] / "receipt.json"
            )
            for row in completed
        },
        "runtime_gate": json.loads((OUT / "runtime_gate.json").read_text()),
    }
    write_json_atomic(OUT / "production.json", result)
    return result


def report() -> dict:  # noqa: C901 - prespecified contrast and decision boundary
    """Analyze prespecified endpoints and paired patient contrasts without selection."""
    started = time.monotonic()
    data = load_data()
    production = json.loads((OUT / "production.json").read_text())
    if production["fingerprint"] != data["fingerprint"]:
        raise ValueError("Production and report source identities differ")
    dev_labels, patients = data["dev_y"], data["patients"]
    logits = {
        "released_probe": np.load(V7 / "released_refit_logits.npy"),
        "off_joint": np.load(V7 / "off_joint_logits.npy"),
        "v7_refit": np.load(V7 / "off_refit_logits.npy"),
    }
    with np.load(OUT / "production/reference/reference.npz") as d:
        d_raw = to_raw(d["joint"], data["mu"], data["scale"])
        logits["D"] = data["dev_x"] @ d_raw[:-1] + d_raw[-1]
    np.save(OUT / "production/reference/D_logits.npy", logits["D"])
    endpoints = {}
    for seed in (44, 45):
        for arm in ARMS:
            key = f"seed{seed}_{arm}"
            directory = OUT / "production" / f"seed{seed}" / arm
            receipt = json.loads((directory / "receipt.json").read_text())
            if receipt["fingerprint"] != data["fingerprint"] or receipt["updates"] != 2400:
                raise ValueError(f"Incomplete {key} endpoint")
            for name, digest in receipt["artifacts_sha256"].items():
                if sha256_file(directory / name) != digest:
                    raise ValueError(f"Changed {key} artifact: {name}")
            endpoints[key] = receipt["diagnostics"]
            for update in (240, 2400):
                logits[f"{key}_{update}"] = np.load(directory / f"update{update}_logits.npy")
    metrics = {name: development_metrics(dev_labels, patients, values) for name, values in logits.items()}
    pairs = {}
    for seed in (44, 45):
        prefix = f"seed{seed}_"
        for update in (240, 2400):
            pairs[f"C_minus_B_{update}_seed{seed}"] = (f"{prefix}C_{update}", f"{prefix}B_{update}")
            pairs[f"B_minus_A_{update}_seed{seed}"] = (f"{prefix}B_{update}", f"{prefix}A_{update}")
        pairs[f"A_minus_joint_240_seed{seed}"] = (f"{prefix}A_240", "off_joint")
        pairs[f"A_minus_v7_refit_240_seed{seed}"] = (f"{prefix}A_240", "v7_refit")
        for arm in ARMS:
            pairs[f"{arm}_terminal_minus_D_seed{seed}"] = (f"{prefix}{arm}_2400", "D")
            pairs[f"{arm}_terminal_minus_release_seed{seed}"] = (f"{prefix}{arm}_2400", "released_probe")
    pairs["mean_C_minus_B_240"] = (("seed44_C_240", "seed44_B_240"), ("seed45_C_240", "seed45_B_240"))
    bootstrap = paired_bootstrap(dev_labels, patients, logits, pairs)
    ref = production["D"]["from_joint"]["J"]
    gaps = {
        key: {
            label: max(0.0, row["J"] - ref) if abs(row["J"] - ref) <= 1e-7 else row["J"] - ref
            for label, row in values.items()
        }
        for key, values in endpoints.items()
    }
    primary_seed = {
        str(seed): metrics[f"seed{seed}_C_240"]["auroc"] - metrics[f"seed{seed}_B_240"]["auroc"]
        for seed in (44, 45)
    }
    conditioning = all(
        primary_seed[str(seed)] >= 0.005
        and gaps[f"seed{seed}_B"]["update240"] > 1e-6
        and gaps[f"seed{seed}_C"]["update240"] <= 0.5 * gaps[f"seed{seed}_B"]["update240"]
        for seed in (44, 45)
    )
    catches_up = all(
        abs(metrics[f"seed{seed}_C_2400"]["auroc"] - metrics[f"seed{seed}_B_2400"]["auroc"]) <= 0.002
        and gaps[f"seed{seed}_B"]["update2400"] <= 1e-5
        and gaps[f"seed{seed}_C"]["update2400"] <= 1e-5
        for seed in (44, 45)
    )
    extra_adamw = all(
        metrics[f"seed{seed}_A_240"]["auroc"] - metrics["off_joint"]["auroc"] >= 0.005
        and metrics[f"seed{seed}_A_240"]["auroc"] >= metrics["v7_refit"]["auroc"] - 0.002
        for seed in (44, 45)
    )
    regularization = all(
        metrics[f"seed{seed}_B_240"]["auroc"] - metrics[f"seed{seed}_A_240"]["auroc"] >= 0.005
        for seed in (44, 45)
    )
    mean_interval = bootstrap["contrasts"]["mean_C_minus_B_240"]["interval_95"]
    result = {
        "fingerprint": data["fingerprint"],
        "scope": "postmortem development-only fixed Off features",
        "records": {"train": N_TRAIN, "development": N_DEV, "development_patients": len(np.unique(patients))},
        "fixed_lambda": LAMBDA,
        "metrics": metrics,
        "training_diagnostics": endpoints,
        "full_training_objective_gaps": gaps,
        "D": production["D"],
        "bootstrap": bootstrap,
        "decision": {
            "primary_C_minus_B_240_by_seed": primary_seed,
            "primary_C_minus_B_240_mean": float(np.mean(list(primary_seed.values()))),
            "conditioning_screen_passed": conditioning,
            "mean_bootstrap_interval_includes_zero": mean_interval[0] <= 0 <= mean_interval[1],
            "B_catches_C_by_2400": catches_up,
            "extra_stationary_AdamW_suffices": extra_adamw,
            "regularization_package_contributes": regularization,
            "retain_released_probe": True,
        },
        "runtime_gate": json.loads((OUT / "runtime_gate.json").read_text()),
        "report_seconds": time.monotonic() - started,
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUT / "report.json", result)
    lines = [
        "# Experiment 016 v8: matched-objective head optimization",
        "",
        "Development-only postmortem of frozen seed-43 Off features; no encoder update.",
        "",
        "| Readout | AUROC | AP | BCE | Patient-fold sensitivity | Patient-fold specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['auroc']:.5f} | {row['average_precision']:.5f} | "
            f"{row['bce']:.5f} | {row['mean_fold_sensitivity']:.5f} | "
            f"{row['mean_fold_specificity']:.5f} |"
        )
    lines += [
        "",
        f"Primary C−B at 240 by seed: {primary_seed}; "
        f"mean {result['decision']['primary_C_minus_B_240_mean']:+.6f}.",
        f"Mean paired patient 95% interval: {mean_interval}; "
        f"{bootstrap['valid']}/{bootstrap['requested']} valid draws.",
        f"Conditioning screen: {conditioning}; B catches C by 2400: {catches_up}.",
        f"Extra AdamW suffices: {extra_adamw}; regularization package contribution: {regularization}.",
        "Intervals condition on one frozen encoder and a repeatedly inspected development cohort.",
        "No calibration/test, encoder update, backbone selection or hyperparameter search.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    """Execute one verified v8 CPU stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train", "report"), required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        result = {"check": check, "profile": profile, "train": train, "report": report}[args.stage]()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": "complete",
                "gate": result.get("projected_total_seconds"),
                "decision": result.get("decision"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
