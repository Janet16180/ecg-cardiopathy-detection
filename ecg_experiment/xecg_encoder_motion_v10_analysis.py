"""Compatibility, cost and paired-patient analysis for Experiment 016 v10."""

from __future__ import annotations

import ast
import copy
import difflib
import inspect
from collections.abc import Mapping

import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_experiment.xecg_encoder_motion_replication_v10 import run_arm
from scripts.experiments import run_xecg_encoder_motion016_v9 as v9

H_SECONDS = 2693.7965717150364
P_SECONDS = 1098.4113020410005
CEILING_SECONDS = 7200.0
BOOTSTRAP_SEED = 16020
BOOTSTRAP_DRAWS = 2000


def run_arm_compatibility() -> dict:  # noqa: C901 - audited AST normalization has explicit guards
    """Prove the inherited scientific loop AST differs only in identity/cost guards."""
    old_source = inspect.getsource(v9.run_arm)
    new_source = inspect.getsource(run_arm)
    old = ast.parse(old_source).body[0]
    new = copy.deepcopy(ast.parse(new_source).body[0])
    if not isinstance(old, ast.FunctionDef) or not isinstance(new, ast.FunctionDef):
        raise RuntimeError("V9/V10 arm runners are not comparable functions")
    if new.args.args[-1].arg != "cost_guard" or len(new.args.defaults) != 1:
        raise RuntimeError("Unexpected V10 arm-runner argument change")
    new.args.args.pop()
    new.args.defaults.pop()
    if "seed != 47" not in ast.unparse(new.body[1]):
        raise RuntimeError("Missing fresh seed-47 production identity guard")
    new.body.pop(1)
    removed = 0
    for node in ast.walk(new):
        if not hasattr(node, "body") or not isinstance(node.body, list):
            continue
        retained = []
        for statement in node.body:
            if isinstance(statement, ast.If) and ast.unparse(statement.test) == "cost_guard is not None":
                if ast.unparse(statement.body[0]) != "cost_guard()":
                    raise RuntimeError("Cost guard changed scientific update behavior")
                removed += 1
            else:
                retained.append(statement)
        node.body = retained
    if removed != 1 or ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
        raise RuntimeError("V10 production loop differs from V9 beyond frozen allowances")
    inherited = {
        "datasets": v9.datasets,
        "checkpoint_cpu_roundtrip": v9._verify_checkpoint,
        "sequential_final_replay": v9._replay,
        "fixed_final_probe": v9._fit_final_probe,
        "released_model_and_full_gradient_update": v9.build_model,
        "full_gradient_train_update": v9.train_update,
        "fixed_training_diagnostic": v9.fixed_diagnostic,
        "full_feature_extraction": v9.infer_features,
    }
    if any(value is None for value in inherited.values()):
        raise RuntimeError("An inherited v9 scientific primitive is unavailable")
    return {
        "scientific_training_loop_ast_identical_after_identity_and_gate_guards": True,
        "new_loop_allowances": ["seed47_and_production_identity_guard", "checkpoint_boundary_cost_guard"],
        "inherited_callables": {
            name: f"{value.__module__}.{value.__name__}" for name, value in inherited.items()
        },
        "reviewed_unified_diff": "".join(
            difflib.unified_diff(
                old_source.splitlines(keepends=True),
                new_source.splitlines(keepends=True),
                fromfile="frozen_v9.run_arm",
                tofile="v10.run_arm",
            )
        ),
    }


def cost_projection(new_preparation_seconds: float, completed_pipeline_seconds: list[float]) -> dict:
    """Apply the frozen H+V remaining-cost rule to zero, one or two finished arms."""
    if (
        new_preparation_seconds < 0
        or len(completed_pipeline_seconds) > 2
        or any(value < 0 for value in completed_pipeline_seconds)
    ):
        raise ValueError("Malformed V10 cost ledger")
    remaining = 2 - len(completed_pipeline_seconds)
    slowest = max([P_SECONDS, *completed_pipeline_seconds])
    projected = (
        H_SECONDS
        + new_preparation_seconds
        + sum(completed_pipeline_seconds)
        + 1.25 * (remaining * slowest + 300)
    )
    return {
        "historical_profile_charge_H_seconds": H_SECONDS,
        "historical_slowest_pipeline_P_seconds": P_SECONDS,
        "new_preparation_V_seconds": new_preparation_seconds,
        "completed_pipeline_seconds": completed_pipeline_seconds,
        "projected_total_seconds": projected,
        "ceiling_seconds": CEILING_SECONDS,
        "passed": projected <= CEILING_SECONDS,
    }


def _contrasts(labels: np.ndarray, logits: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Evaluate the five frozen AUROC contrasts for one training seed."""
    auc = {name: roc_auc_score(labels, values) for name, values in logits.items()}
    return {
        "D": float(auc["F_joint"] - auc["M_joint"]),
        "R": float(auc["M_refit"] - auc["F_refit"]),
        "gap_M": float(auc["M_refit"] - auc["M_joint"]),
        "gap_F": float(auc["F_refit"] - auc["F_joint"]),
        "G": float((auc["M_refit"] - auc["M_joint"]) - (auc["F_refit"] - auc["F_joint"])),
    }


def paired_seed_bootstrap(
    labels: np.ndarray,
    patients: np.ndarray,
    seed_logits: Mapping[str, Mapping[str, np.ndarray]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """Resample whole patients once per draw across both fixed optimization seeds."""
    if set(seed_logits) != {"46", "47"}:
        raise ValueError("Combined analysis requires fixed optimization seeds 46 and 47")
    required = {"M_joint", "M_refit", "F_joint", "F_refit"}
    if labels.ndim != 1 or patients.shape != labels.shape or not 0 < draws <= BOOTSTRAP_DRAWS:
        raise ValueError("Malformed development rows or bootstrap count")
    for seed in ("46", "47"):
        if set(seed_logits[seed]) != required or any(
            values.shape != labels.shape or not np.isfinite(values).all()
            for values in seed_logits[seed].values()
        ):
            raise ValueError(f"Incomplete {seed} development readouts")
    unique, inverse = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = ("D", "R", "gap_M", "gap_F", "G")
    values = {seed: {name: [] for name in names} for seed in ("46", "47", "mean")}
    invalid = 0
    for _ in range(draws):
        sampled = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(labels[sampled])) != 2:
            invalid += 1
            continue
        per_seed = {
            seed: _contrasts(
                labels[sampled], {name: logits[sampled] for name, logits in seed_logits[seed].items()}
            )
            for seed in ("46", "47")
        }
        for name in names:
            values["46"][name].append(per_seed["46"][name])
            values["47"][name].append(per_seed["47"][name])
            values["mean"][name].append((per_seed["46"][name] + per_seed["47"][name]) / 2)
    observed = {seed: _contrasts(labels, seed_logits[seed]) for seed in ("46", "47")}
    observed["mean"] = {name: (observed["46"][name] + observed["47"][name]) / 2 for name in names}
    valid = draws - invalid
    return {
        "seed": BOOTSTRAP_SEED,
        "requested": draws,
        "valid": valid,
        "single_class_invalid": invalid,
        "interval_inference_resolved": valid >= 1900,
        "contrasts": {
            seed: {
                name: {
                    "observed": observed[seed][name],
                    "interval_95": np.quantile(values[seed][name], [0.025, 0.975]).tolist()
                    if valid
                    else None,
                }
                for name in names
            }
            for seed in ("46", "47", "mean")
        },
        "seed47_decision": {
            "joint_harm_replicated": observed["47"]["D"] >= 0.005,
            "readout_gap_pattern_replicated": (
                observed["47"]["D"] >= 0.005
                and observed["47"]["R"] >= -0.002
                and observed["47"]["G"] >= 0.005
            ),
            "primary_interval_excludes_zero": (
                valid >= 1900 and bool(values["47"]["D"]) and float(np.quantile(values["47"]["D"], 0.025)) > 0
            ),
        },
    }
