"""Matched raw/standardized linear-head objectives and CPU optimizers for xECG v8."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

N_TRAIN = 15_359
N_DEV = 1_306
WIDTH = 1_024
C = 0.01
LAMBDA = 1.0 / (N_TRAIN * C)
BATCH = 64
UPDATES_PER_EPOCH = 240
EPOCHS = 10
CLIP_NORM = 3.0
ARMS = ("A", "B", "C")


def clean_cohort_join(
    full: list[dict[str, str]],
    overlay: list[dict[str, str]],
    development: list[dict[str, str]],
    expected_id_digest: str,
) -> list[dict[str, str]]:
    """Join the frozen clean overlay in original train order and verify feature IDs."""
    clean = {row["record_id"]: row for row in overlay}
    train = [row for row in full if f"ptbxl:{row['ecg_id']}" in clean]
    if len(clean) != len(overlay) or len(full) != N_TRAIN + 1 or len(train) != N_TRAIN:
        raise ValueError("Frozen clean cohort count/uniqueness differs")
    if {row["ecg_id"] for row in full} - {row["ecg_id"] for row in train} != {"12722"}:
        raise ValueError("Frozen clean exclusion differs")
    for row in train:
        matched = clean[f"ptbxl:{row['ecg_id']}"]
        if matched["target"] != row["target"] or matched["patient_id"] != f"ptbxl:{row['patient_id']}":
            raise ValueError("Clean label or patient identity differs")
    ids = np.asarray([int(row["ecg_id"]) for row in train + development], dtype=np.int64)
    if sha256(ids.tobytes()).hexdigest() != expected_id_digest:
        raise ValueError("Feature-cache row identities/order differ")
    return train


@dataclass
class HeadState:
    """Serializable Adam state and exact next-minibatch position."""

    theta: np.ndarray
    first: np.ndarray
    second: np.ndarray
    updates: int
    epoch: int
    next_index: int
    order: np.ndarray
    rng_state: dict
    clip_events: int
    clip_factors: list[float]

    def copy(self) -> HeadState:
        """Copy parameters, moments, minibatch order and generator state."""
        return HeadState(
            self.theta.copy(),
            self.first.copy(),
            self.second.copy(),
            self.updates,
            self.epoch,
            self.next_index,
            self.order.copy(),
            deepcopy(self.rng_state),
            self.clip_events,
            list(self.clip_factors),
        )


def train_scaler(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit v7-compatible population statistics on training rows only."""
    scaler = StandardScaler().fit(np.asarray(train, dtype=np.float64))
    return scaler.mean_, scaler.scale_


def to_standardized(raw: np.ndarray, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map raw affine parameters to standardized coordinates."""
    return np.concatenate((scale * raw[:-1], [raw[-1] + mu @ raw[:-1]]))


def to_raw(standardized: np.ndarray, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map standardized affine parameters to raw coordinates."""
    weight = standardized[:-1] / scale
    return np.concatenate((weight, [standardized[-1] - mu @ weight]))


def raw_gradient(standardized_gradient: np.ndarray, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply the exact chain rule to a standardized-coordinate gradient."""
    return np.concatenate(
        (
            scale * standardized_gradient[:-1] + mu * standardized_gradient[-1],
            [standardized_gradient[-1]],
        )
    )


def objective_standardized(
    theta: np.ndarray, z: np.ndarray, labels: np.ndarray, penalty: float = LAMBDA
) -> tuple[float, np.ndarray, float, float]:
    """Return full/minibatch mean BCE, exact probe penalty and gradient."""
    logits = z @ theta[:-1] + theta[-1]
    bce = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    regularizer = float(0.5 * penalty * np.dot(theta[:-1], theta[:-1]))
    residual = (expit(logits) - labels) / len(labels)
    gradient = np.concatenate((z.T @ residual + penalty * theta[:-1], [np.sum(residual)]))
    return bce + regularizer, gradient, bce, regularizer


def objective_raw(
    theta: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    penalty: float = LAMBDA,
) -> tuple[float, np.ndarray, float, float]:
    """Evaluate the same probe objective and gradient in raw coordinates."""
    standardized = to_standardized(theta, mu, scale)
    z = (x - mu) / scale
    value, gradient, bce, regularizer = objective_standardized(standardized, z, labels, penalty)
    return value, raw_gradient(gradient, mu, scale), bce, regularizer


def initial_state(
    raw_head: np.ndarray, mu: np.ndarray, scale: np.ndarray, arm: str, seed: int, n: int
) -> HeadState:
    """Start a fresh optimizer from one common saved joint head and seeded order."""
    if arm not in ARMS:
        raise ValueError("Unknown v8 arm")
    theta = to_standardized(raw_head, mu, scale) if arm == "C" else raw_head.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return HeadState(
        theta,
        np.zeros_like(theta),
        np.zeros_like(theta),
        0,
        0,
        0,
        order,
        deepcopy(rng.bit_generator.state),
        0,
        [],
    )


def _next_batch(state: HeadState, n: int) -> np.ndarray:
    """Consume one full-coverage batch, advancing the saved RNG at epoch boundaries."""
    if state.next_index == n:
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state.rng_state)
        state.order = rng.permutation(n)
        state.rng_state = deepcopy(rng.bit_generator.state)
        state.epoch += 1
        state.next_index = 0
    end = min(state.next_index + BATCH, n)
    index = state.order[state.next_index : end]
    state.next_index = end
    return index


def _gradient(
    theta: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    arm: str,
    mu: np.ndarray,
    scale: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Compute the specified minibatch head gradient before clipping."""
    if arm == "C":
        return objective_standardized(theta, features, labels, penalty)[1]
    if arm == "B":
        return objective_raw(theta, features, labels, mu, scale, penalty)[1]
    logits = features @ theta[:-1] + theta[-1]
    residual = (expit(logits) - labels) / len(labels)
    return np.concatenate((features.T @ residual, [np.sum(residual)]))


def step(
    state: HeadState,
    raw_features: np.ndarray,
    standardized_features: np.ndarray,
    labels: np.ndarray,
    arm: str,
    mu: np.ndarray,
    scale: np.ndarray,
    penalty: float = LAMBDA,
) -> tuple[np.ndarray, float]:
    """Advance exactly one Adam/AdamW update with raw-norm gradient clipping."""
    n = len(labels)
    index = _next_batch(state, n)
    features = standardized_features if arm == "C" else raw_features
    gradient = _gradient(
        state.theta, features[index], labels[index], arm, mu, scale, 0.0 if arm == "A" else penalty
    )
    raw_grad = raw_gradient(gradient, mu, scale) if arm == "C" else gradient
    norm = float(np.linalg.norm(raw_grad))
    factor = min(1.0, CLIP_NORM / norm) if norm > 0 else 1.0
    gradient *= factor
    if factor < 1:
        state.clip_events += 1
    state.clip_factors.append(factor)
    state.updates += 1
    state.first = 0.9 * state.first + 0.1 * gradient
    state.second = 0.999 * state.second + 0.001 * gradient * gradient
    corrected_first = state.first / (1.0 - 0.9**state.updates)
    corrected_second = state.second / (1.0 - 0.999**state.updates)
    lr = 0.001 * (state.updates - 1) / 240 if state.updates <= 240 else 0.001
    if arm == "A":
        state.theta *= 1.0 - lr * 0.1  # decoupled weight and bias decay
    state.theta -= lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return index, factor


def diagnostics(
    state: HeadState,
    arm: str,
    raw_features: np.ndarray,
    standardized_features: np.ndarray,
    labels: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    initial_raw: np.ndarray,
    development: np.ndarray,
) -> dict:
    """Measure the common full-training probe objective and head movement."""
    standardized = state.theta if arm == "C" else to_standardized(state.theta, mu, scale)
    value, gradient, bce, regularizer = objective_standardized(standardized, standardized_features, labels)
    raw = to_raw(standardized, mu, scale)
    train_shift = raw_features @ (raw[:-1] - initial_raw[:-1]) + raw[-1] - initial_raw[-1]
    dev_shift = development @ (raw[:-1] - initial_raw[:-1]) + raw[-1] - initial_raw[-1]
    return {
        "updates": state.updates,
        "epoch": state.epoch + (state.next_index == len(labels)),
        "exposures": state.epoch * len(labels) + state.next_index,
        "full_train_bce": bce,
        "probe_penalty": regularizer,
        "J": value,
        "standardized_gradient_inf_norm": float(np.max(np.abs(gradient))),
        "raw_weight_norm": float(np.linalg.norm(raw[:-1])),
        "standardized_weight_norm": float(np.linalg.norm(standardized[:-1])),
        "mean_absolute_train_logit_shift": float(np.mean(np.abs(train_shift))),
        "mean_absolute_development_logit_shift": float(np.mean(np.abs(dev_shift))),
        "clipped_updates": state.clip_events,
        "clipping_frequency": state.clip_events / state.updates if state.updates else 0.0,
        "mean_clipping_factor": float(np.mean(state.clip_factors)) if state.clip_factors else 1.0,
    }


def fit_reference(
    z: np.ndarray,
    labels: np.ndarray,
    start: np.ndarray,
    maxiter: int = 3000,
) -> tuple[np.ndarray, dict]:
    """Solve the frozen full-training probe objective with L-BFGS-B."""

    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _, _ = objective_standardized(theta, z, labels)
        return value, gradient

    result = minimize(
        fun,
        start,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-8, "ftol": 64 * np.finfo(float).eps, "maxls": 50},
    )
    value, gradient, bce, regularizer = objective_standardized(result.x, z, labels)
    return result.x, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "J": value,
        "bce": bce,
        "penalty": regularizer,
        "standardized_gradient_inf_norm": float(np.max(np.abs(gradient))),
    }


def paired_bootstrap(
    labels: np.ndarray,
    patients: np.ndarray,
    logits: dict[str, np.ndarray],
    contrasts: dict[str, tuple[str, str] | tuple[tuple[str, str], ...]],
    seed: int = 16018,
    draws: int = 2000,
) -> dict:
    """Draw paired whole patients once; average per-seed contrasts within draws."""
    unique, inverse = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    normalized = {name: (pairs,) if isinstance(pairs[0], str) else pairs for name, pairs in contrasts.items()}
    samples = {name: [] for name in contrasts}
    invalid = 0
    for _ in range(draws):
        index = np.concatenate([groups[position] for position in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(labels[index])) != 2:
            invalid += 1
            continue
        auc = {name: roc_auc_score(labels[index], values[index]) for name, values in logits.items()}
        for name, pairs in normalized.items():
            samples[name].append(float(np.mean([auc[left] - auc[right] for left, right in pairs])))
    return {
        "seed": seed,
        "requested": draws,
        "valid": draws - invalid,
        "single_class_invalid": invalid,
        "contrasts": {
            name: {
                "observed": float(
                    np.mean(
                        [
                            roc_auc_score(labels, logits[left]) - roc_auc_score(labels, logits[right])
                            for left, right in pairs
                        ]
                    )
                ),
                "interval_95": np.quantile(samples[name], [0.025, 0.975]).tolist(),
            }
            for name, pairs in normalized.items()
        },
    }
