"""Fixed paired-response and control-suppression objectives and factorial statistics."""

from typing import Any

import numpy as np
import torch
from torch.nn import functional

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_counterfactual_factorial_training import paired_penalty

ARMS = ("B", "P", "K", "PK")
COMPONENTS = ("bce", "hinge", "consistency", "control", "total")


def control_penalty(probabilities: torch.Tensor) -> torch.Tensor:
    """Average the stable top600 probabilities equally across four control cases.

    Parameters
    ----------
    probabilities : torch.Tensor
        Native batch-eight, twelve-lead probability maps, with gradients attached.

    Returns
    -------
    torch.Tensor
        Differentiable equal-control mean of the highest600 cells per control.
    """
    if probabilities.shape != (8, 12, 5000):
        raise ValueError("Control penalty requires the fixed native batch geometry")
    controls = probabilities[[0, 1, 4, 5]].flatten(1)
    return torch.sort(controls, dim=1, descending=True, stable=True).values[:, :600].mean(dim=1).mean()


def objective(logits: torch.Tensor, target: torch.Tensor, weight: float, arm: str) -> dict[str, torch.Tensor]:
    """Form the frozen four-arm objective from one observed-input forward.

    Parameters
    ----------
    logits, target : torch.Tensor
        Native float32 logits and Boolean exact masks for the same batch.
    weight : float
        Unchanged TRAIN-only capped positive weight.
    arm : str
        One of B, P, K or PK.

    Returns
    -------
    dict
        BCE, active paired/control terms, zeros for absent terms and total loss.
    """
    if arm not in ARMS or logits.shape != target.shape or target.dtype != torch.bool:
        raise ValueError("Unknown arm or mismatched exact target geometry")
    bce = functional.binary_cross_entropy_with_logits(
        logits, target.float(), pos_weight=logits.new_tensor(weight)
    )
    hinge = consistency = control = logits.new_zeros(())
    if arm != "B":
        probabilities = logits.sigmoid()
        if arm in {"P", "PK"}:
            hinge, consistency = paired_penalty(
                probabilities,
                target,
                torch.tensor([2, 3, 6, 7], device=logits.device),
                torch.tensor([1, 1, 5, 5], device=logits.device),
                0.2,
            )
        if arm in {"K", "PK"}:
            control = control_penalty(probabilities)
    total = bce if arm == "B" else bce + hinge + consistency + control
    return dict(zip(COMPONENTS, [bce, hinge, consistency, control, total], strict=True))


def control_top600(probabilities: np.ndarray) -> dict[str, Any]:
    """Summarize saved control probabilities without a model call.

    Parameters
    ----------
    probabilities : np.ndarray
        Sixty-four backgrounds with four native float32 maps each.

    Returns
    -------
    dict
        Per-background and mean top600 probabilities for each control and their equal mean.
    """
    if probabilities.shape != (64, 4, 12, 5000) or probabilities.dtype != np.float32:
        raise ValueError("Control summaries require the fixed fresh panel")
    rows = {}
    for position, name in [(0, "sham"), (1, "nuisance")]:
        values = probabilities[:, position].reshape(64, -1)
        top = np.sort(values, axis=1, kind="stable")[:, -600:]
        rows[name] = top.astype(np.float64).mean(axis=1)
    rows["pooled"] = (rows["sham"] + rows["nuisance"]) / 2
    return {
        name: {"record_means": values.tolist(), "mean": float(values.mean())} for name, values in rows.items()
    }


def endpoint_matrices(fits: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Collect identically ordered fresh endpoints for all twelve fixed models.

    Parameters
    ----------
    fits : list
        Fit1,2,3 records, each containing complete arm metric dictionaries.

    Returns
    -------
    dict
        Arm then endpoint arrays with shape three fits by64 backgrounds.
    """
    if [fit["fit_id"] for fit in fits] != [1, 2, 3]:
        raise ValueError("All three fixed fit identities are required in order")
    result = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for fit in fits:
            rows = fit[arm]["cases"]
            if len(rows) != 64 or any(
                row["index"] != 2000 + i or row["seed"] != 2026105000 + i for i, row in enumerate(rows)
            ):
                raise ValueError("Ordered fresh panel identities differ")
        for mode in ["focal", "repeated"]:
            for metric in ["edited_ap", "response_ap", "selectivity"]:
                result[arm][f"{mode}_{metric}"] = np.array(
                    [[row[mode][metric] for row in fit[arm]["cases"]] for fit in fits], dtype=np.float64
                )
        result[arm]["pooled_edited_ap"] = (
            result[arm]["focal_edited_ap"] + result[arm]["repeated_edited_ap"]
        ) / 2
        for mode in ["sham", "nuisance", "pooled"]:
            result[arm][f"{mode}_top600"] = np.array(
                [fit[arm]["top600_control"][mode]["record_means"] for fit in fits], dtype=np.float64
            )
    return result


def contrast_matrices(fits: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build prespecified primary and exploratory factorial contrasts.

    Parameters
    ----------
    fits : list
        Complete fixed-order twelve-model evaluation metrics.

    Returns
    -------
    tuple
        Primary contrasts and all seven signed factorial contrasts by endpoint.
    """
    values = endpoint_matrices(fits)
    primary = {}
    for mode in ["focal", "repeated", "pooled"]:
        primary[f"PK_minus_B_direct_AP_{mode}"] = (
            values["PK"][f"{mode}_edited_ap"] - values["B"][f"{mode}_edited_ap"]
        )
    for mode in ["sham", "nuisance", "pooled"]:
        primary[f"B_minus_PK_top600_{mode}"] = values["B"][f"{mode}_top600"] - values["PK"][f"{mode}_top600"]
    for mode in ["focal", "repeated"]:
        for metric in ["response_ap", "selectivity"]:
            primary[f"PK_{mode}_{metric}"] = values["PK"][f"{mode}_{metric}"]
    secondary = {}
    endpoints = [f"{mode}_edited_ap" for mode in ["focal", "repeated", "pooled"]] + [
        f"{mode}_top600" for mode in ["sham", "nuisance", "pooled"]
    ]
    for endpoint in endpoints:
        b, p, k, pk = [values[arm][endpoint] for arm in ARMS]
        contrasts = {
            "P_minus_B": p - b,
            "K_minus_B": k - b,
            "PK_minus_K": pk - k,
            "PK_minus_P": pk - p,
            "half_P_minus_B_plus_PK_minus_K": 0.5 * ((p - b) + (pk - k)),
            "half_K_minus_B_plus_PK_minus_P": 0.5 * ((k - b) + (pk - p)),
            "PK_minus_P_minus_K_plus_B": pk - p - k + b,
        }
        secondary.update({f"{name}__{endpoint}": value for name, value in contrasts.items()})
    return primary, secondary


def intervals(vectors: dict[str, np.ndarray], crossed: bool) -> dict[str, Any]:
    """Use the prospective shared fit-then-background draws for all contrasts.

    Parameters
    ----------
    vectors : dict
        Float64 arrays with shape(3,64), or(1,64) for conditional estimates.
    crossed : bool
        Whether to resample three fit rows before the shared background vector.

    Returns
    -------
    dict
        Means, linear percentile intervals and exact resampling metadata.
    """
    shape = (3, 64) if crossed else (1, 64)
    if any(value.shape != shape or not np.isfinite(value).all() for value in vectors.values()):
        raise ValueError("Invalid crossed/conditional endpoint geometry")
    seed = 2026105501 if crossed else 2026105502
    rng = np.random.default_rng(seed)
    groups = patient_groups(np.arange(64))
    samples = {key: np.empty(2000, dtype=np.float64) for key in vectors}
    for draw in range(2000):
        rows = rng.integers(0, 3, 3) if crossed else np.array([0])
        columns = patient_resample(groups, rng)
        for key, value in vectors.items():
            samples[key][draw] = value[rows[:, None], columns[None, :]].mean()
    return {
        key: {
            "mean": float(value.mean()),
            "ci_low": float(np.quantile(samples[key], 0.025, method="linear")),
            "ci_high": float(np.quantile(samples[key], 0.975, method="linear")),
            "draws": 2000,
            "seed": seed,
            "skipped_draws": 0,
            "fit_count": shape[0],
            "background_count": 64,
        }
        for key, value in vectors.items()
    }


def aggregate(fits: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the complete fifteen-gate primary and retain every factorial contrast.

    Parameters
    ----------
    fits : list
        The three fixed fits, with all four arm metrics and all64 fresh backgrounds.

    Returns
    -------
    dict
        Primary and secondary intervals, all conditional results, fit ranges and decision.
    """
    primary, secondary = contrast_matrices(fits)
    combined = intervals({**primary, **secondary}, True)
    summary = {key: combined[key] for key in primary}
    gates = {}
    for key in ["PK_minus_B_direct_AP_pooled", "B_minus_PK_top600_pooled"]:
        gates[key + "_mean"] = summary[key]["mean"] >= 0.02
        gates[key + "_lower"] = summary[key]["ci_low"] > 0
    for mode in ["focal", "repeated"]:
        gates[f"PK_minus_B_direct_AP_{mode}_guard"] = (
            summary[f"PK_minus_B_direct_AP_{mode}"]["ci_low"] > -0.01
        )
        for metric in ["response_ap", "selectivity"]:
            key = f"PK_{mode}_{metric}"
            gates[key + "_lower"] = summary[key]["ci_low"] > 0
    for index in range(3):
        gates[f"fit_{index + 1}_direct_AP_gain_positive"] = (
            float(primary["PK_minus_B_direct_AP_pooled"][index].mean()) > 0
        )
    for mode in ["sham", "nuisance"]:
        key = f"B_minus_PK_top600_{mode}"
        gates[key + "_lower"] = summary[key]["ci_low"] > 0
    if len(gates) != 15:
        raise ValueError("Primary scientific conjunction must contain fifteen gates")
    return {
        "primary": summary,
        "secondary": {key: combined[key] for key in secondary},
        "conditional_primary": [
            intervals({key: value[i : i + 1] for key, value in primary.items()}, False) for i in range(3)
        ],
        "conditional_secondary": [
            intervals({key: value[i : i + 1] for key, value in secondary.items()}, False) for i in range(3)
        ],
        "fit_mean_ranges": {
            key: {
                "fit_means": value.mean(axis=1).tolist(),
                "minimum": float(value.mean(axis=1).min()),
                "maximum": float(value.mean(axis=1).max()),
            }
            for key, value in {**primary, **secondary}.items()
        },
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "primary_fit_count": 3,
        "primary_background_count": 64,
        "uncertainty_scope": "exploratory three-fit resampling; no patient or clinical uncertainty",
    }
