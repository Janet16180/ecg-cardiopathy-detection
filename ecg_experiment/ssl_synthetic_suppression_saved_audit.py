"""Independent objective and saved-map arithmetic for paired suppression."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_control_diagnosis_audit import top_selection
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require

ARMS = ("B", "P", "K", "PK")


def objective_components(logits: np.ndarray, target: np.ndarray, weight: float, arm: str) -> np.ndarray:
    """Recompute one captured pre-update objective independently in float64.

    Parameters
    ----------
    logits, target : np.ndarray
        Saved native logits and independently reconstructed Boolean batch targets.
    weight : float
        Unchanged training-only positive class weight.
    arm : str
        Frozen factorial objective identity.

    Returns
    -------
    np.ndarray
        BCE, active hinge/consistency/control terms and total, in declared order.
    """
    require(arm in ARMS and logits.shape == target.shape == (8, 12, 5000), "objective batch geometry")
    require(
        logits.dtype == np.float32 and target.dtype == bool and np.isfinite(logits).all(),
        "objective native values",
    )
    values = logits.astype(np.float64)
    bce = float(np.where(target, weight * np.logaddexp(0, -values), np.logaddexp(0, values)).mean())
    probability = np.empty_like(values)
    nonnegative = values >= 0
    probability[nonnegative] = 1 / (1 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    probability[~nonnegative] = exponential / (1 + exponential)
    hinge, consistency, control = 0.0, 0.0, 0.0
    if arm in ("P", "PK"):
        hinges, consistencies = [], []
        for positive, nuisance in ((2, 1), (3, 1), (6, 5), (7, 5)):
            mask = target[positive]
            require(
                mask.any() and (~mask).any() and not target[nuisance].any(), "nonempty matched pair supports"
            )
            difference = probability[positive] - probability[nuisance]
            hinges.append(float(np.maximum(0, 0.2 - difference[mask]).mean()))
            consistencies.append(float(np.abs(difference[~mask]).mean()))
        hinge, consistency = float(np.mean(hinges)), float(np.mean(consistencies))
    if arm in ("K", "PK"):
        means = []
        for position in (0, 1, 4, 5):
            require(not target[position].any(), "zero control loss target")
            selected, _ = top_selection(probability[position])
            means.append(float(probability[position][selected].mean()))
        control = float(np.mean(means))
    return np.array([bce, hinge, consistency, control, bce + hinge + consistency + control], dtype=np.float64)


def verify_components(components: np.ndarray, losses: np.ndarray, arm: str) -> None:
    """Validate complete logged active-term arithmetic without asserting a neural trajectory.

    Parameters
    ----------
    components : np.ndarray
        Native float64 saved rows in BCE/H/C/K/total order.
    losses : np.ndarray
        Native float64 total-loss vector.
    arm : str
        Objective identity determining exactly absent terms.
    """
    require(
        components.dtype == losses.dtype == np.float64 and components.shape == (len(losses), 5),
        "component geometry",
    )
    require(np.isfinite(components).all() and (components >= 0).all(), "finite nonnegative loss components")
    absent = {"B": (1, 2, 3), "P": (3,), "K": (1, 2), "PK": ()}[arm]
    require(not components[:, absent].any(), "exactly absent objective components")
    require(np.array_equal(components[:, 4], losses), "native saved losses equal component totals")
    require(
        bool(np.all(np.abs(components[:, :4].sum(axis=1) - components[:, 4]) <= 1e-6)),
        "component total tolerance",
    )


def control_top_values(probability: np.ndarray) -> dict[str, Any]:
    """Independently select ranked control values and equal-control summaries.

    Parameters
    ----------
    probability : np.ndarray
        Full native fresh64 map bank.

    Returns
    -------
    dict
        Per-background and aggregate sham, nuisance and pooled top600 means.
    """
    require(
        probability.shape == (64, 4, 12, 5000) and probability.dtype == np.float32, "top600 panel geometry"
    )
    rows = {}
    for position, name in ((0, "sham"), (1, "nuisance")):
        values = []
        for index in range(64):
            scores = probability[index, position]
            selected, _ = top_selection(scores)
            values.append(float(scores[selected].astype(np.float64).mean()))
        rows[name] = np.array(values, dtype=np.float64)
    rows["pooled"] = (rows["sham"] + rows["nuisance"]) / 2
    return {
        name: {"record_means": value.tolist(), "mean": float(value.mean())} for name, value in rows.items()
    }


def endpoint_values(fits: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Gather fixed-order full-grid endpoints from independently reconstructed metrics.

    Parameters
    ----------
    fits : list
        Three fit identities with all four complete arm metric records.

    Returns
    -------
    dict
        Arm and endpoint to three-fit by sixty-four-background arrays.
    """
    require([fit["fit_id"] for fit in fits] == [1, 2, 3], "all three fixed fit identities")
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for fit in fits:
            rows = fit[arm]["cases"]
            require(
                len(rows) == 64
                and all(
                    row["index"] == 2000 + index and row["seed"] == 2026105000 + index
                    for index, row in enumerate(rows)
                ),
                "ordered fresh panel identities",
            )
        for mode in ("focal", "repeated"):
            for measure in ("edited_ap", "response_ap", "selectivity"):
                result[arm][f"{mode}_{measure}"] = np.array(
                    [[row[mode][measure] for row in fit[arm]["cases"]] for fit in fits], dtype=np.float64
                )
        result[arm]["pooled_edited_ap"] = (
            result[arm]["focal_edited_ap"] + result[arm]["repeated_edited_ap"]
        ) / 2
        for mode in ("sham", "nuisance", "pooled"):
            result[arm][f"{mode}_top600"] = np.array(
                [fit[arm]["top600_control"][mode]["record_means"] for fit in fits], dtype=np.float64
            )
    return result


def contrast_values(fits: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Independently construct primary and explicitly signed exploratory factorial effects.

    Parameters
    ----------
    fits : list
        All four objective metrics for every predeclared fit.

    Returns
    -------
    tuple
        Ten primary matrices and forty-two factorial endpoint matrices.
    """
    values = endpoint_values(fits)
    primary = {}
    for mode in ("focal", "repeated", "pooled"):
        primary[f"PK_minus_B_direct_AP_{mode}"] = (
            values["PK"][f"{mode}_edited_ap"] - values["B"][f"{mode}_edited_ap"]
        )
    for mode in ("sham", "nuisance", "pooled"):
        primary[f"B_minus_PK_top600_{mode}"] = values["B"][f"{mode}_top600"] - values["PK"][f"{mode}_top600"]
    for mode in ("focal", "repeated"):
        for measure in ("response_ap", "selectivity"):
            primary[f"PK_{mode}_{measure}"] = values["PK"][f"{mode}_{measure}"]
    secondary = {}
    for endpoint in (
        "focal_edited_ap",
        "repeated_edited_ap",
        "pooled_edited_ap",
        "sham_top600",
        "nuisance_top600",
        "pooled_top600",
    ):
        baseline, paired, suppress, both = (values[arm][endpoint] for arm in ARMS)
        effects = {
            "P_minus_B": paired - baseline,
            "K_minus_B": suppress - baseline,
            "PK_minus_K": both - suppress,
            "PK_minus_P": both - paired,
            "half_P_minus_B_plus_PK_minus_K": ((paired - baseline) + (both - suppress)) / 2,
            "half_K_minus_B_plus_PK_minus_P": ((suppress - baseline) + (both - paired)) / 2,
            "PK_minus_P_minus_K_plus_B": both - paired - suppress + baseline,
        }
        for name, value in effects.items():
            secondary[f"{name}__{endpoint}"] = value
    return primary, secondary


def interval_values(vectors: dict[str, np.ndarray], crossed: bool) -> dict[str, Any]:
    """Resample fit rows and one shared background vector using the prospective seeds.

    Parameters
    ----------
    vectors : dict
        Matched endpoint contrast arrays with common geometry.
    crossed : bool
        Sample three fits or condition on one fixed fit.

    Returns
    -------
    dict
        Complete shared-draw interval metadata.
    """
    shape = (3, 64) if crossed else (1, 64)
    require(
        all(value.shape == shape and np.isfinite(value).all() for value in vectors.values()),
        "suppression interval geometry",
    )
    seed = 2026105501 if crossed else 2026105502
    rng = np.random.default_rng(seed)
    groups = patient_groups(np.arange(64))
    samples = {name: np.empty(2000) for name in vectors}
    for draw in range(2000):
        rows = rng.integers(0, 3, 3) if crossed else np.array([0])
        columns = patient_resample(groups, rng)
        for name, values in vectors.items():
            samples[name][draw] = values[np.ix_(rows, columns)].mean()
    result = {}
    for name, values in vectors.items():
        low, high = np.percentile(samples[name], [2.5, 97.5])
        result[name] = {
            "mean": float(values.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "draws": 2000,
            "seed": seed,
            "skipped_draws": 0,
            "fit_count": shape[0],
            "background_count": 64,
        }
    return result


def suppression_summary(fits: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently evaluate all fifteen primary gates without outcome-dependent selection.

    Parameters
    ----------
    fits : list
        Three complete B/P/K/PK records on the reserved fresh panel.

    Returns
    -------
    dict
        All crossed/conditional effects, ranges and the prospective decision.
    """
    primary, secondary = contrast_values(fits)
    intervals = interval_values({**primary, **secondary}, True)
    gates = {}
    for name in ("PK_minus_B_direct_AP_pooled", "B_minus_PK_top600_pooled"):
        gates[name + "_mean"] = intervals[name]["mean"] >= 0.02
        gates[name + "_lower"] = intervals[name]["ci_low"] > 0
    for mode in ("focal", "repeated"):
        gates[f"PK_minus_B_direct_AP_{mode}_guard"] = (
            intervals[f"PK_minus_B_direct_AP_{mode}"]["ci_low"] > -0.01
        )
        for measure in ("response_ap", "selectivity"):
            name = f"PK_{mode}_{measure}"
            gates[name + "_lower"] = intervals[name]["ci_low"] > 0
    for index in range(3):
        gates[f"fit_{index + 1}_direct_AP_gain_positive"] = (
            float(primary["PK_minus_B_direct_AP_pooled"][index].mean()) > 0
        )
    for mode in ("sham", "nuisance"):
        name = f"B_minus_PK_top600_{mode}"
        gates[name + "_lower"] = intervals[name]["ci_low"] > 0
    require(len(gates) == 15, "fifteen primary gates")
    ranges = {}
    for name, values in {**primary, **secondary}.items():
        means = values.mean(axis=1)
        ranges[name] = {
            "fit_means": means.tolist(),
            "minimum": float(means.min()),
            "maximum": float(means.max()),
        }
    return {
        "primary": {name: intervals[name] for name in primary},
        "secondary": {name: intervals[name] for name in secondary},
        "conditional_primary": [
            interval_values({name: values[index : index + 1] for name, values in primary.items()}, False)
            for index in range(3)
        ],
        "conditional_secondary": [
            interval_values({name: values[index : index + 1] for name, values in secondary.items()}, False)
            for index in range(3)
        ],
        "fit_mean_ranges": ranges,
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "primary_fit_count": 3,
        "primary_background_count": 64,
        "uncertainty_scope": ("exploratory three-fit resampling; no patient or clinical uncertainty"),
    }
