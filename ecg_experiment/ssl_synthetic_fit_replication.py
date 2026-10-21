"""Fixed-fit schedules and panel scoring for prospective synthetic localization replication."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from ecg_experiment import ssl_synthetic_randomized_localization as learning


def training_schedule(seed: int) -> np.ndarray:
    """Generate the unchanged 400 complete-variant minibatches with a declared schedule seed.

    Parameters
    ----------
    seed : int
        Prospective initialization-independent schedule seed.

    Returns
    -------
    np.ndarray
        Integer case indices with shape (400,8), using all 320 TRAIN cases ten times.
    """
    rng = np.random.default_rng(seed)
    pairs = np.concatenate([rng.permutation(80).reshape(40, 2) for _ in range(10)])
    return (pairs[:, :, None] * 4 + np.arange(4)[None, None, :]).reshape(400, 8)


def score_panel(
    probabilities: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    indices: np.ndarray,
    seeds: np.ndarray,
    prior: np.ndarray,
) -> list[dict[str, Any]]:
    """Apply the frozen full-grid scoring equations to a declared 48- or 64-background panel.

    Parameters
    ----------
    probabilities, observed, support : np.ndarray
        Probability maps, observed variants and separate masks, in four-variant order.
    indices, seeds : np.ndarray
        Fixed panel case identities; no identity is a model input.
    prior : np.ndarray
        Existing TRAIN-only spatial-prior score map.

    Returns
    -------
    list
        Complete per-background AP, response, selectivity and baseline measurements.
    """
    count = len(indices)
    expected = (count, 4, 12, 5000)
    if count not in {48, 64} or probabilities.shape != expected or probabilities.dtype != np.float32:
        raise ValueError("Panel maps must match the prospective float32 48/64 geometry")
    if observed.shape != expected or support.shape != expected or support.dtype != np.bool_:
        raise ValueError("Observed/target panel geometry differs")
    if (
        len(seeds) != count
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Panel identities or finite probability range failed")
    rows = []
    for index in range(count):
        row = {"index": int(indices[index]), "seed": int(seeds[index])}
        for position, mode in [(2, "focal"), (3, "repeated")]:
            mask = support[index, position].ravel()
            if not mask.any():
                raise ValueError("A positive evaluation target is empty")
            edited = float(average_precision_score(mask, probabilities[index, position].ravel()))
            baseline = float(average_precision_score(mask, probabilities[index, 1].ravel()))
            difference = probabilities[index, position].astype(np.float64) - probabilities[index, 1].astype(
                np.float64
            )
            inside, outside = float(difference.ravel()[mask].mean()), float(difference.ravel()[~mask].mean())
            row[mode] = {
                "edited_ap": edited,
                "nuisance_ap": baseline,
                "response_ap": edited - baseline,
                "delta_inside": inside,
                "delta_outside": outside,
                "selectivity": inside - outside,
                "amplitude_ap": float(
                    average_precision_score(mask, np.abs(observed[index, position]).ravel())
                ),
                "prevalence": float(mask.mean()),
                "spatial_prior_ap": float(average_precision_score(mask, prior.ravel())),
            }
        row["mean_positive_ap"] = (row["focal"]["edited_ap"] + row["repeated"]["edited_ap"]) / 2
        rows.append(row)
    return rows


def control_scores(probabilities: np.ndarray) -> dict[str, Any]:
    """Apply unchanged descriptive control summaries to either prospective panel size.

    Parameters
    ----------
    probabilities : np.ndarray
        Saved float32 four-variant maps for 48 or 64 generated backgrounds.

    Returns
    -------
    dict
        Sham/nuisance record means, maxima and strict maximum-above-0.5 fraction.
    """
    count = probabilities.shape[0]
    if count not in {48, 64} or probabilities.shape[1:] != (4, 12, 5000):
        raise ValueError("Control scores require an admitted complete panel")
    result = {}
    for position, mode in [(0, "sham"), (1, "nuisance")]:
        values = probabilities[:, position].reshape(count, -1).astype(np.float64)
        result[mode] = {
            "mean_probability": float(values.mean()),
            "maximum_probability": float(values.max()),
            "record_maximum_gt_0p5_fraction": float(np.mean(values.max(axis=1) > 0.5)),
            "record_mean_probabilities": values.mean(axis=1).tolist(),
            "record_maximum_probabilities": values.max(axis=1).tolist(),
        }
    return result


def verify_legacy_scoring(
    probabilities: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    prior: np.ndarray,
) -> bool:
    """Require exact generic-versus-frozen metric equality on the old 48-background schema."""
    old = learning.score_maps(probabilities, observed, support)
    learning.add_spatial_prior(old, support, prior)
    new = score_panel(
        probabilities, observed, support, np.arange(80, 128), np.arange(2026103080, 2026103128), prior
    )
    return old == new and control_scores(probabilities) == learning.control_scores(probabilities)


def _contrast_vectors(fits: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Build matched fit-by-background contrasts using the unchanged full-grid endpoints.

    Parameters
    ----------
    fits : list
        Three new fits, each containing supervised/permuted fresh-panel metric rows.

    Returns
    -------
    dict
        Every old-rule contrast as a float64 array of shape (3,64).
    """
    if len(fits) not in {1, 3} or any(
        len(fit[arm]) != 64 for fit in fits for arm in ["supervised", "permuted"]
    ):
        raise ValueError("Contrasts require one conditional or three primary fits and 64 fresh seeds")
    expected_ids = list(zip(range(1000, 1064), range(2026104000, 2026104064), strict=True))
    for fit in fits:
        for arm in ["supervised", "permuted"]:
            if [(row["index"], row["seed"]) for row in fit[arm]] != expected_ids:
                raise ValueError("Crossed comparisons require identical ordered fresh background identities")
    result = {}
    for comparator in ["absolute_amplitude", "permuted"]:
        mode_values = []
        for mode in ["focal", "repeated"]:
            values = np.array(
                [
                    [
                        row[mode]["edited_ap"]
                        - (
                            other[mode]["edited_ap"]
                            if comparator == "permuted"
                            else row[mode]["amplitude_ap"]
                        )
                        for row, other in zip(fit["supervised"], fit["permuted"], strict=True)
                    ]
                    for fit in fits
                ],
                dtype=np.float64,
            )
            result[f"supervised_minus_{comparator}_{mode}"] = values
            mode_values.append(values)
        result[f"supervised_minus_{comparator}_pooled"] = np.mean(mode_values, axis=0)
    for mode in ["focal", "repeated"]:
        for metric in ["response_ap", "selectivity"]:
            result[f"supervised_{mode}_{metric}"] = np.array(
                [[row[mode][metric] for row in fit["supervised"]] for fit in fits], dtype=np.float64
            )
    return result


def contrast_vectors(fits: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Admit only the three declared new fits into the primary crossed contrasts."""
    if [fit.get("fit_id") for fit in fits] != [1, 2, 3]:
        raise ValueError("Primary fit identities must be exactly [1,2,3]; fit0 is descriptive")
    return _contrast_vectors(fits)


def conditional_fresh_summary(fit: dict[str, Any]) -> dict[str, Any]:
    """Report one fixed fit's 64-background contrasts without crossed fit resampling."""
    return bootstrap_summaries(_contrast_vectors([fit]), crossed=False)


def bootstrap_summaries(vectors: dict[str, np.ndarray], crossed: bool) -> dict[str, Any]:
    """Apply shared crossed fit/background draws or conditional single-fit background draws.

    Parameters
    ----------
    vectors : dict
        Matched contrasts with shape (3,64), or (1,64) for a conditional fit interval.
    crossed : bool
        Use the prospective three-fit crossed resampling rule when true.

    Returns
    -------
    dict
        Mean and percentile bounds for every contrast with identical draw vectors.
    """
    from ecg_experiment.intervals import patient_groups, patient_resample

    expected = (3, 64) if crossed else (1, 64)
    if any(value.shape != expected or not np.isfinite(value).all() for value in vectors.values()):
        raise ValueError("Bootstrap contrast geometry or finiteness failed")
    groups = patient_groups(np.arange(64))
    seed = 2026104501 if crossed else 2026104502
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in vectors}
    for _ in range(2000):
        fits = rng.integers(0, 3, 3) if crossed else np.array([0])
        backgrounds = patient_resample(groups, rng)
        for key, value in vectors.items():
            draws[key].append(float(value[np.ix_(fits, backgrounds)].mean()))
    summaries = {}
    for key, value in vectors.items():
        low, high = np.percentile(draws[key], [2.5, 97.5])
        summaries[key] = {
            "mean": float(value.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "draws": 2000,
            "seed": seed,
            "skipped_draws": 0,
            "unit": "fit_and_generator_seed" if crossed else "generator_seed_conditional_on_fit",
            "fit_count": expected[0],
            "background_count": 64,
        }
    return summaries


def aggregate_replication(fits: list[dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Return the frozen eighteen-gate fresh-panel replication decision without selecting a fit.

    Parameters
    ----------
    fits : list
        All three independently initialized/scheduled matched arm pairs on fresh64.

    Returns
    -------
    dict
        Crossed intervals, per-fit conditional intervals, all means/ranges and gate conjunction.
    """
    if len(fits) != 3:
        raise ValueError("The aggregate excludes fit0 and requires all three new fits")
    vectors = contrast_vectors(fits)
    contrasts = bootstrap_summaries(vectors, crossed=True)
    gates = {}
    for comparator in ["absolute_amplitude", "permuted"]:
        for mode in ["focal", "repeated"]:
            key = f"supervised_minus_{comparator}_{mode}"
            gates[key + "_guard"] = contrasts[key]["ci_low"] > -0.01
        key = f"supervised_minus_{comparator}_pooled"
        gates[key + "_mean"] = contrasts[key]["mean"] >= 0.05
        gates[key + "_lower"] = contrasts[key]["ci_low"] > 0
        for fit in range(3):
            gates[f"fit_{fit + 1}_{key}_mean_positive"] = float(vectors[key][fit].mean()) > 0
    for mode in ["focal", "repeated"]:
        for metric in ["response_ap", "selectivity"]:
            key = f"supervised_{mode}_{metric}"
            gates[key + "_lower"] = contrasts[key]["ci_low"] > 0
    per_fit = [
        bootstrap_summaries({key: value[index : index + 1] for key, value in vectors.items()}, crossed=False)
        for index in range(3)
    ]
    ranges = {
        key: {
            "fit_means": value.mean(axis=1).tolist(),
            "minimum": float(value.mean(axis=1).min()),
            "maximum": float(value.mean(axis=1).max()),
        }
        for key, value in vectors.items()
    }
    return {
        "contrasts": contrasts,
        "per_fit_contrasts": per_fit,
        "fit_mean_ranges": ranges,
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "primary_fit_count": 3,
        "primary_background_count": 64,
        "uncertainty_scope": "crossed three-fit/background resampling; no patient or clinical uncertainty",
    }


def hotspot_scores(
    probabilities: np.ndarray, support: np.ndarray, indices: np.ndarray, seeds: np.ndarray
) -> dict[str, Any]:
    """Describe unchanged top-600 support ranking and paired-background persistence without activity analysis.

    Parameters
    ----------
    probabilities, support : np.ndarray
        Saved four-variant maps and masks for one 48/64-background panel.
    indices, seeds : np.ndarray
        Fixed case identities for reporting.

    Returns
    -------
    dict
        Per-seed scalar rows and descriptive per-arm/panel means.
    """
    from ecg_experiment.ssl_synthetic_control_diagnosis import ranked_mask

    rows = []
    for index in range(len(indices)):
        selected = [ranked_mask(value)[0] for value in probabilities[index]]
        row = {
            "index": int(indices[index]),
            "seed": int(seeds[index]),
            "sham_nuisance_persistence": float(np.count_nonzero(selected[0] & selected[1]) / 600),
        }
        for position, mode in [(2, "focal"), (3, "repeated")]:
            mask = support[index, position]
            hits = int(np.count_nonzero(selected[position] & mask))
            baseline = int(np.count_nonzero(selected[1] & mask))
            row[mode] = {
                "support_precision": hits / 600,
                "support_recall": float(hits / mask.sum()),
                "matched_nuisance_support_precision": baseline / 600,
                "precision_response": hits / 600 - baseline / 600,
                "nuisance_hotspot_persistence": float(
                    np.count_nonzero(selected[position] & selected[1]) / 600
                ),
            }
        rows.append(row)
    means = {"sham_nuisance_persistence": float(np.mean([row["sham_nuisance_persistence"] for row in rows]))}
    for mode in ["focal", "repeated"]:
        means[mode] = {key: float(np.mean([row[mode][key] for row in rows])) for key in rows[0][mode]}
    return {"cases": rows, "means": means}
