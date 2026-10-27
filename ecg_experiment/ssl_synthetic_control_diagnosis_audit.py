"""Independent saved-map rank and activity diagnostic arithmetic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require

MODES = ("sham", "nuisance", "focal", "repeated")
TOP_CELLS = 600


def top_mask(scores: np.ndarray) -> np.ndarray:
    """Select exactly 600 cells by descending score and ascending C-flat index.

    Parameters
    ----------
    scores : np.ndarray
        Finite twelve-lead map.

    Returns
    -------
    np.ndarray
        Boolean lead-time selection, independent of the production ranking helper.
    """
    return top_selection(scores)[0]


def top_selection(scores: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Select cells lexicographically and independently retain complete cutoff tie counts.

    Parameters
    ----------
    scores : np.ndarray
        Finite canonical lead-time score array.

    Returns
    -------
    tuple
        Exact selected mask and global cutoff statistics.
    """
    require(scores.shape == (12, 5000) and bool(np.isfinite(scores).all()), "rank map geometry/finite")
    values = np.asarray(scores).ravel(order="C").astype(np.float64)
    ranked = np.lexsort((np.arange(values.size, dtype=np.int64), -values))
    selection = np.zeros(values.size, dtype=bool)
    selection[ranked[:TOP_CELLS]] = True
    cutoff = float(np.min(values[selection]))
    above = int(np.count_nonzero(values > cutoff))
    equal = int(np.count_nonzero(values == cutoff))
    return selection.reshape(12, 5000), {
        "score": cutoff,
        "strictly_above_count": above,
        "equal_cutoff_count": equal,
        "selected_tied_count": TOP_CELLS - above,
    }


def activity_mask(observed: np.ndarray) -> np.ndarray:
    """Independently form strict per-lead upper-decile smoothed derivative energy.

    Parameters
    ----------
    observed : np.ndarray
        Canonical observed signal in dimensionless study units.

    Returns
    -------
    np.ndarray
        Boolean activity mask over all canonical lead-time cells.
    """
    require(observed.shape == (12, 5000) and bool(np.isfinite(observed).all()), "signal geometry/finite")
    values = observed.astype(np.float64)
    derivative = np.zeros_like(values)
    derivative[:, 1:] = values[:, 1:] - values[:, :-1]
    energy = np.array([np.convolve(row * row, np.full(20, 0.05), mode="same") for row in derivative])
    cutoff = np.quantile(energy, 0.9, axis=1, method="linear")
    return energy > cutoff[:, None]


def shift_for_index(index: int) -> int:
    """Regenerate the common frozen circular shift for one evaluation seed.

    Parameters
    ----------
    index : int
        Global background index, 80 through 127.

    Returns
    -------
    int
        Shift in samples, shared across leads, variants and arms.
    """
    require(80 <= index <= 127, "evaluation index only")
    return int(np.random.default_rng(2026103600 + index).integers(500, 4501))


def overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Return selected-cell coincidence normalized by the fixed 600-cell budget.

    Parameters
    ----------
    first, second : np.ndarray
        Boolean maps over the same full grid.

    Returns
    -------
    float
        Intersection count divided by 600, not Jaccard.
    """
    return float(np.count_nonzero(first & second) / TOP_CELLS)


def record_values(
    probability: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    prior: np.ndarray,
    index: int,
    arm: str,
) -> dict[str, Any]:
    """Recompute complete per-seed rank, activity, prior and support diagnostics.

    Parameters
    ----------
    probability, observed, support : np.ndarray
        Four saved variants, observed signals and target masks for one held-out seed.
    prior : np.ndarray
        Frozen training-derived spatial-prior score map.
    index : int
        Global evaluation index.
    arm : str
        Frozen supervised or permuted model name.

    Returns
    -------
    dict[str, Any]
        Independently computed per-seed diagnostic row.
    """
    require(probability.dtype == np.float32 and probability.shape == (4, 12, 5000), "native map geometry")
    require(bool(((probability >= 0) & (probability <= 1)).all()), "probability range")
    require(observed.dtype == np.float32 and observed.shape == probability.shape, "observed geometry")
    require(support.dtype == bool and support.shape == probability.shape, "mask geometry")
    shift = shift_for_index(index)
    prior_selection = top_mask(prior)
    shifted_prior = np.roll(prior_selection, shift, axis=1)
    selections = [top_selection(values) for values in probability]
    selected = [item[0] for item in selections]
    variants, positives = {}, {}
    for position, mode in enumerate(MODES):
        activity = activity_mask(observed[position])
        native_fraction = overlap(selected[position], activity)
        shifted_fraction = overlap(selected[position], np.roll(activity, shift, axis=1))
        prior_fraction = overlap(selected[position], prior_selection)
        shifted_prior_fraction = overlap(selected[position], shifted_prior)
        lead_counts = selected[position].sum(axis=1)
        variants[mode] = {
            "activity_precision": native_fraction,
            "shifted_activity_precision": shifted_fraction,
            "activity_enrichment": native_fraction - shifted_fraction,
            "activity_occupancy": float(activity.mean()),
            "activity_occupancy_per_lead": activity.mean(axis=1).tolist(),
            "lead_counts": lead_counts.tolist(),
            "lead_fractions": (lead_counts / TOP_CELLS).tolist(),
            "prior_precision": prior_fraction,
            "shifted_prior_precision": shifted_prior_fraction,
            "prior_enrichment": prior_fraction - shifted_prior_fraction,
            "cutoff": selections[position][1],
        }
        if position >= 2:
            mask = support[position]
            require(bool(mask.any()), "nonempty positive support")
            hits = np.count_nonzero(selected[position] & mask)
            baseline_hits = np.count_nonzero(selected[1] & mask)
            positives[mode] = {
                "target_precision": float(hits / TOP_CELLS),
                "target_recall": float(hits / mask.sum()),
                "nuisance_target_precision": float(baseline_hits / TOP_CELLS),
                "precision_response": float((hits - baseline_hits) / TOP_CELLS),
                "nuisance_hotspot_persistence": overlap(selected[position], selected[1]),
            }
    return {
        "variants": variants,
        "control_persistence": overlap(selected[0], selected[1]),
        "positives": positives,
        "primary_activity_enrichment": (
            variants["sham"]["activity_enrichment"] + variants["nuisance"]["activity_enrichment"]
        )
        / 2,
    }


def seed_values(
    probability: dict[str, np.ndarray],
    observed: np.ndarray,
    support: np.ndarray,
    prior: np.ndarray,
    index: int,
) -> dict[str, Any]:
    """Combine both arm rows using the same prospective reference shift.

    Parameters
    ----------
    probability : dict
        Both arms' four-variant probability arrays for this seed.
    observed, support, prior : np.ndarray
        Saved signals, masks and training-derived prior.
    index : int
        Global held-out background index.

    Returns
    -------
    dict[str, Any]
        Complete independently reconstructed per-seed diagnostic row.
    """
    return {
        "index": index,
        "seed": 2026103000 + index,
        "common_shift": shift_for_index(index),
        "arms": {
            arm: record_values(probability[arm], observed, support, prior, index, arm)
            for arm in ("supervised", "permuted")
        },
    }


def seed_interval(values: np.ndarray) -> dict[str, Any]:
    """Compute a descriptive paired whole-seed interval with the frozen shared draws.

    Parameters
    ----------
    values : np.ndarray
        One value per 48 evaluation generator seeds.

    Returns
    -------
    dict[str, Any]
        Mean and fixed percentile seed-bootstrap interval.
    """
    require(values.shape == (48,) and bool(np.isfinite(values).all()), "48 finite seed values")
    groups = patient_groups(np.arange(48))
    generator = np.random.default_rng(2026103701)
    draws = [float(values[patient_resample(groups, generator)].mean()) for _ in range(2000)]
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 2026103701,
        "skipped_draws": 0,
        "unit": "generator_seed",
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently summarize the three primary estimates and all descriptive means.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Complete ordered 48-seed diagnostic rows.

    Returns
    -------
    dict[str, Any]
        Primary intervals and per-arm descriptive means, with no diagnostic PASS.
    """
    require(len(rows) == 48, "48 complete diagnostic seeds")
    primary, secondary, vectors = {}, {}, {}
    for arm in ("supervised", "permuted"):
        arms = [row["arms"][arm] for row in rows]
        vectors[arm] = np.array([value["primary_activity_enrichment"] for value in arms])
        primary[arm] = seed_interval(vectors[arm])
        variants, positives = {}, {}
        for mode in MODES:
            fields = [key for key in arms[0]["variants"][mode] if key != "cutoff"]
            variants[mode] = {
                key: np.mean(np.array([value["variants"][mode][key] for value in arms]), axis=0).tolist()
                for key in fields
            }
        for mode in ("focal", "repeated"):
            positives[mode] = {
                key: float(np.mean([value["positives"][mode][key] for value in arms]))
                for key in arms[0]["positives"][mode]
            }
        secondary[arm] = {
            "variants": variants,
            "positives": positives,
            "control_persistence": float(np.mean([value["control_persistence"] for value in arms])),
        }
    primary["supervised_minus_permuted"] = seed_interval(vectors["supervised"] - vectors["permuted"])
    return {"primary": primary, "secondary_means": secondary}
