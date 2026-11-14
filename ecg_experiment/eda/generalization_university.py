"""Translate historical hospital operating points into university-facing illustrations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def budget_comparison(pilot: dict[str, Any]) -> pd.DataFrame:
    """Read matched local-normal budget simulations for one-source and pooled fits."""
    names = {"ptbxl": "PTB-XL only", "pooled": "Several sources"}
    return pd.DataFrame(
        [
            {
                "Training": names[row["score"]],
                "Budget (%)": row["budget"] * 100,
                "Caught (%)": row["sensitivity_mean"] * 100,
                "Low": row["sensitivity_ci"][0] * 100,
                "High": row["sensitivity_ci"][1] * 100,
                "Normal referral rate (%)": row["rate_mean"] * 100,
            }
            for row in pilot["summary"]
            if row["score"] in names and row["encoder"] == "xecg" and row["m"] == 200
        ]
    ).sort_values(["Training", "Budget (%)"])


def workload_scenarios(
    pilot: dict[str, Any], prevalences: tuple[float, ...] = (0.01, 0.02, 0.05)
) -> pd.DataFrame:
    """Illustrate counts per 1,000 assuming SPH operating rates persist at hypothetical prevalences.

    Parameters
    ----------
    pilot : dict
        Completed Experiment 030 aggregate result; use pooled xECG, 200 local normals, 5% budget.
    prevalences : tuple of float
        Hypothetical shares of positive annotations, not measured student prevalences.

    Returns
    -------
    pandas.DataFrame
        Expected counts for all four annotation/referral outcomes in each hypothetical population.
    """
    if any(not 0 <= prevalence <= 1 for prevalence in prevalences):
        raise ValueError("Hypothetical prevalences must lie between zero and one")
    candidates = [
        row
        for row in pilot["summary"]
        if row["score"] == "pooled" and row["encoder"] == "xecg" and row["m"] == 200 and row["budget"] == 0.05
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one matching historical operating point")
    operating = candidates[0]
    sensitivity, false_referral = operating["sensitivity_mean"], operating["rate_mean"]
    return pd.DataFrame(
        [
            {"Assumed positive share": f"{prevalence:.0%}", "Outcome": outcome, "ECGs": count}
            for prevalence in prevalences
            for outcome, count in (
                ("Positive annotations caught", 1000 * prevalence * sensitivity),
                ("Positive annotations missed", 1000 * prevalence * (1 - sensitivity)),
                ("Normal annotations referred", 1000 * (1 - prevalence) * false_referral),
                ("Normal annotations not referred", 1000 * (1 - prevalence) * (1 - false_referral)),
            )
        ]
    )
