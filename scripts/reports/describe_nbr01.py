"""Post-hoc NBR-01 descriptives: territory parts, group share profiles and the self-swap paired gap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ecg_experiment.files import write_json_atomic
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.localization import nbr01_analysis as na
from ecg_experiment.localization import nbr01_features as nf
from ecg_experiment.localization.nbr01_selection import TERRITORY_LEADS
from ecg_experiment.localization.nbr01_study import OUTPUT


def territory_parts(lead_shares: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Each territory's own contrast against the other territory group."""
    result = {}
    for name in ("inferior", "anterior"):
        expected = lead_shares[:, na.lead_mask(TERRITORY_LEADS[name])].sum(axis=1)
        other = tuple(group for group in ("inferior", "anterior") if group != name)
        result[name] = na.group_contrast(expected, groups, (name,), other)
    return result


def main() -> None:
    """Write ``analyze/posthoc.json`` from the scored records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.input_root / OUTPUT / "score/scores.npz")
    groups, patients = data["group"], data["patient"]
    table = na.record_table(data["scores"], data["energy"], nf.thresholds(args.input_root))
    result = {"note": "post hoc, written after the declared results; no decision uses it"}
    maps = {"energy": table["energy_lead"]}
    for key in ("all_age_blend", "all_age_hgb", "all_age_xecg", "ptb_baseline_blend"):
        maps[key] = table[key]["lead_shares"]
        maps[key + "_swap"] = table[key]["swap_lead_shares"]
    result["territory_parts"] = {name: territory_parts(value, groups) for name, value in maps.items()}
    result["mean_lead_shares"] = {name: {group: np.round(value[groups == group].mean(axis=0), 4).tolist()
                                         for group in na.ABNORMAL + ("normal",)}
                                  for name, value in maps.items() if not name.endswith("_swap")}
    components = {"energy": table["energy_component"],
                  **{key: table[key]["component_shares"] for key in ("all_age_blend", "ptb_baseline_blend")}}
    result["mean_component_shares"] = {name: {group: np.round(value[groups == group].mean(axis=0), 4).tolist()
                                              for group in na.ABNORMAL + ("normal",)}
                                       for name, value in components.items()}
    rng = np.random.default_rng(na.SEED)
    resample = patient_groups(patients)
    gaps = {key: [] for key in ("all_age_blend", "ptb_baseline_blend")}
    for _ in range(na.DRAWS):
        rows = patient_resample(resample, rng)
        for key in gaps:
            gaps[key].append(na.territory_contrast(table[key]["lead_shares"][rows], groups[rows])
                             - na.territory_contrast(table[key]["swap_lead_shares"][rows], groups[rows]))
    result["territory_minus_swap_territory"] = {
        key: {"point": na.territory_contrast(table[key]["lead_shares"], groups)
              - na.territory_contrast(table[key]["swap_lead_shares"], groups),
              "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}
        for key, values in gaps.items()}
    flagged = table["all_age_blend"]["flagged"]
    result["all_age_blend_no_positive_lead_drop"] = int(
        (np.clip(table["all_age_blend"]["lead_drop"], 0, None).sum(axis=1) == 0).sum())
    result["all_age_blend_flagged_territory"] = int(flagged[np.isin(groups, ("inferior", "anterior"))].sum())
    write_json_atomic(args.input_root / OUTPUT / "analyze/posthoc.json", result)
    print(result["territory_parts"], result["territory_minus_swap_territory"])


if __name__ == "__main__":
    main()
