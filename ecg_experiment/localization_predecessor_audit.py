"""Exact read-only reconstruction of Experiment 042 aggregate localization scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file
from ecg_experiment.fragment_localization import premature_windows, r_peaks
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.lead_wave_maps import ANTERIOR, INFERIOR, UnitMap, ecg_score, premature_hit, top_lead


def reproduce_042(root: Path) -> dict:
    """
    Recompute baseline detection, lead, focal and mark metrics from audited live maps.

    Parameters
    ----------
    root : Path
        Current repository root with shared local predecessor outputs and PTB-XL data.

    Returns
    -------
    dict
        Exact reproduced metrics, input receipt identity and largest numerical discrepancy.
    """
    directory = root / "outputs/experiment042_lead_wave_maps_v1"
    receipt = json.loads((directory / "result.json").read_text())
    archive = directory / "unit_scores.npz"
    if sha256_file(archive) != receipt["outputs_sha256"]["unit_scores.npz"]:
        raise ValueError("Predecessor archive hash mismatch")
    with np.load(archive) as saved:
        ids, offsets = saved["ecg_ids"], saved["U_B_offsets"]
        fields = ("scores", "leads", "starts", "ends")
        maps = {
            int(ecg_id): UnitMap(
                *(saved[f"U_B_{field}"][offsets[index] : offsets[index + 1]] for field in fields)
            )
            for index, ecg_id in enumerate(ids)
        }
    development = cohorts(ptb_table())["development"]
    if not np.array_equal(ids, development["ecg_id"].to_numpy()):
        raise ValueError("Predecessor development order differs")
    caches = (
        root / "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
        root / "outputs/experiment016_xecg_probe_finetune/features/ecg_ids.npy",
        root / "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
    )
    common = set.intersection(*(set(np.load(path).astype(int)) for path in caches))
    evaluation = development[
        development["original"] & development["standard"].notna() & development["ecg_id"].isin(common)
    ]
    labels = evaluation["standard"].to_numpy(int)
    score = np.array([ecg_score(maps[i]) for i in evaluation["ecg_id"]])
    metadata, statements = load_metadata(), load_statements()
    diagnostic = statements[statements["diagnostic"] == 1]
    subclass, superclass = (
        diagnostic["diagnostic_subclass"].to_dict(),
        diagnostic["diagnostic_class"].to_dict(),
    )
    codes = metadata["scp_codes"].apply(set)
    subclasses = codes.map(lambda found: {subclass[c] for c in found if c in subclass})
    superclasses = codes.map(lambda found: {superclass[c] for c in found if c in superclass})
    location = {
        "anterior": set(
            subclasses.index[
                subclasses.map(lambda found: "AMI" in found and not found & {"IMI", "LMI", "PMI"})
            ]
        ),
        "inferior": set(
            subclasses.index[
                subclasses.map(lambda found: "IMI" in found and not found & {"AMI", "LMI", "PMI"})
            ]
        ),
    }
    benign = {
        i
        for i, found in codes.items()
        if superclasses[i] == {"NORM"}
        and found <= {"NORM", "SR", "SBRAD", "SARRH"}
        and found & {"SBRAD", "SARRH"}
    }
    found = {"auroc": float(roc_auc_score(labels, score)), "lead_contrast": {}}
    for name, leads, first, second in (
        ("anterior", ANTERIOR, "anterior", "inferior"),
        ("inferior", INFERIOR, "inferior", "anterior"),
    ):
        first_ids = [i for i in ids if i in location[first]]
        second_ids = [i for i in ids if i in location[second]]
        found["lead_contrast"][name] = float(
            np.mean([top_lead(maps[i]) in leads for i in first_ids])
            - np.mean([top_lead(maps[i]) in leads for i in second_ids])
        )
    pvc_ids = set(codes.index[codes.map(lambda found: "PVC" in found)])
    pvc = development[development["ecg_id"].isin(pvc_ids)]
    pairs = []
    for ecg_id, stem in zip(pvc["ecg_id"], pvc["filename_hr"], strict=True):
        peaks = r_peaks(read_ptb_float64(stem), 500)
        windows = premature_windows(peaks, 500)
        if len(peaks) >= 4 and windows:
            pairs.append(premature_hit(maps[ecg_id], windows))
    values = np.array(pairs)
    found.update(
        {
            "pvc_records": len(values),
            "hit_rate": float(values[:, 0].mean()),
            "chance_rate": float(values[:, 1].mean()),
            "hit_minus_chance": float((values[:, 0] - values[:, 1]).mean()),
        }
    )
    threshold = receipt["thresholds"]["U_B"]
    groups = {
        "normal": evaluation.loc[labels == 0, "ecg_id"].tolist(),
        "positive": evaluation.loc[labels == 1, "ecg_id"].tolist(),
        "benign": [i for i in ids if i in benign],
    }
    found["any_red"] = {
        name: float(np.mean([ecg_score(maps[i]) > threshold for i in chosen]))
        for name, chosen in groups.items()
    }
    differences = metric_discrepancies(found, receipt["maps"]["U_B"])
    if max(differences) != 0 or found["pvc_records"] != 73:
        raise ValueError(f"Predecessor exact aggregate reproduction failed: {max(differences)}")
    return {
        "metrics": found,
        "max_absolute_metric_difference": max(differences),
        "score_archive_sha256": sha256_file(archive),
        "receipt_sha256": sha256_file(directory / "result.json"),
    }


def metric_discrepancies(found: dict, prior: dict) -> list[float]:
    """
    Compare every prespecified reproduced scalar against its predecessor receipt.

    Parameters
    ----------
    found : dict
        Recomputed scalar metrics.
    prior : dict
        Frozen predecessor aggregate receipt.

    Returns
    -------
    list[float]
        Absolute scalar discrepancies, including detection, leads, PVC and marks.
    """
    differences = [abs(found[name] - prior[name]) for name in ("auroc", "hit_rate", "chance_rate")]
    differences.append(abs(found["hit_minus_chance"] - prior["hit_minus_chance"]["value"]))
    differences.extend(
        abs(found["lead_contrast"][name] - prior["lead_contrast"][name]["value"])
        for name in ("anterior", "inferior")
    )
    differences.extend(abs(found["any_red"][name] - prior["any_red"][name]) for name in found["any_red"])
    return differences
