"""Aggregate existing development experiments for the generalization notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecg_experiment.paths import from_stored, to_stored

EVIDENCE = {
    "transfer": "outputs/experiment022b_multisource_readout_v1/result.json",
    "labels": "outputs/experiment025b_label_efficiency_multisource_v1/result.json",
    "pilot": "outputs/experiment030_referral_budget_v1/result.json",
}
EXPECTED_HASHES = {
    "transfer": "217e866340071e66844bcdfd4bc3700bee151f1e329cf5e2b13842ecbab4865c",
    "labels": "8c28968dd70518fd234118965903af22c478e1c09a1f0744d3f64bdb0eb74da6",
    "pilot": "b20c7b1c26e2e5af307bee236fcad62b2ec241bafbcd844178ddc5c8cc1c0056",
}
ENCODERS = {"xecg": "xECG", "jepa": "ECG-JEPA", "cpc": "CPC"}
FAMILIES = {
    "chapman_ningbo": "Chapman / Ningbo",
    "georgia": "Georgia",
    "cpsc": "CPSC / CPSC-Extra",
}


def load_evidence() -> dict[str, Any]:
    """Load completed aggregate results, refusing runs that read closed test sets."""
    evidence = {}
    for name, stored in EVIDENCE.items():
        raw = from_stored(stored).read_bytes()
        if hashlib.sha256(raw).hexdigest() != EXPECTED_HASHES[name]:
            raise ValueError(f"Evidence hash changed: {stored}")
        result = json.loads(raw)
        if result["status"] != "complete":
            raise ValueError(f"Incomplete evidence: {stored}")
        for flag in ("challenge_test_read", "ptbxl_test_read"):
            if result[flag]:
                raise ValueError(f"Closed data used: {stored}: {flag}")
        evidence[name] = result
    return evidence


def provenance() -> pd.DataFrame:
    """Return repository-relative evidence paths and hashes of their current bytes."""
    return pd.DataFrame(
        [
            {
                "Evidence": name,
                "File": stored,
                "SHA-256": hashlib.sha256(from_stored(stored).read_bytes()).hexdigest(),
            }
            for name, stored in EVIDENCE.items()
        ]
    )


def verify_saved_scores(transfer: dict[str, Any]) -> pd.DataFrame:
    """Recompute SPH AUROC locally and verify prediction hashes and predecessor identity."""
    stored = "outputs/experiment022b_multisource_readout_v1/predictions.npz"
    path = from_stored(stored)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != transfer["outputs_sha256"]["predictions.npz"]:
        raise ValueError("Saved prediction hash differs from its receipt")
    rows = []
    with np.load(path, allow_pickle=False) as predictions:
        for encoder, display in ENCODERS.items():
            for arm in ("ptbxl", "pooled"):
                actual = roc_auc_score(predictions["sph_labels"], predictions[f"sph_{encoder}_{arm}"])
                saved = transfer["ranking"][encoder]["evaluations"]["sph"]["metrics"][arm]["auroc"]
                if actual != saved:
                    raise ValueError(f"AUROC differs from saved result: {encoder}, {arm}")
                identity = transfer["ptbxl_arm_difference_from_022"][encoder]
                if any(value != 0 for value in identity.values()):
                    raise ValueError(f"Predecessor reproduction failed: {encoder}")
                rows.append(
                    {"Model": display, "Training": arm, "Saved AUROC": saved, "Recomputed AUROC": actual}
                )
    return pd.DataFrame(rows)


def source_counts(transfer: dict[str, Any]) -> pd.DataFrame:
    """Describe the eligible labeled training pool used by Experiment 022b."""
    names = {
        "ptbxl": "PTB-XL",
        "chapman_shaoxing": "Chapman",
        "ningbo": "Ningbo",
        "georgia": "Georgia",
        "cpsc_2018": "CPSC",
        "cpsc_2018_extra": "CPSC-Extra",
    }
    return pd.DataFrame(
        [
            {"Source": names[source], "Label": label, "ECGs": count}
            for source, (total, positive) in transfer["counts"]["training"].items()
            for label, count in (("Annotation negative", total - positive), ("Annotation positive", positive))
        ]
    )


def sph_ranking(transfer: dict[str, Any]) -> pd.DataFrame:
    """Return matched SPH ranking comparisons for three frozen encoders."""
    return pd.DataFrame(
        [
            {
                "Model": display,
                "Training": label,
                "AUROC": transfer["ranking"][encoder]["evaluations"]["sph"]["metrics"][arm]["auroc"],
            }
            for encoder, display in ENCODERS.items()
            for arm, label in (("ptbxl", "PTB-XL only"), ("pooled", "Several sources"))
        ]
    )


def transfer_matrix(transfer: dict[str, Any], encoder: str) -> pd.DataFrame:
    """Compare three training strategies within each evaluation source family."""
    rows = []
    for family, display in FAMILIES.items():
        metrics = transfer["ranking"][encoder]["evaluations"][f"family:{family}"]["metrics"]
        for arm, strategy in (
            ("ptbxl", "PTB-XL only"),
            (f"loso_{family}", "Other families only"),
            ("pooled", "Target family included"),
        ):
            rows.append(
                {"Evaluation family": display, "Training strategy": strategy, "AUROC": metrics[arm]["auroc"]}
            )
    return (
        pd.DataFrame(rows)
        .pivot(index="Evaluation family", columns="Training strategy", values="AUROC")
        .reindex(columns=["PTB-XL only", "Other families only", "Target family included"])
    )


def paired_gains(transfer: dict[str, Any]) -> pd.DataFrame:
    """Read paired bootstrap gains without constructing intervals from marginal scores."""
    rows = []
    for encoder, display in ENCODERS.items():
        gain = transfer["ranking"][encoder]["evaluations"]["sph"]["contrasts"]["pooled_minus_ptbxl"]["auroc"]
        rows.append(
            {"Model": display, "Gain": gain["difference"], "Low": gain["ci_low"], "High": gain["ci_high"]}
        )
    return pd.DataFrame(rows)


def operating_points(transfer: dict[str, Any]) -> pd.DataFrame:
    """Describe historical transferred thresholds alongside ranking performance."""
    rows = []
    for arm, label in (("ptbxl", "PTB-XL only"), ("pooled", "Several sources")):
        operating = transfer["sph_operating"]["operating_points"][arm]["xecg"]
        auroc = transfer["ranking"]["xecg"]["evaluations"]["sph"]["metrics"][arm]["auroc"]
        for metric, value in (
            ("Ranking (AUROC)", auroc),
            ("Abnormal annotations caught", operating["sensitivity"]),
            ("Normal annotations referred", 1 - operating["specificity"]),
        ):
            rows.append({"Training": label, "Measure": metric, "Value": value})
    return pd.DataFrame(rows)


def label_curve(labels: dict[str, Any]) -> pd.DataFrame:
    """Read equal total-label budgets; bands show variability across label selections."""
    return pd.DataFrame(
        [
            {
                "Training": display,
                "Labels": budget,
                "AUROC": metric["mean"],
                "Low": metric["p2_5"],
                "High": metric["p97_5"],
            }
            for arm, display in (("ptbxl", "PTB-XL only"), ("pooled", "Several sources"))
            for budget in labels["budgets"]
            for metric in [labels["summaries"]["sph"][arm][str(budget)]["xecg"]["auroc"]]
        ]
    )


def pilot_curve(pilot: dict[str, Any]) -> pd.DataFrame:
    """Read the 5% local-normal pilot simulation for the historical pooled xECG head."""
    return pd.DataFrame(
        [
            {
                "Local normals": row["m"],
                "Mean referral rate (%)": 100 * row["rate_mean"],
                "Low": 100 * row["rate_p5"],
                "High": 100 * row["rate_p95"],
                "Sensitivity": row["sensitivity_mean"],
                "Within one point": row["share_rate_within_1pp"],
            }
            for row in pilot["summary"]
            if row["score"] == "pooled"
            and row["encoder"] == "xecg"
            and row["budget"] == 0.05
            and row["m"] > 0
        ]
    ).sort_values("Local normals")


def save_aggregate_report(evidence: dict[str, Any], destination: Path) -> None:
    """Write aggregate provenance and comparisons without patient-level outputs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "reanalysis_of_completed_development_experiments",
        "new_training_runs": 0,
        "age_subgroup_scores_computed": False,
        "italian_data_available": False,
        "sources": provenance().to_dict("records"),
        "analysis_module": to_stored(Path(__file__)),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "integrity": verify_saved_scores(evidence["transfer"]).to_dict("records"),
        "sph_paired_gains": paired_gains(evidence["transfer"]).to_dict("records"),
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
