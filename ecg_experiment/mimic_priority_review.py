"""Small review packet for acute-MI machine alerts scored negative by CPC."""

from __future__ import annotations

import csv
import gzip
import os
from collections import Counter

import numpy as np

from . import ROOT
from .cpc_pool import Pool
from .files import sha256_file, write_json_atomic

CANDIDATES = ROOT / "outputs/mimic_cpc_machine_disagreement/candidate_disagreements.csv.gz"
NEGATIVE_CLUSTERS = ROOT / "outputs/mimic_cpc_negative_clustering/negative_ecg_clusters.csv.gz"
CACHE = ROOT / "data/processed/cpc_pool_40k"
OUTPUT = ROOT / "outputs/mimic_cpc_machine_disagreement"
ALERT = "consider acute st elevation mi"


def run() -> dict[str, object]:
    """Select explicit machine-alert discordances and audit basic signal quality."""
    with gzip.open(CANDIDATES, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with gzip.open(NEGATIVE_CLUSTERS, "rt", newline="") as handle:
        groups = {row["ecg_id"]: row for row in csv.DictReader(handle)}
    selected = [{**row, "negative_cluster": groups[row["ecg_id"]]["cluster"],
                 "negative_membership": groups[row["ecg_id"]]["membership"]}
                for row in rows if ALERT in row["machine_report"].casefold()]
    pool = Pool(CACHE)
    maxima = []
    flat_leads = 0
    high_amplitude = 0
    for row in selected:
        signal = np.asarray(pool.signals[pool.index[row["ecg_id"]]])
        maximum = float(np.max(np.abs(signal)))
        maxima.append(maximum)
        flat_leads += int((signal.std(axis=1) < 0.001).any())
        high_amplitude += int(maximum > 5)
    target = OUTPUT / "machine_acute_alert_cpc_negative.csv.gz"
    temporary = target.with_name(target.name + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    os.replace(temporary, target)
    report: dict[str, object] = {
        "definition": "Machine text contains 'CONSIDER ACUTE ST ELEVATION MI' and CPC negative",
        "interpretation": "Unconfirmed machine alert, prioritized for independent clinical review",
        "ecgs": len(selected),
        "patients": len({row["patient_id"] for row in selected}),
        "cpc_score_below_0_1": sum(float(row["cpc_score"]) < 0.1 for row in selected),
        "negative_cluster_counts": dict(Counter(row["negative_cluster"] for row in selected)),
        "records_with_flat_lead": flat_leads,
        "records_above_5_mv": high_amplitude,
        "median_maximum_abs_mv": float(np.median(maxima)) if maxima else None,
        "candidate_source_sha256": sha256_file(CANDIDATES),
        "negative_cluster_source_sha256": sha256_file(NEGATIVE_CLUSTERS),
        "source_sha256": sha256_file(ROOT / "ecg_experiment/mimic_priority_review.py"),
        "review_table": str(target.relative_to(ROOT)),
        "review_table_sha256": sha256_file(target),
    }
    write_json_atomic(OUTPUT / "priority_review.json", report)
    return report
