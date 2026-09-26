"""Audit CPC-negative MIMIC ECGs against independent machine report lines."""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

from . import ROOT
from .files import sha256_file, write_json_atomic

MACHINE = ROOT / "data/raw/mimic-iv-ecg/1.0/machine_measurements.csv"
OFFICIAL_SHA256 = "56f6b1413221bce95bd6f48b28ca1acf27ae0b073d6f2c1d12f3af7500eabbb6"
SCORES = ROOT / "outputs/mimic_cpc_flag_audit/identified_scores.csv.gz"
CLUSTERS = ROOT / "outputs/mimic_cpc_clustering/patient_sample_clusters.csv.gz"
OUTPUT = ROOT / "outputs/mimic_cpc_machine_disagreement"
REPORT_FIELDS = tuple(f"report_{index}" for index in range(18))
EXPECTED_THRESHOLD = 0.24229153990745544
MENTION_PATTERNS = {
    "atrial_fibrillation_or_flutter": re.compile(r"\batrial (?:fibrillation|flutter)\b", re.I),
    "ischemia_or_infarct": re.compile(r"\b(?:ischemi\w*|infarct\w*)\b", re.I),
    "conduction": re.compile(r"\b(?:bundle branch block|a-v block|av block|"
                              r"atrioventricular block|intraventricular conduction)\b", re.I),
    "hypertrophy": re.compile(r"\bhypertrophy\b", re.I),
    "st_t_change": re.compile(r"\b(?:st-t|t wave|st elevation|st depression)\b", re.I),
}


def machine_status(lines: list[str]) -> str:
    """Classify only exact machine summary lines, with conflicts left separate."""
    statuses = set()
    for line in lines:
        normalized = re.sub(r"^summary:\s*", "", line.strip().casefold())
        normalized = normalized.strip(" .;:")
        if normalized in {"abnormal ecg", "borderline ecg", "normal ecg"}:
            statuses.add(normalized.split()[0])
    if len(statuses) > 1:
        return "conflicting"
    return next(iter(statuses), "unclassified")


def text_mentions(lines: list[str]) -> tuple[str, ...]:
    """Record broad text mentions, including uncertain and negated wording."""
    report_text = " | ".join(lines)
    return tuple(name for name, pattern in MENTION_PATTERNS.items() if pattern.search(report_text))


def read_local_tables() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Load fixed CPC scores and prior cluster IDs by prefixed study ID."""
    with gzip.open(SCORES, "rt", newline="") as handle:
        scores = {row["ecg_id"]: row for row in csv.DictReader(handle)}
    with gzip.open(CLUSTERS, "rt", newline="") as handle:
        clusters = {row["ecg_id"]: row["cluster"] for row in csv.DictReader(handle)}
    if len(scores) != 39457 or len(clusters) != 7892:
        raise RuntimeError("Unexpected saved CPC score or cluster population")
    return scores, clusters


def join_machine_rows(
    scores: dict[str, dict[str, str]], clusters: dict[str, str]
) -> dict[str, dict[str, object]]:
    """Read official machine rows matching the fixed CPC cohort."""
    by_study = {key.removeprefix("mimic:"): value for key, value in scores.items()}
    if len(by_study) != len(scores):
        raise RuntimeError("Duplicate MIMIC study IDs")
    joined: dict[str, dict[str, object]] = {}
    with MACHINE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"study_id", "subject_id", "cart_id", *REPORT_FIELDS}.issubset(
            reader.fieldnames or ()
        ):
            raise RuntimeError("Unexpected machine-measurements columns")
        for row in reader:
            score_row = by_study.get(row["study_id"])
            if score_row is None:
                continue
            ecg_id = score_row["ecg_id"]
            if score_row["patient_id"] != f"mimic:{row['subject_id']}":
                raise RuntimeError(f"Machine subject mismatch for {ecg_id}")
            if ecg_id in joined:
                raise RuntimeError(f"Duplicate machine study: {ecg_id}")
            lines = [row[field].strip() for field in REPORT_FIELDS if row[field].strip()]
            joined[ecg_id] = {
                "ecg_id": ecg_id,
                "patient_id": score_row["patient_id"],
                "cluster": clusters.get(ecg_id, ""),
                "cart_id": row["cart_id"],
                "machine_status": machine_status(lines),
                "cpc_score": float(score_row["cpc_probability_ptb_calibrated"]),
                "cpc_flagged": score_row["flagged"] == "1",
                "machine_report": " | ".join(lines),
                "machine_text_mentions": text_mentions(lines),
            }
    return joined


def summarize(
    scores: dict[str, dict[str, str]], clusters: dict[str, str],
    joined: dict[str, dict[str, object]]
) -> dict[str, dict[str, float | int | None]]:
    """Count machine status and discordance in all-ECG and cluster views."""
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for ecg_id, score_row in scores.items():
        cluster = clusters.get(ecg_id)
        # The all-ECG and one-per-patient views are separate denominators.
        views = ("all_ecgs",) if cluster is None else ("all_ecgs", f"cluster_{cluster}")
        item = joined.get(ecg_id)
        for view in views:
            counter = groups[view]
            counter["ecgs"] += 1
            if item is None:
                counter["machine_missing"] += 1
                continue
            status = str(item["machine_status"])
            counter[f"machine_{status}"] += 1
            if score_row["flagged"] == "1" and status == "normal":
                counter["machine_normal_cpc_positive"] += 1
            if score_row["flagged"] == "0":
                counter["cpc_negative"] += 1
                if status == "abnormal":
                    counter["candidate_disagreements"] += 1
    summary = {}
    for view, counts in sorted(groups.items()):
        machine_abnormal = counts["machine_abnormal"]
        summary[view] = {**dict(counts),
                         "disagreement_fraction_among_machine_abnormal":
                         counts["candidate_disagreements"] / machine_abnormal
                         if machine_abnormal else None}
    return summary


def run() -> dict[str, object]:
    """Join exact MIMIC IDs and report machine/CPC disagreements by cluster."""
    if sha256_file(MACHINE) != OFFICIAL_SHA256:
        raise RuntimeError("Machine-measurements CSV differs from the official release")
    scores, clusters = read_local_tables()
    score_report = json.loads((SCORES.parent / "report.json").read_text())
    if score_report["threshold"] != EXPECTED_THRESHOLD:
        raise RuntimeError("CPC threshold changed from the frozen audit")
    joined = join_machine_rows(scores, clusters)
    summary = summarize(scores, clusters, joined)
    candidates = sorted((item for item in joined.values()
                         if item["machine_status"] == "abnormal" and not item["cpc_flagged"]),
                        key=lambda item: (item["cpc_score"], item["ecg_id"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "candidate_disagreements.csv.gz"
    temporary = target.with_name(target.name + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ecg_id", "patient_id", "cluster", "cart_id", "cpc_score",
                         "machine_status", "machine_text_mentions", "machine_report"))
        writer.writerows((item["ecg_id"], item["patient_id"], item["cluster"], item["cart_id"],
                          f"{item['cpc_score']:.8f}", item["machine_status"],
                          ";".join(item["machine_text_mentions"]), item["machine_report"])
                         for item in candidates)
    os.replace(temporary, target)
    mentions_by_view: dict[str, Counter[str]] = defaultdict(Counter)
    for item in candidates:
        views = ["all_ecgs"]
        if item["cluster"]:
            views.append(f"cluster_{item['cluster']}")
        for view in views:
            mentions_by_view[view].update(item["machine_text_mentions"])
    carts_by_cluster: dict[str, Counter[str]] = defaultdict(Counter)
    for item in joined.values():
        if item["cluster"]:
            carts_by_cluster[str(item["cluster"])][str(item["cart_id"])] += 1
    report: dict[str, object] = {
        "source_url": "https://physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv",
        "definition": "Exact machine 'Abnormal ECG' summary AND frozen CPC negative",
        "interpretation": "Machine/model disagreement, not confirmed cardiopathy or false negative",
        "threshold": EXPECTED_THRESHOLD,
        "matched_machine_rows": len(joined),
        "group_summary": summary,
        "candidate_machine_text_mentions_by_view": {
            view: dict(counts) for view, counts in sorted(mentions_by_view.items())
        },
        "machine_cart_diversity_by_cluster": {
            cluster: {
                "matched_ecgs": sum(counts.values()),
                "distinct_carts": len(counts),
                "largest_cart_fraction": max(counts.values()) / sum(counts.values()),
            }
            for cluster, counts in sorted(carts_by_cluster.items())
        },
        "candidate_count": len(candidates),
        "candidate_patient_count": len({item["patient_id"] for item in candidates}),
        "candidate_cpc_score_quantiles": {
            str(quantile): float(np.quantile([item["cpc_score"] for item in candidates], quantile))
            for quantile in (0.1, 0.5, 0.9)
        } if candidates else {},
        "machine_sha256": OFFICIAL_SHA256,
        "analysis_source_sha256": sha256_file(ROOT / "ecg_experiment/mimic_machine_disagreement.py"),
        "scores_sha256": sha256_file(SCORES),
        "clusters_sha256": sha256_file(CLUSTERS),
        "candidate_table": str(target.relative_to(ROOT)),
        "candidate_table_sha256": sha256_file(target),
    }
    write_json_atomic(OUTPUT / "report.json", report)
    return report
