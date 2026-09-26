"""Read-only waveform quality comparison for exploratory MIMIC clusters."""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict

import numpy as np

from . import ROOT
from .cpc_pool import Pool
from .files import sha256_file, write_json_atomic

CLUSTERS = ROOT / "outputs/mimic_cpc_clustering/patient_sample_clusters.csv.gz"
CACHE = ROOT / "data/processed/cpc_pool_40k"
OUTPUT = ROOT / "outputs/mimic_cpc_clustering/signal_qc.json"


def run() -> dict[str, object]:
    """Compare simple waveform amplitude and flat-lead markers by cluster."""
    pool = Pool(CACHE)
    with gzip.open(CLUSTERS, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7892 or len({row["ecg_id"] for row in rows}) != len(rows):
        raise RuntimeError("Cluster table is incomplete or has duplicate ECG IDs")
    by_cluster: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        signal = np.asarray(pool.signals[pool.index[row["ecg_id"]]])
        std_by_lead = signal.std(axis=1)
        group = by_cluster[int(row["cluster"])]
        group["median_lead_std_mv"].append(float(np.median(std_by_lead)))
        group["maximum_abs_mv"].append(float(np.max(np.abs(signal))))
        group["flat_lead_count"].append(float((std_by_lead < 0.001).sum()))
        group["high_amplitude"].append(float(np.max(np.abs(signal)) > 5))
    summary = []
    for cluster, measurements in sorted(by_cluster.items()):
        summary.append({
            "cluster": cluster,
            "records": len(measurements["maximum_abs_mv"]),
            "median_lead_std_mv": float(np.median(measurements["median_lead_std_mv"])),
            "median_maximum_abs_mv": float(np.median(measurements["maximum_abs_mv"])),
            "records_with_flat_lead": int(np.count_nonzero(measurements["flat_lead_count"])),
            "records_above_5_mv": int(np.count_nonzero(measurements["high_amplitude"])),
        })
    report: dict[str, object] = {
        "purpose": "Signal-quality context, not diagnosis",
        "flat_lead_definition": "lead standard deviation <0.001 mV over 10 seconds",
        "high_amplitude_definition": "any sample absolute amplitude >5 mV",
        "clusters": summary,
        "cluster_ids_sha256": sha256_file(CLUSTERS),
        "cache_rows_sha256": sha256_file(CACHE / "rows.csv"),
    }
    write_json_atomic(OUTPUT, report)
    return report
