# Exploratory MIMIC CPC clustering

This first pass tests whether the existing MIMIC ECG pool contains repeatable
groups of similar waveforms. It selects **one ECG per patient** with seed 42
from the 39,457 accepted MIMIC records, avoiding extra influence from patients
who have many hospital ECGs. It uses the frozen, self-supervised Experiment 004
CPC encoder, before the binary fine-tuning head, to extract 512-dimensional
features. The encoder already saw MIMIC waveforms in self-supervised training.

The preprocessing and clustering choices are fixed before inspecting cluster
outcomes: original train-only per-lead normalization, PCA to 16 components
fitted to these selected ECGs without labels, unit-length PCA vectors,
Euclidean HDBSCAN with `min_cluster_size=80`, `min_samples=20`, and excess-of-mass
cluster selection. HDBSCAN's `-1` means unassigned/noise. The inputs are held
only in memory; outputs are a small per-selected-ECG ID/cluster/score table and
aggregate cluster statistics under `outputs/mimic_cpc_clustering/`.

The CPC diagnostic-proxy flags are joined **after** clustering to describe each
group. They are not used to fit PCA or HDBSCAN. Because the binary classifier
uses a related CPC encoder and is unvalidated on MIMIC, flag enrichment is not
independent evidence of disease. A cluster is not a diagnosis or a count of
patients with a particular condition. Clusters need manual signal review and
independent diagnostic labels before condition names or clinical claims.

Run `uv run --no-sync python -m scripts.reports.cluster_mimic_cpc` from the
repository root. The script reads the completed CPC cache and previous local
flag table, without modifying raw signals or creating another waveform copy.

## Completed exploratory result, 26 September 2026

PCA retained **86.44%** of the selected feature variance. HDBSCAN found two
dense clusters and marked **5,353 / 7,892 (67.8%)** ECGs unassigned. Cluster 0
has **1,497** patients, with **1,434 (95.8%)** CPC flags and median CPC score
**0.993**. Cluster 1 has **1,042** patients, with **722 (69.3%)** CPC flags and
median score **0.483**. The unassigned group has **4,078 / 5,353 (76.2%)**
flags. The selected one-per-patient cohort has **6,234 / 7,892 (79.0%)** flags;
this differs from the all-ECG MIMIC rate because repeated ECGs do not receive
extra weight here.

An explicitly **post hoc** parameter check kept two clusters with identical
assignments when `min_cluster_size` rose from 80 to 120 (adjusted Rand index
1.0). Increasing PCA dimensions from 16 to 32 also found two clusters, but
marked **5,872** ECGs unassigned and gave adjusted Rand index **0.783** against
the primary partition, counting noise as a group. This is moderate partition
sensitivity, not proof of stable clinical subtypes.

The separate read-only [signal-quality audit](../outputs/mimic_cpc_clustering/signal_qc.json)
found median maximum absolute amplitudes of **1.60 mV** in both dense clusters;
flat leads were rare (3 and 0 recordings). This simple check gives no obvious
large-amplitude or flat-lead explanation for the split, but it cannot rule out
other acquisition artifacts or source shifts. See the local [aggregate report](../outputs/mimic_cpc_clustering/report.json)
and [selected ECG IDs with cluster assignments](../outputs/mimic_cpc_clustering/patient_sample_clusters.csv.gz).
