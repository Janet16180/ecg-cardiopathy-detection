# Are machine-abnormal ECGs close in the CPC feature space?

This exploratory follow-up uses the already frozen one-ECG-per-patient MIMIC
sample and cluster assignments. It does not train or retune CPC. The only
external comparison is the ECG cart's exact `Abnormal ECG`, `Normal ECG`, or
`Borderline ECG` summary in the official MIMIC-IV-ECG machine-measurement table.
These are machine interpretations, not adjudicated cardiopathy diagnoses.

Before computing results, the comparisons are fixed as follows:

1. For each original HDBSCAN group, report all three exact machine categories,
   unclassified status, CPC flag counts within abnormal and normal categories,
   and the abnormal share among ECGs with any exact summary. Compare cluster 0
   with the other groups descriptively, using one ECG per patient.
2. Re-extract the frozen self-supervised CPC features and reproduce the original
   PCA-16 unit vectors. Among ECGs with **exact abnormal or normal** summaries,
   find each ECG's ten nearest other patients by Euclidean distance. Report the
   mean proportion of abnormal neighbors for abnormal versus normal queries;
   their difference is the primary proximity statistic. The neighbor search
   uses neither machine labels nor CPC classifier scores to fit features.
3. Compare that statistic with 1,000 fixed-seed permutations of the two machine
   labels within each ECG cart ID. This checks whether proximity exceeds a
   simple cart-mix explanation, but small cart strata and other acquisition
   effects remain possible. Also report the unstratified permutation as a
   secondary diagnostic. Do not treat a post hoc p-value as confirmation of
   clinical validity or a new disease subtype.

This is an **association** audit. The cart summary is an imperfect reader and
the PTB-XL calibrated CPC threshold is not validated for MIMIC. A machine-normal
ECG can still have disease; a machine-abnormal ECG may lack cardiopathy. The
study cannot answer whether CPC correctly diagnoses patients without
independent cardiologist adjudication.

## Result, 26 September 2026

The command
`UV_CACHE_DIR=/tmp/ecg_uv_cache uv run --no-sync python -m scripts.reports.profile_mimic_cpc_cluster_labels`
finished on CPU. The [local aggregate receipt](../outputs/mimic_cpc_cluster_label_profile/report.json)
records the source, protocol, model, score, cluster, and official machine-table
hashes. Raw ECGs and frozen model artifacts were not changed.

The largest cluster (cluster 0) contains **1,497 patients**. CPC flags
**1,434/1,497 (95.8%)**. The cart explicitly calls **675 abnormal**, **112
normal**, and **346 borderline**; 364 have no exact summary. Among the 1,133
explicit summaries, 675 (59.6%) are abnormal, versus 1,763/4,863 (36.3%)
in cluster 1 plus unassigned ECGs. The exploratory two-sided Fisher comparison
gives odds ratio 2.59 and p≈4.1e-46; cluster selection and post hoc inspection
make this descriptive evidence, not a confirmatory p-value. CPC flags
**674/675** machine-abnormal ECGs in cluster 0, but also **85/112**
machine-normal ECGs. Thus its high cluster flag rate cannot be read as clinical
accuracy. The original cart-diversity audit found 126 devices in this cluster,
with the largest contributing 13.6% of ECGs.

Across **4,134** patients with an exact abnormal or normal machine summary,
the 10 nearest patients in frozen CPC feature space were machine-abnormal on
average **82.0%** of the time for abnormal queries, versus **23.9%** for normal
queries. This 58.1-point difference exceeded all 1,000 fixed-seed permutations
within device IDs (one-sided Monte Carlo p=0.001; null mean 2.4 points, 95%
interval 1.0–3.8 points). Only 3.5% of neighbor links share a device ID. For
cluster-0 abnormal queries, 93.8% of the global ten nearest neighbors were
machine-abnormal; the corresponding figure for cluster-0 normal queries was
38.3%. The same general separation appears in the other groups. This supports
that the representation captures ECG features associated with **broad machine
abnormality**, not that a particular cardiopathy forms a confirmed cluster.

The next validation step is blinded cardiologist adjudication or linkage to the
separate cardiologist reports, especially for the machine-normal/CPC-positive
and machine-abnormal/CPC-negative disagreements. No MIMIC threshold selection
or model training was performed in this audit.
