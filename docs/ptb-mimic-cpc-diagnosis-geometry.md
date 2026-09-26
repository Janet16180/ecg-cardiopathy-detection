# Do PTB-XL diagnostic groups form CPC neighborhoods, and do they transfer?

Protocol frozen before looking at outcomes, 26 September 2026. This is a
read-only exploratory analysis of the historical self-supervised CPC encoder.
It does not train a model, tune a classifier threshold, or inspect PTB-XL
calibration/test patients.

Use one seeded ECG per **PTB-XL training patient** from the frozen 40k CPC
cache. Join PTB-XL `scp_codes` to official `scp_statements.csv` using exact ECG
and patient IDs. A diagnostic code contributes a superclass only when its
released confidence is at least 50. Examine MI, CD, STTC and HYP separately;
labels can overlap. `NORM` is an ECG annotation, not patient health, and is
counted as a reference only when no other diagnostic superclass is present.
Rows with no qualifying diagnostic superclass remain in the geometry fit but
are excluded from each label-specific comparison.

Extract frozen 512-dimensional CPC pooled features. Fit PCA-16 **only on PTB
training patients**, normalize to unit length, and use Euclidean distance.
For each diagnosis, the primary statistic is the proportion of the ten nearest
other PTB training patients sharing that diagnosis among positive queries,
compared with its prevalence among all sampled PTB training patients.
Compare with 1,000 fixed-seed global label permutations; these exploratory
permutations retain each class count but cannot remove every acquisition or
clinical confound. Report the multi-label co-occurrence and exact cohort size.

Fit HDBSCAN on the same PTB vectors (`min_cluster_size=80`, `min_samples=20`,
excess-of-mass selection). Summarize every cluster and unassigned ECGs by
diagnostic superclass. No disease label enters PCA or HDBSCAN. A diagnosis
concentrated in a cluster is a post hoc enrichment, not a validated subtype.

**Cost amendment before any result:** The initial complete-cohort CPU command
was stopped after more than six minutes without a report. It produced no
outcome. Keep the full PTB patient cohort for PCA fitting and the primary
neighbor test, but perform HDBSCAN only on a uniform 4,000-patient subset
drawn without replacement with NumPy seed 42. This subset is exploratory and
its counts must not be presented as full-cohort cluster prevalence. The MIMIC
transfer comparison remains the original full one-patient-per-person sample.

For an out-of-source check, transform the frozen one-per-patient MIMIC sample
using the **PTB-fitted** PCA. Compare ten-nearest-neighbor homophily for the
exact cart-generated `Abnormal ECG` versus `Normal ECG` summaries, with 1,000
fixed-seed permutations within cart IDs. Also summarize those machine labels
in the original MIMIC clusters. These are broad machine interpretations, not
the four PTB diagnosis labels; this transfer check therefore tests general
ECG abnormality structure, **not** diagnosis-specific external validation.
No MIMIC or PTB label is used to fit the feature transform.

The development and test partitions remain closed. A stronger future test
would link independent cardiologist reports to MIMIC or use a separately
adjudicated cohort with the same diagnosis definitions. Neither data source
establishes a patient's complete clinical health from a ten-second ECG.

## Completed result, 26 September 2026

The CPU command
`UV_CACHE_DIR=/tmp/ecg_uv_cache uv run --no-sync python -u -m scripts.reports.profile_ptb_mimic_cpc_diagnoses`
completed after the documented cost amendment and a reporting-variable fix.
The [aggregate receipt](ptb-mimic-cpc-diagnosis-geometry-aggregate.json)
is a committed copy of the [local run output](../outputs/ptb_mimic_cpc_diagnosis_geometry/report.json) and
contains the exact input, protocol and analysis-source hashes. The
[protocol snapshot at execution](ptb-mimic-cpc-diagnosis-geometry-protocol-at-run.md)
matches its protocol hash; the result text was appended afterward. No raw ECGs,
model weights, PTB held-out patients, or MIMIC score thresholds were changed.

The PTB analysis used **15,023 training patients**, one ECG per person. For
each diagnosis, the table compares the observed fraction of positive ten-nearest
neighbors with the diagnosis prevalence in that patient sample:

| PTB ECG annotation | Positive patients | Prevalence | Positive query's positive neighbors | Ratio to prevalence |
| --- | ---: | ---: | ---: | ---: |
| MI (infarction-pattern superclass) | 2,648 | 17.6% | 33.4% | 1.90× |
| CD (conduction disturbance) | 3,263 | 21.7% | 36.4% | 1.68× |
| STTC (ST/T changes) | 3,305 | 22.0% | 38.6% | 1.75× |
| HYP (hypertrophy) | 1,494 | 9.9% | 17.1% | 1.72× |

All four exceeded their 1,000 globally permuted label baselines (one-sided
Monte Carlo p=0.001 each). This is exploratory association, because this CPC
encoder had already learned from these *unlabeled* PTB training waveforms and
because demographics, devices and co-occurring diagnoses were not controlled.
It is not clinical diagnostic accuracy or an independent-patient test.

On the seeded **4,000-patient PTB density subset**, HDBSCAN assigned 907 ECGs
to cluster 0 and 758 to cluster 1; 2,335 were unassigned. Cluster 1 had
**508/758 (67.0%)** `NORM`-only annotations versus **1,014/2,335 (43.4%)**
among unassigned ECGs. Cluster 0 was mixed: **328/907 STTC**, **232/907 CD**,
**221/907 MI**, and **129/907 HYP**, with overlaps. Thus the local geometry
shows broad abnormal-versus-normal structure, but **no pure, identified
cardiopathy cluster**. In particular, an MI-pattern annotation is not proof of
an acute heart attack.

The PTB-fitted PCA was then applied unchanged to the original **7,892 MIMIC
patients**. Among the 4,134 with an exact machine `Abnormal ECG` or `Normal
ECG` summary, abnormal queries had **82.0%** abnormal ten-nearest neighbors,
versus **25.4%** for normal queries. The 56.6-point difference exceeded all
1,000 within-cart permutations (one-sided Monte Carlo p=0.001). Only 3.4% of
neighbor pairs shared a cart ID. This supports transfer of a broad
abnormality-related signal, but cannot externally validate MI, CD, STTC or
HYP because the MIMIC comparison label is a different, machine-generated
binary summary. The next scientifically useful step is matched,
independently adjudicated diagnosis labels for the MIMIC ECGs, or a separate
cohort with the same PTB diagnosis definitions.
