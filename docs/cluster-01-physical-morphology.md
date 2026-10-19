# CLUSTER-01: physical morphology clusters under reservation controls

Prospective bounded CPU study, 18 October 2026. Commit this protocol before extracting features or
computing a new clustering score. Freeze implementation with a second local commit before execution.
No push, pull request, GPU, new dependency, download or clinical label change is authorized here.

## Question and prior evidence

Do interpretable physical ECG measurements support stable partitions associated with mixtures of
existing ECG annotations after accounting for rate, amplitude, quality and acquisition metadata?
This concerns recurring morphology, not causes of disease or mutually exclusive patient diagnoses.

024 found CPC k-means clusters more associated with device and rate than diagnosis; 024b showed that
this was encoder-dependent. 034 found that more elaborate one-class algorithms failed to improve
its primary Mahalanobis comparison. The MIMIC CPC study left 67.8% unassigned and used model-derived
flags, not ground truth. Their caches/receipts are missing locally, so this study cannot reproduce those
scores or claim a matched improvement. Its directly reproducible predecessor is NTT-01's identity
count: 17,418 records and 15,023 patients in PTB-XL folds 1–8. Reproduce only those identity counts;
NTT-01's full label counts would access the newly reserved patients and must not be rerun.

## Cohort and quarantine

Read identifiers, patient IDs, folds and filenames first. Exclude every patient appearing in fold 9/10.
Assign remaining canonical integer patient IDs by SHA256 of
`ssl_synthetic_partition_v1:20261018:{patient_id}`, integer digest modulo 10000. Only buckets below
8000 (PR67 train) are eligible. Buckets 8000–8999 and 9000–9999 are reserved and never contribute
signals, diagnoses, quality or device metadata. Validate against the coordinator's metadata-only
partition manifest if supplied. Select each patient's lowest ECG ID, then the first 2,000 patients
ordered by SHA256 of `cluster01:20261018:{patient_id}`. No replacement for failed measurements.
Read non-identity metadata only for selected records. This is exploratory training-set analysis.
No ECGAD, Challenge, EchoNext, SPH or MIMIC data are accessed. No age analysis is performed.

## Fixed representation

Read original 500 Hz PTB-XL signals in mV with canonical lead order. Reuse the existing
`ecg_experiment.eda.avance2_features.ecg_measurements` helper. Its measurements are screening
estimates, not validated clinical delineations. Record failures and nonfinite values explicitly.
Features: QRS and QT durations, RR coefficient of variation, frontal QRS axis sine/cosine, and R, S,
ST and T amplitudes in all 12 leads (53 columns). Do not add threshold-based clinical criteria.
Replace nonfinite values by column medians, winsorize at sample 1st/99th percentiles, robust scale,
and fit PCA to 95% variance, at most 16 components. Drop rows missing over 20% of these features.
Reject any representation with an entirely missing column. Identical transforms feed all methods.

Two prespecified views: raw morphology, and morphology residualized by ridge regression (alpha=10)
on device/site one-hot indicators, heart rate, signal standard deviation, baseline fraction,
beat consistency, constant-lead count and source quality flag. Numeric nuisances use median
imputation and robust scaling; missing categorical metadata is an explicit level. Residualization
is descriptive and can remove meaningful correlated physiology; it is not causal adjustment.
Refit imputation/scaling/residualization/PCA for every stability subsample.

## Algorithms and selection

Common seed 20261018. PCA plus k-means (`n_init=10`), diagonal Gaussian mixture (`n_init=3`,
`max_iter=300`, `reg_covar=1e-4`), and spectral clustering (15-nearest-neighbor graph,
`assign_labels=kmeans`, `n_init=10`) each consider k=2,4,8. HDBSCAN uses Euclidean distance,
`min_cluster_size=40` or 80, `min_samples=10`, excess-of-mass selection. sklearn's implementation
counts the point itself in min_samples; see its [official API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

For each method/view select the highest silhouette on assigned observations, computed on at most
1,000 fixed randomly selected assigned rows, among configurations with at least two clusters,
at least 70% assigned, and minimum cluster size >=2% of the whole cohort. Break exact ties by
candidate order. If no configuration qualifies, report that failure and display the highest finite
silhouette as a descriptive fallback, explicitly ineligible. HDBSCAN noise is never a disease class.
Report all candidates. No diagnostic annotation is read by any selection function.

For each selected configuration perform five seeded 80% patient subsamples without replacement,
refitting the complete representation and algorithm. Report adjusted Rand index on overlapping
assigned rows and all rows separately, coverage, and minimum/median/maximum (five draws are not a
confidence interval). The same rows and draws are used for all methods/views. Compare the raw and
residualized full-cohort assignments with adjusted Rand index. No UMAP tuning or visual selection.

## Post-selection interpretation and decision

Only after all configurations and assignments are fixed, join the existing NORM/MI/STTC/CD/HYP
superclass flags, preserving co-occurrence. Report prevalence and multi-abnormal fraction per
cluster, one-sided Fisher enrichment against the remaining cohort, and one Benjamini-Hochberg
family covering every reported cluster-by-label test across both views and all selected methods.
Report adjusted mutual information with the superclass combination, device, site, quality flag,
heart-rate quartile, amplitude quartile, baseline-noise quartile and consistency quartile. Quartiles
are descriptive, not operating points. Include noise in AMI, but exclude it from enrichment tests.
For the selected residualized baseline and the unsupervised best alternative, give 200 paired
whole-patient bootstrap percentile intervals for their AMI difference with superclass combinations;
these intervals condition on fitted clusters and are not a new independent generalization test.

Primary comparison: the eligible residualized non-k-means alternative with highest silhouette,
against residualized PCA+k-means. Declare a promising morphology organization only if both are
eligible, the alternative's median assigned-row stability ARI >=0.70, minimum stability >=0.50,
and silhouette exceeds baseline by >=0.02; also require at least two distinct abnormal superclasses
with BH q<0.05 and prevalence lift >=1.5 somewhere, and superclass-combination AMI greater than
both device and heart-rate-quartile AMI. Otherwise the prespecified result is negative/inconclusive.
This is a conservative exploration gate, not a clinical success claim. Report every condition and
never change selection based on labels. A successful gate would still require an independent source.

## Execution and artifacts

At most two CPU threads; 2,000 patients; no new training. Target under 15 minutes CPU. Save local
feature, assignment and selection tables; source/protocol/input hashes; precise environment and
runtime; status including failures; aggregate JSON; generic aggregate plots of stability, nuisance
associations and cluster prevalence. No raw waveform or patient identifier in committed figures.
Commit a results report written from executed outputs, including negative results and deviations.
Coordinator owns queue/backlog updates. Suggested follow-up must be ranked there before new work.
