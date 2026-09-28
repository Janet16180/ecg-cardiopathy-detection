# Experiment 024: CPC embedding geometry audit and diagnosis prototypes

**Frozen 28 September 2026, before any development score is computed.** The user asked whether a reference vector per
cardiopathy can classify ECGs by cosine similarity, and what clustering the CPC embeddings would show. This
experiment answers the prerequisite question: is the frozen CPC feature space organized by diagnosis, or
mainly by nuisance factors such as device, heart rate and demographics? No encoder is trained, no GPU is
used, and calibration and test patients stay closed.

## What is already known

- [PTB/MIMIC diagnosis geometry](ptb-mimic-cpc-diagnosis-geometry.md): with PCA-16 unit vectors from the
  Experiment 004 encoder, the ten nearest training neighbors of a diagnosis-positive ECG share that diagnosis
  1.7 to 1.9 times more often than prevalence. HDBSCAN found broad normal and abnormal regions, with no pure
  cardiopathy cluster.
- [Experiment 020](experiment-020-full-development-readout-results.md): a logistic probe on the same encoder's
  512 features reaches AUROC 0.889 on full development with the standard label. A probe identifies the
  `CS100 3` device with AUROC 0.974.

- [Experiment 025](experiment-025-label-efficiency-results.md): with the same fixed linear readout on the
  original development ECGs, released ECG-JEPA and xECG features are far stronger than this CPC encoder
  (AUROC 0.959 and 0.962 versus 0.921 with all labels, and about 0.035-0.040 higher at every label
  budget). This does not change the analyses: section 5 already compares these encoders.

Neighbor enrichment and a good linear probe can coexist with a geometry dominated by other factors. The
earlier study did not compare a distance-based classifier with the probe, measure what the neighborhoods
share besides diagnosis, or control for device.

Two published results bear on this. ECG-JEPA (arXiv:2410.08559v5, appendix A.3, Table 10) reports a
nearest-class-mean classifier on frozen features for six encoders: it works without training but ranks the
encoders differently from the linear probe. The ECG foundation-model benchmark (arXiv:2509.25095v2,
section 4.3, Tables 4 and 5) finds the released ECG-CPC encoder much weaker with a linear head on pooled
features than with an attention-pooling head (PTB-XL superclasses 0.863 versus 0.919). The authors
attribute this to a token-level objective not shaping the pooled vector. Our own
[local readout study](cpc-local-readout-v1-results.md) found the same direction. A CPC pooled vector may
therefore be a poor space for prototypes even when the information is present.

## Questions

1. **Prototypes (primary).** How close does a nearest-prototype classifier, built from class means of
   training embeddings, come to the linear probe?
2. **Nuisance.** How strongly do the features encode device, sex, age and heart rate, and how much of the
   neighbor structure survives when neighbors from the query's own device are excluded?
3. **Structure.** Do unsupervised partitions of the training embeddings align more with diagnosis or with
   device and demographics?
4. **Reference ECGs.** Which real training ECGs are closest to each diagnosis prototype? These medoids are
   the data-derived "reference signal per condition" for visual and clinical review.

## Inputs

- Features: `outputs/experiment020_full_development_v2/features.npz`, the unchanged starting CPC encoder's
  512 pooled features, 17,417 training and 1,604 full-development ECGs. The file must match the
  `features_sha256` in that directory's `result.json` (`09376ee9...`). Rows follow
  `ecg_experiment.full_development.cohorts(ptb_table())`, as in Experiment 020.
- Labels: the standard binary label as in Experiment 020, and multi-label diagnostic superclasses
  (`ecg_experiment.eda.ptbxl.superclasses`, any diagnostic statement, no likelihood threshold, matching the
  standard label). Secondary: diagnostic subclasses with at least 100 training and 20 development ECGs.
- Nuisance variables: `device`, `male` and `age` from `ptb_table` (age 300 is missing), and `heart_rate`
  from `outputs/eda/features/ptbxl_500hz.parquet`.

Development patients were inspected by earlier readouts, so every result is exploratory.

## Feature space

The training-only `StandardScaler` of Experiment 020, applied to float64 features, then scaled to unit
length. Cosine similarity is the dot product in this space. Nothing is fitted on development ECGs.

## Analyses

**Integrity.** Refit Experiment 020's `cpc_standard` head. It must reproduce the saved development
probabilities to 1e-5 and AUROC 0.888642 on the 1,572 labeled development ECGs. The run stops otherwise.

**1. Prototype versus probe (primary).** For the binary standard label:

- `probe`: the Experiment 020 `cpc_standard` head.
- `prototype`: cosine similarity to the mean of positive training embeddings minus cosine similarity to
  the mean of NORM-only training embeddings, with each mean re-normalized.
- `knn`: fraction of positives among the 25 nearest training ECGs by cosine similarity.
- `multi_prototype_k`: k-means (seed 24024, 10 initializations) within positives and within NORM-only
  training ECGs, k in {2, 4, 8}. Score is the highest cosine similarity to a positive centroid minus the
  highest to a NORM centroid. All three k are reported; none is selected.

Primary statistic: `prototype` minus `probe` AUROC on full development. The same comparison is repeated
per superclass (one-vs-rest, positive = superclass present, negative = NORM-only) and per eligible
subclass.

**2. Nuisance audit.** The same fixed logistic readout (Experiment 018 settings) predicts, from training
to development: each device with at least 50 development ECGs (one-vs-rest AUROC) and sex (AUROC). Ridge
regression (alpha 1.0) predicts age and heart rate (development R²). For each development query, report the
fraction of its 25 nearest training ECGs from the same device, and compare it with that device's training
share. Two controls:

- `knn_other_device`: `knn` using only training neighbors from a different device than the query.
- `prototype_device_balanced`: each prototype is the unweighted mean of its per-device class means, using
  devices with at least 20 training ECGs of that class.

Report both AUROCs next to `knn` and `prototype`.

**3. Structure.** On the training embeddings: k-means with k in {2, 4, 8, 16} (seed 24024), and HDBSCAN on
PCA-16 unit vectors with the earlier settings (`min_cluster_size=80`, `min_samples=20`, excess of mass).
HDBSCAN uses the seeded 4,000-patient subset, as in the earlier study, because it did not finish on the
whole cohort. For each partition, report the adjusted mutual information with the superclass combination,
device, sex, age decade and heart-rate band (<60, 60 to 100, >100 bpm). No label enters any fit.

**4. Reference ECGs.** For NORM-only and each single-superclass group of training ECGs (for example MI
with no other superclass), list the five training ECGs most similar to the group prototype, and plot the
top one, 10 s, 12 leads. PTB-XL is public, so the plots may be committed.

**5. Encoder comparison (secondary).** Repeat analyses 1 and 2 for three cached frozen encoders: released
ECG-JEPA (768 features, `data/processed/pretrained/ecg-jepa-full-public`), released xECG (1,024,
`outputs/experiment016_xecg_probe_finetune/features`) and released ECG-CPC (512,
`outputs/experiment004_cpc_40k/released_features`). These caches hold other PTB-XL partitions too, so rows
are selected by ECG ID and only training and original development IDs are read. The comparison uses the
original 1,306 development ECGs, the only ones present in every cache, and the standard label. Each encoder
gets its own training-only scaler and the same fixed probe. This asks which representation is organized by
diagnosis, not which is the best classifier.

**6. Label audit (training only).** For each training ECG, compute the fraction of its 25 nearest training
neighbors (itself excluded) whose standard label differs, in the CPC and the ECG-JEPA spaces. List ECGs
where both fractions are at least 0.8 as candidates for review, with their SCP codes and
`validated_by_human`. This follows ECG-JEPA appendix E, where embedding outliers led to records that looked
mislabeled. No label is changed; a list is the only output.

## Uncertainty

AUROC differences use 2,000 paired whole-patient bootstrap draws with seed 24024. Draws with one class
are counted and skipped.

## How results will be read

Set before any score is computed:

- Prototype within 0.02 AUROC of the probe: the geometry is diagnostic, and per-condition reference
  vectors are usable as they are. Next: prototypes from several human-labeled sources and a frozen external
  readout on SPH.
- Gap above 0.05: diagnosis is linearly present but does not dominate the distances. Reference vectors and
  clustering would need a supervised projection first, such as the probe's subspace or metric learning.
- `knn_other_device` more than 0.02 below `knn`, or device AMI above diagnosis AMI: device drives the
  neighborhoods. Multi-source prototypes and clustering must then be device- or source-controlled.
- If another encoder's prototype-to-probe gap is much smaller than CPC's, that encoder is the better base
  for reference vectors and clustering, whatever its probe AUROC.

## Feasibility details fixed at freeze

These settle how the design above is executed. They were fixed from metadata counts only, before any
feature was loaded or any score computed. They do not change an analysis.

- **Joins.** `heart_rate` comes from `outputs/eda/features/ptbxl_500hz.parquet`, which has one row per lead
  with the same value on all 12 leads of a record; the first value per `record_id` (the PTB-XL `ecg_id`) is
  used. It is the EDA's QRS-energy screening estimate (`ecg_experiment.eda.signals.heart_rate`), not a
  clinical measurement. All 21,799 records have one. `validated_by_human`, `report` and `scp_codes` come
  from `data/raw/ptb-xl/1.0.3/ptbxl_database.csv`, joined by `ecg_id`.
- **Subclasses.** From `data/raw/ptb-xl/1.0.3/scp_statements.csv`, rows with `diagnostic == 1`, column
  `diagnostic_subclass`. A subclass is present when any listed code maps to it, with no likelihood
  threshold. The NORM subclass is excluded. On the 17,083 labeled training and 1,572 labeled development
  ECGs, 15 subclasses qualify: IMI, AMI, STTC, LVH, LAFB/LPFB, ISC_, IRBBB, ISCA, _AVB, IVCD, NST_, CRBBB,
  CLBBB, LAO/LAE and ISCI.
- **CPC rows.** Analysis 1 uses the 17,083 training ECGs with a standard label (the `cpc_standard` rows) as
  prototype and neighbor pool, and scores the 1,572 labeled full-development ECGs. The feature space is the
  scaler of the refitted `cpc_standard` head. Per-superclass and per-subclass tasks keep only rows that are
  positive for the task or NORM-only, on both sides. Each task computes all four methods; its probe is the
  same fixed head refitted on the task's training rows. The nuisance probes of analysis 2 use all 17,417
  training and 1,604 full-development ECGs, as Experiment 020's device probe did; age uses rows with a known
  age. The same-device fraction and `knn_other_device` use the analysis 1 pool and queries.
- **Devices.** With at least 50 full-development ECGs: CS-12, CS-12 E, AT-6 C 5.5, AT-6 6, AT-60 3, AT-6 C
  and AT-6 C 5.8. CS100 3 has 49 and is therefore not probed here; Experiment 020 already reported it
  (AUROC 0.974). In the encoder comparison (1,306 ECGs) the eligible devices are the first six.
- **Methods.** k-means is scikit-learn `KMeans` on the unit vectors, `n_init=10`, `random_state=24024`
  (analyses 1 and 3); centroids and all prototypes are re-normalized. A device-balanced prototype averages
  the unit-length per-device class means with equal weight. Ridge regression is a training-only
  `StandardScaler` followed by `Ridge(alpha=1.0)`.
- **Structure.** Analysis 3 uses all 17,417 training ECGs. PCA-16 is fitted on their unit vectors
  (randomized solver, `random_state=24024`) and its output re-normalized. The HDBSCAN subset follows the
  earlier procedure with this experiment's seed: permute the training rows with
  `numpy.random.default_rng(24024)`, keep the first ECG of each patient, and take the first 4,000. HDBSCAN
  noise (-1) counts as one label. AMI is scikit-learn's `adjusted_mutual_info_score`. The superclass
  combination is `ecg_experiment.eda.ptbxl.superclasses` (for example `CD+MI`, or `none`); age decade is
  floor(age / 10) with missing age as its own category.
- **Reference ECGs.** Groups are drawn from the 17,083 labeled training ECGs by superclass combination:
  NORM-only (7,243), MI (2,043), STTC (1,903), CD (1,353) and HYP (415). Candidates are group members.
  Plotted records are read from the raw 500 Hz files after checking them against the official
  `SHA256SUMS.txt`; the display subtracts each lead's median and applies no filter.
- **Encoder comparison.** The training pool is Experiment 025's: 15,359 training ECGs with a standard label
  present in all three released caches. CPC is repeated on the same rows with its own pool-fitted scaler as
  the reference for the three released encoders.
- **Label audit.** The pool is the same 15,359 ECGs. The CPC space is the analysis 1 space; the ECG-JEPA
  space uses the scaler fitted on the pool. Neighbors come from the pool only.
- **Bootstrap.** `ecg_experiment.full_development.patient_bootstrap` with seed 24024 and 2,000 draws, for:
  each CPC method minus `probe`, `knn_other_device` minus `knn`, and `prototype_device_balanced` minus
  `prototype` on the binary label; `prototype` minus `probe` for each superclass and subclass; and, per
  encoder, `prototype` minus `probe`, `knn_other_device` minus `knn` and `prototype_device_balanced` minus
  `prototype`.
- **Integrity tolerance.** Probabilities to 1e-5 and AUROC within 1e-6 of Experiment 020's 0.8886421.
- **Reading.** A prototype-to-probe gap between 0.02 and 0.05 has no prescribed reading and is reported as
  intermediate.

## Outputs

`scripts/experiments/run_embedding_geometry024.py`, CPU only, uses `ecg_experiment/embedding_geometry.py`.
It hashes every input, source file and this protocol into `result.json`, refuses to overwrite an existing
run, and writes `result.json`, `medoids.csv` (record IDs and similarities), `label_audit.csv` and figures to
`outputs/experiment024_embedding_geometry_v1/`. Copies of the figures go to `docs/figures/experiment-024/`.
The results go to `docs/experiment-024-embedding-geometry-results.md`, and a review sheet for a cardiologist
to `docs/experiment-024-clinician-review.md`.

```bash
OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_embedding_geometry024
```

## Not in scope

Synthetic or textbook ECGs as references: a digitized image is far out of distribution for this encoder,
and one example captures one patient, not a condition. Text prototypes (ECG-report alignment) are a
separate, later study. SPH, EchoNext and other sources are not used here.
