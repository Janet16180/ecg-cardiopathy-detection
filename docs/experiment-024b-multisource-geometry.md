# Experiment 024b: embedding geometry across hospitals

**Frozen 29 September 2026, before any new score, cluster or distance is computed.** This is the rerun of
[Experiment 024](experiment-024-embedding-geometry.md) that the Ningbo data made possible (backlog item
`rerun_024_multisource`). The user authorized the reruns of the experiments Ningbo could affect. Before this
freeze only aggregate 024, 022b, 025b and 026b results and metadata were read: record counts, labels, source
families, feature record lists and 022b's saved training quality flags. No encoder is trained, no GPU is used.

## Question

Within PTB-XL, 024 found that k-means clusters of the frozen CPC features aligned more with the recording
device (AMI 0.22-0.23 for k = 4, 8, 16) than with diagnosis (superclass combination, 0.06-0.09), and that a
class-mean prototype trailed a linear probe by 0.031 (ECG-JEPA), 0.034 (xECG) and 0.073-0.077 (CPC) AUROC.

With PTB-XL and the Challenge hospitals together:

1. **Primary.** Do unsupervised clusters of the pooled training embeddings follow the source hospital more
   than the diagnosis?
2. Do class prototypes built from the pooled labels still trail the probe, read at SPH (an unseen hospital)?
3. How well does a linear readout identify the hospital from the embedding?
4. Do the reference ECGs (medoids) of each class come from one hospital, as the MI medoids came from one
   device in 024?

For the student screen, a new university is a new source. If the embeddings are organized by hospital,
cluster-, prototype- and neighbor-based methods will place its ECGs in their own region.

## What stays fixed

- **Encoders.** The frozen `xecg`, `jepa` and `cpc` features of 022b: CPC from Experiment 020, JEPA and xECG
  from 022's caches, the Challenge features of `outputs/features_challenge_v1/`, and 022's SPH features, all
  hash-checked against their receipts. **xECG is primary**, as in 022b and 026b; JEPA and CPC are secondary.
- **Label.** The standard label: PTB-XL's superclass label, the primary label of the
  [Challenge mapping](challenge-label-mapping.md) (positive for any MI, STTC, CD or HYP code, negative for
  sinus rhythm alone) and SPH's primary label (022).
- **Superclass combination** (024's diagnosis factor). PTB-XL: `ecg_experiment.eda.ptbxl.superclasses`, as
  in 024. Challenge: `NORM` for a primary negative, otherwise the present superclasses among MI, STTC, CD and
  HYP joined in alphabetical order with `+` (for example `CD+MI`). PTB-XL positives can carry NORM in the
  combination (for example `CD+NORM`, 353 training ECGs); Challenge positives cannot.
- **Methods.** 024's module, unchanged: `unit_space`, `prototype_scores`, `device_balanced_scores` (here
  with the source family as the device), `neighbor_indices` (25 neighbors), `medoids`, `ami_summary`, the
  fixed logistic readout of Experiment 018 (`full_development.fit_logistic`, used by 024's probes), and
  k-means (`KMeans`, `n_init=10`, `random_state=24024`). 022b's readout (`multisource_readout.fit_readout`,
  `C=0.01`) is the probe at SPH.

## Rows

**Pooled training rows** are 022b's: the 17,083 PTB-XL training ECGs with a standard label, then the 22,494
Challenge training-group ECGs that are evaluable, have features and pass the training quality policy (39,577
in all). The quality flags are read from 022b's `training_rows.csv` (hash checked against 022b's result), and
the rows must match 022b's order and counts.

**The balanced draw.** 026b's `equal_family_subsample` with seed **33033** draws the same number of records
from each of the four source families, as many as the smallest family has, without replacement:

| Family | Sources in the draw | Records | Positive |
| --- | --- | ---: | ---: |
| `ptbxl` | PTB-XL | 4,356 | 2,498 |
| `chapman_ningbo` | Ningbo 3,329, Chapman/Shaoxing 1,027 | 4,356 | 3,181 |
| `georgia` | Georgia 4,356 | 4,356 | 3,485 |
| `cpsc` | CPSC 2018 2,600, CPSC-Extra 1,756 | 4,356 | 3,805 |
| All | | 17,424 | 12,969 |

The family is the hospital unit, as in the [Challenge split](challenge-splits-v1.md); the six sources are a
secondary factor. The positive share differs by family (57% to 87%), so family and label are correlated; the
controls below address this.

**Held-out rows** (for the source probe and neighbor checks): PTB-XL's 1,572 labeled full-development ECGs
(884 positive) and the 7,612 Challenge calibration-group ECGs with features that 022b scored (Chapman/Ningbo
4,432, Georgia 1,718, CPSC 1,462). The calibration groups are used only to evaluate methods not fitted on
them, which the split's access rule allows.

**SPH:** 022's 21,008 evaluation ECGs (20,364 patients, 7,190 positive).

The Challenge test groups, PTB-XL calibration and test ECGs stay closed. No age or other subgroup analysis is
done.

## Spaces

- **Draw space.** A `StandardScaler` fitted on the balanced draw, then unit length (024's `unit_space`).
  Used for the clusters and the neighbor checks, so that no family dominates the scaling.
- **Readout space.** 024's rule, the scaler of the probe the prototypes are compared with: the scaler of
  022b's `pooled` readout (fitted on the 39,577 pooled rows) for pooled prototypes, and that of the `ptbxl`
  readout (17,083 rows) for the PTB-XL-only reference. Used for the SPH prototypes and the medoids.

## Reference: reproduce 024 and 022b

The run stops if any check fails.

- **R1, CPC structure within PTB-XL.** 024's k-means on its CPC space (the 17,417 PTB-XL training ECGs, the
  scaler of its refitted `cpc_standard` head, which must pass 024's integrity check): AMI with the superclass
  combination and the device for k = 2, 4, 8 and 16 must equal 024's `result.json` to within 1e-6.
- **R2, prototype minus probe within PTB-XL.** 024's encoder comparison for `cpc`, `jepa` and `xecg`
  (025's 15,359-ECG pool, 1,306 original development ECGs, 024's `evaluate_task`): every method's AUROC must
  equal 024's to within 1e-9, and the `prototype` minus `probe` interval (seed 24024) must equal it exactly.
- **R3, CPC medoids.** 024's five reference ECGs per group (NORM, MI, STTC, CD, HYP) in its CPC space must be
  the same records, in order, as `medoids.csv`, with similarities within 1e-9.
- **R4, the probes.** Refitting 022b's `ptbxl` and `pooled` readouts must reproduce 022b's saved SPH
  probabilities to within 1e-9.

## Analyses

**1. Structure (primary).** In the draw space, k-means with k in {2, 4, 8, 16} on the 17,424 balanced-draw
embeddings. No label or family enters any fit. For each partition, report AMI (`ami_summary`) with:

- the source family (4 levels) and the source (6 levels);
- the standard label (2 levels) and the superclass combination;
- the family within standard-negative rows only and within positive rows only (the label held fixed), and the
  standard label within each family (the family held fixed);
- the fraction of each factor's entropy explained by the partition (scikit-learn `homogeneity_score` with the
  factor as the true labels), which does not depend on the factor's number of levels.

Also HDBSCAN (024's settings: `min_cluster_size=80`, `min_samples=20`, excess of mass, noise as one label)
on PCA-16 unit vectors (randomized PCA fitted on the whole draw, `random_state=33033`) of a 4,000-record
subset, 1,000 per family, drawn from the balanced draw with `numpy.random.default_rng(33033)` visiting
families in sorted order. It is descriptive, without intervals, as in 024.

For comparison with 024, the same k-means (k = 2, 4, 8, 16) is run for each encoder on the 17,083 PTB-XL
training ECGs alone, each in a scaler fitted on those rows, with AMI against the device, the standard label
and the superclass combination. This is the within-PTB-XL device reference for JEPA and xECG, which 024 did
not cluster.

**2. Prototype versus probe at SPH (secondary).** For each encoder and each of the `ptbxl` and `pooled`
training sets, in that set's readout space:

- `probe`: 022b's readout for that set;
- `prototype`: 024's score, cosine to the positive class mean minus cosine to the negative class mean;
- `prototype_family_balanced` (`pooled` only): each class mean is the unweighted mean of the four families'
  class means (`device_balanced_scores` with the family as the device, at least 20 ECGs per family and
  class);
- `knn`: the positive fraction among the 25 nearest training ECGs.

The statistic is `prototype` minus `probe` SPH AUROC for `pooled`, with the same for `ptbxl` and the change
between them (`pooled` gap minus `ptbxl` gap). Also `prototype_family_balanced` minus `prototype` and `knn`
minus `probe`. Average precision is reported alongside.

**3. Source probe (secondary).** For each family, the fixed logistic readout (`fit_logistic`), one family
against the rest, fitted on the balanced draw's raw features and applied to the held-out rows. Report each
family's one-vs-rest AUROC, their mean, and the balanced accuracy of the four-way assignment to the family
with the highest probability (chance 0.25). Also the same with the six sources as classes (chance 1/6). For
comparison: 024's device probes within PTB-XL reached AUROC 0.75-0.96, and Leinonen et al. identify the source
at 96.9%.

**Neighbor checks.** In the draw space, with the balanced draw as the neighbor pool:

- for each held-out ECG, the fraction of its 25 nearest draw ECGs from its own family (expected 0.25 by
  shares);
- for each held-out family, `knn` and `knn_other_family` (024's `knn_other_device`, neighbors only from the
  other three families) AUROC on the standard label. `knn_other_family` imitates a new site: none of the
  neighbors comes from the query's hospital;
- for SPH, the share of its ECGs' 25 nearest draw ECGs from each family (which hospital a new hospital
  resembles). Descriptive.

**4. Reference ECGs (secondary).** For NORM-only and each single-superclass group (MI, STTC, CD, HYP) of the
39,577 pooled training ECGs, in the pooled readout space: the 50 members most similar to the group
prototype (024's `medoids`). Report the families of the top 5 and top 50, against the family shares of the
group. A group's medoids "come from one hospital" when all top 5 share one family. The top ECG of each group
for xECG is plotted from its raw file (10 s, 12 leads, the window its features were computed from, each lead
median-centred, no filter) after checking it against the official checksums.

024's label audit, subclass tasks and device, sex, age and heart-rate probes are not rerun.

## Uncertainty

- **Structure:** 2,000 record-level bootstrap draws of the balanced draw with seed 33033, the partition held
  fixed; the AMI of each factor is recomputed on each resample. Intervals are the 2.5 and 97.5 percentiles.
- **SPH:** 2,000 paired whole-patient draws with seed 33033 (`normal_manifold.patient_resamples`,
  `bootstrap_metrics`, `contrast`), the same draws for every score and encoder; fits held fixed.
- **Held-out rows (source probe and kNN):** 2,000 record-level draws with seed 33033 per set (the Challenge
  sources have no patient IDs); single-class draws are skipped and counted.

## Primary comparison and decision rule

- **Primary comparison:** xECG, k-means with k = 8, AMI with the source family minus AMI with the standard
  label, on the balanced draw, with its record-bootstrap interval.
- **Decision:**
  - Interval entirely above 0: **the clusters follow the hospital more than the diagnosis.** Unsupervised
    clusters, prototypes and neighbors from pooled data must be source-controlled, and a new site's ECGs
    should be expected to form their own region.
  - Interval entirely below 0: **the clusters follow the diagnosis more than the hospital.**
  - Otherwise: **not distinguished.**
- **Size.** A difference below 0.02 AMI is called small in the results; this does not change the decision.
- **Robustness (described, no separate decision).** The same contrast for JEPA and CPC, for k = 2, 4 and 16,
  with the superclass combination in place of the standard label, and with the entropy-explained fraction in
  place of AMI. The results say whether the sign agrees across them, and whether the family AMI within a fixed
  label stays above the label AMI.
- **Prototypes (secondary).** Read with 024's thresholds: a `prototype` minus `probe` gap within 0.02 means
  the prototypes are usable as they are; above 0.05 means diagnosis does not dominate the distances. Pooling
  **narrows** the gap if the change interval lies entirely above 0 and **widens** it if entirely below 0.
- All other contrasts are described by where their interval lies (above 0, below 0, includes 0).

## Caveats written into the results

- ECG-JEPA and xECG report Chapman and Ningbo in their self-supervised pretraining data, so they have likely
  seen the `chapman_ningbo` waveforms without labels; CPC was pretrained on PTB-XL and MIMIC (Experiment 004)
  and has seen no Challenge record. No encoder saw a label here; SPH is unseen by every encoder.
- Family differs from hospital in two ways: Chapman/Shaoxing and Ningbo are two hospitals in one family, and
  CPSC 2018 and CPSC-Extra come from one collection. Family also carries country, device, recording protocol,
  population and labeling practice; the analysis cannot say which of these the embeddings encode.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM, and the families differ
  in positive share, so the label and the family are not independent.
- AMI depends on the number of levels of each factor (4 families against 2 label values); the
  entropy-explained fraction and the superclass combination are the checks on that.
- The Challenge split is by record; a patient may have ECGs in the train and calibration groups, which can
  inflate same-family neighbors and the source probe for the Challenge families. PTB-XL development patients
  are disjoint from training.
- One draw, one seed per encoder checkpoint, one k-means seed; the bootstrap holds the partitions and fits
  fixed. PTB-XL development was inspected by earlier experiments. The label is an ECG annotation proxy, not
  confirmed disease or a referral decision.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -u -m scripts.experiments.run_multisource_geometry024b
```

CPU only. The reproduction checks R1-R3 run in the main process with four threads, as 024 did; then one process
per encoder with one BLAS thread each (three threads). `scripts/experiments/run_multisource_geometry024b.py`
uses the frozen 024, 022b, 026b, 027 and 022 runners and modules and the new
`ecg_experiment/multisource_geometry.py`; no frozen module changes. It hashes every input, source and this
protocol into the result, refuses to overwrite an existing run, and writes
`outputs/experiment024b_multisource_geometry_v1/` (`result.json`, `medoids.csv`, figures, `run.log`). Figures
are copied to `docs/figures/experiment-024b/`: AMI by factor and k per encoder, SPH AUROC by method, and one
reference ECG per group. Results go to `docs/experiment-024b-multisource-geometry-results.md`.
