# Experiment 034: one-class baselines on the frozen embeddings

Frozen 29 September 2026, before any score of this experiment is computed. This covers the backlog item
`one_class_embedding_baselines`, which the [novelty search](normal-manifold-novelty-search.md) lists as a
baseline set a reviewer would ask for before accepting the distance-from-normal result as a contribution.
Before this freeze only aggregate 026 and 026b results, their receipts and metadata (record counts, feature
shapes, file hashes) were read, and the candidate methods were timed on random numbers of the same shape. No
fit-set or evaluation feature entered a method of this experiment.

**SPH is development data.** Experiments 022, 022b, 024b, 026, 026b, 027, 027b, 029 and 030 have all read
it. It is the unseen hospital for every encoder, but it is not a final test.

## Question

026b adopted a squared Mahalanobis distance (StandardScaler, PCA-64, Ledoit-Wolf) fitted on 10,846 normal
ECGs from PTB-XL and the Challenge training groups (the `pooled` arm). On the same frozen embeddings, fit
set and evaluation rows, does that Gaussian beat other one-class scores? Put differently: is the Gaussian
the right density for the normal reference, or just the simplest one?

## What stays fixed

- **Encoders.** The frozen features of 026 and 026b: xECG (1,024 dimensions) primary, ECG-JEPA (768)
  secondary, CPC (512) secondary and optional. Loaded exactly as 026b does, every cache checked against its
  receipt (`run_normal_manifold026.identity`, `load_sph`; `run_label_efficiency025.select_rows`,
  `load_features`; 026b's `split_table`, `challenge_features` and 027b's `feature_identity`).
- **Fit set.** Exactly 026b's `pooled` fit set per encoder: the rows of
  `outputs/experiment026b_multisource_manifold_v1/fit_sets.csv` (hash checked against 026b's `result.json`)
  with `pooled` true, in that file's order: 5,872 PTB-XL NORM training ECGs, then 2,581 Ningbo, 819
  Chapman/Shaoxing, 1,023 Georgia and 551 CPSC 2018 training normals, 10,846 in all. The runner rebuilds
  the feature rows from the loaders above and stops unless their record IDs equal that file's, in order.
- **Evaluation rows and label.** The standard label on the rows of 026b:
  - SPH: 21,008 `use_evaluation` ECGs with a primary label (20,364 patients, 7,190 positive).
  - PTB-XL development: 1,306 ECGs (1,173 patients, 843 abnormal), the binary task.
  - Challenge calibration groups, per family: Chapman/Ningbo 4,432 ECGs (3,254 positive), Georgia 1,718
    (1,372), CPSC 1,462 (1,279).
- **Reference.** The Mahalanobis score is refitted with the frozen `normal_manifold.fit_mahalanobis` and
  `mahalanobis_scores` on this fit set. It must reproduce 026b's saved `<encoder>_pooled` scores in
  `sph_scores.csv`, `development_scores.csv` and `challenge_scores.csv` (hashes checked against 026b's
  receipt) to within 1e-9 relative, or the run stops. The expected difference is 0, so the reference AUROCs
  are 026b's: xECG SPH 0.878, development 0.920.

## Methods

Every method is fitted on the fit set only, once per encoder. Higher scores mean farther from normal.

Two input spaces, fixed here:

- **Unit-length features** for the kNN scores, as in Sun et al. (ICML 2022): each raw embedding divided by
  its L2 norm, with no standardization.
- **Whitened PCA-64** for every other method: the Mahalanobis reference's own fitted scaler and PCA, and
  each of the 64 components divided by the square root of its fit-set variance
  (`pca.explained_variance_`). The fit set then has zero mean and unit variance per component. This holds
  the projection fixed, so these methods differ from the reference only in the density or boundary they
  place on the same 64 coordinates. Isolation forest and the full-covariance mixture do not depend on the
  per-component scaling; the SVM kernel and the networks do.

| Method | Space | Score | Hyperparameters |
| --- | --- | --- | --- |
| `mahalanobis` | scaler, PCA-64 | squared Mahalanobis distance, Ledoit-Wolf (reference) | 026, unchanged |
| `knn_kth` | unit-length | Euclidean distance to the k-th nearest fit normal | k = 10, fixed |
| `knn_mean` | unit-length | mean Euclidean distance to the k nearest fit normals | k = 10, fixed |
| `knn026` | standardized, unit-length | 026's mean cosine distance to the 25 nearest (`normal_manifold.knn_scores`) | 026, unchanged |
| `ocsvm` | whitened PCA-64 | minus `OneClassSVM.decision_function` | sklearn defaults: RBF, `nu` 0.5, `gamma` "scale" |
| `iforest` | whitened PCA-64 | minus `IsolationForest.score_samples` | sklearn defaults: 100 trees, 256 samples per tree; seed 38038 |
| `gmm` | whitened PCA-64 | minus `GaussianMixture.score_samples` (log density) | full covariance, K tuned (below), `n_init` 3, seed 38038 |
| `deep_svdd` | whitened PCA-64 | squared distance of the network output to the centre c | fixed network, epochs tuned (below) |
| `autoencoder` | whitened PCA-64 | mean squared reconstruction error | fixed network, epochs tuned (below) |
| `flow` | whitened PCA-64 | minus the log density of a RealNVP flow | fixed network, epochs tuned (below) |

Why these fixed values:

- kNN: Sun et al. used k = 50 for 50,000 CIFAR-10 training images and k = 1,000 for 1.28 million ImageNet
  images, about 0.1% of the training set; 0.1% of 10,846 is about 10. `knn026` is 026's score, kept so that
  the kNN result links to 026 (where it was not adopted).
- One-class SVM and isolation forest: the library defaults, as used by the standard benchmarks (PyOD,
  ADBench) when no labelled anomalies are available for tuning.

### Hyperparameters chosen on held-out normals only

No evaluation row enters any choice. A tuning slice is drawn once from the fit set: 10% of the fit set's
units, where a unit is a PTB-XL patient (026's patient IDs) or a Challenge record (the Challenge headers
have no patient ID), permuted with `numpy.random.default_rng(38038)`. The first ceil(0.1 U) units form the
held-out slice; the rest is the tuning-training part. The projection (scaler, PCA) is the one fitted on
the whole fit set; the slice is held out only from the method being tuned. The criterion is always a
normal-only quantity on the held-out normals:

- `gmm`: K in {1, 2, 4, 8, 16} components, chosen by the largest mean held-out log-likelihood; ties go to
  the smaller K. Then refitted on the whole fit set with that K.
- `deep_svdd`, `autoencoder`, `flow`: the number of epochs. Each network is trained on the tuning-training
  part for up to 300 epochs, with the held-out loss (below) measured after every epoch; the chosen number
  of epochs is the one with the lowest held-out loss (the first, on ties), and training stops once 30
  epochs pass without a new lowest. The network is then trained again from the same initialization on the
  whole fit set for the chosen number of epochs, and that network scores the evaluation rows.

Networks, all in float32 on the CPU with one thread, `torch.manual_seed(38038)` before building each
network, Adam with learning rate 1e-3, batches of 256 drawn by a `torch.Generator` seeded 38038, and
gradient norms clipped at 5:

- `deep_svdd` (Ruff et al., ICML 2018, one-class objective): a bias-free MLP 64-128-64-32 with LeakyReLU
  (slope 0.1), no autoencoder pretraining. The centre c is the mean output of the untrained network on the
  training rows, with any coordinate smaller than 0.1 in absolute value set to ±0.1 (Ruff's rule). Loss:
  mean squared distance to c, weight decay 1e-6. Held-out loss: the same mean on the held-out normals.
- `autoencoder`: MLP 64-128-16-128-64 with ReLU and biases, a 16-dimensional bottleneck. Loss and held-out
  loss: mean squared reconstruction error.
- `flow`: RealNVP with 8 affine coupling layers. Each layer updates the 32 coordinates chosen by a fixed
  random permutation (drawn once from `numpy.random.default_rng(38038)`) from the other 32, through an MLP
  32-128-128-64 with ReLU; the log-scale passes through tanh. Standard normal base. Loss and held-out loss:
  mean negative log-likelihood.

### Wall-time cap

A method is dropped for an encoder if its tuning, fit and scoring take more than 30 minutes of wall time in
its process (one thread each). Network training checks the clock after every epoch and stops, marking the
method dropped. A library method that finishes over the cap is also dropped. A dropped method is reported
as dropped, with its elapsed time, and takes no part in the decision. The normalizing flow is optional in
the backlog; it runs under the same cap, and a drop is not a failure of the experiment.

## Evaluation

Each method is scored once on each set. Nothing is fitted or selected on an evaluation set.

1. **SPH (primary).** AUROC per method and encoder; average precision is reported beside it.
2. **PTB-XL development (secondary).** AUROC per method and encoder.
3. **Challenge calibration groups, per family (secondary).** AUROC per method and encoder. The `pooled` fit
   set includes each family's training normals, so this is an in-distribution readout (patients may repeat
   across the Challenge record split).
4. **Sensitivity at a 5% referral budget (secondary).** 030's rule (`referral_budget.budget_threshold`,
   type-1 quantile, referral strictly above), with the threshold from the 1,707 Challenge calibration
   normals (030's source quantile for the distance score, m = 0), applied to all 21,008 SPH ECGs. Reported:
   the achieved false-referral rate on the 13,818 SPH normals and the sensitivity on the 7,190 positives,
   with whole-patient intervals from `intervals.metric_intervals` (2,000 draws, seed 38038); only its
   sensitivity and specificity entries are read. Also, as a threshold-free companion, the sensitivity at
   the budget threshold set on the SPH normals themselves (sensitivity at 95% specificity), without an
   interval.

### Intervals

All from the shared `ecg_experiment/intervals.py`; no bootstrap code is copied.

- **Contrasts:** each method minus `mahalanobis`, AUROC, on SPH, development and each family:
  `intervals.paired_auroc_difference`, 2,000 draws, seed 38038. SPH and development resample whole
  patients; the Challenge families resample records (each record its own unit). The same seed gives the
  same draws for every method of a set, so the contrasts are paired across methods too. Skipped
  single-class draws are counted and reported.
- The fits are held fixed in every draw.

## Primary comparison and decision rule

- **Primary comparison:** on SPH, xECG, the AUROC of each method minus that of `mahalanobis`, with its
  paired whole-patient interval.
- **Decision:** Mahalanobis stays the reference unless a method meets both conditions on xECG:
  1. its SPH interval lies entirely above 0; and
  2. its observed PTB-XL development AUROC is at most 0.01 below Mahalanobis's (difference ≥ −0.010).

  If one or more methods meet both, the one with the largest observed SPH difference becomes the candidate
  replacement, and a follow-up must confirm it before the paper's reference changes. If none does,
  Mahalanobis stays.
- No multiplicity correction is applied across the nine alternatives. The rule is one-sided in favour of
  the reference, and a replacement also has to hold at the home site and in a follow-up, so a single
  chance win would not change the claim.
- ECG-JEPA and CPC, the families, average precision and the referral-budget readout are described in the
  same interval language (above 0, below 0, includes 0). No decision attaches to them.

## Closed data and exclusions

- PTB-XL calibration and test ECGs and the Challenge test groups stay closed.
- No age or other subgroup analysis.
- No hyperparameter is changed after a score is seen. A bug found after scoring is fixed only if it
  violates this protocol, and the fix and the discarded run are reported.

## Caveats written into the results

- xECG and ECG-JEPA were pretrained without labels on Chapman and Ningbo waveforms; SPH is unseen by every
  encoder.
- Seven of the ten methods (all but the three kNN scores) see only the 64 principal components, so they
  cannot use information the PCA discards. That is by design: it isolates the density.
  The kNN scores use the full embedding.
- One fit per method and encoder. The networks and isolation forest depend on their seed; no seed spread is
  measured. The defaults and small networks are not the best each method could do with labels for tuning,
  which a one-class screen does not have.
- The tuning slice is held out from the tuned method but not from the shared projection.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_one_class_baselines034
```

One CPU stage, one process per encoder with one thread each (three in all). New files only:
`ecg_experiment/one_class_baselines.py` and `scripts/experiments/run_one_class_baselines034.py`; the frozen
`normal_manifold.py`, `multisource_manifold.py`, `referral_budget.py`, `intervals.py` and the 026 and 026b
runners are imported, not changed. The runner hashes every input, source and this protocol into the
result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment034_one_class_baselines_v1/` (`result.json`, `sph_scores.csv`, `development_scores.csv`,
`challenge_scores.csv`, `run.log`). Results go to `docs/experiment-034-one-class-baselines-results.md`,
with one figure under `docs/figures/experiment-034/`.
