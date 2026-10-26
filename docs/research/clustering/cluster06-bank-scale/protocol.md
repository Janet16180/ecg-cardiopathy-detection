# CLUSTER-06: support-bank averaging and covariance-scale controls

## Prospective question and decision

Does averaging five reference-bank coordinate matrices improve conditional bank stability when the
regional tied GMM configuration is held fixed after RMS normalization? This is a finite development
pool sensitivity experiment, without disease-discovery, clinical-validity or causal endpoints.
Commit this protocol before new scores and source/tests before execution. Keep all work local.

The primary comparison is the 10th percentile of all 45 pairwise validation-anchor ARIs among ten
five-bank means minus the 10th percentile of all 1,225 pairwise ARIs among 50 individual banks, using
regional tied GMM k=2 and RMS coordinates with the unchanged absolute covariance floor 1e-3.
A meaningful conditional averaging benefit requires a difference >=0.10, at least 45/50 admissible
individual banks and 9/10 admissible ensembles. Report the signed difference even if negative. This
operational threshold is not a population effect-size test. Quantiles use NumPy's linear method.
All finite fits enter every denominator, including nonconverged or imbalanced fits. Do not infer
independent-pair confidence intervals: banks overlap supports and share the same patients.

## Inputs and integrity

Reuse exactly CLUSTER-04's 50 ordered support draws, consecutive disjoint blocks of five, 1,993
retained development patients, 1,582 fitting patients and 411 validation anchors. All 27 supports
and whole patients touching PTB folds 9/10 remain excluded. Verify identifiers against the existing
PR67 training-role reservation before feature consumption. No waveform access, new encoder inference,
GPU, downloads, new cohort, closed Challenge/EchoNext test data, age/sex analysis or label changes.

Pin CLUSTER-04 result SHA256 `c672be6b2fc9fe6d8c14d974179f04345381b068349df611d1656723e0427baa`
and CLUSTER-05 `cbe2d5af704e6a1ee86ba535ea03e2087bd667b33573ec676952d4e65275f885`.
Verify their immutable source/input/artifact bindings and completed statuses. Reuse
`bank_inputs.cached_inputs`, `committed_sources`, `replay_predecessor`, `bank_robustness` and
`matched_controls` helpers. Reproduce the eight CLUSTER-03 selected/individual-bank fits and native
NTT scores. Reproduce every CLUSTER-04 bank/ensemble assignment and check, and its numeric sensitivity
summaries within 1e-12. Reproduce CLUSTER-05's six relevant fixed NTT raw/RMS reference configurations
and eight patient refits each, with assignment ARI=1 and silhouette error <=1e-12. Replay the saved
NTT raw/RMS coordinates within 1e-12. Exact label integers may differ only by permutation. Abort on
identity, source, hash or replay disagreement; never convert an error into an exclusion.

## Fixed factorial conditions and numerical control

Use three configurations: regional tied GMM k=2, regional diagonal GMM k=2, record full GMM k=3.
Every fit has seed 20261018, n_init=3, init_params=kmeans, max_iter=400 and tol=1e-3. There is no
configuration selection. For each configuration fit each individual bank and each five-bank mean
under three conditions: fitting-centered raw coordinates with reg_covar=1e-3; fitting-centered
coordinates divided by s=sqrt(mean(centered_fitting_coordinates**2)) with reg_covar=1e-3; and the
same RMS coordinates with reg_covar=1e-3/s². Fit centering and s separately for every bank/ensemble.
Require finite positive s. Average bank coordinates before centering/scaling, never assignments.
Admissibility means convergence and >=5% occupancy separately in fitting patients and anchors.

The third condition tests coordinate equivalence to raw geometry. The scikit-learn k-means
initialization uses the same seed and is theoretically invariant to positive scalar transforms;
finite precision, initialization choices, multi-start ranking and EM stopping may still differ.
Use fitting-only maximum-overlap alignment with lexicographic permutation ties. Require whole-cohort
and anchor ARI=1, identical convergence flags, and absolute errors <=1e-6 for weights, responsibilities,
raw-coordinate means, raw-coordinate covariances and adjusted lower bounds. Backtransform RMS means
by s and covariance by s², and compare lower_bound_RMS-d*log(s) with raw. Compare centered raw means,
since both fits use the same fitting center. Save all maximum errors and each failed condition.
Do not tune tolerances, seeds or settings after scoring. An equivalence failure means numerical
equivalence was not demonstrated; no isolated covariance-floor attribution is then justified.
This numerical gate qualifies interpretation and does not replace the primary averaging decision.

## Persistence and reporting

For every configuration/scale fit the original three-bank-mean coordinates as its own fixed
reference. Describe reference-aligned agreement using fitting-only alignment, pairwise ARI,
convergence, occupancy, silhouette, anchor modal frequencies and entropy. For each own-reference
cluster report its anchor count and distribution of agreement, avoiding majority masking. Also
report persistence for each of the original CLUSTER-03 regional reference's two anchor groups,
including its 43-anchor minority group, for both regional configurations. Match by fitting overlap;
group identity comes from the frozen partition, never from target annotations. Record k=3 views
retain their own three-group reference summaries. Cluster numbers have no clinical meaning.

Use two CPU threads and a measured 1,200-second ceiling covering integrity, 540 bank/ensemble fits,
nine own-reference fits and summaries, checked between operations. Exceptions preserve typed failure
and progress; no retries or dropping fits. No annotation parsing or target classification is needed.
Save identities, samples, bank manifest, coordinates, assignments, model parameters, diagnostics,
versions, commit, runtime and all source/input/artifact hashes locally under
`outputs/cluster06_bank_scale/`, using repository-relative stored paths. Only aggregate-safe explicit
fields enter `reports/clustering/cluster06-bank-scale/`. The executing agent writes results from
outputs, including negatives and numerical-control failures. The coordinator updates queue/backlog
and indexes. No study rescues the failed CLUSTER-03/05 primary results or completes expert review.
