# CLUSTER-04: reference-bank robustness on cached native ECG vectors

## Prospective question and decision

Does the NTT regional partition persist when its three support patients per finding change within the
existing pool? This is a conditional sensitivity study, not an independent clinical test or new disease
label definition. Commit this protocol before scores and commit the implementation before execution.
User authorization on 19 October 2026 permits continued local autonomous work; no push or merge.

The primary representation and clustering configuration are frozen from CLUSTER-03: regional affinity
mean/max/population-standard-deviation coordinates (nine axes), centered on fitting patients only,
with tied-covariance Gaussian mixture k=2, n_init=3, reg_covar=1e-3, max_iter=400, tol=1e-3 and seed20261018.
No new search, individual-axis scaling, backbone training, inference, downloads or GPU use. The primary operational
robustness gate requires all three conditions:

1. At least45 of50 support-bank fits converge and have every cluster occupy at least5% separately among
   fitting patients and among development validation anchors.
2. The10th percentile of all1,225 pairwise anchor adjusted Rand indices is at least0.50.
3. At least80% of the411 anchors agree with the original three-bank-mean CLUSTER-03 partition in at
   least80% of the50 banks, after fitting-only alignment of cluster names.

Use `numpy.quantile(method="linear")`. Include labels from finite nonconverged or imbalanced fits in
all50-bank denominators; never discard a bank to improve a summary. An exception, nonfinite representation
or invalid label shape aborts execution and produces a failure receipt. A passing gate means persistence
within this finite development support pool. A coherent alternative partition may fail the reference
agreement gate. Secondary analyses cannot rescue failure or reverse CLUSTER-03's failed primary gate.

## Inputs, roles and exact predecessor replay

Use exactly CLUSTER-03's1,993 already explored PTB development patients,1,582 fitting patients and411
validation anchors, excluding all27 NTT-06 support patients and whole patients touching sourcefold9/10.
Verify identifiers against the PR67 train reservation before feature or diagnostic payload consumption.
Never read closed Challenge/EchoNext records, add age/sex analyses or change source labels.

Pin CLUSTER-03 result SHA256 `64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9`,
cache receipt `6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3`, and NTT-06
result `3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3`. Verify their source,
input and artifact bindings, omitting documented mutable status snapshots. Reuse cached native768-axis
patient and automatic-QRS-region unit vectors. No waveform access or backbone forward passes.

Before new scores, reconstruct all five selected CLUSTER-03 reference partitions with ARI=1 and
silhouettes within1e-10, and its three original individual regional-bank fits with ARI=1 and matching
convergence/occupancy. Recompute original NTT record/regional coordinates with max error<=1e-12.
Reproduce all180 NTT-06 native-QRS choices/scores from saved vectors, with maximum score error<=1e-12
and exact integer correct counts. Bind predecessor file hashes, fitted identities and all source bytes.

## Frozen support sampling and fitting

Extract the existing nine distinct support patients per NORM/CRBBB/CLBBB finding from NTT-06's27 supports.
Sort each class pool by numeric patient ID then ECG ID. Use `numpy.random.default_rng(20261020)` to draw
three patients without replacement independently within each finding, sort each selected triple, and
accept the first50 distinct complete banks. A duplicate complete bank is redrawn; no outcome or feature
value selects banks. Save the complete local bank manifest before computing new affinities.

Each prototype is the unit-normalized mean of its three unit patient vectors, retaining the unchanged
class order from the frozen reference helper. Compute patient cosines and automatic-region cosines.
Per patient summarize regions as three means, three maxima and three population standard deviations.
Each new bank contributes one such view; do not count computational replicas as independent banks.
Fit the frozen regional configuration separately for every bank using all1,582 fitting patients and
predict all1,993 patients. Anchors never fit centering, mixture parameters or cluster-name alignment.

For reference-aligned agreement, form k-by-k fitting-patient contingency tables versus the original
CLUSTER-03 selected partition. Enumerate all k! permutations in lexicographic order, maximize total
matched fitting count and choose the first permutation on ties. This exactly defines deterministic
assignment alignment without looking at anchor or clinical labels. Report each anchor's matching-bank
fraction, its aligned modal-cluster frequency and entropy in bits. These quantities describe bank sensitivity,
not posterior diagnostic probabilities. Save full local bank assignments and coassignment frequencies
on anchors, which are invariant to cluster-name permutations.

## Prespecified secondary analyses

Apply the same50 banks to the fixed CLUSTER-03 record-affinity configuration (three centered axes,
full-covariance mixture k=3; all other parameters unchanged). Report every bank's admissibility, pairwise
anchor ARIs and fitting-aligned reference agreement. No record result enters the regional primary gate.

Form ten disjoint five-bank ensembles from consecutive blocks of the saved bank manifest. Average the
five regional coordinate matrices before fitting the frozen regional configuration; do not vote cluster
names. Report the same diagnostics and all45 ensemble pairwise ARIs. The support draws overlap patients,
and ensembles remain correlated through the finite support pool and shared cohort. No independent-sample
confidence interval is inferred from pairwise ARIs or the number of banks.

On the unchanged60 NTT-06 targets (20 per existing finding, disjoint from all27 supports), use the same
50 banks to classify frozen patient-QRS vectors by largest cosine, with unchanged class-order argmax ties.
Report the range/median/10th/90th percentiles of balanced accuracy and each patient's fraction of correct
bank predictions. Reuse the exact unchanged three-bank-average physical full-context baseline from the predecessor; its
own bank uncertainty is not resampled here, so this comparison is conditional and asymmetric. Do not
fit a new classifier or claim new independent accuracy. This is a secondary illustration of conditional
support uncertainty, with fixed previously explored targets and no new clinical evidence labels.

Only after bank sampling, fits and the primary decision are frozen, parse the same permitted source
superclass annotations and describe anchor annotation-combination AMI distributions for regional and
record banks. No label selects a bank, configuration, ensemble or threshold. No new enrichment hypothesis
tests, clinical masks, causal claims or statistical rescue. Heart rate/quality associations from
CLUSTER-03 remain qualifications; apparent stable morphology is not disease etiology.

## Runtime, outputs and reporting

Use two CPU threads and a1,200-second measured execution ceiling checked between fits. The budget covers
integrity replays,100 new single-bank fits and10 ensemble fits, coordinates and fixed summaries. Never
silently drop expensive configurations or replace an inadmissible bank. Record versions, seeds, counts,
source/input/artifact hashes, timing and failures. No dependency or pinned-file edits.

Keep bank identities, per-patient vectors/assignments and private plots under `outputs/cluster04_bank_robustness/`.
Only explicitly allowlisted aggregates and standalone aggregate figures enter
`reports/clustering/cluster04-bank-robustness/`. Write `results.md` from actual executed outputs, including
a failed gate; update both queues and rank follow-ups. Blinded expert review is a separate readiness task,
not an endpoint completed by this study. PR#71 stays draft and all work stays local.
