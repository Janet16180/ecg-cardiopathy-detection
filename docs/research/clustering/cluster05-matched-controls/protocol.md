# CLUSTER-05: dimension, pooling and scale controls

## Prospective question and decision

CLUSTER-03 found stable native and NTT partitions but failed its geometric persistence gate. Its
native PCA16 versus NTT3/9 comparisons confound dimension, pooling and scale. This follow-up asks
whether native regional summaries and NTT regional summaries agree after matching their dimension,
pooling operators and total fitting variance. It cannot reverse CLUSTER-03's negative primary result.

The primary comparison is `native_region9_rms` versus `ntt_region9_rms`. Both selected configurations
must be eligible under the unchanged CLUSTER-03 rule, and their validation-anchor adjusted Rand index
(ARI) must be >=0.50. Failure of either condition is a negative primary result. This measures
concordance between related views of previously explored development ECGs; it is not independent
replication, disease discovery, diagnosis, specialist localization, or evidence about etiology.
Secondary comparisons, unscaled results, support-label controls and annotations cannot rescue failure.

Commit this protocol before scores and commit all source/tests before execution. The scientific
reviewer and coordinating agent must approve the draft before its protocol commit. No further
experiment is automatically authorized by this study.

## Verified inputs and exact predecessor replay

Use the unchanged CLUSTER-03 retained cohort and cached native vectors: 1,993 patients, 1,582 fitting
patients and 411 deterministic validation anchors. Verify CLUSTER-03 result SHA256
`64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9` and native-cache result SHA256
`6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3`, their source, input and artifact
bindings, and the complete original identity/PR67 training-role checks before consuming feature or
annotation payloads. Exclude whole patients touching PTB folds9/10 and all27 NTT supports. Use the exact
saved fitting/anchor identities and eight CLUSTER-03 subsamples; bind them again. No new cohort,
protected dataset, age/sex subgroup, label definition, waveform read, encoder inference, GPU,
dependency, download, network access, push or merge is allowed.

Before any new fit, reproduce each of CLUSTER-03's five selected reference configurations on its
original view and fitting rows, with the frozen reference_geometry helpers. Require reconstructed raw views within1e-12 of saved raw representations, then refit their original
preprocessing and require reference anchor assignments ARI=1 and silhouette error<=1e-10. Replay all three original single-bank
regional fits and their pairwise ARIs within1e-12. Also reproduce NTT mean-bank record/regional views
from the original frozen support vectors and prototypes within1e-12 of their saved representations.
Abort on any identity, hash, input, source or replay disagreement; do not turn errors into exclusions.

## Fixed matched views

For every fitting sample independently fit centered, unwhitened PCA3 and PCA9 to its normalized
native patient QRS vectors, using deterministic full SVD. Orient each component so its largest-absolute loading is positive
(first coordinate breaks exact ties), before region pooling. No individual coordinate standardization or
whitening is allowed. PCA3 is fitted to one vector per fitting patient, rather than treating all
regions as independent fitting rows; this avoids weighting patients by their number of beats. Report
patient-QRS explained variance and the separate fraction of fitting-region variance captured by the
PCA3 basis, giving each fitting patient equal total weight across their regions. Anchors do not enter
this diagnostic.

The five raw geometries are:

1. `native_region9`: transform each unit QRS-region vector through the fitting-patient PCA3 basis;
   concatenate each patient's coordinate means, maxima and population standard deviations (nine axes).
2. `ntt_region9`: the original mean-bank NTT regional view, with the same mean/max/population-std
   pooling and nine axes.
3. `native_patient3`: fitting-patient PCA3 coordinates of unit native patient QRS vectors.
4. `ntt_record3`: the original mean-bank patient-to-prototype cosine view with three axes.
5. `native_patient9`: fitting-patient PCA9 coordinates of unit native patient QRS vectors.

Center every resulting patient view using only its current fitting sample. Produce an unscaled `_raw`
view and an `_rms` view divided by one strictly positive fitting-derived scalar
`sqrt(mean_ij(centered_ij**2))`. This sets average coordinate population variance to1, retaining relative
coordinate variances and anisotropy. Refit the PCA basis, patient centering and RMS within every
subsample. This is a global scalar, so k-means geometry should be invariant; fixed GMM regularization
has a different relative effect after scaling. NTT prototypes remain frozen across patient refits.

Matching three directions and pooling operators does not make PCA and NTT equivalent. PCA chooses
variance directions and its fixed loading-based coordinate signs affect maxima. NTT averages nonlinear regional
summaries over three distinct support banks; native PCA uses one basis. These residual differences
limit attribution. No favorable sign, component, view or support bank will be selected after results.

## Grid, selection and controls

Run the original k=2/3/4/8 and k-means/diagonal/tied/full GMM configurations, with the unchanged
CLUSTER-03 seeds, n_init, regularization, iteration limits, eight patient refits, convergence checks,
minimum5% fitting/anchor occupancy, silhouette>=0.05, median stability>=0.80 and minimum>=0.60.
Rank by median stability, minimum stability, silhouette, smaller k, then the original algorithm order.
Run all16 configurations for each of10 views (1,440 reference/refit fits), without annotation access.
An ineligible same-ranked fallback is descriptive only. Report all candidates and all selected
pairwise anchor ARIs, with the primary comparison explicitly identified. Secondary matched comparisons
are native_patient3 versus ntt_record3 and native_patient9 versus ntt_region9, separately for raw/RMS.
Report raw-versus-RMS ARI for every matching k-means configuration; expect ARI=1, and expose numerical
or optimization disagreements rather than treating a failed invariant as scientific scale dependence.

Fit the selected scaled NTT regional configuration to each of the three original support-bank views,
with fitting-only centering and scalar normalization; report occupancy, convergence and pairwise ARIs.
Rebuild20 support-label controls using the original CLUSTER-03 seeds, shuffling individual support
labels within each bank while preserving three supports per finding and rebuilding prototypes. Use
only the selected scaled NTT regional configuration; fit centering/RMS anew and report anchor ARI to
the original selected partition, convergence, occupancy and silhouette. These controls are conditional
sensitivity descriptions, not confirmatory p-values or evidence that labels cause disease.

Use two CPU threads and a900-second measured wall budget covering integrity replay, transforms,
clustering and controls, checked between operations. Preserve partial progress and typed failure on
budget exhaustion; do not drop expensive configurations or alter settings after seeing scores.

## Interpretation and deliverables

Save and hash the complete label-free selection/decision before parsing existing target annotations.
Only then describe selected-view AMI with the unchanged overlapping NORM/MI/STTC/CD/HYP annotation
combinations and original nuisance/quality variables. Report original and20 support-control annotation
AMIs. No enrichment significance claims, new taxonomy, subgroup analysis or clinical recommendation.
Annotation association cannot select or rescue a representation, configuration or primary decision.

Local artifacts under `outputs/cluster05_matched_controls` include exact identities/samples,
transform coordinates, native feature/cache/input/source hashes, assignments, convergence, runtime,
replay details and the frozen unsupervised decision. Only aggregate-safe JSON/figures go under
`reports/clustering/cluster05-matched-controls/`. The executing agent writes `results.md` from the
completed outputs, including negative findings and limitations. The coordinator updates both queue
files and the backlog/ranking. Work remains local and PR71 remains draft.
