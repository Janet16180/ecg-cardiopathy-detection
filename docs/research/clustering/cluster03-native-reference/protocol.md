# CLUSTER-03: native encoder and No Time to Train reference geometry

## Question and prospective decision

Do native ECG representations form stable patient groups, and how does the No Time to Train
reference-matching stage change those groups? Compare raw encoder geometry, reference-conditioned
geometry and physical morphology on the same broad development cohort. The NTT pipeline has no new
trained backbone: its region and record similarities derive from the frozen ECG-JEPA features and
labeled reference prototypes. Reference-conditioned clustering is not unsupervised disease discovery.

Commit this protocol before new inference or clustering scores, and freeze each stage's source before
execution. Primary success requires eligible winners for both native QRS and NTT regional-affinity
views, anchor ARI >=0.50 between their partitions, and every pairwise ARI >=0.50 across the three
individual-support-bank NTT regional fits at the selected NTT configuration. Other views, label
associations or density plots cannot rescue failure. A pass means exploratory geometric persistence,
not independent clinical validity, source transfer, specialist localization or disease etiology.

## Cohort and exact integrity

Start with CLUSTER-01's exact 1,999 already explored training patients and 53 physical features.
Verify result SHA256 `74f0c82918f158a61368d308acdab5d5e98fe759852c966de76867e5c99bc8e7`, identity,
feature, physical-measurement and reservation hashes. Skip its documented mutable running-status hash.
Check identities against the PR67 training manifest and metadata identity columns before consuming
individual features or waveforms. Exclude whole patients touching PTB source folds 9/10 and require
one ECG per patient. No closed Challenge/EchoNext records, age/sex subgroup, new labels or SPH training.

Verify NTT-06 result SHA256
`3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3` and its bound support selection,
pooled vectors, predictions and predecessor receipts. Exclude all 27 reference patients globally from
the clustering cohort: six overlaps leave 1,993 candidates, without selecting on their ECG findings.
The 60 class-balanced NTT-06 targets are not used as the cohort definition. Broadening the cohort does
not create an independent untouched test: these physical-clustering patients have been explored.

Before new cluster scores, reproduce all six native NTT-06 methods from saved reference/target vectors,
all 180 choices and scores, and its exact 2,000-draw primary interval using the frozen summary helper.
Also reproduce CLUSTER-01 raw/residual k-means assignments with ARI=1 and silhouettes within 1e-10
on its original saved 1,999 rows, before support exclusion. No broad label-audit replay is permitted.

## New bounded native extraction

This study authorizes new local CPU inference; it does not extend NTT-06's 87-record receipt. Reuse
its exact official ECG-JEPA source, strict checkpoint loader, preprocessing and embedding identity.
No new weights, dependency changes, downloads, GPU or training. Documented pretraining sources exclude
PTB, but the checkpoint lacks an independently auditable patient-level training manifest; retain that
qualification. Signals remain local and are never submitted to an external API.

A separate source-bound CPU profile loads the real frozen checkpoint and runs a synthetic input plus
the first four permitted candidates in numeric ECG-ID order. Require valid native shape/finite values,
all parameters frozen, at most three seconds per complete real-record detection/feature operation,
peak RSS <=6 GiB and profile stage <=180 seconds. Conservative projected extraction is model-load time
plus 1.5 times the slowest profiled real-record time times 1,993; require <=2,400 seconds. Bind actual
numerical-library/thread/device versions. A completed profile with identical source, checkpoint and
runtime fingerprints gates extraction; a profile proposal or random-weight smoke test cannot admit it.

Extract each candidate once with two CPU threads and a 2,400-second measured wall deadline, checked
between records. Validate raw 500 Hz, ten-second, twelve-lead physical-mV data and hash each source
header/data file before reading. Keep the exact NTT-06 automatic detector and complete-beat bounds.
Prespecified quality exclusions, without replacement, cover invalid signal shape, nonfinite values,
wrong sampling rate/units, or fewer than three complete automatic beats. Missing files, changed hashes,
role/identity failures, checkpoint/model errors or unexpected inference exceptions abort the stage;
they cannot be converted into quality exclusions. All profile records must pass rather than be replaced.

Require at least 95% retention before admitting any clustering. Save every reason locally and report
aggregate retention plus post-selection exclusions by existing annotations and quality, so overall
retention cannot conceal concentrated failures. Intersect the retained patients across every view.
Save unit patient QRS vectors, full-record V1/V6 vectors, each unit QRS-region vector, region offsets,
original-time intervals and row identities. Do not save unnecessary full dense maps. No target disease
annotation is needed for extraction or clustering selection.

## Fixed representation views

All NTT supports remain outside the clustering cohort. Build the three frozen NTT-06 banks, each with
three support patients per NORM/CRBBB/CLBBB finding, exactly as originally defined. A reference bank
represents those directions only; other diseases cannot be treated as absent or forced into a complete
three-diagnosis taxonomy.

1. `native_qrs`: original 768-dimensional normalized average of normalized complete-QRS region vectors.
2. `native_global`: original 768-dimensional full-record V1/V6 pooled vector.
3. `ntt_affinity`: three QRS patient-vector cosine similarities to the three prototypes, averaged over
   the three banks. This is a low-dimensional support-label-informed projection, not another encoder.
4. `ntt_regions`: for every automatic QRS region, compute its three prototype cosines. Summarize the
   mean, maximum and population standard deviation of each finding's similarities, nine coordinates,
   then average these summaries equally across the three banks. This uses the native NTT region
   matching stage without calibrated diagnosis probabilities or expert interval truth.
5. `physical`: original 53 physical features on the same retained patient rows.

For native views fit centered, unwhitened PCA on fitting patients, retain the number explaining 95%
variance subject to cap16, and report actual retained variance when capped. Do not standardize each
native feature coordinate. For NTT views center on fitting patients and retain all three/nine axes;
no PCA, whitening or variance-based axis removal. This preserves the declared similarity summaries.
For physical features retain the prior fit-only median imputation, 1st/99th-percentile winsorization,
robust scaling and centered unwhitened PCA95% capped at16. No nuisance residualization is a primary
view; quality associations are interpretation, not proof that physiology is artifact.

## Fitting, anchor selection and stability

Reuse CLUSTER-02's deterministic identity-only anchor hash (`cluster02_anchor:20261018:<patient>`,
SHA256 integer modulo5 equals0). Anchors never enter preprocessing or model fits. They do enter the
selection rule and are development validation anchors, not untouched test patients. Use all remaining
patients for the reference fit. Refit the entire preprocessing/model pipeline on eight deterministic
80% fitting-patient samples without replacement, seeds20261019 through20261026. Use identical patient
samples in every view/configuration. Cache transformations once per view/sample, never across refits.

Configurations per view: k=2/3/4/8 for k-means (n_init10) and diagonal/tied/full Gaussian mixtures
(n_init3, reg_covar1e-3, max_iter400, tol1e-3). Reference seed20261018, refit seed20261019+j.
This is 16 configurations per view, 80 total and 720 reference/refit model fits. Require at least100
anchors and500 fitting patients; otherwise stop rather than change the split. Bind all patient-sample
hashes and final anchor assignment matrices locally. Bound clustering plus declared control fits to
900 measured seconds, checking the deadline between fits. A timeout preserves partial receipts and
cannot be rescued by dropping expensive algorithms or changing settings after seeing results.

Eligibility requires every fit to converge, all k clusters to contain at least5% separately in each
fit's fitting predictions and each anchor prediction, reference anchor silhouette >=0.05, median
anchor ARI across eight refits >=0.80 and minimum ARI >=0.60. Reject collapsed/empty anchor clusters.
Rank eligible candidates by median stability, minimum stability, silhouette, smaller k, then k-means
before mixture and fixed covariance order diag/tied/full. Choose one winner per view without target
labels or nuisance associations. If none qualifies, retain the same-ranked descriptive fallback,
explicitly ineligible; it cannot pass the primary gate. No diagnostic AMI or enrichment chooses a model.

Fit the selected `ntt_regions` configuration separately in each single support-bank view, with its
complete fitting-only centering/model, and predict identical anchors. Report pairwise anchor ARIs,
all-cluster occupancy and convergence; each single-bank fit must retain >=5% occupancy in fitting and
anchor predictions and converge, in addition to the primary bank-agreement requirement. Cross-view
agreement compares related transformations of the same ECGs and is concordance, not independent
replication. Report every view's selected partition and all pairwise anchor ARIs.

## Support-label and density controls

Cyclic relabeling of class prototypes only permutes affinity coordinates and cannot test clustering
semantics. Instead, generate20 fixed permutations (seed20261019+i) of the nine individual support
labels inside each bank, preserving exactly three supports per named finding and rebuilding the class
prototypes. No target labels enter the permutations. Recompute mean-bank regional affinities and fit
only the already selected regional configuration for each permutation, with fitting-only centering.
Record anchor ARI versus true-reference clusters, occupancy/convergence and silhouette. Do not select
a new model or winner from this control.

After unsupervised selection, describe the true-reference annotation AMI versus the20 randomized-support
AMI values. These are conditional sensitivity comparisons, not a confirmatory p-value: selection was
performed in the original geometry, and source/support uncertainty is incompletely sampled. The
permutation does not invalidate the usefulness of labeled supports for a supervised downstream task.

For an additional descriptive density view, run HDBSCAN on each reference fitting-only transformed
full retained cohort with min_cluster_size40/80, min_samples10 and n_jobs2. This transductive secondary
analysis includes anchor observations in the density fit and has no anchor prediction rule; it cannot
enter selection, persistence admission or a held-out claim. Report both configurations, noise coverage,
cluster counts/sizes and assigned-only silhouette. Do not force all patients into clusters or rescue
noise-only outcomes by changing parameters. It shares the900-second clustering/control budget.

## Post-selection interpretation and artifacts

Only after the label-free selections and primary decision are frozen, parse existing human-read SCP
annotations for the permitted cohort and excluded candidates. Preserve overlapping NORM/MI/STTC/CD/HYP
flags; do not invent new clinical normality or athlete criteria. Report anchor cluster multilabel
prevalence/enrichment with one Benjamini-Hochberg family across every displayed winner's cluster-label
test. Report AMI with existing annotation combinations and device/site/quality, heart-rate, signal
amplitude, baseline fraction and beat-consistency quartiles. No age/sex analysis or etiologic inference.
Density cluster interpretation, if shown, is descriptive and excluded from the primary enrichment family
unless explicitly included in the same family; no separate uncorrected significance claim.

Save source/input/model/cache hashes, exact identities, feature coordinates, fits/sample hashes,
partitions, exclusions, runtime/failures and per-patient artifacts locally under `outputs/cluster03_*`.
Use new code/tests/commands under the clustering topic subdirectories. Protocol/results live here;
only safe aggregate artifacts and standalone figures go in `reports/clustering/cluster03-native-reference/`.
Write a results report from the executed outputs even if a gate fails. Update both queues and rank
follow-ups. No push or merge; PR #71 stays draft. No additional experiment runs automatically.
