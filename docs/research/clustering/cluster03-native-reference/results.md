# CLUSTER-03 results: stable groups, but reference-dependent geometry

Clustering was executed for both the native ECG-JEPA representations and the No Time to Train
reference-matching outputs on the same 1,993 development patients. Both produced stable partitions
under patient subsampling, but the prespecified persistence gate failed: native QRS and NTT regional
partitions have anchor ARI 0.103, and one pair of reference-bank partitions has ARI 0.387, below 0.50.
Stable clusters therefore do not establish stable disease categories across these representations or
reference choices.

The clearest descriptive finding is that the selected raw QRS clusters associate much more strongly
with heart-rate quartiles (AMI 0.396) than with the existing overlapping annotation combinations
(AMI 0.053). The NTT record-affinity view produces three stable groups, including a small cluster with
22/28 conduction-disturbance annotations, but its representation already incorporates labeled BBB
supports. These are useful exploratory morphology/reference patterns, not new disease diagnoses,
independent subtype discovery, specialist evidence localization or causes of heart disease.

## What was actually clustered

The [protocol](protocol.md) was committed as `98f42b2` before new inference. The existing CLUSTER-01
1,999-patient cohort was verified, and six patients overlapping the 27 NTT support patients were
excluded globally. All remaining 1,993 patients passed native quality checks; none was replaced.
They are previously explored PTB patients inside PR67's training role, with no patient touching source
folds 9/10. The cohort is broader than NTT-06's deliberately balanced 60 targets, but remains development
data and is not an independent hospital or student-screening population.

The same 1,993 patients enter every view. There are 1,582 fitting patients and 411 deterministic
validation anchors; anchors never enter preprocessing/model fits but do participate in selection.
The existing annotation counts are NORM 889, MI 492, STTC 474, CD 445 and HYP 253. These flags overlap;
NORM means the source code is present and does not certify absence of all other findings. All source
annotations, clinical definitions and operating points remain unchanged.

| View | Features used for clustering |
| --- | --- |
| Native QRS | 768-dimensional normalized average of automatic QRS vectors |
| Native full-record V1/V6 | 768-dimensional contextual whole-record vector |
| NTT record affinity | Three prototype cosines, averaged over three banks |
| NTT regional affinity, primary reference view | Mean/max/population-std of three finding cosines over automatic QRS regions; nine coordinates, averaged across banks |
| Physical morphology | The original 53 interval/axis/amplitude features on matched patients |

The NTT views use the actual native reference stage: the frozen NORM/CRBBB/CLBBB banks from NTT-06,
with three support patients per finding/bank and equal patient weights. No support patient appears in
the clustering cohort. NTT record affinity is a low-dimensional projection of normalized encoder
vectors onto labeled prototype directions; regional affinity also summarizes within-record variation.
Neither is an independently trained encoder or a label-free representation of all heart diseases.

Native preprocessing was centered unwhitened PCA, with 95% variance retention subject to cap 16; the
cap actually retained 71.99% QRS and 74.33% full-record variance. Physical fit-only imputation/clipping/
robust scaling/PCA retained 92.13% at 16 dimensions. NTT views retained all three/nine centered axes.
Every transform was independently refitted within each fitting-patient subsample. These differences
in dimension and scaling limit causal attribution of cross-view differences; fixed GMM regularization
also has different relative strength across representations. No parameter was retuned after scores.

## Stability and selected partitions

All 80 prespecified configurations ran: k-means and diagonal/tied/full GMM at k=2/3/4/8 in each of five
views. The reference fit plus eight 80% fitting-patient refits produced 720 model fits. Eligibility
required convergence, at least 5% occupancy in every requested cluster separately in each fitting and
anchor population, silhouette >=0.05, median refit ARI >=0.80 and minimum ARI >=0.60. Selection used only
these geometric/stability quantities, never target annotations or nuisance association.

| Selected view | Algorithm | Anchor cluster sizes | Minimum / median refit ARI | Anchor silhouette |
| --- | --- | --- | ---: | ---: |
| Native QRS | k-means, k=2 | 169 /242 | 0.914 /0.961 | 0.188 |
| Native full-record | tied GMM, k=2 | 29 /382 | 1.000 /1.000 | 0.276 |
| NTT record affinity | full GMM, k=3 | 329 /54 /28 | 0.887 /0.992 | 0.524 |
| NTT regional affinity | tied GMM, k=2 | 368 /43 | 0.935 /1.000 | 0.551 |
| Physical morphology | full GMM, k=2 | 261 /150 | 0.904 /0.952 | 0.195 |

Every selected partition is eligible; qualifying configuration counts are 6/7/8/8/3 respectively.
All configurations, including failed conditions, are retained in the
[aggregate receipt](../../../../reports/clustering/cluster03-native-reference/aggregate.json).
Validation-anchor stability is conditional on this source, cohort and selection rule; it is not an
unbiased population estimate or a clinical validation result.

| Prespecified primary condition | Observed | Pass? |
| --- | ---: | --- |
| Eligible native QRS winner | Yes | Yes |
| Eligible NTT regional winner | Yes | Yes |
| Native-QRS versus NTT-regional anchor ARI >=0.50 | 0.1026 | No |
| Every individual-bank fit converges and has required occupancy | Yes | Yes |
| Every pairwise individual-bank ARI >=0.50 | 0.7261 /0.3867 /0.5844 | No |

The primary persistence gate is false. The three banks use the same fixed algorithm and k with
fitting-only centering; they were not reselected separately. NTT record-versus-regional concordance is
0.5494, whereas native QRS-versus-global concordance is 0.0673. Different views of the same ECGs can
produce different stable partitions. Concordance between related transformations is not independent
replication, and the secondary record-affinity result cannot rescue the primary regional gate.

## Annotation and quality interpretation

The complete unsupervised decision was written and hashed before the target SCP annotations were
joined. All enrichment tests across the five selected views form one 55-test Benjamini-Hochberg family.
No density output or permutation output was used to select or rescue a configuration.

| Selected view | Annotation-combination AMI | Heart-rate-quartile AMI | Device AMI | Amplitude-quartile AMI |
| --- | ---: | ---: | ---: | ---: |
| Native QRS | 0.053 | 0.396 | -0.001 | 0.048 |
| Native full-record | 0.030 | 0.032 | 0.009 | 0.005 |
| NTT record affinity | 0.092 | 0.052 | 0.018 | 0.050 |
| NTT regional affinity | 0.047 | 0.074 | 0.022 | 0.077 |
| Physical morphology | 0.116 | 0.040 | 0.027 | 0.120 |

AMI is an association measure adjusted for chance, not diagnostic accuracy; small negative values
are possible. The table does not identify the cause of the associations. Heart rate and amplitude may
carry real physiology, so a stronger association is not proof of acquisition artifact or confounding.
The native-QRS partition is rate-associated rather than a clean disease taxonomy. The physical view's
annotation association is higher than either NTT view's; the study did not select views by these values.

There are informative, mixed reference-associated groups. In the secondary NTT record-affinity view,
the 28-patient anchor cluster has 22 CD annotations (78.57%, lift 3.20, q=5.48e-9), zero NORM annotations,
10 MI, four STTC and six HYP. Another 54-patient group has 21 MI and 24 STTC annotations. These are not
mutually exclusive diagnoses; the support dictionary already encodes conduction-block directions.

In the primary NTT regional view, the 43-patient anchor group has 18 MI (41.86%), 19 STTC (44.19%),
22 CD (51.16%) and 12 HYP (27.91%) annotations, with only two NORM annotations. Corresponding corrected
q values are 0.00449/0.00274/0.000253/0.02008. This is broad abnormal-annotation enrichment, not a pure
single-disease cluster. The 368-patient group still includes 72 MI, 75 STTC, 79 CD and 45 HYP annotations;
its 52.72% NORM prevalence cannot make it a clinically normal group.

## Reference-label and density controls

Twenty fixed within-bank permutations reassigned the nine individual support labels while preserving
three supports per named finding, rebuilt normalized prototypes, recomputed regional affinities and
refit only the already selected configuration. All 20 fits converged and met occupancy requirements.
Cyclically renaming complete prototypes was not used, because it only permutes coordinate columns.

True-reference regional annotation AMI is 0.04694, versus 0.01104 to 0.02059 across the 20 rebuilt-label
controls (median 0.01333). Anchor ARI between each control and the true-reference partition ranges
0.6186 to 0.6948 (median 0.6499). Support labels affect the geometry/annotation relationship, while much
of the grouping persists. These are conditional descriptive sensitivity comparisons, not a confirmatory
p-value or independent evidence of clinical subtypes. Selection was performed in the true-reference
geometry, and 20 permutations do not resolve support-population uncertainty.

HDBSCAN was run at both fixed min-cluster sizes 40/80 in each reference-transformed full cohort.
These ten fits include validation anchors in their density fit and have no held-out prediction rule;
they are transductive secondary outputs. At size 40, both native views report two clusters with 183 noise
patients (90.82% assigned). All remaining eight density configurations return all 1,993 patients as
noise. No parameter was changed to force disease groups or use density to rescue the primary gate.
Assigned/noise coverage and every configuration remain in the aggregate receipt.

## Integrity, runtime and limitations

[Profile results](profile-results.md) preserve the first reader failure and actual corrected profile.
The original profile stopped before a real forward because it rejected uppercase lead aliases.
[Execution v2](execution-v2.md), committed as `10cddc1`, introduced a separate canonical reader rather
than editing pinned sources. Corrected cache implementation is `61adaa0`. The analysis implementation
was committed as `d43891d`; a synthetic pre-score check then caught NumPy integer PCA metadata that
could not be serialized, and `efbb81b` fixed only that receipt type before any clustering score.
All original scientific settings and failure artifacts are retained.

| Executed stage | Measured wall seconds |
| --- | ---: |
| Corrected actual CPU profile | 11.921 |
| New 1,993-record native extraction | 921.056 |
| Complete clustering, controls and interpretation | 31.004 |

There were 1,993 extraction forwards, four separate real profile forwards, one successful-profile
synthetic forward and one first-attempt synthetic forward. Profile-record features were not silently
counted as free extraction. Cache peak RSS was 1,490,554,880 bytes. The cache contains 1,993 QRS and
full-record vectors plus 22,156 unit QRS-region vectors and their original-time coordinates. Retention
is 100% for every reported annotation and source-quality category; no exclusion analysis hides failures.
No image model input, encoder fine-tuning, GPU, new source download or closed test was used. The
clustering algorithms themselves were fitted, without target labels.

The native predecessor's six methods, all 180 choices/scores and its 2,000-draw primary interval reproduced
exactly, with maximum score error 0.0. Physical predecessor raw/residual k-means assignments reproduced
with ARI 1; silhouette differences were 2.78e-16 and 5.55e-17. The actual inference/model preprocessing,
source/checkpoint/runtime fingerprints, cohort roles and profile admission were bound in local receipts.
The released checkpoint's documented pretraining sources exclude PTB, but no independently auditable
patient-level training manifest is available. Tokens retain full-record/eight-lead context; QRS anchors
are not independent pathological boundaries. No clinical localization endpoint was evaluated here.

Earlier [024b](../../../experiment-024b-multisource-geometry-results.md) already reported clustering of
released global ECG-JEPA features. This study adds matched native QRS/reference-conditioned geometry,
patient stability, bank changes and rebuilt-label controls; it does not claim to be the first ECG-JEPA
clustering experiment. The old 024b feature caches were not rerun or presented as new evidence.

Local receipts are `outputs/cluster03_native_cache_v2/result.json` SHA256
`6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3` and
`outputs/cluster03_native_reference/result.json` SHA256
`64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9`.
Patient IDs, vectors, original-time coordinates, assignments and per-patient artifacts remain local.
Only aggregate receipts and figures enter Git. All work stays local and PR #71 stays draft; no push
or merge occurred. Follow-ups will be ranked separately rather than changing this completed result.

## Independent verification

An independent reviewer verified the cache source/input/artifact hashes, exact 1,993-patient roles,
all 22,156 regional vectors/coordinates and unit norms, and patient pooling (maximum error 0.0).
The reviewer then rebuilt all 720 candidate fits, three individual-bank fits, 20 support-label fits and
ten density fits from the saved features with the unchanged settings. Every saved reference/refit
partition reproduced with ARI 1, all iteration/convergence/occupancy conditions and selections matched,
and maximum silhouette difference was 1.11e-16. Independent affinity reconstruction error was 0.0.
The reviewer also independently reproduced every annotation/nuisance AMI, all 55 Fisher/BH rows and
all 20 permutation AMIs exactly, parsing diagnostic payloads only for the role-verified retained cohort
after the unsupervised receipt was frozen. No age/sex analysis, raw waveform read or encoder inference
was used in verification.

HDBSCAN left its input coordinates unchanged. The equal native-view density counts do not correspond
to identical memberships or noise masks. Aggregate JSON shared fields match the local result and
contain no per-patient payload; the coordinator visually checked all three aggregate figures.
The final integrated CPU suite passed **1,528 tests with six skipped in 65.16 seconds**; the focused
clustering suite passed 33 tests in 3.53 seconds. Repository-wide Ruff and the offline package build passed,
and all three new clustering module files in the wheel match their final source bytes. The
[verification summary](../../../../reports/clustering/cluster03-native-reference/verification.json)
records these checks; detailed logs and audits remain local under
`outputs/cluster03_native_reference_verification/`. No further scientific settings were tried after
the failed primary result.
