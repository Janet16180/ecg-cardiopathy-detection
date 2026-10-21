# NTT-10: matched-union control of reference contribution consistency

## Question and prospective primary classification

Does normalized five-bank aggregation add conditional beat-ranking consistency beyond using the
same unique support people uniformly? This supports the native ECG reference-comparison objective
and its lead/time evidence. It does not test clinical localization, disease etiology or new label
definitions. QRS windows and V1/V6 are inherited fixed pooling priors; no independent selection among
P, QRS, ST and T evidence is established. NTT-10 is unrelated to deferred historical Experiment 010.
Commit the scientifically reviewed protocol before new scores, and source/tests before execution.

For each unchanged target, compute the mean of defined positive-quarter beat-set Jaccards across
all 45 pairs among ten fixed blocks, separately for original normalized-bank aggregation A and
uniform unique-patient aggregation B. Define the patient's difference as mean_A minus mean_B; then
average these differences equally across all 60 targets. Freeze resolution tau=0.05 Jaccard points:

- Difference >=+0.05: material conditional consistency advantage for A.
- Difference <=-0.05: material conditional consistency advantage for uniform union B.
- Absolute difference <0.05: no material mean difference at this operational resolution.

These symmetric categories are finite-design engineering descriptions, not population equivalence,
noninferiority or clinical adequacy. If any patient mean in A or B is undefined, or either method
has fewer than 43/45 defined pairs within ANY patient, classify the primary as unassessable. Retain
all 60 targets, signed effects, absolute agreement levels, coverage and lower-tail patient summaries.
Do not replace missing values with zero or average a favorable subset. This classification does not
select an aggregation method for deployment. Secondary method C cannot change the primary category.
Report the full patient-difference distribution and counts by sign and by the same +/-0.05 materiality
bands, including undefined patients. Opposite patient effects must remain visible even when their
mean is small.

The positive-quarter set uses strictly positive contributions among the first ceil(B/4) beats in
signed-descending, original-index tie order. Both-empty sets have null Jaccard; one-empty gives zero.
Preserve all pair denominators, negative fixed margins, zero masses and constant-rank nulls. Shared
supports and patients make pairs dependent; no independent-pair confidence interval or p-value.

## Frozen inputs, roles and predecessor integrity

Use exactly NTT-09's 60 targets, 27 disjoint supports, 50 three-shot banks and ten consecutive
five-bank blocks. Each patient's ordered winner/runner contrast remains frozen from NTT-09's
full-nine-per-finding prototypes, including negative margins under any new method. No reranking,
target annotation selection, new patient, waveform, encoder inference/training, GPU, download,
age/sex analysis, clinical annotation, closed Challenge/EchoNext test access, upload or external
contact. Verify original reservation and whole-patient fold-9/10 exclusions before feature payloads.

Pin NTT-09 result SHA256 `009eacb3c72b0dcfe85e155a9036542cc504e788ddd7179e097860fe4f4e8f64`
and all its immutable source/input/artifact bindings and completed status. Reuse frozen identity,
cached geometry, fixed-contrast, pair, mass and patient-summary helpers without editing their files.
Before new comparisons reproduce all 3,600 NTT-09 decompositions, saved geometry/prototypes,
76,200 pairs with all four metrics and null masks, all family/patient aggregates, exposure counts
and its primary decision. Require exact identities, choices, integer counts and null masks; numeric
arrays and summaries must agree within 1e-12. Preserve the existing 87-descriptor/native-score
replay and prior NTT-08 bindings. This successor need not repeat NTT-08's 54,000 random deletions.
Abort on any role, identity, hash, source, replay or undefined-normalization disagreement; preserve
typed failure/progress rather than dropping or retrying a scientific fit.

## Matched-union aggregation and analytic identity

For each block and finding, let U be the exact union of its five three-shot support sets. Every
method uses those same people and their unchanged unit patient-QRS vectors u_i. Canonically sort
patients by numeric patient ID, then ECG ID before each weighted sum. Let s_j=sum_{i in bank_j}u_i,
q_j=s_j/||s_j||, and n_i be the number of banks in the block containing patient i. Define:

1. A = unit(mean_j q_j), the original NTT-09 normalized prototype ensemble.
2. B = unit(sum_{i in U}u_i), uniform weighting of the exact unique support union.
3. C = unit(sum_{i in U}n_i*u_i), occurrence weighting without individual-bank normalization.

Analytically reconstruct A as unit(sum_i alpha_i*u_i), with
alpha_i=sum_{j containing i}1/||s_j||. Save all five bank norms and normalized patient weights
alpha_i/sum(alpha), 1/|U| and n_i/15. Require direct versus analytic A prototype error <=1e-12 and
their fixed-margin/beat/token contribution errors <=1e-10. Require direct A to match NTT-09's
saved ensemble prototypes within 1e-12. Identical per-finding union signatures must produce exactly
identical B prototypes; canonical summation prevents iteration-order artifacts. Undefined region,
bank or aggregate normalization is an execution error, never a favorable exclusion.

Record distinct union counts per finding and distinct three-finding union triples across the ten
blocks, together with counts of the 45 block pairs sharing each signature. Repeated unions can
mechanically improve B consistency and do not create independent validation observations. All
methods match unique patients within each block; A/C retain 15 occurrence slots per finding, while
B uses one contribution per unique patient. Different weighting and arithmetic costs remain, so
this is not an equal-computation comparison and does not isolate every consequence of aggregation.

## Secondary descriptions and execution

Prespecify C-minus-B and A-minus-C patient-mean Jaccard differences as mechanistic descriptions.
Occurrence weighting and individual-bank normalization act through nonlinear prototype and rank
operations; these differences are not independent causal effects. Report within-method signed and
absolute rank correlation, absolute-quarter Jaccard and all null/tie counts. Report paired A/B/C
prototype cosines and same-block contribution-map agreement, margin reversals, patient-first mass,
cancellation, effective fraction, absolute top shares and signed/absolute map variability using the
unchanged NTT-09 definitions. Save normalized weight effective support count 1/sum_i(w_i**2) and its
fraction of |U| for each method/block/finding. Report raw absolute agreement even when the primary
classifies a small difference; two weak methods do not become adequate through similar averages.
For matching blocks compare A/B, A/C and B/C positive-quarter set Jaccard and signed/absolute rank
correlation, absolute fixed-margin differences and prototype cosine/L2 differences. Also report raw
signed contribution L1 differences and total variation between normalized absolute maps,
0.5*sum(abs(abs(c_first)/mass_first - abs(c_second)/mass_second)), separately for beats and unique
active tokens. Total variation is null if either absolute mass is zero. Average blocks within each
patient before population summaries and preserve all undefined counts. Small differences in mean
within-method stability must not be described as equal highlighted sets or equivalent maps.

Use two CPU threads and a measured 1,200-second ceiling including role/hash verification, exact
predecessor replay, 1,800 new A/B/C target-block decompositions, 600 analytic-A checks, all summaries
and writing. Check time between patients. No dependency or pinned-file changes. Save source commit,
runtime, versions, all source/input/artifact hashes through repository-relative stored paths and
complete local identities, unions, weights, norms, geometry, contributions, pairs and manifests
under `outputs/ntt10_matched_union/`. Only explicit aggregate fields and standalone aggregate figures
enter `reports/localization/ntt10-matched-union/`; no individual identities, coordinates or maps.
The executing agent writes results from live outputs, including unassessable or negative outcomes;
independent reconstruction follows. The coordinator owns queue/backlog/index updates. No specialist
labels are invented and no result substitutes for expert review or rescues NTT-06's failed primary.
Private data, identities and maps stay local. Under the renewed 21 October user authorization,
source, protocols, tests and aggregate reports may be pushed to the existing draft PR #71 by the
coordinator. No PR merge, clinician contact or private data upload is authorized.
