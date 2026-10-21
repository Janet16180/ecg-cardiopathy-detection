# NTT-08: exact contextual representation contributions

## Question and prospective decision

Without access to a cardiologist, can the frozen native-signal reference classifier expose an exact,
auditable decomposition of its decision into contextual beat and lead/time-token contributions?
This is an engineering and model-behavior study. It does not measure pathological localization,
clinical sufficiency, disease causes or agreement with an expert. A correctly reconstructed score
cannot establish that the corresponding waveform segment contains an abnormality.

Commit this protocol before computing new contributions, concentrations or perturbation results;
commit implementation and meaningful synthetic tests before execution. Scientific review precedes
execution. No model, support bank, target, label or threshold is selected from the new outputs.

The primary engineering gate requires every original score and choice to replay within 1e-12, every
cached patient descriptor to reconstruct from its original dense tokens within 1e-12, and signed beat
and signed token contributions each to sum to the original fixed winner-minus-runner-up margin
within 1e-10 for all 180 original bank-target evaluations. All source, role and identity checks must
pass. Report the maximum errors and each failed gate. This algebraic gate validates implementation,
not clinical highlighting. No descriptive result rescues a failed gate or the NTT-06 primary failure.

## Inputs and exact integrity

Use the unchanged NTT-06 sixty targets, three original three-shot-per-finding banks and 27 disjoint
support patients, all in the frozen PR67 training role. Retain all sixty targets regardless of the
original prediction's correctness. Verify patient roles and whole-patient PTB fold-9/10 exclusion
before consuming feature or token payloads. No new patient, signal, annotation or test-set payload.

Pin `outputs/ntt06_native_encoder/result.json` SHA256
`3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3` and all its immutable
source/input/artifact bindings. Reuse `predecessor_inputs`, `reproduce_predecessor`,
`native_predictions`, `native_summary`, `descriptors`, `regions_from_saved`, `lead_patch_tokens`
and the shared receipt/path helpers. Reproduce the predecessor's complete native and historical
predictions, all native summary values and 165/180 native-QRS correct choices exactly. The historical
physical baseline remains 167/180. Neither count is a new independent accuracy estimate.

Verify dense-cache ECG identifiers against the complete admitted 87-record selection before token
access. Reconstruct all four saved descriptors for all 87 records, including QRS and shifted pools,
from the pinned lead-major 400-by-768 float32 token arrays. Keep all original automatic QRS regions
and their half-open original-second coordinates. Token anchors are 200 ms patches on eight measured
leads, while the original classifier pools V1/V6 QRS regions. No image conversion, waveform parser,
encoder inference, weight training, GPU, download, clinician contact or external data upload.

## Exact signed decomposition

For a target and original bank, rank its original QRS cosine scores descending with lower finding
index breaking an exact tie. Freeze winner w and runner-up r before any intervention. With token
vectors h_t and the unchanged occupancy-weighted region average v_b=sum_t a_bt*h_t, where the
nonnegative weights a_bt sum to one within each region, let u_b=v_b/||v_b||, S=sum_b u_b,
D=||S|| and delta=p_w-p_r. The original margin is delta dot S/D. Define beat contributions
c_b=delta dot u_b/D and beat-token contributions
c_bt=a_bt*(delta dot h_t)/(||v_b||*D). Sum c_bt over beats using the original token identity before
token summaries; overlapping beat windows do not create distinct token observations. Explicitly
save occupancy weights, normalizers, original region/token coordinates and signed contributions
locally. Verify region vectors against the frozen pooling implementation as well as the final sum.

Report positive mass, negative mass, total absolute mass A, cancellation 1-|margin|/A, effective
count 1/sum((|c|/A)^2), effective fraction divided by the full beat count, and absolute-mass shares
of the top 10%, 25% and 50% of beats and unique active tokens. Use ceiling counts and deterministic
original-index ties. For A=0 report null cancellation/effective counts and zero top shares; retain
the case. Do not divide signed contributions by a tiny margin or call these shares probabilities.
Lead summaries sum unique signed token contributions by V1/V6; all other anchored leads have zero
direct pooling weight, which does not imply zero influence on contextual representations.

## Descriptive representation interventions and bank consistency

Remove whole cached beat vectors and renormalize the surviving aggregate. Keep the original winner
and runner-up prototypes fixed throughout. This intervenes on the downstream representation, not
the waveform. A zero surviving vector is an undefined intervention: count it and retain a null
margin/drop. For each target-bank pair report every single-beat margin drop, including negative
drops, and Spearman association with signed contributions; a constant input produces a null
correlation rather than an invented zero. Use SciPy's average-rank Spearman definition.

At fractions f=0.10, 0.25 and 0.50, set k=max(1,min(B-1,ceil(f*B))). Delete the k largest signed
contributions, breaking ties by beat index. Compare the resulting fixed-contrast margin drop with
100 uniformly drawn size-k subsets without replacement within a draw, using a separate PCG64 seed
derived from SHA256(`ntt08:20261020:<ecg_id>:<original_bank_seed>:<fraction>`), interpreted as a
base-16 integer. Duplicate subsets across draws are allowed and counted. Report random mean,
quantiles and selected-minus-random-mean drop without hypothesis-test or independent-validation
claims. Attribution and deletion share the same score algebra.

Across original bank pairs, count changed winner/runner-up contrasts first. Only pairs with the
same ordered contrast enter descriptive signed-rank correlation and top-positive-quarter Jaccard
agreement. Retain the full eligible-pair denominator; do not choose pairs by correctness, finding,
margin or apparent stability. Report null constant-rank comparisons. Aggregate each patient's
three banks before population summaries; 180 evaluations represent 60 people. No clinical subgroups,
age/sex analysis, source-label changes, calibration or referral thresholds. No confidence intervals
are needed for these conditional descriptive results.

## Execution, privacy and reporting

Use the default pinned environment with two CPU numerical threads, at most 1,200 seconds including
input verification, exact replay, decomposition, interventions and writing. Check the deadline
between records. Save typed failure/progress and preserve any failed output directory. Do not
silently retry a failed scientific run. Save source commit, versions, all source/input/artifact
hashes with repository-relative paths and complete private numeric outputs under
`outputs/ntt08_contextual_contributions/`. Public reports under
`reports/localization/ntt08-contextual-contributions/` contain explicit aggregate fields only.
Never export real ECG traces, per-person tokens, coordinates, identifiers, hashes or attribution maps.
Synthetic examples may illustrate the math if clearly identified as synthetic.

The executing coordinator writes results from the live outputs, arranges an independent reconstruction,
updates both queues and ranks follow-ups. Specialist annotation remains unavailable and is not a
completion requirement for this engineering study. Existing clinical-review work stays pending;
no public silver annotation is relabeled as independent expert evidence. Keep all work local and
do not push or merge.
