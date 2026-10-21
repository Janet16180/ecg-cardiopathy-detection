# Review of a paired-loss factorial

18 October 2026. Prospective design advice only. No new waveform access, model fitting,
inference or scores were used for this review. The preceding paired-background experiment and
saved-map diagnostic are complete; their frozen decisions remain unchanged.

## Question and admissible conclusion

Cross two initializations, random (R) and existing real-waveform SSL (S), with two objectives,
the existing weighted BCE and weighted BCE plus the fixed paired loss below. The question is
whether adding explicit response supervision improves localization on newly introduced
voltage perturbations, conditional on this architecture, training bank, schedule and initialization.

The diagnostic motivates the question: R had high pre-addition target-site probabilities and
little response, whereas S responded strongly but retained more zero-introduction record flags.
These observations do not establish internal causal features, native disease labels or the cause
of the optimization behavior. A new objective is a hypothesis, not a demonstrated remedy.

The single primary outcome should be direct edited-signal AP, rather than AP-response. A response
gain can otherwise arise predominantly from lowering baseline-site AP without improving edited
localization. The factorial interaction is useful secondary evidence; it need not have a positive
sign for the objective to help.

## Fixed objective and matching

Retain the existing 500 TRAIN patients, all 2,000 fixed inputs and their exact masks, the same
600 updates of four patients by four variants, the original schedule, AdamW settings, positive
weight, architecture, scale and original head. Do not rerender the training bank with a new seed.
Use the exact original random encoder for R and the original SSL encoder for S. No new SSL.

Let p_e and p_b be sigmoid outputs for an edited signal and its already-present matched control.
The model receives one signal at a time. Pairing is used only to construct the training loss.
Let d = p_e - p_b, T be the exact introduced support, and N be every lead-time cell outside T.
For each positive case define:

    L_pair(case) = mean_T max(0, 0.20 - d) + mean_N abs(d)
    L_total = L_existing_BCE + 1.0 * mean_positive_cases L_pair(case)

Average focal and repeated cases equally within each patient, then patients equally in the
batch. Backpropagate through both edited and control outputs; do not detach either branch.
The same matched control may legitimately serve both positives, and its paired-loss gradient
then contributes to both terms. Existing BCE still averages all 16 inputs unchanged. A BCE arm
must use the unchanged BCE path, not an approximate reimplementation.

The hinge is per cell before averaging. A hinge on the mean target response can be satisfied by
a responding subset while the other target cells remain unresponsive. The per-cell version is
better aligned with the direct localization primary. Outside-target consistency opposes a global
output shift; separately measured T-minus-U selectivity remains necessary. Both loss terms are
in probability units. Margin 0.20 and weight 1.0 are fixed engineering choices, not empirically
optimized constants. No weight, margin, duration or checkpoint sweep is admitted.

The losses have compatible ideal targets: edited target cells high, matched controls low,
and unchanged cells low in both. The added term changes the training gradients rather than
introducing a different clinical label definition. It can still fail with finite capacity or
finite optimization. No success across paired learning generally is implied by one tested formula.

Four arms are scientifically useful. Reusing existing frozen BCE endpoints would be adequate
if exact training identity is retained; if all four fits are rerun, require exact BCE final
tensor-state reproduction against the original R/S endpoints before any new calibration or
evaluation. Compare tensor-state digests, not serializer-dependent checkpoint-container bytes.
Stop on a mismatch instead of accepting a nearly identical new comparator.

## One primary contrast and fixed guards

For each of 128 evaluation patients, average focal and repeated full-grid AP within each arm.
Compute the patientwise primary contrast:

    g = 0.5 * [(AP_R,pair - AP_R,BCE) + (AP_S,pair - AP_S,BCE)]

Use 2,000 joint whole-patient bootstrap draws for all four aligned arms, with one new prospectively
committed seed. Retain every patient, both modes and tied-score convention. The twelve conditions
for engineering admission are:

1. Mean g is at least 0.02.
2. Its paired 95% interval has lower bound above zero.
3. For R, the direct edited-AP pair-minus-BCE interval has lower bound greater than -0.01.
4. The corresponding S lower bound is greater than -0.01.
5. The R paired-loss arm has AP-response interval lower bound above zero.
6. The S paired-loss arm has AP-response interval lower bound above zero.
7. The R paired-loss arm has probability T-minus-U response interval lower bound above zero.
8. The S paired-loss arm has probability T-minus-U response interval lower bound above zero.
9. R paired-minus-BCE unchanged-control flag-rate point increase is at most 0.05.
10. The corresponding R nuisance-control increase is at most 0.05.
11. The corresponding S unchanged-control increase is at most 0.05.
12. The corresponding S nuisance-control increase is at most 0.05.

The old R BCE arm's failed selectivity is not a requirement that must retrospectively become
positive. Every arm gets its own frozen threshold from the same 64 new calibration patients:
the higher empirical 95th percentile of each patient's maximum over its two control maps.
Use strict greater-than for flags. Report control paired intervals even though the frozen gate
uses point increases. These are zero-introduction controls, not clinically adjudicated normals.

Predeclare secondary direct-AP effects by initialization and mode; AP-response, baseline-site AP
and edited AP components; T/U/O response; positive flags; and the initialization-by-loss
interaction:

    interaction = (AP_S,pair - AP_S,BCE) - (AP_R,pair - AP_R,BCE)

Also report S-minus-R under each loss. No interaction-sign gate, favorable-mode selection or
post hoc switch of primary is allowed. The primary supports an average objective benefit,
not a claim that both initializations or modes improve. Secondary intervals are descriptive,
without a family of separately promoted discoveries.

If the primary passes but a guard fails, report the measured direct benefit and the failed
engineering admission separately. If the primary fails, the finding is that this fixed formula
did not demonstrate the prespecified benefit under this setup, not that paired learning is
ineffective. The original experiments' failed decisions stay unchanged.

## Fresh patients and technical truth

Hash-freeze 64 new calibration and 128 new evaluation patients from the original permitted
VALIDATION partition, excluding all 384 previously used real-validation patients. Preserve
the engineering-test patients, folds 9-10 and other protected test cohorts. Use the same
lowest-record selection and unchanged algorithmic anchors/editor/gates, profile A for
calibration and B for evaluation. Keep native labels unknown.

Freeze all new patient lists before payload access. Qualification failure stops the selected
bank without replacement, subjective exclusion or altered gate. Bind the old TRAIN bank and
qualification, and qualify all 192 new backgrounds and 768 new cases before profiling.
Truth remains the exact introduced post-processing difference, not exhaustive native pathology.

All four arms use identical new cases. Calibration outputs and thresholds are frozen before
evaluation output generation. Explicitly distinguish old TRAIN mask indices, new background
cache rows, new mask/map rows and within-role patient indices. Add artificial tests for the
first calibration row, first evaluation row and last row, plus paired-control identities.

## Feasibility and evidence limits

Existing cumulative experiment costs are about 569.934 CPU seconds and 226.413 GPU wall seconds.
The existing ceilings remain 14,400 CPU seconds, 7,200 GPU wall seconds and 4 GiB local output.
Do not raise them because the hypothesis is promising.

Reuse the old TRAIN background/mask bank without copying it. One new float32 background bank,
two new bool mask banks and four combined float32 probability maps require:

| New payload | Shape/count | Bytes |
|---|---|---:|
| Backgrounds | 192 x 12 x 2500, float32 | 23,040,000 |
| Support and candidate | 2 x 768 x 12 x 2500, bool | 46,080,000 |
| Four probability banks | 4 x 768 x 12 x 2500, float32 | 368,640,000 |
| Total array payload | Seven arrays | 437,760,000 |

With seven 128-byte NPY headers and a 64 MiB metadata/checkpoint reserve, the prospective
increment is 504,869,760 bytes. Current storage near 3.48 GB leaves enough headroom for that
design, but admission must use measured current bytes and actual header/checkpoint sizes.
Do not archive duplicated edited/baseline waveforms, training banks or separate copies of
calibration map slices. Freeze slice hashes instead. Source-bound audit files and failures
also consume the cap.

Profile the actual four-arm code paths, including paired gradient flow, real batch copies,
optimizer state, model output and new inference path. Require deterministic replay and
next-update recovery, finite gradients/outputs, and a conservative forecast covering all
fits, calibration/evaluation, checkpoints and reporting reserves. Verify an artificial paired
case sends gradients to both branches, keeps BCE exactly unchanged and handles the existing
matched-control indices correctly. Real GPU admission, shared lock and cumulative live limits
remain mandatory.

A single matched schedule is informative about the conditional recipe effect and makes the
four-arm comparison feasible. Exact BCE reproduction removes implementation drift as an
explanation for changed comparators. It does not establish robustness over training order,
random initialization, SSL runs, hospitals, editors or native clinical conditions. Patient
bootstrap intervals quantify patient sampling conditional on these fitted models only.

The next important result can be a verified positive or negative answer to this one frozen
objective question. It does not require another adaptive search if the result is unfavorable.
Expert annotation recovery, if trustworthy artifacts actually exist, remains a separate
truth-quality task; seven persistent zero-introduction flags alone do not supply clinical labels.
