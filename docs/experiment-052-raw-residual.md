# Experiment 052: raw within-ECG morphology residual localization

Frozen 2 October 2026 before any experiment score. This is an unsupervised location map for an already
referred ECG. Referral, clinical labels and existing detection models remain unchanged.

## Method

Read only PTB-XL training normals and the 1,604 development ECGs already used by 042. Existing `r_peaks`
align beats; RR intervals are never anomaly features. Extract each complete beat at R - 0.25 through
R + 0.45 seconds. Subtract its lead-specific median in R - 0.10 through R - 0.06 seconds. Its prediction
is the leave-one-beat-out median of the other complete beats. Divide absolute residuals by their
pointwise median absolute deviation times 1.4826, with a 0.02 mV floor. Assign samples to beats using
adjacent R midpoints, so a preceding beat's T span cannot own an early complex. No fitted disease label,
supervised head, external API, download, GPU, calibration waveform or closed test waveform is used.

The primary `raw_residual` map averages squared normalized residuals in a 140 ms window, at 10 ms stride,
for each of the 12 leads. `U_B_fixed` broadcasts the saved U_B piece scores to their exact lead/time
support, taking the maximum on overlaps, and averages in exactly the same candidate windows. Each map
displays its highest scoring window. The fixed width cannot change after inspection. The minimum MAD
floor prevents zero-MAD or flat leads from producing infinite scores. Fewer than three complete beats
is an explicit exclusion, paired in every comparison.

## Integrity before new scores

Receipt-check 042's unit scores, its result receipt, the metadata and relevant historical helper sources.
Reconstruct 042's full rows and automatic premature-beat windows; reproduce every field in the existing
U_B map metrics, including bootstrap intervals, with seed 42042 and 2,000 draws. Any discrepancy stops
the run. New shared forward-only library helpers can factor the necessary frozen historical runner
functions once; new code never imports scripts and frozen sources are unchanged.

## Primary development endpoint

For 041's 73 PVC ECGs with an automatic early-beat window, use the R peak that starts each window
(window start + 0.10 seconds) and its QRS support R - 0.06 through R + 0.08 seconds. A temporal hit
requires the center of the top 140 ms window to lie inside that support. Uniform chance is the share of
the identical candidate windows whose centers do so; leads are not PVC location truth. Report the paired
hit-minus-chance gain, median distance of the top center from the closest target R, target support
overlap fraction, and exclusions. Patient-bootstrap intervals use 2,000 draws and seed 52052.

The raw map improves on U_B_fixed only if the point gain is at least 0.10 and its paired interval lower
bound is above zero. These RR-derived automatic targets are exploratory proxies, never expert beat
annotations. Legacy 041 overlap and strict beat-ownership hits, worst-window AUROC/AP, red shares at
each map's development-normal 95th percentile, and lead contrasts are descriptive. They cannot change
the primary rule. A within-record map subtracts persistent disease and is not a replacement detector.

Pre-score clarification: both maps use only candidate windows fully covered by the same complete-beat
support mask (midpoint-owned R - 0.25 through R + 0.45 segments). No zero-filled unscored gap is eligible.
This mask is intersected with actual saved U_B piece coverage, whose original edge rule requires
R >= 0.30 seconds. Raw extraction stays at R >= 0.25 seconds, but unsupported historical edge windows
are ineligible for both maps.
For both maps, tied maxima use `numpy.isclose` with rtol 1e-10 and atol 1e-10. Primary hit, joint hit,
distance and overlap are expectations over every tied maximum, not earliest-argmax tie breaking.
Report tie counts. Figures use the median-time tied maximum for a deterministic descriptive mark.

## Independent exact-support synthetic checks

Select 200 training NORM-only ECGs from unique patients by ascending ECG ID, excluding every development
patient. These ECGs are held out of any fit in this experiment; there is no population fit. With seed
52052 choose an interior complete beat and lead(s) before evaluating the map. Keep R peaks fixed to the
unmodified ECG in every perturbation/control, so rhythm changes cannot create synthetic success.
Eligibility requires at least five complete beats, ensuring an interior target and another interior
beat for the time-shift control. Skip counts and ECG IDs are recorded; eligibility never depends on a
score. The time-shift control moves the ST edit to the first other interior beat. The lead-permutation
control cyclically permutes all twelve leads and moves the known ST lead identically.

1. ST: add a 0.15 mV raised-cosine bump on one random lead at R + 0.10 through R + 0.20 seconds.
2. QRS: replace the QRS of three random leads by a 1.6 times temporally stretched median QRS of other
   complete beats from the same ECG, on R - 0.096 through R + 0.128 seconds; taper replacement edges.
   This is a synthetic morphology transplant, not a physiological ectopic diagnosis.
3. Sham: unchanged copies. Nuisance: a constant 0.15 mV offset on all leads and a 0.05 mV 0.2 Hz
   baseline drift. Time-shift and lead-permutation checks verify the injected location follows the edit.

For each edit, success requires the top window center inside the exact edited time support AND its lead
among the changed leads. Report joint success, support-overlap fraction, distance, worst-score paired
change against unchanged controls, and nuisance false flags using the unchanged normal 95th percentile.
Supporting evidence requires at least 80% joint localization separately for ST and QRS, with the paired
score-change interval lower bound above zero. Loose overlap alone is insufficient. Synthetic checks
show recovery of known signal edits, not correctness on human pathology. Do not present synthetic
comparisons against U_B without recomputing its raw-input encoder/reference pipeline.

Promotion for user review requires both the real primary improvement rule and both synthetic supporting
rules above. A synthetic success alone does not promote the method. The nuisance threshold fitted on
the same 200 unchanged records is an in-sample descriptive calibration, not an independently validated
normal false-positive rate.

## Outputs and review

New code is `ecg_experiment/raw_residual052.py`, `ecg_experiment/localization_inputs052.py`, and
`scripts/experiments/run_raw_residual052.py`, with meaningful tests. Limit BLAS and OpenMP threads to two.
Write source/protocol/input/waveform hashes, seed, exclusion counts and receipts into
`outputs/experiment052_raw_residual_v1/`, including `result.json`, maps, per-record tables and `run.log`.
Do not overwrite a prior run; stage into `.partial` and rename on success. Save local review figures for
the seven established examples (47, 219, 8, 184, 287, 30, 69), including complete-LBBB ECG 287, and the
lowest-ID real win, real loss, synthetic success and synthetic failure where available. Each compares
equal-duration marks; clinical/anatomical interpretations are left to the cardiologist. No waveform
leaves this machine. The running agent writes `docs/experiment-052-raw-residual-results.md` from these
live outputs and reports all negative findings and deviations.
