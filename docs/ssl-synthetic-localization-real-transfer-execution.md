# Frozen real-background transfer execution details

18 October 2026. Additive execution specification for the committed
[study](ssl-synthetic-localization-real-background-study.md) and
[once-cast/admission clarification](ssl-synthetic-localization-real-background-clarifications.md).
Commit this supplement and tested execution code before the actual GPU profile or any model score.
No fitting, checkpoint selection, extra model or clinical label is introduced.

Use only the original fixed final R/S models with deterministic model-v2 output. Require all 192
patients/768 cases to qualify and all four profile-B/all-mode R/S counterfactual CI-lower-bound gates
to pass. Bind the original numerical audit and explicit reproduction closure, original failed
advancement, counterfactual result, data selection, preparation, dependencies and checkpoint bytes.
Do not open the unused original real-TRAIN waveform cache; verify its digest against frozen metadata.

Case rows, saved masks and the two combined float32 probability arrays use GLOBAL case_index 0–767.
The calibration cases are global rows 0–255 and engineering-evaluation cases 256–767. Background
access uses cache_index. Metrics/bootstraps use role_subject_index, with 64 calibration and 128
evaluation subjects. Assert every mapping and all four variants instead of inferring role from a
zero-based subset offset. Store no duplicate edited/baseline waveform banks.

The actual-path profile uses the fixed R checkpoint, eval/inference mode, batch 16, ten warm-up
batches then twenty measured batches. Traverse the calibration-A case list cyclically in its immutable
order, starting batch b at position 16*b. Each batch regenerates observed inputs from cached backgrounds,
checks their immutable waveform/mask hashes, copies the observed float32 arrays to CUDA, computes
native logits/sigmoid, copies probabilities to CPU, checks finiteness, then discards them. Synchronize
CUDA before/after measurement. No AP, maximum-map threshold, target performance or saved probe map
is calculated. Compare model-state digest and source/input hashes before/after; record zero updates.

For 768 cases and two arms there are 96 full batch-16 inference batches. The admission forecast is
actual total profile wall plus twice measured mean batch wall times 96 plus 120 seconds reserve,
under 900 stage and 7,200 cumulative GPU-wall seconds. CPU uses preparation cost plus actual profile
CPU plus twice measured mean batch CPU times 96 plus 120 reserve, under 1,800 stage and 14,400
cumulative CPU seconds. Original/counterfactual prior charges remain recorded. Enforce live deadlines
and entire-stream 4 GiB. The total study storage forecast counts the 192 backgrounds, 768 support/
candidate maps and TWO 768-by-12-by-2500 float32 probability arrays exactly once: 253,440,000 payload
bytes plus exact NPY headers and a shared 64 MiB metadata reserve. Subtract already-present study
artifacts from the additional-space request; never double-count the preparation arrays.

Only after profile PASS and coordinator READY, fill each arm's calibration slices first. Fit its
64-patient joint unchanged/nuisance higher-95% threshold, using strict greater-than flags. Write and
hash calibration.json BEFORE any B inference. Freeze separate hashes of the calibration slices,
since the remaining rows of the combined probability files are deliberately filled afterward.
Verify calibration bytes/slices again after B inference. Save every native R/S map for independent
recomputation. Reconstruct each positive and matching control byte-exactly; the counterpart map is
already present, so input-response diagnostics need no extra inference.

Score all 128 evaluation patients on full-grid exact intervention masks with the original AP,
prevalence and 300-cell expected-tie metrics. Report focal/repeated separately and subject-macro
S-minus-R with 2,000 whole-patient draws, seed 20261023. Report unchanged/nuisance/positive flags,
paired control differences, contextual uniform/raw-voltage/all-anchor candidate maps, and paired
input-response diagnostics. Keep original advancement failed, introduce no new pass rule, and do not
interpret an old-to-new-domain AP difference as a causal domain-alignment effect. Zero masks denote
absence of the added perturbation, not native clinical normality.

Profile and evaluation destinations are new `real_background_v1/profile/` and `evaluation/` directories.
Refuse overwrite and retain source-bound failures. All work remains local; real waveforms, IDs,
masks, predictions and checkpoints never enter tool output, figures, Git or remote services.
