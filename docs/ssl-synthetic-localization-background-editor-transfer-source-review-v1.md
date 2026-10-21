# Scientific source review of the frozen background/editor diagnostic

21 October 2026. Source GO for integration commits `9d306f4`, `d2f2bdf` and final documentation-only
cleanup `2ca6700`, subject to the actual complete identity/header preflight, committed admission,
GPU profile/resource gates and independent verification. No new model outputs, waveform arrays,
intervention outcomes or experimental scores were read for this review. The study remains unrun
at this review checkpoint. Artificial tests and source approval are not experimental results.

The prospective [protocol](ssl-synthetic-localization-background-editor-transfer-v1.md) and
[design review](ssl-synthetic-localization-background-editor-transfer-science-review-v1.md)
remain unchanged. The user authorizes GPU execution and local commits only, with no pushes.

| Final reviewed file | SHA-256 |
|---|---|
| `ecg_experiment/ssl_synthetic_background_editor_transfer.py` | `6d6c535abd2878627c703517362ce8fbdd2d01140b1068e0d421ad808cb1c13b` |
| `ecg_experiment/ssl_synthetic_background_editor_transfer_run.py` | `2757fde86c7750b6532fbefd9d780459667fc4e0b8389901139795e65f3f6e77` |
| `scripts/experiments/ssl_synthetic_background_editor_transfer.py` | `581f7a8226138085c2374d2a730f08734717a5539f7f11237e38a7f0e8d91d69` |
| `tests/test_ssl_synthetic_background_editor_transfer.py` | `c8364ee70cc63b22efaa4345891a6d8357faee5323e8ff416e21c7156019cda2` |

## Input construction and numerical truth

`prepare_inputs` takes the first 64 ascending nonparser real IDs from the saved 160-record bank.
It extracts already-standardized native channels without applying the robust transform again.
Generated raw native arrays receive the frozen transformation once. Both then enter the same
editor. Real canonical shams must equal the earlier input bytes; generated original first-four
variants, strict masks and scoring masks must exactly equal their paired-study artifacts.

The original frozen parameter function supplies the same lead, sign, amplitude, width, centers
and nuisance phase for each real/generated slot. The held-out kernel implements precisely the
prospective float64 Hann-times-sinusoid and L2 normalization, followed by signed amplitude cast
to float32 before addition. Its measured peak, mean and squared L2 are retained. There is no
shape replacement, post-edit normalization, threshold search or diagnosis parsing.

The new constructor measures final canonical float64 differences from the exact nuisance input.
Strict !=0 and abs(difference)>0.01 targets are saved separately. Nonempty/nonfull masks, threshold
subset and declared affected-lead/time hulls are checked. Actual change energy, retained fraction
and residual maxima are saved rather than inferred from an intended window. Canonical lead
arithmetic is the same frozen helper, including induced limb changes.

## Scoring, controls and uncertainty

`score_record` checks the six-variant float32 geometry, finiteness and probability range. It
reconstructs the target from saved final inputs before scoring. Direct AP, same-target nuisance
AP, their response, observed-amplitude AP/gain and actual next-slot donor-mask AP/gain use the
complete 60,000-cell grid. Probability change inside/outside the actual target and their
selectivity difference retain the parent definition. Controls remain separate, with mean, q95,
maximum, stable-sort top-600 mean and the explicitly uncalibrated >0.5 descriptor.

`score_maps` indexes the original geometry slot for donor (slot+1)%64, including profile slots;
it does not redefine donor order inside a profile or bootstrap sample. The complete input/target
bank exists before profile scoring, so slot 7's donor can correctly be slot 8. No overlap filter,
new donor draw or empty-mask exclusion is introduced.

`summarize` requires every source/slot/fit/arm row exactly once. It averages fixed fits and modes
within each slot before a single shared 2,000-draw whole-slot bootstrap. The three primary
components are exactly real/B/biphasic response, amplitude gain and donor gain, with strict mean
and lower-bound positivity: six conjunctive checks. All sixteen source/family/arm cells,
explicitly signed source/family interactions and PK-minus-B secondary contrasts remain visible.
The fixed-donor bootstrap is conditional on that target/geometry assignment, not a new-patient
population or a universal permutation null. No fit is resampled or selected.

The final source also saves all 96 source/arm/fit/positive-variant point summaries. This fixes
the initial draft's omission of explicit per-fit/per-mode summaries while leaving the primary
vectors and bootstrap unchanged. Unfavorable modes and fixed fits cannot disappear behind pooling.

## GPU operations, persistence and resource accounting

All twelve original states are strictly loaded through the frozen architecture helper, checked
against their actual named-tensor parent identities, set to evaluation mode and rechecked after
inference. Parent map replay is retained before its 1e-6 gate. The actual locked CUDA operation,
deterministic runtime, disabled TF32 and minimum free-device memory are required. There is no
optimizer, generator execution or CPU main-inference fallback.

Both source cohorts use the same batch-eight inference, float32 memmap assignment, flush and
hash path for profile and full execution. Profile/full first-eight-slot maps must match exactly.
Each model bank is written/hashed/released before moving to the next; scoring likewise maps one
model bank at a time. This addresses the risk of making all 2.212 GB of maps resident under a
2 GiB process-RSS limit. The conservative retained-byte forecast includes full/profile maps,
observations, masks, replay evidence and reserve before allocation; final exact byte accounting
and all live limits remain mandatory.

The actual profile scores/persists all twelve states on both sources and all six variants.
CPU scoring is outside the shared GPU lock but inside CPU/wall forecasts. Model loading, old
replay and the fixed figure are included in elapsed setup; recurring inference and scoring are
scaled by the prespecified factor and reserves. Separate profile/full lock intervals accumulate.
The final outer-finally repair measures occupancy through lock release and persists phase JSON
after release, including failure evidence. The earlier draft stopped its clock while still
holding the lock during that write; this was corrected before any execution.

The CLI captures scientific clocks before heavy imports. Exclusive output creation, immutable
source/input hashes, original parent output-manifest anchors, exact predecessor reconstruction,
actual stage-34 prior verification and additive terminal accounting are preserved. Successful
main status remains provisional until independent verification. Failed scientific checks do not
become technical failures or select another arm, and a technical failure remains retained.

Eight artificial tests cover original-editor equality, held-out shape/L2, final canonical targets,
matched-control scoring, shared fixed-fit bootstrap and roster completeness, a failed donor
component despite other favorable components, preflight failure and cumulative GPU accounting.
The integration agent reports them and Ruff passing; the reviewer inspected their assertions.
The last `2ca6700` diff changes only NumPy docstrings, not numerical expressions.

No source-level mathematical blocker remains for the concrete admitted protocol. This is a
bounded introduced-change transfer test using spent development backgrounds. Its gates do not
validate physiological realism, disease control, clinical localization, an SSL benefit, a causal
background-source mechanism or new training. The previous paired/suppression conjunction stays
FAIL with fourteen of fifteen gates passing. Clinical definitions, closed tests, local data and
weight ownership, draft PR67 and the user's no-push instruction remain unchanged.
