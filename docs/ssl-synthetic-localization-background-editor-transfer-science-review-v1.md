# Scientific review of the background-by-editor transfer diagnostic

21 October 2026. Scientific design GO for the prospective protocol committed by the coordinator
as `6d6c65a`. This is permission to finish source/admission review, not an executed result or a
substitute for actual-path GPU profiling and independent numerical verification. No new waveform,
mask or model-output score was read or computed for this review. The user requests GPU execution
and local commits only; no push or GitHub Actions are authorized.

| Reviewed declaration | SHA-256 |
|---|---|
| `docs/ssl-synthetic-localization-background-editor-transfer-v1.md` | `974fc96b1f0cbbc7dd491eeae8559ead41245dba56e824d1595524b93216c19a` |
| `configs/ssl_synthetic_background_editor_transfer_v1.json` | `39d7df5e26552d1286dc187333ec515ab1cd6550f42b362e87dbb42076e72750` |

## Why this question follows the actual results

The released generator supplied usable numerical backgrounds, and correct added-change supervision
beat controls and replicated across three fixed fits. Persistent synthetic background responses
remained. The later real-ECG diagnostic did not support simple positive concentration of B maps at
exact released N timestamps; weak whole-span correlation coexisted with modest upper-tail overlap.
That result does not establish clinical specificity or exclude broader wave-related responses.

A direct introduced-change transfer test is therefore more useful than widening an N window after
seeing a negative result. It asks whether the same existing maps respond to a known change on a
real background and whether that response survives a different intervention construction.
The experiment remains about numerical intervention support, not real disease boundaries.

## Cohort and representation decisions are sound

All 64 generated backgrounds were excluded from the 80 backgrounds used to fit the twelve states.
They have nevertheless already been evaluated in the paired/suppression study and are spent
development evidence. They are not fresh patients or a new untouched test. Current generator
training membership and clinical status remain unverified.

The first 64 ascending nonparser LUDB IDs use a simple prospective ID rule. The remaining 96
measurement and 40 parser records are not selected by performance. Every LUDB record remains spent
public development evidence. Source pairing is by shared numerical geometry slot, not patient,
clinical finding, demographic matching or acquisition device. Exactly balanced 64-by-64 panels
avoid silently changing the geometry mixture through a 160-to-64 source imbalance.

Generated native outputs receive the frozen robust transformation once. Saved real canonical
inputs are already transformed: extract native indices [0,1,6,7,8,9,10,11] without applying it again.
Require derived real shams to equal the saved inputs exactly and the original four generated
variants/masks to equal their parent arrays. This is a strong guard against source-dependent or
double normalization. No filtering, clipping or renormalization follows the edits.

## The held-out family and its truth policy are appropriate

The original positive target used one-polarity Hann pulses. The held-out positive uses a
sign-changing Hann-windowed sinusoid, normalized in float64 to the original Hann L2 energy, then
cast to float32 after the same sign and amplitude. Centers, widths, lead, repetition, nuisance and
source-slot assignments remain paired. This is a held-out target-editor construction; ordinary
ECG backgrounds may already contain biphasic waves. It is not proof of unseen physiological
morphology or a new clinical disease family.

The kernel is nominally zero-DC and L2 matched before addition. Its exact peak and threshold-mask
occupancy differ from Hann; final rounded changes need not have exactly zero mean. Saved kernel
means/peaks/L2, change energy and support sizes prevent concealing those differences. Neither
source-by-family contrast identifies a causal physiological effect of the generator.

The target is the final float64 canonical input difference from the exact nuisance counterpart,
with strict !=0 and scoring abs(difference)>0.01 masks. This includes induced limb changes and
float32 arithmetic. A zero target means no added target intervention, not a healthy ECG. Models
see only observed signals. Intended windows, diagnosis fields and N marks are not disease masks.

## Primary comparisons and controls answer distinct questions

The primary remains fixed B on real backgrounds and the held-out editor, equally averaging focal
and repeated modes, the three fitted states and then the 64 backgrounds. For each positive case:

- Response is AP(edited map, actual target) minus AP(nuisance map, the same actual target).
- Amplitude gain is AP(edited map, actual target) minus AP(abs(observed edited input), actual target).
- Donor gain is AP(edited map, actual target) minus AP(edited map, the fixed donor target).

These ask whether the model changes its ranking, exceeds a simple observed-amplitude reference,
and aligns with its own intervention rather than another case's support. All three mean gains
and their conditional lower 95% bounds must be strictly positive: six predeclared checks, with no
clinical operating point or minimum useful effect size. An arbitrarily small positive effect can
pass this narrow rule; report magnitude and all per-mode/fit values without claiming accurate
localization. A failed component leaves the conjunction failed.

Donor slot (i+1)%64 is fixed before scoring, within source/family/mode. No donor is changed to avoid
chance overlap. The complete bank preserves the support-size distribution under permutation,
but an individual target and donor can differ in size. This is a fixed alignment comparison, not
a universal null. The donor assignment and complete target bank stay fixed during resampling;
do not reassign donors according to a bootstrap vector or select a favorable permutation.
Donor intervals are conditional on that fixed geometry/control assignment and are not a valid
claim about a new patient population or a permutation-null hypothesis.

All sixteen source/family/arm cells and their descriptive transfer contrasts must be shown.
P/K/PK are not candidates from which to select a winner. No secondary cell or new narrow transfer
conjunction rescues the earlier paired/suppression FAIL: fourteen of fifteen gates passed there.
Raw control probabilities and their maxima remain uncalibrated engineering descriptors.

## Resampling and implementation conditions

One shared vector of 64 whole geometry slots is drawn 2,000 times with seed 2026106601 using the
shared patient bootstrap helpers. Each selected slot keeps both sources, both families, both
modes, all leads and all fixed fitted states. Average fixed fits, do not resample them as a training
population. Leads, variants and the twelve checkpoints are not independent patients. No skipped
draws or post-hoc fit/mode subset is permitted. Intervals condition on this development design.

The full-path GPU profile must include both sources, all six variants and every checkpoint,
persistence/flush/hash and the actual scoring path. Keep CPU scoring outside the GPU lock while
including it in CPU/wall forecasts. Load/replay/setup costs remain counted, with actual GPU memory
and lock checks. Retain the fixed profile and failed forecasts; no CPU fallback or hidden retry.

Full map files occupy 2,211,840,000 bytes, already above the 2 GiB process-RSS ceiling if all remain
resident. Open, flush, hash and close one model bank at a time, or explicitly release mapped pages;
apply the same bound during independent verification. Large free disk does not waive process-RSS
or the new 3 GiB output budget. Original scientific budgets remain unchanged.

If AP range assertions or construction-oracle checks are added, carry forward the established
floating-point boundary admission of eight float64 epsilons without clipping or changing raw
scores; avoid exact floating-point AP==1 as a hidden scientific criterion. All score tolerances,
source identities, JSON container schemas and operation counts must be checked before execution.
The independent 96 CPU replays use their declared 1e-5 absolute/zero-relative tolerance; CUDA
parent replay uses 1e-6, while profile/full maps must replay exactly.

The complete old paired aggregate and its failed conjunction, real-map primary and all old draws,
actual successful stage34 and once-only cost lineage are prerequisites. Independent reconstruction
must verify final inputs/masks, all new rows/draws and gates, not only a favorable aggregate.
Technical/resource failure, a scientific negative and a positive narrow transfer result are three
different outcomes. None establishes physiological realism, disease conditioning, clinical
localization, an SSL advantage or authorization to train. Clinical labels, closed tests, local
file ownership and draft-PR status remain unchanged.
