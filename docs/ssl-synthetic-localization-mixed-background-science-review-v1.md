# Scientific review of mixed-background localizer training

Prospective review, 22 October 2026. Design GO, conditional on source review,
literal admission, actual-path GPU profiling and independent output verification.
This review contains no new training results, waveform inspection or model scores.

Reviewed coordinator commit30300d2, reproduced locally by214abda:

| Frozen file | SHA256 |
| --- | --- |
| docs/ssl-synthetic-localization-mixed-background-v1.md | 2b4835255962de801a5ad89b621ce6da86b35cd6ffaea9670b52fef6c2d6be6c |
| configs/ssl_synthetic_mixed_background_v1.json | a7121ae3285e29719fe44b5e11827e5eaa57db0961770070da2aca887490ec90 |

## Question and interpretation

Replacing half the training backgrounds is a useful next intervention after the
verified, positive but modest frozen-model transfer result. It tests whether a
localizer learns added numerical changes more effectively on real backgrounds
when some training backgrounds are real. It does not require changing Pulse2Pulse
or assuming its outputs are clinically normal. Generator adaptation is not part
of this experiment.

The comparison changes the identities and source of40of80 training backgrounds.
Matching initialization, geometry, schedules and optimizer controls several
training differences, but does not identify a population-level causal effect of
background source. The single selected real cohort, released generator, selected
generated backgrounds and already spent evaluation panels remain conditional.

## Cohort and representation

The missing v3/v4 manifests and original partition payload remain absent historical
artifacts. This new supervised numerical-target cohort is not claimed to reproduce
or replace a permitted SSL/pretraining cohort. The reviewed source-local selection
uses the original TRAIN hash role and quarantines every patient with any fold8,
9or10 record. The additional fold8 exclusion preserves known calibration separation.
No age, diagnosis, sex, waveform quality or model output selects a patient.

Lowest ECG ID per eligible patient, followed by the frozen patient-hash ordering,
fixes40distinct patients before waveform decoding. A technical failure stops this
version without replacement. Metadata-role exclusions support source-local role
separation; they do not recover absent waveform duplicate catalogs, establish
cross-dataset biological identity, or reveal GAN pretraining membership.

The official metadata and checksum identities, four interpreted metadata fields,
local manifest saved before decoding, selected-file hashes and sanitized structural
reader are appropriate safeguards. All per-record details stay local. A source
review must verify that physical native signals enter original.make_case directly:
its original robust scaling must run exactly once. No post-edit normalization may
alter the final-difference target or destroy nuisance matching.

## Matched learning comparison

Both arms retain80backgrounds, the original four variants,400batch-eight updates,
three matched initial states, original Adam options and positive weight20. Original
Hann editors alone provide numerical supervision. Shared original schedules expose
each slot/variant ten times and yield equal real/generated presentations overall
for the mixed arm. Rebalancing individual batches would change the experiment.

Full generated-only replay is a strong integrity requirement: every loss/component,
final tensor and complete Adam state/options must equal the original B artifacts.
The scientific mixed runs begin from untouched initial tensors and fresh Adam,
including after disposable profiling. No partial state, selected fit, extra update
or evaluation-driven checkpoint can replace the committed comparison.

Zero targets identify absence of an ADDED editor. They do not identify healthy
ECGs or absence of existing abnormalities. Such supervision may suppress existing
findings. Even an improved numerical-localization result cannot establish real
disease sensitivity, safety or clinical specificity.

## Primary comparison and complete engineering decision

The primary is mixed-minus-generated direct AP for real backgrounds with the held-
out biphasic editor, averaged over both modes and three fixed fits. The two primary
checks require gain at least0.01 and conditional95%lower bound above0. The full
14-check conjunction also requires:

- Six checks: mixed edited-minus-nuisance, edited-minus-absolute-input and edited-
  minus-donor AP each have positive point value and positive lower interval bound.
- Two checks: real focal and repeated direct-AP contrasts each have lower bound
  above−0.01, preventing a pooled improvement from concealing a large mode loss.
- Two checks: generated Hann and biphasic direct-AP contrasts each have lower bound
  above−0.01, limiting loss on the previous development source.
- Two checks: real sham and nuisance top600-probability contrasts each have upper
  interval bound strictly below+0.02, bounding background-score worsening.

The control bounds do not require reduction and are not clinical false-positive
limits. Numerical margins are engineering criteria rather than clinical utility
thresholds. Low absolute AP and unfavorable arm/mode results must remain visible.
Passing a subset cannot rescue a failed conjunction.

Whole-slot2000draw resampling with seed2026106701, shared across all contrasts,
preserves geometry matching and paired model comparisons. Modes and the three
fixed fits are averaged within each slot before resampling. Fits are not sampled
as independent patients. Fixed donor assignment remains fixed during resampling;
the resulting intervals are conditional on that control assignment and cohort,
not a donor-permutation null or uncertainty over future training populations.

The same spent64real/64generated panel and exact saved observations/supports make
the comparison useful for development. Reusing it cannot create an untouched test.
Full12-lead numerical masks describe final input differences, not pathology masks.

## Execution, resources and verification

The proposed actual-path profile covers both training banks, complete model/Adam
serialization and restore/replay, frozen-B inference, persistence, hashing and CPU
scoring. Charging the ten disposable updates and all cold/recurrent work is
necessary. GPU-lock time must not include unlocked CPU scoring. Forecasts must
pass before the2400scientific updates; a busy lock or failed gate stops execution.

Streaming the approximately553MB mixed maps and referencing immutable baseline
artifacts fits the1GiB output design without duplicating old trees. Actual memory,
disk and timing guards remain necessary. The reviewed limits are1800CPU/2400wall/
900GPU-lock seconds and2GiB RSS/CUDA. The separately bounded audit has no training
or GPU forwards, independently reconstructs all new metrics/draws and includes24
fixed CPU mixed-state functional replays.

Source/input admission, complete baseline closure, selected-record identity checks
and terminal rehashes must be implemented before execution. Actual costs append
to verified stage37 once, including failed attempts. Historical14of15FAIL, the
negative released-N finding and unavailable older payloads remain unchanged.
Results require actual output-derived main and independent reports. No push,
GitHub runner, disease-label change or closed-test access is authorized here.
