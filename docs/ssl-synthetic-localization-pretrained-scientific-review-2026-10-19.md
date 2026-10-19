# Scientific review of the pretrained generator continuation

19 October 2026. Source review supports one bounded standalone Pulse2Pulse technical probe, followed
only after its reviewed PASS by the separately frozen 128-seed quality/support study. This document
records design and source inspection, not generated-sample results. The executing coordinator must
write results from retained local outputs and preserve failures.

## Reviewed contract

Inspected the full new standalone runner, protocol, configuration and artificial orchestration tests
in generator commit `2a7c36cb4fd335d3299df1b2952d233b0bd3e9f2`, plus its shared adapter's AST comparison,
restricted selected-class loading, strict finite tensor-state checks, canonical-lead construction,
forward/export implementation, output gates and CPU configuration. Reviewed new runner SHA-256:
`f9146d94e925f19672f4e9a49659bb3df83f0a8729e23f722064904c0821eb0f`.

The exact six CPU forwards comprise one profile, four prespecified seeds and one same-input replay.
The gates measure reproducible finite geometry, diversity, state integrity and arithmetic closure.
The canonical array is derived from the author's integer export; these values are not established
mV. The source does not attach healthy/diseased labels or regional disease truth. Missing historical
receipts remain explicitly unavailable; new measured costs do not quietly absorb reported old costs.
No scientific blocker was found in this contract. Numerical implementation verification, passing
artificial tests and committed integration remain the coordinator's separate launch conditions.

The proposed next study was frozen in `e899cf6`, protocol
[generated quality v1](ssl-synthetic-localization-generated-quality-v1.md). It deliberately separates
operational support integrity from descriptive signal quality. Its paired circular-shift comparison
asks whether independent-channel derivative-energy events align in time. Algebraically derived leads
are excluded from that contrast to prevent a constructed dependency from inflating it. Neither a
positive contrast nor visually plausible traces establish physiology. Independent-lead scaling also
changes native amplitude relationships, so the transformed arrays are explicitly dimensionless.

Focal and repeated edits remain separate, preserving the historical warning that their performance
can move in different directions. Masks describe the final numerical intervention after lead
reconstruction; subthreshold differences are retained and quantified. Paired-difference AP is an
oracle closure check, and fixed lead/time fixtures make position priors trivial. These particular
fixtures cannot support a useful localization learning claim or train/test comparison. A further
learning study would need independently varied placement and edit families, then a genuinely held-out
seed/family assessment and a profiled training path. No classifier or screening comparison is needed
for the present generator question.

## Public annotations that can reduce dependence on a new clinician

Only public documentation was read for this assessment; no waveform, annotation payload or clinical
report was downloaded. These are candidates for a future prospective study, not newly admitted data.

| Resource | Verified annotation scope | Useful next question | Constraint |
|---|---|---|---|
| [LUDB 1.0.1](https://physionet.org/content/ludb/1.0.1/) | 200 ten-second twelve-lead recordings; expert P/QRS/T boundaries and peaks | Validate an algorithmic timing anchor or delineation output against existing expert boundaries | Wave boundaries locate normal waveform components as well as abnormal ones; they are not disease-region masks |
| [QT Database 1.0.0](https://physionet.org/content/qtdb/1.0.0/) | 105 two-lead records; selected manually delineated beats; second annotator for 11 records | Independent waveform-boundary evaluation with explicit uncertain/unalabeled regions | `.q1c`/`.q2c` are manual; `.pu`, `.pu0`, `.pu1` are automatic. Unannotated beats must not become negatives; constituent source records need overlap checks |
| [European ST-T Database 1.0.0](https://physionet.org/content/edb/1.0.0/) | 90 two-hour, two-channel records from 79 subjects; expert ST/T episode starts, extrema and ends, with reconciled annotation | Evaluate a narrowly defined real ST/T episode localization task if separately adopted | Episode support spans time, not exact abnormal lead-time waveform cells; repeated records share patients, and this population/task differs from student screening |

LUDB is the closest geometry match for examining timing anchors without recruiting a new cardiologist.
QTDB provides a complementary boundary check but requires explicit manual-annotation eligibility.
European ST-T is potentially closer to a true localized-change endpoint, but adopting it changes the
research endpoint and cannot silently redefine the project's binary clinical labels. Its annotated
axis-shift episodes offer an additional nuisance category, according to the
[official annotation specification](https://www.physionet.org/physiobank/database/edb/annotations.shtml).

Ranked follow-up recommendations for the coordinator's backlog are: first review the actual fixed
samples and descriptive quality outputs; then qualify varied synthetic supports without training;
then design a small LUDB anchor validation with prespecified patient/source-overlap admission. A
separate varied-support learned-localization comparison is justified only after these input/truth
checks and actual-path training profiling. Access to public expert annotations makes that route
possible without adding a clinician recruitment requirement.

## Quality implementation source review

Before any quality-study GPU profile or score, reviewed the complete transformation/metric helper,
CUDA orchestration, command wrapper, seven artificial tests, original protocol/config and separate
predecessor admission amendment. Scientific GO applies to these reviewed helper/runner digests:

- `ecg_experiment/ssl_synthetic_generated_quality.py`:
  `3de5105471558685f532da2df5bc3d70e86874d30168074820c52e3a2f9a1cb4`.
- `ecg_experiment/ssl_synthetic_generated_quality_run.py`:
  `534cfa134efc339a75bbb78edbccb1c03715a931a4b5c158e00613610fe5a96b`.
- `tests/test_ssl_synthetic_generated_quality.py`:
  `5e9072ab288d0bfa623862b8fd4f9b6f89c4e60237fdddb6ff94aa05a7c90e7a`.

Pre-execution review caught a missing aVL cell group in the affected-lead hull, failure-path GPU cost
loss, missing consumed-predecessor array bindings, and delayed persistence of completed signals.
The reviewed version fixes all four: independent-II edits propagate to canonical indices 1–5;
GPU-lock cost is updated in a finally block; predecessor assets and saved outputs are checked against
literal admission anchors and rechecked at completion; and each generated raw array is flushed before
quality processing, with replay saved before device metadata collection. Attempted and completed
forwards are recorded separately. Fixed noise is saved before profiling. No scientific samples or
metric values informed those corrections.

The helper implements the frozen float64 quantiles, dimensionless float32 transforms, derivative
activity/Jaccard comparison, paired seed bootstrap, half-open periodogram bands and final-array
intervention supports. The gate list excludes clinical quality and includes the constructed limb
algebra, exact replay/sham closure and numerical support checks. CUDA is deterministic with TF32
disabled; the same actual path is profiled before the 128-member bank. The separate admission
amendment binds the immediate predecessor and its audit and reserves additive saved-output audit
resources. Source review therefore finds no remaining scientific blocker for this exact bounded
study after committed integration and the coordinator's required test/numerical-review checks.

## Runtime-only v2 amendment review

Read the actual v1 `profile.json`, `failure.json` and terminal ledger. The attempt correctly stopped
before generating the bank: four predecessor CPU forwards and one GPU profile completed, with zero
bank or replay forwards. Its conservative wall forecast was 1,550.6452264455147 seconds against a
1,200-second limit. The terminal ledger retains 0.772192178 CPU seconds and 6.083721050992608 GPU-lock
seconds. That is a resource-profile failure, not an unfavorable primary quality comparison.

Scientific GO applies to the new v2 protocol/config and successor controller SHA-256
`fed0c9a93a87c69b25005aa7781322beda3c6091e22d50c42d404035e529f0b8`, reviewed with its four new
artificial tests. Only wall/GPU-lock limits increase to 1,800 seconds. CPU, memory, storage, profile
multiplier, fixed seeds, numerical operations, primary comparison and operational gate definitions
stay unchanged. The controller explicitly rejects other configuration differences and directly
reuses the frozen numerical/GPU functions. A new actual profile must still pass. The failed attempt
is bound and preserved, with no endpoint reclassification or favorable seed/device selection.

The additive ledger uses four distinct stages' own costs: standalone main, standalone audit, failed
v1 and its narrow failure-metadata audit. It does not sum their cumulative totals. New v2 costs are
added once; unavailable older factorial costs remain unknown. This transparent prospective resource
amendment is scientifically appropriate for the authorized bounded generation task. Numerical review,
committed integration and passing tests remain separate launch conditions; this review creates no
v2 result and authorizes no training or clinical interpretation.

## Saved-result scientific interpretation review

Read the completed v2 main summary and terminal ledger and inspected all three saved figures without
new inference or new scoring. The main summary SHA-256 is
`0067dc16c609a7209cb626e8d31600f6b654d192e3b0803968ca3275db1660b2`. All 15 reported operational gates
pass, with 128 completed seeds and the exact authorized forward counts. The reported paired activity
coincidence difference is 0.8094248365473182, seed-bootstrap interval
[0.8038931804338517, 0.8148564292963815]. This supports the narrow conclusion that high derivative-energy
activity is aligned between independent generated leads relative to independently shifted controls.
It is not an expert waveform-boundary score or clinical quality threshold.

The predetermined first-eight gallery visibly contains repeated, aligned sharp events in the native
signals. The intervention figure shows the fixed temporal windows and changes propagated through the
five affected limb leads; it does not depict verified disease. The summary and plots are consistent
with the intended engineering interpretation. Figure inspection is qualitative and adds no clinical
verdict or sample-selection rule. Numerical verification belongs to the separately executed audit.
The executing coordinator's results report should retain the v1 profile failure, unknown physical
calibration, generator provenance limits and the fixed-support shortcut caveat. No further model
training or claims about real abnormality localization follow from these saved results.

## Randomized-support learning source review

The new [learning protocol](ssl-synthetic-localization-randomized-learning-v1.md) was frozen in
`68a4931` before new randomized interventions or fits. Scientific GO applies to implementation commit
`b612efc` and these reviewed digests:

- Numerical helper `ecg_experiment/ssl_synthetic_randomized_localization.py`:
  `a8af5f609994ace4ce93acb0e8217ff85285d969ef9003f58267162a0c87d6c0`.
- Controller `ecg_experiment/ssl_synthetic_randomized_localization_run.py`:
  `9fb927be7459a82e0790fb786b42e63c7327e9f5cea90475bd7b6df982afb169`.

Reviewed the complete numerical helper, controller, wrapper and artificial tests. Parameter draws,
canonical-lead hulls, final-array masks, training-only class weights and mask derangement, fixed
observed-only network, identical initialization, interleaved batch schedule and all twelve scientific
gates match the prospective design. The secondary spatial prior uses training masks only. Both arms
finish fitting before evaluation; paired responses use matched nuisance maps only for scoring. The
same grouped seed-resampling sequence underlies pooled and mode-specific intervals.

Pre-execution review identified missing full step-two recovery verification, delayed offending-case
persistence, missing explicit sham/limb checks, and a draft arm-by-arm schedule differing from the
interleaved protocol. The frozen source fixes these before any real optimization. Original and
recovered step-two and step-three model/Adam states and losses are checked exactly, while scientific
fits start from the common initial model with fresh Adam. Failed support cases are retained before
rejection. Actual full-path profiling separately records cold work, recurring steps and inference;
CPU, whole-wall and GPU-lock forecasts and guards use their declared separate limits.

This is a scientifically useful bounded test of learning introduced changes because locations,
leads, widths, signs and amplitudes vary, background seeds are held out from fitting, and the matched
permuted-target arm controls for learning broad target priors. It remains one smooth-offset family,
one fitted randomness and generated-background evaluation. A PASS would not establish disease
localization, physiological perturbation validity, self-supervised advantage or an untouched patient
test. No such claims are encoded in the source verdict. Independent numerical review, committed
integration and required tests remain separate launch conditions.

## Learning result interpretation review

Read the executed learning summary, profile and saved control metrics, and inspected both first-eight
mode figures plus learning diagnostics without new inference or scoring. Main summary SHA-256:
`fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f`. Its twelve scientific and nine
integrity gates pass. The supervised pooled direct-AP gain is 0.053016083946753285 versus amplitude,
95% interval [0.02736853182550996, 0.08305224283469469], and 0.05618453026201692 versus permuted
targets, [0.031927902910476966, 0.0836068382567016]. Both mode-specific gains also have positive lower
bounds, as do paired AP response and selectivity. This is evidence of learned correspondence to the
introduced smooth offsets under this fixed recipe.

The positive advancement rule is narrower than practical localization adequacy. The amplitude gain
only slightly exceeds the prespecified0.05 point-estimate margin, and the study uses one fitted
randomness. Fixed plots show diffuse background activation rather than clean target-only maps.
Supervised sham and nuisance record maxima exceed the descriptive0.5 cutoff in23/48 and25/48 cases,
respectively, compared with0/48 in the permuted arm. These are engineering control responses, not
calibrated clinical false-positive rates. Supervised control means are lower than the permuted arm's
means, so the concern is concentrated non-target responses rather than uniform probability inflation.

The result report should show absolute AP and mode-specific values beside gains, retain the high
control-maxima counts, and identify the seed-bootstrap intervals as conditional on one fit. Calling
this an accurate or clinically useful abnormality localizer would overstate the evidence. A bounded
saved-map diagnosis of control activations and a separately frozen objective/fit-seed replication
belong in the backlog; no extra fit is required to establish the present result or authorized by
this review. Numerical training/checkpoint and metric verification remains the independent audit's
responsibility, rather than an implied independent re-execution of optimization here.
