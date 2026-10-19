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
