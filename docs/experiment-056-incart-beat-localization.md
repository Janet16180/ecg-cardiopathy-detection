# Experiment 056: independent expert beat-identity localization on INCART

Frozen 2 October 2026 before any score. The user authorized autonomous work with new public data.
INCART's expert-corrected beat types supply an independent beat-identity benchmark; its annotation
times were not manually corrected. This tests which beat is unusual, not an expert ST/lead region or
an anatomical diagnosis. Source: [PhysioNet INCART 1.0.0](https://physionet.org/content/incartdb/1.0.0/),
Open Data Commons Attribution License v1.0. Public waveforms stay local and ignored by Git.

## Prospective metadata audit and closed-data boundary

The committed [record manifest](experiment-056-incart-records.json) audits all 75 headers before
scores: 32 patients, 30 minutes per record, 257 Hz, all twelve standard leads in the same order, and
per-record gains of 240–1,063 ADC units/mV. Header patient numbers agree with the official
`files-patients-diagnoses.txt`. All records sharing a patient number are one evaluation group.

The repository's actual closed Challenge manifest has 65,824 rows from only Ningbo, Chapman/Shaoxing,
Georgia, CPSC2018 and CPSC-Extra; none has an INCART `I<digits>` record name. The proof and manifest hash
are committed. INCART appeared in the broader historical Challenge2020, but is absent from this
repository's frozen source split. No closed Challenge, reserved calibration, EchoNext or MIMIC signal
is read. The new methods have no fitted INCART labels or encoder parameters.

## Inputs and fixed algorithms

Read all 75 INCART records. WFDB physical conversion uses actual header gain/baseline, mapping
AVR/AVL/AVF to canonical aVR/aVL/aVF without inventing leads. Polyphase-resample the full record once
from 257 to 500 Hz with shared `resample_full(up=500, down=257)`, then crop nonoverlapping 10-second
cores with 0.5 seconds context on either side. Context contributes label-free templates only; a
detected beat belongs to one core by its original absolute R time. No annotation position enters peak
detection, alignment, template fitting, normalization, threshold fitting or score construction.

Use frozen waveform-only `r_peaks` for both maps. With at least three complete beats in the context:

- **`aligned_residual` (one primary recipe):** frozen 054 `aligned_field`, with shared multilead QRS
  shifts limited to +/-5 samples, recording-level lead scale (median QRS peak-to-peak, floor 0.1 mV),
  leave-one-out template, no per-beat amplitude fit and original-timeline residuals. Use its
  uncalibrated `aligned_only` energy: the empirical phase-CDF recipe is not reused or tuned here.
- **`U_B_fixed` comparator:** reconstruct frozen 042's 48 lead/wave Ledoit-Wolf normal references from
  its exact 5,872 PTB-XL training normals and original kept beats, using shared `beat_pieces`,
  `fit_wave_references`, `wave_scores` and `beat_unit_map`. No INCART patient enters this reference.
  Broadcast exact piece supports as in 052.

Both maps average dense energies in identical 140 ms windows, 10 ms stride, on the shared midpoint-owned
complete-beat mask intersected with actual U_B piece coverage. A window belongs to its nearest original
complete detected R peak, using midpoint ownership. Each beat's scalar score is the maximum eligible
window score belonging to it; its displayed location uses the median-time tied maximum. All top
choices use the frozen `isclose(rtol=1e-10, atol=1e-10)` expectation over tied maximal windows/beats.
No method sweep, new outcome-selected width or RR anomaly feature is permitted.

## Expert matching and eligible evaluation

Only after all scores for a record are computed, open its `.atr` annotation. Recognize WFDB beat
symbols using its standard beat-type list; nonbeat rhythm/quality annotations are counted separately.
Match waveform-detected R times to reference beat annotations one to one: enumerate pairs with absolute
distance at most 150 ms, sort by distance then reference index then detected index, greedily accept
unused pairs. This frozen tolerance accommodates uncorrected annotation positions. Report unmatched
reference/detected beats, distances, per-type detection coverage and skipped core reasons. Annotation
positions are evaluation coordinates, never oracle inference anchors.

Primary labels are expert ventricular ectopic `V` versus ordinary `N`. Other beat types (including
L/R conduction beats, atrial ectopy, fusion and unknown) are excluded from this binary comparison and
reported separately; they are never silently relabeled normal. All eligible matched N/V beats are
evaluated. A patient contributes AUROC only with both classes across its pooled records. A core
contributes the top-beat endpoint only with at least one matched V and one matched N. Report all
patient/core exclusions and original annotation counts. Coverage is common to both maps; missed V
beats are explicitly included in an end-to-end sensitivity denominator at the fixed threshold.

## Co-primary decision and guardrails

Macro-average patient AUROC (each eligible patient equal weight) and macro-average each patient's
mixed-core top-beat hit minus its matched V share among matched N/V beats. Tied top beats receive
their expected V hit. The candidate must pass both:

1. AUROC gain over U_B_fixed at least +0.05 with paired patient-bootstrap lower bound above zero.
2. Top-beat hit-minus-chance gain at least +0.10 with paired patient-bootstrap lower bound above zero.

Patient intervals use 2,000 draws, seed 56056, shared `patient_groups`/`patient_resample` through
`bootstrap_mean`; every record and core of a drawn patient stays together. Report AP and V prevalence
descriptively. Conditional matched-beat AUROC does not establish full detector sensitivity.

Fit each method's threshold to the 95th percentile of the *same beat-level score* on PTB training-normal
records; this is neither the old ECG-max threshold nor an INCART annotation-based threshold. Report
patient-macro N false flags and conditional/end-to-end V sensitivity. The candidate must additionally
have the N false-flag difference's upper bound no greater than +0.05. These thresholds may transport
poorly to long Holter recordings; their absolute values are descriptive, while the paired increase is
an explicit guardrail. Coverage cannot favor one method because both use exactly the same detections.

For the first 10-second core of every record, score the unchanged signal and a fixed nuisance copy
(+0.15 mV constant offset and 0.05 mV 0.2 Hz drift) using the unchanged peaks. Report per-beat score
changes, rank changes and threshold flag-share increases. A candidate nuisance flag-share increase
over +0.05 prevents promotion. Do not adjust a method after these checks.

## Integrity, execution and outputs

Before every new benchmark score, reproduce 054's full primary metric dictionary from its live table,
all synthetic summaries and nuisance fields, and verify every recorded source/protocol/input/output
hash. Reproduce 052 and the entire saved historical U_B metrics exactly through its shared library.
After rebuilding U_B references, recompute all 1,604 development unit maps and require shape/time/lead
equality plus maximum score absolute difference <=1e-8 and relative <=1e-10 to frozen 042 units.
Any discrepancy stops before INCART scores. New normal-reference fitting is independent of INCART.

New files only: `ecg_experiment/incart_localization056.py`, runner
`scripts/experiments/run_incart_localization056.py`, and meaningful tests for header gain/lead order,
patient grouping, deterministic one-to-one matching, midpoint ownership, equal supports and
patient-weighted metrics. Every executed predecessor remains frozen. CPU only, two threads, no GPU.
Do not import scripts from new code. No new dependency is needed.

Write through `outputs/experiment056_incart_localization_v1.partial/` and rename on success, refusing
overwrite. Persist normal references and thresholds, header/waveform/annotation hashes, resampling
settings, all detected and reference beat identities/matches/scores, per-core and per-patient metrics,
coverage/exclusions, nuisance checks, source/protocol/manifest hashes and run log. Save local raw
waveform comparisons for deterministic lowest-ID wins/losses and difficult expert V cases, including
false positives and missed annotations. The running agent writes
`docs/experiment-056-incart-beat-localization-results.md` from executed outputs, reporting every gate
and limitation. A positive result establishes independent ectopic beat ranking only, not correct
ST/lead labeling, structural disease detection, or performance in university students.
