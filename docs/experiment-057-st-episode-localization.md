# Experiment 057: independently annotated ST episode localization

Frozen 2 October 2026 before any score. The user reports that neither the promised Italian data nor
cardiologist access is available and asks us to overcome that limitation. This authorizes bounded
public local acquisition and independent development benchmarks on the existing PR #61. No uploads,
project diagnosis label changes, clinical decisions or closed-test access are authorized.

## Question and source

Does a phase-specific, unsupervised ST change score locate independently adjudicated transient ST
episodes more accurately than a whole-beat morphology residual with the same reference and support?
Use all 90 records of the [European ST-T Database, version 1.0.0](https://physionet.org/content/edb/1.0.0/).
The source has 79 subjects, two channels, 250 Hz and two hours per recording. Two cardiologists
independently annotated ST/T changes; disagreements were adjudicated. Records and channels are used
as published, including modified leads. Neither channel is invented or mapped to absent 12-lead data.

Patient groups follow the source's explicit repeated-subject list: e0118-e0122, e0123-e0126,
(e0129,e0133), (e0136,e0139), (e0147,e0148), (e0154,e0155), (e0162,e0163); all others are distinct.
Require exactly 90 records and 79 patient groups. Verify every .dat/.hea/.atr against the published
SHA256SUMS.txt before scoring; save the acquisition receipt locally. There are no trained parameters,
selected hyperparameters or annotation-informed references. All records are evaluation-only.

These annotations describe *changes* relative to an early reference, including changes superimposed
on existing abnormalities. They do not mark every pathological wave or the anatomical heart lesion.
The two-hour context is also different from the project's ten-second screening recordings.

## One prospective comparison

This is a conditional waveform-localization benchmark: both methods receive the same annotated beat
times, with their beat types discarded before feature computation. Beat detection is deliberately
held constant; findings cannot establish accuracy of an autonomous end-to-end detector. Other
annotation fields, including ST/T events, quality and beat identity, never reach feature computation.

Extract both available channels on offsets [-0.25,+0.45) seconds, rounded to the native 250 Hz sample
grid. Require the complete support. Subtract the median voltage on [-0.10,-0.06) seconds separately
for each beat and channel. The reference is the median of all complete beats whose timestamps fall
in the first 30 seconds, without using beat labels. Fewer than three reference beats is an explicit
exclusion with no replacement. Record all exclusions and require at least 85 evaluable records.

The comparator is the root mean square difference from that reference over [-0.06,+0.45), per beat
and channel. The candidate is the absolute difference between observed and reference mean voltage
on [+0.12,+0.16), per beat and channel. This fixed approximation to the ST segment is not claimed
to measure J+80 ms exactly; no boundary estimates or expert wave marks are supplied.

For both methods, average those nonnegative per-beat scores within the same nonoverlapping 30-second
windows, starting at 30 seconds and ending at 7,200 seconds. A channel/window requires at least three
complete beats. The reference interval is never an evaluated candidate. All eligible windows and
both channels are shared exactly. Scores remain in mV. No empirical CDF cap, RR anomaly feature,
per-record outcome selection, hyperparameter sweep or later recipe retuning is permitted.

## Expert targets and decision

Parse uppercase ST annotations into onset, extremum and end episodes for the corresponding channel.
Keep T episodes and lowercase positional axis-shift episodes separately. Validate chronological
pairing and channel identity; malformed episodes raise, rather than disappearing silently. An expert
ST hit requires the top window's center inside an uppercase ST episode on the *same channel*.
Average all maxima tied with isclose(rtol=1e-10, atol=1e-10). Also report the deterministic displayed
top window separately. The target is episode time and channel, not within-beat wave boundaries.

The primary population consists of all evaluable records with at least one ST episode containing an
eligible window center. For each record compute hit and uniform eligible-window hit probability,
then average within subject. Bootstrap those subject means with 2,000 whole-patient draws, seed 57057,
using ecg_experiment/intervals.py. Promote this method *for review as an ST-change localization
prototype* only if patient-macro paired hit-minus-chance gain is at least +0.10, its 95% lower bound
exceeds zero, and the candidate's patient-macro hit rate reaches 70%. Require at least 30 subjects
with eligible ST targets. The unchanged screening/explanation pipeline is not replaced by this test.

Report window-level ST-versus-non-ST AUROC (subject-macro), top-5% window episode coverage,
channel/time chance, ties, deterministic hit rate, extremum proximity and each subject's counts.
Axis-shift localization and T-only/non-ST top marks are explicit stress analyses; an axis-shift top
hit increase over the comparator greater than +0.05 prevents general ST-only promotion, even if the
primary passes. A no-ST record has no correct ST top highlight, so top-one forced selection on those
records is not mislabeled as specificity. Thresholded episode detection is outside this experiment.

## Integrity, resources and evidence

Before any new score, verify every recorded source/input hash of Experiment 054 and reproduce its
primary comparison and synthetic summaries exactly from saved outputs. This checks predecessor
integrity; its automatic targets are not used for 057 evaluation or method selection. Source files
already pinned by receipts are never edited. Use new ecg_experiment/st_episode057.py,
scripts/experiments/run_st_episode057.py and meaningful scientific tests. Scripts import shared
library functions, never other scripts. Record repository-relative paths with paths.py.

CPU only, at most two numerical threads, no GPU, no training or pretraining. Acquisition is bounded
to 500 MB plus metadata and execution to 30 CPU minutes. Save output to a new
outputs/experiment057_st_episode_v1/ directory, with source/protocol/data hashes, exclusions, beat
features, window scores, episodes, patient statistics and source-backed grouping. Save local plots
for the first three ST successes and first three failures by record ID, plus all six known positional
axis-shift records. Raw recordings and individual plots/results stay local. The executing agent
writes docs/experiment-057-st-episode-localization-results.md from completed outputs, including
negative outcomes and the ten-second/12-lead transport limitation. Queue and backlog are updated.

Source annotation specification:
[European ST-T annotation guide](https://physionet.org/content/edb/1.0.0/annotations.shtml).
The first-30-second reference follows that guide; it is fixed from source methodology before scores.
