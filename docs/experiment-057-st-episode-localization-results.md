# Experiment 057 results: ST-change time and channel localization improved

The ST-specific unsupervised score passed the frozen review rule against independent expert ST
episode annotations. Its patient-macro top-window hit rate was **70.4%**, against **53.6%** for this
experiment's whole-beat residual comparator. The paired gain was **+16.8 percentage points
[+7.2,+27.6]**. This is a review-worthy ST-change prototype, not a replacement for the screening or
explanation pipeline and not a measured improvement over U_B on this dataset.

The comparison is conditional on supplied beat timestamps and an evaluable expert ST episode.
It uses two-hour recordings with two available channels and an early 30-second reference. It does
not establish autonomous beat detection, ST detection specificity, anatomical heart-lesion
localization, or accuracy on a standalone ten-second 12-lead screening ECG.

## Executed evidence

- Original protocol: [Experiment 057](experiment-057-st-episode-localization.md), committed `b72539a`.
- Prospective format/censoring amendment: [annotation audit](experiment-057-annotation-audit-amendment.md).
- Frozen source/amendment commit: `f0feff4`; CPU only, two numerical threads, no trained model.
- Complete receipt: `outputs/experiment057_st_episode_v1/result.json`.
- Local data: public European ST-T 1.0.0, 270 record files, 487,662,449 bytes, all published checksums match.
- Run elapsed: 42.04 seconds, including predecessor reproduction; 90 records/79 subjects, zero exclusions.
- Eleven scientific tests and Ruff passed before execution.

Before any new score, 6,221 predecessor source/waveform hashes were verified and Experiment 054's
complete primary and synthetic summaries reproduced exactly. The independent scientific reviewer
then verified all 12 current source hashes, 272 data/metadata hashes and 194 output hashes. They
reconstructed targets, censor masks, candidate centers, top marks, chance and paired intervals from
the saved arrays, and rebuilt all 90 early-reference templates directly from native mV samples;
every template matched exactly. All 90 headers specify mV and 200 ADC units/mV in both channels.

## Primary comparison

The primary population contains 85 records from 76 subjects with at least one completed expert ST
episode containing an eligible 30-second window center. Scores are computed without episode labels.
Beat identities are discarded; both methods receive identical beat timestamps, reference intervals
and candidate windows. A hit requires the window center and selected channel to belong to a
completed uppercase ST episode. Each subject receives equal weight after averaging their records.
Intervals use 2,000 whole-subject draws, seed 57057.

| Endpoint | ST-specific score | Whole-beat residual |
| --- | ---: | ---: |
| Patient-macro top time+channel hit |70.39% [60.53,79.61]|53.62% [42.43,64.14]|
| Uniform same-domain chance |11.03% [8.81,13.60]|11.03% [8.81,13.60]|
| Hit minus chance |59.37 points [48.92,69.11]|42.59 points [31.61,52.85]|
| Displayed deterministic hit |70.39%|53.62%|
| Records with tied maxima |0|0|
| Mean episode coverage at top 5% of windows, descriptive |68.77%|58.38%|
| Subject-macro window ST/non-ST AUROC, descriptive |0.8878|0.8533|

The paired gain is 16.78 points [7.24,27.63]. It exceeds the frozen 10-point practical threshold, its
lower bound is positive, and the candidate exceeds the frozen 70% hit threshold. All record/subject
count gates pass. Raw record hits are 60/85 versus 44/85; 19 records improve and three regress.
Those record-weighted counts differ from the primary patient-macro estimates because some subjects
have repeated recordings. The AUROC analysis skips three single-class subjects explicitly.

The positional-ST-shift stress analysis has six subjects. The candidate's top mark hits an annotated
positional ST shift in one record, versus two for the comparator: paired increase −16.67 points
[−50.00,0.00]. This passes the frozen increase guard, but six subjects cannot establish general
artifact robustness. Lowercase T-shift annotations are counted separately and do not enter this guard.

## Annotation incompleteness and observable domain

Four strings contained embedded NUL terminators followed by padding. They were parsed according to
the prospective amendment. Twelve unfinished episode keys in nine records remain explicitly
unknown. Four concern ST in three records. Both channels' evaluation windows are conservatively
restricted before the earliest unfinished ST/st onset in each affected record; all waveform scores
remain saved, with the evaluation mask visible. The discarded tail is never imputed as positive or
negative. Unfinished T episodes do not erase independently complete ST annotations.

| Record | Earliest unknown ST tail, seconds | Censored windows per channel |
| --- | ---: | ---: |
|e0405|6951.404|8|
|e0409|506.376|223|
|e0704|6761.996|15|

The parser finds 364 complete ST episodes plus four unfinished ST episodes, compared with the
source page's 367 episodes. That discrepancy is preserved, not corrected by changing the parser.
The 476 complete T intervals include nested extreme-T intervals and are not directly comparable to
the source's count of base T episodes. There are 11 complete positional ST and 12 positional T intervals.

Five records have no evaluable completed ST target: e0133, e0155, e0409, e0509, e0611. This includes
e0409 after conservative censoring; it does not imply that the entire recording lacks ST changes.
A forced top mark on these records is not counted as correct ST localization and cannot measure
clinical detection specificity. The frozen receipt field `promote_general_st_only_localizer=true`
means the study's primary/count/stress gates passed; its broad name does not authorize general
clinical deployment. The receipt also records `clinical_single_ecg_localization_established=false`.

## Local review and next work

The local review page is `outputs/localization_public_review_2026_10_02/index.html`. It includes
original ECG excerpts and early-reference beat overlays, alongside expert episode timelines.
Examples are chosen by record-ID order from the first three corrected cases and first three failures,
plus all three regressions; failures are retained. Raw waveforms and individual outputs stay local.

In e0103, the ST score marks MLIII near 16 minutes inside an expert episode while the whole-beat
residual marks a region near 32 minutes outside it. In e0116, both methods mark a V4 excursion near
117 minutes outside expert ST episodes. Thus the improvement is visible, but transient/artifact
susceptibility remains.

This result overcomes the missing collaborator for a specific expert-annotated localization endpoint.
It warrants user review. The next ranked studies should test waveform-derived beat anchors, retain
the first 30-second reference comparison, and separately investigate transfer to ten-second 12-lead
recordings. No project clinical label, referral rule, model weight, closed test or current explanation
recommendation changed. No further recipe was tuned after the successful comparison.

Sources: [European ST-T 1.0.0](https://physionet.org/content/edb/1.0.0/),
[expert annotation specification](https://physionet.org/content/edb/1.0.0/annotations.shtml).
