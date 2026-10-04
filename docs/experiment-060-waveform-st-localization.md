# Experiment 060: ST localization with waveform-derived beat anchors

Prospective protocol, 2 October 2026. The user authorized continued autonomous
localization iterations after proposing synthetic teaching ECGs, while freezing
the first screening stage and keeping existing PR #61. Experiments 058 and 059
address synthetic controls and wave boundaries; this experiment removes the
expert-supplied beat timestamps required by the successful Experiment 057.
No classifier, clinical labels, closed tests or referral thresholds change.

## Fixed recipe and data

Use the same local European ST-T 1.0.0 recordings and published subject grouping
as 057: 90 two-hour, two-channel recordings at 250 Hz, 79 subjects. This source
has already been inspected and is a development/regression benchmark, not an
untouched final test. Verify all 057 source/data/output hashes and reproduce its
primary ST and whole-beat summaries and paired interval exactly before scoring.

Inference receives only the original waveform. Call the frozen
`fragment_localization.r_peaks(signal.T, 250)` on the complete recording;
its channel-energy sum works for both available channels. Do not resample,
retune the detector, supply annotation beat times, or use beat identities.
Pass detected timestamps to the frozen `st_episode057.change_windows` recipe:
PR baseline subtraction, median reference from all complete detected beats in
the first 30 seconds, ST mean change at R+[120,160) ms and whole-beat RMS
change at R+[-60,450) ms. Use native voltages and sample coordinates.
These are fixed ST approximations, not exact J-point boundaries.

Use the exact 057 evaluation window centers and its annotation-completeness
mask for both new arms, regardless of detected beat counts. If fewer than
three detected beats occupy an original window, assign zero to both scores
and retain that window, recording insufficient coverage. An inference failure
or fewer than three reference beats gives zero ST hit for that record in every
primary comparison; it never removes a difficult record. Do not invent a
correct forced highlight when no valid score exists. Every original window
is retained, and missing scores cannot shrink the expert-positive denominator.

Only after waveform features have been saved may the evaluator load expert
episodes/beat annotations. Copy 057's complete ST/positional-ST target masks
and censoring exactly, verifying against saved local arrays. All 85 original
ST-eligible records / 76 subjects remain in the primary population; all 90
records remain in coverage and stress reporting. Real failures count as zero.

## Primary decision, controls and limits

Use expected hit over tied maxima, same-channel expert ST membership at the
window center, and record averages within each published subject. Reuse 057's
2,000 whole-subject bootstrap draws and seed 57057 via shared interval helpers.
There is one recipe and no parameter sweep or score-directed retuning.

A worthwhile operational improvement requires all of:

- Waveform ST minus waveform whole-beat patient-macro hit gain >=0.10 and
  paired 95% lower bound >0.
- Waveform ST patient-macro hit >=0.70.
- Waveform ST minus supplied-anchor 057 ST hit paired 95% lower bound >-0.05,
  a prospective five-point noninferiority margin.
- At least 85 successfully scored records, all 85 original primary records
  retained (failed primary records score zero), and at least 30 subjects.
- Positional-ST top-hit increase over waveform whole-beat <=0.05, with the
  small stress population explicitly disclosed.
- At least 99% of original evaluated window/channel pairs have three or more
  waveform-detected beats. Zero-filled windows remain visible and scored.

Report supplied-anchor and waveform-anchor comparisons side by side, missed
reference/windows, counts, ties, no-ST forced marks, subject-macro window AUROC
when defined, detector sensitivity/precision using post-inference one-to-one
matching within 150 ms, and paired success/regression record counts. Matching
uses all expert beat identities together and never enters either scoring arm.
Do not claim specificity from forced top-one highlights or anatomical lesion
identification. Even success establishes two-hour/two-channel ST-change
localization without supplied beat anchors, not ten-second student screening.

## Resources and artifacts

CPU only, two numerical threads, no training, downloads or GPU. Budget 30
minutes and 500 MB new artifacts. Use new library/runner/tests only; preserve
all pinned predecessors. Hash protocols, sources, predecessor receipt, official
inputs and outputs; store repository-relative paths. Write new local outputs
under `outputs/experiment060_waveform_st_v1/`, never overwrite prior evidence.
Save detected anchors, full scores, common centers, valid-window mask, targets,
record/subject comparisons, and plots for the first three corrected and first
three regressed records, or first three failures if there are no regressions.
Raw traces and individual artifacts remain local. The executing parent writes
the aggregate report from completed outputs and updates queue/backlog, including
negative results. An independent agent audits any proposed improvement.
