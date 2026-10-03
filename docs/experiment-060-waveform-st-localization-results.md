# Experiment 060 results: waveform-derived anchors for ST changes

The waveform-only ST method found the expert-marked episode and channel in
71.71% of subjects, versus 51.97% for its matched whole-beat comparator. The
paired gain was +19.74 percentage points [9.21, 31.58]. Nevertheless, the full
prospective decision failed: noninferiority to Experiment 057's supplied-anchor
method was not established within the frozen five-point margin. No method is
promoted and the screening classifier remains unchanged.

## Executed comparison

Protocol `3b19a76` preceded all new scores; source `fd4e5bf` was committed after
four scientific tests and Ruff passed. The completed local receipt is
`outputs/experiment060_waveform_st_v1/result.json`, SHA-256
`f2291ef5d07138aef5498dd04eaf53882f7c7f1878d9a0183f518c8b38a961d9`.
The recorded scoring time was 19.12 seconds, on CPU with two numerical threads;
figures were generated afterward. No training or GPU was used.

All 90 two-hour/two-channel European ST-T recordings were scored. The original
85 ST-eligible records and 76 subject groups remained in every primary
comparison, with no inference failures. Every evaluated thirty-second window
had at least three complete waveform-detected beats: coverage was 100%.
The original annotation-completeness mask and expert targets were rebuilt
exactly. Both new methods received waveform-detected anchors only; expert beat
marks were loaded after inference features had been saved.

The predecessor integrity check verified 284 source/data files and 194 output
hashes and reproduced 057's primary statistics and paired interval exactly.
An independent audit verified all 13 new source hashes, 272 data hashes and
188 immutable output hashes, rebuilt both paired bootstrap intervals and all
90 evaluation domains, and reran the detector exactly for e0103 and all six
regressions. Its local receipt is
`outputs/experiment060_waveform_st_audit_v1/audit.json`; it confirms the failed
noninferiority decision and found no implementation defect.

| Frozen measurement | Result | Decision |
| --- | --- | --- |
| Waveform ST hit | 71.71% [61.18, 80.92] | >=70% passed |
| Waveform whole-beat hit | 51.97% [40.79, 62.50] | Comparator |
| ST minus whole-beat hit | +19.74 points [9.21, 31.58] | >=10 points / positive lower bound passed |
| Supplied-anchor 057 ST hit | 70.39% [60.53, 79.61] | Predecessor |
| Waveform ST minus supplied-anchor ST | +1.32 points [-7.89, 10.53] | Lower bound >-5 points failed |
| Positional-ST hit increase over waveform whole-beat | 0.00 points [0.00, 0.00], six subjects | Stress gate passed |
| Evaluated-window coverage | 100%; no failed records | Counts/coverage passed |

Intervals use 2,000 whole-subject draws, seed 57057, averaging records within
each published subject. The same 85-record population contains seven records
corrected and six regressed relative to supplied-anchor ST. This is why a
slightly higher aggregate hit rate does not establish safe replacement of the
predecessor within the frozen noninferiority margin.

## Secondary evidence and local review

Post-inference one-to-one matching within 150 ms found 789,742 matches among
790,565 expert beat marks and 792,035 waveform detections: pooled sensitivity
99.896% and precision 99.710%. These describe beat matching, not exact R timing
or within-beat wave boundaries. Per-record detector measurements are saved;
the lowest sensitivity was 97.45% in e0119.

Descriptive subject-macro window AUROC was 0.90985 for ST and 0.84921 for
whole-beat changes, skipping three subjects whose targets had only one class.
This secondary endpoint does not replace the failed primary decision.

Local paired timelines retain the first three corrected records by ID
(e0166, e0204, e0303) and first three regressions (e0115, e0154, e0161), under
`outputs/experiment060_waveform_st_v1/figures/`. All raw recordings and
individual outputs remain local. The six-subject positional stress analysis
does not establish general artifact robustness.

## Scope and follow-up

This completed development comparison shows that a fixed waveform detector can
support useful ST episode/channel localization without supplied beat times. It
does not pass all conditions for replacing 057, and it does not validate
ten-second, twelve-lead student screening or locate an anatomical lesion.
No scores were used to retune the detector or ST recipe. Preserve this run.

Prospective follow-ups belong in the backlog: synthetic stress tests of anchor
timing and phase measurements, followed by a frozen timing-robust recipe and
new independently annotated data. The confidence interval must not be narrowed
by treating windows or multiple records from one subject as independent.
