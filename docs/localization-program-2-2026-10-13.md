# Localization programme, part 2 (13 October 2026)

The first programme ([results](localization-program-results-2026-10-13.md)) confirmed a learned lead x wave combiner
(`units_v2`) on the ECGAD holdout, but it localizes mostly by diagnosing and painting the expected region, and the
holdout is now spent. The user reported that the Italian collaboration is no longer available, so no new
cardiologist marks are expected. The user authorized a day of unsupervised experiments on approaches that need no
new marks.

## Questions

| Experiment | Approach | Question |
| --- | --- | --- |
| 069 | Measurement-based criteria localizer | Do per-lead measurements (PTB-XL+ 12SL, Uni-G and ECGdeli features, and our own) turned into guideline criteria give patient-specific lead and wave evidence that beats the textbook region? |
| 070 | Replacement faithfulness test | If the region a map highlights is replaced by its normal counterpart, does a detector's abnormal probability fall more than for a random region of equal size? A label-free test of every map so far. |
| 071 | Same-patient serial change maps | Within PTB-XL folds 1-8, does the aligned difference between a patient's normal and later abnormal ECG light the leads and waves of the new statement, beyond normal-normal pairs and beyond the textbook region? |
| 072 | DINO v2: whole record, lead-local tokens | Does SimDINO localize when it sees the whole 10 s record, keeps leads separable, and is scored without per-bin z calibration? |

## Rules

- ECGAD development (198 records, 92 marked usable) remains the only marked set, and it has now been compared many
  times. Every ECGAD reading in part 2 is exploratory; no result is "confirmed" without a new test set. Independent
  evidence comes from label-free tests (070), the noise control, and patient-specific null comparisons (071).
- The ECGAD holdout half is spent and is not scored again.
- PTB-XL keeps each patient in one fold; no ECGAD development patient has an ECG in folds 1-8.
- Each experiment commits its protocol before any score, reproduces a predecessor number, and ends with a results
  report. Closed data rules are unchanged.
