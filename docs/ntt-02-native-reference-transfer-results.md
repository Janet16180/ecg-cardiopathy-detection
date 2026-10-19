# NTT-02 results: phase priors beat masked morphology on supplied windows

18 October 2026. The frozen ECG-native reference comparison beat duration matching, but failed its
advancement rule because simpler QRS-phase and derivative-energy descriptors did better. This is a real
bounded CPU candidate-selection result on generated silver answers. It establishes neither independent
clinical localization nor the original paper's SAM2/DINOv2 performance.

The [protocol](ntt-02-native-reference-transfer.md) was committed as `4a5fefc`; clarification and executable
sources were committed as `9289e29` before the single scored run. Both commits use the user's personal
identity. The run completed in **68.82 seconds**, including 65.17 seconds for integrity checks, loading,
feature extraction and input hashing, with at most two numerical threads. No GPU, download, new library,
training, push or PR publication was used by this study.

## The executed denominator

Every reasoning and local-label aggregate in the existing NTT-01 receipt reproduced exactly, including
2,499 permitted cases, 2,322 ECGs, 2,174 patients, 4,854 wave answers, 4,813 absolute intervals and 41
beat-ordinal answers. This was the predecessor integrity replay; it allocated no new evaluation sample.
Original historical ECGAD localization metrics still cannot be reproduced without their original marks.

New scoring was restricted to PR67 train patients. An independent identity-only implementation matched
all 13,938 train ECGs / 12,078 train patients in the shared manifest exactly. Whole patients touching
folds 9/10 were excluded. Reserved validation/engineering-test content and signals were skipped before
interpretation. The shared manifest SHA256 is
`efaa51a62f52fc2d10881a58994d2da206ae5683cff08872c09cc5ec08ef253c`.

| Selection stage | Executed count |
| --- | ---: |
| Eligible train QA cases | 1,982 |
| Ineligible cases skipped using identities only | 1,102 |
| Nonpositive diagnosis cases excluded | 1,084 |
| Wave questions excluded for nonaffirmative finding | 424 |
| Nonabsolute or nondistinct wave-option questions excluded | 3 |
| Positive absolute questions before patient/class cap | 1,550 |
| Questions removed by the maximum-three cap | 1 |
| Candidate reference classes before minimum-patient rule | 38 |
| Eligible classes / excluded classes | 32 / 6 |
| Scored questions / ECGs / patients | 1,507 / 827 / 769 |
| Prediction rows / support manifests | 9,042 / 576 |

The six class failures are retained in the table below; none was dropped after observing a score.
Each target patient was absent from both its matched and shuffled reference banks. References used the
released affirmative finding and a named criterion, with budgets one/three independent patients and
three frozen draws. Scalar summaries and masked waveform prototypes gave each support patient equal
weight. Three inner patient folds existed only inside PR67 train.

The model pooled all twelve native 100 Hz leads into 32 temporal bins per masked interval, preserving
physical mV and waveform polarity after per-lead median removal. It used unit-normalized instance
vectors and patient-then-class prototype averaging, with no learned representation. Although 890 scored
questions have some lead answer in their reasoning step, those answers are not necessarily linked to
every wave question. The study preserved that metadata and made no lead-localization claim.

## Candidate-choice results

Values below are patient-macro exact released-option agreement, averaged over the three support draws.
The uniform row is the expectation for the same option counts. References for every method and control
were selected without the target patient's inner fold.

| Method / control | One support patient | Three support patients |
| --- | ---: | ---: |
| Masked physical morphology | 64.84% | 75.12% |
| Duration only | 52.77% | 58.67% |
| Absolute center only | 23.40% | 23.61% |
| QRS phase only | 87.48% | 87.07% |
| Physical RMS amplitude | 73.46% | 76.79% |
| Derivative energy | 76.30% | 79.27% |
| Wrong-window references | 16.85% | 11.36% |
| Shuffled-class references | 45.49% | 50.87% |
| Always option zero | 24.24% | 24.24% |
| Uniform-choice expectation | 24.39% | 24.39% |

For the frozen primary comparison, budget-three morphology minus duration was **+16.45 percentage
points**, with paired whole-patient bootstrap 95% interval **+13.31 to +19.52 points** (2,000 draws,
seed 20261018; 769 equally weighted patients). This narrow comparison passed.

The full advancement rule failed. Morphology trailed QRS phase by **11.95 points** (95% interval
−14.16 to −9.69), derivative energy by **4.15 points** (−7.13 to −1.10), and physical amplitude by
1.67 points (−4.49 to +1.35). These secondary intervals are descriptive comparisons on the same
development cohort, not multiplicity-adjusted confirmatory evidence. The frozen decision is therefore
**do not advance this morphology method on the strength of this benchmark result**.

The QRS-phase descriptor measures the distance from a candidate center to a detected QRS center, then
matches it to the reference descriptor. It uses no disease interpretation. It reached 100% agreement in
26 of 32 classes. This is consistent with many supplied tasks being separable by generic wave identity
or timing, rather than by the named abnormal morphology. It does not prove which construction shortcut
caused every individual answer or that any published model used such a shortcut.

The supplied correct interval is explicitly one of the candidates, so the answer-proposal oracle is
100% by construction. Every observed candidate-window IoU score equaled exact-choice accuracy: the
alternative windows did not provide a graded overlap endpoint. The experiment therefore supplies no
independent free-localization or boundary-quality result. Selected window widths were 70/110/140/160/390 ms
at the minimum/25th/50th/75th/maximum percentiles; other-option widths were 60/130/190/230/360 ms.
The target answer positions were 361/441/344/334/27 for indices 0/1/2/3/4. These distributions are
construction descriptions, not verified pathological-negative annotations.

## Reference sensitivity and illustration

One-reference morphology varied from **60.76% to 67.93%** across the three fixed draws; three-reference
morphology varied from **72.27% to 79.82%**. More support patients helped its mean score, but did not
remove dependence on the chosen reference patients. Wrong-window and shuffled-class references reduced
agreement, showing sensitivity to the supplied reference content. Those are limited controls: the first
unselected reference option can still be clinically meaningful, and 36 of 576 shuffled banks retain the
same criterion with a different wave question. Neither control establishes finding-specific causal
localization or that another beat is truly normal.

The prospectively fixed first eligible class/patient/question was illustrated without choosing by score.
Its target options were [5.73, 5.95], [8.20, 8.31], [8.48, 8.78], [9.39, 9.52] seconds; morphology chose
index 1, matching the released answer. The displayed reference mask was [7.67, 7.79] seconds. This plot
shows lead II for readability while the method uses all twelve leads, and displays one of the three
reference patients. It illustrates candidate geometry; it is not a specialist validation example.

The locally inspected [PNG](../outputs/ntt02_native_reference_transfer/example.png),
[SVG](../outputs/ntt02_native_reference_transfer/example.svg), and
[coordinate record](../outputs/ntt02_native_reference_transfer/example_coordinates.json) preserve seconds,
mV, support/target identity and option coordinates. Waveforms and patient-level artifacts stay local.

## Receipts, checks and reproduction

The completed [result](../outputs/ntt02_native_reference_transfer/result.json) has SHA256
`5e7f4052b60d3bfb3ae589bb6cc482c32c7af49d6858d8cc5ae37e6cb044047b`. It binds committed source bytes,
the pinned public annotation, metadata/statements, the exact shared partition and all 827 waveform
header/data pairs. The public annotation SHA256 remains
`353ddc413b68c751a0e2c5c44a11a182de4519e278c5ae49fe0bf089c674b39b`.
[Local verification](../outputs/ntt02_native_reference_transfer/verification_local.json) independently
rebuilt target/reference patient separation, PR67 train membership, inner folds, stored choices and the
primary difference with standard-library aggregation (agreement within 1e-15), and binds artifact hashes.

The focused quarantine/coordinate/reference-patient suite passed 37 tests, including the NTT-01 tests.
Repository-wide Ruff passed. The full CPU suite in the required symlinked worktree produced 1,245
passed, six skipped and one existing test failure in 74.48 seconds. The frozen test
`test_graph_rejects_missing_and_conflicting_leaves` expects a missing output-file error, while frozen
`safe_leaf` rejects the worktree's required `outputs` symlink first. Those three existing source/test
files are byte-unchanged from the worktree base. Running that exact existing test from the main checkout,
where `outputs` is a real directory, passed (one test, 2.60 seconds). No frozen source/test or worktree
symlink was changed to hide the environment-specific failure.
There was one scored execution and no post-score feature, support, group or threshold tuning. No
historical spent/closed evaluation set was opened for new scoring. The statistical intervals describe
these generated questions and fixed draws; shared support banks and source annotation errors limit
interpretation beyond this development feasibility study.

Reproduce in a fresh output directory, with the existing venv and local annotations/signals already present:

```bash
PYTHONPATH="$PWD" UV_CACHE_DIR=/tmp/ecg-ntt02-uv \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MPLCONFIGDIR=/tmp/ecg-ntt02-mpl uv run --no-sync python -m scripts.reports.reference_transfer_ntt02 \
  --partition outputs/ntt02_native_reference_transfer/partition_local.json \
  --output outputs/ntt02_native_reference_transfer_replay
```

Follow-up candidates for the root-owned backlog: retain the QRS-phase descriptor as a mandatory control;
construct independently reviewed within-wave distractors and genuinely linked lead/time findings before
claiming pathological evidence localization; keep any visual paper-model feasibility result separate.
This negative gate does not establish whether the original paper's pretrained image representation can
help on a clinically meaningful independent endpoint.

## Every prospective reference class

The table retains all 38 classes, including the six failures of the minimum-12-patient rule. Distinct
wave questions under the same criterion remain separate classes. Percentages use three support patients
and the three frozen draws; class denominators overlap because one patient can carry several findings.

| Class | Named criterion and exact wave question | Patients | Questions | Morphology | QRS phase | Duration |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Cornell Product ((R amplitude in aVL + S amplitude in V3) x QRS duration) > 0.244mV*s; In which of the following segments can you locate a QRS wave to measure the R amplitude in lead aVL? | 42 | 44 | 90.48% | 100.00% | 59.13% |
| 2 | Cornell Product ((R amplitude in aVL + S amplitude in V3) x QRS duration) > 0.244mV*s; In which of the following segments can you locate a QRS wave to measure the S amplitude in lead V3? | 42 | 44 | 95.63% | 100.00% | 62.30% |
| 3 | Presence of ST-segment depression in anterior leads; Within the selected leads, in which of the following waves can you observe the QRS complex with ST-segment depression? | 31 | 31 | 97.85% | 100.00% | 87.10% |
| 4 | Presence of ST-segment depression in inferior leads; Within the selected leads, in which of the following segments can you observe the QRS complex with ST-segment depression? | 10 | 11 | Excluded | Excluded | Excluded |
| 5 | Presence of ST-segment depression in lateral leads; Within the selected leads, in which of the following segments can you observe the QRS complex with ST-segment depression? | 30 | 30 | 91.11% | 100.00% | 65.56% |
| 6 | Presence of ST-segment elevation in anterior leads; Within the selected leads, in which of the following segments can you observe the QRS complex with ST elevation? | 25 | 26 | 89.33% | 100.00% | 61.33% |
| 7 | Presence of ST-segment elevation in inferior leads; Within the selected leads, in which of the following segments can you observe the QRS complex with ST-segment elevation? | 33 | 37 | 79.46% | 100.00% | 39.56% |
| 8 | Presence of ST-segment elevation in lateral leads; Within the selected leads, in which of the following segments can you observe the QRS complex with ST-segment elevation? | 10 | 12 | Excluded | Excluded | Excluded |
| 9 | Presence of a broad monophasic R wave or qR pattern in lead V1; Within the lead V1, in which of the following segments can you observe the QRS complex with broad monophasic R wave or qR pattern? | 11 | 11 | Excluded | Excluded | Excluded |
| 10 | Presence of a broad monophasic R wave with a prolonged R wave peak time in leads V5 and V6; Within the selected leads, in which of the following segments can you observe the QRS complex with broad monophasic R waves, accompanied by a prolonged R wave peak time? | 28 | 31 | 100.00% | 100.00% | 55.36% |
| 11 | Presence of a dominant R wave in V1; In which of the following segments can you locate a QRS wave to measure the R amplitude in lead V1? | 2 | 2 | Excluded | Excluded | Excluded |
| 12 | Presence of a dominant S wave in leads V1 and V2; Within the leads V1 and V2, in which of the following segments can you observe the QRS complex with dominant S wave? | 62 | 66 | 100.00% | 100.00% | 77.42% |
| 13 | Presence of a non-conducted P wave within the dropped interval; In which of the following segments can you observe the non-conducted P wave, specifically within the prolonged RR interval? | 2 | 2 | Excluded | Excluded | Excluded |
| 14 | Presence of a notched R wave in lateral leads; Within the selected leads, in which of the following segments can you observe the notched R wave? | 34 | 35 | 100.00% | 100.00% | 63.73% |
| 15 | Presence of a positive monophasic QRS complex without a Q wave in lateral leads; Within the selected leads, in which of the following segments can you observe the positive monophasic QRS complexes without Q waves? | 62 | 66 | 100.00% | 100.00% | 70.43% |
| 16 | Presence of a qR pattern in leads I and aVL; Within the selected leads, in which of the following segments can you observe the QRS complex with qR pattern? | 61 | 66 | 94.81% | 100.00% | 49.73% |
| 17 | Presence of a qR pattern in leads II, III, and aVF; Within the selected leads, in which of the following segments can you observe the QRS complex with qR pattern? | 17 | 18 | 99.02% | 100.00% | 41.18% |
| 18 | Presence of a slurred S wave in lateral leads; Within the selected leads, in which of the following segments can you observe the QRS complex with slurred S wave? | 58 | 61 | 91.95% | 100.00% | 73.56% |
| 19 | Presence of an LV strain pattern in lateral leads; Within the selected leads, in which of the following segments can you locate the ST segment and T wave showing the LV strain pattern? | 22 | 23 | 81.82% | 66.67% | 81.06% |
| 20 | Presence of an RSR' pattern in leads V1 and V2; Within the selected leads, in which of the following segments can you observe the QRS complex with RSR' pattern? | 48 | 50 | 97.22% | 100.00% | 69.79% |
| 21 | Presence of an rS pattern in leads I and aVL; Within the selected leads, in which of the following segments can you observe the QRS complex with rS pattern? | 17 | 18 | 92.16% | 100.00% | 50.98% |
| 22 | Presence of an rS pattern in leads II, III, and aVF; Within the selected leads, in which of the following segments can you observe the QRS complex with rS pattern? | 61 | 66 | 95.63% | 100.00% | 47.27% |
| 23 | Presence of inverted T waves in anterior leads; Within the selected leads, in which of the following segments can you observe the inverted T wave? | 34 | 34 | 73.53% | 73.53% | 59.80% |
| 24 | Presence of inverted T waves in inferior leads; Within the selected leads, in which of the following segments can you observe the inverted T waves? | 58 | 61 | 85.63% | 84.20% | 66.09% |
| 25 | Presence of inverted T waves in lateral leads; Within the selected leads, in which of the following segments can you observe the inverted T wave? | 38 | 38 | 92.98% | 95.61% | 58.77% |
| 26 | Presence of left axis deviation; In which of the following segments can you locate a QRS wave that would be used to calculate the QRS axis? | 61 | 66 | 89.62% | 100.00% | 47.81% |
| 27 | Presence of pathological Q waves in anterior leads; Within the selected leads, in which of the following segments can you observe the QRS complex with pathological Q waves? | 27 | 27 | 51.85% | 100.00% | 61.73% |
| 28 | Presence of pathological Q waves in inferior leads; Within the selected leads, in which of the following segments can you observe the QRS complex with pathological Q waves? | 30 | 31 | 82.22% | 100.00% | 51.67% |
| 29 | Presence of pathological Q waves in lateral leads; Within the selected leads, in which of the following segments can you observe the QRS complex with pathological Q waves? | 49 | 56 | 64.63% | 100.00% | 55.56% |
| 30 | Presence of premature beats; In which of the following segments can you observe the premature beat? | 125 | 127 | 31.20% | 39.33% | 34.13% |
| 31 | Presence of right axis deviation; In which of the following segments can you locate a QRS wave that would be used to calculate the QRS axis? | 18 | 20 | 92.59% | 100.00% | 46.30% |
| 32 | Prolongation of the PR interval; In which of the following segments can you observe a P wave that demonstrates the prolonged PR interval? | 65 | 66 | 59.74% | 84.62% | 63.08% |
| 33 | Prolongation of the QRS duration; In which of the following segments can you observe a QRS wave that demonstrates the prolonged QRS duration? | 120 | 127 | 57.22% | 100.00% | 77.08% |
| 34 | QRS axis > +60 degrees; In which of the following segments can you locate a QRS wave that would be used to calculate the QRS axis? | 21 | 21 | 100.00% | 100.00% | 90.48% |
| 35 | R wave amplitude in V1 > 1.5mV; In which of the following segments can you locate a QRS wave to measure the R amplitude in lead V1? | 4 | 4 | Excluded | Excluded | Excluded |
| 36 | R wave amplitude in lead aVL > 1.1mV; In which of the following segments can you locate a QRS wave to measure the R amplitude in lead aVL? | 36 | 36 | 93.52% | 100.00% | 77.78% |
| 37 | Sum of S amplitude in V1 and R amplitude in V5/V6 > 3.5mV; In which of the following segments can you locate a QRS wave to measure the R amplitude in the selected lead? | 42 | 42 | 94.44% | 100.00% | 64.29% |
| 38 | Sum of S amplitude in V1 and R amplitude in V5/V6 > 3.5mV; In which of the following segments can you locate a QRS wave to measure the S amplitude in lead V1? | 43 | 43 | 94.57% | 100.00% | 66.67% |
