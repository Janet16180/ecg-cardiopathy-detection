# NTT-06 results: native signal matching works; physical-baseline gain gate fails

The native ECG adaptation is implemented and executed with the real frozen ECG-JEPA checkpoint.
On the unchanged 60 development targets and three reference draws, QRS-based native references reach
91.67% balanced record-finding accuracy, versus 92.78% for the strongest physical baseline and 82.78%
for the historical compact visual pipeline. Native minus strongest physical baseline is -1.11 percentage
points, with a selection-aware paired 95% patient interval [-5.98, +4.31]. The prespecified advancement
gate fails. This does not establish superiority, equivalence or noninferiority.

QRS pooling exceeds the prospectively specified shifted-window control by 8.33 points [3.67, 13.01].
This secondary result suggests that the token-pooling location carries useful finding information.
It does not validate an abnormality boundary: contextual tokens can carry information from the rest of
the ECG, and QRS versus whole-record V1/V6 pooling has an interval that crosses zero.

The [protocol](protocol.md) was committed as `6c29873` before inference. Final scored source is
`7078768`, following the strict encoder adapter and pre-score precision correction. The measured
synthetic profile took 23.68 seconds; development evaluation took 60.27 seconds. No image entered the
native model, no model weight was trained, and no GPU or closed test was used. Work remains local and
PR #71 remains draft without merge authorization.

## Implemented signal pipeline

1. Verify the pinned official architecture and all checkpoint bytes; load every encoder state key
   strictly on CPU, freeze parameters and use evaluation/inference mode.
2. Read canonical physical-mV signals at 500 Hz. Preserve input precision through Fourier resampling,
   then cast to float32, following the official PTB path. Select measured I, II, V1-V6 at 250 Hz;
   no amplitude normalization or fabricated derived-lead features.
3. Extract 400 native contextual tokens: eight leads by 50 patches, 768 features per token. Each
   lead/time anchor spans 200 ms in the original recording's seconds.
4. Detect complete beats without labels. Pool V1/V6 tokens around each QRS with fractional occupancy,
   normalize each region, average equally within patient and equally over support patients per finding.
5. Match target vectors and automatic regions against finding prototypes by cosine similarity. Retain
   original coordinates, all class scores, embedding identity and context. Apply separate temporal
   redundancy decay while keeping disjoint recurring beats.

The reusable [encoder](../../../../ecg_experiment/localization/jepa_native.py) and
[reference study helpers](../../../../ecg_experiment/localization/native_study.py) build on NTT-05's
verified native-region contracts. The [runner](../../../../scripts/experiments/localization/run_ntt06.py)
implements separate profile and evaluation stages. The
[aggregate export](../../../../reports/localization/ntt06-native-encoder/aggregate.json) and
[comparison figure](../../../../reports/localization/ntt06-native-encoder/comparison.png) contain no
per-patient data. Signal plots are output-only local artifacts; they are never converted back into
model inputs.

## Record-finding results

All methods use the same 60 unique targets, 27 unique supports, three draws and three supports per
finding/draw. Targets are 20 strict project normals, 20 CRBBB and 20 CLBBB. These are already explored
PTB development patients, all inside PR67's permitted training role; no patient overlaps a support bank
and target set. The 180 patient-draw choices are repeated evaluations of 60 people.

| Method | Balanced accuracy |
| --- | ---: |
| Native QRS references, primary | 91.67% |
| Native shifted-window references | 83.33% |
| Native QRS query against shifted references | 86.11% |
| Native full-record V1/V6 pooling | 88.33% |
| Native full-record eight-lead pooling | 81.11% |
| Native cyclically relabeled QRS prototypes | 3.89% |
| Frozen physical full-beat references | 92.78% |
| Frozen physical QRS references | 88.33% |
| Historical compact visual QRS-pooled references | 82.78% |

The native primary has 165 correct patient-draw choices, versus physical full-beat 167. Historical
physical/visual methods observe a median beat; the native encoder observes the original ten seconds.
Consequently these are pipeline comparisons, not an isolated representation experiment. Cyclic relabeling
is a deliberately adversarial control, so its below-chance result is not a random-label expectation.
The eight-lead control contains additional input leads and is descriptive rather than a matched primary.

| Prespecified paired comparison | Difference, points | 95% patient interval |
| --- | ---: | ---: |
| Native QRS minus strongest physical baseline, primary | -1.11 | [-5.98, +4.31] |
| Native QRS minus shifted-window native | +8.33 | [+3.67, +13.01] |
| Native QRS minus full-record V1/V6 native | +3.33 | [-0.17, +7.35] |
| Native QRS minus full-record eight-lead native | +10.56 | [+6.25, +15.06] |

Primary admission required at least +5 points and a positive lower interval bound; both fail. The
2,000 patient bootstrap draws retained all three findings, with zero skipped draws. The strongest
physical baseline was reselected in every primary draw. Secondary control intervals are exploratory,
without a multiplicity correction. All intervals condition on these three support banks and this source;
they do not establish hospital transfer, independent confirmation or support-population uncertainty.

| Finding | Native QRS | Physical full-beat | Historical compact visual |
| --- | ---: | ---: | ---: |
| Strict project normal | 98.33% | 98.33% | 81.67% |
| CRBBB | 76.67% | 91.67% | 71.67% |
| CLBBB | 100.00% | 88.33% | 95.00% |

Each row still represents 20 unique people across three draws. These descriptive differences cannot
justify selecting or fusing class-specific winners after seeing the results. Native QRS accuracy varies
from 81.67% to 96.67% across the three frozen support draws, demonstrating material reference-bank
sensitivity. Coexisting annotations are unchanged: MI occurs in 12/20 CRBBB versus 2/20 CLBBB targets,
with additional HYP/STTC co-occurrence. Performance cannot be attributed to isolated BBB morphology.

## What the interval output means

The implementation produced 14,802 candidate assignments across 60 targets and three support draws:
2,022 automatic QRS assignments and 12,780 fixed multiscale-grid assignments. These correspond to
4,934 unique target-coordinate candidates evaluated against three different banks, including 674
unique automatic QRS intervals. Counts are proposals, not independent expert labels.

Each local candidate has half-open start/stop seconds, V1/V6 leads, all three class cosines, chosen
prototype, margin, raw and redundancy-adjusted score, record frame and embedding-space identity.
The grid uses 0.4/0.8/1.6-second windows at 0.4-second stride, including the right edge. Multiple
recurring beats remain separate. Minimum cosine -1 and margin 0 retain every nearest-prototype
assignment; no clinical absence, referral or abstention threshold was calibrated. A selected finding
is not a disease probability and a low decayed score is not an absence label.

Mean QRS-vector cosine agreement is 0.668 with shifted windows, 0.889 with whole-record V1/V6 and
0.750 with whole-record eight-lead pooling. Region placement changes features, but the whole-record
context remains substantial. Token anchors are 200 ms and cannot certify precise 70-160 ms abnormality
boundaries. No independent acceptable-evidence annotations were available, so expert localization
accuracy, proposal recall against expert evidence and absence sensitivity remain unmeasured.

## Integrity and resources

The new runner reproduced every NTT-04 method's saved scores with maximum error 0.0, all 180 choices
and the complete predecessor summary exactly, including the 167/180 physical count replayed in NTT-05.
Original float32 visual descriptors and float64 physical descriptors were restored explicitly from the
JSON cache before replay. No visual model was rerun. Immutable predecessor/manifest hashes and
train-role identities were checked before feature payload consumption; all 174 selected waveform
files matched the predecessor hashes before reading.

| Actual pretrained CPU check | Result |
| --- | ---: |
| Parameters, all frozen | 85,374,720 |
| Synthetic profile load | 22.457 s |
| Synthetic first / repeat dense forward | 0.414 / 0.407 s |
| Maximum dense-mean difference from official pooled output | 1.55e-6, within 1e-5 tolerance |
| Maximum repeated-token difference | 0.0 |
| Synthetic profile peak RSS | 1,419,485,184 bytes |
| Real clinical forwards | 87 |
| Real forward total / mean / maximum | 36.968 / 0.425 / 0.471 s |
| Entire measured development stage | 60.267 s |
| Development peak RSS | 1,514,463,232 bytes |

The profile passed every prospectively declared shape, determinism, equivalence, time and memory gate.
Evaluation admission required matching source, checkpoint and runtime fingerprints plus a completed
profile-status receipt binding its result bytes. Two CPU threads were used. No dependency/lock file
changed. The official checkpoint SHA256 is
`61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e`.
The original paper and pinned pretraining code document Chapman/Ningbo and CODE15 pretraining, which
exclude PTB at source level; the released checkpoint lacks a patient-level training manifest, so exact
patient membership cannot be independently certified. No new pretraining was performed.

The immutable local result SHA256 is
`3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3`.
Private artifacts under `outputs/ntt06_native_encoder/` include dense tokens, pooled feature vectors,
fixed selection, original-time coordinates, predictions, all regional scores and a fixed-first-target
waveform overlay. These remain local and ignored. The native vectors can support a future clustering
protocol, but clustering was not executed in this study and the small finding-balanced cohort cannot
establish unsupervised disease groups or etiology.

Independent saved-output review reproduced all six native methods and all nine historical methods
with zero score error, all 2,000 bootstrap intervals exactly, and all 14,802 candidate assignments.
It checked 24 source hashes, eight input hashes, seven output-artifact hashes, committed source bytes,
checkpoint bytes/size, patient roles, profile admission and aggregate privacy. Dense pooling agreed to
2.78e-16, candidate scores to 7.77e-16 and redundancy decay to 6.91e-15. The reviewer read no waveform
or model: waveform hash claims were cross-bound to the predecessor; the runner performed the actual
174-file byte verification before reading.

The full integrated CPU suite passed **1,491 tests with six skipped in 76.43 seconds**. The focused
native/predecessor suite passed 75 tests in 2.57 seconds. Repository-wide Ruff, offline package build
and new research links passed; the two native modules in the built wheel exactly match their source
bytes. The [verification summary](../../../../reports/localization/ntt06-native-encoder/verification.json)
records these checks. Logs and independent audit details remain local under
`outputs/ntt06_native_encoder_verification/`. No additional inference or scientific score was run during
verification. Follow-ups were added to the backlog and ranked; no subsequent experiment is scheduled.
