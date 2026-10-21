# Frozen real-background transfer results

18 October 2026. Executed once from the local receipts below. No new training or checkpoint
selection.

The fixed SSL-initialized model S reached subject-macro intervention-localization AP 0.289397,
versus 0.232088 for the random-initialized supervised model R. The paired difference was 0.057309
(95% patient bootstrap interval [0.037000, 0.077527]). The advantage came from repeated edits; focal
edits had no clear S advantage. Both models responded to added perturbations compared with exact
matching control maps. S also had higher baseline-site agreement before the edit, so its higher
edited-map AP does not by itself demonstrate a larger response to the added intervention.

This is a separate engineering-development endpoint on known added voltage changes over real ECG
backgrounds. There is no new advancement rule. The original toy-pilot advancement remains **FAIL**.
No clinical normality, diagnosis, native ST localization or clinical screening claim is made.

## Full-grid localization

All 128 evaluation patients were retained: 128 focal and 128 repeated positives, plus 128 unchanged
and 128 added-nuisance controls. Each patient contributes the mean of its two positive modes to
overall AP. Intervals use 2,000 paired whole-patient bootstrap draws, seed 20261023; no draw was
skipped. These intervals describe patient sampling for one fixed trained seed, not training
variability.

| Positive mode | R AP | S AP | S minus R, 95% interval |
|---|---:|---:|---:|
| all | 0.232088 | 0.289397 | 0.057309 [0.037000, 0.077527] |
| focal | 0.149927 | 0.144173 | -0.005754 [-0.033576, 0.019552] |
| repeated | 0.314250 | 0.434621 | 0.120371 [0.097324, 0.144656] |

All methods score the complete 12-by-2,500 map. The uniform map equals intervention prevalence. The
algorithmic-anchor prior uses every candidate window on every lead; it receives privileged
algorithmic timing but no realized edited lead or anchor. Raw absolute voltage uses only the
observed signal.

| Method | Overall AP | Focal AP | Repeated AP | Overall top-1% precision | Overall top-1% recall |
|---|---:|---:|---:|---:|---:|
| R | 0.232088 | 0.149927 | 0.314250 | 0.260833 | 0.103993 |
| S | 0.289397 | 0.144173 | 0.434621 | 0.319792 | 0.110257 |
| uniform | 0.038240 | 0.006032 | 0.070447 | 0.038240 | 0.010000 |
| absolute_voltage | 0.065504 | 0.012029 | 0.118979 | 0.000729 | 0.000209 |
| algorithmic_anchor_prior | 0.099383 | 0.016209 | 0.182556 | 0.099383 | 0.026997 |

Top-1% metrics allocate 300 cells and use the frozen expected allocation across boundary ties.
Overall prevalence was 0.038240; focal 0.006032 and repeated 0.070447. The much greater occupied
area of repeated edits is part of this endpoint, so mode-specific results matter.

## Frozen record thresholds and coverage

Each arm used 64 separate calibration-A patients. Its threshold was the empirical higher 95th
percentile of each patient’s maximum across its unchanged and nuisance-control probability maps;
flags require a strictly greater maximum. Thresholds were fitted separately; native probability
values themselves were not probability-calibrated. Calibration bytes and both calibration map slices
were frozen before any B inference and verified unchanged afterward.

| Arm | Threshold | Unchanged flags /128 | Nuisance flags /128 | Added-edit flags /256 | Focal /128 | Repeated /128 |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.999988317489624 | 10 (7.81%) | 8 (6.25%) | 16 (6.25%) | 9 | 7 |
| S | 0.9993564486503601 | 11 (8.59%) | 9 (7.03%) | 34 (13.28%) | 16 | 18 |

S minus R unchanged-control flag rate was 0.007812 [-0.031250, 0.046875]; the nuisance-control
difference was 0.007812 [-0.015625, 0.031445]. Added-edit flag coverage remained low for both
models. These are flags for known interventions under the stated calibration scheme, not clinical
sensitivity or clinical false-positive rates. An unchanged control can contain unknown native
abnormalities.

## Response to the known addition

For all 256 positives, the regenerated baseline was byte-identical to the same patient’s appropriate
unchanged or nuisance control. The saved control map is therefore the pre-addition map, requiring no
extra inference. Baseline-site AP scores that map against the same fixed edit mask; it measures
agreement at intervention sites before addition, not pathology accuracy.

| Arm / mode | Edited AP | Baseline-site AP | AP change, 95% interval | Target minus unedited-candidate response, 95% interval |
|---|---:|---:|---:|---:|
| R / all | 0.232088 | 0.144730 | 0.087359 [0.055703, 0.121192] | 0.072747 [0.037850, 0.112212] |
| R / focal | 0.149927 | 0.031004 | 0.118923 [0.077736, 0.162819] | 0.093599 [0.044752, 0.144466] |
| R / repeated | 0.314250 | 0.258456 | 0.055794 [0.032418, 0.081611] | 0.051896 [0.026103, 0.079651] |
| S / all | 0.289397 | 0.223988 | 0.065409 [0.038837, 0.093961] | 0.045901 [0.016296, 0.075073] |
| S / focal | 0.144173 | 0.046488 | 0.097685 [0.061692, 0.134192] | 0.064823 [0.025039, 0.105207] |
| S / repeated | 0.434621 | 0.401488 | 0.033133 [0.011107, 0.056367] | 0.026979 [0.005710, 0.049251] |

The response contrast is mean probability change inside the exact intervention support T minus that
in unedited candidate cells U. O is outside all candidate windows. The following values are
within-arm descriptive probability changes; uncalibrated scores across arms do not establish
comparative probability calibration.

| Arm | Mean change T, 95% interval | Mean change U, 95% interval | Mean change O, 95% interval |
|---|---:|---:|---:|
| R | 0.083646 [0.040306, 0.131615] | 0.010899 [0.001371, 0.020884] | -0.001272 [-0.003895, 0.001008] |
| S | 0.053470 [0.018059, 0.088157] | 0.007570 [0.001065, 0.013643] | 0.000530 [-0.000147, 0.001347] |

Both AP-change and selectivity lower bounds were positive in both modes for each model. This
supports responsiveness to the engineered input addition. S’s overall baseline-site AP (0.223988)
exceeded R’s (0.144730), while its point AP response was smaller (0.065409 versus 0.087359). The
higher S edited-map AP therefore cannot be attributed solely to greater edit responsiveness. No new
pass criterion or model change was introduced after inspecting these maps.

## Qualification, provenance and resources

Before scoring, the data preparation retained all 192 selected validation patients and all 768
variants, with 768 deterministic replays and 384 exact positive/control baseline checks. There were
zero integrity, anchor or case failures and no replacements. Retained anchor counts ranged from 7 to
17. The selected set happened to contain zero technical quality exclusions and zero review flags; no
selection filter removed such records. The separately reserved 500 TRAIN patients’ waveforms were
not opened.

All support masks were computed from the final once-cast float32 observations at a strict 0.005 mV
floor. Maximum support occupancy was 0.148467, below 0.18; maximum observed delta was 0.342431 mV.
Maximum change outside reported support was 0.004998684 mV, below the frozen floor. Maximum
quantization residual was 4.52795e-7 mV; added-limb algebra error was at most 1.19209e-7 mV and
passed its per-case rounding bound. Native limb leads were preserved rather than reconstructed.
These checks qualify implementation truth, not physiological beat or ST validity.

The original passed numerical audit and explicit binding closure, fixed checkpoint/component hashes,
completed counterfactual diagnostic and all four prospective response-admission conditions were
freshly verified. The unused original TRAIN cache waveform payload was not reopened: its digest
remained bound through frozen metadata, as declared. Selected real-background data were loaded only
locally. No diagnostic field, original real engineering-test payload, closed test set or extra
cohort was consumed.

The reviewed implementation was committed as `ae9de9b`, after prospective execution supplement
`42d36ae`; the coordinator integrated these as `958bb36` and `64e0d43`. Eleven artificial execution
tests passed, and the coordinator’s combined background/transfer suite passed 28 tests. Ruff passed.
Tests covered calibration/global-index isolation, exact matching, observed-input-only inference,
four-condition admission failure, storage arithmetic and stop receipts.

One actual-path profile used calibration A only, fixed R, batch 16, ten warm-up and twenty timed
batches. It discarded probe maps and computed no performance metric. Model state was unchanged. The
measured mean batch wall time was 0.027044 seconds; the prospective 2x-plus-reserve forecast was
131.719 GPU-wall and 136.749 CPU seconds for the stage, below its 900/1,800-second limits.

| Executed cost | Wall / GPU-lock seconds | CPU seconds |
|---|---:|---:|
| Profile | 6.526349 | 6.507916 |
| Evaluation including summaries/hash verification | 20.072364 | 19.986500 |
| New stage including preparation/post-verification | 26.598715 GPU-lock | 31.540014 |
| Cumulative charged history at evaluation | 136.022255 GPU-wall | 421.240948 |
| Both independent audits, additional | 12.894763 wall | 12.891983 |
| Complete cumulative CPU including both audits | unchanged GPU use | 434.132931 |

All numerical and PyTorch interop thread counts were one. Peak allocated CUDA memory was 43,825,152
bytes. Whole-stream storage at completion was 3,030,613,071 bytes, below 4 GiB. The new-stage array
forecast counted 253,440,000 payload bytes once, plus five NPY headers and a shared 64 MiB metadata
reserve. No optimizer step, fit, new checkpoint or extra inference for counterfactual diagnostics
occurred. No profile/evaluation failure occurred in this stage; earlier failed profile, stopped
counterfactual admission and audit history remain retained.

All 1,536 native probability maps were saved in two local float32 arrays with shape
768-by-12-by-2,500 and global case indices. All 768 regenerated cases matched four frozen array
hashes before inference. After execution, all 75 source, 154 input and six adjacent output bindings
passed fresh verification; both calibration slices were unchanged and every probability was finite.
This is production verification; independent saved-map recomputation is reported separately when
complete.

## Independent saved-map verification

The first independent audit made 89,802 comparisons and retained eight schema-key mismatches:
its earlier schema omitted the two per-arm positive-mode flag groups and six arm/mode groups of
extra top-1% response summaries. All values it compared matched; its receipt remains failed and
unchanged. An additive completion checked the omitted fields and retained bindings. The completed
combined audit passed 90,376 checks with zero mismatches at numerical tolerance 1e-12, including
thresholds, native map coordinates, all positive-mode flags, AP/top-1% metrics, all thirteen response
metrics, patient-macro arithmetic and paired intervals. It verified 185 input bindings. Both audit
attempts are counted in the final CPU charge above.

The independent auditor read saved masks, maps and metadata, with zero waveform payload reads,
model inferences, optimizer updates or GPU use. It did not independently regenerate native waveform
identity, anchors or physical intervention truth; those remain bound through qualification sources
and hashes. Raw-voltage per-case values were not independently recomputed, only their aggregate
arithmetic. Historical calibration ordering is supported by bound execution sources and unchanged
slice hashes rather than an independent observer of call order. This verifies the reported
engineering calculation within those limits; it does not validate native pathology or establish
that SSL produces a stronger perturbation response.

## Local receipts

All paths below are under `outputs/ssl_synthetic_localization/real_background_v1/`; payloads,
patient IDs, masks, predictions and checkpoints remain local and untracked.

| Receipt | SHA-256 |
|---|---|
| `preparation/summary.json` | `32b387379d3fc8d11e52d2bd9e022f9cadc6b69b4e58f1f00f06de7286921b7b` |
| `preparation/post_verification.json` | `da2fd1595e987f5304d2eb7cde5cedbe1a65ab2e8adcf17d110cc36f3b13b40f` |
| `profile/summary.json` | `895d13375fb0ffb45f7b86500a9f903307a96c274fb6a2ef8862d5a19b831332` |
| `evaluation/calibration.json` | `3d73bc5c54a70a7e8c2dceb7a72149fc3ace1f540530235b49179d9b0e6f4c24` |
| `evaluation/summary.json` | `ed578802c4864a4534bc06ba115f7a0835922d5b0c6c16d0be839be148db08ce` |
| `independent_audit/summary.json` (schema failure retained) | `1abb154769d86615460e7b66e6f44760c6994afd0d9408a68b89535b7878278d` |
| `independent_audit_completion/summary.json` (combined pass) | `b701dd865d7f8991521cd9199fc3fab39963fb89e177c3e383e0a033289e13f8` |

The evaluation receipt binds native map files, per-case metrics, regeneration checks and admission
metadata. The full protocol is in [the study](ssl-synthetic-localization-real-background-study.md),
[once-cast clarification](ssl-synthetic-localization-real-background-clarifications.md) and
[execution supplement](ssl-synthetic-localization-real-transfer-execution.md).

Unknown native abnormalities can resemble the added change. Algorithmic timing is not expert
verification, and the original B mechanism was already spent development evidence. These results do
not explain the causal effect of domain mismatch by comparing AP with the old toy endpoint. A new
fitting or domain-alignment comparison would need its own prospective protocol on a fixed endpoint;
none was launched here.
