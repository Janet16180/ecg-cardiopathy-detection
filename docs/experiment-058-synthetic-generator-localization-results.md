# Experiment 058 results: paired generators qualify, localization robustness fails

The paired raw waveform generators pass all frozen engineering checks. The frozen aligned residual
does not pass the synthetic localization robustness rule. Its strong focal T-source results do not
compensate for weak QRS/ST localization, severe extra-noise sensitivity and persistent-ST failures.
No localizer, classifier, clinical label or referral rule is changed or promoted.

The ECGSYN-equation family's persistent T-source change is especially broad: its observable mask
occupies **85.9–89.2% of all lead/time samples**, with eligible-candidate chance 85.5–88.5%. Its 97.5%
raw hit is therefore not evidence of precise physiological T-wave localization. The remaining ODE
component tails are part of the actual counterfactual change, not expert wave boundaries. Lead credit
similarly means an observed projected voltage change, not a clinical territory or anatomical lesion.

## Frozen execution and generator validity

The [protocol](experiment-058-synthetic-generator-localization.md) was committed as `089a7c0` before
scores. Prospective amendment `e7081f0` separates unchanged respiratory/initial-state background from
the T component and requires support occupancy/chance reporting. Implementation was frozen as
`cf45db4`; its sources remain unchanged. Eight new scientific tests passed after this clarification,
and Ruff passed. The combined 052/054/056/058 suite had passed 26 tests before the background split;
the subsequent eight tests exercise that split and its paired-support invariants.

The established-equation family independently implements the mathematical ECGSYN model of McSharry
et al. (IEEE TBME 50(3), 289–294, 2003), using the
[official ECGSYN 1.0.0 algorithms](https://physionet.org/content/ecgsyn/1.0.0/). The official 161 KB
source archive stays local under its published license; its checksum matches the protocol. The C
implementation cannot compile directly without omitted Numerical Recipes files, its 32-bit Linux
binary lacks a local loader, and Java/Octave are unavailable. No upstream code was copied into the
repository and no dependency was added.

Our independent implementation uses an analytic unit-cycle phase and exponential linear-state
recurrence, rather than the official Matlab ODE solver. It also projects component vectors through
virtual electrodes to construct coupled twelve-lead voltages. The separate compact-pulse family is
an engineering control. Neither arm is a clinically validated heart/torso or disease model; exact
lead algebra qualifies a controlled signal generator, not physiological realism.

There are 600 seeded subjects per generator: 200 normal calibration, 200 IID tests and 200 tests with
shifted geometry/heart-rate/morphology ranges. Generator versions share latent subject IDs and are
reported separately. All 1,200 clean renders and their programmed edits passed qualification:

| Frozen check | Executed result |
| --- | ---: |
| Maximum processed sham/counterpart difference | 0 |
| Maximum exact limb-lead algebra error | 1.776e-15 mV |
| Observable masks reconstructed from actual noiseless paired differences | All match |
| Uninformative pathology edits, without replacement | 0 |
| Excluded subjects or unscorable test traces | 0 |

Masks use the actual absolute paired difference >=0.01 mV, then retain leads with at least 20 ms
above that floor. The localizers receive observed signals only. They never receive a clean test
counterpart, latent source anchor, intervention identity or target support. The frozen 056 waveform
detector, aligned residual and U_B reference construct the same 140 ms / 10 ms candidate domain.
All tied maxima receive expected localization credit; a hit requires both lead and center to be
inside the observable mask.

Before new scoring, all required predecessor hashes and exact metrics were reproduced: 056's full
benchmark, nuisance fields, thresholds and coverage, plus 057's primary and stress comparisons.
No closed data was accessed. Generator qualification took 133.83 seconds; the entire two-thread CPU
run took 1,099.17 seconds. It used no GPU.

After completion, an independent read-only audit reproduced all 7,200 cases' 14,400 map metrics and
flags to within 5.6e-17, all 468 per-kind intervals, four transient intervals, twelve nuisance
intervals and all four gates to within 1e-13. It verified every one of the twelve source hashes,
two input hashes and 8,460 output hashes, plus the cached parameter splits/projections, limb checks
and calibration thresholds. The negative robustness verdict was independently confirmed.

## Held-out transient localization

Each row contains 200 independent test subjects. Entries are aligned-residual / U_B hit percentages.
The final column averages QRS, ST and T within each subject before paired whole-subject bootstrap:
2,000 draws, seed 58058. Generator versions and distributions are never pooled.

| Generator / distribution | Focal QRS | Focal ST | Focal T source | Mean transient paired gain, points [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| ECGSYN equations / IID | 75.5 / 78.0 | 64.5 / 53.0 | 100 / 5.5 | +34.50 [29.83, 39.00] |
| ECGSYN equations / shifted | 49.5 / 41.5 | 49.5 / 45.5 | 100 / 8.0 | +34.67 [29.50, 39.67] |
| Compact engineering / IID | 50.5 / 93.0 | 86.0 / 42.5 | 100 / 5.5 | +31.83 [27.67, 36.00] |
| Compact engineering / shifted | 56.5 / 64.0 | 70.0 / 45.5 | 100 / 3.5 | +37.83 [33.33, 42.33] |

All mean transient gains exceed ten points with positive lower bounds. Those aggregate gains are
dominated by the focal T-source edit: **every QRS cell fails the required 80% hit**, and three of four
ST cells fail it. Compact IID QRS regresses by -42.50 points [-51.00, -34.50]. ECGSYN shifted ST gains
only +4.00 points [-4.50, +12.00]; its confidence interval includes zero. Perfect focal T-source hits
cannot rescue these failed component requirements.

Focal support occupancy is approximately 1.1% of all lead/time samples for QRS, 0.8% for ST and
1.6–2.0% for T. Uniform eligible-candidate chance is 1.9–2.2%, 1.4–1.5% and 2.5–2.7%, respectively.
The single 140 ms lead highlight captures only 4.7–9.1% of total QRS difference energy, 1.2–2.2% of
ST energy and 2.1–3.1% of T energy. Correct center/lead credit does not mean the full multilead change
was recovered. Exact-mask IoU and all distance/tie readings remain in the per-case tables.

## Persistent changes and support-size limits

Persistent edits cover all latent beats or the entire T component, leaving no normal beat for a
within-record template. They are separate diagnostic conditions, not alternate targets selected
after outcome inspection.

| Generator / distribution | Persistent ST hit, residual / U_B | Persistent T-source hit, residual / U_B | T-source chance | T-source full-mask occupancy |
| --- | ---: | ---: | ---: | ---: |
| ECGSYN equations / IID | 14.0 / 58.0% | 97.5 / 99.2% | 88.5% | 89.2% |
| ECGSYN equations / shifted | 13.0 / 46.0% | 97.5 / 99.1% | 85.5% | 85.9% |
| Compact engineering / IID | 0.0 / 52.0% | 80.5 / 22.7% | 38.0% | 25.2% |
| Compact engineering / shifted | 6.0 / 46.5% | 75.5 / 37.1% | 34.3% | 26.3% |

Persistent ST is a clear failure: aligned-residual hit is below its 18.7–19.4% uniform candidate
chance in every distribution. Paired changes are -44.0 [-53.0,-35.0], -33.0 [-41.5,-24.5],
-52.0 [-59.0,-45.0] and -40.5 [-47.5,-34.0] points in the table's row order. The observable persistent
ST mask occupies about 11.3% of all lead/time samples.

ECGSYN's broad persistent T-source mask arises from the stateful component's long decay, even though
respiratory/background contribution remains unchanged. Do not interpret its high hit as precise
T-wave recognition. The compact family's support is narrower but still includes many repeated
changes; it is not an independent physiological wave annotation. Neither result establishes correct
pathological wave ownership, territory or anatomy.

## Sham and artifact controls

Record-max thresholds are fitted to the 95th percentile of the 200 unchanged calibration subjects
per generator, then frozen before test scoring. These are simulator controls, not a student referral
budget or a deployable clinical operating point.

| Generator / distribution | Test sham flags, residual / U_B | Residual extra-noise flags | Paired extra-noise flag increase, points [95% interval] |
| --- | ---: | ---: | ---: |
| ECGSYN equations / IID | 5.5 / 7.5% | 71.5% | +66.0 [59.0, 72.5] |
| ECGSYN equations / shifted | 14.5 / 60.5% | 68.5% | +54.0 [46.5, 61.0] |
| Compact engineering / IID | 6.0 / 12.5% | 94.0% | +88.0 [83.0, 92.5] |
| Compact engineering / shifted | 11.5 / 59.5% | 88.0% | +76.5 [70.0, 82.5] |

Every extra-noise guard fails the maximum five-point upper bound. Calibration background electrode
noise is 0.003 mV; the prespecified added-noise stress has standard deviation 0.02 mV. The candidate
reacts strongly to this change and cannot distinguish it from a localization-worthy signal change
under the frozen control criterion. This is a specific stress finding, not a claim that every noisy
clinical ECG is benign.

Constant/slow-drift and common 1.1 gain controls pass: their paired upper bounds are at most 3.5 and
2.5 points. Shifted clean sham flag rates rise for both methods, especially U_B. Thus apparent gains
against the real-normal reference also reflect generator-domain mismatch; they cannot establish
clinical superiority. The full criterion is conjunctive, and all four generator/distribution cells
fail. The receipt records `provisional_synthetic_robustness_pass=false` and `clinical_promotion=false`.

## Artifacts, review and next implications

Complete outputs are `outputs/experiment058_synthetic_generator_v1/`: 1,200 clean subject caches,
the locked parameter manifest, qualification, normal cutoffs, 7,200 paired cases with exact differences,
energy/masks and both candidate maps, 14,400 method rows, final summaries, 55 local figures and run log.
The receipt pins 8,460 artifacts. Output size is approximately twelve GB, bounded by the frozen subject
and intervention counts; raw traces and upstream source archives remain local and were not staged.
Receipt SHA-256: `5e0356a0f2bed20d11254817a0b39082a393df865f5d90c6f3fb999c90d09cdb`.

Plots use gray clean counterparts for evaluation display, black observed intervened signals, green
actual observable support and red 140 ms model marks. The clean counterpart is never supplied to
inference. Examples are the first win/loss/tie in each generator/distribution/intervention category:
they are outcome-stratified diagnostics, not blind representative sampling. Every case is retained
in the aggregate metrics. The inspected `ecgsyn_shifted_persistent_st_loss_400.png` shows the aligned
residual selecting outside repeated ST changes while U_B selects inside them.

This completes the proposed generator investigation and leaves a reusable controlled stress bench.
Any future simulator-developed method must itself pass an independent real expert comparison;
successes of different algorithms in 059/060 cannot validate this generator's clinical transfer.
The present classifiers remain frozen. Further recipe trials stop for user review of the separate
independent real-data improvement; this negative synthetic robustness result accompanies that review.

```bash
CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/ecg-mpl058 \
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=/tmp/ecg-localization052 \
.venv/bin/python -m scripts.experiments.run_synthetic_generator058
```

The runner refuses existing partial or complete output directories. All executed sources remain pinned.
