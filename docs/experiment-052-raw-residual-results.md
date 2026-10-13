# Experiment 052 results: raw within-ECG morphology residuals

The raw residual map does not improve localization overall. It correctly marks the odd wide beat in
ECG 287, but fails the prospective improvement rule and both synthetic supporting rules. Referral and
the existing explanation maps remain unchanged.

Completed 2 October 2026 under the [frozen protocol](experiment-052-raw-residual.md): initial freeze
`7071dc4`, prospective scientific clarifications `614016e` and `2baa625`, implementation/run commit
`8411072`. All clarifications preceded every new score. The CPU run took 84.36 seconds, with two threads
and no GPU. Live outputs are `outputs/experiment052_raw_residual_v1/`: `result.json`,
`development_maps.npz`, `development_locations.csv`, `synthetic_locations.csv`, `run.log`, and local
`figures/`. No waveform was uploaded, downloaded or sent to an external service. No calibration, closed
test, EchoNext or MIMIC waveform was read.

## Primary result

All 1,604 development ECGs were usable; there were no exclusions. The primary comparison uses the same
140 ms windows, 10 ms stride, and exact common sample eligibility for both maps. U_B pieces produce
flat plateaus, so both maps' primary metrics average over tied maximal candidates using the frozen
`isclose` tolerance. The target is the automatically identified early complex's QRS support, R - 0.06
through R + 0.08 seconds. This target is an exploratory proxy, not a cardiologist annotation.

| On 73 PVC ECGs with automatic targets | Raw residual | U_B fixed support |
| --- | ---: | ---: |
| Window-center temporal hit | 0.479 | 0.488 |
| Uniform eligible-window chance | 0.048 | 0.048 |
| Hit minus chance, patient interval | 0.432 [0.314, 0.552] | 0.440 [0.334, 0.546] |
| Median distance from closest target R | 126 ms | 72 ms |
| Mean target overlap / displayed width | 0.415 | 0.504 |
| Records with tied maxima | 0 | 40 |
| Mean maximum tie count | 1.00 | 5.92 |

The paired gain is **-0.009 [-0.166, +0.139]**, failing both the required +0.10 point gain and a lower
bound above zero. Intervals use 2,000 whole-patient draws and seed 52052. The identical candidate space
gives identical chance on each paired ECG; no wider marks or RR anomaly score can create a gain.

## Exact-support synthetic checks

The run evaluated 200 training normals from 200 unique patients, disjoint from all development
patients. Selection was ascending ECG ID with the frozen requirement of five complete beats; no record
was skipped. No population model was fitted in this experiment. These are training records previously
available to the project, not a new clinical holdout. Synthetic R positions stay fixed to the original
ECG, so an edit cannot benefit by changing the peak detector.

| Known edit | Exact time-and-lead joint hit, patient interval | Worst-score change from unchanged copy | Mean overlap / displayed width | Supporting rule |
| --- | --- | --- | ---: | --- |
| Single-lead 0.15 mV ST bump | 0.205 [0.150, 0.265] | +0.80 [-0.44, +1.70] | 0.146 | Fails |
| Three-lead stretched QRS transplant | 0.775 [0.715, 0.830] | +87.34 [+72.07, +102.96] | 0.784 | Fails |
| ST moved to another interior beat | 0.225 [0.170, 0.285] | +3.18 [+1.04, +7.04] | 0.163 | Descriptive |
| ST with cyclic lead permutation | 0.205 [0.150, 0.265] | +0.80 [-0.44, +1.70] | 0.146 | Descriptive |

The ST map often selects an unmodified part of the ECG; its median distance from the exact edit's center
is 2.148 seconds. The QRS transplant's median distance is 10 ms, but its 77.5% joint hit still falls
below the prespecified 80%. The exact lead-permutation result matches the original ST result. Synthetic
QRS edits are template-generated signal changes and naturally favor a beat-template method; even a
success would not validate clinical PVC localization. The unchanged comparison does not include a
synthetic U_B claim: its raw reference pipeline was not recomputed on edited ECGs.

The combined constant-offset and slow-drift nuisance control flags 5% of records, the same as unchanged
copies, using the unchanged records' 95th percentile threshold of 630.753. Its mean score change is
-17.51 [-52.41, +11.06]. This threshold is calibrated on these same 200 records, so this false-flag
share is descriptive and in sample.

## ECG 287 and deterministic review examples

The raw map's top window is **V2 at 6.04–6.18 seconds**, centered at 6.11 seconds: 4 ms from the automatic
R target at 6.114 seconds. It overlaps 90% of that QRS support. U_B's equal-width top window is V1 at
9.17–9.31 seconds, 3.126 seconds from the target. The raw score is 2,364.42. This demonstrates a useful
location on the motivating example, but does not overcome the negative overall result.

Local figures include the seven prespecified ECGs (47, 219, 8, 184, 287, 30, 69), the lowest-ID real gain
(287), the lowest-ID real loss (1984), and the lowest-ID successes and failures for each synthetic
perturbation. Files are under `outputs/experiment052_raw_residual_v1/figures/`; green shading identifies
the known synthetic or automatic target, and red identifies the equal-duration descriptive map mark.
They are review material, not validated anatomy or diagnoses. For tied maxima, the descriptive mark is
the median-time maximum; primary metrics average all maxima instead.

## Descriptive detection and historical window rules

| Descriptive metric | Raw residual | U_B fixed support |
| --- | ---: | ---: |
| Worst-window AUROC, 1,306 binary-evaluation ECGs | 0.652 | 0.739 |
| Average precision | 0.772 | 0.839 |
| Any red: 463 normals | 0.052 | 0.052 |
| Any red: positives | 0.165 | 0.292 |
| Any red: 84 PVC ECGs | 0.857 | 0.798 |
| Any red: 52 benign variants | 0.058 | 0.077 |
| Anterior lead contrast | +0.018 [-0.102, +0.132] | +0.085 [-0.038, +0.202] |
| Inferior lead contrast | +0.038 [-0.041, +0.121] | +0.129 [+0.014, +0.243] |

A within-record residual subtracts abnormalities shared by all beats, so its weak detection is expected.
It cannot replace the referral model. The red thresholds are each map's development-normal 95th
percentile; they are not final clinical operating points.

The legacy broad 041 window overlap, evaluated with tie averaging on the new fixed maps, is 0.877 for
raw and 0.888 for U_B. Strict midpoint beat ownership is 0.904 and 0.932. The additional copied 042
descriptive metric block preserves that historical helper's earliest-argmax convention, giving U_B
overlap 0.863. This difference is caused by ties; it does not enter the primary decision.
That reused block also retains an `improves_on_041` reading against the older 041 map. It is a
historical descriptive comparison and does not mean the failed 052 primary rule improved on U_B.

## Integrity and limitations

Before every new score, the full historical U_B metric dictionary and its threshold reproduced
Experiment 042 exactly, including bootstrap intervals and 041 contrasts. The unit scores match their
receipt. The 042 result receipt matches the hash stored by 051. Metadata, relevant frozen helper
sources, 041 maps/result, pinned dependencies and the historical runner match 042's recorded hashes.
The new receipt records source/protocol/input hashes and all read waveform header/data hashes. Seven
meaningful tests and Ruff passed before launch.

There are no outcome-driven exclusions or method changes after scoring. All real data are reused
development data. The automatic QRS target derives from rhythm timing; timing is used for alignment,
never as a score feature. Lead contrasts concern whole-record annotations and cannot validate a local
wave mark. The synthetic controls recover fixed signal edits, not heart disease. Synthetic candidate
maps were not separately persisted; their exact locations and scores are in the per-record table and
representative figures. Development maps are fully persisted.

## Follow-up

The pointwise leave-one-out MAD map finds the motivating wide beat, but does not robustly distinguish
the smaller ST edit from existing waveform variation. A prospective successor could reduce normal
QRS alignment variability and calibrate scores by wave phase using independent training normals,
retaining equal candidate supports and untouched synthetic evaluation patients. This is a new
hypothesis requiring its own committed protocol and outputs. Experiment 052 is negative and is not
promoted to the existing explanation pipeline.
