# Experiment 056 results: expert ventricular-beat localization on INCART

The unsupervised aligned morphology residual passes both prospective localization endpoints and both
guardrails on independent expert beat types. It is promoted for user review. Patient-macro top-beat
accuracy rises from 74.13% to 89.55%, and V-versus-N AUROC rises from 0.8701 to 0.9550. These results
establish improved ventricular ectopic **beat identity**, not exact pathological wave, lead, ST region,
anatomy or diagnosis. The existing referral detector and clinical label definitions are unchanged.

Completed 2 October 2026 under the [frozen protocol](experiment-056-incart-beat-localization.md) and
[record manifest](experiment-056-incart-records.json). Protocol commits were `17b0292`, `a2c1642` and
`42dacd5`, all before scoring; source commit was `da9ef7a`. The two amendments make the endpoint stricter:
every scored waveform-detected beat is a possible top choice, and expert mixed V/N cores remain in the
denominator even if their V is missed. No recipe, tolerance, endpoint or threshold changed after scores.

## Independent evidence and integrity

All 75 public INCART records, grouped into 32 distinct header patients, were evaluated. The official
[PhysioNet INCART 1.0.0 description](https://physionet.org/content/incartdb/1.0.0/) states that experts
corrected beat types but did not manually correct annotation times. Waveforms, headers and annotations
were downloaded from its official public S3 mirror and remain local. The metadata manifest verifies
patient grouping against the official grouping text, twelve actual leads, 257 Hz and gains of
240–1,063 ADC units/mV. Every record passed its digital checksum and initial-value checks before physical
gain conversion and deterministic full-record polyphase resampling to 500 Hz.

The frozen metadata proof checks all 65,824 rows of the repository's Challenge split: its five source
families contain no INCART record. No closed Challenge or reserved calibration waveform was accessed.
INCART's appearance in the broader historical Challenge2020 does not make it a member of this
repository's split. No INCART labels fit a reference, detector, alignment, scale or threshold. File
bytes were hashed for provenance; expert annotation symbols were parsed only after waveform scoring
finished for the corresponding record.

Before INCART scores, the runner verified predecessor sources, input and output hashes and reproduced
052's and 054's complete required metrics exactly. Rebuilding the U_B normal reference from exactly
5,872 PTB training normals and 63,479 retained beats reproduced all 1,604 saved development unit maps:
geometry identical, maximum absolute score difference **0**, maximum relative difference **0**. The
same 63,479 training-normal beat scores supplied method-specific 95th-percentile thresholds.

Both methods use the same waveform-only R peaks, complete-beat support, midpoint ownership and
140 ms / 10 ms candidate grid. The single primary candidate is frozen 054's uncalibrated aligned-only
residual, with limited shared QRS alignment and no per-beat amplitude fitting. It uses neither an RR
anomaly feature nor INCART annotation positions as inference anchors. This is a prospectively chosen
new external comparison; it does not overturn 054's failed primary phase-calibrated/synthetic gates.

## Co-primary patient comparisons

Intervals are paired whole-patient bootstraps with 2,000 draws and seed 56056. Each eligible patient has
equal weight across its pooled records; aggregate beat counts do not determine patient weight.

| Patient-macro endpoint | Aligned residual | U_B fixed | Paired gain, 95% interval | Frozen rule |
| --- | ---: | ---: | ---: | --- |
| Matched expert V versus N AUROC | 0.95501 | 0.87013 | +0.08488 [0.04298, 0.13184] | Pass: gain >=0.05, lower >0 |
| Mixed-core top-beat hit | 0.89551 | 0.74130 | +0.15420 | Descriptive absolute hit |
| Mixed-core hit minus candidate-pool chance | 0.69137 | 0.53717 | +0.15420 [0.06571, 0.25813] | Pass: gain >=0.10, lower >0 |
| Average precision | 0.58431 | 0.35642 | +0.22789 | Descriptive |

Both co-primary endpoints include 30 patients. Patient 8 has no ordinary N beats, and patient 30 has
no V beats; neither supplies a binary AUROC or mixed V/N core. They remain in applicable coverage and
control tables. There are 5,685 expert mixed cores. Top selection includes unmatched detections and
other beat types, with ties evaluated by expected V hit. Chance is the matched V share of the entire
scored candidate pool and is identical for both maps: patient-macro 20.41%.

The candidate has better AUROC in 26 patients and worse AUROC in four. Mixed-core excess improves in
19, worsens in nine and ties in two. Across individual mixed cores it improves top-beat hit in 899,
worsens it in 409 and ties in 4,377. Pooled-core accuracy is 88.53% versus 79.90%; it differs from the
primary patient-macro comparison because patients supply different numbers of mixed cores.

## Coverage, specificity and nuisance

There are 175,907 expert beat annotations and 176,214 scored detections. Of 20,013 expert V beats,
19,998 are matched (99.925%); of 150,410 N beats, 150,303 are matched (99.929%). In total 124 reference
beats are unmatched and 431 detections are unmatched. Coverage and matching are shared by both methods.
Median matching distance is 5.67 ms, 95th percentile 23.56 ms and maximum 133.15 ms under the frozen
150 ms tolerance. The greedy distance-first matcher is not maximum-cardinality; neither timing nor
unmatched-reference counts are manually adjudicated or changed after the result.

Other expert types are retained separately: R 3,174; A 1,944; F 219; j 92; n 32; S 16; Q 6; B 1. Twelve
nonbeat annotations are excluded. Two cores, I02/66 and I02/67, have fewer than three complete detected
beats and cannot be scored. Their expert beats remain in coverage, and any expert mixed-core eligibility
remains in the localization denominator. No record or patient was replaced.

| Threshold/control endpoint | Aligned residual | U_B fixed |
| --- | ---: | ---: |
| PTB training-normal beat 95th percentile | 0.118205 | 430.585059 |
| Patient-macro matched N false flags, 31 N-positive patients | 19.09% | 50.42% |
| Conditional V sensitivity, 31 V-positive patients | 96.76% | 96.45% |
| End-to-end V sensitivity, 31 V-positive patients | 96.70% | 96.39% |
| First-core unchanged beat flag share | 27.99% | 52.03% |
| First-core nuisance beat flag share | 27.99% | 52.03% |
| Mean first-core nuisance score change | +0.001380 | -0.014745 |
| Mean first-core unchanged/nuisance rank correlation | 0.77612 | 1.00000 |

The paired patient N false-flag change is -0.31331 [-0.40445, -0.23158], comfortably passing the required
upper bound <=+0.05. Both nuisance flag-share changes are zero and pass the candidate's <=+0.05 gate.
The nuisance test uses the first core of every record, unchanged peaks, +0.15 mV constant offset and
0.05 mV / 0.2 Hz drift. It is not a test of all artifact types. Residual ranking changes despite stable
flags, so the candidate is not fully invariant to drift.

The absolute 19.09% normal-beat false-flag rate remains substantial. These are in-sample PTB training
thresholds transported to long clinical Holter recordings, not a validated student referral budget.
The receipt's all-32-patient end-to-end means are 93.68% and 93.37%, because its implementation assigns
zero to the one patient without V; the table above descriptively restricts to the 31 V-positive
patients using the saved counts. Neither reading affects the frozen promotion gates.

## Local examples and limits

Four deterministic raw waveform comparisons were saved under
`outputs/experiment056_incart_localization_v1/figures/` and inspected:

In the win/loss figures, green is +/-100 ms around a detected R anchor matched to an expert V
annotation, for orientation only; INCART annotation times were not manually corrected. Red is the
140 ms anomalous window assigned
to its nearest detected beat by midpoint ownership. A hit validates beat identity, not gold-standard
QRS delineation or pathological lead/wave support. The unmatched-V diagnostic instead uses +/-150 ms
to show the matching tolerance.

- `win_I01_13.png`: the residual chooses the expert V's owned beat neighborhood while U_B chooses a
  later ordinary beat. The red residual window is on a post-QRS morphology segment; a correct beat
  identity does not establish the precise abnormal wave or lead.
- `loss_I01_0.png`: the residual selects an ordinary beat while U_B selects the expert V. This failure
  remains in the aggregate results.
- `normal_false_alarm_I01.png`: a matched ordinary N exceeds the fixed PTB-normal threshold.
- `unmatched_expert_v_I18.png`: an expert V has no accepted waveform-detected match. The green tolerance
  band is approximate annotation timing, not a clinical QRS boundary.

Within-ECG template subtraction can suppress persistent abnormalities shared by every beat. The model
does not diagnose bundle-branch block, localize structural heart disease, or establish ST/lead ground
truth. INCART is a clinical Holter cohort with repeated patient records and ectopy, not a prospective
university-student screening cohort. Conditional AUROC and mixed-core localization cover different
parts of this limitation; the saved unmatched beats and fixed-threshold errors must accompany review.

All four prospective gates pass, so `promote_for_review` is true. This is a concrete improvement worth
reviewing with the user. Further recipe trials stop while this review package and the separate 057
improvement are assessed; no diagnosis-label change or cardiologist decision is required to review
these beat-localization results.

## Artifacts and validation

Outputs are `outputs/experiment056_incart_localization_v1/`: receipt, all detected/reference beat rows,
mixed-core and patient tables, coverage/exclusions, nuisance pairs, normal reference, training-normal
beat scores, four local figures and run log. All receipt-listed output hashes were rechecked after
completion. Receipt SHA-256:
`124e8f250715dcb51ca211e23582115b1a07da79052563efa5dc46094d43f452`.
Raw data and output waveforms were not staged, uploaded or sent to an external service.

The two-thread CPU run took 808.63 seconds and used no GPU. Eighteen meaningful 052/054/056 tests and
Ruff passed before launch, including exact U_B piece extraction, physical gains/canonical leads,
resampling, one-to-one matching, shared ownership and equal patient weighting with missed-V cores.

```bash
CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/ecg-mpl056 \
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=/tmp/ecg-localization052 \
.venv/bin/python -m scripts.experiments.run_incart_localization056
```

The runner refuses an existing final or partial output directory. Executed sources remain pinned.
