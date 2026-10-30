# NBR-01 protocol: where the screening blend sees the problem, by normal-beat replacement

Frozen 28 October 2026, before any replacement score. Branch `research/normal-beat-replacement-20261028`,
started from `research/ml-improvement-20261028` (`906f9fe`). The choices below were made by the agent from
the lead's brief and are fixed here.

## Question

Take an SPH ECG, replace one region with its normal counterpart, and score it again with the screening
blend. A region is one lead, one wave component (P, QRS or ST-T) in one lead, or one component in all leads.
The regions whose replacement lowers the score most, or remove the referral, are where the model's evidence
lies. Do these regions agree with what the SPH labels imply anatomically (inferior MI in II, III and aVF;
anterior MI in V1-V4; BBB in the QRS; ST-T statements in the ST-T), and do they agree more than a
disease-blind map of where the counterpart simply changes the signal most?

Experiment 070 showed that replacement scores mostly follow replaced energy and did not agree with
cardiologist marks. NBR-01 differs in two ways: it explains the project's screening blend rather than a
separate detector, and it validates against label-derived expectations on SPH, with an energy map as the
qualifying baseline.

## Models

Both models come from the PR 74 from-raw bundle
`outputs/pr74_from_raw_validation_20261028/evaluate/model_bundle.joblib`, which HEADS-01 and CLF-01
reproduced exactly:

- Primary: `all_age_blend`, the fixed 0.5/0.5 mean of the frozen HistGradientBoosting on the 130
  measurements of `measurement_extractor042.extract_array` and the L2 logistic regression on the frozen
  xECG's 1,024 pooled features, trained on the all-age 25k cohort (SPH sensitivity 0.773996, specificity
  0.947737, AUROC 0.936770).
- Secondary: `ptb_baseline_blend`, the app default, trained on PTB-XL only (SPH sensitivity 0.6864).

Each replaced variant's measurements (`pr74_screening_features.measurements`, the frozen six-digit
serialization) and xECG features (`external_encoders.xecg_input` and `xecg_features`, released weights,
full batches of 16, the last batch padded with its final input) are computed once and scored by both
models. Thresholds are the frozen SPH 5% calibration-normal thresholds in the bundle, with strict `>`:
`all_age_blend` 0.6569658573983748 and `ptb_baseline_blend` 0.7289262124029245 (the HGB and xECG
components use their own bundle thresholds). A record is flagged when its unreplaced blend score is above
the threshold.

## Normal counterpart (option 1 of the brief: the 062/066/070 machinery)

The files `normative_map062.py`, `combination066.py`, `replacement070.py` and their imports
(`morphology_boundaries.py`, `fragment_localization.py`, `mil065.py`, `locbench.py`, `locbench_data.py`,
`locbench_cache.py`, `ecgad.py`) and `docs/locbench-partition-v1.json` were brought unchanged from
`8902c13` with `git checkout 8902c13 -- <paths>`. None of them existed on the starting branch, so no pinned
file changed.

062's saved model (`outputs/experiment062_normative_v1/normative.pkl`) no longer exists locally. The
rank-0 population model was therefore refitted with 062's own functions, exactly as
`normative_map062.fit_normative` computes `models["rank0"]`: `record_features` of the 2,666 LocBench `fit`
clean normals (2,566 patients, all usable) from the existing LocBench cache, the 062 `design` (heart rate,
age, sex) and `fit_population(..., 0)`. No SPH record is involved.

Same-operator check, already run (stage `counterpart`, no SPH data): 070's sets A and N were redrawn with
070's own selection function and seeds, and `replacement070.counterpart_difference` with the refitted model
was compared with 070's saved `difference.npy` (`/tmp/ecg070-work/full`) for the first 100 prepared records
of each set. The largest absolute difference was 0.0 mV in both sets (gate: 1e-6 mV). The counterpart is
therefore 070's operator: the population mean beat for the record's heart rate, age and sex, warped to each
complete beat, blended with 30-sample trapezoid ramps, `c = x - clean + T`, so the record keeps its
baseline wander and content above 40 Hz.

SPH covariates: age in years; sex mapped to 062's PTB-XL coding (0 male, 1 female) as `1 - male`.

## Regions and variants

Every sample gets a 066 wave label (`replacement070.composition_labels`). The three components are
P = P + PQ, QRS, and ST-T = ST + T. A component mask is restricted to samples 100-4899 (070's scored range)
where the complete-beat counterpart windows carry full weight (summed trapezoid weight at least 1), so
partial edge beats and window ramps are never replaced. Replacement is 070's: `y = x + w (c - x)`, with the
binary lead-by-sample mask convolved with the normalized 11-sample Hann window (22 ms taper). All beats of
the record are replaced at once.

105 variants per record:

- the original (nothing replaced);
- with the normal counterpart: 36 grid cells (12 leads x 3 components), 12 whole leads (the union of the
  three components in one lead), 3 whole components (one component in all 12 leads) and the full
  replacement (all three components in all leads);
- the same 52 regions with the self-swap counterpart.

Self-swap control: each beat's region is replaced by the same region of the record's next complete beat
(cyclically), taken from 062's warped beats of the same preprocessed signal and mapped back with the same
warp, trapezoid blend and taper (`nbr01_counterpart.beat_timeline`, which equals 070's
`template_timeline` when all beats share one template). It measures splicing artefact plus normal
beat-to-beat variation.

Energy: for every variant, the sum over leads and samples of the squared change `(w (c - x))^2`, in mV^2
summed over 500 Hz samples. The energy map is disease-blind.

A record is excluded, and counted, if 062 finds fewer than two complete beats or a component mask is empty.

## SPH groups (labels only)

SPH is development data. Only `evaluation`-role ECGs of the frozen baseline table
(`outputs/pr74_fresh_baseline_exact_20261028/features/sph_rows.csv`), the 10,434 ECGs the models are scored
on, are eligible. Statement mapping (SPH AHA codes, modifiers ignored), one group per ECG, in this order:

| Group | Rule | Expected region |
| --- | --- | --- |
| inferior | inferior MI (161), no anterior (160, 165) or extensive anterior (166) MI | leads II, III, aVF |
| anterior | anterior (160) or anteroseptal (165) MI, no inferior or extensive anterior MI | leads V1-V4 |
| lateral | none: SPH has no lateral MI statement | I, aVL, V5, V6 |
| lbbb | LBBB (104), no MI | QRS, emphasis I, aVL, V5-V6 |
| rbbb | complete RBBB (106), no LBBB, no MI | QRS, emphasis V1-V2 |
| stt | ST deviation (145), ST deviation with T change (146) or T-wave abnormality (147), and no MI, conduction (101, 102, 104, 105, 106, 108) or hypertrophy (142, 143, 153) statement | ST-T |
| normal | "Normal ECG" (1) only, negative label | none |

Extensive anterior MI (166) spans the anterior and lateral territories, so it is not single-territory and is
excluded, as are ECGs with both inferior and anterior MI. SPH's ST-T statements carry no lead, so ischaemia
cannot be assigned a territory; only MI statements are territorial.

Eligible counts from labels only (ECGs / patients): inferior 49 / 49, anterior 60 / 55, lateral 0 / 0,
lbbb 36 / 34, rbbb 342 / 323, stt 2,264 / 2,207, normal 6,850 / 6,682.

Decision on the primary endpoint, from these counts alone: the two territories SPH can label both have at
least 40 patients (49 and 55), so territory discrimination stays primary, computed over inferior and
anterior. Lateral cannot be tested because SPH's vocabulary has no lateral MI statement; this is a property
of the label set, not a small count. The wave contrast is the first secondary endpoint. If after exclusions
either territory has fewer than 40 analysed patients, the wave contrast becomes primary and the territory
contrast secondary; this is checked before any score is read.

Sample: one ECG per patient across the whole sample. Groups are drawn in the table's order with one
generator, `numpy.random.default_rng(20261028)`: in each group the sorted patients not already drawn are
permuted, the first ones kept, and one of each kept patient's group ECGs is drawn with the same generator
(070's `one_per_patient` rule). Cap: 300 patients per abnormal group, and 300 normal patients. Expected
sample: inferior 49, anterior 55, lbbb 34, rbbb 300, stt 300, normal 300 (1,038 ECGs).

## Profile (already run, no score)

Stage `profile` on 20 SPH ECGs (3 per abnormal group present and 5 normals, drawn with seed 20261029), with
no model scoring:

- preparation (counterparts, masks, 105 variants and their 130 measurements): 3.63 CPU-seconds per record;
  12 single-thread workers prepared the 20 in 6.9 s;
- xECG encoding of the 105 variants on GPU 0: 3.51 s per record, 0.62 GB peak, 10.6 GB free at start;
- the original variant's measurements equal the saved SPH measurements exactly (largest difference 0.0),
  and its xECG features are bit-identical to the saved SPH features for 20 of 20 records.

Forecast for 1,038 ECGs: preparation about 5 minutes with 12 workers; encoding about 61 minutes on one GPU,
or about 31 minutes split across both GPUs, each shard under its own device lock and only if at least 4 GiB
are free; scoring and analysis a few minutes. This fits well inside the 2.5-hour budget, so the cap stays
at 300.

## Primary endpoint: territory discrimination (`all_age_blend`)

For each ECG, the score drop of whole-lead replacement l is `s(original) - s(lead l replaced)`. Lead
attribution is the positive part of the 12 drops normalized to sum to 1; an ECG with no positive drop gets
uniform shares (1/12), so no ECG is dropped on its result.

For each territory group T present (inferior, anterior), C_T = mean share in T's expected leads among ECGs
of T minus the mean share in the same leads among ECGs of the other territory group. The primary statistic
is the mean of C_T over the territory groups. Interval: 95% percentile interval of 2,000 whole-patient
bootstrap draws (`intervals.patient_groups` and `patient_resample`), `numpy.random.default_rng(20261028)`,
resampling all analysed patients jointly; every statistic in this protocol is computed on the same draws,
so all differences are paired. A draw in which a statistic is undefined (a group absent) is skipped for that
statistic and counted.

Decision: territory localization is shown if the lower bound is above 0. Otherwise it is not shown.

Key qualifier: the same contrast computed with the energy map (lead shares of the whole-lead replacement
energies), and the paired difference model minus energy. Only if the lower bound of that difference is
above 0 can the report say that the model localizes beyond where the counterpart differs most from the
record (070's lesson). If the primary passes and the qualifier does not, the report says the model's lead
evidence agrees with the territory but is not shown to go beyond replaced energy.

## Secondary endpoints (intervals, no decision)

All on the same bootstrap draws, for the blend and separately for the HGB and xECG component scores (each
component's drops and its own threshold), and for both models:

1. Wave contrast: component shares from the three whole-component replacements. QRS share among BBB ECGs
   (lbbb and rbbb) minus among the other abnormal groups (inferior, anterior, stt); ST-T share among stt
   ECGs minus among the other abnormal groups (inferior, anterior, lbbb, rbbb); and their mean. Each with
   the energy map and the paired model-minus-energy difference.
2. BBB lead contrast: RBBB share in V1-V2 minus LBBB share there, and LBBB share in I, aVL, V5, V6 minus
   RBBB share there, averaged; with energy and the paired difference.
3. Top-1 and top-3 territory hit rates: share of territory ECGs whose largest (or one of three largest)
   positive whole-lead drop is in the expected leads; ties keep lead order. Chance for top-1 is 3/12
   (inferior) and 4/12 (anterior); for top-3, 0.618 and 0.745. Energy-map hit rates and the paired
   difference.
4. Decision-flip rates: among flagged ECGs of each group, of all abnormal groups and of normals, the share
   for which at least one region's replacement leaves the score at or below the threshold, separately for
   grid cells, whole leads, whole components and the full replacement, and for self-swap grid cells, leads
   and full.
5. Full-replacement sanity: mean blend score after full replacement per group, its drop, and the share of
   flagged ECGs that it unflags; compared with the normals' original scores. Operator check (as 070's check
   4): the mean full-replacement drop over the abnormal groups must have a 95% lower bound above 0;
   otherwise the counterpart is not normal for this model and every result is flagged.
6. Self-swap artefact: mean absolute score change over the 36 self-swap grid cells and the 12 self-swap
   leads, against the same for the normal counterpart; the self-swap full drop; and the territory contrast
   computed from self-swap lead drops (a splicing-artefact control of the primary).
7. Normals: original, full-replacement and absolute grid and lead changes, and flip rates of flagged
   normals.
8. Counts of records and flagged records per group.

## Integrity checks

1. 070 operator reproduction (done): 0.0 mV on 100 + 100 records.
2. The unreplaced variant of every analysed ECG, computed from the raw signal through this pipeline, must
   reproduce the saved `all_age` and `ptb_baseline` HGB, xECG and blend SPH scores
   (`outputs/pr74_from_raw_validation_20261028/evaluate/sph_predictions.npz`) with a largest absolute
   difference of at most 1e-12 and identical referral decisions. Justification: the profile showed
   identical measurements and bit-identical xECG features, so any remaining difference can only come from
   summation order when the frozen heads score a matrix of a different size (HEADS-01 used the same 1e-12
   gate for refitted heads). Failure stops the study.
3. The bundle thresholds must equal the thresholds recomputed with `pr74_screening_metrics.budget_threshold`
   from the saved SPH calibration-normal scores.
4. The original variant equals the raw signal bit for bit (unit test), and the encoding stage rebuilds every
   record's 105 variants and requires their SHA-256 to equal the preparation stage's.
5. Raw SPH bytes and canonical windows are verified by `pr74_fresh_baseline.read_signal` on every read.

## Page export

`outputs/nbr01_normal_beat_replacement_20261028/examples.json`, schema as in the brief, plus
`hgb_original` and `xecg_original` in each model's scores so the page can compute component drops. 16
examples drawn without regard to any score: within each group, sorted ECG IDs, `default_rng(20261028)`
choice without replacement, quotas 3 inferior, 3 anterior, 2 lateral, 2 rbbb, 2 lbbb, 2 stt and 2 normal;
because SPH has no lateral ECG, its two slots go to inferior and anterior (4 each). Signals are averaged in
sample pairs to 250 Hz and rounded to 0.001 mV; component ranges are half-open 250 Hz sample ranges of the
replaced masks.

## Data rules

SPH never fits, scales or selects anything: the counterpart model comes from PTB-XL LocBench `fit` clean
normals, the screening models and thresholds are frozen, and SPH labels only define groups and expected
regions. No Challenge test group, EchoNext test set, PTB-XL fold 10 or final frozen test is read. No label
definition changes and no age subgroup is analysed.

## Outputs

Small results in `outputs/nbr01_normal_beat_replacement_20261028/` (stage receipts, `select/selection.csv`,
`score/scores.npz`, `score/integrity.json`, `analyze/results.json`, `examples.json`). Plans and features
(about 1 GB) in `/tmp/nbr01-work` on the local NVMe. Code: `ecg_experiment/localization/nbr01_*.py`,
`scripts/experiments/run_nbr01.py`, `tests/localization/test_nbr01_*.py`. Report:
`docs/nbr01-normal-beat-replacement-results-2026-10-28.md`.

## Limitations stated in advance

- Replacement attribution is model-specific and counterfactual; it says where the score responds, not where
  disease is.
- The population template is more typical than most real normals (070), so full replacement may also lower
  normal scores; the energy map and the normals measure this.
- SPH MI statements are mostly old MI and have no lateral territory; ST-T statements carry no lead.
- SPH has been read by many experiments; results are development evidence.

## Amendment 1 (28 October 2026, after integrity check 2 failed, before any replacement score was read)

Integrity check 2 failed as registered. The `score` stage scored all 1,037 prepared ECGs and compared only
their unreplaced variants with the saved SPH scores: HGB was exact for both models, but the xECG scores
differed by up to 4.12e-8 (`all_age`) and 2.46e-8 (`ptb_baseline`), and the blends by up to 2.06e-8 and
1.23e-8, above the 1e-12 gate. Referral decisions were identical for both blends, and the thresholds replayed
exactly. The stage raised and wrote no completion receipt. No endpoint, drop, share or replaced-variant score
has been read.

Diagnosis, from the unreplaced variants only: 1,035 of the 1,037 unreplaced xECG feature vectors are
bit-identical to the saved SPH features; 2 differ by at most 7.15e-6. Scoring the saved features of the same
1,037 records reproduces the saved xECG scores to 6.1e-16, so the frozen heads are not the cause, and the
records whose features are identical reproduce to 6.1e-16. The 2 gaps come from the encoder: in this study
each record's variants share a batch of 16, while the saved features were encoded in consecutive batches of
16 SPH rows. The batch shape is the same, so this is a batch-content effect of the FP32 kernels, a smaller
relative of the batch-shape effect documented in `pr74-batch-shape-reproduction-2026-10-28.md`.

Additive successor check, in the spirit of that precedent; no tolerance, threshold, endpoint or decision rule
changes. Stage `parity` re-encodes every record whose unreplaced features are not bit-identical inside its
historical batch (rows `16 * (p // 16)` to `16 * (p // 16) + 15` of the frozen SPH table, read and verified
with `read_signal`) and requires bit-identical features and xECG scores within 1e-12 of the saved scores for
both models. The `score` stage then records the registered check as failed and passes only if (a) referral
decisions are identical, (b) every record whose unreplaced score differs by more than 1e-12 is one that the
parity stage reproduced exactly in its historical batch. If either fails, the study stops.

The analysis keeps the scores as computed: all 105 variants of a record were encoded the same way, in
batches drawn from that record's own variants, so drops compare like with like. The report states that the
registered check failed and gives the size of the batch-content effect (at most 2.06e-8 on the primary blend).
