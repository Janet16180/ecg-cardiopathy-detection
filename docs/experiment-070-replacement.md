# Experiment 070: replacement faithfulness test of the localization maps

Frozen 13 October 2026 before any replacement score. Part 2 of the localization programme
([plan](localization-program-2-2026-10-13.md)). The user authorized a day of unsupervised experiments; the
choices below were made by the agent and are fixed here.

## Question

If the region a localization map highlights is replaced by its normal counterpart, does an abnormal-ECG
detector's output fall more than when a random region of the same size, the same leads count and the same wave
composition is replaced? The test needs no marks, so it can rank every map built so far (061-068). It is then
compared with the ECGAD development A2 ranking to see whether a label-free test orders maps the way the
cardiologist marks do.

## Records

- Set A (primary): `evaluation`-partition records of the LocBench detection set with `y = 1` (a diagnostic
  statement of superclass MI, STTC, CD or HYP at likelihood at least 50; 2,974 records, 2,537 patients). One
  record per patient: the sorted unique patients are permuted with `numpy.random.default_rng(70070)`, the first
  1,500 patients are kept, and one of each patient's eligible records is drawn with the same generator.
- Set N (control): `evaluation` clean normals (1,400 records, 1,343 patients), 400 patients drawn the same way
  with seed 70071.
- Set E (exploratory): the 198 ECGAD development records. The ECGAD holdout half is not read, mapped or scored.

A record is excluded, and counted, if the normal counterpart cannot be built (062 `record_features` returns
None) or the 066 delineation finds no beat.

## Maps (14)

| Map | Source | How it is obtained |
| --- | --- | --- |
| `random`, `amplitude`, `qrs_prior`, `U_B`, `residual056` | 061 | rebuilt with `holdout067.record_maps` |
| `qrs_lexicographic` | 062 | rebuilt with `holdout067.record_maps` |
| `normative_v1` | 062 | saved maps (`maps_normative_v1.npy`, float16) |
| `raw` (reconstruction error) | 063 | saved maps (`maps_v1/evaluation_raw.npy`, float16) |
| `dino_dist` | 064 | saved maps (`maps_dino_dist.npy`, float16) |
| `mil_lse` | 065 | saved maps (`maps_mil_lse_evaluation.npy`, float16) |
| `factor`, `textbook`, `units_learned` | 066 | rebuilt with `holdout067.record_maps` |
| `units_v2` | 068 | rebuilt with `holdout067.record_maps` |

Set E uses the scored float32 development maps saved in each experiment's LocBench `tables` folder
(`ecgad_maps.npy`) for all 14 maps. Rebuild checks: the rebuilt development maps must equal the saved scored
development maps (largest absolute difference at most 1e-5, as in 067), and rebuilt evaluation maps are compared
with 066's and 068's saved float32 maps (`/tmp/ecg066-work/full`, `/tmp/ecg068-work/full`) and 061's float16
maps where those files still exist.

## Detectors

(a) Primary: an independent detector that no map uses. A 1D ResNet (four residual stages of widths 32, 64, 128
and 128, kernel 9, strides 1, 2, 2, 2 after a stride-2 stem, global average pooling over time, one logit) on the
raw signal in mV, averaged in pairs to 250 Hz and clipped to +-10 mV. Training data: `fit`-partition records
only, positive when the LocBench detection rule calls them positive (6,091 records) and negative when they are
clean normals (2,666 records); the other 2,860 `fit` records are not used. This is the project's binary label
(MI, STTC, CD or HYP against sinus-only normal) with LocBench's likelihood and clean-normal rules. 15% of `fit`
patients (seed 70070) are held out for early stopping. Training uses random 9-second crops (a random time shift of
up to 1 s), AdamW (learning rate 1e-3, weight decay 1e-2), batch 64, binary cross-entropy, at most 40 epochs,
patience 6 on the held-out AUROC computed on full 10-second records, seed 70070, float32. The detector is adopted
if its held-out AUROC is at least 0.85. Its AUROC on the whole `evaluation` detection set (4,374 records) is
reported as a description; no choice depends on it.

(b) Secondary: the 065 `mil_lse` ensemble (`weights_lse_6506[5-7].pt`). Its abnormal logit is minus the mean
NORM bag logit of the three networks, the logit of 1 - p(NORM). It is run in float32. 065's zero-phase 0.5-40 Hz
band-pass is applied as an exact linear operator (`scipy.signal.sosfiltfilt` applied to the 5,000 unit impulses,
then multiplied on the GPU), followed by 065's +-10 mV clip. Under (b), `mil_lse` is fully circular (it is the
detector's own evidence), and `factor`, `textbook`, `units_learned` and `units_v2` are partly circular (they use
065 bag probabilities or evidence).

The score of a record under a detector is its logit; a drop is the original logit minus the logit after
replacement.

## Normal counterpart

For each record, built from the frozen 062 machinery and the 062 normative model fitted on the 2,666 `fit` clean
normals (`outputs/experiment062_normative_v1/normative.pkl`):

1. `features = normative_map062.record_features(x)` gives each complete beat's aligned warp knots (QRS onset - 300
   ms, QRS onset, QRS offset, T onset, T end, T end + 100 ms).
2. The template is the population mean beat for this record's heart rate, age and sex:
   `design(HR, age, sex) @ models["rank0"].coefficients`, shape (12, 430), in mV, PR-segment baseline at zero,
   40 Hz low-pass. The bounded normal subspace is not used, so no abnormal shape is absorbed.
3. Each beat's template is mapped back to the timeline through the inverse of that beat's piecewise-linear warp.
   Each beat window carries a trapezoid weight with 30-sample ramps at both ends; where windows overlap the
   templates are averaged, and where the summed weight is below 1 the template fades to zero (the baseline).
4. `clean = normative_map062.preprocess(x, delineate(x)["qon"])` is the record's 40 Hz low-passed waveform minus
   its PR-segment baseline interpolated between beats.
5. The counterpart is `c = x - clean + T`: the record keeps its own baseline wander and its content above 40 Hz,
   and its cardiac waveform below 40 Hz becomes the warped normal template (the isoelectric baseline where no
   beat window covers a sample). It is level-matched to the record at every beat.

Replacement of a region with binary lead-sample mask M: `y = x + w * (c - x)`, where `w` is M convolved along
time with a normalized 11-sample Hann window (a 22 ms cosine taper). `w` lies in [0, 1], so no step is created,
and away from the scored-range edges it carries the same total weight as M. Only samples 100-4899 are ever
selected.

## Replacement and control

- Composition labels: every sample gets one label from the frozen 066 delineation
  (`combination066.delineated_components`, `component_masks`): P, PQ, QRS, ST, T, or other (outside these). Where
  components overlap, QRS > ST > T > PQ > P.
- Map region at k in {5, 10, 20}%: the map's top n = round(k x 12 x 4,800) lead-samples within samples 100-4899
  (2,880, 5,760, 11,520), ties broken by a per-record uniform random key (`default_rng([70070, ecg_id])`).
- Matched random control, R = 8 draws per record, shared by all maps and all k of that record: a uniform random
  lead permutation, and for each composition label c one uniform random offset over that label's scored samples.
  The control moves the map's selection in lead l to lead pi(l), and moves its selection of the j-th label-c sample
  to the (j + offset)-th label-c sample (cyclically), the same offset for every lead. This keeps exactly the
  number of selected lead-samples per lead and per label, the number of leads, and the contiguity of the selection
  within each label (runs split only where they wrap across a component boundary). A map whose region is a random
  draw from the same distribution (the `random` map) has an expected score of exactly 0.
- Whole-lead variant: replace samples 100-4899 of the map's top 3 leads by lead score (061 rule: 95th percentile
  of the map over samples 100-4899; ties broken by the record's random key), against 20 random 3-lead sets per
  record, shared by all maps.

Per record and map: drop_map(k) = z0 - z(map region replaced); drop_ctrl(k) = mean over the 8 draws of z0 -
z(control replaced); S(k) = drop_map(k) - drop_ctrl(k). AOPC_raw is the mean of drop_map over the three k and
AOPC_matched the mean of S over the three k. W = drop_map(top 3 leads) - mean drop over the 20 random 3-lead sets.

## Readings

- Primary: mean S(10%) over set A under detector (a), per map, with a whole-patient bootstrap 95% interval (2,000
  draws, seed 61061, `intervals.patient_groups` and `patient_resample`, `np.unique` patient order), and the paired
  difference `units_v2` minus `qrs_prior` on the same draws.
- Secondary, each with intervals: S(5%), S(20%), AOPC_raw, AOPC_matched, drop_map and drop_ctrl at 10%, and W, on
  sets A and N under detectors (a) and (b); paired `units_v2` minus `qrs_prior` for each.
- Validation: across the 14 maps, the Spearman correlation between mean S(10%) on set A under detector (a) and
  ECGAD development A2 (mean of the saved per-record `ecgad.csv` tables of each map). Interval: 2,000 joint draws
  that resample set-A patients (`default_rng(61061)`) and, independently, the patients of the 92 A2 records
  (`default_rng(61062)`). Sensitivities: the same without `random`; a bootstrap over maps (14 maps resampled with
  replacement, seed 61061, draws with a constant vector skipped and counted); detector (b).
- Exploratory: the same readings on set E, and its Spearman correlation with A2. Descriptive: each map's
  composition (share of P, PQ, QRS, ST, T and other) in its top 10%; S(10%) per detection superclass; drops after
  replacing the whole scored range (all 12 leads, samples 100-4899).

## Decision rules

- Per map (set A, detector (a), S(10%)): faithful if the interval lies above 0, anti-faithful if below 0,
  otherwise not separable from the matched control.
- Primary comparison: `units_v2` minus `qrs_prior` on S(10%). Above 0: `units_v2` points to regions the detector
  relies on more than `qrs_prior` does, beyond wave composition. Below 0: the reverse. Interval containing 0: no
  difference.
- Validation: Spearman interval above 0 means the label-free test ranks maps like the marks; containing 0 means no
  agreement is shown; below 0 means disagreement. Disagreement would say that the regions a detector relies on
  differ from the regions the cardiologists marked (the test measures faithfulness to a model, not clinical
  correctness), or that the operator or the control is inadequate. It cannot by itself say which maps are
  clinically right.
- Set N: S near 0 is expected. A map whose S on normals is as large as on set A is not reacting to abnormality.

## Integrity checks

1. Detector (a) held-out AUROC at least 0.85.
2. Replacing nothing (an all-zero mask) changes every record's logit by exactly 0 under both detectors.
3. `random`: S(10%) on set A under detector (a) has an interval that contains 0. If not, the control is flagged as
   biased.
4. Operator: replacing the whole scored range lowers the mean logit of set A under detector (a). If not, the
   counterpart is not normal for the detector and the test is flagged.
5. Predecessor reproduction: 065 bag logits recomputed with `holdout067.mil_summaries` equal 065's saved
   `bags_evaluation.npz` and `bags_ecgad_dev.npz` (largest difference at most 1e-3, as in 067); the rebuilt maps
   equal the saved development maps (above); the rebuilt development `qrs_prior` and `units_v2` maps rescored with
   `locbench.ecgad_record_metrics` reproduce the saved per-record A2 within 1e-12; the A2 means of the 14 saved
   tables equal the published values.
6. The GPU-path detector (b) logits of unmodified records are compared with the CPU-path (065 preprocessing, bf16)
   logits; the largest difference is reported.

## Data rules

Nothing is fitted on ECGAD or `evaluation` records: detector (a) uses only `fit` records, and every other fitted
quantity is frozen from 062, 065, 066 and 068. ECGAD development marks are read only through the saved A2 tables
and, for check 5, the development rows of the label file. Challenge test and calibration groups, the EchoNext test
set and PTB-XL folds 9-10 outside ECGAD are not read.

## Outputs

`outputs/experiment070_replacement_v1/`: `result.json` (inputs and hashes, record lists, detector training and
held-out AUROC, every reading with intervals, decisions and integrity), per-record drop tables (CSV), detector (a)
weights and history, and example figures of the operator (PNG). Large intermediates (maps, ranks, counterparts) go
to `/tmp/ecg070-work`. Code: `ecg_experiment/replacement070.py`, `scripts/experiments/run_replacement070.py`,
`tests/test_replacement070.py`.
