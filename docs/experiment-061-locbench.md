# Experiment 061: LocBench v1, a localization benchmark with real truth, and baselines

Frozen 12 October 2026 before any score. Part of the [localization programme](localization-program-2026-10-12.md).
The user authorized the ECGAD carve-out and split ([ecgad-benchmark-v1.md](ecgad-benchmark-v1.md)) and the
PTB-XL+ download (CC BY 4.0, `data/raw/ptb-xl-plus/1.0.1/`, features, labels and ECGdeli global fiducials only).

## Question

How well do the existing localization maps, and maps that know nothing about disease, localize abnormalities
against cardiologist marks, against clinically expected regions, and how strongly are they drawn to noise?
The answer sets the bar for Experiments 062-066.

## Map contract

A method returns, for a list of PTB-XL `ecg_id`s, a finite float32 map of shape (n, 12, 5000) aligned to the
500 Hz PTB-XL record in canonical lead order (I, II, III, aVR, aVL, aVF, V1-V6); higher means more abnormal.
Inputs a method may use: the raw waveform, and age and sex. It may fit on the `fit` partition of
`docs/locbench-partition-v1.json` (label-free methods: its 2,666 `clean_normal` records only), on unlabeled
cohorts v4 minus the `evaluation` partition, and on public delineators. It may never use an ECGAD record or an
`evaluation` record for fitting. Only samples 100-4899 are scored (as in the ECGAD code), to avoid filter
edges.

## Evaluation sets

1. **ECGAD development**: the 198 `development` records of `docs/ecgad-split-v1.json` (94 marked).
2. **Region set**: `evaluation`-partition records with exactly one `REGION_TABLE` statement at likelihood at
   least 50 (`ecg_experiment/locbench_data.py`), and no rhythm statement outside SR, SARRH, SBRAD and STACH.
   Statements are grouped into families: inferior MI (IMI), anterior MI (ASMI, AMI), lateral MI (ALMI, LMI),
   CLBBB, CRBBB, LAFB, 1AVB, WPW, long QT (LNGQT), ST injury (INJAS, INJAL, INJIN), ischemia (ISCAS, ISCAL,
   ISCIN, ISCLA). Expected region = expected leads x expected components, from the table. A strict variant
   (no other diagnostic statement at any likelihood) is reported as a sensitivity analysis.
3. **Noise set**: `evaluation`-partition records that are clean normals in every respect except noise: NORM at
   100, no other diagnostic or form statement, clean rhythm, no extra beats or pacemaker, and a noise
   annotation naming 1 to 11 leads (`noisy_leads`).
4. **Detection set**: all `evaluation`-partition records that are `clean_normal` (negatives) or carry a
   diagnostic statement of superclass MI, STTC, CD or HYP at likelihood at least 50 (positives).

Component boundaries come from PTB-XL+ ECGdeli global fiducials (`fiducial_points/ecgdeli/*_points_global.atr`,
500 Hz). Per beat: P = [P onset, P offset]; PR = [P onset, QRS onset]; QRS = [QRS onset, QRS offset];
ST = [QRS offset, T onset]; T = [T onset, T offset]. An onset is paired with the next matching offset of the
same beat within 300 ms (P), 250 ms (QRS), 600 ms (T); PR pairs a P onset with the next QRS onset within
400 ms; ST pairs a QRS offset with the next T onset within 400 ms. Unpaired marks are dropped. A region-set
record whose expected component has no interval in 100-4899 is excluded and counted.

## Metrics

All AUROCs are Mann-Whitney AUROCs with ties counted as half. "Lead score" is the 95th percentile of a lead's
map over samples 100-4899; "time score" is the maximum over leads per sample; "record score" is the 99th
percentile over all leads and samples.

ECGAD development:

- **A2, primary**: within-record point AUROC (marked against unmarked lead-samples), mean over the 94 marked
  records.
- A1: pooled point AUROC over all 198 records (the published metric; mixes detection and localization).
- A3: lead AUROC (lead score against lead marked), mean over marked records with 1-11 marked leads.
- A4: time AUROC (time score against any lead marked), mean over marked records.
- A5: record AUROC, marked against unmarked records.
- A6: precision at top 5%: share of a marked record's top 5% lead-samples that are marked, mean, beside the
  mean marked share (its chance level).

Region set, per family with at least 20 records and as an unweighted macro mean over those families:

- B1: within-record region AUROC (expected leads x expected components against all other lead-samples).
- B2: lead AUROC (lead score against expected lead), for families whose lead set is not all 12 leads.
- B3: component AUROC (time score against the expected components in any lead).

Noise set: C1 noisy-lead AUROC (lead score against named noisy lead). A cardiac map should stay near 0.5;
higher means it is drawn to noise. C2: record AUROC of noise-set records against `evaluation` clean normals.

Detection set: D1 record AUROC.

Intervals: 2,000 whole-patient bootstrap draws (`ecg_experiment/intervals.py` convention), seed 61061, for each
metric and for paired differences between methods on the same draws.

## Baselines

1. `random`: iid uniform values, seed 61061.
2. `amplitude`: absolute value of the signal after subtracting a 0.6 s running median per lead.
3. `qrs_prior`: 1 inside QRS intervals of the 059 hybrid delineation (`hybrid_boundaries059_v2.joint_hybrid`
   on frozen waveform-only R peaks), 0 elsewhere, plus 1e-3 times `amplitude` to break ties. It uses no ECGdeli
   output, so it is not circular with the region set.
4. `U_B`: Experiment 042's lead/wave Ledoit-Wolf map, refitted on the 2,666 `fit` clean normals with the frozen
   `lead_wave_maps` functions, each piece's score broadcast to its samples (maximum where pieces overlap).
5. `residual056`: Experiment 056's uncalibrated aligned residual energy (frozen `raw_residual054.aligned_field`
   and 056's 140 ms window averaging), broadcast to samples.

Uncovered samples (outside complete beats) get the map's minimum value.

## Integrity checks

- Reproduce 042: recompute `U_B` with its original 5,872-normal reference on 20 fold-9 records and match the
  saved `outputs/experiment042_lead_wave_maps_v1/unit_scores.npz` units exactly (relative 1e-10).
- Reproduce 056: rescore three INCART records and match the saved beat scores exactly (relative 1e-10).
- `random` A2 must lie in [0.48, 0.52]; if not, the metric code is wrong and nothing is reported.
- The ECGAD labels are read from the local copy whose hash is in `ecgad-benchmark-v1.md`; the partition and
  split CSV hashes are recorded in the receipt.
- ECGAD sanity reading (descriptive): for marked IMI and ASMI records, the share of marks in inferior and
  anterior leads.

## Decision rule

The benchmark is adopted if every integrity check passes. The bar for Experiments 062-066 is the baseline with
the highest A2 point estimate; its A2, B1 macro, C1 and D1 are frozen in this experiment's results report.
No method choice, threshold or set definition changes after scores.

## Outputs

`outputs/experiment061_locbench_v1/`: `result.json` (inputs, hashes, every metric with intervals, counts and
exclusions), per-record metric tables, maps for the ECGAD development records (float16), and `run.log`.

## Amendment 1, 12 October 2026, before any score

Implementation clarifications written while building `ecg_experiment/locbench.py`, before any map was scored.
Only set counts from metadata had been read. No definition above changes; the points below fix details the
protocol leaves open.

Component boundaries:

- An interval is half-open, [start mark, end mark), as in the Experiment 059 evaluator. "Same beat" means that
  no other start mark of the same kind lies between the start mark and its end mark (P onset for P and PR, QRS
  onset for QRS, QRS offset for ST, T onset for T). The end mark must come strictly after the start mark, at most
  the stated window later (P 150, QRS 125, T 300, PR 200, ST 200 samples).
- "An interval in 100-4899" means an interval that overlaps samples 100-4899. The four records without an
  ECGdeli file (7120, 13196 and 18247 in `evaluation`, 21602 in `fit`) are excluded from the region set and
  counted.

Sets:

- Region set: a rhythm statement is an SCP statement with `rhythm` = 1, at any likelihood. A WPW record needs
  both PR and QRS intervals. The strict variant keeps region-set records with no other `diagnostic` = 1
  statement at any likelihood; it reports the families with at least 20 strict records and their macro mean.
- Noise set: "clean rhythm" is `CLEAN_RHYTHMS` (SR, SARRH, SBRAD), as for clean normals. The named leads are the
  union over the four noise columns (`locbench_data.noisy_leads`), and that union must hold 1-11 leads.
- ECGAD: a record is marked if any sample in 100-4899 is marked (94 records, as in the split). A2 skips the
  marked records whose every scored lead-sample is marked and A4 those whose every scored sample is marked in
  some lead; both are counted.

Metrics:

- Percentiles are `numpy.percentile` with linear interpolation. Every AUROC is the Mann-Whitney statistic with
  ties counted as half. A6 uses the top k = 2,880 lead-samples (5% of 12 x 4,800); values tied with the k-th
  largest are counted by expectation, as if ties were broken at random.
- Intervals: each metric is bootstrapped over the patients of the records it averages or ranks, with the
  `intervals.py` draw convention (patients in `np.unique` order, `rng.integers(0, P, P)` on a fresh
  `default_rng(61061)`, 2,000 draws, 2.5 and 97.5 linear percentiles). Record-level AUROCs (A5, C2, D1) skip
  single-class draws and count them. B macro draws over the patients of the eligible families' records; a draw
  that misses a family is skipped and counted. A paired difference uses the same draws for both methods.
- A1 point estimate is the exact Mann-Whitney AUROC over all points. Its bootstrap ranks all points once,
  quantizes the average ranks into 65,536 equal-count bins (tied values share a bin), keeps per-record
  positive and negative counts per bin, and computes each draw from the weighted sum of the record histograms,
  counting pairs within a bin as ties. The binned full-set value must match the exact one to 1e-4.

Maps and baselines:

- The signal is the PTB-XL physical record in float64, recovered exactly from the float32 cache as
  round(1000 x) / 1000 (PTB-XL gain 1,000, baseline 0; equality with `wfdb.rdsamp` is checked in the run).
- Maps are evaluated in float32. The ECGAD development maps are saved in float32 so that A1 paired differences
  are exact; region and noise maps are saved in float16 for display; detection records keep lead and record
  scores only. Uncovered samples get the minimum of that record's covered values.
- `random`: `default_rng([61061, ecg_id])` per record, so a record's map does not depend on the set.
- `amplitude`: `scipy.ndimage.median_filter` with a centred 301-sample (0.6 s) window per lead, edge mode
  `nearest`.
- `qrs_prior`: the half-open QRS intervals `hybrid[:, 1]` of `joint_hybrid(signal, fs=500)`, the same on all
  12 leads; invalid (-1) intervals are skipped.
- `U_B`: references from every kept beat of the `fit` clean normals with at least two kept beats (042 rule);
  each piece's score fills [round(500 start), round(500 end)) on its lead, maximum where pieces overlap.
- `residual056`: `incart_localization056.score_signal` on the 10-second record with the refitted `U_B`
  references (they only set the piece-coverage mask, not the residual), its `aligned_residual` 140 ms windows
  broadcast to their samples with the maximum where windows overlap.
- A record for which a baseline cannot be built (fewer than two kept beats for `U_B`, fewer than three complete
  beats for `residual056`) gets a constant zero map and is counted.

Integrity: the 042 check uses the first 20 records of the saved `unit_scores.npz` order (all fold 9) and its
5,872-record fit set rebuilt with the 042 selection rule; the 056 check rescores INCART records I01, I02 and
I03 completely with 056's saved `normal_reference.pkl` (hash checked against its receipt) and compares both
methods' beat scores with `detected_beats.csv`.
