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
