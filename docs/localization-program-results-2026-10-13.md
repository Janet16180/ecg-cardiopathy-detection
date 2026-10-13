# Localization programme 061-068: results

13 October 2026. Summary of the programme planned in [localization-program-2026-10-12.md](localization-program-2026-10-12.md).
Every number comes from the experiments' `result.json` files and results reports linked below. A2 is the within-record
point AUROC of a map against the ECGAD cardiologist's lead-by-sample marks (marked against unmarked samples of the same
record); intervals are paired 95% whole-patient bootstrap intervals (2,000 draws, seed 61061).

## Holdout result

The ECGAD holdout half (201 records, 96 marked, 92 patients) was scored once, by one run, after the protocol and the
map hashes were committed ([067](experiment-067-holdout-results.md)).

| Map | What it is | Holdout A2 | Minus `qrs_prior` | Development A2 |
| --- | --- | ---: | --- | ---: |
| `units_v2` (068) | Learned lead x wave combiner | 0.764 | +0.037 [+0.012, +0.065] | 0.767 |
| `units_learned` (066) | Same combiner, fewer training records | 0.759 | +0.032 [+0.007, +0.060] | 0.760 |
| `textbook` (066) | Predicted diagnosis, painted on its textbook leads and waves | 0.755 | +0.028 [+0.000, +0.057] | 0.749 |
| `qrs_prior` (061) | QRS indicator, disease-blind | 0.727 | bar | 0.742 |
| `factor` (066) | Diagnosis x MIL lead evidence x wave prior | 0.724 | -0.002 [-0.031, +0.026] | 0.725 |
| `qrs_lexicographic` (062) | Normative map ranked inside the QRS prior | 0.709 | -0.018 [-0.037, +0.001] | 0.753 |
| `residual056` | Within-record beat residual (earlier work) | 0.689 | -0.038 [-0.071, -0.001] | 0.680 |
| `U_B` (042) | Earlier lead/wave map | 0.649 | -0.078 [-0.108, -0.047] | 0.632 |

- The primary comparison, `units_v2` minus `qrs_prior`, is confirmed. The top two maps moved by less than 0.004 between
  halves; part of the margin is the bar falling by 0.015 on the holdout, whose marks are sparser (14% of samples against
  19%).
- The combiner's gain over textbook painting is not separable on the holdout: +0.009 [-0.005, +0.022].
- Which lead: `units_v2` lead AUROC 0.616 against 0.441 for the bar. Inside the QRS (A2_qrs): +0.067 [+0.023, +0.116].
  Outside the QRS: +0.047 [+0.007, +0.091]. Time within the record: -0.009 [-0.041, +0.024].
- The QRS prior is still best for LAFB, CRBBB and CLBBB on both halves: the marks for conduction findings cover the
  QRS in more leads than the textbook leads.

## What each experiment found

| Experiment | Approach | Development A2 | Finding |
| --- | --- | ---: | --- |
| [061](experiment-061-locbench-results.md) | LocBench v1 benchmark and baselines | 0.742 (`qrs_prior`) | The cardiologist marks concentrate on QRS (36% of marks on 16% of samples), so a disease-blind QRS indicator beats every earlier map. |
| [062](experiment-062-normative-map-results.md) | Normative phase-aligned map against clean normals | 0.694 | Good lead and detection signal; puts evidence on ST, T and PR, not QRS. |
| [063](experiment-063-masked-reconstruction-results.md) | Masked cross-lead and cross-beat reconstruction | 0.602 | Dividing by predicted spread removes the QRS emphasis and draws the map to noise. |
| [064](experiment-064-dino-patch-knn-results.md) | SimDINOv2 beat-patch tokens, phase-matched kNN | 0.670 | Learned features add nothing over raw 40 ms patches; strong detection (D1 0.887), no lead information. |
| [065](experiment-065-finding-mil-results.md) | Finding-specific multiple-instance heads | 0.662 | Right leads (lead AUROC 0.644), best detection (D1 0.962), wrong time: heads light whole leads. |
| [066](experiment-066-combination-results.md) | Factorized combination, lead-quality filter, learned combiner | 0.725 / 0.760 | Textbook painting beats MIL lead weights; the noise filter barely acts; a small learned combiner is best. |
| [068](experiment-068-combiner-v2-results.md) | Combiner trained on 3,517 records | 0.767 | Twelve times more records add +0.0015: the weak-label objective is saturated. |
| [067](experiment-067-holdout-results.md) | One-time holdout confirmation | - | `units_v2` confirmed against the bar. |

## Why the earlier attempts fell short

- Label-free anomaly maps (062, 063, 064, and 041-060 before them) measure how unusual a waveform is, not whether it is
  clinically abnormal. Their calibration inflates flat, low-variance segments (P, PR, ST, TP) and noisy limb leads,
  while the cardiologist marks sit on QRS complexes. Noise attraction (C1_std 0.58-0.76) is the visible symptom.
- Population references detect abnormal records well but do not say where; within-record references see focal beats
  but not changes shared by every beat. The map that works combines a diagnosis with where that diagnosis lives.
- Statement labels say which finding is present, not which wave carries it, so supervised heads learn the right leads
  but not the right time (065).
- Self-supervised DINO tokens did not beat raw waveform patches for localization in this setting (064); their value is
  detection, not localization.

## Meaningful abnormality against unusual waveform

- The noise control (C) asks whether a map lights leads that a PTB-XL note calls noisy. The confirmed maps are only
  mildly drawn to noise on development (C1_std 0.537-0.540 against 0.560 for the QRS prior and 0.71-0.76 for the
  label-free maps).
- The region test (B) asks whether a map lights the leads and waves a statement implies; it is not independent for the
  combiners, which were trained on the same region table.
- The holdout says the confirmed maps localize mostly by diagnosing and painting the expected region: textbook painting
  from the true statements reaches 0.767 on the holdout, against 0.764 for `units_v2`. Record-specific localization beyond the
  diagnosis is not demonstrated with 96 marked records.

## Limits and next steps

- The ECGAD holdout is spent; any new localization method needs a new marked test set (backlog
  `loc_new_marked_test_set`).
- ECGAD marks are coarse (about 7 leads and 14-19% of samples per marked record), which caps what A2 can reward.
- Questions for the cardiologist: why ASMI marks spread outside V1-V4, and why conduction-finding marks cover the QRS
  in all leads (backlog `loc_asmi_mark_review`, `loc_conduction_component_weight`).
- Ranked follow-ups are in `docs/experiment-priorities.md`.
