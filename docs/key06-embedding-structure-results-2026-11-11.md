# KEY-06 results: structure of the xECG embedding space beyond the 14 conditions

11 November 2026. Executes the frozen [KEY-06 protocol](key06-embedding-structure-protocol-2026-11-11.md)
(commit `951d7b7`). Only the included train (discovery) and validation (replication) records of MLC-01 v4
were read. No feature row, label, waveform or prediction of the test roles (PTB-XL fold 10, both SPH halves)
was parsed, and no Challenge test group, EchoNext or MIMIC record was opened. Nothing here is a clinical
finding: the sources code statements differently, no second reading exists, and clinical review is pending
(validación pendiente). The sheet readings below are an analyst's description of what is measurable in the
plots, not a diagnosis.

Every number comes from a file under `outputs/key06_embedding_structure_20261111/`. Most are printed by
`PYTHONPATH=. .venv/bin/python -m scripts.reports.key06_tables --section <name>` (cited as [section `<name>`]);
the others name their file.

## Result

1. What we already know explains 57% of the embedding's variance on validation (0.572; 0.628 inside the 91
   principal components that keep 90%). The measurements carry most of it: 0.505 alone and 0.273 that
   nothing else explains. The 14 labels explain 0.197 alone but only 0.024 uniquely, heart rate 0.135 and
   0.023, age and sex 0.053 and 0.005, the source 0.019 and 0.003, the signal flags 0.014 and 0.001
   [section `variance`].
2. The source is a small share of the variance but easy to read: a linear classifier names it with accuracy
   0.738 (balanced 0.624, majority 0.461). Removing the linear effects of source, heart rate, age, sex and the
   reliable flags takes away 0.1965 of the variance and leaves the source unreadable (accuracy 0.461, balanced
   0.167, the chance level for six sources) [sections `reverse`, `residual`].
3. 606 clusters came out of 20 solutions; 27 meet all five rules of the protocol. None is unexplained. By
   their measurements, statements and sheets they fall into five kinds [sections `clusters-candidates`,
   `statements-candidates`, `undefined`, `notes`]:
   - normal-looking ECGs and normal variants (9): the dense core of sinus rhythm, and a vertical-axis group with
     tall inferior QRS and prominent T waves where the early repolarization statement is enriched;
   - gaps in the labels (10): flat or inverted T waves in records whose T-wave or ST label is undefined (73-88%
     of members against 19-22% in their subset), and left-axis groups where left axis deviation is positive in
     at most a third of members or undefined;
   - a measurement gradient below a label threshold (2): long PR (pr_ms +0.90 within source) and short PR
     (-1.21);
   - a recording artefact below the flag thresholds (3): baseline drift (noise_drift_frac +0.99 to +1.19);
   - three patterns outside the 14 conditions: a pre-excitation pattern (the WPW statement on 19 of 33 train
     members, only 3 validation members), low limb-lead QRS voltage, and a terminal rightward QRS deflection.
4. The protocol's artefact rule, read literally, rejects 435 of the 606 clusters, and in 248 of them the
   failing flag is on at most four members [section `artefact`]. 85 clusters fail that rule and nothing else.
   They include the clearest findings outside the 14 conditions: ten clusters where the PTB-XL complete LBBB
   statement is on 40-54% of members, nine where the pacemaker statement is on 42-88%, and three where the
   Ningbo atrial rhythm statement is on 32-44%. Fifty of the 85 would pass if a flag needed five members, but
   the frozen rule decides, so none is a candidate.
5. The embedding separates 18 statements outside the 14 conditions that the protocol's rule admits (100
   train positives, two source families). Validation AUROC within source ranges from 0.978 (right ventricular
   hypertrophy) and 0.977 (left bundle branch block) to 0.782 (nonspecific ST-T abnormality) [section
   `statements`].
6. The 1% most isolated train records are enriched for AF, ventricular ectopy, wide-QRS or very fast rhythms
   and recording defects: 24% carry a reliable flag against 8% of the others, and median heart rate is 101
   against 72 [section `outliers`].

In the protocol's terms: question 1 is answered by items 1 and 2. Question 2: after the linear effects are
removed, HDBSCAN finds no group (every record is noise), while k-means partitions stay stable; 16 of the 27
candidates come from the residual spaces and are of the same kinds as the raw-space ones. Question 3 is
answered by items 3 and 4, question 4 by item 5.

## Integrity

[section `integrity`]

- P0's validation selection metric was reproduced exactly from its stored predictions: 0.9642697112976489,
  equal to the MLC-02 selection record. The validation label matrix read here has the selection record's
  SHA-256, so the role-guarded reader returns the same labels as MLC-02.
- The feature rows (`rows.csv`) reproduce the manifest's record keys, sources, roles and first-10-s hashes for
  all 71,388 records, and the five feature files equal their receipt's SHA-256.
- Records: train 56,250 and validation 15,138, per source as in MLC-01.

| source | train | validation |
|---|---|---|
| chapman_shaoxing | 6097 | 2036 |
| cpsc_2018 | 3763 | 1251 |
| cpsc_2018_extra | 1936 | 653 |
| georgia | 6117 | 2043 |
| ningbo | 20919 | 6972 |
| ptbxl | 17418 | 2183 |

The records carry 139 raw statements outside the 14 conditions (codes that are not a positive code of one of
them); 96 of them are on at least 30 train records.

## 1. Variance accounting

[section `variance`]

| group | columns | r2_alone_validation | r2_unique_validation | r2_alone_validation_kept | r2_unique_validation_kept | r2_alone_train |
|---|---|---|---|---|---|---|
| source | 5 | 0.019 | 0.003 | 0.021 | 0.003 | 0.021 |
| labels | 14 | 0.197 | 0.024 | 0.218 | 0.027 | 0.192 |
| heart_rate | 7 | 0.135 | 0.023 | 0.150 | 0.025 | 0.124 |
| measurements | 254 | 0.505 | 0.273 | 0.555 | 0.297 | 0.516 |
| signal_flags | 15 | 0.014 | 0.001 | 0.015 | 0.001 | 0.017 |
| age_sex | 9 | 0.053 | 0.005 | 0.058 | 0.005 | 0.052 |
| all | 304 | 0.572 | 0.572 | 0.628 | 0.628 | 0.580 |

Kept components (90%): 91.

"Alone" is the validation R^2 of one group; "unique" is what the group adds to all the others. The groups
overlap heavily: the labels add 0.024 to the other groups, an eighth of their 0.197 alone, as most labels are
defined by what the measurements measure.

[section `components`] The first principal component (10.5% of the variance) is explained 0.87 by the
measurements, 0.71 by the labels and 0.58 by heart rate. The source explains at most 0.101 of any of the
first ten components (component 7).

[section `reverse`] In the other direction, the embedding predicts within source:

- the 14 labels with stratified AUROC from 0.911 (any MI) to 0.998 (sinus tachycardia);
- heart rate with R^2 0.975 and age with 0.700; sex with AUROC 0.922;
- 117 of the 127 other measurements with R^2 above 0.5 (median 0.839); the weakest are the Q depths in V1-V4
  (0.255-0.382);
- the reliable flags with AUROC from 0.849 (contact loss) to 0.983 (flat lead).

## 2. Residual space

[section `residual`]

Variance removed on validation: 0.1965; left: 0.8035. Source from the residual space: accuracy 0.4606, balanced 0.1667, majority 0.4606.

Norm of each source's validation mean:

| source | raw | residual |
|---|---|---|
| ptbxl | 4.835 | 1.964 |
| ningbo | 3.598 | 0.346 |
| chapman_shaoxing | 4.733 | 0.723 |
| georgia | 6.428 | 0.775 |
| cpsc_2018 | 5.204 | 0.954 |
| cpsc_2018_extra | 7.421 | 1.395 |

PTB-XL keeps the largest mean offset after residualisation (1.96). In the residual space the labels stay
readable (step 7 reference, [section `labels`]): AUROC changes by less than 0.04 for every label except sinus
bradycardia, which falls from 0.996 to 0.840 because the heart rate was removed linearly.

## 3. Clusters

[section `solutions`]

| space | train | validation | kept_components | kept_share |
|---|---|---|---|---|
| all__raw | 56250 | 15138 | 91 | 0.9005 |
| all__residual | 56250 | 15138 | 106 | 0.9002 |
| label_negative__raw | 15763 | 3740 | 83 | 0.9003 |
| label_negative__residual | 15763 | 3740 | 97 | 0.9011 |

| solution | clusters | noise_train | stable | source | replicated | artefact | labelled | stable_and_source | stable_source_replicated | artefact_min5 | candidates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all/raw/kmeans_10 | 10 | 0.000 | 10 | 10 | 10 | 3 | 6 | 10 | 10 | 5 | 1 |
| all/raw/kmeans_20 | 20 | 0.000 | 15 | 18 | 20 | 3 | 9 | 13 | 13 | 10 | 1 |
| all/raw/kmeans_40 | 40 | 0.000 | 22 | 37 | 40 | 6 | 16 | 19 | 19 | 18 | 1 |
| all/raw/kmeans_80 | 80 | 0.000 | 25 | 76 | 80 | 22 | 29 | 22 | 22 | 48 | 0 |
| all/raw/hdbscan | 4 | 0.457 | 4 | 3 | 4 | 2 | 2 | 3 | 3 | 2 | 1 |
| all/residual/kmeans_10 | 10 | 0.000 | 10 | 10 | 9 | 2 | 8 | 10 | 9 | 7 | 2 |
| all/residual/kmeans_20 | 20 | 0.000 | 16 | 20 | 20 | 5 | 14 | 16 | 16 | 15 | 2 |
| all/residual/kmeans_40 | 40 | 0.000 | 22 | 38 | 40 | 10 | 22 | 20 | 20 | 24 | 3 |
| all/residual/kmeans_80 | 80 | 0.000 | 21 | 76 | 76 | 22 | 43 | 18 | 18 | 54 | 1 |
| all/residual/hdbscan | 0 | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| label_negative/raw/kmeans_10 | 10 | 0.000 | 9 | 9 | 10 | 3 | 10 | 8 | 8 | 6 | 2 |
| label_negative/raw/kmeans_20 | 20 | 0.000 | 10 | 16 | 19 | 7 | 20 | 7 | 7 | 13 | 3 |
| label_negative/raw/kmeans_40 | 40 | 0.000 | 12 | 34 | 31 | 14 | 40 | 7 | 7 | 32 | 2 |
| label_negative/raw/kmeans_80 | 80 | 0.000 | 12 | 71 | 65 | 25 | 80 | 6 | 5 | 63 | 0 |
| label_negative/raw/hdbscan | 2 | 0.054 | 2 | 1 | 2 | 0 | 2 | 1 | 1 | 1 | 0 |
| label_negative/residual/kmeans_10 | 10 | 0.000 | 8 | 10 | 10 | 1 | 10 | 8 | 8 | 4 | 1 |
| label_negative/residual/kmeans_20 | 20 | 0.000 | 12 | 18 | 20 | 8 | 20 | 10 | 10 | 16 | 5 |
| label_negative/residual/kmeans_40 | 40 | 0.000 | 9 | 37 | 31 | 10 | 40 | 6 | 5 | 33 | 0 |
| label_negative/residual/kmeans_80 | 80 | 0.000 | 11 | 74 | 67 | 28 | 80 | 8 | 7 | 68 | 2 |
| label_negative/residual/hdbscan | 0 | 1.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| total | 606 |  | 230 | 558 | 554 | 171 | 451 | 192 | 188 | 419 | 27 |

- HDBSCAN (minimum cluster size 100) found 4 clusters in the raw space with 45.7% of train as noise, 2 in the
  label-negative raw space, and none in either residual space: there every train record is noise.
- Of the 606 clusters, 230 are stable, 558 are not one source, 554 replicate, 171 pass the literal artefact
  rule and 451 are not already labelled. 192 are stable and not one source.
- The 192 stable, multi-source clusters fail as follows (`failed_rules` in `describe/clusters.csv`): the
  artefact rule alone 85, artefact and already labelled 58, already labelled alone 18, not replicated (with or
  without artefact) 4, none 27.

The full table of the 192 is printed by [section `clusters-stable`] with their statements in [section
`statements-stable`]; all 606 are in `describe/clusters.csv`.

### How the artefact rule fails

[section `artefact`] Of the 435 clusters failing the literal rule, the failing flag with the most flagged
members has 1 member in 124 clusters, 2-3 in 98, 4 in 26 and 5 or more in 187. In Appendix A most flags that
fail on one to three members are rare ones (`reversal_la_ra`, `reversal_neutral`, `near_flat_lead`,
`edge_hold`, `identity_violation`, `adc_rail`), whose rate outside is so low that one flagged member among
thousands exceeds twice it. The per-cluster flag rates in `describe/flag_rates.csv` reproduce both
artefact columns of `clusters.csv` for all 606 clusters.

Clusters that fail only the literal rule and pass with five flagged members (50), grouped by their first
enriched statement and the share of members carrying it:

| statement | clusters | lowest_member_share | highest_member_share |
|---|---|---|---|
| scp:CLBBB (complete left bundle branch block) | 10 | 0.40 | 0.54 |
| scp:PACE (normal functioning artificial pacemaker) | 9 | 0.42 | 0.88 |
| scp:NORM (normal ECG) | 4 | 0.22 | 0.56 |
| snomed:106068003 (atrial rhythm) | 3 | 0.32 | 0.44 |
| scp:axis:AXL (heart_axis = AXL) | 3 | 0.01 | 0.01 |
| snomed:428417006 (early repolarization) | 3 | 0.02 | 0.04 |
| scp:SR (sinus rhythm) | 2 | 0.41 | 0.49 |
| scp:axis:LAD (heart_axis = LAD) | 2 | 0.16 | 0.20 |
| scp:INJAL (subendocardial injury in anterolateral l) | 2 | 0.02 | 0.02 |
| scp:LVOLT (low QRS voltages in the frontal and hori) | 1 | 0.01 | 0.01 |
| scp:LNGQT (long QT-interval) | 1 | 0.00 | 0.00 |
| scp:HVOLT (high QRS voltage) | 1 | 0.00 | 0.00 |
| scp:axis:MID (heart_axis = MID) | 1 | 0.21 | 0.21 |
| scp:axis:RAD (heart_axis = RAD) | 1 | 0.01 | 0.01 |
| snomed:164909002 (left bundle branch block) | 1 | 0.07 | 0.07 |
| snomed:253352002 (left atrial abnormality) | 1 | 0.01 | 0.01 |
| snomed:251170000 (blocked premature atrial contraction) | 1 | 0.00 | 0.00 |
| snomed:426761007 (supraventricular tachycardia) | 1 | 0.04 | 0.04 |
| snomed:426783006 (sinus rhythm) | 1 | 0.62 | 0.62 |
| snomed:733534002 (complete left bundle branch block) | 1 | 0.01 | 0.01 |
| snomed:74390002 (wolff parkinson white pattern) | 1 | 0.01 | 0.01 |

The LBBB, pacemaker and atrial rhythm groups are findings outside the 14 conditions by the sources' own
statements. The others name statements carried by 0-4% of members, or normal and sinus-rhythm records. The
full list of the 85 is in Appendix A.

## 4. Candidate findings

[section `clusters-candidates`] "Strongest" is the enrichment with the largest absolute log odds ratio among
those significant on train, so it can be a depletion (a negative log odds ratio): the normal core's strongest
enrichment is the absence of left axis deviation or LBBB.

| solution | c | n train | n val | jaccard | families | top label | strongest (train/val log OR) | worst flag | HR | QRS | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all/raw/kmeans_10 | 0 | 9408 | 2226 | 0.92 | ptbxl 0.44, chapman_ningbo 0.29, cpsc 0.15, georgia 0.11 | t_wave_abnormality 0.08 | left_axis_deviation -5.56/-3.93 | near_flat_lead x1.00 (2/9408) | 77 | 82 |  |
| all/raw/kmeans_20 | 0 | 5780 | 1451 | 0.63 | ptbxl 0.37, chapman_ningbo 0.35, cpsc 0.16, georgia 0.12 | t_wave_abnormality 0.09 | left_axis_deviation -5.20/-3.38 | near_flat_lead x1.75 (2/5780) | 77 | 82 |  |
| all/raw/kmeans_40 | 24 | 1240 | 314 | 0.76 | georgia 0.38, ptbxl 0.31, chapman_ningbo 0.17, cpsc 0.14 | t_wave_abnormality 0.46 | scp:NORM -6.42/-4.15 | hf_noise x1.04 (4/1240) | 70 | 88 |  |
| all/raw/hdbscan | 0 | 26711 | 8384 | 0.88 | chapman_ningbo 0.50, ptbxl 0.35, cpsc 0.08, georgia 0.06 | sinus_bradycardia 0.33 | scp:CLBBB -6.27/-4.67 | baseline_wander x0.72 (645/26711) | 64 | 84 |  |
| all/residual/kmeans_10 | 3 | 6126 | 1625 | 0.89 | chapman_ningbo 0.48, ptbxl 0.32, cpsc 0.10, georgia 0.09 | first_degree_av_block 0.22 | scp:CLBBB -4.02/-2.63 | reversal_la_ra x1.64 (1/6126) | 75 | 84 |  |
| all/residual/kmeans_10 | 4 | 5681 | 1606 | 0.87 | chapman_ningbo 0.47, ptbxl 0.33, georgia 0.11, cpsc 0.09 | left_axis_deviation 0.25 | lafb +3.36/+3.27 | near_flat_lead x1.78 (2/5681) | 71 | 90 |  |
| all/residual/kmeans_20 | 5 | 4209 | 1206 | 0.65 | chapman_ningbo 0.49, ptbxl 0.32, georgia 0.10, cpsc 0.09 | sinus_bradycardia 0.23 | scp:CLBBB -4.32/-1.61 | edge_hold x1.37 (1/4209) | 71 | 88 |  |
| all/residual/kmeans_20 | 8 | 3042 | 908 | 0.88 | chapman_ningbo 0.41, ptbxl 0.36, georgia 0.16, cpsc 0.07 | sinus_bradycardia 0.21 | scp:CLBBB -2.69/-2.52 | hf_noise x1.65 (15/3042) | 70 | 84 |  |
| all/residual/kmeans_40 | 1 | 2387 | 638 | 0.67 | chapman_ningbo 0.41, ptbxl 0.36, cpsc 0.12, georgia 0.11 | sinus_bradycardia 0.21 | left_axis_deviation -4.32/-2.74 | missing_samples x1.46 (4/2387) | 71 | 86 |  |
| all/residual/kmeans_40 | 13 | 2083 | 565 | 0.64 | chapman_ningbo 0.59, ptbxl 0.25, georgia 0.08, cpsc 0.08 | sinus_tachycardia 0.29 | scp:ISC_ -3.50/-2.13 | isolated_spike x1.63 (41/2083) | 83 | 82 |  |
| all/residual/kmeans_40 | 15 | 1615 | 437 | 0.68 | chapman_ningbo 0.44, ptbxl 0.34, georgia 0.14, cpsc 0.08 | sinus_bradycardia 0.35 | scp:axis:MID -4.39/-4.06 | missing_samples x1.06 (2/1615) | 62 | 90 |  |
| all/residual/kmeans_80 | 32 | 846 | 238 | 0.63 | chapman_ningbo 0.44, ptbxl 0.35, georgia 0.15, cpsc 0.06 | sinus_bradycardia 0.41 | scp:axis:MID -5.14/-3.39 | missing_samples x1.01 (1/846) | 60 | 92 |  |
| label_negative/raw/kmeans_10 | 1 | 2695 | 620 | 0.86 | ptbxl 0.58, chapman_ningbo 0.22, cpsc 0.10, georgia 0.10 | sinus_tachycardia 0.00 | scp:CLBBB -4.55/-2.72 | missing_samples x1.45 (3/2695) | 65 | 86 |  |
| label_negative/raw/kmeans_10 | 6 | 1294 | 346 | 0.92 | cpsc 0.39, ptbxl 0.27, georgia 0.23, chapman_ningbo 0.12 | sinus_tachycardia 0.00 | scp:ISCAS +4.86/+4.01 | contact_loss x1.69 (10/1294) | 76 | 84 |  |
| label_negative/raw/kmeans_20 | 4 | 1159 | 241 | 0.75 | ptbxl 0.57, chapman_ningbo 0.23, georgia 0.10, cpsc 0.10 | sinus_tachycardia 0.00 | scp:axis:LAD -5.27/-3.19 | hf_noise x1.15 (2/1159) | 65 | 86 |  |
| label_negative/raw/kmeans_20 | 7 | 1089 | 252 | 0.68 | ptbxl 0.52, chapman_ningbo 0.22, cpsc 0.13, georgia 0.13 | sinus_tachycardia 0.00 | scp:CLBBB -3.38/-1.43 | mains_interference x1.18 (7/1089) | 72 | 86 |  |
| label_negative/raw/kmeans_20 | 12 | 829 | 222 | 0.85 | cpsc 0.42, ptbxl 0.28, georgia 0.22, chapman_ningbo 0.08 | sinus_tachycardia 0.00 | scp:NORM -5.08/-5.77 | amplitude_outlier x1.90 (2/829) | 75 | 86 |  |
| label_negative/raw/kmeans_40 | 26 | 384 | 85 | 0.68 | ptbxl 0.67, georgia 0.20, chapman_ningbo 0.10 | sinus_tachycardia 0.00 | scp:PACE -2.85/-0.84 | baseline_wander x1.51 (16/384) | 69 | 86 |  |
| label_negative/raw/kmeans_40 | 29 | 311 | 88 | 0.66 | cpsc 0.45, ptbxl 0.31, georgia 0.20 | sinus_tachycardia 0.00 | scp:NORM -5.77/-4.75 | isolated_spike x1.76 (5/311) | 69 | 86 |  |
| label_negative/residual/kmeans_10 | 3 | 1936 | 478 | 0.64 | ptbxl 0.54, chapman_ningbo 0.24, georgia 0.14, cpsc 0.09 | sinus_tachycardia 0.00 | scp:axis:LAD -5.09/-3.82 | missing_samples x1.30 (2/1936) | 69 | 84 |  |
| label_negative/residual/kmeans_20 | 5 | 1072 | 247 | 0.72 | ptbxl 0.52, chapman_ningbo 0.23, cpsc 0.14, georgia 0.11 | sinus_tachycardia 0.00 | scp:CLBBB -3.35/-1.45 | amplitude_outlier x1.44 (2/1072) | 72 | 82 |  |
| label_negative/residual/kmeans_20 | 7 | 1006 | 242 | 0.65 | ptbxl 0.58, chapman_ningbo 0.20, georgia 0.12, cpsc 0.10 | sinus_tachycardia 0.00 | scp:axis:LAD -5.12/-3.19 | isolated_spike x0.74 (7/1006) | 66 | 84 |  |
| label_negative/residual/kmeans_20 | 9 | 928 | 237 | 0.66 | ptbxl 0.45, chapman_ningbo 0.30, cpsc 0.16, georgia 0.10 | sinus_tachycardia 0.00 | scp:PACE -3.35/-1.22 | identity_violation x2.00 (1/928) | 73 | 92 |  |
| label_negative/residual/kmeans_20 | 12 | 826 | 158 | 0.60 | ptbxl 0.51, chapman_ningbo 0.26, cpsc 0.13, georgia 0.10 | sinus_tachycardia 0.00 | scp:CLBBB -3.05/-1.06 | amplitude_outlier x1.90 (2/826) | 68 | 86 |  |
| label_negative/residual/kmeans_20 | 14 | 601 | 146 | 0.77 | ptbxl 0.35, cpsc 0.34, georgia 0.17, chapman_ningbo 0.14 | sinus_tachycardia 0.00 | scp:NORM -4.86/-4.01 | hf_noise x1.10 (1/601) | 74 | 84 |  |
| label_negative/residual/kmeans_80 | 58 | 151 | 30 | 0.79 | ptbxl 0.70, georgia 0.20, chapman_ningbo 0.06 | sinus_tachycardia 0.00 | snomed:47665007 +1.97/+2.18 | baseline_wander x1.67 (7/151) | 71 | 86 |  |
| label_negative/residual/kmeans_80 | 77 | 33 | 3 | 0.69 | ptbxl 0.58, chapman_ningbo 0.18, cpsc 0.12, georgia 0.12 | sinus_tachycardia 0.00 | scp:WPW +9.63/+7.02 | missing_samples x0.00 (0/33) | 70 | 124 |  |

Readings, from both review sheets of every candidate (24 records nearest the centre, 24 random members) and
the measurements and statements of [section `statements-candidates`]:

| kind | clusters | what is measurable |
|---|---|---|
| normal core | all/raw k10 c0, k20 c0, hdbscan c0; label_negative/raw k20 c7 | normal-looking sinus ECGs; sinus rhythm and normal ECG statements enriched (SNOMED sinus rhythm OR 18.1 in all/raw k10 c0, PTB-XL normal ECG OR 18.3 in the HDBSCAN core of 26,711); left axis and LBBB depleted |
| vertical axis, prominent T | all/residual k40 c1; label_negative/raw k10 c1, k20 c4; label_negative/residual k10 c3, k20 c7 | vertical axis with tall inferior QRS (qrs_area_II +1.24) and tall T in the label-negative groups (t_amp_II +1.06 to +1.30); early repolarization statement OR 2.8-13.5, on 17-36 members |
| T waves in the label gap | all/raw k40 c24; label_negative/raw k10 c6, k20 c12, k40 c29; label_negative/residual k20 c14 | flat or inverted lateral and inferior T (t_amp_V5 -1.33 to -1.63); PTB-XL ischemic statements enriched (ISCAS or ISCAL OR 44-137 in the label-negative groups); T-wave label undefined in 79-88% of label-negative members, ST depression undefined in 73% of all/raw k40 c24, which also has high precordial voltage |
| left axis in the label gap | all/residual k10 c4, k20 c5, k40 c15, k80 c32; label_negative/residual k20 c12 | negative QRS in II, III, aVF (QRS area in II or aVF -1.17 to -1.43 within source); in the four all-record groups left axis deviation is positive in 21-32% of members and LAFB in 16-21% (`describe/clusters.csv`); PTB-XL `heart_axis` LAD enriched (OR 4.0-10.1); left axis deviation undefined in 58% of label_negative/residual k20 c12 against 38% of its subset |
| long PR | all/residual k10 c3 | P onset about 180-220 ms before R in the median beats, pr_ms +0.90; first-degree AV block positive in 22% |
| short PR | label_negative/residual k20 c5 | P onset about 120-140 ms before R, pr_ms -1.21; no QRS widening or delta wave visible on the sheets |
| baseline drift | all/residual k20 c8; label_negative/raw k40 c26; label_negative/residual k80 c58 | normal-looking ECGs with slow baseline movement, noise_drift_frac +0.99, +1.10, +1.19 within source; the `baseline_wander` flag is 1.5-1.7 times as frequent as outside (`describe/flag_rates.csv`), under the rule's factor of two |
| pre-excitation pattern | label_negative/residual k80 c77 | short PR with slurred, widened QRS onset in the median beats; WPW statement on 19 of 33 (OR 15,270 train, 1,115 validation); QRS median 124 ms; 3 validation members |
| low limb-lead voltage | all/residual k40 c13 | low limb-lead QRS (ptp_II -0.92); SNOMED low QRS voltage on 110 of 2,083 (OR 3.7 train, 4.2 validation); the twelve-lead sheet shows the limb leads are also the noisier ones |
| terminal rightward QRS | label_negative/residual k20 c9 | S in I and V6, late R in aVR, QRS about 90-105 ms (qrs_dur_ms +0.56); IVCD statement OR 2.9 on train, 0.8 on validation; no rSR' in V1 on the sheets |

[section `undefined`] For each candidate, the two labels whose undefined share among train members most
exceeds the subset's:

| solution | c | undefined in members (in subset) |
|---|---|---|
| all/raw/kmeans_10 | 0 | left_axis_deviation 0.38 (0.28); sinus_tachycardia 0.11 (0.07) |
| all/raw/kmeans_20 | 0 | left_axis_deviation 0.34 (0.28); sinus_tachycardia 0.11 (0.07) |
| all/raw/kmeans_40 | 24 | st_depression 0.73 (0.22); mi_any 0.45 (0.20) |
| all/raw/hdbscan | 0 | left_axis_deviation 0.28 (0.28); af_or_flutter 0.00 (0.00) |
| all/residual/kmeans_10 | 3 | sinus_arrhythmia 0.19 (0.18); incomplete_rbbb 0.19 (0.18) |
| all/residual/kmeans_10 | 4 | left_axis_deviation 0.33 (0.28); complete_rbbb 0.01 (0.01) |
| all/residual/kmeans_20 | 5 | left_axis_deviation 0.34 (0.28); af_or_flutter 0.00 (0.00) |
| all/residual/kmeans_20 | 8 | left_axis_deviation 0.30 (0.28); mi_any 0.21 (0.20) |
| all/residual/kmeans_40 | 1 | lvh 0.15 (0.13); sinus_tachycardia 0.08 (0.07) |
| all/residual/kmeans_40 | 13 | lafb 0.21 (0.21); sinus_arrhythmia 0.18 (0.18) |
| all/residual/kmeans_40 | 15 | left_axis_deviation 0.35 (0.28); mi_any 0.20 (0.20) |
| all/residual/kmeans_80 | 32 | left_axis_deviation 0.34 (0.28); complete_rbbb 0.01 (0.01) |
| label_negative/raw/kmeans_10 | 1 | sinus_bradycardia 0.09 (0.07); sinus_tachycardia 0.08 (0.07) |
| label_negative/raw/kmeans_10 | 6 | t_wave_abnormality 0.79 (0.19); st_depression 0.81 (0.21) |
| label_negative/raw/kmeans_20 | 4 | sinus_tachycardia 0.08 (0.07); sinus_bradycardia 0.08 (0.07) |
| label_negative/raw/kmeans_20 | 7 | st_depression 0.25 (0.21); t_wave_abnormality 0.23 (0.19) |
| label_negative/raw/kmeans_20 | 12 | t_wave_abnormality 0.86 (0.19); st_depression 0.86 (0.21) |
| label_negative/raw/kmeans_40 | 26 | st_depression 0.26 (0.21); mi_any 0.21 (0.19) |
| label_negative/raw/kmeans_40 | 29 | t_wave_abnormality 0.88 (0.19); st_depression 0.90 (0.21) |
| label_negative/residual/kmeans_10 | 3 | mi_any 0.21 (0.19); lvh 0.09 (0.09) |
| label_negative/residual/kmeans_20 | 5 | t_wave_abnormality 0.22 (0.19); st_depression 0.23 (0.21) |
| label_negative/residual/kmeans_20 | 7 | sinus_bradycardia 0.08 (0.07); lvh 0.09 (0.09) |
| label_negative/residual/kmeans_20 | 9 | sinus_tachycardia 0.11 (0.07); lvh 0.12 (0.09) |
| label_negative/residual/kmeans_20 | 12 | left_axis_deviation 0.58 (0.38); t_wave_abnormality 0.20 (0.19) |
| label_negative/residual/kmeans_20 | 14 | t_wave_abnormality 0.81 (0.19); st_depression 0.78 (0.21) |
| label_negative/residual/kmeans_80 | 58 | st_depression 0.26 (0.21); left_axis_deviation 0.40 (0.38) |
| label_negative/residual/kmeans_80 | 77 | left_axis_deviation 0.61 (0.38); st_depression 0.24 (0.21) |

## 5. Context clusters

The 165 stable, multi-source clusters that are not candidates were drawn as well; 57 were read (all of
all/raw k10, k20 and k40, 12 of all/raw k80, and 6 that fail only the artefact rule; 51 on both sheets, the 6
on the nearest sheet only) [section `notes`]. They show:

- clusters of the 14 conditions (AF, flutter, sinus bradycardia and tachycardia, complete RBBB, ventricular
  bigeminy, Q-wave MI, LAFB, first-degree AV block), which fail "already labelled" and usually the literal
  artefact rule as well;
- LBBB clusters (all/raw k20 c15, k40 c27) and paced clusters (all/raw k10 c8, k20 c16, k40 c30), outside the
  14, with no one of the 14 labels positive in more than half; all/raw k40 c27 fails only the literal rule (three members each with `adc_rail`
  and `identity_violation`), the others also fail with five flagged members (`mains_interference` 23 of
  1,122 in k20 c15, `flat_lead` 75 of 888 in k20 c16);
- groups held together by recording defects, such as Ningbo records with flat chest leads in all/raw k10 c9
  (`flat_lead` 477 of 1,709) and k40 c39;
- among the near-candidates read: low or negative inferior P waves in label-negative records
  (label_negative/raw k80 c62, atrial rhythm statement on 46 of 118), flat lateral T with tall lateral R
  (all/residual k20 c9), left axis with deep right precordial S (all/residual k80 c53), slow sinus with long
  normal PR (all/residual k20 c10), and two ordinary normal groups (label_negative/raw k20 c13,
  all/residual k40 c8).

## 6. Label-negative subset

15,763 train and 3,740 validation records have no positive among their defined labels. 15 of the 27
candidates come from this subset. In five of them most members sit in the undefined part of the label that
would describe them (T-wave abnormality in CPSC-Extra and in PTB-XL records with ischemic statements; left
axis deviation in PTB-XL), so the subset is not a clean normal set. Among its near-candidates
are LBBB and pacemaker groups (label_negative/raw k10 c8 with PTB-XL complete LBBB on 161 of 347; k40 c36 with
the pacemaker statement on 102 of 125) [section `artefact`].

## 7. Cross-encoder check

[section `cross-encoder`] Membership of the 192 stable, multi-source clusters was predicted from ECG-CPC
embeddings, whose pretraining saw none of these sources. Validation AUROC within source: mean 0.980 over the
192 (lowest 0.860, label_negative/residual k10 c2); over the 27 candidates mean 0.961, lowest 0.886
(label_negative/residual k20 c9, the terminal rightward QRS group), highest 1.000 (the pre-excitation group,
from 3 validation members). xECG itself gives 0.986 on the candidates. So the groups are not peculiar to
xECG, but the check is weak: the clusters follow waveform properties that any ECG encoder represents.

## 8. Outliers

[section `outliers`] 562 train records per space (1%), 383 of them outliers in both spaces. In the raw space:

- by source, 1.5% of Georgia, 1.2% of Ningbo, Chapman and CPSC-Extra, 0.8% of CPSC and 0.5% of PTB-XL;
- flagged 24% against 8%; median heart rate 101 against 72, QRS 104 against 86 ms, age 69 against 62;
- labels: AF or flutter 39% against 14%, ventricular ectopy 19% against 4%, complete RBBB 14% against 5%;
- enriched statements: paroxysmal ventricular tachycardia (20 records, OR 56), bigeminy (8, OR 30),
  ventricular escape rhythm (13, OR 28), complete heart block (10, OR 17), ventricular fibrillation (6, OR 14),
  WPW pattern (6, OR 13), supraventricular tachycardia (42, OR 12).

The residual space gives the same picture, with more flagged outliers (35%) and more of them with a flat lead
(98 against 43 in the raw space).
All eight raw sheets and two of the eight residual sheets (48 random outliers per space, six per sheet) were
read [section `notes`]: frequent or multiform ventricular beats, wide-complex and very fast rhythms, AF at
extreme rates, and defects such as square clipping, steps, heavy limb-lead noise and one to six flat chest
leads.

The outlier share depends on the Ningbo record number [section `ningbo-numbers`]:

| block | records | outliers | outlier_share | label_negative_share | under_18_share |
|---|---|---|---|---|---|
| (10000, 20000] | 5601 | 153 | 0.027 | 0.017 | 0.045 |
| (20000, 30000] | 5949 | 66 | 0.011 | 0.073 | 0.022 |
| (30000, 40000] | 6071 | 1 | 0.000 | 0.346 | 0.076 |
| (40000, 45000] | 3005 | 12 | 0.004 | 0.280 | 0.101 |
| (45000, 50000] | 293 | 25 | 0.085 | 0.321 | 0.140 |

label_negative/raw/kmeans_80 cluster 62: 61 Ningbo train members, 44 numbered JS20900-JS21100; the label-negative Ningbo train records in that range: 98 of 3568.

The outlier share of Ningbo train records is 2.7% in JS10000-JS20000, 0.0% in JS30000-JS40000 and 8.5% above
JS45000, and the label-negative share runs from 1.7% to 34.6% across the same blocks. The atrial-rhythm group
label_negative/raw k80 c62 draws 44 of its 61 Ningbo members from JS20900-JS21100, a range that holds 98 of
the 3,568 label-negative Ningbo train records. On the outlier sheets the late block shows records with no
positive label and markedly abnormal-looking tracings (a sinusoidal wide oscillation at 193/min, a wide slow
rhythm at 32/min, flat chest leads). The numbering seems to group recordings by something we cannot see
(diagnosis, period or device).

## 9. Statements the embedding separates

[section `statements`] Logistic regression on the standardized embedding (C = 1), within-source validation
AUROC with a 2,000-draw patient bootstrap, for the statements with at least 100 train positives in two
families. `train_by_source` follows the order ptbxl/ningbo/chapman_shaoxing/georgia/cpsc_2018/cpsc_2018_extra;
"space C positive" marks statements MLC-01 already maps to a condition outside the 14.

| feature | auroc raw | auroc residual | ci_high raw | ci_high residual | ci_low raw | ci_low residual | description | mapping | train_positives | validation_positives | train_by_source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snomed:426783006 | 0.973 | 0.927 | 0.977 | 0.935 | 0.970 | 0.919 | sinus rhythm |  | 6443 | 2151 | 0/3759/1102/1028/551/3 |
| snomed:164909002 | 0.977 | 0.972 | 0.986 | 0.983 | 0.964 | 0.956 | left bundle branch block | space C positive: complete_lbbb | 418 | 144 | 0/18/133/138/108/21 |
| snomed:89792004 | 0.978 | 0.976 | 0.989 | 0.989 | 0.961 | 0.960 | right ventricular hypertrophy | space C positive: rvh | 132 | 39 | 0/65/4/51/0/12 |
| snomed:47665007 | 0.968 | 0.962 | 0.977 | 0.975 | 0.957 | 0.945 | right axis deviation | space C positive: right_axis_deviation | 566 | 187 | 0/383/132/51/0/0 |
| snomed:428417006 | 0.968 | 0.964 | 0.978 | 0.975 | 0.956 | 0.952 | early repolarization |  | 314 | 109 | 0/215/11/88/0/0 |
| snomed:284470004 | 0.959 | 0.957 | 0.968 | 0.968 | 0.947 | 0.946 | premature atrial contraction | space C positive: premature_atrial_complex | 1518 | 526 | 0/626/152/376/334/30 |
| snomed:251146004 | 0.944 | 0.942 | 0.955 | 0.954 | 0.933 | 0.930 | low qrs voltages | space C positive: low_qrs_voltage | 857 | 271 | 0/476/156/225/0/0 |
| snomed:426761007 | 0.968 | 0.965 | 0.994 | 0.992 | 0.928 | 0.927 | supraventricular tachycardia | space C positive: supraventricular_tachycardia | 427 | 153 | 0/72/334/20/0/1 |
| snomed:713422000 | 0.951 | 0.899 | 0.971 | 0.952 | 0.926 | 0.832 | atrial tachycardia | space C positive: supraventricular_tachycardia | 194 | 69 | 0/104/74/11/0/5 |
| snomed:233917008 | 0.937 | 0.892 | 0.958 | 0.944 | 0.911 | 0.829 | av block | ambiguous: first_degree_av_block | 199 | 65 | 0/49/102/46/0/2 |
| snomed:698252002 | 0.923 | 0.908 | 0.940 | 0.930 | 0.904 | 0.885 | nonspecific intraventricular conduction disorder | space C positive: nonspecific_ivcd | 553 | 214 | 0/312/130/109/0/2 |
| snomed:6374002 | 0.929 | 0.881 | 0.954 | 0.931 | 0.901 | 0.825 | bundle branch block | ambiguous: complete_rbbb; ambiguous: complete_lbbb | 320 | 79 | 0/247/0/73/0/0 |
| snomed:111975006 | 0.905 | 0.900 | 0.921 | 0.920 | 0.888 | 0.877 | prolonged qt interval | space C positive: long_qt | 1046 | 380 | 0/196/34/814/0/2 |
| snomed:164917005 | 0.892 | 0.881 | 0.913 | 0.905 | 0.870 | 0.857 | qwave abnormal | ambiguous: mi_any | 915 | 294 | 0/507/137/271/0/0 |
| snomed:164931005 | 0.881 | 0.883 | 0.915 | 0.917 | 0.847 | 0.849 | st elevation | space C positive: st_elevation | 312 | 105 | 0/0/111/77/86/38 |
| snomed:164930006 | 0.841 | 0.827 | 0.865 | 0.855 | 0.814 | 0.798 | st interval abnormal | ambiguous: st_depression; ambiguous: t_wave_abnormality | 1337 | 461 | 0/485/1/573/0/278 |
| snomed:426627000 | 0.863 | 0.838 | 0.906 | 0.887 | 0.813 | 0.795 | bradycardia | ambiguous: sinus_bradycardia | 160 | 50 | 0/5/0/5/0/150 |
| snomed:428750005 | 0.782 | 0.756 | 0.801 | 0.776 | 0.764 | 0.736 | nonspecific st t abnormality | ambiguous: st_depression; ambiguous: t_wave_abnormality | 2537 | 815 | 0/0/714/1120/0/703 |

- PTB-XL statements never qualify, because the two vocabularies were not merged (departure 5). The
  single-family list [section `statements-one-family`] adds, among others, complete LBBB in Ningbo (0.998),
  pacing rhythm (0.984) and PTB-XL pacemaker (0.983), PTB-XL right axis (0.979), the Ningbo atrial rhythm
  (0.976) and tall P wave (0.973), and PTB-XL complete LBBB (0.969).
- Sinus rhythm, bradycardia and AV block restate labels the embedding already encodes. Bradycardia has 150 of
  its 160 train positives in CPSC-Extra.
- For comparison, the 14 labels probed the same way give 0.911-0.998 [section `labels`].

## Review sheets

`outputs/key06_embedding_structure_20261111/review/<subset>__<space>/<method>/cNN_nearest.png` and
`cNN_random.png` for the 192 stable, multi-source clusters; `cNN_detail_1.png` (twelve leads of the 6
nearest records) for the 27 candidates; `outliers/<space>/sheet_1.png` to `sheet_8.png`. The analyst's
readings are in `review/notes.csv` (96 rows) [section `notes`].

## Departures and doubts

The operational choices the protocol left open were fixed before any full run, in the module docstring of
`ecg_experiment/key06_embedding_structure.py` (commit `1047c65`) and in the run script (`a682a5a`). They are:

1. Signal flags. The protocol lists the 18 MLC-03 flags. Steps 1 and 2 use the 15 that passed MLC-03's visual
   check. `reversal_ra_ll` and `rhythm_unmeasurable` failed it, and `duplicate_lead` flags no record. Two of the
   15 (`reversal_la_ra`, `reversal_neutral`) passed only in a check that reused the first viewing; they hold 8
   records each and do not matter to any share.
2. Factor groups. Heart rate is a cubic spline of `hr_bpm` with a missing indicator. `rr_mean_ms` and `n_beats`
   restate the rate and enter no group, so "other measurements" are the remaining 127, as normal scores with
   missing indicators. Age is a spline with a missing indicator; sex is two indicators.
3. Variance shares. "Cross-validated linear regression onto the principal components, fitted on train, read on
   validation" was read as one least-squares fit on train, scored on validation. Over all 1,024 standardized
   dimensions this equals regression onto all principal components; the 91 components that keep 90% of the
   variance are reported beside it.
4. Reverse direction. Binary factors use logistic regression (C = 1) and continuous ones ridge regression
   (alpha = 1), without tuning. "Within source" is the stratified AUROC (a positive against a negative of the
   same source) and the R^2 around each source's own mean. The source itself is scored by accuracy.
5. Statements. A statement is a raw code that is not a positive code of the 14 conditions: PTB-XL SCP
   statements of any likelihood, PTB-XL's `heart_axis` as `axis:X`, and SNOMED CT codes of the Challenge
   sources. The two vocabularies are not merged, so a PTB-XL statement is always in one family.
6. Enrichment. Mantel-Haenszel odds ratio across the six sources, 0.5 added to the cells of a stratum with an
   empty cell, Cochran-Mantel-Haenszel test without continuity correction. Benjamini-Hochberg runs once over
   all cluster-statement pairs of the study on train (20 solutions together), and once over all
   cluster-label pairs.
7. Replication. "Its strongest enrichment" is the label or statement with the largest absolute log odds ratio
   among those significant on train after Benjamini-Hochberg. Its interval is the 2.5-97.5 percentile range of
   the validation log odds ratio over 2,000 patient resamples. A cluster with no significant enrichment fails
   the rule.
8. Artefact rule. Read literally: a flag present among members and absent outside counts as more than twice as
   frequent, however few members carry it. A second reading that also needs five flagged members is reported
   beside it and decides nothing.
9. "Positive in more than half its members" counts undefined members in the denominator.
10. HDBSCAN validation records take the label of their nearest train record, noise included; k-means validation
    records go to the nearest centre.
11. Step 5 was run for every candidate and, as context, for every cluster that is stable and not one source.
    The protocol's "each candidate cluster of step 6" was read as the candidates of step 4.
12. Step 7 reads "present in at least two source families" as at least five train positives (the MLC-01
    annotation rule) in sources of two families. Statements with 100 train positives in one family only,
    which includes every PTB-XL statement, are listed separately and are outside the protocol's list.
13. Review sheets. Each cluster has two sheets (24 nearest the centre, 24 random members), each showing the
    median beats of all 12 leads overlaid and a 10 s lead II strip per record. One twelve-lead sheet of the 6
    nearest records was drawn for every candidate.
14. The integrity step hashes the whole feature files, so the bytes of test-role rows pass through SHA-256; no
    test-role value is parsed or kept.

Added after the full run:

15. Reading coverage. All 27 candidates were read on both sheets; of their twelve-lead sheets only two were
    read (the pre-excitation and low-voltage groups). 57 of the 165 context clusters were read, the other 108
    are described by their tables only. Two of the eight residual outlier sheets were read.
16. The describe step used the per-cluster flag rates without keeping them. A `flags` step (commit
    `24ee16c`) writes them to `describe/flag_rates.csv`; it reproduces both artefact columns for all 606
    clusters and changes no result.
17. The report script reads train label states and record numbers through the role-guarded loader for the
    `undefined` and `ningbo-numbers` sections; no validation or test value enters them.
18. The artefact rule cannot see an artefact that no flag measures: three candidates are baseline-drift groups
    below the `baseline_wander` threshold. In the other direction, the literal rule rejects LBBB and pacemaker
    groups on one to four rare flags.
19. The replication rule tests only the strongest enrichment, which is often a depletion (LBBB, left axis or
    normal ECG absent). A cluster can therefore replicate without its defining trait being tested on
    validation; for the pre-excitation group the validation interval rests on 3 members.
20. The candidate kinds and their counts are the analyst's grouping of the readings, not an output of the
    rules.
21. The cross-encoder check cannot tell a finding from an artefact: both are waveform properties that ECG-CPC
    also represents.
22. The experiment used dataset v4 because v5 did not exist when the protocol was frozen.

## Follow-up ideas

These are for the backlog, ranked there; none was run.

- Decide whether the artefact rule should need a minimum number of flagged members. With five, 50 more
  clusters pass, among them the LBBB and pacemaker groups.
- Candidate labels for later, if the clinical review wants them: left bundle branch block, right ventricular
  hypertrophy, right axis deviation, early repolarization, premature atrial contraction and low QRS voltage
  (each separated with a validation AUROC lower bound of at least 0.93 in two families), and paced rhythm and
  pre-excitation, which form clusters but miss step 7's rule: each vocabulary codes pacing in one family, and
  neither WPW statement reaches 100 train positives (both are absent from both statement lists).
- A baseline-drift measure as a quality covariate, below MLC-03's `baseline_wander` threshold.
- The label-negative set is not a normal set: T-wave abnormality is undefined in most CPSC-Extra records and in
  PTB-XL records with ischemic statements, and left axis deviation is undefined in PTB-XL. Any use of
  label-negative records as normals (for example the referral budget of Experiment 030) should check this.
- Ningbo record numbers: find what the blocks correspond to, and check the late block (JS45000 and above)
  against its source codes.

## How to rerun

```bash
export PYTHONPATH=. OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for step in integrity variance residual prepare cluster assemble describe flags review cross-encoder \
    outliers statements; do
  .venv/bin/python -m scripts.experiments.run_key06_embedding_structure "$step" --workers 12
done
.venv/bin/python -m scripts.experiments.run_key06_embedding_structure detail --solution <solution> \
    --cluster <n> --count 6
.venv/bin/python -m scripts.reports.key06_tables
```

Scratch files go to `/var/tmp/janetrivera/key06/`; `--smoke` runs every step on a small sample under the
scratch directory. The full run took 77 minutes of wall time, most of it HDBSCAN on the subsamples. Each step
writes a receipt under `outputs/key06_embedding_structure_20261111/receipts/`.

## Appendix A: clusters that fail only the artefact rule

[section `artefact`]

| solution | cluster | n train | top label | strongest | first statement (OR train/val [with/members]) | failing flags (members flagged/members) | passes min-5 | failing with 5 or more |
|---|---|---|---|---|---|---|---|---|
| all/raw/kmeans_10 | 5 | 5598 | left_axis_deviation 0.22 | left_axis_deviation | scp:axis:AXL (heart_axis = AXL) OR 14.7/7.9 [59/5598] | reversal_la_ra 2/5598; reversal_neutral 2/5598 | True |  |
| all/raw/kmeans_10 | 6 | 4878 | t_wave_abnormality 0.33 | scp:NORM | snomed:733534002 (complete left bundle branch block) OR 10.5/16.0 [33/4878] | near_flat_lead 3/4878; reversal_neutral 2/4878 | True |  |
| all/raw/kmeans_10 | 7 | 3857 | sinus_bradycardia 0.20 | scp:CLBBB | scp:BIGU (bigeminal pattern (unknown origin, SV or) OR 10.3/6.8 [33/3857] | adc_rail 14/3857; identity_violation 9/3857; amplitude_outlier 8/3857 | False | adc_rail 14/3857; identity_violation 9/3857; amplitude_outlier 8/3857 |
| all/raw/kmeans_10 | 8 | 2046 | af_or_flutter 0.36 | scp:NORM | snomed:81898007 (ventricular escape rhythm) OR 294.5/185.6 [48/2046] | edge_hold 2/2046; flat_lead 80/2046; near_flat_lead 2/2046; identity_violation 7/2046; reversal_la_ra 1/2046; amplitude_outlier 20/2046; isolated_spike 56/2046; contact_loss 24/2046 | False | flat_lead 80/2046; identity_violation 7/2046; amplitude_outlier 20/2046; isolated_spike 56/2046; contact_loss 24/2046 |
| all/raw/kmeans_10 | 9 | 1709 | sinus_tachycardia 0.42 | snomed:426761007 | snomed:426761007 (supraventricular tachycardia) OR 1069.8/1028.2 [415/1709] | edge_hold 3/1709; flat_lead 477/1709; mains_interference 64/1709; hf_noise 24/1709; isolated_spike 45/1709 | False | flat_lead 477/1709; mains_interference 64/1709; hf_noise 24/1709; isolated_spike 45/1709 |
| all/raw/kmeans_20 | 6 | 3597 | t_wave_abnormality 0.40 | scp:NORM | scp:INJAL (subendocardial injury in anterolateral l) OR 10.9/7.7 [56/3597] | near_flat_lead 3/3597; reversal_neutral 2/3597 | True |  |
| all/raw/kmeans_20 | 9 | 2890 | sinus_bradycardia 0.19 | scp:PACE | scp:BIGU (bigeminal pattern (unknown origin, SV or) OR 16.2/9.0 [36/2890] | adc_rail 12/2890; identity_violation 8/2890; amplitude_outlier 7/2890; baseline_wander 162/2890 | False | adc_rail 12/2890; identity_violation 8/2890; amplitude_outlier 7/2890; baseline_wander 162/2890 |
| all/raw/kmeans_20 | 15 | 1122 | af_or_flutter 0.21 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 1326.8/527.3 [396/1122] | edge_hold 1/1122; identity_violation 4/1122; amplitude_outlier 4/1122; mains_interference 23/1122 | False | mains_interference 23/1122 |
| all/raw/kmeans_20 | 16 | 888 | af_or_flutter 0.35 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 1018.1/1128.2 [183/888] | edge_hold 1/888; flat_lead 75/888; near_flat_lead 2/888; identity_violation 4/888; amplitude_outlier 14/888; hf_noise 6/888; isolated_spike 35/888; contact_loss 24/888 | False | flat_lead 75/888; amplitude_outlier 14/888; hf_noise 6/888; isolated_spike 35/888; contact_loss 24/888 |
| all/raw/kmeans_40 | 10 | 1973 | sinus_bradycardia 0.19 | scp:PACE | snomed:253352002 (left atrial abnormality) OR 8.9/3.7 [17/1973] | baseline_wander 119/1973 | False | baseline_wander 119/1973 |
| all/raw/kmeans_40 | 27 | 818 | left_axis_deviation 0.17 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 1080.0/354.3 [364/818] | adc_rail 3/818; identity_violation 3/818; amplitude_outlier 2/818 | True |  |
| all/raw/kmeans_40 | 28 | 755 | sinus_tachycardia 0.46 | sinus_bradycardia | snomed:426761007 (supraventricular tachycardia) OR 20.7/17.6 [124/755] | identity_violation 2/755; amplitude_outlier 2/755; mains_interference 23/755; hf_noise 12/755; isolated_spike 22/755 | False | mains_interference 23/755; hf_noise 12/755; isolated_spike 22/755 |
| all/raw/kmeans_40 | 30 | 640 | af_or_flutter 0.41 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 2588.7/1882.1 [178/640] | identity_violation 2/640; amplitude_outlier 2/640; isolated_spike 22/640 | False | isolated_spike 22/640 |
| all/raw/kmeans_40 | 39 | 239 | af_or_flutter 0.20 | snomed:81898007 | snomed:81898007 (ventricular escape rhythm) OR 1296.0/691.3 [45/239] | edge_hold 1/239; adc_rail 1/239; flat_lead 80/239; near_flat_lead 2/239; identity_violation 1/239; amplitude_outlier 10/239; hf_noise 4/239; isolated_spike 10/239; contact_loss 20/239 | False | flat_lead 80/239; amplitude_outlier 10/239; isolated_spike 10/239; contact_loss 20/239 |
| all/raw/kmeans_80 | 42 | 708 | first_degree_av_block 0.46 | sinus_bradycardia | snomed:251170000 (blocked premature atrial contraction) OR 11.3/5.9 [2/708] | reversal_la_ra 1/708 | True |  |
| all/raw/kmeans_80 | 58 | 464 | af_or_flutter 0.34 | snomed:10370003 | snomed:10370003 (pacing rhythm) OR 1523.6/819.3 [268/464] | identity_violation 2/464; amplitude_outlier 1/464; isolated_spike 15/464 | False | isolated_spike 15/464 |
| all/raw/kmeans_80 | 78 | 178 | af_or_flutter 0.16 | snomed:81898007 | snomed:81898007 (ventricular escape rhythm) OR 1257.2/1040.4 [43/178] | edge_hold 1/178; adc_rail 1/178; flat_lead 80/178; near_flat_lead 2/178; identity_violation 1/178; amplitude_outlier 9/178; hf_noise 3/178; isolated_spike 9/178; contact_loss 20/178 | False | flat_lead 80/178; amplitude_outlier 9/178; isolated_spike 9/178; contact_loss 20/178 |
| all/raw/kmeans_80 | 79 | 163 | af_or_flutter 0.33 | snomed:425856008 | snomed:425856008 (paroxysmal ventricular tachycardia) OR 151.0/330.1 [16/163] | flat_lead 11/163; amplitude_outlier 1/163; baseline_wander 12/163; mains_interference 3/163; hf_noise 3/163; baseline_step 14/163 | False | flat_lead 11/163; baseline_wander 12/163; baseline_step 14/163 |
| all/raw/hdbscan | 3 | 144 | af_or_flutter 0.37 | snomed:10370003 | snomed:10370003 (pacing rhythm) OR 5391.0/2599.7 [92/144] | baseline_step 9/144; isolated_spike 5/144 | False | baseline_step 9/144; isolated_spike 5/144 |
| all/residual/kmeans_10 | 1 | 8907 | sinus_bradycardia 0.21 | scp:CLBBB | scp:LNGQT (long QT-interval) OR 4.0/4.0 [39/8907] | edge_hold 3/8907 | True |  |
| all/residual/kmeans_10 | 2 | 8858 | sinus_bradycardia 0.35 | lafb | snomed:428417006 (early repolarization) OR 7.9/10.7 [186/8858] | reversal_neutral 2/8858 | True |  |
| all/residual/kmeans_10 | 6 | 4314 | t_wave_abnormality 0.34 | scp:NORM | snomed:733534002 (complete left bundle branch block) OR 34.7/90.6 [81/4314] | edge_hold 2/4314; near_flat_lead 3/4314; reversal_neutral 1/4314; amplitude_outlier 9/4314 | False | amplitude_outlier 9/4314 |
| all/residual/kmeans_10 | 7 | 4286 | sinus_bradycardia 0.20 | snomed:733534002 | scp:BIGU (bigeminal pattern (unknown origin, SV or) OR 16.8/6.9 [40/4286] | identity_violation 5/4286 | False | identity_violation 5/4286 |
| all/residual/kmeans_10 | 9 | 1388 | af_or_flutter 0.34 | scp:NORM | scp:PACE (normal functioning artificial pacemaker) OR 212.6/222.5 [187/1388] | flat_lead 48/1388; identity_violation 2/1388; isolated_spike 34/1388; contact_loss 15/1388 | False | flat_lead 48/1388; isolated_spike 34/1388; contact_loss 15/1388 |
| all/residual/kmeans_20 | 0 | 5177 | sinus_bradycardia 0.21 | scp:CLBBB | scp:NORM (normal ECG) OR 2.9/2.2 [1127/5177] | edge_hold 2/5177 | True |  |
| all/residual/kmeans_20 | 2 | 4756 | sinus_bradycardia 0.43 | scp:ISCAL | snomed:428417006 (early repolarization) OR 9.1/12.1 [147/4756] | reversal_la_ra 1/4756 | True |  |
| all/residual/kmeans_20 | 4 | 4355 | sinus_bradycardia 0.31 | scp:CLBBB | scp:HVOLT (high QRS voltage) OR 4.6/2.2 [15/4355] | reversal_neutral 1/4355 | True |  |
| all/residual/kmeans_20 | 6 | 3956 | t_wave_abnormality 0.19 | left_axis_deviation | scp:axis:MID (heart_axis = MID) OR 3.3/1.6 [836/3956] | reversal_neutral 1/3956 | True |  |
| all/residual/kmeans_20 | 9 | 2776 | t_wave_abnormality 0.45 | scp:NORM | scp:INJAL (subendocardial injury in anterolateral l) OR 15.9/18.7 [55/2776] | near_flat_lead 3/2776; reversal_neutral 2/2776 | True |  |
| all/residual/kmeans_20 | 10 | 2644 | first_degree_av_block 0.47 | scp:CLBBB | snomed:426761007 (supraventricular tachycardia) OR 7.7/9.7 [94/2644] | reversal_la_ra 1/2644 | True |  |
| all/residual/kmeans_20 | 16 | 1175 | af_or_flutter 0.21 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 1381.2/759.4 [391/1175] | edge_hold 2/1175; identity_violation 2/1175; amplitude_outlier 4/1175; mains_interference 23/1175 | False | mains_interference 23/1175 |
| all/residual/kmeans_20 | 17 | 681 | af_or_flutter 0.38 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 2355.8/2391.1 [180/681] | identity_violation 2/681; isolated_spike 21/681 | False | isolated_spike 21/681 |
| all/residual/kmeans_40 | 8 | 2172 | sinus_bradycardia 0.21 | scp:PACE | snomed:253352002 (left atrial abnormality) OR 10.1/6.4 [17/2172] | near_flat_lead 1/2172 | True |  |
| all/residual/kmeans_40 | 11 | 2136 | sinus_bradycardia 0.19 | scp:CLBBB | scp:axis:AXL (heart_axis = AXL) OR 7.7/10.3 [21/2136] | edge_hold 1/2136 | True |  |
| all/residual/kmeans_40 | 12 | 2128 | sinus_bradycardia 0.48 | scp:axis:LAD | snomed:428417006 (early repolarization) OR 9.2/13.0 [84/2128] | reversal_la_ra 1/2128 | True |  |
| all/residual/kmeans_40 | 26 | 924 | af_or_flutter 0.20 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 1786.8/612.4 [377/924] | edge_hold 1/924; identity_violation 2/924; mains_interference 18/924 | False | mains_interference 18/924 |
| all/residual/kmeans_40 | 34 | 594 | af_or_flutter 0.39 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 4846.7/4784.4 [175/594] | identity_violation 2/594; isolated_spike 19/594 | False | isolated_spike 19/594 |
| all/residual/kmeans_80 | 53 | 554 | t_wave_abnormality 0.33 | scp:NORM | snomed:74390002 (wolff parkinson white pattern) OR 32.5/44.5 [8/554] | near_flat_lead 1/554 | True |  |
| all/residual/kmeans_80 | 58 | 429 | left_axis_deviation 0.38 | scp:NORM | snomed:164909002 (left bundle branch block) OR 193.7/284.0 [112/429] | adc_rail 2/429; identity_violation 2/429; amplitude_outlier 2/429; mains_interference 8/429 | False | mains_interference 8/429 |
| all/residual/kmeans_80 | 61 | 423 | first_degree_av_block 0.15 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 451.6/270.3 [179/423] | edge_hold 1/423; identity_violation 1/423; amplitude_outlier 4/423 | True |  |
| all/residual/kmeans_80 | 62 | 408 | af_or_flutter 0.33 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 2698.6/1760.7 [132/408] | identity_violation 2/408; amplitude_outlier 1/408; isolated_spike 12/408 | False | isolated_spike 12/408 |
| all/residual/kmeans_80 | 66 | 338 | complete_rbbb 0.50 | scp:NORM | snomed:425856008 (paroxysmal ventricular tachycardia) OR 93.6/111.7 [14/338] | isolated_spike 9/338 | False | isolated_spike 9/338 |
| label_negative/raw/kmeans_10 | 0 | 2879 | sinus_tachycardia 0.00 | snomed:426761007 | scp:SR (sinus rhythm) OR 5.5/10.7 [1184/2879] | near_flat_lead 1/2879; reversal_neutral 1/2879 | True |  |
| label_negative/raw/kmeans_10 | 2 | 2531 | sinus_tachycardia 0.00 | scp:CLBBB | scp:axis:LAD (heart_axis = LAD) OR 8.9/10.5 [511/2531] | edge_hold 1/2531; reversal_la_ra 1/2531 | True |  |
| label_negative/raw/kmeans_10 | 4 | 1436 | sinus_tachycardia 0.00 | snomed:10370003 | scp:axis:RAD (heart_axis = RAD) OR 4.5/8.0 [8/1436] | edge_hold 1/1436; adc_rail 5/1436; flat_lead 100/1436; identity_violation 2/1436; reversal_la_ra 1/1436; mains_interference 18/1436 | False | adc_rail 5/1436; flat_lead 100/1436; mains_interference 18/1436 |
| label_negative/raw/kmeans_10 | 5 | 1412 | sinus_tachycardia 0.00 | scp:CLBBB | scp:LVOLT (low QRS voltages in the frontal and hori) OR 3.2/0.4 [15/1412] | hf_noise 9/1412 | False | hf_noise 9/1412 |
| label_negative/raw/kmeans_10 | 7 | 452 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 7238.5/3318.3 [190/452] | edge_hold 1/452; flat_lead 71/452; near_flat_lead 1/452; identity_violation 3/452; amplitude_outlier 10/452; isolated_spike 14/452; contact_loss 18/452 | False | flat_lead 71/452; amplitude_outlier 10/452; isolated_spike 14/452; contact_loss 18/452 |
| label_negative/raw/kmeans_10 | 8 | 347 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 9599.1/3171.0 [161/347] | adc_rail 3/347; identity_violation 3/347; amplitude_outlier 2/347 | True |  |
| label_negative/raw/kmeans_20 | 11 | 834 | sinus_tachycardia 0.00 | snomed:10370003 | scp:axis:RAD (heart_axis = RAD) OR 3.4/9.5 [5/834] | missing_samples 2/834; adc_rail 4/834; identity_violation 1/834; baseline_step 41/834 | False | baseline_step 41/834 |
| label_negative/raw/kmeans_20 | 13 | 680 | sinus_tachycardia 0.00 | scp:PACE | scp:NORM (normal ECG) OR 2.9/1.0 [379/680] | hf_noise 2/680 | True |  |
| label_negative/raw/kmeans_20 | 14 | 337 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 9599.1/1508.3 [161/337] | adc_rail 3/337; identity_violation 3/337; amplitude_outlier 2/337 | True |  |
| label_negative/raw/kmeans_20 | 15 | 321 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 10661.0/6765.0 [186/321] | identity_violation 2/321; amplitude_outlier 1/321; isolated_spike 6/321 | False | isolated_spike 6/321 |
| label_negative/raw/kmeans_40 | 25 | 399 | sinus_tachycardia 0.00 | snomed:10370003 | scp:axis:RAD (heart_axis = RAD) OR 4.5/7.6 [3/399] | missing_samples 1/399; adc_rail 2/399; identity_violation 1/399 | True |  |
| label_negative/raw/kmeans_40 | 30 | 195 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 4464.6/3736.8 [88/195] | identity_violation 1/195; amplitude_outlier 1/195 | True |  |
| label_negative/raw/kmeans_40 | 32 | 188 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 3317.5/120.7 [98/188] | adc_rail 1/188; identity_violation 1/188 | True |  |
| label_negative/raw/kmeans_40 | 35 | 150 | sinus_tachycardia 0.00 | scp:NORM | scp:CLBBB (complete left bundle branch block) OR 498.1/990.5 [65/150] | adc_rail 2/150; identity_violation 2/150; amplitude_outlier 2/150 | True |  |
| label_negative/raw/kmeans_40 | 36 | 125 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 11521.7/282.3 [102/125] | identity_violation 1/125 | True |  |
| label_negative/raw/kmeans_80 | 62 | 118 | sinus_tachycardia 0.00 | snomed:106068003 | snomed:106068003 (atrial rhythm) OR 189.0/108.2 [46/118] | hf_noise 1/118 | True |  |
| label_negative/raw/kmeans_80 | 64 | 111 | sinus_tachycardia 0.00 | snomed:713422000 | snomed:713422000 (atrial tachycardia) OR 232.9/102.0 [19/111] | adc_rail 1/111; identity_violation 1/111; amplitude_outlier 1/111; mains_interference 2/111; baseline_step 9/111 | False | baseline_step 9/111 |
| label_negative/raw/kmeans_80 | 65 | 111 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 8623.7/379.9 [86/111] | identity_violation 1/111 | True |  |
| label_negative/raw/kmeans_80 | 68 | 89 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 1261.5/62.8 [44/89] | mains_interference 1/89 | True |  |
| label_negative/raw/kmeans_80 | 72 | 57 | sinus_tachycardia 0.00 | scp:NORM | snomed:164909002 (left bundle branch block) OR 27.2/30.5 [4/57] | contact_loss 1/57 | True |  |
| label_negative/raw/hdbscan | 0 | 14732 | sinus_tachycardia 0.00 | snomed:426761007 | scp:NORM (normal ECG) OR 69.9/169.4 [6253/14732] | reversal_la_ra 2/14732; reversal_neutral 1/14732 | True |  |
| label_negative/residual/kmeans_10 | 1 | 2112 | sinus_tachycardia 0.00 | scp:CLBBB | scp:SR (sinus rhythm) OR 3.1/4.7 [1031/2112] | reversal_la_ra 1/2112 | True |  |
| label_negative/residual/kmeans_10 | 2 | 2065 | sinus_tachycardia 0.00 | scp:CLBBB | scp:NORM (normal ECG) OR 5.4/3.7 [894/2065] | missing_samples 4/2065; adc_rail 8/2065 | False | adc_rail 8/2065 |
| label_negative/residual/kmeans_10 | 4 | 1849 | sinus_tachycardia 0.00 | scp:CLBBB | scp:axis:LAD (heart_axis = LAD) OR 5.1/4.9 [299/1849] | reversal_la_ra 1/1849 | True |  |
| label_negative/residual/kmeans_10 | 6 | 1705 | sinus_tachycardia 0.00 | scp:CLBBB | scp:axis:AXL (heart_axis = AXL) OR 15.6/3.5 [18/1705] | edge_hold 1/1705; hf_noise 5/1705 | False | hf_noise 5/1705 |
| label_negative/residual/kmeans_10 | 7 | 1379 | sinus_tachycardia 0.00 | scp:CLBBB | scp:LVOLT (low QRS voltages in the frontal and hori) OR 4.1/1.0 [16/1379] | edge_hold 1/1379; hf_noise 7/1379 | False | hf_noise 7/1379 |
| label_negative/residual/kmeans_10 | 8 | 410 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 11268.5/7729.4 [162/410] | edge_hold 1/410; adc_rail 2/410; flat_lead 28/410; identity_violation 2/410; amplitude_outlier 2/410; contact_loss 7/410 | False | flat_lead 28/410; contact_loss 7/410 |
| label_negative/residual/kmeans_10 | 9 | 362 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 8764.9/6765.0 [189/362] | flat_lead 22/362; identity_violation 2/362; amplitude_outlier 1/362; isolated_spike 8/362; contact_loss 4/362 | False | flat_lead 22/362; isolated_spike 8/362 |
| label_negative/residual/kmeans_20 | 4 | 1073 | sinus_tachycardia 0.00 | scp:CLBBB | scp:axis:AXL (heart_axis = AXL) OR 13.3/6.5 [13/1073] | edge_hold 1/1073 | True |  |
| label_negative/residual/kmeans_20 | 6 | 1014 | sinus_tachycardia 0.00 | scp:CLBBB | scp:NORM (normal ECG) OR 2.5/4.3 [453/1014] | adc_rail 3/1014; hf_noise 3/1014 | True |  |
| label_negative/residual/kmeans_20 | 13 | 753 | sinus_tachycardia 0.00 | scp:PACE | scp:LVOLT (low QRS voltages in the frontal and hori) OR 3.9/0.7 [10/753] | hf_noise 4/753 | True |  |
| label_negative/residual/kmeans_20 | 15 | 345 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 17429.7/4977.5 [162/345] | adc_rail 2/345; identity_violation 2/345; amplitude_outlier 1/345 | True |  |
| label_negative/residual/kmeans_20 | 17 | 198 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 1354.5/1925.2 [83/198] | identity_violation 1/198; amplitude_outlier 1/198; isolated_spike 4/198 | True |  |
| label_negative/residual/kmeans_40 | 33 | 215 | sinus_tachycardia 0.00 | snomed:106068003 | snomed:106068003 (atrial rhythm) OR 190.6/134.8 [68/215] | adc_rail 1/215; identity_violation 1/215; amplitude_outlier 1/215; mains_interference 3/215; hf_noise 2/215; contact_loss 3/215 | True |  |
| label_negative/residual/kmeans_40 | 34 | 192 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 7847.9/3152.4 [81/192] | identity_violation 1/192; amplitude_outlier 1/192 | True |  |
| label_negative/residual/kmeans_40 | 35 | 184 | sinus_tachycardia 0.00 | scp:CLBBB | scp:CLBBB (complete left bundle branch block) OR 10202.7/145.4 [99/184] | adc_rail 1/184; identity_violation 1/184 | True |  |
| label_negative/residual/kmeans_40 | 36 | 162 | sinus_tachycardia 0.00 | scp:NORM | scp:CLBBB (complete left bundle branch block) OR 407.4/990.5 [65/162] | adc_rail 2/162; identity_violation 2/162; amplitude_outlier 2/162 | True |  |
| label_negative/residual/kmeans_40 | 37 | 132 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 13266.6/282.3 [110/132] | identity_violation 1/132 | True |  |
| label_negative/residual/kmeans_80 | 46 | 208 | sinus_tachycardia 0.00 | snomed:426783006 | snomed:426783006 (sinus rhythm) OR 4.9/3.1 [129/208] | missing_samples 1/208; adc_rail 2/208; identity_violation 1/208; amplitude_outlier 2/208; isolated_spike 4/208 | True |  |
| label_negative/residual/kmeans_80 | 59 | 139 | sinus_tachycardia 0.00 | snomed:106068003 | snomed:106068003 (atrial rhythm) OR 285.0/219.4 [61/139] | mains_interference 2/139; hf_noise 2/139 | True |  |
| label_negative/residual/kmeans_80 | 65 | 97 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 3139.1/735.8 [41/97] | identity_violation 1/97; amplitude_outlier 1/97; isolated_spike 2/97 | True |  |
| label_negative/residual/kmeans_80 | 69 | 76 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 5932.9/379.9 [67/76] | identity_violation 1/76 | True |  |
| label_negative/residual/kmeans_80 | 74 | 49 | sinus_tachycardia 0.00 | scp:PACE | scp:PACE (normal functioning artificial pacemaker) OR 3047.1/111.1 [40/49] | isolated_spike 1/49 | True |  |
