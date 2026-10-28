# NBR-03 results: on PTB-XL fold 9 the territory signal replicates, but weakly and not beyond replaced energy

28 October 2026, under the committed [protocol](nbr03-ptbxl-replication-protocol-2026-10-28.md) (`544fc55`;
code `f97f29d`). The branch is stacked on the NBR-02 branch (pull request #79), which is stacked on NBR-01. Every
number comes from `outputs/nbr03_ptbxl_replication_20261028/` (`analyze/results.json`, `analyze/counts.csv`,
`score/integrity.json`, `parity/parity.json`, `checks/checks.json`, `select/selection.json`,
`prepare/prepare.csv`, `profile/profile.json`, `encode_*/encode.json`, `export/export.json`). Intervals are
95% whole-patient bootstraps, 2,000 draws, seed 20261028, all statistics on the same draws. Unless stated,
scores are `all_age_blend` probabilities with NBR-01's P0 counterpart.

## Answer

- Primary passed: localization replicates. The territory contrast (inferior against anterior MI) is +0.098
  [+0.052, +0.143] on 307 territory patients, against +0.306 on SPH.
- Key qualifier not met. The fold-9 energy map gives +0.060 [+0.036, +0.086], and model minus energy is
  +0.038 [-0.004, +0.078]. On PTB-XL the all-age blend's lead evidence is not shown to go beyond where the
  normal counterpart changes the signal most.
- The PTB-only default, which is in-domain here (trained on PTB-XL folds 1-7), localizes clearly and beyond
  energy: +0.211 [+0.161, +0.257], +0.150 [+0.103, +0.195] beyond energy.
- Lateral (13 patients, secondary) has a part contrast of +0.089 [-0.007, +0.189]; it cannot be judged at this
  size. With all three territories the contrast is +0.095 [+0.047, +0.140], +0.044 [+0.004, +0.081] beyond
  energy.
- The wave and BBB lead contrasts replicate in direction. ST-T statements load the ST-T (+0.180 beyond energy),
  but QRS evidence in BBB is not beyond energy here (+0.008 [-0.088, +0.097]).

## Primary endpoint

| Score | Territory contrast | Energy map | Model minus energy |
| --- | --- | --- | --- |
| `all_age_blend` (primary) | +0.098 [+0.052, +0.143] | +0.060 [+0.036, +0.086] | +0.038 [-0.004, +0.078] |
| `all_age` HGB | +0.086 [+0.044, +0.124] | +0.060 | +0.025 [-0.015, +0.063] |
| `all_age` xECG | +0.104 [+0.049, +0.159] | +0.060 | +0.044 [-0.008, +0.094] |
| `ptb_baseline_blend` (in-domain) | +0.211 [+0.161, +0.257] | +0.060 | +0.150 [+0.103, +0.195] |
| `ptb_baseline` HGB | +0.189 [+0.143, +0.233] | +0.060 | +0.129 [+0.082, +0.172] |
| `ptb_baseline` xECG | +0.227 [+0.172, +0.282] | +0.060 | +0.167 [+0.119, +0.216] |

Decision under the protocol: localization replicates (lower bound above 0); localization beyond replaced
energy is not shown (paired lower bound -0.004).

Per-territory parts (all three territories; each territory against the other two):

| Map | Inferior (II, III, aVF) | Anterior (V1-V4) | Lateral (I, aVL, V5, V6) |
| --- | --- | --- | --- |
| `all_age_blend` | +0.136 [+0.082, +0.186] | +0.060 [+0.008, +0.110] | +0.089 [-0.007, +0.189] |
| `ptb_baseline_blend` | +0.252 [+0.194, +0.306] | +0.160 [+0.107, +0.216] | +0.083 [-0.044, +0.210] |
| Energy map | +0.028 [+0.007, +0.049] | +0.090 [+0.058, +0.123] | +0.036 [-0.038, +0.108] |

The energy map favours the precordial leads, so it helps the anterior part and not the inferior one. For the
all-age blend the anterior part (+0.060) is below the energy map's (+0.090), while the inferior part (+0.136)
is well above it.

## Secondary endpoints

| `all_age_blend` | Model | Energy map | Model minus energy |
| --- | --- | --- | --- |
| Wave contrast | +0.196 [+0.136, +0.255] | +0.102 [+0.082, +0.121] | +0.094 [+0.036, +0.150] |
| QRS share, BBB against other abnormal | +0.130 [+0.029, +0.222] | +0.122 [+0.097, +0.146] | +0.008 [-0.088, +0.097] |
| ST-T share, stt against other abnormal | +0.262 [+0.203, +0.318] | +0.082 [+0.054, +0.110] | +0.180 [+0.126, +0.233] |
| BBB lead contrast | +0.414 [+0.324, +0.498] | -0.083 [-0.128, -0.037] | +0.497 [+0.390, +0.591] |
| Top-1 territory hit (chance 0.286, group-weighted) | 0.423 [0.368, 0.477] | 0.423 [0.369, 0.483] | +0.000 [-0.080, +0.083] |
| Top-3 territory hit (chance 0.673, group-weighted) | 0.691 [0.638, 0.741] | 0.554 [0.497, 0.610] | +0.137 [+0.057, +0.214] |

`ptb_baseline_blend`: wave +0.305 [+0.249, +0.356] (+0.203 beyond energy), BBB lead +0.484 [+0.389, +0.571],
top-1 hit 0.492, top-3 hit 0.785. Selectivity (abnormal over normal mean absolute whole-lead change, as in
NBR-02): `all_age_blend` 0.982 [0.851, 1.141] (SPH 0.935), `ptb_baseline_blend` 1.198 [1.047, 1.372].

Flips at the PTB thresholds (fold-8 calibration, patient-disjoint from fold 9; `all_age_blend` 0.90074), share
of flagged ECGs whose flag one replacement removes:

| Group | Flagged | Any grid cell | Any lead | Any component | Full | Self-swap grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inferior | 92 / 173 | 0.163 | 0.250 | 0.543 | 0.913 | 0.033 |
| anterior | 113 / 134 | 0.150 | 0.204 | 0.372 | 0.876 | 0.009 |
| lateral | 7 / 13 | 0.429 | 0.429 | 0.714 | 1.000 | |
| lbbb | 38 / 39 | 0.000 | 0.026 | 0.079 | 0.763 | 0.000 |
| rbbb | 23 / 23 | 0.174 | 0.348 | 0.391 | 0.826 | 0.000 |
| stt | 86 / 180 | 0.233 | 0.279 | 0.698 | 0.814 | 0.081 |
| all abnormal (NBR-01 pool, without lateral) | 352 / 549 | 0.159 [0.122, 0.197] | 0.224 [0.182, 0.269] | 0.466 [0.416, 0.516] | 0.855 [0.816, 0.890] | 0.031 [0.014, 0.051] |
| normal | 7 / 300 | 0.286 | 0.286 | 0.429 | 0.714 | 0.000 |

The PTB threshold is high (0.90), so more flagged ECGs sit near it: even the self-swap control removes 3.1% of
flags (0.6% on SPH).

Full-replacement sanity: the mean abnormal score falls from 0.825 to 0.359 (drop 0.466 [0.441, 0.489];
operator check passed), and the normals' from 0.247 to 0.080 (drop 0.168 [0.144, 0.191]). Self-swap artefact:
mean absolute change per grid cell 0.0029 against 0.0099 for the counterpart on abnormal ECGs, 0.0036 against
0.0111 on normals; per whole lead 0.0053 against 0.0241 and 0.0068 against 0.0246. The self-swap territory
contrast is +0.039 [-0.020, +0.100].

## Integrity

| Check | Result |
| --- | --- |
| Data rules | fold 10 absent from the table and refused by the reader; fold 9 unseen by both models (all-age PTB rows folds 1-7, `ptb_baseline` fit folds 1-7, threshold calibration fold 8), with both tables matching their receipts; zero fold-9 patients in any training, calibration or LocBench fit set |
| 1. NBR-01's SPH primary recomputed by the NBR-03 code | +0.3056275517940945, exact |
| 2. P0 counterparts rebuilt on 100 SPH records | 0.0 mV |
| 3. Unreplaced fold-9 scores against the saved `ptb_predictions.npz` | largest difference 0.0 for all six scores on 862 ECGs; xECG features bit-identical for 862 of 862, so the pre-registered parity fallback was not needed; decisions identical; PTB thresholds replay exactly |
| 4. Variant SHA-256 between preparation and encoding | all 862 matched |

## Run

Sample (labels only): inferior 173, anterior 134, lateral 13, lbbb 39, rbbb 23, stt 180, normal 300; 862 ECGs,
one per patient, none excluded by 062. Preparation 97 s with 16 workers; encoding 431 ECGs on each GPU in 425
and 457 s, in parallel under the two locks; bootstrap 41 s.

## Deviations and disclosures

1. The protocol expected 873 ECGs. The draw gives 862, because a patient already drawn into an earlier group
   is not drawn again (lateral 13 instead of 14, lbbb 39 instead of 42, rbbb 23 instead of 26, stt 180
   instead of 184). The rule is NBR-01's and was not changed; only the expected counts were wrong.
2. The `checks` and `profile` stages ran before the protocol was committed; they read no score.
3. The brief described `ptb_baseline` as fit on folds 1-8; its receipts show folds 1-7, with fold 8 used
   only for its thresholds.

## What it means

- The direction of NBR-01's territory finding holds on a second, independently labelled dataset, but the
  all-age blend's effect is a third of SPH's and cannot be separated from the energy map. The SPH finding
  that the model localizes beyond replaced energy (+0.190) therefore does not generalize as strongly to
  PTB-XL for the all-age blend.
- On PTB-XL the in-domain PTB-only blend localizes about twice as strongly as the all-age blend and beyond
  energy. Localization appears to depend on how close the evaluation data are to the training data, which
  is a hypothesis these numbers suggest; no paired model comparison was declared.
- Inferior MI evidence is localized clearly by both models; anterior MI evidence is weak for the all-age
  blend; lateral evidence cannot be judged with 13 patients.
- For the page: lead attributions on PTB-XL-like data should carry the same caution as on SPH, and more for
  anterior findings.

## Limitations

- PTB-XL fold 9 was evaluated before (PR 74, HEADS-01): development data, not a fresh test.
- Territory groups use statements of any likelihood, as the project's label does; partly automatic reports.
- Lateral has 13 patients, and fold 10 stays closed, so a powered lateral test is not possible within PTB-XL
  development data.
- Attribution is model-specific and on the probability scale.

## Page export

`outputs/nbr03_ptbxl_replication_20261028/examples.json` (6,184,850 bytes; validated): NBR-01's final schema
with `dataset: "ptbxl_fold9"`, an `age_note` (ages above 89 are exported as 90) and the PTB thresholds in
`models`. 16 examples drawn within groups by NBR-01's rule: 3 inferior, 3 anterior, 2 lateral, 2 RBBB, 2 LBBB,
2 ST-T and 2 normal. `summary.statistics` holds 369 statistics, including the `all_territories/`, `parts/` and
lateral keys.

## Follow-ups (added to the backlog)

- `nbr_lateral_territory_ptb` is marked done: lateral could only be described.
- `nbr_domain_localization_gap`: a declared paired comparison of the all-age and PTB-only blends' territory
  contrasts on fold 9 and SPH, to test whether localization strength follows training-domain proximity.
