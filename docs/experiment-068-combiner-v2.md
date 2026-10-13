# Experiment 068: learned unit combiner v2

Frozen 13 October 2026 (UTC), before the cross-fit networks were trained, before any 068 map was built from
them and before any LocBench score of a 068 map. Part of the [localization programme](localization-program-2026-10-12.md);
scored on LocBench v1 ([Experiment 061](experiment-061-locbench.md), with amendments 2 and 3). It follows
[Experiment 066](experiment-066-combination.md) and its [results](experiment-066-combination-results.md).

Before this freeze, only the following were read or run:

- the protocols, results reports and code of 061, 065 and 066, and the programme note;
- `fit`-partition metadata counts (the training-set counts below);
- a GPU timing profile of one cross-fit fold (40 training steps, no held-out reading);
- a smoke run of the whole pipeline (1,200 `fit` records, one epoch of 5 steps per fold, 60 training and 16
  scored records): shapes, finiteness and timing only. In that run the 066 `units_learned` map rebuilt from
  this experiment's feature code with 066's saved weights equalled 066's saved map on the 16 scored records
  (largest absolute difference 0.0). That is a map-equality check, not a score.

No ECGAD mark, ECGAD statement label or LocBench score of a 068 map has been read.

## Question

Does the 066 learned unit combiner improve when it is trained on all `fit`-partition records with region
statements, using out-of-fold MIL evidence from a patient-level cross-fit, including records with several
statements, and given a softened MIL lead factor and explicit diagnosis-by-component weights as features?

Motivation from 066: `units_learned` (A2 0.7604) was the best development map, but it was trained on only
289 records (held-out 065 rows with exactly one region statement), and its weights were mostly `textbook`
(+1.25) with a learned QRS-over-ST-T preference. Multiplying by the full MIL lead weights (`factor`) helped
inside QRS and hurt outside it (066 follow-ups 1 and 5).

## MIL cross-fit (train-time evidence)

- Records: the 11,617 `fit`-partition records. Folds: K = 5 patient-level folds from
  `combination066.patient_folds` with seed 66066, the folds of 066's lead-quality classifier, so the MIL
  evidence and the q feature of a training record are both out of fold under the same split.
- Network: the 065 `mil_lse` architecture and training code, by import (`mil065.LeadMIL(49, pooling="lse",
  cross_lead=False)`, `mil065.train` with the 065 `Recipe`: AdamW 1e-3, weight decay 1e-2, batch 64, random
  4,480-sample crops, bf16 autocast, at most 40 epochs, patience 6), the same 49 heads in the same order
  (065 `result.json`) and the same labels (`mil065.statement_labels`) and preprocessing (`mil065.preprocess`).
- Early stopping inside each fold: 065's held-out rule (`mil065.heldout_mask`, 15% of patients, seed 65065)
  applied to the patients of the four training folds. The out-of-fold records are never used for training
  or early stopping.
- One seed, 68068 (`torch.manual_seed` before building each fold's network; NumPy generator for batch order
  and crops), for all five folds.
- Out-of-fold outputs per `fit` record, computed exactly as 066 computed them from the 065 ensemble, but from
  the single fold network that did not see the record: bag logits, per-head lead means of the evidence over
  the scored steps (`combination066.lead_means`), rhythm-head evidence, and the combined map
  (`mil065.combined_map`, sum over the 48 non-NORM heads).
- GPU: GPU0 with the shared lock (`ecg_experiment.gpu.gpu_lock`), one fold per lock hold, 6 GB process cap.
  Profile (fold 0, 7,905 training rows): 0.115 s per step, 2.5 GB peak, about 14 s per epoch; with 065's
  best epochs (11-20) about 5 minutes per fold and under 40 minutes for the cross-fit.
- Checks (reported, in `result.json`): out-of-fold macro AUROC over heads with at least 10 positives and 10
  negatives, on all 11,617 records and on the 1,766 rows that were 065's held-out rows; for comparison the
  065 three-seed ensemble on those rows and 065's per-seed held-out values (0.9440, 0.9436, 0.9439). The
  out-of-fold value is expected to be near, and slightly below, 065's per-seed values (each fold network
  trains on 68% of `fit` instead of 85%).

## Test-time evidence and the calibration gap

Scored records (ECGAD development and the `evaluation` records LocBench needs) keep the 065 three-seed
ensemble evidence, exactly as in 066, so the scored inputs of the 066 features are identical to 066's. They
are read from 066's work folder (`/tmp/ecg066-work/full/mil_*.npy`), whose bag logits are checked against
065's saved `bags_*.npz` (largest difference at most 1e-3; 066 found them identical), and whose use is
validated by the 066 rebuild check below. So train-time MIL features come from the single-seed cross-fit
networks and test-time MIL features from the 065 three-seed ensemble. The gap is reported: the distribution
of non-NORM bag probabilities (mean, median, 90th and 99th percentiles, mean per-record sum and maximum) for
(a) cross-fit out-of-fold on the training records, (b) the 065 ensemble on the scored records, and on the
same records (065's held-out rows) for (c) the cross-fit out-of-fold and (d) the 065 ensemble, with the mean
and mean absolute logit difference (c minus d) and their correlation. Every continuous unit feature is
z-scored within the record, which removes a record-level scale shift but not a change in the ranking of
heads.

## Training set

- Every `fit`-partition record with at least one `REGION_TABLE` statement at likelihood at least 50 (the
  partition's `region_statements`), of any rhythm: 3,517 records of 2,983 patients. 2,288 carry one
  statement, 967 two, 217 three, 43 four and 2 five. 467 would fail the region-set rhythm rule (a rhythm
  statement outside SR, SARRH, SBRAD, STACH); they are included, unlike in 066, because this experiment asks
  about all region records. The count is reported.
- Records per LocBench family (a record counts once in each of its families): anterior MI 1,185, IMI 892,
  LAFB 842, ischemia 572, 1AVB 417, CRBBB 297, CLBBB 294, lateral MI 188, ST injury 147, long QT 59, WPW 45.
- Weak labels: a unit is positive if it lies in the expected leads and components of any of the record's
  statements (`REGION_TABLE`, PR = P and PQ): the union of the 066 weak labels over statements. Records
  without units, or whose units are all positive or all negative, are dropped and counted.
- `fit` clean normals are excluded from the ranking objective (a within-record objective needs positives).
  No clean normal carries a region statement; this is asserted.

## Units and features

Units are 066's: (beat, lead, component) for the five components P, PQ, QRS, ST, T of every beat delineated
by `combination066.delineated_components`, kept if non-empty. Unit features are means over the unit's
samples, built with 066's code (`combination066.factor_sum`, `textbook_sum`, `unit_table`, `unit_means`):

- The 24 features of 066, unchanged: `factor`, `textbook`, `mil`, `normative` (062 `normative_v1`),
  `residual` (`residual056`), `amplitude` (log1p for the last three), each z-scored within the record; `q`
  (066 lead quality; out-of-fold q for training records, 066's final model for scored records); one-hot
  component (5) and lead (12).
- Softened lead factor `factor_soft`: `factor` with the MIL lead weight `L_k` replaced by `(1 - a) + a L_k`
  for every table head (rhythm heads unchanged), divided by its record maximum, z-scored within the record.
  `a` is chosen from {0.25, 0.5, 0.75} (below). At a = 1 it is `factor`; at a = 0 it ignores lead evidence.
- Diagnosis-by-component weights: the z-scored `textbook` feature times each component one-hot (5
  features), so the weight on the diagnosed region can differ between P, PQ, QRS, ST and T. With the
  component one-hots (a per-component time weight) this is the explicit component-time weighting.
- Out of sample for the training records: the MIL features (cross-fit), q (066 out-of-fold), 062
  `normative_v1` (fitted on the 2,666 `fit` clean normals only, which hold no training record), `residual056`
  (061's `ub_references` from the `fit` clean normals set only the coverage mask) and amplitude (label-free).
  No 063 or 064 feature is used.

## Models (declared maps)

All linear models use 066's objective: within-record listwise softmax with several positives,
`loss_r = logsumexp_u(s_u) - mean_{u positive}(s_u)`, averaged over records, plus `L2 ||w||^2`, no
intercept, L-BFGS from zero (a vectorized form of `combination066.listwise_loss`, tested equal).

| Map | Features | Selection | Role |
| --- | --- | --- | --- |
| `units_v2` | 066's 24 + `factor_soft` + 5 interactions (30) | (a, L2) jointly | **primary** |
| `units_v2_nolead` | `units_v2` minus every MIL lead feature (`factor`, `factor_soft`, `mil`): 27 | L2 | ablation |
| `units_v2_hgb` | `units_v2` features at the selected a | none (fixed) | nonlinear secondary |
| `units_v1_data` | 066's 24 | L2 | diagnostic, never a candidate |

- Selection: patient-grouped five-fold cross-validation on the training records only, with the cross-fit
  folds; criterion the mean held-out listwise loss (data term) over records; L2 from {1e-4, 1e-3, 1e-2, 1e-1};
  ties go to the larger L2. The final model is refitted on all training records. ECGAD is never used.
- `units_v2_nolead` keeps `textbook`, which uses the MIL bag probabilities (the diagnosis) but no lead
  evidence. It asks whether the lead evidence earns its place.
- `units_v2_hgb`: `sklearn` `HistGradientBoostingClassifier` (max depth 3, learning rate 0.1, 200
  iterations, L2 1.0, at least 100 units per leaf, no early stopping, seed 68068), pointwise logistic loss
  on the weak labels with unit weights `1 / units in the record`, score = decision function. No ranking-loss
  library (LightGBM, XGBoost) is installed and the dependencies are pinned, so a pointwise loss replaces the
  ranking loss of the brief.
- `units_v1_data`: 066's features on this experiment's training set, so that the change of training data
  and the change of features can be separated. It is scored but is not a candidate.
- Reported from the training records (fit partition, not LocBench): the CV grid, the chosen settings, the
  held-out within-record weak-label AUROC of each model, the weights, the training counts by family and
  statement, and the within-record weak-label AUROC of 066's saved weights on the training records outside
  066's own 289.

Maps: unit scores painted as in 066 (`combination066.units_map`: maximum where units overlap, record minimum
elsewhere, min-max scaled within the record, plus `1e-3 x amplitude`), float32 (n, 12, 5000), for every
record `locbench.scored_ecg_ids()` returns (ECGAD development and `evaluation`). No ECGAD holdout record is
mapped.

## Primary comparison and decision rule

- Primary: paired A2 (within-record point AUROC, the 92 marked ECGAD development records) of `units_v2`
  minus `qrs_prior`, 2,000 whole-patient bootstrap draws, seed 61061, with
  `scripts/experiments/run_locbench_score.py --compare` against a copy of 061's `qrs_prior` tables.
- Decision: `units_v2` **localizes** if the 95% interval is above 0; **matches** if it contains 0;
  **below** if it is below 0.
- Secondary paired A2, same draws: `units_v2` minus 066 `units_learned`; minus 066 `textbook`; minus
  `units_v2_nolead`. Also reported: minus `units_v2_hgb` and minus `units_v1_data`, and every map against
  `qrs_prior`, `units_learned`, `textbook` and `residual056`.
- Reported for every map: every LocBench metric (A1-A6, B1-B3 macro, per family and strict, C1, C2, D1),
  `A2_qrs` and `A2_nonqrs` (`scripts/experiments/run_locbench_strata.py`, `qrs_prior` reference first) and
  `C1_std` (`scripts/experiments/run_locbench_noise.py`).
- Independence: every 068 map is trained on `REGION_TABLE` weak labels, so its B readings are not
  independent tests; only the A (ECGAD) and C (noise) readings are.
- 067 (holdout confirmation, a separate experiment run later) adds `units_v2` to its candidates if the
  development A2 of `units_v2` exceeds `units_learned`'s 0.7604 (0.7603613741425232); otherwise 067 keeps
  `units_learned`. No other 068 map can become a candidate.

## Reference tables

Both secondary CLIs write into every folder they are given, so the reference tables are copied into
`outputs/experiment068_combiner_v2/ref/` first (061 `qrs_prior` and `residual056`; 066 `units_learned` and
`textbook`, without their `strata.*` and `noise_std.*`), and every CLI receives the copies. No 061, 065 or
066 file is written.

## Integrity checks (in `result.json`; a failure stops the scoring)

- Bar: rescore the saved 061 `qrs_prior` ECGAD maps with `locbench.ecgad_record_metrics`; every per-record A2
  within 1e-12 of its `ecgad.csv`, and the mean within 1e-12 of 0.7419383358406554.
- 066 `units_learned`: (1) rescore its saved ECGAD maps and match its `ecgad.csv` per record within 1e-12
  and its A2 0.7603613741425232; (2) rebuild it from this experiment's features of the scored records with
  066's saved weights, and match 066's per-record A2 within 1e-12 (the largest map difference is reported).
- Test-time MIL: 066's recomputed 065 bag logits match 065's saved bags (at most 1e-3) and cover every scored
  record.
- Cross-fit: out-of-fold macro AUROC reported beside 065's held-out values (above).
- Data separation, asserted: no ECGAD record and no `evaluation` record enters the cross-fit, the combiner or
  any selection; every training record is a `fit` record and none is a `fit` clean normal; no ECGAD holdout
  record is mapped. The ECGAD holdout half (only `ecgad.load_split` development records are mapped), folds 9
  and 10 outside ECGAD, the Challenge test or calibration groups and the EchoNext test set are never read.
- Every map is finite.

## Outputs

`outputs/experiment068_combiner_v2/`: `result.json` (inputs and hashes, integrity, GPU profile, cross-fit
metrics and calibration gap, training counts, CV grids, weights, every reading, decisions),
`gpu_profile.json`, `crossfit_fold*.json`, `crossfit_summary.json`, `features_receipt.json`,
`fit_receipt.json`, `locbench/<map>/` (summary, tables, strata, noise), `ref/`, `run.log`. Large
intermediates (preprocessed signals, cross-fit weights and evidence, unit features, maps) live in
`/tmp/ecg068-work` and are not kept in `outputs/`. Smoke outputs go to `outputs/experiment068_combiner_smoke/`
and are not results. Code: `ecg_experiment/combiner068.py`, `scripts/experiments/run_combiner068.py`,
`tests/test_combiner068.py`. The running agent writes `docs/experiment-068-combiner-v2-results.md` from
`result.json`.
