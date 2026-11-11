# MLC-02 Stage 3 code: selection record, operating points, test predictions, evaluation

10 November 2026. This document describes the code of Stage 3 of the
[MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md) and its dry runs. **No test-role file was
opened.** No prediction, label or metric of PTB-XL fold 10, the SPH evaluation half or the SPH calibration half
was computed. Every number below comes from validation records, and the selection table is provisional
because candidate F was still training.

| File | Content |
|---|---|
| `ecg_experiment/mlc02_stage3.py` | E's family rule, the selection record and its Markdown document, the operating points, the guards |
| `ecg_experiment/mlc02_stage3_predict.py` | `test-predict` and its validation dry run, through one code path |
| `ecg_experiment/mlc02_stage3_evaluate.py` | `evaluate` from saved test predictions only, and its validation dry run |
| `ecg_experiment/mlc02_calibration.py` | Logistic recalibration by damped Newton (see "Calibration fits") |
| `scripts/experiments/run_mlc02_stage3.py` | Command line: `select`, `fix`, `test-predict`, `evaluate` |
| `tests/test_mlc02_stage3.py` | 17 tests |

Every metric, the selection metric, the tie rule, the 90% specificity threshold, the bootstrap and the paired
comparison come from the tested scoring library (`mlc02_evaluation`, `mlc02_bootstrap`); the new modules
arrange calls to it. The only new estimator is the damped Newton fit.

## Commands for the real run, in order

Run from a worktree with `data`, `outputs`, `.venv` and `third_party` linked, with the code committed (every
real step checks that no tracked file is modified). `<wt>` is the worktree.

1. Merge F's branch (`research/mlc02-finetune-20261110`) and wait for F's `finalize` to write
   `candidates/F/model.json`, `candidate.json` and `validation__seeds-20261110-20261111-20261112.npz`.
   `select` refuses an unfinished F (`model.json` without `candidate.json`), a smoke run, or any candidate
   whose recomputed selection metric differs from the stored one.
2. Select, without `--allow-missing`:

   ```
   OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=<wt> .venv/bin/python \
       -m scripts.experiments.run_mlc02_stage3 select
   ```

   It writes `outputs/mlc02_condition_models_20261110/stage3/selection_record.json`,
   `stage3/predictions/E/validation__seeds-<union>.npz`, `stage3/receipts/select.json` and
   `docs/mlc02-selection-record-2026-11-10.md`.
3. Commit the record document by path:
   `git add docs/mlc02-selection-record-2026-11-10.md && git commit -m "Add the MLC-02 selection record"`.
4. Fix the operating points:

   ```
   OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=<wt> .venv/bin/python \
       -m scripts.experiments.run_mlc02_stage3 fix
   ```

   It writes `stage3/operating_points.json` and `stage3/receipts/fix.json`.
5. Optional, recommended: dry run of `test-predict` on validation with every candidate, F included, on the
   GPU (all records):

   ```
   CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=<wt> .venv/bin/python \
       -m scripts.experiments.run_mlc02_stage3 test-predict --dry-run-role validation --device cuda \
       --scratch /var/tmp/janetrivera/mlc02/stage3_dry_run
   ```

6. Test predictions, once:

   ```
   CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=<wt> .venv/bin/python \
       -m scripts.experiments.run_mlc02_stage3 test-predict --device cuda --i-am-the-final-run
   ```

   On GPU 1 add `--gpu-lock /tmp/ecg_project_gpu1.lock` (the default lock is GPU 0's,
   `/tmp/ecg_project_gpu.lock`). It writes `stage3/predictions/<id>/<role>__seeds-<...>.npz` for every
   candidate and the three roles, and `stage3/test_predictions_receipt.json`.
7. Evaluate:

   ```
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=<wt> .venv/bin/python \
       -m scripts.experiments.run_mlc02_stage3 evaluate --workers 8
   ```

   It writes `stage3/evaluation/evaluation.json`, `conditions.csv`, `macro.csv` and `comparisons.csv`.

## Guards

| Step | Refuses when |
|---|---|
| `select` | `selection_record.json` exists and `--overwrite-before-tests` is not given; any test prediction file exists anywhere under `outputs/mlc02_condition_models_20261110/`, the Stage 3 directory or the candidate directory (always, flag or not); a candidate is absent and not named in `--allow-missing`; a candidate is unfinished, a smoke run, or says it scored a test role; a recomputed selection metric differs from the stored one by more than 1e-12; the code is uncommitted (unless `--uncommitted`, for scratch runs) |
| `fix` | `operating_points.json` exists without `--overwrite-before-tests`; any test prediction file exists; no selection record; the validation labels or a validation file differ from those in the record (SHA-256) |
| `test-predict` | no `--i-am-the-final-run`; any test prediction file exists (it runs once); no selection record; the record is provisional; the record document is outside the repository, untracked, modified, or differs from the text generated from `selection_record.json` (byte for byte, with the JSON's SHA-256 inside); no operating points, or operating points fitted for another record; uncommitted code; a validation file changed; a feature file differs from the Stage 1 receipt; a cache block or record of another role; a cached window that does not reproduce its manifest hash; a model whose seeds or artefact hashes differ from the record or its own index |
| `test-predict --dry-run-role` | any role but `validation`; writes only to `--scratch` and never with `--i-am-the-final-run` |
| `evaluate` | the selection record, the operating points or any test prediction file differs from the SHA-256 pinned by the test-predict receipt; uncommitted code |

A test prediction file is any file named `<test role>__*` at any depth (`test_same_source__`,
`test_external__`, `sph_calibration__`), the `mlc02_data.prediction_path` convention. `test-predict` computes
every prediction in memory and writes files only after all candidates and roles succeeded, so a run that
fails leaves no file and can be repeated; once files exist, the step is closed.

## What each step computes

`select` (validation only):

- loads each candidate's prediction file of all its seeds: `model.json` seeds for the feature candidates and
  F, `settings.json` seeds for S-R;
- recomputes the selection metric with `mlc02_evaluation.selection_metric` and checks it against the value the
  Stage 2 run stored (`candidate.json`, or `results.json` for S-R);
- builds E and adds it to the table;
- applies `mlc02_evaluation.selection_record` (highest selection metric; ties within 0.001, inclusive, go to
  the earliest of P0, L-J, L-C, G-M, H-X, H-J, H-C, H-All, S-R, F, E).

E's rule, as written into the record: a family is a set of candidates with the same input features and a
different head: xECG (P0, H-X, F), ECG-JEPA (L-J, H-J) and ECG-CPC (L-C, H-C); G-M, H-All and S-R are each
their own family. A family's best member is its candidate with the highest validation selection metric (the
earliest in the table on exact equality). Families are ranked by that member's metric, and E is the mean
probability of the best members of the three best families, summed in table order. The 0.001 tie margin does
not apply inside E's construction. E's files carry the union of its members' seeds.

`fix` (validation only), per candidate (E included) and condition:

- the threshold of `mlc02_evaluation.specificity_threshold`: the lowest validation score with specificity at
  least 0.90 on the records where the condition is defined, pooled over sources;
- a Platt calibration fitted on the same records by damped Newton;
- the validation sensitivity and specificity at the threshold, for reference.

`test-predict`, per role (PTB-XL fold 10, SPH evaluation half, SPH calibration half):

- feature candidates: Stage 1 rows from `/var/tmp/janetrivera/mlc02/features/final/` (checked against the
  Stage 1 receipt and the manifest by key, hash, role, source and inclusion), as float64, into
  `mlc02_feature_models.load_candidate` and `predict_loaded`;
- S-R and F: the cached first 10 s from `/var/tmp/janetrivera/mlc02/waveforms/`. Only the blocks of that role
  are memory-mapped, after a role check of every record and block. Every window's `sha256_10s` is checked, and
  the windows go in chunks of 2,048 into `mlc02_scratch_network.predict` (after checking each checkpoint's
  SHA-256 against `settings.json`) or `mlc02_finetune.predict` (imported only when F is in the record; it
  checks its own `model.json` hashes);
- E: the mean of its members, in table order.

`evaluate`, from saved test predictions only. For PTB-XL fold 10 and the SPH evaluation half, with 2,000
patient-cluster resamples per set shared by all candidates (seed 20261110):

- per candidate, raw probabilities, by `mlc02_bootstrap.interval_report`:
  - per-condition and macro AUROC, and average precision with the prevalence;
  - sensitivity and specificity at the validation thresholds;
  - the library's calibration slope, calibration in the large and joint-fit intercept, and the Brier score;
  - the imprecise flag (fewer than 100 positives) and the usability rule (lower AUROC bound at least 0.80
    and at least 100 positives);
  - intervals for every metric;
- the same after the validation Platt calibration, without thresholds;
- beside both, the damped-Newton joint-fit intercept and slope with intervals on the same resamples, and a
  count of where the library's fit gave NaN;
- every candidate minus P0, by `mlc02_bootstrap.paired_comparison`: the macro AUROC difference over the
  conditions with at least 20 positives and 20 negatives, and per-condition differences with Holm over those
  conditions. Only the selected model minus P0 on the SPH evaluation half is labelled confirmatory ("better"
  iff the lower 95% bound is above 0). If the selected model is P0, the record says so and "better" is false;
- the SPH secondary analysis: thresholds and the damped Platt fit refitted on the SPH calibration half and
  applied to the evaluation half. It reports sensitivity, specificity and calibration there, and the local
  minus validation-fitted differences with intervals on the same resamples;
- the likelihood analysis at PTB-XL fold 10 (statements with likelihood below 50 undefined, `labels_lik50`),
  with its own resamples. SPH has no likelihoods;
- audits, point estimates: by sex, by flat or missing lead, one condition against several, and the derived
  multiclass view (top-1 and top-2 under both readings, exact set) on raw probabilities, with top-k after Platt
  beside it.

## Dry runs (validation only)

All dry runs wrote to the scratch directory of this session, `<scratch>` =
`/tmp/claude-218201143/-home-janetrivera-ecg-cardiopathy-detection/23ce6ee8-fd51-4059-a2f6-6ded0be3cc80/scratchpad`.
None touched `outputs/` except for reading Stage 2 files.

### Provisional selection table, without F

PROVISIONAL: F was missing (`select --allow-missing F`). This is not the selection record.

Code `a4d5278`, no modified tracked file. Scratch outputs are in `<scratch>/stage3_committed/`:
`selection_record.json` (SHA-256 `d93454fa...139457`) and `selection-record-provisional.md`. Within-source
macro AUROC per validation source and the selection metric:

| candidate | family | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | selection metric |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | xecg | 0.9654 | 0.9850 | 0.9834 | 0.9538 | 0.9548 | 0.9433 | 0.9643 |
| L-J | jepa | 0.9639 | 0.9812 | 0.9837 | 0.9475 | 0.9543 | 0.9378 | 0.9614 |
| L-C | cpc | 0.9391 | 0.9626 | 0.9705 | 0.9166 | 0.9407 | 0.9114 | 0.9401 |
| G-M | measurements | 0.9676 | 0.9781 | 0.9824 | 0.9587 | 0.9477 | 0.9444 | 0.9632 |
| H-X | xecg | 0.9682 | 0.9865 | 0.9864 | 0.9552 | 0.9613 | 0.9397 | 0.9662 |
| H-J | jepa | 0.9680 | 0.9848 | 0.9855 | 0.9525 | 0.9611 | 0.9360 | 0.9646 |
| H-C | cpc | 0.9547 | 0.9733 | 0.9773 | 0.9312 | 0.9560 | 0.9229 | 0.9526 |
| H-All | all_features | 0.9732 | 0.9881 | 0.9873 | 0.9611 | 0.9668 | 0.9469 | 0.9706 |
| S-R | scratch_waveform | 0.9695 | 0.9851 | 0.9862 | 0.9531 | 0.9642 | 0.9405 | 0.9664 |
| E | ensemble | 0.9745 | 0.9891 | 0.9880 | 0.9628 | 0.9694 | 0.9474 | 0.9719 |

- Family ranking by best member: H-All 0.970563, S-R 0.966440, xECG (H-X) 0.966215, JEPA (H-J) 0.964640,
  G-M 0.963160, CPC (H-C) 0.952567. E is the mean of H-X, H-All and S-R.
- E has the highest metric, 0.971868. The next, H-All at 0.970563, is 0.0013 below, beyond the 0.001 margin,
  so nothing is tied and the provisional selection is E.
- The third family is close: S-R and H-X differ by 0.0002. F belongs to the xECG family, so if F scores above
  0.966215 it replaces H-X in E, and above 0.966440 it moves the xECG family to second place. The members of E
  and the selection can therefore change once F is in.
- All nine stored selection metrics were reproduced exactly from the saved validation files (difference 0).

### Operating points

On the provisional record (`operating_points.json`, SHA-256 `7379ea5d...61fb9`):

- validation specificity at the fitted thresholds is 0.9000-0.9001 for every candidate and condition;
- sensitivity ranges from 0.619 (L-C) and 0.709 (H-C) at the lowest to 0.999;
- no threshold is infinite;
- Platt slopes range from 0.500 (G-M) to 2.058 (S-R), and intercepts from -0.506 to 2.340 (S-R).

### test-predict on validation

Same code path as the real run, role `validation`, CPU (`CUDA_VISIBLE_DEVICES=""`, 8 threads), compared with
the saved validation files of the record. Tolerance 1e-6 per probability.

| candidate | records | largest absolute difference | passed |
|---|---:|---:|---|
| P0, L-J, L-C, G-M, H-X, H-J, H-C, H-All | 15,138 each | 0 | yes |
| S-R | 15,138 | 6.1e-7 | yes |
| E | 15,138 | 2.0e-7 | yes |
| S-R (48 random records, seed 20261110) | 48 | 2.9e-7 | yes |
| E (same 48 records) | 48 | 9.6e-8 | yes |
| F | absent | not run | |

The full run took 126.5 s, 119 s of it for S-R on the CPU, with a peak resident memory of 5.0 GB; the
48-record run took 6.8 s. Files: `dry_run_predict_full/` and `dry_run_predict_48/`
(`dry_run_test_predict.json` and the scratch prediction files).

### evaluate on validation stand-ins

Code `5e40dac`. The validation predictions of the record stand in for test predictions, and the files are
prefixed `DRY_RUN_validation_standin_` (`<scratch>/stage3_committed/dry_run_evaluate/`):

- fold 10 is played by validation PTB-XL fold 9 (2,183 records, 1,942 patients, 14 entering conditions);
- the SPH evaluation half by the Ningbo validation records with an even key hash (3,472 records, 13 entering);
- the SPH calibration half by the odd ones (3,500).

The operating points were fitted on the same validation records, so **no number of this run is a result**. It
shows only that every part of `evaluate` runs end to end on real labels and predictions.

- 2,000 resamples per set, no redraw. Every part was produced: the raw and Platt interval reports, the damped
  calibration reports, the comparisons of all nine other candidates with P0, the confirmatory entry (E minus
  P0, with its top-level `better` flag), the secondary analysis for all ten candidates, the likelihood analysis
  (179 entries undefined at PTB-XL), and the audits.
- Wall time 214.7 s with 8 threads and BLAS limited to one thread; peak resident memory 0.59 GB; JSON 4.1 MB.
  The real sets are about four times larger at SPH (12,698 and 12,883 records) and there will be eleven
  candidates. I expect 12-20 minutes; this is my extrapolation, not a measurement.
- Library calibration on the stand-ins: the point slope is NaN for 26 of 140 entering candidate-condition pairs
  at PTB-XL and 19 of 130 at Ningbo, raw and after Platt alike. 45,282 and 33,656 of the 280,000 and 260,000
  resampled raw fits failed.
- The damped fit gave no NaN point estimate. It gave NaN in 129 resamples, all for left axis deviation at
  PTB-XL (AUROC 0.9992-0.9997), where some resamples separate the classes perfectly and no finite estimate
  exists.

## Calibration fits

`mlc02_evaluation.logistic_fit`, used by the library's `fit_platt` and by its calibration slope and
intercept, takes full Newton steps from the null model (slope 0). On scores that separate the classes well, the
steps oscillate and diverge. For H-X and PVC at validation, the slope goes 1.38, -0.18, 1.82, -2.71, 22.1,
-9,585, then NaN. The fit is reported as not converged although the maximum-likelihood estimate exists:
sklearn's unpenalized fit finds it (slope 0.985 there).

On the validation predictions of this record:

- `fit_platt` fails on 13 of the 140 pooled candidate-condition fits: G-M (PVC, complete RBBB, LAFB, left axis
  deviation), H-X (PVC), S-R (first-degree AV block, complete RBBB, LAFB, left axis deviation) and E (PVC,
  first-degree AV block, LAFB, left axis deviation). `fix` cannot run with it.
- The library's calibration slope is NaN in 80 of the 610 entering (candidate, validation source, condition)
  cells.
- Calibration in the large, a one-parameter fit, converged on all 140 pooled fits.

`mlc02_evaluation.py` is pinned by the Stage 2 receipts and was not changed. `mlc02_calibration.py` maximizes
the same likelihood by Newton steps halved until the log-likelihood does not decrease, starting from the
identity `(a, b) = (0, 1)`.

- On all 750 cells (140 pooled and 610 per source) it converged.
- Where the library converged, the two agree within 2.8e-14.
- It agrees with sklearn within 1.3e-7 (sklearn's tolerance).
- On perfectly separated data it returns NaN, as it should (tested).

The resampled fits start at the observed estimate.

## Tests and checks

- `tests/test_mlc02_stage3.py`, 17 tests:
  - E's family rule: best member per family, F replacing H-X, variants never twice, exact equality to the
    earlier candidate, refusal of E as a member and of fewer than three families;
  - the tie rule with E last;
  - the record end to end on synthetic candidates: table order, E equal to the mean of its members, union seeds,
    exact reproduction of stored metrics, refusal of a metric that does not reproduce;
  - absent, unfinished and smoke candidates;
  - the 90% specificity threshold with undefined labels masked: scores of undefined entries moved to 0 or 1
    leave every threshold unchanged;
  - the damped Platt fit: recovery of `(a, b) = (-0.7, 1.6)`, equality with the library where it converges,
    NaN under separation, intervals around the point estimate;
  - on the real G-M left-axis-deviation validation scores, the library returns NaN and the damped fit matches
    sklearn within 1e-5;
  - the guards: `select` and `fix` refusing to overwrite and closing after a test file appears; `test-predict`
    without the flag, without a record, with a document outside the repository, with a committed document
    that is not the generated one, and after a test file exists; the dry run and the block reader refusing
    test roles before opening anything;
  - an audit hook on `open` showing that `select` and `fix` open no test-role file, feature array or waveform
    block, on synthetic candidates with test-role files planted beside them and on the real outputs;
  - `evaluate_sets` end to end on synthetic sets, including the selected-equals-P0 case.
- `tests/test_mlc02_*.py` together: 110 passed at `5e40dac`, with no GPU visible.
- `ruff check ecg_experiment scripts tests conftest.py`: passed.
- The dry runs above were run by `<scratch>/rerun_committed.sh` (`select --allow-missing F`, `fix`, the two
  `test-predict --dry-run-role validation --device cpu` runs, then `evaluate --dry-run-validation-standin`),
  and the evaluate dry run was repeated at `5e40dac`.

## Audits not implemented

- **Space A beside KEY-04.** The candidates output the 14 space-B conditions only; KEY-04 scored space A (MI,
  STTC, CD, HYP). Space-A metrics need one of two things, fixed before `test-predict`. Either a protocol
  addendum with a rule mapping the 14 outputs to the four groups (for example the highest probability among
  each group's member conditions; the member lists would have to be stated), or space-A heads trained in
  Stage 2. With either, the evaluation is
  `interval_report(labels_a, scores, cluster_resamples(groups, labels_a), conditions=GROUPS_A)`, beside
  KEY-04's 0.9608. The space-A label counts are already in the MLC-01 report.
- **Window mean for SPH records longer than 10 s.** The waveform cache holds the first 10 s only. The analysis
  needs:
  - the full SPH evaluation-half recordings read from the raw files;
  - every consecutive 10 s window passed through each candidate's input pipeline: the three encoders and the
    measurements for the feature candidates, waveforms for S-R and F;
  - the window probabilities averaged per record, then `metrics_table` on the averages.

  This is a Stage 1-sized job (feature extraction per window) and opens SPH raw files.
- Source identification and leave-one-family-out need no test input. They were run in Stage 2
  (`source_identification/`, `lofo/`).

## Departures and doubts

All dated 10 November 2026.

1. **Calibration fits by damped Newton** (previous section). The Platt calibration of `fix` and of the SPH
   secondary analysis uses `fit_platt_damped`, the same estimator as the library's `fit_platt`, which does not
   converge on 13 validation fits. `evaluate` keeps the library's calibration columns as computed and puts the
   damped ones beside them. *Ruling requested:* which slope the results report calls primary. My proposal is
   the damped fit, since it is the maximum-likelihood fit the protocol names, with the library's value beside
   it.
2. **E's rule.** The family definition is the lead's: same input features, different head. H-All also reads
   the xECG features, but its inputs are not the same as H-X's, so it is its own family. Inside E, the best
   member and the family ranking use the exact metric (earlier in the table on exact equality), not the 0.001
   margin. E's validation metric is optimistic twice, since its members are chosen on the same validation
   records it is scored on. Without F, E is the mean of H-X, H-All and S-R.
3. **Where test predictions go.** `stage3/predictions/<id>/<role>__seeds-<...>.npz`, not `candidates/<id>/`,
   so that Stage 3 outputs stay together and Stage 2 directories stay as written. The once-only guard scans the
   whole `outputs/mlc02_condition_models_20261110/` tree, so either location would be caught.
4. **What is committed.** `selection_record.json` lives under `outputs/` (local, not in Git). The committed
   artefact is the Markdown document, which carries the JSON's SHA-256. `test-predict` regenerates the document
   from the JSON and compares it byte for byte. The operating points are not committed. They are bound to the
   record by its SHA-256, `fix` refuses to rewrite them once any test file exists, the test-predict receipt pins
   their SHA-256, and `evaluate` checks it.
5. **Shared files.** The manifest is one file for all roles: `select`, `fix` and the dry runs parse it and keep
   the validation rows, as Stage 2 did. The Stage 1 feature arrays also hold every role: the `test-predict` dry
   run memory-maps them and reads validation rows only. `select` and `fix` never open a feature array, a
   waveform block or a test-role file; a test with a Python audit hook on `open` checks this on synthetic and on
   real outputs. The waveform reader of `test-predict` maps only the blocks of the requested role, and the
   `mlc02_waveforms.load_cache` helper, which maps every block, is not used.
6. **S-R on the CPU.** The S-R dry run ran on the CPU; its saved predictions came from the GPU. The largest
   difference was 6.1e-7 over all 15,138 validation records, within the 1e-6 tolerance. The real run uses the
   GPU, where S-R's own check found a difference of 0.
7. **F was not exercised.** F was absent. Its path imports `ecg_experiment.mlc02_finetune` at run time; that
   module is on F's branch, which must be merged before `select` and `test-predict`. F's `predict` reloads its
   three models for every chunk of 2,048 records, 16 chunks over the three roles. The GPU check asks for
   3 GiB free plus S-R's 1 GiB headroom, a figure I chose, not one profiled for F's evaluation batch (16). Step
   5 of the commands is the place to see F reproduce its validation file before the real run.
8. **What the dry run proves.** It runs the same `role_predictions` as the real run on the validation role and
   compares with the saved files, so it checks the loaders, the model artefacts, the seed means and E. It does
   not open a test-role block or feature row, so the first read of those happens in the real run, guarded by
   the same role and hash checks.
9. **Audits without intervals.** Sex, flat leads, one condition against several and the multiclass view are
   point estimates; the protocol asks for no intervals there.
10. **Secondary analysis interval.** The local-minus-validation intervals resample the SPH evaluation half
    with the local thresholds and Platt fit held fixed, so they do not include the variability of the fit on
    the calibration half.
11. **Likelihood analysis** at PTB-XL fold 10 only; SPH statements carry no likelihood.
12. **Selected model equal to P0.** If that happens, the confirmatory entry says so and "better" is false
    (no demonstrated gain), without a degenerate paired bootstrap.
13. **Integrity.** The nine Stage 2 candidates' stored selection metrics were reproduced exactly (difference 0)
    from their saved validation files; `select` refuses a difference above 1e-12.
14. **Queue documents.** Not updated here; the orchestrator updates them when Stage 3 changes state.
