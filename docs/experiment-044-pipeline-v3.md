# Experiment 044: pipeline v3, the candidate pipeline with the concatenated-feature readout

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiment 043 Stage 1 found that
`logistic_concat`, pipeline v2's readout refitted on the concatenated xECG and ECG-JEPA features, beats the
v2 readout R by its prespecified rule (SPH AUROC 0.9404 against 0.9387, +0.0017 [+0.0008, +0.0026]; full
PTB-XL development 0.9347 against 0.9306, +0.0041 [+0.0008, +0.0077]). An AUROC gain does not have to show
at the top 5% of normals, where the referral rule works (037 found two readouts with equal SPH AUROC that
differ there). This experiment asks whether pipeline v2 with only R replaced, **pipeline v3**, catches more
abnormal ECGs at the same referral budget. The user authorized follow-up experiments when results show
improvements. Before this freeze only the protocols, results reports and aggregate `result.json` fields of
037 and 043, and the key names and shapes of their saved prediction files, were read. No score, threshold or
metric of v3 or v3b on any draw was computed.

## The three pipelines

| | v2 (current candidate) | v3 (candidate) | v3b (secondary) |
| --- | --- | --- | --- |
| Binary readout | 037's R: xECG features (1,024) | the same fit on the concatenated xECG and JEPA features (1,792, xECG first) | v3's |
| Readout fit | `multisource_readout.fit_readout(x, y, weights)` with 035's `dropped_upweighted` weights | the same | the same |
| Training rows | 037's 39,577 (27,360 positive) | the same | the same |
| Finding heads | 032's xECG PVC and WPW heads (`full_development.fit_logistic`) | the same | the same rows, labels and fit on the concatenated features |
| Standardization | 1,707 Challenge calibration clear normals (`finding_screen.finding_z`) | the same constants | the same recipe on the concatenated heads |
| Rule | 033 `combined_50` and `binary` alone | the same | the same |

Everything else (rows, labels, weights, solver settings, the threshold rule, the draws, the budgets and the
resamples) is 037's, unchanged.

### Features and rows

- Rows, labels and weights come from Experiment 043's `training_data` (imported from its runner, an accepted
  exception), which returns the xECG and JEPA features of the same rows in the same order. Its xECG part must
  equal Experiment 037's `training_data` (043's `check_against_037`).
- The JEPA features of the Challenge calibration rows come from the same `challenge_table` call that gives
  037's calibration rows and xECG features; 037 used the calibration rows only for the finding-head
  standardization, and so does this experiment. Nothing else of the calibration groups is read.
- v2 in the draws is the frozen v2: its binary score is 037's saved `sph_v2_binary` (hash checked) and its
  finding scores are 033's `load_scores`, exactly as 037 used them. The refits below only check them.

### Reproduction checks (the run stops if any fails)

- Every input file is checked against its receipt: 037's `predictions.npz`, `draws.csv` and
  `pipeline_v2_heads.npz`, 043 Stage 1's `predictions.npz`, 035's `predictions.npz`, and those that 033's
  `load_rows` and `load_scores` check themselves (030, 032).
- The xECG rows, labels, weights and evaluation rows equal 037's `training_data`.
- The v2 refit reproduces 037's saved `sph_v2_binary` (25,577) and `development_v2_binary` (1,604) to 1e-12
  (037's tolerance), and the refitted xECG PVC and WPW heads reproduce 037's saved `sph_z_pvc`, `sph_z_wpw`
  and `sph_z_combined` and the standardization constants in `pipeline_v2_heads.npz` to 1e-12.
- 033's z-scores equal 037's saved z-scores to 1e-12.
- The v3 readout's logits reproduce 043 Stage 1's saved `sph_logistic_concat` and
  `development_logistic_concat` to 1e-10, and its AUROCs equal 043's to 1e-12.
- The v2 draws (`binary`, `combined_50` and the other 033 rules at 5% with 200 normals) reproduce the v2 rows
  of 037's `draws.csv` exactly: every threshold, outcome and local-pool column, difference 0.
- v2's AUROCs on 037's four sets equal 037's to 1e-12.
- The fitted parameters of v3's readout and of the four finding heads, with their standardization constants,
  are saved to `pipeline_v3_heads.npz`; scoring from the saved parameters alone must reproduce the fitted
  heads to 1e-12. Fits run with one BLAS thread and the OpenBLAS Haswell kernels (`OPENBLAS_CORETYPE=Haswell`),
  which 043 used.

## The simulated site, draws and rules: 037's, exactly

- **SPH is development data** (read by 022-043). This simulates a new site; it is not a final test.
- Rows, split and outcomes: 033's `load_rows`; evaluation half 12,759 ECGs of 12,320 patients (6,895 normal,
  3,584 binary positive, 4,052 athlete-criteria composite); local pool of 6,923 normals.
- Threshold rule (030), `combined_50` (033, `finding_screen.split_thresholds`), 200 draws per m from
  `numpy.random.default_rng([30030, m, draw])`, m in {50, 100, 200, 500, 1,000, 2,000}, budgets
  {1%, 2%, 5%, 10%}; `combined_50` only where k = floor(b m) ≥ 1. The other 033 rules run at 5% with 200
  normals only, for the local-pool selection step. The same draws serve every pipeline (paired).
- Intervals: 037's, unchanged. Whole-patient bootstrap over the 12,320 evaluation patients, 2,000 resamples
  from `numpy.random.default_rng(41041)` (`pipeline_v2.resample_counts`), each resample's statistic the mean
  over the 200 draws with the draw thresholds fixed, 2.5th and 97.5th percentiles. AUROC contrasts use
  `intervals.paired_auroc_difference`, 2,000 draws, seed 41041.

## Primary comparison and decision

- **Primary:** the share of athlete-criteria composite ECGs (4,052) caught by rule `combined_50` at a 5%
  budget with 200 local normals, **v3 minus v2**, paired over 037's identical draws and resamples, with its
  95% interval.
- **Decision.**
  - `adopt_v3` if the lower bound of that interval is above 0;
  - otherwise `no_worse_keep_v2` if the lower bound is above −0.005 (037's margin): v3 is reported as no
    worse, but not adopted;
  - otherwise `keep_v2`.
- v3b carries no decision.

## Secondary outcomes (no decision)

- At 5% with 200 normals, for v2, v3 and v3b: binary-label sensitivity (3,584), the achieved false-referral
  rate (share of the 6,895 evaluation normals referred), PVC, WPW, frequent PVC, AF/flutter, high-grade AV
  block, long QT, composite without a binary label, "other" referred, and the four superclasses; referrals and
  cases caught per 1,000 at an assumed 5% prevalence (037's definition).
- The same outcomes at every other budget and m, for `combined_50` and `binary` alone.
- Contrasts, each paired: v3 − v2, v3b − v2, v3b − v3, and each pipeline's `combined_50` − `binary`.
- 033's local-pool selection step repeated with v3 and v3b.
- AUROC of the binary readout on 037's sets (PTB-XL full development 1,572 ECGs, ordinary 1,306, hard 266;
  SPH 21,008), v2 and v3, with the paired v3 − v2 interval. v3b's readout is v3's.
- Finding-head AUROC on the SPH ECGs with a defined PVC or WPW label (both halves, as 032), xECG head
  against concatenated head, with the paired difference.

**What to expect (reasoning before any score).** The SPH AUROC gain is small (+0.0017), so the composite share
at a fixed budget should move by less than about 0.01. 037 showed that AUROC and budget sensitivity can
disagree, so the sign is not certain. The JEPA features add morphology information that may help ST-T
findings more than ectopy; the PVC head already carries the PVC ECGs, so most of any gain should come from
binary-positive ECGs. v3b might help WPW (15 ECGs, too few to tell) and PVC.

## Closed data and exclusions

- The PTB-XL calibration and test ECGs and the Challenge test groups are not read. The Challenge calibration
  groups are read only for the finding-head standardization, exactly as 037.
- No age or other demographic subgroup analysis. No label definition changes.
- New files only: `ecg_experiment/pipeline_v3.py`, `tests/test_pipeline_v3.py`,
  `scripts/experiments/run_pipeline_v3_044.py`, this protocol and the results report. If v3 is adopted, the
  specification goes to a new `docs/pipeline-v3.md`; `docs/pipeline-v2.md` is not edited.

## Caveats written into the results

- SPH is development data, an older Chinese hospital cohort at 34% prevalence; the local pool and evaluation
  half come from the same hospital, the most favourable case.
- 043 chose `logistic_concat` on SPH and full development AUROC, both read here again; this experiment is a
  second look at the same data, not an independent confirmation.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs.
- The labels are ECG annotation proxies, not confirmed disease or referral decisions.
- One fit per head; only the local normals and the evaluation patients vary.
- v3 needs a second frozen encoder (ECG-JEPA) at the student site, which adds compute and one more
  dependency to freeze.

## Execution

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v3_044 --smoke \
    --output outputs/experiment044_pipeline_v3_smoke
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v3_044
```

CPU only, at most four threads (fits with one). The smoke mode fits and scores training rows only (a held-out
part of 4,000 binary training rows, 043's `smoke_split`), and no development, calibration or SPH ECG is
scored. The runner hashes every input, source and this protocol into the result, performs the checks above,
writes to `<output>.partial` and renames it when complete, and refuses to overwrite an existing output. The
full run writes `outputs/experiment044_pipeline_v3_v1/` (`result.json`, `draws.csv`, `predictions.npz`,
`pipeline_v3_heads.npz`, `run.log`). Results go to `docs/experiment-044-pipeline-v3-results.md`.
