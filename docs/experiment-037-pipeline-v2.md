# Experiment 037: the candidate screening pipeline (pipeline v2) and its operating numbers

**Frozen 30 September 2026, before any score of this experiment is computed.** This runs the top-ranked
backlog item `pipeline_v2_rethreshold`. The user asked to keep experiments running during a machine
migration. Before this freeze only the aggregate results and protocols of 022b, 030, 032, 033 and 035 were
read, plus the key names and array shapes of the saved prediction files of 022b, 032 and 035 and the column
names of 030's and 033's `draws.csv`. No score, threshold or metric of this experiment was computed.

## Why

Experiment 035 found that the adopted pooled readout (022b `pooled`, PTB-XL plus the Challenge training
groups) loses on the PTB-XL hard added subset because pooling dilutes the 1,724 PTB-XL training ECGs without a
project label, and that giving them back their PTB-XL-only share of the weight (`dropped_upweighted`) fixes
most of it: xECG hard subset 0.655 to 0.712, SPH 0.939 unchanged. Experiment 030 (referral-budget threshold on
local normals) and 033 (the `combined_50` PVC and pre-excitation rule) were computed on the 022b readout.
This experiment defines the candidate screening pipeline, **pipeline v2**, from these three pieces,
recomputes 030's and 033's operating numbers with the v2 readout, and writes the specification that a later
one-time final test (`final_frozen_test`) would freeze. It runs no final test.

## The two pipelines

Both use the frozen xECG features only, and both use 033's finding heads unchanged.

| | v1 (current) | v2 (candidate) |
| --- | --- | --- |
| Binary readout R | 022b `pooled`: train-only `StandardScaler` and L2 logistic regression (`C=0.01`, `lbfgs`, `tol=1e-8`, float64), unweighted | the same fit with 035's `dropped_upweighted` weights in the scaler and the fit |
| Training rows | 17,083 PTB-XL training ECGs with the standard label and 22,494 Challenge training rows with a primary label that pass the training quality policy (39,577, 27,360 positive) | the same rows |
| Weights | 1 | the 1,724 PTB-XL rows without a project label carry 1,724 / 17,083 = 10.09% of the total weight (2.3167 each), every other row 0.9400 (weights average 1) |
| Finding heads | 032's xECG `ventricular_ectopy` (PVC) and `preexcitation` (WPW) heads, logit standardized on 032's 1,707 Challenge calibration clear normals (033) | the same |
| Rule | 033 `combined_50`: refer if R is above its threshold or F = max(z_PVC, z_WPW) is above its threshold | the same |

Both pipelines are also reported with the binary readout alone (rule `binary`, 030's rule), so that 030's
numbers are recomputed with v2 as well.

### Refitting and saved heads

- The v2 readout is refitted once here from 032's loaders (PTB-XL rows and features, the 25,577 SPH
  evaluation ECGs, the Challenge train and calibration rows and features), with the training quality policy
  taken from 032's saved `training_rows.csv` (hash checked) instead of recomputed from waveforms. The binary
  rows must equal 022b's `training_rows.csv` rows with an empty reason, in order.
- **Reproduction checks (the run stops if any fails):**
  - an unweighted refit on the same rows (v1) must reproduce 022b's saved SPH probabilities on the 21,008
    rows to 1e-12, and 032's saved `sph_xecg_binary` on all 25,577 rows to 1e-12;
  - the v2 refit must reproduce 035's saved `sph_xecg_dropped_upweighted` (21,008 rows) and
    `development_xecg_dropped_upweighted` (1,572 rows) to 1e-12;
  - the PVC and WPW heads are refitted with 032's rows and fit (`full_development.fit_logistic`) and must
    reproduce 032's saved xECG SPH and Challenge calibration probabilities to 1e-12.
- The fitted parameters of all three heads (scaler mean and scale, coefficients, intercept) and the two
  finding-head standardization constants are saved to `pipeline_v2_heads.npz`, and scoring from the saved
  parameters alone must reproduce the fitted heads' SPH probabilities to 1e-12. This file is what a final
  test would load. Fits run with one BLAS thread, as in 032 and 035.
- The 4,569 SPH ECGs outside 030's rows (no binary label) get v2 scores for the first time here; 035 scored
  only the 21,008.

## The simulated site: 030's and 033's, exactly

- **SPH is development data.** Experiments 022-035 have read it. This is a simulation of a new site, not a
  final test.
- Rows, split and outcome labels are 033's `load_rows` (030's `load_site` inside it): the 25,577
  `use_evaluation` SPH ECGs; 030's split of the 21,008 labeled ones (029's `split_site`, seed 29029); the
  4,569 without a binary label added with 033's `extend_split` (seed 33033).
- Evaluation half: 12,759 ECGs of 12,320 patients: 6,895 normal, 3,584 binary positive, 4,052 athlete-criteria
  composite (binary positive or PVC, WPW, AF/flutter, high-grade AV block, long QT), 531 PVC, 183 frequent PVC,
  15 WPW, 370 AF/flutter, 12 high-grade AV block, 10 long QT, 468 composite without a binary label, 1,812
  other. Superclasses (on the 10,479 evaluation ECGs of 030): MI 122, STTC 2,507, CD 1,184, HYP 117.
- Local pool: 6,923 normals (030's). Nothing is fitted or chosen on the evaluation half.

## Thresholds, draws, budgets and rules

- **Threshold rule (030).** With k = floor(b m) in integers, R's threshold for `binary` is the (k + 1)-th
  highest of the m local normal scores, and an ECG is referred when its score is strictly above it.
- **`combined_50` (033).** F gets rank r = floor(50 k / 1000), so r local normals lie above its threshold;
  R gets the largest rank at which at most k − 1 of the m normals lie above either threshold. 033's
  `finding_screen.split_thresholds` is used unchanged. It needs k ≥ 1: at 1% with m = 50 (k = 0) the rule is
  undefined and only `binary` is reported there. The finding rank r by m and budget:

| m | 1% | 2% | 5% | 10% |
| ---: | ---: | ---: | ---: | ---: |
| 50, 100 | 0 (undefined at 50) | 0 | 0 | 0 |
| 200 | 0 | 0 | 0 | 1 |
| 500 | 0 | 0 | 1 | 2 |
| 1,000 | 0 | 1 | 2 | 5 |
| 2,000 | 1 | 2 | 5 | 10 |

- **Draws (030).** m in {50, 100, **200**, 500, 1,000} and 2,000, 200 draws per m from
  `numpy.random.default_rng([30030, m, draw])` over the 6,923 local normals. The same draws serve every
  pipeline and rule (paired).
- **Budgets:** b in {1%, 2%, **5%**, 10%}.
- **Not included:** 030's m = 0 source quantile. It needs v2 scores of the PTB-XL calibration ECGs, which this
  experiment keeps closed, and pipeline v2 requires local normals by definition.

### Reproduction of 030 and 033 (the run stops otherwise)

- v1 `binary` must equal 030's `pooled` xECG draws: threshold, achieved rate, sensitivity and the four
  superclass sensitivities of every draw, for every m in 50-2,000 and every budget, to 1e-12.
- v1 `binary` and v1 `combined_50` must equal 033's draws (m 200 and 1,000; budgets 2%, 5%, 10%): both
  thresholds and every evaluation outcome of 033's `draws.csv`, and the local-pool columns, to 1e-12.
- The finding z-scores must equal 033's (recomputed with 033's `load_scores`, which also rechecks 032).

## Outcomes on the evaluation half

Per pipeline (v1, v2), rule (`binary`, `combined_50`), m and budget, the mean over the 200 draws with the
5th-95th percentile across draws and a 95% patient-bootstrap interval:

- binary-label sensitivity (3,584);
- athlete-criteria composite sensitivity (4,052);
- PVC (531) and WPW (15) sensitivity; secondary: frequent PVC, AF/flutter, high-grade AV block, long QT, the
  composite ECGs without a binary label, and the share of "other" ECGs referred;
- per-superclass sensitivity (MI, STTC, CD, HYP);
- the achieved false-referral rate (share of the 6,895 evaluation normals referred);
- at an assumed 5% prevalence, per 1,000 ECGs: referrals 1000 (0.05 s + 0.95 rate) and cases caught 50 s,
  with s the composite sensitivity (the pipeline's abnormal definition) and, as 030 did, also with s the
  binary-label sensitivity.

### Intervals

All intervals use `ecg_experiment/intervals.py`'s whole-patient convention: patients sorted by `np.unique`,
one draw `rng.integers(0, P, P)` from `numpy.random.default_rng(41041)` (`patient_groups`,
`patient_resample`), 2,000 draws over the 12,320 evaluation patients, every ECG of a drawn patient kept,
2.5th and 97.5th percentiles. In each resample the statistic is the mean over the 200 local-normal draws with
each draw's thresholds fixed (thresholds come from the local pool only). The same resamples serve every
pipeline, rule, m, budget and outcome, so every contrast is paired. A resample without an ECG of an outcome is
left out of that outcome's interval and counted (possible only for the smallest outcomes). The AUROC contrasts
use `intervals.paired_auroc_difference` with 2,000 draws and seed 41041.

## Primary contrast and decision

- **Primary:** composite sensitivity of v2 minus v1, rule `combined_50`, 5% budget, m = 200, with its paired
  95% interval.
- **Decision.** Adopt v2 as the candidate pipeline **unless it is worse**, meaning the whole interval lies
  below −0.005 (upper bound < −0.005). Otherwise v1 stays the candidate. Whether the lower bound is above
  −0.005 is reported beside it (a stricter non-inferiority reading); no decision attaches to that.
- **Secondary, no decision:** v2 minus v1 for every other outcome, rule, m and budget; binary-label
  sensitivity and achieved rate at 5%, m = 200; 033's three adoption conditions (composite gain interval
  above 0, binary cost at most 0.010, rate increase at most 0.25 points) for v2 `combined_50` against v2
  `binary`; and 033's local-pool selection step repeated with v2 (the seven 033 rules at 5%, m = 200, on the
  local pool's binary positives and composite ECGs) to see whether it still selects `combined_50`. The rule
  stays `combined_50` either way.
- **For completeness (AUROC, xECG, v1 and v2 with the v2 − v1 interval):** PTB-XL ordinary development (1,306
  ECGs, 843 positive), hard added subset (266, 41), full development (1,572), from 035's saved predictions
  (v2 reproduced here), and SPH (21,008, 7,190).

**What to expect (reasoning before any score).** SPH AUROC was 0.939 for both readouts in 035, so the
ranking of SPH ECGs changes little and the composite and binary sensitivities at a fixed budget should move by
less than about 0.01 either way. The hard-subset kind of ECG (sinus variants, ectopy, incomplete RBBB) is a
small share of SPH. The decision is therefore expected to be "adopt v2", and the interest is in the operating
numbers themselves.

## The specification file

`docs/pipeline-v2.md` lists exactly: the readout, features, training rows and weights; the finding heads and
their standardization; the threshold rule, budget choices and rule; the local-normal requirement (how many
normals, read as normal by the cardiologist); and every code path and output hash needed to reproduce it,
including `pipeline_v2_heads.npz`. It is written after the run from the outputs. If the decision keeps v1,
the file describes v1 instead and says so.

## Closed data and exclusions

- The PTB-XL calibration and test ECGs and the Challenge test groups are not read. The Challenge
  calibration groups are read only to reproduce 032's finding-head probabilities and standardization, as 032
  and 033 did.
- No age or other demographic subgroup analysis.
- New files only: `ecg_experiment/pipeline_v2.py`, `scripts/experiments/run_pipeline_v2_037.py` and
  `tests/test_pipeline_v2.py`. No frozen, hashed module changes.

## Caveats written into the results

- SPH is development data, an older Chinese hospital cohort at 34% prevalence; its normals are hospital
  normals, and the local pool and evaluation half come from the same hospital, the most favourable case.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs.
- The labels are ECG annotation proxies, not confirmed disease or referral decisions. A PVC code marks at
  least one PVC.
- 033's rule was selected on the local pool with v1; it is kept fixed here.
- One fit per head; only the local normals and the evaluation patients vary. The hard-subset and development
  AUROCs were read by 035; they are reported, not tested again.

## Execution

```bash
PYTHONPATH=/home/JanetRivera/ecg-wt-cpu OMP_NUM_THREADS=3 uv run --no-sync python -u \
    -m scripts.experiments.run_pipeline_v2_037 | tee outputs/experiment037_pipeline_v2_v1.log
```

CPU only, one process with at most three BLAS threads (fits with one). The runner hashes every input, source
and this protocol into the result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment037_pipeline_v2_v1/` (`result.json`, `draws.csv`, `predictions.npz`,
`pipeline_v2_heads.npz`; the log is moved in as `run.log`). Results go to
`docs/experiment-037-pipeline-v2-results.md` and the specification to `docs/pipeline-v2.md`.
