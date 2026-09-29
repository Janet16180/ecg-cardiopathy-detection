# Experiment 029: local adaptation at a new site

**Frozen 29 September 2026, before any score of this experiment is computed.** This runs the two
top-ranked backlog items as one experiment: `site_recalibration` (how many local labeled ECGs restore the
95%-sensitivity operating point) and `local_normal_manifold` (how many local normal ECGs the
distance-from-normal reference needs). The user asked to continue the work. Before this freeze only
aggregate results of 022b, 026, 026b, 027 and 027b and metadata were read: record counts, labels, patient
IDs, the saved Platt coefficients and thresholds of 022b, and the counts of the split below (computed from
patient IDs and labels only).

## Why

The target deployment is a university: a nurse records student ECGs and a cardiologist reads them. 027,
027b and 022b showed that no threshold fitted at other hospitals keeps 95% sensitivity at SPH: PTB-XL
calibration gives 0.922 for xECG, the adopted multi-hospital readout calibrated on its own hospitals 0.902,
and the hospitals miscalibrate in opposite directions. 026 and 026b showed that "normal" also shifts
between sites: normals from other hospitals close about a third of the SPH gap of the distance-from-normal
score. The remaining route is local data. This experiment treats SPH as the new site and measures how many
local ECGs the cardiologist must read to restore the operating point (Part A), and how many local normal
ECGs the normal reference needs (Part B). The answer sets the pilot size.

## The simulated site

- **Rows.** The 21,008 `use_evaluation` ECGs of `data/processed/sph_clean_v1` with a primary label
  (20,364 patients, 7,190 positive), in the order of 022b's `predictions.npz` (whose ECG IDs, patient IDs
  and labels must equal the manifest rows). The label is 022's primary label (the standard label).
- **Split** (`local_adaptation.split_site`, seed 29029). By patient, stratified by the patient's label
  (positive if any of their ECGs is positive; 85 patients have ECGs of both labels). Within each stratum the
  patients are shuffled (negative stratum first, one generator) and the first half, rounded down, forms the
  evaluation half. Every arm of both parts is scored on the same evaluation half, and nothing is fitted on
  it.

| Half | ECGs | Patients | Positive | Negative | Prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Evaluation | 10,479 | 10,181 | 3,584 | 6,895 | 34.2% |
| Local pool | 10,529 | 10,183 | 3,606 | 6,923 | 34.2% |

No patient is in both halves; the runner stops if a count differs.

## Part A: recalibrating the operating point

### Readouts

Frozen, with their saved SPH and calibration-pool probabilities from
`outputs/experiment022b_multisource_readout_v1/predictions.npz` (hash checked against 022b's
`result.json`). Nothing in a readout is refitted.

- `pooled`: the adopted 022b readout (PTB-XL plus the Challenge training groups). Its source operating
  point is 022b's: Platt and threshold fitted on the 8,176 PTB-XL and Challenge calibration ECGs.
- `ptbxl`: the PTB-XL-only readout (022, for reference). Its source operating point is 027's: Platt and
  threshold on the 564 PTB-XL calibration ECGs.
- Encoders: **xECG primary**, ECG-JEPA and CPC secondary. Six heads in all.
- **Reproduction.** The source Platt fits and thresholds are refitted from the saved calibration
  probabilities with `screening_threshold.fit_platt` and `screening_threshold.screening_threshold`, and must
  reproduce 022b's recorded coefficients and thresholds and its saved SPH calibrated probabilities
  (`sph_calibrated`) to 1e-9, or the run stops.

### Local draws

- Sizes n in {0, 50, 100, 200, 500, 1,000} ECGs; n = 0 is the source operating point (022b and 027),
  scored once.
- For n > 0, 200 draws per n (`local_adaptation.draw_records`): a simple random sample of n ECGs from the
  local pool, without replacement, from `numpy.random.default_rng([29029, 1, n, draw])`. The same draw
  serves every head and option (paired).
- The number of local positives in each draw is recorded, since the threshold depends on them.

### Options

Each option turns a local draw into a Platt mapping and a threshold (`local_adaptation.adapt`). An ECG is
referred when its calibrated probability is at or above the threshold.

| Option | Platt mapping | Threshold | Role |
| --- | --- | --- | --- |
| `threshold_only` | the source one | at least 95% of the draw's positives, on the source-calibrated probability | secondary |
| `platt_threshold` | refitted on the draw (`fit_platt`, C=1e6) | at least 95% of the draw's positives, on the new calibrated probability | **primary** |
| `blend` | refitted on the source calibration ECGs plus the draw, weighted so both have the same total weight (each local ECG weighs N_source / n; 027b's `fit_weighted_platt`) | weighted share of positives at least 95% (027b's `weighted_threshold`) | secondary |
| `conservative` | the source one | the r-th lowest positive score of the draw, r the largest rank with P(Binomial(k, 0.05) >= r) >= 0.90 for k local positives; the lowest positive when no rank qualifies (k < 45) | secondary |

- **Why `platt_threshold` and `threshold_only` refer the same ECGs.** Platt scaling with a positive slope is
  strictly increasing, and the threshold is the calibrated probability of one of the draw's positives, so
  both options put the cut at the same positive and refer the same evaluation ECGs. They differ only in the
  probability shown for each ECG (its calibration) and in their fallbacks. The run counts the draws in which
  their referrals differ and reports it; any difference is a tie or a fallback, not a finding.
- **What to expect from the 95% rule (reasoning before any score).** With the cut at a positive order
  statistic of k local positives, the evaluation sensitivity is roughly Beta-distributed around
  j / (k + 1), where j = ceil(0.95 k). For large k this centres on 0.95, so about half the draws fall
  below 0.95 however large n is; small k is conservative only through rounding. The `conservative` option
  is the distribution-free tolerance rule that targets "at least 95% with 90% confidence" directly; it is
  added for this reason, before any score, and is secondary.
- **Fallbacks, fixed in advance.** A draw without a local positive keeps the source threshold
  (`threshold_only`, `conservative`); `platt_threshold` keeps the source Platt and threshold if the draw lacks
  either class or its Platt slope is not positive; `blend` keeps the source operating point only if its
  weighted slope is not positive. A fallback draw is scored with the source operating point, so it counts
  as a failure when that point misses 95%. Fallbacks are counted per n and option.

### Outcomes on the evaluation half

Per head, n and option, over the draws:

- **Sensitivity:** the share of draws reaching at least 0.95, the mean, and the 5th percentile.
- **Specificity:** the mean and the 5th and 95th percentiles.
- **Referrals per 1,000 ECGs** at the evaluation half's prevalence (34.2%), and at an assumed 5%
  prevalence from the draw's sensitivity and specificity: 1000 (0.05 sens + 0.95 (1 − spec)).
- **Calibration of the shown probability** (`platt_threshold` and `blend`, and the fixed source mapping):
  mean Brier score, 10-bin ECE (027b's `weighted_rates` with unit weights) and calibration intercept and
  slope (`screening_threshold.calibration_fit`).
- **Local positives per draw:** mean, 5th and 95th percentiles.

### Low prevalence (secondary)

A university screen will have a far lower positive share than SPH's 34%. To see how many ECGs, and how many
positives, are then needed, the draws are repeated at an assumed 5% prevalence:

- n in {50, 100, 200, 500, 1,000, 2,000, 5,000}, 200 draws each, seeds `[29029, 2, n, draw]`.
- Each draw takes k ~ Binomial(n, 0.05) local positives and n − k local negatives, each without replacement
  from the local pool (`draw_records` with `prevalence=0.05`).
- Same heads, options, fallbacks and evaluation half. Sensitivity and specificity do not depend on the
  evaluation prevalence, so the 34% evaluation half still measures them; referrals are also given at 5%.
  Calibration metrics are not reported here (a 5% local draw calibrates to a 5% site, not to the 34%
  evaluation half).

## Part B: a local normal reference

### Score and references

- The Mahalanobis distance of 026 (`normal_manifold.fit_mahalanobis`, `mahalanobis_scores`: StandardScaler,
  PCA with `svd_solver="full"`, Ledoit-Wolf, float64), on the frozen `xecg` (primary), `jepa` and `cpc`
  features of 026 and 026b.
- **Non-local reference:** 026b's adopted `pooled` reference, fitted on its 10,846 normals (5,872 PTB-XL
  NORM training ECGs and 4,974 Challenge training normals that passed the training quality policy). The
  fit set is rebuilt with 026b's loaders; its membership is read from 026b's `fit_sets.csv` (hash checked
  against 026b's `result.json`), so the waveform quality check is not repeated. The rebuilt reference must
  reproduce 026b's saved SPH `pooled` scores of every encoder to 1e-9 relative, or the run stops.
- SPH features: 022's `features.npz`, loaded with 026's `load_sph`.

### Local normals

- m in {50, 100, 200, 500, 1,000} local normal ECGs (primary label 0), drawn without replacement from the
  local pool's 6,923 normals, 50 draws per m, seeds `[29029, 3, m, draw]`. The same draw serves every
  encoder and option.

### Options

| Option | Fit | PCA components | Role |
| --- | --- | ---: | --- |
| `local_only` | the m local normals only | min(64, m // 4): 12, 25, 50, 64, 64 | secondary |
| `local_plus_pooled` | the 10,846 non-local normals plus the m local normals, unweighted | 64 | **primary** |
| `recentered` | the non-local reference unchanged; each ECG's features are shifted by the reference mean minus the local normals' mean before scoring | 64 | secondary |

- **The PCA rule is fixed here, not chosen after the results.** PCA-64 on 50 normals is ill-posed (sklearn
  cannot keep more components than rows, and 64 directions from 50 samples would be noise). `local_only`
  keeps at least four normals per component, `min(64, m // 4)`, with Ledoit-Wolf shrinkage on those
  components as in 026. No other number of components is tried.
- `recentered` estimates only a mean from the local normals, so it is well posed at every m. It is added
  because a site offset in feature space (device, filtering) is the simplest explanation of 026's SPH gap.

### Outcomes on the evaluation half

Per encoder, m and option, over the 50 draws: mean AUROC with its 5th and 95th percentiles, mean average
precision, the share of draws whose AUROC is above the non-local reference's, and the mean gain over it.
Reference values on the evaluation half, each computed once: the non-local `pooled` reference, 026's
PTB-XL-only reference (026b's saved `ptbxl` scores), and the supervised readouts of Part A (`pooled` and
`ptbxl` raw probabilities).

## Pre-registered questions and reading

- **(A) primary.** The smallest n in {50, 100, 200, 500, 1,000} at which at least 90% of draws reach an
  evaluation sensitivity of at least 0.95, for xECG, the `pooled` readout, `platt_threshold`. If no n
  qualifies, the answer is "not reached by 1,000", and the results give the largest share reached.
- **(B) primary.** The smallest m in {50, 100, 200, 500, 1,000} at which `local_plus_pooled` beats the
  non-local reference on evaluation AUROC in at least 90% of draws, for xECG. If none qualifies, "not
  reached by 1,000". A mean gain below 0.005 AUROC is called negligible in practice (the project's noise
  level), even when the share is met; this wording does not change the answer.
- Everything else (other heads, options, encoders, the low-prevalence draws, specificity, referrals,
  calibration and the comparison with the supervised readout) is described with the same "smallest size at
  which at least 90% of draws meet the criterion" language and no decision attaches to it.
- The draws measure the variability of the local sample only. The evaluation half is fixed; its own
  sampling error (about ±0.007 on a sensitivity near 0.95 with 3,584 positives, a 95% binomial interval)
  is not added.

## Closed data and exclusions

- The Challenge test groups, PTB-XL calibration as a test, and the PTB-XL test ECGs stay closed. PTB-XL
  and Challenge calibration probabilities are used only as the source calibration data, as in 022b.
- No age or other subgroup analysis is done (the user wants that after the cardiologist meeting).
- New files only: `ecg_experiment/local_adaptation.py`, `scripts/experiments/run_local_adaptation029.py`,
  `scripts/reports/plot_local_adaptation029.py` and tests. No frozen, hashed module changes.

## Caveats written into the results

- **SPH is no longer untouched.** Experiments 022, 022b, 024b, 026, 026b, 027 and 027b have all read it.
  This is a simulation of a new site, not a final test.
- SPH is a Chinese hospital cohort with a 34% positive share, heavily band-pass filtered, and older and
  sicker than students. A university site will have a far lower prevalence, a different device and young
  people; the pilot numbers transfer only roughly.
- The label is an ECG annotation proxy (SPH's AHA codes mapped to the standard label), not confirmed
  disease or a referral decision. The cardiologist's own reads may differ from SPH's.
- The local pool and the evaluation half come from the same hospital and period, which is the most
  favourable case for local adaptation.
- Readouts, encoders and references are one fit each; only the local calibration or normal data vary.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_local_adaptation029
```

One CPU stage, run once: Part A, then Part B, each with one process per encoder and one BLAS thread per
process (three threads, within the four allowed). No GPU. The runner hashes every input, source and this
protocol into the result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment029_local_adaptation_v1/` (`result.json`, `split.csv`, `part_a_draws.csv`,
`part_b_draws.csv`, `run.log`). Figures (sensitivity against n; AUROC against m) go to
`docs/figures/experiment-029/` and results to `docs/experiment-029-local-adaptation-results.md`.
