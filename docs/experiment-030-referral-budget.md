# Experiment 030: an operating point set by a referral budget

**Frozen 29 September 2026, before any score of this experiment is computed.** This runs the top-ranked
backlog item `referral_budget_operating_point`. The user asked to keep running experiments while they
prepare a move to a new machine. Before this freeze only aggregate results of 022b, 026b and 029 and
metadata were read: record counts, labels and superclass flags, patient IDs, family names, the file
receipts, and 029's split. One listing of 022b's prediction file printed the first three values of each
array; no rate, quantile or other metric was computed from them.

## Why

The target deployment is a university: a nurse records student ECGs and a cardiologist reads the referred
ones. Most students are healthy. 029 showed that a 95%-sensitivity threshold is impractical at a new site:
a distribution-free guarantee needs at least 45 local positives, and it then refers 49-75% of normals; at a
5% prevalence that is about 800 referrals per 1,000 students.

The alternative is to fix the referral budget instead of the sensitivity: flag a fixed share of the normal
ECGs. That threshold is a quantile of the scores of normal ECGs only, which a mostly healthy pilot supplies
without anyone having to find the rare abnormal ECG. Expert readers using the 2017 international criteria
for athletes' ECGs call 1.3-6.8% of ECGs abnormal, so budgets in that range match what a cardiologist
already works with. This experiment measures how many local normal ECGs fix such a threshold, and what
sensitivity each budget gives.

## The simulated site

- **Rows.** The 21,008 `use_evaluation` ECGs of `data/processed/sph_clean_v1` with a primary label, in the
  order of 022b's `predictions.npz` (whose ECG IDs, patient IDs and labels must equal the manifest rows, as
  in 029). The label is 022's primary label (the standard label).
- **Split.** 029's split, reused exactly: `local_adaptation.split_site` with seed 29029 must reproduce the
  `evaluation` column of 029's `split.csv` (hash checked against 029's `result.json`), or the run stops.

| Half | ECGs | Patients | Positive | Normal | MI | STTC | CD | HYP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evaluation | 10,479 | 10,181 | 3,584 | 6,895 | 122 | 2,507 | 1,184 | 117 |
| Local pool | 10,529 | 10,183 | 3,606 | 6,923 | 133 | 2,530 | 1,181 | 110 |

Every superclass ECG is positive on the standard label; an ECG can carry more than one superclass. Nothing
is fitted on the evaluation half.

## Scores

Nothing is refitted. Each score ranks ECGs, higher meaning more likely abnormal.

- `pooled`: the adopted 022b readout (PTB-XL plus the Challenge training groups). **xECG primary**, ECG-JEPA
  and CPC secondary.
- `ptbxl`: the PTB-XL-only readout (022), for reference, all three encoders.
- `normal_ref`: 026b's adopted `pooled` distance-from-normal score (squared Mahalanobis distance to 10,846
  training normals), all three encoders, as an unsupervised comparison.

The supervised scores are 022b's raw probabilities (`sph_<encoder>_<readout>` and
`calibration_<encoder>_<readout>`); a Platt step is strictly increasing and would refer the same ECGs. The
distance scores are 026b's `sph_scores.csv` and `challenge_scores.csv`. Every file is hash checked against
its receipt, and row orders must equal 022b's.

## The threshold rule

For a budget b and a set of m normal scores, let k = floor(b m), computed in integers (b in thousandths).
The threshold is the (k + 1)-th highest normal score, and an ECG is referred when its score is **strictly
above** it. So exactly k of the m normals (a share of at most b) lie above the threshold. This is the
empirical (1 − b) quantile without interpolation (Hyndman and Fan type 1, numpy's `inverted_cdf`). It is
fixed here and no other rule is tried.

**What to expect (reasoning before any score).** For exchangeable continuous scores, the share of new
normals above the (k + 1)-th highest of m is Beta(k + 1, m − k) distributed, with mean (k + 1)/(m + 1). The
rule therefore overshoots the budget slightly at small m. At a 5% budget:

| m | k | Expected rate | 5th-95th percentile | P(within ±1 pp) |
| ---: | ---: | ---: | --- | ---: |
| 50 | 2 | 5.9% | 1.7-12.1% | 0.26 |
| 100 | 5 | 5.9% | 2.6-10.2% | 0.35 |
| 200 | 10 | 5.5% | 3.1-8.3% | 0.48 |
| 500 | 25 | 5.2% | 3.7-6.9% | 0.69 |
| 1,000 | 50 | 5.1% | 4.0-6.3% | 0.85 |
| 2,000 | 100 | 5.05% | 4.3-5.9% | 0.96 |

The same calculation gives P(within ±1 pp) at m = 1,000 of 0.99 for a 1% budget, 0.97 for 2% and 0.71 for
10%. Draws without replacement from a pool of 6,923 normals vary somewhat less than this, and the fixed
evaluation half adds one common offset. The prediction for (A) is therefore "about 1,000 normals, perhaps
not reached by 1,000". To bracket that answer, m = 2,000 is added as a secondary size. At the 1% budget and
m = 50 the rule cannot flag less than the single highest normal (k = 0), so the rate is about 2%.

## Local normals and the source quantile

- **Local draws.** m in {50, 100, 200, 500, 1,000} (and 2,000, secondary) normal ECGs, drawn without
  replacement from the local pool's 6,923 normals, 200 draws per m, from
  `numpy.random.default_rng([30030, m, draw])`. The same draw serves every score and budget (paired).
- **m = 0: the source quantile, no local data.** The same rule on the normal ECGs of 022b's calibration
  pool: 1,923 normals (216 PTB-XL calibration, 1,178 Chapman/Ningbo, 346 Georgia and 183 CPSC calibration)
  for all six supervised heads. 026b did not score the PTB-XL calibration ECGs, so the `normal_ref` source
  quantile uses the 1,707 Challenge calibration normals. Scored once.
- **Budgets.** b in {1%, 2%, 5%, 10%}.

## Outcomes on the evaluation half

Per score, m and budget, over the draws (one value for m = 0):

- **Achieved false-referral rate:** the share of the 6,895 evaluation normals referred. Mean, 5th and 95th
  percentiles, and the share of draws within ±1 percentage point of the budget (and, secondary, within
  ±0.5 pp).
- **Sensitivity** for the standard label (3,584 positives): mean, 5th and 95th percentiles.
- **Sensitivity by superclass** (MI, STTC, CD, HYP): the share of that superclass's evaluation ECGs referred;
  mean, 5th and 95th percentiles.
- **Referrals per 1,000** at assumed prevalences of 2% and 5%: 1000 (π sensitivity + (1 − π) rate), from
  each draw's own sensitivity and achieved rate. Also the abnormal ECGs caught and missed per 1,000.

### Intervals over evaluation patients

The draws measure the variability of the local sample. The evaluation half's own sampling error is measured
by a patient bootstrap: 2,000 resamples of the 10,181 evaluation patients with replacement, from
`numpy.random.default_rng(34034)`; an ECG enters as often as its patient. In each resample the statistic is
the mean over the 200 draws of the sensitivity (or superclass sensitivity, or achieved rate), with each
draw's threshold fixed, since thresholds come from the local pool only. Intervals are the 2.5th and 97.5th
percentiles. The same resamples serve every score, m and budget, so contrasts are paired. They are computed
for m = 0 and every m, all scores, budgets and outcomes; the run stops if a resample holds no ECG of a
superclass.

## Pre-registered questions

- **(A) primary.** The smallest m in {50, 100, 200, 500, 1,000} at which the achieved false-referral rate is
  within ±1 percentage point of a 5% budget (4% to 6%, inclusive) in at least 90% of draws, for xECG, the
  `pooled` readout. If none qualifies, the answer is "not reached by 1,000", with the largest share, and
  the m = 2,000 share is reported beside it.
- **(B) primary.** Sensitivity at the 5% budget with m = 200 local normals, xECG `pooled`: the mean over the
  200 draws, its 5th-95th percentile across draws, and its 95% patient-bootstrap interval.
- **(C) secondary.** (A) and (B) for the 2% and 10% budgets.
- **Secondary contrasts** at the 5% budget and m = 200, paired over the same resamples: m = 200 minus the
  source quantile (m = 0), `pooled` minus `ptbxl` (xECG), `pooled` xECG minus `normal_ref` xECG, and xECG
  minus JEPA and minus CPC (`pooled`). Everything else is described with the same language and no decision
  attaches to it.

## Cross-site check in the Challenge families (secondary)

Each Challenge family's calibration group (Chapman/Ningbo 4,432 ECGs, 1,178 normal; Georgia 1,718, 346;
CPSC 1,462, 183), at every budget, per encoder, with the source quantile. Sensitivity and specificity with
95% Wilson intervals, by record (the Challenge headers carry no patient ID). Three settings:

- `pooled` readout, threshold from all 1,923 source normals. The family's own normals are part of this
  quantile, so it is partly in-sample.
- `pooled` readout, threshold from the source normals outside that family (PTB-XL plus the other two
  families): the threshold comes from other sites.
- The readout trained and calibrated without that family (022b's `loso_<family>`), with the threshold from
  the source normals outside that family: both readout and threshold come from other sites.

The threshold's own sampling error is not added to these intervals. The Challenge test groups stay closed.

## Closed data and exclusions

- The Challenge test groups, the PTB-XL test ECGs and PTB-XL calibration as a test stay closed. The PTB-XL
  calibration normals are used only as source normals for the m = 0 quantile, as 022b and 029 used the
  calibration pool.
- No age or other subgroup analysis beyond the four superclasses (the user wants that after the
  cardiologist meeting).
- New files only: `ecg_experiment/referral_budget.py`, `scripts/experiments/run_referral_budget030.py`,
  `scripts/reports/plot_referral_budget030.py` and `tests/test_referral_budget.py`. No frozen, hashed module
  changes.

## Caveats written into the results

- **SPH is development data now.** Experiments 022, 022b, 024b, 026, 026b, 027, 027b and 029 have all read
  it. This is a simulation of a new site, not a final test.
- SPH is an older Chinese hospital cohort at 34% prevalence, heavily band-pass filtered. Its normals are
  hospital normals, not healthy students; young normal ECGs (higher voltages, early repolarization) may
  score higher, so a threshold from hospital or source normals may over-refer students. That is why the
  local normals matter.
- A budget fixes the share of normals referred, not the sensitivity. The sensitivity at a budget depends on
  the site's mix of abnormalities, which differs between SPH and a student population.
- The label is an ECG annotation proxy (SPH's AHA codes mapped to the standard label), not confirmed disease
  or a referral decision. MI and HYP have only 122 and 117 evaluation ECGs.
- The local pool and the evaluation half come from the same hospital and period, the most favourable case.
- Readouts, encoders and references are one fit each; only the local normals vary.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -u -m scripts.experiments.run_referral_budget030
```

One CPU process with at most four BLAS threads, run once. No GPU. The runner hashes every input, source and
this protocol into the result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment030_referral_budget_v1/` (`result.json`, `draws.csv`, `families.csv`, `run.log`). The
figure (sensitivity against budget; achieved rate against m) goes to `docs/figures/experiment-030/` and the
results to `docs/experiment-030-referral-budget-results.md`.
