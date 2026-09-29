# Experiment 031: a hybrid of the supervised readout and the distance from normal

**Frozen 29 September 2026, before any score of this experiment is computed.** This runs the top-ranked
backlog item `hybrid_screening_score`. The user asked to keep running experiments while they prepare a move
to a new machine. Before this freeze only aggregate results of 022b, 026, 026b, 029 and 030 and metadata
were read: record counts, labels and superclass flags, patient and record IDs, family names and the file
receipts. No score, quantile or rate of this experiment was computed.

## Why

Experiment 030 set the operating point by a referral budget: flag a fixed share of the normal ECGs, with the
threshold taken from local normal ECGs only. At a 5% budget with 200 local normals at SPH, the pooled xECG
readout (022b) referred 0.775 of the abnormal ECGs and the distance from normal (026b, pooled reference)
0.589. Most misses were conduction findings (CD 0.681).

A supervised readout only learns the findings it is shown. A student screen will meet rare findings that
are thin or absent in the training labels. The distance from normal needs no abnormal label, and in 026 it
matched probes that had never seen a held-out superclass. This experiment asks whether combining the two
scores catches more abnormal ECGs at the same referral budget than the readout alone, overall and for
findings the readout never saw.

## The simulated site, draws and threshold rule (030, reused exactly)

- **Rows.** The 21,008 `use_evaluation` SPH ECGs with a primary label, in the order of 022b's
  `predictions.npz`; the standard label. Loaded with 030's `load_site`, which checks every receipt and
  reproduces 029's split (seed 29029).
- **Evaluation half.** 10,479 ECGs, 10,181 patients: 3,584 positive, 6,895 normal; MI 122, STTC 2,507,
  CD 1,184, HYP 117. With a single abnormal superclass (S-only): MI 37, STTC 2,194, CD 995, HYP 27.
  Nothing is fitted on the evaluation half.
- **Local normals.** m in {200 (primary), 1,000} normals from the local pool's 6,923, 200 draws per m from
  `numpy.random.default_rng([30030, m, draw])`, exactly 030's draws. The same draw serves every score.
- **m = 0: source normals, no local data.** The 1,707 Challenge calibration normals (Chapman/Ningbo 1,178,
  Georgia 346, CPSC 183), for every score in this experiment including the readout alone. Both component
  scores exist only there: 026b did not score the PTB-XL calibration ECGs. So the readout-alone m = 0 row
  here differs from 030's, which used 1,923 normals.
- **Budgets.** b in {1%, 2%, 5%, 10%}.
- **Threshold rule.** With k = floor(b m) in integers, the threshold is the (k + 1)-th highest normal score
  and an ECG is referred when its score is strictly above it (`referral_budget.budget_threshold`).

## Component scores (nothing refitted)

Per encoder, **xECG primary** and ECG-JEPA secondary (CPC is left out; 030 showed it far behind):

- `readout`: 022b's `pooled` raw probability, used as its logit R (`screening_threshold.head_logits`).
- `distance`: 026b's `pooled` squared Mahalanobis distance, used as its logarithm D.

Both transforms are strictly increasing, so the single scores refer exactly the ECGs 030 referred. The SPH
scores come from 022b's `predictions.npz` and 026b's `sph_scores.csv`; the Challenge calibration scores from
022b's `calibration_<encoder>_pooled` and 026b's `challenge_scores.csv`, through 030's `load_heads`, which
checks every hash and row order.

## Hybrid rules

Every hybrid's fitted parts use Challenge calibration data only (7,612 records, 5,905 positive, 1,707
normal): PTB-XL plus Challenge training data were used to fit the readout and the reference, and SPH is
never used to fit anything. PTB-XL calibration is not used, because 026b gave it no distance score.

- **`zmean` (primary hybrid).** Standardize R and D with the mean and standard deviation (ddof 0) of the
  1,707 source normals, then average: H = (z_R + z_D) / 2. Equal weights, no label is used. The budget rule
  is applied to H.
- **`stack` (secondary).** An unpenalized logistic regression on (z_R, z_D) fitted on the 7,612 Challenge
  calibration records with the standard label (`penalty=None`, `lbfgs`, `tol=1e-8`, `max_iter=5000`, float64,
  unweighted). The score is its linear predictor; the budget rule is applied to it. Its coefficients are
  reported.
- **`either` (secondary): the OR rule.** Refer an ECG if R is above its threshold or D is above its own.
  Both thresholds come from the same m normals with the same rank: the (j + 1)-th highest R and the
  (j + 1)-th highest D, with j the largest integer from 0 to k at which at most k of the m normals lie above
  either threshold. The budget is split equally in rank, and the false-referral share on the threshold
  normals stays at most b, as for the single scores.

No other rule, weight or transform is tried.

## Held-out conditions (026's design, secondary)

A normal reference should matter most for findings the readout never saw. For each superclass S in MI,
STTC, CD and HYP, the pooled readout is refitted without every training ECG that carries S, as 026's
`probe_without_S` did: 022b's 39,577 pooled training rows, PTB-XL superclasses from 026's `record_classes`
and Challenge superclasses from the split manifest. The fit is 022b's (`multisource_readout.fit_readout`:
train-only scaler, L2 logistic regression, C = 0.01), with one BLAS thread.

| Readout without | Training ECGs | Positive | Removed |
| --- | ---: | ---: | ---: |
| MI | 34,319 | 22,102 | 5,258 |
| STTC | 22,719 | 10,502 | 16,858 |
| CD | 29,763 | 17,546 | 9,814 |
| HYP | 35,540 | 23,323 | 4,037 |

- Features: 022b's loaders (PTB-XL training features from 022's caches, Challenge features from
  `outputs/features_challenge_v1/`, SPH features from 022's `features.npz`), every hash checked. The kept
  Challenge training rows are 022b's `training_rows.csv` (hash checked; its record order must equal the
  loaded rows). PTB-XL calibration and development features are not used.
- **Reproduction first.** The same code fitting all 39,577 rows must reproduce 022b's pooled SPH and
  Challenge calibration probabilities for xECG and JEPA to 1e-9, or the run stops.
- For each S, the readout without S replaces R. The hybrids are rebuilt from it: `zmean` standardized on the
  same 1,707 source normals, `stack` refitted on the Challenge calibration records without S (MI 7,332
  records, STTC 3,358, CD 5,581, HYP 7,014), `either` unchanged in form. The distance D is unchanged, since
  the reference never saw an abnormal ECG.
- Outcome: the share of the evaluation half's S ECGs referred, and of its S-only ECGs, which removes the
  ECGs the held-out readout could catch through a co-occurring superclass it did see. Also overall
  sensitivity and the achieved rate.

## Outcomes on the evaluation half

Per score (readout, distance, zmean, stack, either), encoder, m and budget, over the draws, exactly as 030:

- **Achieved false-referral rate**: share of the 6,895 evaluation normals referred. Mean, 5th and 95th
  percentiles across draws, and the share of draws within ±1 percentage point of the budget.
- **Sensitivity** (3,584 positives) and **sensitivity by superclass** (MI, STTC, CD, HYP): mean, 5th and 95th
  percentiles.
- **Referrals per 1,000** at assumed prevalences of 2% and 5%, as in 030.

### Intervals

A patient bootstrap of the evaluation half: 2,000 resamples of the 10,181 evaluation patients with
replacement, from `numpy.random.default_rng(35035)` (`referral_budget.bootstrap_counts`). In each resample
the statistic is the mean over the 200 draws, with each draw's thresholds fixed. Intervals are the 2.5th and
97.5th percentiles. The same resamples serve every score, encoder, m and budget, so contrasts are paired.
The run stops if a resample holds no ECG of a group (including the S-only groups).

## Pre-registered primary comparison and decision

- **Primary comparison.** Sensitivity at the 5% budget with m = 200 local normals, xECG:
  Δ = `zmean` minus `readout`, with its paired 95% patient-bootstrap interval. The paired difference in
  achieved rate is reported beside it.
- **Decision.**
  - **Adopt the hybrid** for the student-screen pipeline if the interval of Δ lies entirely above 0, Δ is at
    least +0.010 (one more abnormal ECG caught per 100) and the achieved-rate difference is at most
    +0.5 percentage points.
  - **The hybrid hurts** if the interval lies entirely below 0.
  - **Otherwise** the two are not distinguished in practice, and the readout alone stays, being simpler.
- **Secondary, no decision attached**: `stack` and `either` minus `readout` with the same contrast; JEPA; the
  other budgets and m; the superclasses; the rates. If a secondary rule meets the adoption criteria while
  `zmean` does not, it is reported as a hypothesis for a new test, not adopted.
- **Held-out reading (secondary).** For each S, at 5% and m = 200, xECG: `zmean` without S minus the readout
  without S, on the S ECGs and on the S-only ECGs. The normal reference **helps for an unseen S** if the
  interval on the S ECGs lies above 0. Also reported: readout without S minus the full readout (what the
  missing labels cost) and `zmean` without S minus the full readout.

**What to expect (reasoning before any score).** At SPH the readout ranks at AUROC 0.939 and the distance at
0.877; the two come from the same features and are likely strongly correlated. An equal-weight average with a
much weaker score usually lowers overall ranking unless the scores err on different ECGs, so `zmean` may
lose a little overall sensitivity; `stack` should weight the readout heavily and stay close to it, and
`either` spends half of the budget on the weaker score. The case for a hybrid is the held-out one: when S is
absent from training, the readout loses S ECGs and the distance should recover some.

## Closed data and exclusions

- The Challenge test groups, the PTB-XL test ECGs and PTB-XL calibration stay closed (PTB-XL calibration
  features are not loaded).
- No age or other subgroup analysis beyond the four superclasses.
- New files only: `ecg_experiment/hybrid_score.py`, `scripts/experiments/run_hybrid_score031.py`,
  `scripts/reports/plot_hybrid_score031.py` and `tests/test_hybrid_score.py`. No frozen, hashed module
  changes.

## Integrity

- 030's checks: 029's split, SPH order, evaluation counts and every receipt.
- The readout and distance alone reproduce 030's `draws.csv` rows (m = 200 and 1,000, every budget, xECG and
  JEPA): thresholds on the transformed scale map back exactly, and the rates and sensitivities are equal.
  030's `draws.csv` is hash checked against its receipt.
- The pooled refit reproduces 022b to 1e-9 before any held-out readout is fitted.
- The held-out training and calibration counts equal the tables above.
- Per-ECG referral shares reproduce every draw-averaged sensitivity to 1e-9.

## Caveats written into the results

- **SPH is development data.** Experiments 022, 022b, 024b, 026, 026b, 027, 027b, 029 and 030 have all read
  it. This is a simulation of a new site, not a final test.
- SPH is an older Chinese hospital cohort at 34% prevalence, heavily band-pass filtered; its normals are
  hospital normals, not healthy students. The label is an ECG annotation proxy (SPH's AHA codes mapped to the
  standard label), not confirmed disease or a referral decision. MI and HYP have only 122 and 117 evaluation
  ECGs, and 37 and 27 S-only ECGs.
- The Challenge calibration families were in the readout's training groups (record split), so the readout
  looks better there than at a new site; the stack may give it too much weight.
- The held-out readout still sees findings that co-occur with S; the S-only rows address this. Removing
  STTC removes 43% of the training rows, so that readout is also a smaller fit.
- One split, one fit per readout and reference; only the local normals and the evaluation patients vary.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -u -m scripts.experiments.run_hybrid_score031
```

One CPU process, at most four threads, no GPU, run once. The runner hashes every input, source and this
protocol into the result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment031_hybrid_score_v1/` (`result.json`, `draws.csv`, `run.log`). The figure goes to
`docs/figures/experiment-031/` and the results to `docs/experiment-031-hybrid-screening-score-results.md`.
