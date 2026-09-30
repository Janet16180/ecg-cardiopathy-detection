# Experiment 025b: label efficiency with pooled PTB-XL and Challenge labels

**Frozen 29 September 2026, before any readout is fitted on a pooled label draw and before any new SPH,
PTB-XL development or Challenge score is computed.** Backlog item `rerun_025_ningbo`, the Ningbo-era rerun of
[Experiment 025](experiment-025-label-efficiency.md). The user authorized the reruns of the experiments
Ningbo could affect. Before this freeze only aggregate 025 and 022b results and metadata were read: record
counts, labels, feature record lists, 022b's kept training rows, and the source mix of the pooled label draws
below (which depend only on the pool table, not on any score).

## Question

025 drew 100 to 4,000 PTB-XL training labels and fitted a fixed logistic readout on frozen features. JEPA and
xECG beat CPC by 0.034-0.040 AUROC at every budget, and the curves were nearly flat above 4,000 labels. It
scored PTB-XL development only. 022b then showed that fitting the readout on PTB-XL plus the Challenge training
groups (Ningbo, Chapman/Shaoxing, Georgia, CPSC) raises SPH AUROC by about 0.02 with all labels, but loses
about 0.07 on 022's hard added PTB-XL development subset.

**Does pooling the Challenge labels change the label-efficiency curves or the encoder ranking (CPC, ECG-JEPA,
xECG), and does it help at SPH?** Concretely: with the same number of labels, is a readout trained on labels
drawn from PTB-XL plus the Challenge sources better at SPH than one trained on PTB-XL labels only?

## What stays fixed (from 025)

- **Readout.** 025's primary readout only: `full_development.fit_logistic` (train-only `StandardScaler`, L2
  logistic regression, `C=0.01`, L-BFGS, `max_iter=5000`, `tol=1e-8`, float64, seed 42). A solver that does
  not converge stops the run. 025's secondary readout (C chosen by cross-validation) is not rerun: it changed
  no 025 conclusion and was within 0.004 AUROC of the primary.
- **Encoders.** 025's `cpc`, `jepa` and `xecg` caches, rows selected by PTB-XL ECG ID and hash-checked
  against their receipts by 025's `identity`. **xECG is primary**, as in 022b; JEPA and CPC are secondary.
  025's `released_cpc` and `age_sex` are not rerun: they have no Challenge or SPH features.
- **Budgets, draws and seeds.** N in {100, 250, 500, 1000, 2000, 4000}, 20 draws per budget with seeds
  25025 + draw (draw 0-19), drawn by 025's `label_efficiency.draw_subset` unchanged; plus N = all. The budget
  counts total labels.
- **PTB-XL pool.** 025's pool, by 025's `select_rows`: 15,359 training ECGs, 13,351 patients, 9,487 positive
  (0.6177), labeled with the PTB-XL standard superclass label.

## Label pools and arms

**Challenge training rows.** Exactly 022b's: the `train` groups of the frozen
[Challenge split](challenge-splits-v1.md), evaluable, with a frozen-encoder feature row
(`outputs/features_challenge_v1/`, npz hashes checked against its metadata), kept by the training quality
policy. The runner takes the `kept` column of 022b's `training_rows.csv` (hash checked against 022b's result),
after requiring that its source and record order equal the rows it loads. 22,494 ECGs, 17,520 positive:
Ningbo 9,952, Chapman/Shaoxing 3,050, Georgia 5,136, CPSC 2018 2,600, CPSC-Extra 1,756. Label: the primary
label of the [Challenge mapping](challenge-label-mapping.md) (positive for any MI, STTC, CD or HYP code;
negative for sinus rhythm alone).

**Stacked table.** PTB-XL pool rows in 025's order, then the Challenge rows in 022b's order. Family and source
come from 022b's `stacked`, and the `ptbxl` and `pooled` rows from 022b's `arm_members`. The Challenge headers
carry no patient ID and the split is by record, so each Challenge record counts as its own patient
(`challenge:<source>:<record>`). The pooled pool is 37,853 ECGs from 35,845 patient units, 27,007 positive
(0.7135).

| Arm | Training labels at budget N | Role |
| --- | --- | --- |
| `ptbxl` | 025's draw of N PTB-XL labels (N = all: the 15,359) | reference: must reproduce 025 |
| `pooled` | `draw_subset` of N labels from the pooled pool, same seeds (N = all: the 37,853) | **primary** |
| `ptbxl_plus_challenge` | 025's draw of N PTB-XL labels plus all 22,494 Challenge labels (N + 22,494 in total) | secondary |

**How the pooled draws spread across sources (pre-registered): uniformly over the pool.** 025's rule is
applied unchanged to the pooled pool: permute all 37,853 rows, keep the first ECG of each patient unit, take
the first round(N × 0.7135) positives and the rest negatives. Every patient unit has the same chance of being
labeled, so each source contributes in proportion to what it has. Reasons:

- It is the plain meaning of "pool the labels and draw N": a lab that pools public labels draws from the
  union, and the rule adds no free parameter.
- A fixed per-source share or family balancing would need a share to be chosen. 022b's family-balanced arm
  did not improve SPH ranking over the unweighted pool and cost the most at PTB-XL.
- Draws still come from distinct patients and depend only on the pool table, so every encoder and arm sees
  paired draws.

Counted before this freeze (no score): the PTB-XL share of a pooled draw averages 0.40-0.41 at every budget
(range 0.33-0.47 at N = 100, 0.39-0.41 at N = 4,000). Chapman/Ningbo averages 0.33-0.35, Georgia 0.14 and
CPSC 0.11-0.13. The runner records each draw's source counts.

**Prevalence.** The pooled draws are 71% positive against 62% for PTB-XL draws, because the Challenge rows are
78% positive. This follows from the pool; AUROC is a ranking measure, but a linear head fitted at a different
prevalence can rank slightly differently. The `ptbxl_plus_challenge` arm answers the other practical question:
does a readout with N home labels gain from adding every public label?

## Evaluation sets

Nothing is fitted or selected on an evaluation set.

1. **SPH (primary).** 022's 21,008 `use_evaluation` ECGs with a primary label (20,364 patients, 7,190
   positive), with 022's saved features (hash checked). The standard label (022's primary label).
2. **PTB-XL development (secondary), reported separately:**
   - **ordinary:** 025's 1,306 original development ECGs (1,173 patients, 843 positive), with 025's features.
     The runner requires that 022's feature path gives identical features for these rows.
   - **hard added subset:** 022's 266 added development ECGs with a standard label (41 positive), the ones
     the project label dropped; JEPA and xECG features from 022's `ptb_features.npz` combined with the caches
     (022's `ptb_encoder_features`), CPC from Experiment 020's saved features.
3. **Challenge calibration groups per family (secondary).** 022b's sets: Chapman/Ningbo 4,432 (3,254
   positive), Georgia 1,718 (1,372), CPSC 1,462 (1,279), all quality. For `pooled` and
   `ptbxl_plus_challenge` these families are in distribution: the readout saw their training groups, and a
   patient may repeat across the record split. For `ptbxl` they are unseen hospitals.

The Challenge **test** groups, PTB-XL calibration and the PTB-XL test set stay closed. No age or other subgroup
analysis is done.

## Reproduction of 025 (runs before any other score is reported)

The `ptbxl` arm must reproduce 025's primary readout for `cpc`, `jepa` and `xecg`: every draw's subset hash,
record and positive counts, and development AUROC and average precision in 025's `draws.csv`, and the N = all
development probabilities in 025's `all_budget_predictions.npz` (both hash checked against 025's result). The
run stops if any AUROC, AP or probability differs by more than 1e-9, and reports the maximum difference.

## Metrics

- Per arm, encoder, budget and set: AUROC and average precision, as the mean, SD and 2.5 and 97.5 percentiles
  over the 20 draws (025's `summarize`); N = all is one fit.
- **Draw rule (025's):** per budget, paired by draw, the AUROC difference of two readouts, summarized as the
  mean and the number of draws above 0 (`paired_summary`), and read by 025's 90% rule (`reading`): A is
  better than B if A − B is positive in at least 18 of 20 draws.
- **Paired bootstrap:** 2,000 draws, seed 32032, by whole patient for SPH and PTB-XL development
  (`normal_manifold.patient_resamples`), by record for the Challenge families; draws with one class are
  skipped and counted. One set of resamples per evaluation set serves every arm, encoder and budget. At each
  budget below all the statistic is the draw-mean AUROC (the mean over the 20 draws' AUROCs on the same
  resample), so the interval covers evaluation patients with the label draws held fixed; at N = all it is the
  single fit's AUROC, as in 025. Fits are held fixed. AUROC only; intervals are the 2.5 and 97.5
  percentiles.
- Contrasts: `pooled − ptbxl` and `ptbxl_plus_challenge − ptbxl` per encoder, budget and set (draw rule and
  bootstrap); `jepa − cpc`, `xecg − cpc` and `xecg − jepa` within each arm, per budget and set (draw rule),
  and at N = all with the bootstrap.
- Label efficiency at SPH: the smallest budget below all at which each arm's mean SPH AUROC reaches the
  `ptbxl` arm's SPH AUROC at N = all (all 15,359 PTB-XL labels), or "none" (025's `smallest_budget`).

## Primary comparison and decision rule

- **Primary comparison:** SPH, xECG, AUROC of `pooled` minus `ptbxl` at N = 250 and N = 1,000 (025's read
  budgets).
- **Verdict per read budget:**
  - `pooled better` if `pooled` − `ptbxl` is positive in at least 18 of 20 draws **and** the bootstrap
    interval of the draw-mean difference lies entirely above 0;
  - `ptbxl better` if it is negative in at least 18 of 20 draws and the interval lies entirely below 0;
  - otherwise `not distinguished`.
- **Decision:**
  - `adopt_pooled` if `pooled better` at both read budgets: for a small labeled set, draw readout labels
    from the pooled PTB-XL and Challenge pool rather than PTB-XL alone.
  - `pooling_hurts` if `ptbxl better` at both.
  - Otherwise `mixed`: PTB-XL-only stays the default for a small labeled set (the simpler pool), and the
    results describe the budgets where each wins.
- **Size.** A difference below 0.005 AUROC is called "negligible in practice" even when its interval
  excludes 0 (the priorities page's noise level). This wording does not change the decision.
- **Encoder ranking (prespecified secondary reading).** For each arm, on SPH and on ordinary PTB-XL
  development, at N = 250 and N = 1,000, 025's 90% rule for `jepa − cpc`, `xecg − cpc` and `xecg − jepa`.
  The ranking **changes** if any of these verdicts differs between `ptbxl` and `pooled` on the same set and
  budget; otherwise it **holds**.
- The hard subset, the Challenge families, `ptbxl_plus_challenge` and the other budgets are described by the
  draw rule and where their intervals lie (above 0, below 0, includes 0), with no separate decision.

## Caveats written into the results

- ECG-JEPA and xECG report Chapman and Ningbo in their self-supervised pretraining data, so they have likely
  seen the `chapman_ningbo` waveforms without labels. CPC (Experiment 004, PTB-XL and MIMIC) has seen no
  Challenge record. No encoder saw a Challenge label; SPH is unseen by every encoder.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM and SPH's code 1, and
  the pooled draws are more often positive (0.71 against 0.62).
- The pooled PTB-XL rows are 025's 15,359, not 022b's 17,083 (025's pool is restricted to the released
  caches), so `pooled` at N = all is not 022b's `pooled` fit.
- Challenge records stand in for patients; a Challenge patient with two records can be drawn twice, and the
  family readouts are in distribution for the pooled arms.
- The hard added subset has 41 positives, so its values are imprecise.
- Draw variability reflects label sampling on fixed evaluation sets. The development patients were inspected
  by earlier experiments. One checkpoint per encoder; the label is an ECG annotation proxy, not a clinical
  outcome or a referral decision.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_label_efficiency025b
```

One CPU stage with one process per encoder and one BLAS thread each (three threads in all, within the four
allowed). `scripts/experiments/run_label_efficiency025b.py` reuses the frozen 025, 022, 027, 026b and 022b
runners and modules and the new `ecg_experiment/label_efficiency_multisource.py`; no frozen module changes. It
hashes every input, source and this protocol into the result, performs the checks above, refuses to overwrite
an existing run, and writes `outputs/experiment025b_label_efficiency_multisource_v1/` (`result.json`,
`draws.csv`, `all_budget_predictions.npz`, `run.log`). Results go to
`docs/experiment-025b-label-efficiency-multisource-results.md`.
