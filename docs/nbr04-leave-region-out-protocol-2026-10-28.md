# NBR-04 protocol: leave-region-out personalization of the normal counterpart

Frozen 28 October 2026, before any replacement score of a new arm is read. Same branch as NBR-03
(`research/nbr03-ptbxl-replication-20261028`), after NBR-03's commits; stacked on #79. The user approved this
study; the design comes from the lead's brief and the choices below are fixed here.

## Question

NBR-02's rank-6 personalized counterpart (C6) sharpened lead localization but absorbed part of the
abnormality, so it changed abnormal ECGs relatively less than normal ones and lost to P0 on selectivity. Can
we keep the localization gain without the absorption, by never letting the replaced region inform the
personalization (L6), and by also removing the disease directions from the normal subspace (L6D)?

## Arms

Both use the C fit set and C6 model of NBR-02 (7,339 all-age cohort normals, rank 6, 062 machinery).

- L6, leave-region-out. For each of the 52 replaced regions, the record's subspace coordinates are 062's
  Huber-reweighted PPCA posterior (`subspace_coefficients`, five iterations, clipped at 2.58 SD) with the
  Huber weights forced to 0 on that region's canonical bins in the replaced leads
  (`nbr04_templates.masked_subspace_coefficients`, which returns 062's coordinates exactly when every bin is
  kept). The template is `covariates @ coefficients + scale * (basis @ coordinates)` and is warped, blended,
  tapered and masked as in NBR-01 and NBR-02, one template per region.
  - Canonical bins of a component on 062's grid: P = the PR segment (bins 0-149, QRS onset minus 300 ms to
    QRS onset), QRS = bins 150-199, ST-T = the ST and T segments (bins 200-329). The post-T bins 330-379
    belong to no component.
  - A grid cell drops its component's bins in its lead; a whole lead drops bins 0-329 of that lead; a whole
    component drops its bins in all leads; the full replacement drops every bin, so its coordinates are 0
    and its template is C0's.
- L6D, L6 plus disease-direction removal, using four directions, one per superclass (MI, STTC, CD, HYP).
  - Reason for four directions rather than one pooled abnormal-minus-normal direction: NBR-02's absorption
    was largest for LBBB, a conduction pattern, while a pooled abnormal mean is dominated by ST-T records
    (12,285 of the 17,526 usable abnormal rows carry STTC), so one direction would mostly protect ST-T
    patterns.
  - Computation (stage `fitsets`, already run; training data only, never SPH or PTB-XL fold 9): the 25,000
    rows of the all-age cohort, 24,865 used after 135 exclusions (missing age; same rule as NBR-02), their
    062 medians standardized with the C model's coefficients and scale. Superclass flags come from
    `heads01_features.superclass_table`, which reproduces every cohort target. Each direction is the mean
    standardized residual of the rows with that superclass (MI 2,038, STTC 12,285, CD 6,030, HYP 2,367) minus
    the mean of the 7,339 normals (the same count as NBR-02's C fit set). The four directions are
    orthonormalized (QR).
  - Removal: the directions are projected out of C6's basis and the result re-orthonormalized (SVD; the six
    singular values are 1.000, 1.000, 0.981, 0.846, 0.801 and 0.379, none below the 1e-3 drop rule, so rank 6
    is kept). The basis is rotated to its principal axes on the fit normals, and the eigenvalues, noise and
    off-subspace scale are refitted on those normals exactly as `fit_population` computes them; coefficients
    and scale are unchanged. Eigenvalues become 677, 465, 406, 351, 209 and 145 (C6: 1,118, 632, 509, 427, 319,
    226); noise 0.891 (C6: 0.677). The orthonormal directions have norms 0.873, 0.582, 0.551 and 0.364 inside
    C6's original span.
  - L6D then applies L6's leave-region-out rule with this model.

## Samples, scores and comparators

- SPH: NBR-01's 1,037 ECGs (comparable to NBR-02). P0 scores are NBR-01's; C6 scores are NBR-02's
  (context only).
- PTB-XL fold 9: NBR-03's 862 ECGs, with NBR-03's P0 scores, groups and primary territories (inferior and
  anterior). Fold 10 is never read; preparation refuses any row outside fold 9.
- Unreplaced and self-swap scores are NBR-01's (SPH) and NBR-03's (PTB-XL). New encoding: 52 variants per arm
  per ECG (104), in full batches of 16 with the last batch padded.
- `all_age_blend` is primary; the HGB and xECG components and `ptb_baseline_blend` are secondary. SPH uses the
  SPH thresholds and PTB-XL the fold-8 PTB thresholds, as in NBR-01 and NBR-03.

## Primary decision

All intervals are 97.5% whole-patient bootstrap intervals (two arms), 2,000 draws, `default_rng(20261028)`,
every statistic of every arm of one dataset on the same draws. An arm replaces P0 only if all four hold:

- A. Selectivity, SPH: arm minus P0, lower bound above 0. Selectivity is NBR-02's: mean absolute
  `all_age_blend` change over the 12 whole-lead replacements on the 737 abnormal ECGs, divided by the same on
  the 300 normals.
- B. Territory, SPH: NBR-01's territory contrast, arm minus P0, lower bound above -0.05.
- C. Tightened guardrail, SPH: the whole-component flip rate among flagged abnormal ECGs (any of the three
  whole-component replacements leaves the score at or below the threshold), arm minus P0, lower bound above
  -0.05. Full replacement is the population template by construction, so the guardrail moves to the
  component level.
- D. Confirmation on PTB-XL fold 9 (NBR-03's sample and primary territory contrast): the selectivity
  difference (arm minus P0) has a point estimate above 0, and the territory difference has a lower bound
  above -0.05. D is reported as confirmed or not confirmed.

Tie rule: if both arms pass, choose L6 unless L6D's SPH selectivity exceeds L6's with a paired lower bound
above 0 (97.5%), in which case choose L6D.

Review of the rule before any score, kept unchanged: A uses absolute changes (NBR-02's caveat applies, so the
numerator and denominator contrasts are reported); C at the component level is the meaningful guardrail
because L6's full replacement equals C0 by construction.

## Secondary (97.5% intervals, no decision), on both datasets

- Every NBR-02 statistic for each arm with its own energy map: selectivity and its numerator and
  denominator, territory contrast, model minus energy, wave contrasts, BBB lead contrast, top-1 and top-3
  hits, flip rates, full-replacement sanity, normals' full drop; contrasts L6 - P0, L6D - P0, L6D - L6, and on
  SPH also L6 - C6 and L6D - C6.
- The HGB and xECG components and `ptb_baseline_blend`.
- Clip rate by group: the share of the 51 non-full region fits of a record with any coordinate at the clip
  bound, averaged by group, for L6 and L6D.
- Template movement by group: the mean over the 51 non-full regions of the mean absolute difference between
  the region template and the C0 (population) template on 062's grid, in mV, especially for LBBB and RBBB.
- On PTB-XL also NBR-03's all-territory and per-territory statistics.

## Integrity checks

1. L6's full-replacement counterpart equals NBR-02's C0 counterpart bit for bit for all 1,037 SPH ECGs
   (checked in `prepare`), and L6's full-replacement scores equal NBR-02's C0 full-replacement scores exactly.
   Pre-registered exception: because the variants of a record are encoded in batches of that record's own
   variants, an xECG score difference with identical signals is reported as a batch-composition effect;
   any HGB (measurement) difference stops the study.
2. The NBR-04 analysis code reproduces the P0 primary statistics exactly from the saved scores: SPH
   +0.3056275517940945 (NBR-01) and PTB-XL fold 9 +0.09848614265592237 (NBR-03).
3. NBR-02's C0 and C6 models share coefficients and scale (checked), and L6D keeps them.
4. Variant SHA-256 per arm between preparation and encoding.
5. 062's knots of every record equal those of the NBR-01 or NBR-03 plan.

## Profile and run

The `profile` stage times preparation and encoding on 10 SPH and 10 fold-9 ECGs before the full run.
Forecast from NBR-02 and NBR-03 (about 1 s per 105 variants per GPU): encoding about 16 minutes over both
GPUs for 1,899 ECGs; preparation (104 templates and measurements per ECG) about 10-15 minutes with 16 workers.
Each GPU shard runs under its own lock with at least 4 GiB free.

## Page export

`outputs/nbr04_leave_region_out_20261028/examples.json`: the chosen arm, or L6 if none passes, on NBR-01's 16
SPH IDs in the same order, NBR-01's final schema plus `arm`, `counterpart_note` and `summary.decisions`. The
`counterpart` array is a composite: lead l comes from the whole-lead-l template, fitted with lead l left out.
Signals, components, unreplaced and self-swap scores are NBR-01's. `summary.statistics` holds every arm's
page statistics and the contrasts, for SPH unprefixed and for PTB-XL under `ptbxl_fold9/`.

## Data rules

The fit set and disease directions use only all-age cohort training rows. SPH and PTB-XL fold 9 never fit,
scale or select anything. No Challenge test or calibration group, EchoNext, MIMIC, CODE-15, PTB-XL fold 10
or final frozen test is read. No age-subgroup analysis.

## Outputs

`outputs/nbr04_leave_region_out_20261028/`; work files in `/tmp/nbr04-work`. Code:
`ecg_experiment/localization/nbr04_*.py`, `scripts/experiments/run_nbr04.py`,
`tests/localization/test_nbr04_leave_region_out.py`. Report:
`docs/nbr04-leave-region-out-results-2026-10-28.md`.

## Amendment 1 (28 October 2026): three arms, written after the L6 and L6D results and before any D6 score

The user asked, through the lead, for a three-arm design before the protocol was frozen. The request arrived
after this protocol had been committed (`e0b32ee`) and after the L6 and L6D replacement scores had been read and
reported (`34b7bfb`). It is therefore an amendment made with the L6 and L6D results known. It is committed before
any score of the new arm D6 is computed. The original two-arm decision stays reported as it was.

Arms, all on the C fit set at rank 6:

- L6 (unchanged): leave-region-out only.
- D6 (new): disease-direction removal only. The L6D model (the C6 basis with the four superclass directions
  removed, refitted on the normals), with coordinates computed on the whole record by 062's
  `subspace_coefficients`, one template per record, as NBR-02's C6 (`nbr02_templates.arm_differences`).
- L6D (unchanged): both.
- NBR-02's C6 (neither change) completes a 2 x 2 factorial: leave-region-out yes/no by direction removal yes/no.

Additional scoring: D6 on the 1,037 SPH ECGs and the 862 PTB-XL fold-9 ECGs, and C6 on the fold-9 ECGs (NBR-02
scored C6 on SPH only). The amendment's code path rebuilt NBR-02's SPH C6 counterparts for 100 records with a
largest difference of 0.0 mV (stage `amendment_check`, no score). Profile (no score): preparation 1.13 CPU-seconds
and encoding 0.78 s per record on 10 SPH and 10 fold-9 ECGs.

Primary decision (replaces the two-arm decision for the page): parts A to D against P0 exactly as above, for each
of L6, D6 and L6D, with 98.33% whole-patient intervals (Bonferroni over three arms), 2,000 draws, seed 20261028,
every statistic of one dataset on the same draws. If more than one arm passes all four parts, choose the one with
the highest SPH selectivity point estimate and report the paired selectivity differences between the passing
arms. If none passes, P0 stays and L6 is exported.

Because L6 and L6D already failed parts A and C with 97.5% intervals, and a 98.33% interval is wider, the
amendment cannot make either pass; only D6 is unseen.

Secondary, with 95% intervals on the same draws, on SPH and on PTB-XL fold 9, for selectivity, the territory
contrast, the top-1 territory hit and the whole-component flip rate (and for every score):

- leave-region-out effect: L6 - C6 (without removal) and L6D - D6 (with removal);
- direction-removal effect: D6 - C6 (without leave-out) and L6D - L6 (with leave-out);
- interaction: (L6D - D6) - (L6 - C6).

Integrity added: the amendment's analysis code must reproduce NBR-04's P0, L6 and L6D point statistics exactly on
both datasets, including the P0 primaries (SPH +0.3056275517940945, fold 9 +0.09848614265592237); variant SHA-256
between preparation and encoding for D6 and C6. The page export follows the amended decision and replaces
`outputs/nbr04_leave_region_out_20261028/examples.json`.
