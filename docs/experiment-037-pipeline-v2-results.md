# Experiment 037 results: the candidate screening pipeline (pipeline v2) and its operating numbers

Completed 30 September 2026 under the [frozen protocol](experiment-037-pipeline-v2.md) (committed as
`54e9aff`; the run recorded the same file hash, `14ecebae…fd70`). Backlog item `pipeline_v2_rethreshold`. Run
once at commit `780d6f6` on CPU in 213 s, one process with at most three threads (fits with one), no GPU.
Local outputs are in `outputs/experiment037_pipeline_v2_v1/` (`result.json` SHA-256 `36ac7f6a…c857`,
`draws.csv`, `predictions.npz`, `pipeline_v2_heads.npz`, `run.log`). No PTB-XL calibration or test ECG and no
Challenge test-group ECG was read. No age or other demographic subgroup analysis was done. The specification
of the adopted pipeline is in [pipeline-v2.md](pipeline-v2.md).

**SPH is development data.** Experiments 022-035 have all read it. This is a simulation of a new site, not a
final test.

## Integrity

- The unweighted refit (v1) reproduced 022b's SPH probabilities and 032's `sph_xecg_binary` exactly
  (largest difference 0.0), and 035's saved development `pooled` probabilities exactly.
- The v2 refit (`dropped_upweighted`) reproduced 035's saved SPH (21,008) and development (1,572)
  probabilities exactly (0.0). The 1,724 dropped PTB-XL rows carry weight 2.3167, the other 37,853 rows 0.9400.
- The refitted PVC and WPW heads reproduced 032's SPH and Challenge calibration probabilities exactly (0.0),
  and their z-scores equal 033's. 1,707 Challenge calibration clear normals set the standardization.
- Scoring from the saved parameters in `pipeline_v2_heads.npz` alone reproduced every head to within
  6e-16.
- v1 with the binary readout alone reproduced 030's pooled xECG draws exactly (threshold, rate, sensitivity
  and the four superclass sensitivities of all 4,800 draws, m = 50 to 2,000, every budget). v1 reproduced
  033's draws exactly (all 3,600 shared draws, every threshold, outcome and local-pool column), and 033's
  local-pool selection of `combined_50`.
- Every count matched 033's. One of the 2,000 bootstrap resamples held no high-grade AV block ECG and is left
  out of that outcome's interval; no other outcome lost a resample.

Intervals are 95% paired whole-patient bootstrap intervals over the 12,320 evaluation patients
(`ecg_experiment.intervals` convention, 2,000 draws, seed 41041). Point values are means over the 200
local-normal draws. v1 is the 022b readout, v2 the 035 `dropped_upweighted` readout; both use 033's
`combined_50` rule unless the row says "binary alone".

## Primary: composite sensitivity at 5%, 200 local normals

| | v1 | v2 | v2 − v1 [95% CI] |
| --- | ---: | ---: | --- |
| Athlete-criteria composite (4,052) | 0.794 | 0.789 | **−0.0048 [−0.0072, −0.0025]** |
| Binary label (3,584) | 0.772 | 0.768 | −0.0044 [−0.0070, −0.0018] |
| Achieved false-referral rate (6,895 normals) | 5.41% | 5.45% | +0.05 pp [−0.07, +0.16] |
| PVC (531) | 0.968 | 0.967 | −0.0014 [−0.0025, −0.0005] |
| WPW (15) | 0.790 | 0.816 | +0.026 [0.000, +0.088] |
| MI (122) | 0.927 | 0.919 | |
| STTC (2,507) | 0.829 | 0.820 | −0.0092 [−0.0118, −0.0066] |
| CD (1,184) | 0.680 | 0.688 | +0.0076 [+0.0025, +0.0128] |
| HYP (117) | 0.901 | 0.885 | |
| Composite without a binary label (468) | 0.963 | 0.955 | −0.0077 [−0.0150, −0.0017] |
| AF/flutter (370) | 0.997 | 0.989 | |
| "Other" ECGs referred (1,812; not counted as false referrals) | 0.254 | 0.223 | |

**Decision: `adopt_v2`.** The pre-registered rule adopts v2 unless the whole interval lies below −0.005; its
upper bound is −0.0025. The stricter non-inferiority reading, reported beside it, is **not** met: the lower
bound −0.0072 lies below −0.005. v2 catches slightly fewer composite ECGs than v1 at this operating point,
about 19 of the 4,052, and the interval excludes zero.

## By budget, 200 and 1,000 local normals

Composite and binary-label sensitivity, achieved rate, and v2 − v1 composite with its interval.

| Budget | m | v1 composite | v2 composite | v2 − v1 composite | v1 binary | v2 binary | v1 rate | v2 rate |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1% | 200 | 0.649 | 0.641 | −0.008 [−0.010, −0.006] | 0.612 | 0.606 | 1.50% | 1.52% |
| 2% | 200 | 0.713 | 0.707 | −0.006 [−0.008, −0.004] | 0.682 | 0.677 | 2.40% | 2.43% |
| 5% | 200 | 0.794 | 0.789 | −0.005 [−0.007, −0.002] | 0.772 | 0.768 | 5.41% | 5.45% |
| 10% | 200 | 0.850 | 0.852 | +0.002 [−0.000, +0.004] | 0.833 | 0.835 | 10.09% | 10.07% |
| 1% | 1,000 | 0.649 | 0.643 | −0.006 [−0.009, −0.003] | 0.618 | 0.615 | 1.15% | 1.16% |
| 2% | 1,000 | 0.712 | 0.705 | −0.007 [−0.010, −0.004] | 0.680 | 0.674 | 1.96% | 1.99% |
| 5% | 1,000 | 0.796 | 0.790 | −0.005 [−0.008, −0.002] | 0.772 | 0.767 | 5.15% | 5.24% |
| 10% | 1,000 | 0.851 | 0.853 | +0.002 [−0.000, +0.005] | 0.834 | 0.837 | 9.97% | 9.82% |

At 5% the v2 − v1 composite difference is −0.005 to −0.006 at every m from 50 to 2,000. Achieved rates across
draws at 5%, m = 200 range 3.0-7.7% for v2 (3.1-8.0% for v1); within ±1 point of the budget in 50% of draws
(v1 57%) with 200 normals and 88% (v1 89%) with 1,000.

Per superclass, the pattern at every budget is the same: v2 refers about 0.01 fewer STTC ECGs and 0.004-0.016
more CD ECGs (the gain grows with the budget), and slightly fewer MI and HYP ECGs (122 and 117 ECGs, wide
intervals).

## The binary readout alone (030 recomputed with v2)

At 5% and 200 normals, v2 alone has binary-label sensitivity 0.769 against v1's 0.775 (−0.006 [−0.009,
−0.003]) and composite 0.744 against 0.757 (−0.013 [−0.017, −0.010]). The larger composite loss comes from PVC
ECGs: v2 alone refers 0.584 of them against 0.647 (−0.063 [−0.079, −0.049]), and 0.552 against 0.619 of the
composite ECGs without a binary label. This is the hard-subset mechanism of 035 working as intended: v2 learns
from the PTB-XL rows that an ECG with ectopic beats can be normal, so it ranks PVC-only ECGs lower. The PVC
head restores them in the full pipeline (0.967 against 0.968).

## 033's finding rule under v2 (secondary)

With v2, `combined_50` against the v2 binary readout alone gains more than it did with v1, and all three of
033's adoption conditions hold:

| Budget (m = 200) | v1 composite gain | v2 composite gain | v2 binary cost | v2 rate difference |
| --- | --- | --- | --- | --- |
| 1% | +0.033 [+0.025, +0.040] | +0.037 [+0.029, +0.045] | 0.024 | −0.02 pp |
| 2% | +0.038 [+0.032, +0.046] | +0.049 [+0.041, +0.056] | 0.006 | −0.03 pp |
| 5% | +0.037 [+0.031, +0.044] | +0.045 [+0.039, +0.053] | 0.001 | −0.11 pp |
| 10% | +0.029 [+0.024, +0.035] | +0.038 [+0.032, +0.045] | 0.003 | −0.22 pp |

033's local-pool selection repeated with v2 still selects `combined_50` (local composite 0.779, local binary
cost 0.003; `combined_100` is the only other eligible rule, 0.776 at cost 0.0097).

## AUROC (xECG, for completeness)

| Set | ECGs (positive) | v1 | v2 | v2 − v1 [95% CI] |
| --- | --- | ---: | ---: | --- |
| PTB-XL hard added subset | 266 (41) | 0.655 | 0.712 | +0.058 [+0.038, +0.079] |
| PTB-XL ordinary development | 1,306 (843) | 0.955 | 0.954 | −0.002 [−0.003, +0.000] |
| PTB-XL full development | 1,572 (884) | 0.927 | 0.931 | +0.004 [+0.002, +0.006] |
| SPH | 21,008 (7,190) | 0.9386 | 0.9387 | +0.000 [−0.000, +0.001] |

These reproduce 035 (whose intervals used seed 39039).

## Numbers for the cardiologist meeting

Pipeline v2 (`combined_50`), threshold set on 200 local normal ECGs, assuming 5% of students have an
athlete-criteria abnormal ECG. "Abnormal" is the binary diagnostic label; "athlete-criteria findings" adds
PVC, WPW, AF/flutter, high-grade AV block and long QT. Referrals per 1,000 use the athlete-criteria share.

| Budget (share of normals flagged) | Referrals per 1,000 | Share of abnormal caught | Share of athlete-criteria findings caught |
| --- | ---: | ---: | ---: |
| 1% | 47 | 0.61 | 0.64 |
| 2% | 58 | 0.68 | 0.71 |
| 5% | 91 | 0.77 | 0.79 |
| 10% | 138 | 0.84 | 0.85 |

At 5% that is about 91 referrals per 1,000 students, 39-40 of the 50 abnormal ones caught (interval 38.9-40.0)
and 10-11 missed. With 1,000 local normals the means barely move (89 referrals, 0.79 of findings), but the
achieved share of normals flagged is much steadier. v1 gives the same table to within one referral per 1,000
and 0.01 in each share.

## What it means, in plain language

- On SPH the two readouts are almost the same screen. At a 5% budget, v2 catches about 5 fewer of every 1,000
  athlete-criteria abnormal ECGs than v1; at 10% it catches about 2 more. Both refer the same number of
  normal ECGs.
- The small loss at 1-5% is real (its interval excludes zero) but within the pre-registered tolerance, so v2
  is adopted as the candidate. It does not pass the stricter reading that it is no more than 0.005 worse.
- What v2 buys is the PTB-XL behaviour that 035 found: it scores normal ECGs with sinus variants, extra beats
  and incomplete RBBB the PTB-XL way. At SPH this shows as fewer "other" ECGs referred (sinus bradycardia or
  tachycardia, atrial premature beats: 22% against 25%), more conduction findings caught and fewer ST-T
  findings caught. At PTB-XL it is +0.058 AUROC on the hard subset.
- For students, that trade probably favours v2. Healthy students often have sinus bradycardia or arrhythmia
  and occasional ectopic beats, which the athlete criteria call normal; a readout that scores them as abnormal
  would push them above the threshold. SPH cannot test this, because its "normal" is sinus rhythm alone.
- The PVC head carries the PVC findings in v2: without it v2 would miss 6 more of every 100 PVC ECGs. The
  finding heads are a required part of pipeline v2, not an option.

## Surprises

- v2's loss on SPH at small budgets comes mostly from STTC (−0.009) and from PVC-only ECGs, not from the
  normals. The 035 AUROC comparison (SPH +0.000) hid it: two readouts can rank the whole set equally and still
  differ at the top 5% of normals.
- The finding rule helps v2 more than v1 (+0.045 against +0.037 at 5%), because v2's binary readout gives up
  more of the PVC ECGs that the head then recovers.
- At 10% v2 is slightly ahead (+0.002), driven by conduction findings (+0.016).

## Caveats

- SPH is development data (022-035), an older Chinese hospital cohort at 34% prevalence, whose normals are
  hospital normals with sinus rhythm alone. The local pool and evaluation half come from the same hospital,
  the most favourable case. "Other" ECGs are not counted as false referrals here, but at a student site many
  of them would be normal.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs; MI and HYP have 122 and 117.
- The labels are ECG annotation proxies, not confirmed disease or referral decisions. A PVC code marks at
  least one PVC.
- 033's rule was chosen on the local pool with v1 and kept fixed; the v2 selection agreed.
- One fit per head; only the local normals and the evaluation patients vary. The hard-subset and development
  AUROCs were read by 035 and are reported, not retested.
- The adoption margin (0.005) was fixed by the protocol; the stricter non-inferiority reading fails, so a
  reader who wants "no worse on SPH" should prefer v1. The final test should report both pipelines if the user
  wants that question settled on unseen data.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=3 uv run --no-sync python -u -m scripts.experiments.run_pipeline_v2_037
```

The runner refuses to overwrite an existing `result.json`. It needs the 020, 022, 022b, 029, 030, 032, 033 and
035 outputs, the JEPA and xECG caches (loaded by 032's PTB-XL loader), `outputs/features_challenge_v1/` and the
SPH manifest. It reads no raw waveform: the training quality policy comes from 032's saved `training_rows.csv`.
