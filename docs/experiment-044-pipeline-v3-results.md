# Experiment 044 results: pipeline v3, the candidate pipeline with the concatenated-feature readout

Completed 1 October 2026 under the [frozen protocol](experiment-044-pipeline-v3.md) (committed as `755c7b0`;
the run recorded the same file hash, `2426971d…2026`). Run once at commit `5f291cf` on CPU in 174 s, one
process with at most four threads (fits with one), OpenBLAS Haswell kernels, no GPU. Local outputs are in
`outputs/experiment044_pipeline_v3_v1/` (`result.json` SHA-256 `f3ec9b58…c16d`, `draws.csv`,
`predictions.npz`, `pipeline_v3_heads.npz`, `run.log`). A training-rows-only smoke run
(`outputs/experiment044_pipeline_v3_smoke/`, 56 s) preceded it. No PTB-XL calibration or test ECG and no
Challenge test-group ECG was read; the Challenge calibration groups were read only for the finding-head
standardization, as in 037. No age or other demographic subgroup analysis was done. The specification of the
adopted pipeline is in [pipeline-v3.md](pipeline-v3.md).

**SPH is development data.** Experiments 022-043 have read it, and 043 chose the concatenated readout partly on
SPH AUROC. This is a simulation of a new site, not a final test.

## Integrity

- The xECG training rows, labels, weights and evaluation rows of 043's `training_data` equal 037's.
- The refitted v2 readout reproduced 037's saved SPH (25,577) and development (1,604) probabilities exactly
  (largest difference 0.0). The refitted xECG PVC and WPW heads reproduced 037's saved z-scores and
  standardization constants exactly (0.0), and 037's saved parameters to within 6e-16. 033's z-scores equal
  037's (0.0).
- The v3 readout (422 iterations) reproduced 043 Stage 1's saved `logistic_concat` logits to within 1.1e-14
  (SPH) and 1.8e-15 (development), and its four AUROCs equal 043's.
- The v2 draws reproduced all 10,600 v2 rows of 037's `draws.csv` exactly (every threshold, outcome and
  local-pool column, largest difference 0.0). v2's AUROCs equal 037's, and 033's local-pool selection with
  v2 still selects `combined_50`.
- Scoring from `pipeline_v3_heads.npz` alone reproduced every saved head to within 7e-16.
- One of the 2,000 bootstrap resamples held no high-grade AV block ECG and is left out of that outcome's
  interval, as in 037; no other outcome lost a resample.

Intervals are 95% paired whole-patient bootstrap intervals over the 12,320 evaluation patients (037's
resamples, seed 41041). Point values are means over 037's 200 local-normal draws. All rows use 033's
`combined_50` rule unless they say "binary alone". v2 is 037's pipeline, v3 replaces only its binary readout
with the readout on xECG + JEPA features, and v3b also refits the PVC and WPW heads on xECG + JEPA.

## Primary: composite sensitivity at 5%, 200 local normals

| | v2 | v3 | v3b | v3 − v2 [95% CI] | v3b − v2 | v3b − v3 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Athlete-criteria composite (4,052) | 0.789 | 0.797 | 0.799 | **+0.0080 [+0.0040, +0.0120]** | +0.0094 [+0.0052, +0.0136] | +0.0014 [+0.0003, +0.0026] |
| Binary label (3,584) | 0.768 | 0.777 | 0.777 | +0.0092 [+0.0048, +0.0137] | +0.0094 [+0.0048, +0.0140] | +0.0002 [−0.0006, +0.0010] |
| Normals referred (6,895) | 5.45% | 5.74% | 5.76% | +0.28 pp [+0.04, +0.52] | +0.31 pp [+0.05, +0.56] | +0.03 pp [−0.05, +0.10] |
| PVC (531) | 0.967 | 0.966 | 0.976 | −0.0003 [−0.0035, +0.0019] | +0.0093 [+0.0027, +0.0170] | +0.0096 [+0.0033, +0.0168] |
| WPW (15) | 0.816 | 0.765 | 0.741 | −0.051 [−0.169, 0.000] | −0.075 [−0.194, +0.004] | −0.025 [−0.067, +0.006] |
| Composite without a binary label (468) | 0.955 | 0.954 | 0.965 | −0.0013 [−0.0070, +0.0037] | +0.0094 [+0.0003, +0.0187] | +0.0107 [+0.0035, +0.0188] |
| MI (122) | 0.919 | 0.921 | 0.921 | +0.002 [−0.018, +0.022] | | |
| STTC (2,507) | 0.820 | 0.827 | 0.827 | +0.0071 [+0.0019, +0.0122] | | |
| CD (1,184) | 0.688 | 0.700 | 0.700 | +0.0117 [+0.0037, +0.0202] | | |
| HYP (117) | 0.885 | 0.914 | 0.915 | +0.029 [+0.010, +0.053] | | |
| AF/flutter (370) | 0.989 | 0.988 | 0.988 | −0.0007 [−0.0067, +0.0050] | | |
| "Other" ECGs referred (1,812; not counted as false referrals) | 0.223 | 0.249 | 0.261 | +0.026 [+0.018, +0.034] | +0.038 [+0.029, +0.048] | +0.012 [+0.008, +0.016] |

**Decision: `adopt_v3`.** The pre-registered rule adopts v3 if the lower bound of v3 − v2 is above 0; it is
+0.0040. v3 catches about 8 more of every 1,000 athlete-criteria abnormal ECGs than v2, and it is ahead in 71%
of the 200 local-normal draws (behind in 28%). v3b carries no decision; it is ahead of v3 by a further 0.0014.

v3 also refers more normal ECGs at the same nominal budget: 5.74% against 5.45%, an interval that excludes
zero. The post hoc check below asks how much of the gain that explains.

Per 1,000 ECGs at an assumed 5% prevalence, v3 gives 94.4 referrals against v2's 91.3 and catches 39.9 of the
50 athlete-criteria abnormal ECGs (interval 39.3-40.5) against 39.5 (38.9-40.0).

## By budget, 200 and 1,000 local normals

| Budget | m | v2 composite | v3 composite | v3b composite | v3 − v2 composite | v2 binary | v3 binary | v2 rate | v3 rate |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1% | 200 | 0.641 | 0.650 | 0.651 | +0.009 [+0.005, +0.013] | 0.606 | 0.616 | 1.52% | 1.56% |
| 2% | 200 | 0.707 | 0.716 | 0.717 | +0.009 [+0.005, +0.013] | 0.677 | 0.688 | 2.43% | 2.55% |
| 5% | 200 | 0.789 | 0.797 | 0.799 | +0.008 [+0.004, +0.012] | 0.768 | 0.777 | 5.45% | 5.74% |
| 10% | 200 | 0.852 | 0.859 | 0.859 | +0.007 [+0.003, +0.011] | 0.835 | 0.843 | 10.07% | 10.49% |
| 1% | 1,000 | 0.643 | 0.646 | 0.644 | +0.003 [−0.002, +0.008] | 0.615 | 0.618 | 1.16% | 1.20% |
| 2% | 1,000 | 0.705 | 0.715 | 0.717 | +0.011 [+0.005, +0.016] | 0.674 | 0.686 | 1.99% | 2.09% |
| 5% | 1,000 | 0.790 | 0.798 | 0.800 | +0.008 [+0.003, +0.013] | 0.767 | 0.776 | 5.24% | 5.46% |
| 10% | 1,000 | 0.853 | 0.861 | 0.862 | +0.007 [+0.002, +0.012] | 0.837 | 0.845 | 9.82% | 10.29% |

At 5% the v3 − v2 composite difference is +0.006 to +0.008 at every m from 50 to 2,000, and every lower bound is
above 0 (smallest +0.0017, at m = 500). The only `combined_50` screens whose interval includes 0 are 1% with
1,000 normals (above) and 1% with 2,000 (−0.001 [−0.007, +0.005]). At every m and budget v3 refers slightly
more normals than v2 on average (+0.004 to +0.70 points; +0.13 to +0.28 at 5%). Achieved rates across
draws at 5%, m = 200 range 3.5-8.5% for v3 (3.0-7.7% for v2); with 1,000 normals 4.4-6.6% (4.3-6.1%), within ±1 point in 77% of draws against 88%.

## The binary readout alone

At 5% with 200 normals, v3's readout alone has composite sensitivity 0.758 against v2's 0.744 (+0.014
[+0.009, +0.018]) and binary-label sensitivity 0.780 against 0.769 (+0.011 [+0.007, +0.016]), at a rate of
5.89% against 5.56%. It refers 0.624 of the PVC ECGs against 0.584, so part of v2's ectopy cost (037) is
recovered by the JEPA features; the PVC head still carries them in the full pipeline (0.966).

## 033's finding rule under v3 (secondary)

| Budget (m = 200) | v2 composite gain | v3 composite gain | v3 binary cost | v3 rate difference | v3b composite gain |
| --- | --- | --- | ---: | ---: | --- |
| 1% | +0.037 [+0.029, +0.045] | +0.031 [+0.023, +0.039] | 0.025 | −0.05 pp | +0.032 [+0.024, +0.040] |
| 2% | +0.049 [+0.041, +0.056] | +0.040 [+0.033, +0.047] | 0.011 | −0.14 pp | +0.041 [+0.034, +0.049] |
| 5% | +0.045 [+0.039, +0.053] | +0.040 [+0.034, +0.047] | 0.003 | −0.16 pp | +0.041 [+0.035, +0.048] |
| 10% | +0.038 [+0.032, +0.045] | +0.033 [+0.027, +0.039] | 0.004 | −0.20 pp | +0.033 [+0.028, +0.040] |

The finding rule still adds about 0.04 composite at 5%, a little less than with v2 because v3's readout
already ranks more PVC ECGs high. 033's local-pool selection with v3 and with v3b still selects `combined_50`
(v3: local composite 0.787, local binary cost 0.0039; v3b: 0.788, 0.0028).

## Post hoc: the gain at a matched false-referral rate (not prespecified, no decision)

Because v3 refers 0.28 points more normals, `scripts/experiments/rate_matched_044.py` set each pipeline's
thresholds on the 6,895 evaluation normals themselves (an oracle no site has), so that every pipeline refers
the same share of them (4.975% at a 5% budget). Its intervals re-estimate the thresholds in each resample, so
they are wider than the main ones. Output: `outputs/experiment044_rate_matched_v1/result.json`
(`298b3baf…33a4`), 38 s.

| Screen at a matched 4.975% | v2 | v3 | v3b | v3 − v2 | v3b − v2 |
| --- | ---: | ---: | ---: | --- | --- |
| `combined_50`, composite | 0.785 | 0.792 | 0.794 | +0.0074 [−0.0027, +0.0137] | +0.0094 [−0.0012, +0.0156] |
| `combined_50`, binary label | 0.761 | 0.770 | 0.770 | +0.0089 [−0.0025, +0.0159] | +0.0095 [−0.0025, +0.0165] |
| `combined_50`, "other" referred | 0.208 | 0.231 | 0.242 | +0.023 [+0.007, +0.036] | +0.034 [+0.014, +0.048] |
| binary alone, composite | 0.734 | 0.742 | 0.742 | +0.0084 [−0.0015, +0.0188] | |

At a matched rate the point gain is almost the same as in the primary comparison (+0.0074 against +0.0080),
so the higher false-referral rate explains little of it. With the thresholds re-estimated per resample the
interval includes 0. With the thresholds set on the whole local pool (6,923 normals), v3 refers 5.38% of the
evaluation normals against v2's 5.31% and catches 0.797 of the composite ECGs against 0.791.

## AUROC (for completeness)

| Set | ECGs (positive) | v2 | v3 | v3 − v2 [95% CI] |
| --- | --- | ---: | ---: | --- |
| PTB-XL hard added subset | 266 (41) | 0.712 | 0.746 | +0.034 [+0.006, +0.063] |
| PTB-XL ordinary development | 1,306 (843) | 0.9536 | 0.9566 | +0.0030 [+0.0001, +0.0060] |
| PTB-XL full development | 1,572 (884) | 0.9306 | 0.9347 | +0.0041 [+0.0007, +0.0076] |
| SPH | 21,008 (7,190) | 0.9387 | 0.9404 | +0.0017 [+0.0008, +0.0027] |

These reproduce 043 (whose intervals used seed 43043). The concatenated finding heads do not rank SPH better:
PVC AUROC 0.9900 against 0.9898 for the xECG head (25,577 ECGs, 1,058 PVC; +0.0001 [−0.0017, +0.0019]) and WPW
0.9902 against 0.9916 (25,566 ECGs, 27 WPW; −0.0014 [−0.0044, +0.0009]).

## Numbers for the cardiologist meeting

Pipeline v3 (`combined_50`), threshold set on 200 local normal ECGs, assuming 5% of students have an
athlete-criteria abnormal ECG:

| Budget (share of normals flagged) | Referrals per 1,000 | Share of abnormal caught | Share of athlete-criteria findings caught | Normals referred |
| --- | ---: | ---: | ---: | ---: |
| 1% | 47 | 0.62 | 0.65 | 1.56% |
| 2% | 60 | 0.69 | 0.72 | 2.55% |
| 5% | 94 | 0.78 | 0.80 | 5.74% |
| 10% | 143 | 0.84 | 0.86 | 10.49% |

At 5% that is about 94 referrals per 1,000 students and 40 of the 50 abnormal ones caught, against 91 and 39-40
with v2.

## What it means, in plain language

- At every budget the screen with the JEPA features added catches slightly more abnormal ECGs: about 8 more
  of every 1,000 at 5%, more ST-T, conduction and hypertrophy findings and as many PVC ECGs. The gain passes the
  pre-registered rule, so v3 becomes the candidate pipeline.
- It also flags about 3 more of every 1,000 normal ECGs at the same nominal budget. A post hoc check with the
  false-referral rate held equal kept about the same gain, so the extra flags explain little of it.
- The cost for students is in the "other" ECGs (sinus bradycardia or tachycardia, atrial premature beats): v3
  refers 25% of them against v2's 22%, which undoes most of what v2 gained over v1 (25%) in 037. At a student
  site many of those would be normal variants. SPH counts them neither as caught nor as false referrals.
- v3 needs a second frozen encoder (ECG-JEPA) at the site.
- Refitting the PVC and WPW heads on the joint features (v3b) catches about 1 more PVC ECG in 100, but it
  refers even more "other" ECGs and does not rank PVC or WPW better by AUROC. It carries no decision here.

## Surprises

- 043's SPH AUROC gain was small (+0.0017), yet the gain at the top of the normals is clear at every budget.
  In 037 the reverse happened: two readouts with equal AUROC differed there.
- At 5%, v3 refers more normals than v2 at every size, even with 2,000 normals (5.46% against 5.26%), and
  with 1,000 normals its achieved rate is less steady across draws (within ±1 point in 77% against 88%).
- WPW sensitivity fell from 0.816 to 0.765 (15 ECGs; the interval reaches 0).

## Caveats

- SPH is development data (022-043), an older Chinese hospital cohort at 34% prevalence, whose normals are
  hospital normals with sinus rhythm alone. 043 chose the concatenated readout partly on SPH AUROC, so this
  is a second look at the same data, not an independent confirmation.
- The local pool and evaluation half come from the same hospital, the most favourable case.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs; MI and HYP have 122 and 117.
- The labels are ECG annotation proxies, not confirmed disease or referral decisions.
- One fit per head; only the local normals and the evaluation patients vary.
- The rate-matched check is post hoc and uses oracle thresholds; it describes, it does not test.

## Reproduce

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v3_044
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
    .venv/bin/python -u -m scripts.experiments.rate_matched_044
```

The runner refuses to overwrite an existing output. It needs the 030, 032, 033, 035, 037 and 043 Stage 1
outputs, the JEPA and xECG caches, `outputs/features_challenge_v1/` and the SPH manifest, and reads no raw
waveform.
