# Experiment 022b results: the readout trained on PTB-XL plus several hospitals, read on SPH

Completed 29 September 2026 under the [frozen protocol](experiment-022b-multisource-readout.md) (frozen at
commit `b102118`, whose file hash the run recorded). Run once at commit `af2fdf3` on CPU in 1,189 s (179 s of
it the quality check), three encoder processes with one thread each. Local outputs are in
`outputs/experiment022b_multisource_readout_v1/` (`result.json` SHA-256 `217e8663…865c`, `predictions.npz`,
`training_rows.csv`, `run.log`). No PTB-XL test ECG, no Challenge test-group ECG and no PTB-XL calibration ECG
as a test was read. This covers the backlog items `ningbo_sph_transfer` and `multisource_lso`.

Before the run, a synthetic smoke test and a pass of the runner up to its count check (stopped before any
readout was fitted) were used to catch errors. Neither computed a score.

## Integrity

- The `ptbxl` arm reproduced 022 exactly for all three encoders: every SPH and development probability
  differed by 0.0, and the SPH AUROCs are identical.
- Its operating point reproduced 027 exactly (Platt coefficients, thresholds and SPH calibrated
  probabilities, difference 0.0).
- Every input matched its receipt: the 020, 022 and 027 artifacts, the JEPA and xECG caches, the Challenge
  features, the Challenge split and the Ningbo manifest.
- The counts matched the protocol: 17,083 PTB-XL and 22,494 Challenge training ECGs (330 excluded by the
  training quality policy), the eight arm sizes, the calibration pools, and every evaluation set. No
  bootstrap draw was skipped. Every readout converged (223-326 iterations).

All intervals are 95% paired bootstrap intervals (2,000 draws, seed 31031): by patient for SPH and PTB-XL
development, by record for the Challenge families.

## 1. SPH ranking (21,008 ECGs, 7,190 positive)

AUROC by training set:

| Encoder | `ptbxl` (022) | `pooled` | `balanced` | `ptbxl_chapman_ningbo` | `ptbxl_ningbo` | `loso_chapman_ningbo` | `loso_georgia` | `loso_cpsc` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xECG | 0.915 | 0.939 | 0.937 | 0.936 | 0.935 | 0.926 | 0.939 | 0.935 |
| ECG-JEPA | 0.911 | 0.934 | 0.934 | 0.930 | 0.929 | 0.919 | 0.934 | 0.930 |
| CPC | 0.876 | 0.897 | 0.897 | 0.893 | 0.892 | 0.884 | 0.896 | 0.893 |

Differences in AUROC:

| Contrast | xECG | ECG-JEPA | CPC |
| --- | --- | --- | --- |
| `pooled` − `ptbxl` (primary) | +0.024 [+0.021, +0.026] | +0.023 [+0.021, +0.025] | +0.021 [+0.019, +0.024] |
| `balanced` − `ptbxl` | +0.022 [+0.019, +0.025] | +0.022 [+0.020, +0.025] | +0.021 [+0.019, +0.024] |
| `ptbxl_chapman_ningbo` − `ptbxl` | +0.021 [+0.019, +0.023] | +0.019 [+0.017, +0.021] | +0.017 [+0.015, +0.020] |
| `ptbxl_ningbo` − `ptbxl` | +0.020 [+0.018, +0.022] | +0.018 [+0.016, +0.019] | +0.016 [+0.014, +0.018] |
| `loso_chapman_ningbo` − `ptbxl` | +0.012 [+0.010, +0.013] | +0.008 [+0.006, +0.009] | +0.008 [+0.007, +0.010] |
| `loso_georgia` − `ptbxl` | +0.024 [+0.022, +0.026] | +0.023 [+0.021, +0.025] | +0.020 [+0.018, +0.023] |
| `loso_cpsc` − `ptbxl` | +0.020 [+0.018, +0.023] | +0.019 [+0.017, +0.021] | +0.018 [+0.015, +0.020] |
| `pooled` − `loso_chapman_ningbo` | +0.012 [+0.011, +0.013] | +0.015 [+0.014, +0.017] | +0.013 [+0.011, +0.015] |

Average precision moved the same way: xECG 0.885 to 0.917 with `pooled` (+0.033 [+0.030, +0.035]), JEPA
0.882 to 0.911 (+0.029), CPC 0.831 to 0.861 (+0.030). Every SPH interval, AUROC or AP, lies above 0.

The raw readout probabilities (before any Platt step) are best calibrated when only Chapman/Ningbo is
added: xECG Brier 0.108 (`ptbxl`), 0.090 (`ptbxl_chapman_ningbo`), 0.120 (`pooled`); calibration intercept
−0.57, −0.51 and −1.42. Georgia and CPSC-Extra, which are 80-100% positive, push the pooled probabilities
up.

## 2. SPH operating point (each arm's own calibration data)

Sensitivity and specificity at the at-least-95%-sensitivity threshold, with the Platt probability's Brier
score and ECE. `D` is the change in `|sensitivity − 0.95|` against `ptbxl`; negative means closer to 95%.

| Arm | Head | Sensitivity | Specificity | Brier | ECE | D |
| --- | --- | --- | --- | ---: | ---: | --- |
| `ptbxl` (027) | xECG | 0.922 [0.915, 0.928] | 0.646 [0.638, 0.654] | 0.124 | 0.105 | |
| | JEPA | 0.916 [0.909, 0.922] | 0.663 [0.655, 0.671] | 0.127 | 0.115 | |
| | CPC | 0.971 [0.967, 0.974] | 0.303 [0.295, 0.311] | 0.202 | 0.237 | |
| `pooled` | xECG | 0.902 [0.895, 0.908] | 0.797 [0.790, 0.804] | 0.123 | 0.149 | +0.020 [+0.014, +0.026] |
| | JEPA | 0.894 [0.886, 0.901] | 0.800 [0.794, 0.807] | 0.120 | 0.140 | +0.022 [+0.016, +0.028] |
| | CPC | 0.926 [0.920, 0.932] | 0.575 [0.567, 0.584] | 0.178 | 0.218 | +0.003 [−0.005, +0.012] |
| `balanced` | xECG | 0.941 [0.936, 0.947] | 0.672 [0.664, 0.680] | 0.155 | 0.209 | −0.019 [−0.025, −0.014] |
| | JEPA | 0.934 [0.929, 0.940] | 0.680 [0.673, 0.689] | 0.148 | 0.197 | −0.019 [−0.025, −0.013] |
| | CPC | 0.958 [0.954, 0.963] | 0.419 [0.410, 0.427] | 0.205 | 0.265 | −0.012 [−0.016, −0.008] |
| `ptbxl_chapman_ningbo` | xECG | 0.811 [0.802, 0.820] | 0.914 [0.909, 0.918] | 0.094 | 0.051 | +0.111 [+0.104, +0.119] |
| | JEPA | 0.813 [0.804, 0.822] | 0.900 [0.895, 0.905] | 0.098 | 0.052 | +0.102 [+0.095, +0.110] |
| | CPC | 0.842 [0.833, 0.851] | 0.757 [0.750, 0.765] | 0.140 | 0.119 | +0.087 [+0.077, +0.098] |
| `ptbxl_ningbo` | xECG | 0.841 [0.833, 0.850] | 0.881 [0.876, 0.886] | 0.096 | 0.060 | +0.081 [+0.074, +0.088] |
| `loso_chapman_ningbo` | xECG | 0.958 [0.953, 0.963] | 0.562 [0.554, 0.571] | 0.195 | 0.271 | −0.020 [−0.030, −0.010] |
| | JEPA | 0.949 [0.944, 0.954] | 0.550 [0.542, 0.559] | 0.204 | 0.280 | −0.033 [−0.037, −0.025] |
| `loso_georgia` | xECG | 0.904 [0.897, 0.910] | 0.792 [0.785, 0.798] | 0.132 | 0.160 | +0.018 [+0.012, +0.024] |
| `loso_cpsc` | xECG | 0.814 [0.805, 0.824] | 0.910 [0.905, 0.914] | 0.091 | 0.045 | +0.108 [+0.100, +0.115] |

The full table for every arm and head is in `result.json` (`sph_operating`). Only one threshold transfers to
SPH by 027's rule (sensitivity interval includes 0.95): `loso_chapman_ningbo` JEPA, 0.949, bought with a
specificity of 0.550.

**Post hoc, not prespecified.** To separate the ranking gain from the threshold shift: at the SPH
sensitivity of the PTB-XL readout (0.922 for xECG, chosen on SPH, so not a deployable threshold), the pooled
xECG readout would pass 0.746 of SPH normals against 0.646, and `ptbxl_chapman_ningbo` 0.724. For JEPA at
0.916: 0.743 against 0.662. For CPC at 0.971: 0.349 against 0.302.

## 3. Held-out hospital families (`multisource_lso`)

Each family's calibration group, scored by the PTB-XL readout and by the readout trained (and calibrated)
without that family. `pooled` saw the family's training rows and is the in-distribution reference. AUROC:

| Family (ECGs, positive) | Encoder | `ptbxl` | `loso_f` | `pooled` | `loso_f` − `ptbxl` | `pooled` − `loso_f` |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Chapman/Ningbo (4,432, 3,254) | xECG | 0.953 | 0.973 | 0.991 | +0.020 [+0.017, +0.022] | +0.018 [+0.015, +0.020] |
| | JEPA | 0.961 | 0.971 | 0.987 | +0.010 [+0.009, +0.013] | +0.016 [+0.013, +0.018] |
| | CPC | 0.926 | 0.944 | 0.976 | +0.018 [+0.015, +0.022] | +0.032 [+0.028, +0.036] |
| Georgia (1,718, 1,372) | xECG | 0.899 | 0.925 | 0.937 | +0.026 [+0.020, +0.032] | +0.012 [+0.010, +0.016] |
| | JEPA | 0.904 | 0.922 | 0.935 | +0.018 [+0.012, +0.023] | +0.013 [+0.010, +0.016] |
| | CPC | 0.882 | 0.906 | 0.915 | +0.024 [+0.018, +0.031] | +0.009 [+0.007, +0.011] |
| CPSC (1,462, 1,279) | xECG | 0.931 | 0.944 | 0.952 | +0.013 [+0.006, +0.021] | +0.008 [+0.004, +0.012] |
| | JEPA | 0.934 | 0.943 | 0.950 | +0.009 [+0.002, +0.015] | +0.007 [+0.004, +0.011] |
| | CPC | 0.893 | 0.901 | 0.911 | +0.008 [−0.002, +0.018] | +0.010 [+0.006, +0.014] |

Adding Ningbo alone also helps the two hospitals it is unrelated to: `ptbxl_ningbo` − `ptbxl` on Georgia
is +0.025 [+0.020, +0.030] (xECG), +0.018 (JEPA), +0.026 (CPC); on CPSC +0.016 [+0.009, +0.023], +0.010,
+0.008 [+0.000, +0.017].

Without the 70 Ningbo records with a lead stored as zero, every Chapman/Ningbo contrast is the same to within
0.003.

At the held-out families the threshold behaved as at SPH: trading sensitivity for specificity. For xECG,
trained and calibrated without the family, sensitivity and specificity were 0.931 and 0.676 at Georgia
(PTB-XL readout 0.964 and 0.425), 0.840 and 0.880 at CPSC (0.922 and 0.694), and 0.979 and 0.660 at
Chapman/Ningbo (0.957 and 0.646).

## 4. PTB-XL development and the hard cases (secondary)

AUROC, standard label:

| Encoder | Set | `ptbxl` | `pooled` | `ptbxl_chapman_ningbo` | `ptbxl_ningbo` | `pooled` − `ptbxl` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| xECG | full (1,572) | 0.939 | 0.927 | 0.932 | 0.933 | −0.013 [−0.018, −0.008] |
| | original (1,306) | 0.960 | 0.955 | 0.958 | 0.958 | −0.005 [−0.009, −0.000] |
| | added (266, 41 positive) | 0.726 | 0.655 | 0.676 | 0.687 | −0.072 [−0.113, −0.032] |
| JEPA | full | 0.936 | 0.927 | 0.930 | 0.932 | −0.010 [−0.015, −0.005] |
| | original | 0.957 | 0.955 | 0.957 | 0.957 | −0.002 [−0.006, +0.003] |
| | added | 0.734 | 0.668 | 0.685 | 0.696 | −0.066 [−0.109, −0.025] |
| CPC | full | 0.889 | 0.874 | 0.876 | 0.879 | −0.015 [−0.021, −0.008] |
| | original | 0.916 | 0.914 | 0.914 | 0.915 | −0.002 [−0.009, +0.004] |
| | added | 0.575 | 0.506 | 0.510 | 0.519 | −0.069 [−0.111, −0.030] |

At the home site the extra hospitals cost little on the ordinary ECGs (0.000-0.005 for `pooled` and the
Ningbo arms) but a lot on the hard cases: on 022's added subset every multi-source arm loses 0.04-0.07 AUROC
for xECG and JEPA (intervals below 0), and pooled CPC falls to chance. The one exception is CPC without
Chapman/Ningbo (−0.012 [−0.040, +0.015]). `balanced`, which gives PTB-XL only a quarter of the weight,
costs the most (xECG 0.726 to 0.619).

## Prespecified reading

- **Primary: the multi-source readout is adopted.** On SPH, xECG, `pooled` − `ptbxl` is +0.024
  [+0.021, +0.026]; the interval lies entirely above 0 and the gain is above the 0.005 "negligible" line.
  The readout should be fitted on PTB-XL plus the Challenge training groups rather than PTB-XL alone.
- **Secondary encoders:** the same holds for ECG-JEPA (+0.023) and CPC (+0.021). Every secondary arm
  improves SPH AUROC and AP for every encoder, intervals above 0.
- **Ningbo (prespecified answer): helps.** Both Ningbo contrasts lie above 0 for xECG: Chapman/Ningbo added
  alone +0.021 [+0.019, +0.023], and added to the other hospitals +0.012 [+0.011, +0.013].
- **Operating point:** no arm fixes the transfer. For JEPA and xECG, `pooled` moves SPH sensitivity further
  from 95% (D above 0), while `balanced` and `loso_chapman_ningbo` bring it closer (D below 0) at a cost in
  Brier score and ECE. Only `loso_chapman_ningbo` JEPA meets 027's transfer rule.
- **Held-out families:** a readout trained on PTB-XL plus the other hospitals ranks an unseen family better
  than PTB-XL alone, with intervals above 0 in eight of nine cases (CPC at CPSC includes 0). A family's own
  training rows add a further 0.007-0.032.
- **Home site:** PTB-XL full-development AUROC falls by 0.010-0.015 (intervals below 0), almost all of it on
  the hard added subset.

## Does Ningbo help?

**Yes, for ranking ECGs at a new hospital; no, for the fixed 95%-sensitivity threshold.**

- **Ranking: better.** Adding Ningbo's labeled training ECGs to PTB-XL raises SPH AUROC by 0.020 for xECG
  (0.915 to 0.935), and by 0.016-0.018 for JEPA and CPC. Ningbo is most of what the full multi-source readout
  gains: removing Chapman/Ningbo from it cuts the SPH gain from 0.024 to 0.012. It also helps the US Georgia
  hospital (+0.025) and CPSC (+0.016), so it is not only similarity to SPH. CPC, which never saw a Ningbo
  waveform in pretraining, gains too.
- **Threshold: worse unless recalibrated locally.** When the threshold is calibrated on PTB-XL plus Ningbo,
  SPH sensitivity drops from 0.922 to 0.841 (xECG) while specificity rises from 0.646 to 0.881. The readout
  ranks Ningbo's own calibration ECGs almost perfectly (AUROC 0.99), so the 95% cut placed there is too
  strict for a hospital it has not seen. The better ranking is real (post hoc, at SPH's 0.922 sensitivity the
  pooled readout passes 75% of normals instead of 65%), but it needs a threshold set at the target site.
- **Home site: slightly worse.** PTB-XL's hardest ECGs rank worse (0.726 to 0.687 for xECG with Ningbo alone).
- **In one sentence:** add Ningbo when training the readout, and do not trust a threshold calibrated on
  hospitals the readout was trained on.

## What this means for the project

- The best external ranking so far is the xECG readout fitted on PTB-XL plus the Challenge training groups:
  0.939 at SPH, against 0.915 for 022. The gain (+0.024) is about two thirds of what replacing CPC with
  xECG gave at SPH in 022 (+0.039), and it costs no GPU time.
- Together with 027b and 026b, the pattern is consistent: more hospitals help a score **rank**, but no pooled
  calibration gives a threshold that transfers. The university screen needs local labeled ECGs to set its
  threshold (backlog `site_recalibration`), whatever readout it uses.
- Calibrating on groups from the same hospitals as the training data is optimistic. A later threshold study
  should calibrate on hospitals the readout was not trained on (like `loso_chapman_ningbo` here), or on the
  target site.

## Surprises

- The Ningbo readout's threshold missed at SPH far more than PTB-XL's (0.81-0.84 against 0.92), even though
  its ranking was better. In 027b, with the PTB-XL readout, adding the same calibration hospitals moved xECG
  sensitivity by only 0.002; here, with a readout trained on those hospitals, it moved by 0.02 (`pooled`) to
  0.11 (Chapman/Ningbo).
- Dropping Georgia from training cost nothing at SPH (`loso_georgia` 0.939, the same as `pooled`), as 026b
  also found for the normal reference.
- Equal family weighting did not help SPH ranking and cost the most at the home site. It came closer to 95%
  sensitivity, probably because it gives less weight to the easy Chapman/Ningbo calibration ECGs.
- Adding the other hospitals hurt the PTB-XL hard cases by about 0.07 AUROC (`pooled`), much more than the
  0.005 on the ordinary ECGs. Those ECGs are the ones the project label dropped (022); the Challenge label's stricter
  normal (sinus rhythm alone) may pull the boundary away from them.

## Caveats

- ECG-JEPA and xECG were pretrained without labels on Chapman and Ningbo waveforms, so the Chapman/Ningbo
  held-out readout is not an unseen-hospital test for them. CPC has seen no Challenge record and shows the
  same pattern. SPH is unseen by every encoder.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM and SPH's code 1, and
  the Challenge training rows are 78% positive (CPSC-Extra 100%). This shifts the raw probabilities and the
  boundary; the SPH gain is in ranking, which prevalence does not affect.
- The Challenge split is by record; patients may repeat across train and calibration groups, which flatters
  the in-distribution `pooled` readout on a family and its calibration. SPH, the primary test, and the `loso`
  readouts do not depend on this.
- The arms differ in training size (17,083 to 39,577 ECGs). Adding Ningbo alone (+9,952) gains 0.020 while
  025 found PTB-XL gains above 4,000 labels under 0.01, so the gain is unlikely to be size alone, but size
  and hospital are not separated here.
- CPSC records longer than 10 s are read through their centred window; the CPSC calibration group has 183
  negatives. The added development subset has 41 positives, so its values are imprecise.
- One fit per arm and encoder, one run, one checkpoint per encoder; the bootstrap holds fits and thresholds
  fixed. The label is an ECG annotation proxy, not confirmed disease or a referral decision. No age or other
  subgroup analysis was done.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_multisource_readout022b
```

The runner refuses to overwrite an existing `result.json`. It reads the raw Chapman, Georgia, CPSC 2018 and
CPSC-Extra training records (official checksums verified) to apply the quality policy, and needs the 020,
022 and 027 outputs, the JEPA and xECG caches and `outputs/features_challenge_v1/`.
