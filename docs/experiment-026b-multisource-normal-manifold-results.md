# Experiment 026b results: the normal-ECG manifold fitted on normals from several hospitals

Completed 29 September 2026 under the [frozen protocol](experiment-026b-multisource-normal-manifold.md)
(frozen at commit `b89db3c`, whose file hash the run recorded). Run once at commit `f2e36d6` on CPU in
615 s, three encoder processes with one thread each. Local outputs are in
`outputs/experiment026b_multisource_manifold_v1/` (`result.json` SHA-256 `89a74c1f…f0e5`, per-row scores for
SPH, development and the Challenge calibration groups, `fit_sets.csv` and `run.log`). No PTB-XL calibration
or test ECG and no Challenge test-group ECG was read.

A first launch stopped at the count check before any score was computed: the runner's constant for the
positives of the Chapman/Ningbo set without zero leads was a guess (3,190; the protocol fixes only the 70
excluded records). The constant was corrected to the manifest's 3,207 and the run started again.

## Integrity

- The `ptbxl` arm reproduced 026's saved Mahalanobis scores exactly for all three encoders: the largest
  difference over every development and SPH ECG was 0.0.
- Every cache matched its receipt: the PTB-XL and SPH features through 026's identity, the Challenge features
  against their `metadata.json`, the Challenge split and the Ningbo manifest against theirs.
- Fit-set counts matched the protocol: PTB-XL 5,872, Ningbo 2,581 (126 failing `use_training`), Chapman 819
  (1 policy exclusion), Georgia 1,023 (4), CPSC 2018 551 (0). Arms: `ptbxl` 5,872, `pooled` 10,846,
  `balanced` 2,204, `loso_chapman_ningbo` 7,446, `loso_georgia` 9,823, `loso_cpsc` 10,295.
- No bootstrap draw was skipped in any set.
- The 64 PCA components kept 89-95% of the standardized fit-set variance in every arm; adding hospitals
  changed that by at most 0.8 points.

All intervals below are 95% paired bootstrap intervals (2,000 draws, seed 30030): by patient for SPH and
PTB-XL development, by record for the Challenge families.

## 1. SPH, primary label (21,008 ECGs, 7,190 positive)

AUROC of the distance-from-normal score by fit set. The last column is the supervised Experiment 022 probe,
fitted on PTB-XL with all labels.

| Encoder | `ptbxl` (026) | `pooled` | `balanced` | `loso_chapman_ningbo` | `loso_georgia` | `loso_cpsc` | 022 probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xECG | 0.858 | 0.878 | 0.875 | 0.867 | 0.879 | 0.876 | 0.915 |
| ECG-JEPA | 0.836 | 0.865 | 0.865 | 0.845 | 0.867 | 0.864 | 0.911 |
| CPC | 0.791 | 0.806 | 0.813 | 0.800 | 0.808 | 0.801 | 0.876 |

Each arm minus `ptbxl`, AUROC:

| Arm | xECG | ECG-JEPA | CPC |
| --- | --- | --- | --- |
| `pooled` (primary) | +0.020 [+0.018, +0.022] | +0.028 [+0.026, +0.031] | +0.015 [+0.013, +0.017] |
| `balanced` | +0.017 [+0.015, +0.019] | +0.028 [+0.026, +0.031] | +0.021 [+0.018, +0.025] |
| `loso_chapman_ningbo` | +0.009 [+0.008, +0.010] | +0.008 [+0.007, +0.009] | +0.009 [+0.008, +0.011] |
| `loso_georgia` | +0.021 [+0.019, +0.022] | +0.030 [+0.028, +0.032] | +0.017 [+0.015, +0.019] |
| `loso_cpsc` | +0.018 [+0.016, +0.020] | +0.028 [+0.025, +0.030] | +0.010 [+0.008, +0.012] |

Average precision moved the same way: xECG 0.806 to 0.834 with `pooled` (+0.028 [+0.025, +0.030]), ECG-JEPA
0.792 to 0.825 (+0.034 [+0.031, +0.036]), CPC 0.736 to 0.758 (+0.022 [+0.019, +0.025]). Every SPH interval,
AUROC or AP, lies above 0.

Share of 026's SPH gap to the 022 probe that `pooled` closed: xECG 35% (0.020 of 0.057), ECG-JEPA 38%, CPC
18%.

## 2. PTB-XL development (1,306 ECGs, 843 abnormal)

The home site, which every arm includes. AUROC, then `pooled` and `balanced` minus `ptbxl`:

| Encoder | `ptbxl` | `pooled` | `balanced` | `pooled` - `ptbxl` | `balanced` - `ptbxl` |
| --- | ---: | ---: | ---: | --- | --- |
| xECG | 0.923 | 0.920 | 0.917 | -0.004 [-0.007, -0.001] | -0.006 [-0.012, -0.002] |
| ECG-JEPA | 0.910 | 0.906 | 0.896 | -0.004 [-0.008, -0.001] | -0.014 [-0.020, -0.008] |
| CPC | 0.836 | 0.818 | 0.801 | -0.018 [-0.025, -0.012] | -0.034 [-0.044, -0.026] |

Widening the reference costs a little at the home site: 0.004 AUROC for xECG and JEPA, 0.018 for CPC. The
`balanced` arm, with only 551 PTB-XL normals, costs more.

## 3. Challenge calibration groups per family (secondary)

AUROC on each family's calibration group. `loso_f` and `ptbxl` never saw family f; `pooled` saw f's training
normals.

| Family (ECGs, positive) | Encoder | `ptbxl` | `loso_f` | `pooled` | `balanced` | `loso_f` - `ptbxl` | `pooled` - `loso_f` |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Chapman/Ningbo (4,432, 3,254) | xECG | 0.932 | 0.937 | 0.944 | 0.949 | +0.005 [+0.004, +0.006] | +0.007 [+0.006, +0.009] |
| | ECG-JEPA | 0.941 | 0.945 | 0.953 | 0.955 | +0.004 [+0.003, +0.005] | +0.008 [+0.007, +0.010] |
| | CPC | 0.915 | 0.920 | 0.943 | 0.947 | +0.005 [+0.003, +0.006] | +0.024 [+0.021, +0.027] |
| Georgia (1,718, 1,372) | xECG | 0.895 | 0.896 | 0.901 | 0.907 | +0.001 [-0.002, +0.005] | +0.004 [+0.003, +0.007] |
| | ECG-JEPA | 0.883 | 0.888 | 0.894 | 0.899 | +0.005 [+0.002, +0.008] | +0.006 [+0.004, +0.009] |
| | CPC | 0.864 | 0.872 | 0.874 | 0.874 | +0.007 [+0.003, +0.012] | +0.003 [+0.001, +0.004] |
| CPSC (1,462, 1,279) | xECG | 0.854 | 0.865 | 0.872 | 0.883 | +0.011 [+0.004, +0.017] | +0.007 [+0.005, +0.009] |
| | ECG-JEPA | 0.857 | 0.867 | 0.872 | 0.878 | +0.011 [+0.004, +0.017] | +0.005 [+0.003, +0.006] |
| | CPC | 0.816 | 0.796 | 0.803 | 0.814 | -0.020 [-0.031, -0.007] | +0.007 [+0.005, +0.009] |

Without the 70 Ningbo records with a lead stored as zero (4,362 ECGs, 3,207 positive), every Chapman/Ningbo
AUROC rises by 0.007-0.011, and every contrast is the same to within 0.001.

## Prespecified reading

- **Primary: the multi-hospital normal reference is adopted.** On SPH, xECG, `pooled` minus `ptbxl` is
  +0.020 [+0.018, +0.022]; the interval lies entirely above 0. The distance-from-normal score should be fitted
  on PTB-XL plus the Challenge training normals rather than PTB-XL alone.
- **Secondary encoders:** the same holds for ECG-JEPA (+0.028) and CPC (+0.015); both intervals lie above 0.
- **Secondary arms at SPH:** every arm is above `ptbxl` for every encoder. `balanced` is no better than
  `pooled` for xECG (0.875 against 0.878), equal for JEPA, and better for CPC (0.813 against 0.806).
- **New-site reading:** for xECG and JEPA, the other hospitals predict a held-out hospital's normals better
  than PTB-XL alone, with intervals above 0 at Chapman/Ningbo and CPSC and, for JEPA, at Georgia. xECG at
  Georgia includes 0 (+0.001). CPC at CPSC is the exception: the other hospitals made it worse (-0.020). A
  family's own training normals always add a little more (`pooled` - `loso_f` +0.003 to +0.024, all above 0).
- **Home site:** PTB-XL development AUROC falls slightly for every encoder (intervals below 0).

## What this means for a normal-reference screen at a new site

- **A broader picture of "normal" helps at an unseen hospital, but it closes only about a third of the gap.**
  Fitted on PTB-XL plus four other collections, the xECG distance score reaches 0.878 at SPH, up from 0.858.
  The supervised probe is still ahead at 0.915. The rest of the gap is not a too-narrow reference.
- **Which hospitals are added matters more than how many.** Chapman/Ningbo carries most of the gain: without
  it, SPH improves by only 0.008-0.009; without Georgia or CPSC, it improves as much as with them, and without
  Georgia slightly more. Chapman, Ningbo and SPH are all Chinese hospitals, so the likely reason is population
  or device similarity to SPH, not the number of sites. This could not be separated from the pretraining
  overlap for JEPA and xECG (below), but CPC, which never saw a Challenge record, shows the same pattern.
- **The cost at the home site is small but real** (0.004 AUROC for xECG). A reference that describes several
  hospitals' normals is slightly less tight around any one of them.
- **For the university screen,** this points the same way as 027b. A reference fitted elsewhere should include
  normals as close to the target population as possible, and the best reference will still be local normal
  ECGs from the students themselves (backlog item `local_normal_manifold`). Gathering normal ECGs is also far
  cheaper than labeling abnormal ones.

## Surprises

- The gain is larger at SPH (+0.020) than at the Challenge families the reference had never seen (+0.001 to
  +0.011), for xECG and JEPA, even though SPH is the hospital furthest from PTB-XL. Much of it comes through
  Chapman/Ningbo.
- Removing Georgia raised SPH AUROC slightly for every encoder (xECG 0.879 against 0.878 pooled). Georgia's
  normals, from a US population, do not help a Chinese site.
- The equal-count `balanced` arm, with 2,204 normals, did about as well as `pooled` with 10,846 at SPH
  (within 0.007 for every encoder), and as well or better on every Challenge family. A few hundred normals per
  hospital are enough for the 64-component covariance; the mix of hospitals matters more than the size.
- CPC behaves differently at CPSC: adding other hospitals lowered its CPSC AUROC, and CPC's home-site cost
  was also the largest. This run does not test why.

## Caveats

- ECG-JEPA and xECG were pretrained without labels on Chapman and Ningbo waveforms, so for them the
  Chapman/Ningbo normals are not new data, and the Chapman/Ningbo held-out readout is not an unseen-hospital
  test. CPC has seen no Challenge record and still shows the same SPH pattern. SPH is unseen by every encoder.
- The Challenge normal (sinus rhythm alone) is stricter than PTB-XL's NORM and SPH's normal code. None of
  these normals are young adults, and a normal ECG is an annotation, not proof of health.
- The Challenge split is by record, so `pooled` on a family it saw may share patients with that family's
  training normals. SPH, the primary test, and the `loso` readouts do not depend on this.
- The fit sets differ in size (2,204 to 10,846 ECGs); `balanced` also has far fewer PTB-XL normals, which
  explains most of its larger home-site cost.
- These results are on the frozen features of one checkpoint per encoder, one fit per arm. The bootstrap
  holds the fits fixed. The development patients were inspected by earlier experiments. No threshold,
  sensitivity or specificity is reported; a distance score still flags filtering, device and noise
  differences as well as disease.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_multisource_manifold026b
```

The runner refuses to overwrite an existing `result.json`. It reads the raw Chapman, Georgia and CPSC 2018
records (official checksums verified) to apply the quality policy to the training normals, and it needs the
Experiment 025/026 caches, the Experiment 022 SPH features and `outputs/features_challenge_v1/`.
