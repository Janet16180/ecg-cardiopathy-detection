# Experiment 022 results: external readout on SPH

Completed 28 September 2026 under the [frozen protocol with addenda v2 and v3](experiment-022-sph-external-readout.md).
This is the first project result on an independent hospital. Nothing was fitted, tuned or selected on SPH,
and SPH was scored once. No PTB-XL calibration or test ECG was used. Local outputs are in
`outputs/experiment022_sph_external_v3/`.

**Integrity.**
- The four Experiment 020 heads reproduced their saved development probabilities exactly.
- JEPA and xECG features recomputed for 32 training ECGs matched the cached features exactly (difference 0.0).
- The SPH counts matched addendum v2: 25,577 ECGs of 24,642 patients, 13,818 negative, 7,190 positive, and
  4,569 with an undefined primary label, which leaves 21,008 scored.
- Features for 2,356 PTB-XL ECGs that were missing from the JEPA and xECG caches were extracted with the
  cached features' own code: 2,058 training and 298 added development ECGs.
- One pre-run check changed. The lead-identity check first required every sampled record's worst
  Einthoven residual to stay below 0.02 mV. It failed on A16746, which has a single corrupted aVR sample
  (1.18 mV). The clean manifest already flags that record as `limb_identity_violated`. The check was
  changed to the per-record median residual (largest 0.0008 mV across the 64 records), and the worst record
  is still reported. This affects that pre-run check only; no feature or score changed.

## 1. Every encoder holds up at a new hospital, and the ranking is preserved

AUROC and average precision on SPH, primary label (21,008 ECGs, 7,190 positive):

| Head | AUROC | AP |
| --- | ---: | ---: |
| `xecg_standard` | 0.915 | 0.885 |
| `jepa_standard` | 0.911 | 0.882 |
| `jepa_standard_c01` | 0.904 | 0.877 |
| `cpc_project_full` | 0.882 | 0.843 |
| `cpc_age_sex_standard` | 0.876 | 0.832 |
| `cpc_standard` (primary) | 0.876 | 0.831 |
| `age_sex_standard` | 0.658 | 0.508 |

| Contrast (paired whole-patient bootstrap, 2,000 draws) | AUROC difference | 95% interval |
| --- | ---: | --- |
| `cpc_standard` minus `age_sex_standard` (primary) | +0.218 | [+0.210, +0.226] |
| `jepa_standard` minus `cpc_standard` | +0.036 | [+0.032, +0.039] |
| `xecg_standard` minus `cpc_standard` | +0.039 | [+0.036, +0.043] |
| `cpc_age_sex_standard` minus `cpc_standard` | +0.0002 | [−0.0004, +0.0008] |
| `cpc_standard` minus `cpc_project_full` | −0.007 | [−0.008, −0.006] |

On PTB-XL full development, the same heads scored:

| Head | PTB-XL full development AUROC |
| --- | ---: |
| `cpc_standard` | 0.889 |
| `jepa_standard` | 0.936 |
| `xecg_standard` | 0.939 |

The populations differ, so the protocol makes no test of the drop. Two things hold anyway:
- JEPA and xECG stay about 0.035-0.04 above CPC at SPH, almost exactly their PTB-XL advantage (Experiment 025
  found +0.035).
- The ranking from PTB-XL development survived an independent country, device and labeling team.

**Age and sex matter much less at SPH.** They reach only 0.658, against 0.756 on PTB-XL, and add nothing to
CPC features (+0.0002). SPH patients are younger: their mean age is about 50, against about 60 in the other
sources.

**The project label edges ahead at SPH.** The head trained on the project label is slightly better there than
the standard-label head (−0.007 for standard minus project). On PTB-XL full development it was the other way
round (+0.010). Both effects are small.

## 2. The secondary label and the quality policy change nothing

- The secondary label also counts sinus-variant-only records as negative (23,809 ECGs). The AUROCs move by at
  most 0.005: CPC 0.874, JEPA 0.909, xECG 0.914.
- Dropping the 98 records the quality policy would exclude (64 near the ADC limit, 34 above 20 mV) changes
  primary AUROC by less than 0.001 for every encoder.

## 3. By superclass (each superclass versus normal ECGs)

| Superclass | Positives | CPC | JEPA | xECG |
| --- | ---: | ---: | ---: | ---: |
| MI | 255 | 0.922 | 0.965 | 0.978 |
| STTC | 5,037 | 0.900 | 0.925 | 0.927 |
| CD | 2,365 | 0.840 | 0.893 | 0.899 |
| HYP | 227 | 0.936 | 0.973 | 0.972 |

Conduction disturbances are the hardest for every encoder, and they account for most positives after STTC.
MI and HYP are rare in SPH (255 and 227 ECGs), so their AUROCs are less precise.

## 4. PTB-XL hard cases: better with JEPA and xECG, but still hard

Standard label on PTB-XL full development. "Added" are the ECGs the project label dropped (266 ECGs,
41 positive), where CPC was at chance in Experiment 020:

| Head | Full (1,572) | Original (1,306) | Added (266) |
| --- | ---: | ---: | ---: |
| `cpc_project_full` | 0.879 | 0.921 | 0.506 |
| `cpc_standard` | 0.889 | 0.916 | 0.575 |
| `jepa_standard` | 0.936 | 0.957 | 0.734 |
| `xecg_standard` | 0.939 | 0.960 | 0.726 |
| `jepa_standard_c01` | 0.936 | 0.956 | 0.751 |

The stronger encoders rank the hard cases far better than CPC, but still well below the other ECGs. The added
subset has only 41 positives, so these values are imprecise.

## Interpretation

- **The CPC encoder generalizes to a new hospital:** 0.876 AUROC, far above age and sex alone.
- **The released frozen encoders keep their advantage externally.** xECG (0.915) and ECG-JEPA (0.911) are the
  best available representations for this task. Together with Experiment 025 (JEPA and xECG with 100 labels
  beat CPC with 15,359), this points the project's baseline toward a frozen JEPA or xECG with a simple
  readout.
- **The C=0.1 JEPA variant is not better at SPH** (0.904 against 0.911 with C=0.01), so the fixed C=0.01
  readout is not handicapping JEPA.

**Caveats.**
- SPH is heavily band-pass filtered compared with PTB-XL (see the [SPH EDA review](sph-echonext-eda-review.md)).
  This is a real shift, which the encoders survived, but it may affect encoders differently: xECG works at
  100 Hz, CPC and JEPA at 250 Hz.
- The SPH label follows the PTB-XL superclass rule only as closely as AHA codes allow. SPH codes "normal ECG"
  as exclusive, and lists 5,037 STTC but only 255 MI records.
- Neither JEPA nor xECG listed SPH in its pretraining data. HuBERT-ECG did, but it is not tested here.
- This is one run, one fit per head, and a readout fitted on PTB-XL only. There is no calibration or
  threshold, so sensitivity and specificity are not reported.

## Reproduce

```bash
uv run --no-sync python -u -m scripts.experiments.run_sph_external022 --stage profile
uv run --no-sync python -u -m scripts.experiments.run_sph_external022 --stage run
```

The profile projected 2,486 s against the 3,600 s ceiling. The run extracted SPH features in 632 s (read 70,
CPC 63, JEPA 216, xECG 282).
