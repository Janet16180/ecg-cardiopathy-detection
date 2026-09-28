# Experiment 025 results: label efficiency of frozen encoders

Completed 28 September 2026 under the [frozen protocol](experiment-025-label-efficiency.md). Development
patients only; no calibration or test ECG was fitted or scored. CPU only, from cached frozen features. Local
outputs are in `outputs/experiment025_label_efficiency_v1/` (`result.json`, per-draw `draws.csv`, and the
N = all development probabilities).

**Integrity.** Refitting Experiment 020's `cpc_standard` head from the cached CPC features reproduced its
saved development probabilities exactly (maximum difference 0.0). Every cache matched its extraction receipt.

**Rows.**

- Training pool: 15,359 ECGs from 13,351 patients, 9,487 positive (prevalence 0.6177). Of the 17,417
  Experiment 020 training ECGs, 334 have no standard label and 1,724 are absent from the released caches.
- Evaluation: all 1,306 original development ECGs (1,173 patients, 843 positive). None dropped.
- On these rows the standard label is identical to the project label (0 disagreements in training and in
  evaluation). As a consequence, CPC at N = all reproduces Experiment 018's full-label AUROC (0.920794).

Runtime: 1,430 seconds on four worker processes with one BLAS thread each (xECG took 1,223 s, JEPA 872 s,
CPC 624 s, released ECG-CPC 613 s, age/sex 71 s). The run completed on the first attempt, with no solver
failure.

## Development AUROC by labeled training ECGs

Primary readout (fixed C = 0.01). Mean over 20 paired draws, SD in parentheses; N = all is one fit.

| N | CPC | ECG-JEPA | xECG | Released ECG-CPC | Age/sex |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.883 (0.011) | 0.922 (0.007) | 0.921 (0.008) | 0.894 (0.012) | 0.767 (0.028) |
| 250 | 0.897 (0.008) | 0.934 (0.006) | 0.931 (0.007) | 0.911 (0.007) | 0.773 (0.019) |
| 500 | 0.904 (0.005) | 0.942 (0.005) | 0.939 (0.006) | 0.918 (0.006) | 0.776 (0.013) |
| 1,000 | 0.910 (0.002) | 0.947 (0.004) | 0.947 (0.004) | 0.927 (0.005) | 0.774 (0.008) |
| 2,000 | 0.914 (0.002) | 0.952 (0.002) | 0.952 (0.002) | 0.933 (0.003) | 0.777 (0.006) |
| 4,000 | 0.916 (0.001) | 0.955 (0.001) | 0.957 (0.001) | 0.937 (0.002) | 0.779 (0.004) |
| all (15,359) | 0.921 | 0.959 | 0.962 | 0.943 | 0.779 |

Average precision follows the same order. At N = 250 it is 0.948 (CPC), 0.967 (JEPA), 0.965 (xECG),
0.956 (released ECG-CPC) and 0.863 (age/sex). At N = all it is 0.961, 0.979, 0.981, 0.972 and 0.867.

## Paired AUROC differences

Mean difference over draws, with the number of the 20 draws above 0 in parentheses.

| N | JEPA - CPC | xECG - CPC | xECG - JEPA | Released - CPC | CPC - age/sex | JEPA - age/sex | xECG - age/sex | Released - age/sex |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | +0.039 (20) | +0.038 (20) | -0.001 (10) | +0.012 (15) | +0.116 (20) | +0.155 (20) | +0.154 (20) | +0.127 (20) |
| 250 | +0.037 (20) | +0.034 (20) | -0.003 (3) | +0.014 (19) | +0.124 (20) | +0.161 (20) | +0.158 (20) | +0.138 (20) |
| 500 | +0.038 (20) | +0.036 (20) | -0.002 (4) | +0.014 (20) | +0.128 (20) | +0.166 (20) | +0.164 (20) | +0.143 (20) |
| 1,000 | +0.037 (20) | +0.036 (20) | -0.001 (11) | +0.016 (20) | +0.137 (20) | +0.173 (20) | +0.173 (20) | +0.153 (20) |
| 2,000 | +0.038 (20) | +0.038 (20) | +0.000 (12) | +0.019 (20) | +0.137 (20) | +0.175 (20) | +0.175 (20) | +0.156 (20) |
| 4,000 | +0.039 (20) | +0.040 (20) | +0.001 (18) | +0.021 (20) | +0.137 (20) | +0.176 (20) | +0.177 (20) | +0.158 (20) |

At N = all, with 2,000 paired whole-patient bootstrap draws (no invalid draws):

| Contrast | AUROC difference | 95% interval |
| --- | ---: | --- |
| JEPA - CPC | +0.039 | [+0.029, +0.050] |
| xECG - CPC | +0.041 | [+0.031, +0.052] |
| xECG - JEPA | +0.003 | [-0.002, +0.008] |
| Released ECG-CPC - CPC | +0.022 | [+0.013, +0.032] |
| CPC - age/sex | +0.142 | [+0.119, +0.167] |
| JEPA - age/sex | +0.181 | [+0.157, +0.207] |
| xECG - age/sex | +0.183 | [+0.160, +0.209] |
| Released ECG-CPC - age/sex | +0.164 | [+0.140, +0.190] |

## Prespecified reading

At N = 250 and at N = 1,000 the reading is the same:

- ECG-JEPA and xECG are each better than CPC (20 of 20 draws).
- Released ECG-CPC is better than CPC (19 of 20 at N = 250, 20 of 20 at N = 1,000).
- Every encoder is better than age and sex alone (20 of 20).
- xECG versus ECG-JEPA: no clear difference (3 of 20 draws favor xECG at N = 250, 11 of 20 at N = 1,000).

The CPC AUROC at N = all is 0.921. ECG-JEPA and xECG exceed it on average at N = 100 (0.922 and 0.921),
the smallest budget tested. Released ECG-CPC first reaches it at N = 1,000 (0.927). CPC and age/sex do not
reach it below N = all.

## Secondary readout (C chosen by training-only cross-validation)

The secondary readout does not change any conclusion. Its mean AUROC is within 0.004 of the primary at every
budget and encoder, and the prespecified reading is identical at N = 250 and 1,000. At N = all it scores
0.923 (CPC), 0.960 (JEPA), 0.962 (xECG), 0.943 (released ECG-CPC) and 0.779 (age/sex). At N = 100-250,
cross-validation chose C = 0.001 in 15-16 of 20 draws for xECG and 10-12 for ECG-JEPA. From N = 1,000 on it
chose C = 0.01 in most draws for every encoder except age/sex. The full per-budget table is in `result.json`.

## Findings

- For a small labeled set, ECG-JEPA or xECG features give the best development AUROC among the cached
  encoders. With 100 labels either one matches what our CPC encoder reaches with all 15,359 labels.
- The gap to CPC is about 0.034-0.040 AUROC at every budget from 100 to 4,000. It does not narrow as labels
  grow.
- ECG-JEPA and xECG cannot be separated. Mean differences are at most 0.0032 in either direction. The draws split
  at most budgets, and the N = all interval includes 0.
- Released ECG-CPC sits between CPC and the two leaders at every budget. Its gap to our CPC grows from
  +0.012 at N = 100 to +0.022 at N = all.
- Draw-to-draw SD shrinks from about 0.01 at N = 100 to about 0.001 at N = 4,000. Which ECGs get labeled
  matters mainly below 500 labels.
- Age and sex alone reach 0.767-0.779 at every budget. Every encoder adds 0.116-0.183 AUROC over them.

## Caveats

- The development patients were inspected by earlier experiments; these results are exploratory.
- Each encoder is one feature set (one checkpoint, preprocessing and pooling). Differences mix pretraining
  data, objective and architecture.
- Draw variability reflects label sampling on one fixed evaluation set. It is not a confidence interval
  for new patients.
- The pool and the evaluation set exclude the ECGs the project label dropped, which Experiment 020 found are
  the hard cases. On these rows the standard label equals the project label.
- The label is the PTB-XL standard superclass label, a diagnostic annotation proxy, not a clinical outcome.

## Reproduce

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.experiments.run_label_efficiency025
```

The runner refuses to overwrite an existing `result.json`.
