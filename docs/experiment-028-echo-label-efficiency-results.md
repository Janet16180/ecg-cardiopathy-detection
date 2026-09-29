# Experiment 028 results: label efficiency for echo-confirmed structural heart disease

Completed 29 September 2026 under the [frozen protocol](experiment-028-echo-label-efficiency.md). CPU only,
from the cached Experiment 023 v3 features. Validation patients only; the EchoNext test split was not opened.
EchoNext is credentialed, so this page holds aggregates only. The per-draw table and the full aggregate result
are in `outputs/experiment028_echo_label_efficiency_v1/` (`draws.csv`, `result.json`).

Integrity checks:
- The features, 023 predictions and rows matched the hashes recorded by Experiment 023.
- The feature keys matched the usable training rows and all validation rows, in order.
- At N = all the primary heads reproduced 023 exactly: maximum probability difference 0.0 for every encoder,
  and identical AUROCs (xECG 0.8381, ECG-JEPA 0.8228, CPC 0.8116, age/sex 0.6547).

Rows:
- Training pool: 71,823 usable ECGs from 26,023 patients, 37,744 positive (prevalence 0.5255).
- Evaluation: 4,575 usable validation ECGs, one per patient, 1,974 positive (0.431).
- Every draw had N ECGs from N distinct patients. Every fit converged.

Runtime: 6,655 s (1 h 51 min) on four worker processes with one BLAS thread each, with other CPU work
running on the machine. Most of it was the cross-validated secondary readout at N = 8,000 and 16,000.

## Validation AUROC by labeled training ECGs

Primary readout (fixed C = 0.01). Mean over 20 paired draws, SD in parentheses; N = all is one fit.

| N | xECG | ECG-JEPA | CPC | Age/sex |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.752 (0.019) | 0.734 (0.022) | 0.728 (0.029) | 0.631 (0.046) |
| 250 | 0.773 (0.013) | 0.764 (0.015) | 0.758 (0.012) | 0.649 (0.008) |
| 500 | 0.787 (0.009) | 0.777 (0.008) | 0.772 (0.008) | 0.651 (0.005) |
| 1,000 | 0.801 (0.006) | 0.791 (0.007) | 0.783 (0.006) | 0.653 (0.004) |
| 2,000 | 0.813 (0.005) | 0.801 (0.004) | 0.791 (0.004) | 0.653 (0.002) |
| 4,000 | 0.824 (0.002) | 0.809 (0.002) | 0.798 (0.002) | 0.655 (0.001) |
| 8,000 | 0.831 (0.002) | 0.815 (0.002) | 0.804 (0.002) | 0.655 (0.000) |
| 16,000 | 0.836 (0.001) | 0.820 (0.001) | 0.809 (0.001) | 0.655 (0.000) |
| all (71,823) | 0.838 | 0.823 | 0.812 | 0.655 |

For reference, the 023 `tabular` head (cart measurements with age and sex, fitted on all training rows)
scores 0.737.

The 2.5-97.5 percentile range over draws is wide at N = 100, for example 0.713-0.776 for xECG and
0.665-0.761 for CPC. It narrows to 0.791-0.811 and 0.774-0.792 at N = 1,000.

Average precision follows the same order. At N = 250 it is 0.730 (xECG), 0.711 (ECG-JEPA), 0.698 (CPC)
and 0.567 (age/sex). At N = all it is 0.809, 0.790, 0.779 and 0.570. Read it against the prevalence of
0.431.

## Paired AUROC differences

Primary readout. Mean difference over draws, with the number of the 20 draws above 0 in parentheses.

| N | xECG - CPC | JEPA - CPC | xECG - JEPA | CPC - age/sex | JEPA - age/sex | xECG - age/sex |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | +0.024 (19) | +0.006 (11) | +0.018 (20) | +0.097 (20) | +0.103 (20) | +0.121 (20) |
| 250 | +0.016 (17) | +0.006 (15) | +0.009 (16) | +0.108 (20) | +0.115 (20) | +0.124 (20) |
| 500 | +0.015 (18) | +0.006 (16) | +0.010 (17) | +0.120 (20) | +0.126 (20) | +0.136 (20) |
| 1,000 | +0.018 (20) | +0.008 (18) | +0.010 (20) | +0.130 (20) | +0.138 (20) | +0.148 (20) |
| 2,000 | +0.022 (20) | +0.010 (20) | +0.012 (20) | +0.138 (20) | +0.147 (20) | +0.160 (20) |
| 4,000 | +0.026 (20) | +0.011 (20) | +0.015 (20) | +0.144 (20) | +0.154 (20) | +0.169 (20) |
| 8,000 | +0.027 (20) | +0.011 (20) | +0.016 (20) | +0.149 (20) | +0.160 (20) | +0.176 (20) |
| 16,000 | +0.027 (20) | +0.012 (20) | +0.016 (20) | +0.154 (20) | +0.165 (20) | +0.181 (20) |

At N = all, 2,000 paired patient bootstrap draws (seed 28028, no invalid draws):

| Contrast | AUROC difference | 95% interval |
| --- | ---: | --- |
| xECG - CPC | +0.026 | [+0.020, +0.033] |
| JEPA - CPC | +0.011 | [+0.005, +0.018] |
| xECG - JEPA | +0.015 | [+0.010, +0.021] |
| CPC - age/sex | +0.157 | [+0.140, +0.174] |
| JEPA - age/sex | +0.168 | [+0.151, +0.185] |
| xECG - age/sex | +0.183 | [+0.167, +0.200] |

## Prespecified reading

All on the primary readout.

1. **Encoder contrasts (90% rule).**
   - At N = 250 every encoder is better than age and sex alone (20 of 20 draws). The encoder contrasts do not
     meet the rule: xECG - CPC 17 of 20, JEPA - CPC 15, xECG - JEPA 16. So each is "no clear difference".
   - At N = 1,000 xECG is better than CPC (20 of 20) and than ECG-JEPA (20 of 20). ECG-JEPA is better than
     CPC (18 of 20). Every encoder is better than age and sex.
   - Described at the other budgets: xECG - CPC meets the rule at N = 100 (19 of 20), 500 (18) and from
     1,000 on; xECG - JEPA at N = 100 (20) and from 1,000 on; JEPA - CPC from 1,000 on.
2. **Near-ceiling budget.**
   - 95% of the own N = all AUROC is first reached on average at N = 500 for CPC (0.772 against 0.771), and at
     N = 1,000 for ECG-JEPA and xECG.
   - The stricter target, 95% of the gain over chance, is first reached at N = 4,000 by all three encoders.
3. **Beating the cart measurements (0.737).**
   - On the mean, xECG reaches it at N = 100 (0.752); ECG-JEPA and CPC at N = 250.
   - In at least 18 of 20 draws, all three reach it at N = 250 (xECG 20, ECG-JEPA 18, CPC 18).
   - At N = 100, 16 of 20 xECG draws reach it, against 7 for ECG-JEPA and 10 for CPC.
   - Age and sex alone never reach it.

## Secondary readout (C chosen by training-only cross-validation)

The secondary readout changes the N = 250 encoder reading but no other conclusion:
- At N = 250 it favors xECG over CPC (20 of 20), ECG-JEPA over CPC (18 of 20) and xECG over ECG-JEPA
  (20 of 20). The N = 1,000 reading is unchanged.
- Cross-validation chose C = 0.001 for xECG in 16-20 of 20 draws at every budget, and at N = all. That
  lifts xECG at small budgets: 0.789 at N = 250 against 0.773 with C = 0.01, and 0.811 against 0.801 at
  N = 1,000.
- ECG-JEPA mostly chose 0.001 up to N = 2,000 and 0.01 from N = 8,000. CPC mostly chose 0.01 from
  N = 1,000.
- At N = all it scores 0.840 (xECG), 0.823 (ECG-JEPA), 0.812 (CPC) and 0.655 (age/sex).

As prespecified, this is a sensitivity analysis. It suggests that the fixed C = 0.01 under-regularizes
xECG's 1,024 features at small budgets.

## Findings

- A frozen encoder with a linear head needs few echo labels to beat the cart's own measurements. With 250
  labeled ECGs, all three encoders exceed the tabular head (0.737) in at least 18 of 20 draws. With 100
  labels xECG does so on average, but only in 16 of 20 draws.
- Returns diminish but do not stop. Going from 1,000 to all 71,823 labels adds 0.029-0.037 AUROC, and from
  N = 250 on each budget doubling still adds 0.005-0.014. Near-ceiling performance (95% of the gain
  over chance) needs about 4,000 labels.
- xECG has the best mean AUROC at every budget. Its lead over CPC is smaller with few labels: +0.015 to
  +0.016 at N = 250-500, growing to +0.027 from N = 8,000. Its lead over ECG-JEPA is +0.018 at N = 100,
  +0.009-0.010 at N = 250-1,000 and +0.016 from N = 8,000. With C = 0.01, the draws do not separate the
  encoders reliably at N = 250 and 500.
- This differs from PTB-XL ([Experiment 025](experiment-025-label-efficiency-results.md)). There, ECG-JEPA
  and xECG led CPC by a constant 0.034-0.040 from 100 labels on, and matched CPC's all-label AUROC with 100
  labels. On echo-confirmed disease the gaps are smaller and ECG-JEPA is only slightly ahead of CPC.
- Which ECGs get labeled matters most below 500 labels: the SD over draws is 0.019-0.029 at N = 100 and
  0.006-0.007 at N = 1,000.
- Age and sex stay at 0.649-0.655 from 250 labels on. Every encoder adds 0.10-0.18 AUROC over them.

For a university cohort with a few hundred echo-labeled ECGs, these results suggest the following. Expect
AUROC around 0.76-0.79 on a population like EchoNext's, above what cart measurements give. xECG with a
more strongly regularized head is the best first choice. Confirm it on local patients.

## Caveats

- EchoNext waveforms were median filtered, clipped and standardized by the dataset authors. The encoders
  see a shifted input, and the encoder gaps may partly reflect how each one reacts to that preprocessing.
- The validation split has one ECG per patient. Draws also use one ECG per patient. Only N = all uses the
  repeated ECGs of patients with several.
- Draws permute ECGs, so patients with many ECGs are more likely to be drawn at small N (the 025 design).
- One academic centre, patients referred for echocardiography, prevalence 0.431. These AUROCs describe
  that referred population, not screening.
- Draw variability reflects label sampling on one fixed validation set. It is not a confidence interval
  for new patients.
- The validation patients were already scored in 023; results are exploratory. The `tabular` comparator is
  fitted on all training rows, not per budget.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_echo_label_efficiency028
```

The runner refuses to overwrite an existing `result.json`.
