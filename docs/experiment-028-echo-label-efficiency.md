# Experiment 028: label efficiency for echo-confirmed structural heart disease

**Frozen 28 September 2026, before any validation score of this experiment is computed.** Echo labels are
expensive, and a university cohort would have few of them. EchoNext is the only project dataset whose label
is measured heart disease rather than a reading of the ECG. This experiment asks how many echo-labeled ECGs a
frozen encoder needs for a useful structural heart disease readout. It is CPU only: no encoder is trained and
no feature is extracted. It reuses the cached features of Experiment 023 v3 and the draw logic of
[Experiment 025](experiment-025-label-efficiency.md). The EchoNext test split stays closed.

## Question

With 100 to 16,000 echo-labeled training ECGs, what validation AUROC does a fixed linear readout reach on
each frozen encoder, how does each compare with age and sex alone and with the cart measurements, and how
many labels does each need to come close to its all-label result?

## Inputs

- Features: `outputs/experiment023_echonext_v3/features.npz`, keys `train_ecg_keys`, `train_cpc`,
  `train_jepa`, `train_xecg` (71,823 usable training ECGs) and `val_all_ecg_keys`, `val_all_cpc`,
  `val_all_jepa`, `val_all_xecg` (all 4,626 validation ECGs). Its SHA-256 must equal
  `outputs_sha256["features.npz"]` in the 023 `result.json`, and `predictions.npz` must match likewise.
- Rows: `data/processed/echonext_250hz_v1/rows.csv`, for the label, `patient_key`, `use`, `age_at_ecg` and
  `sex`. Its SHA-256 must equal the `rows` hash recorded in the 023 identity.
- `train_ecg_keys` must equal, in order, the `ecg_key` of the rows with `split == "train"` and `use`, and
  `val_all_ecg_keys` those with `split == "val"`. Any mismatch stops the run.

## Rows and label

- **Label:** `shd_moderate_or_greater_flag`, the dataset's composite.
- **Training pool:** the 71,823 usable training ECGs, from 26,023 patients, prevalence 0.5255 (counted on
  training metadata only). Patients have 1 to 146 ECGs (median 1).
- **Evaluation:** the 4,575 usable validation ECGs, one per patient. The 51 unusable validation ECGs are not
  scored.

## Encoders

| Name | Inputs | Width |
| --- | --- | ---: |
| `cpc` | 023 pooled CPC features | 512 |
| `jepa` | 023 pooled ECG-JEPA features | 768 |
| `xecg` | 023 pooled xECG features | 1,024 |
| `age_sex` | `echonext_readout.age_sex_inputs`: age / 10 and male, as in 023 | 2 |

## Label budgets and draws

- N in {100, 250, 500, 1,000, 2,000, 4,000, 8,000, 16,000}, plus all 71,823.
- For each N below all, 20 draws with seeds 28028 + draw (draw 0-19), using `label_efficiency.draw_subset`
  unchanged: the pool is permuted with `numpy.random.default_rng(seed)`, the first ECG of each patient in
  that order is kept, and the first round(N x 0.5255) positives and the remaining negatives are taken in the
  same order. Each draw has N ECGs from N distinct patients at the pool prevalence.
- Because the permutation is over ECGs, a patient with many ECGs is more likely to be drawn at small N. This
  is the 025 design and is kept for comparability.
- Draws depend only on the pool table, so every encoder and readout sees the same draws (paired).
- N = all is one fit on the whole pool, including patients with several ECGs. It is exactly the 023 fit.

## Readouts

- **Primary:** `full_development.fit_logistic` exactly as in 023: train-only `StandardScaler`, L2 logistic
  regression, `C=0.01`, L-BFGS, `max_iter=5000`, `tol=1e-8`, float64, seed 42, one BLAS thread.
- **Secondary:** `label_efficiency.select_c` and `fit_logistic_c`: C chosen from {0.001, 0.01, 0.1, 1} by
  5-fold `StratifiedGroupKFold` (groups = patients, `shuffle=True`, `random_state` = the draw seed, 28028 at
  N = all) inside the drawn training subset, highest mean fold AUROC, ties to the smaller C, then refitted on
  the whole subset. Only training rows are used for the choice.
- A solver that does not converge stops the run.

## Integrity

Before any draw is scored, the primary head of each encoder is fitted at N = all and scored on the
evaluation ECGs. Its probabilities must match the 023 `predictions.npz` entry of the same head to 1e-8, and
its AUROC must equal the 023 `result.json` value (`cpc` 0.8116, `jepa` 0.8228, `xecg` 0.8381, `age_sex`
0.6547) to 1e-12. Otherwise the run stops.

## Metrics

- Validation AUROC and average precision per encoder, readout and N: mean, SD and the 2.5 and 97.5
  percentiles over the 20 draws.
- Paired AUROC and AP differences per N and draw: `xecg - cpc`, `jepa - cpc`, `xecg - jepa`, `cpc - age_sex`,
  `jepa - age_sex` and `xecg - age_sex`. Reported as the mean and the fraction of draws above 0.
- At N = all: one fit per encoder and readout; the same AUROC differences with 2,000 paired patient bootstrap
  draws (`full_development.patient_bootstrap`, seed 28028).

## Prespecified reading

All on the primary readout.

1. **Encoder contrasts.** At N = 250 and N = 1,000, as in 025, encoder A is "better" than encoder B if
   A - B AUROC is positive in at least 18 of 20 draws (90%); otherwise "no clear difference". The same rule
   is also reported at every other budget, as a description.
2. **Near-ceiling budget.** The smallest N below all at which an encoder's mean AUROC reaches 0.95 of its own
   N = all AUROC. Because AUROC starts at 0.5, the smallest N reaching 0.5 + 0.95 x (AUROC at all - 0.5),
   95% of its gain over chance, is also reported.
3. **Beating the cart measurements.** The smallest N below all at which an encoder's mean AUROC reaches the
   023 `tabular` AUROC (cart measurements with age and sex, 0.7371), and the smallest N at which at least 18
   of 20 draws reach it.

"None" is reported when no budget below all qualifies. The secondary readout is a sensitivity analysis and
does not change the reading.

## Caveats

- EchoNext waveforms were median filtered, clipped and standardized by the dataset authors; the encoders see
  a shifted input, as in 023.
- The validation split has one ECG per patient, while the training pool has several for many patients; only
  N = all uses the repeated ECGs.
- One academic centre, patients referred for echocardiography, validation prevalence 0.431. Not screening.
- Draw variability reflects label sampling on one fixed validation set; it is not a confidence interval
  for new patients.
- Validation patients were already scored by 023; results are exploratory.
- The `tabular` comparator is fitted on all training rows; it is not re-fitted at each budget.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_echo_label_efficiency028
```

`scripts/experiments/run_echo_label_efficiency028.py` uses `ecg_experiment/echo_label_efficiency.py` and
`ecg_experiment/label_efficiency.py`. It hashes every input, source file and this protocol into the result,
refuses to overwrite an existing run, runs four worker processes with one BLAS thread each, and writes
`outputs/experiment028_echo_label_efficiency_v1/result.json` and a per-draw `draws.csv`. EchoNext is
credentialed: per-row outputs stay under `outputs/`, and the results page,
`docs/experiment-028-echo-label-efficiency-results.md`, holds aggregates only.
