# Experiment 025: label efficiency of frozen encoders

**Frozen 28 September 2026, before any development score is computed.** Backlog item `label_efficiency` in
`docs/experiment-priorities.md` (branch `plan/024-priorities`): with a few hundred labels, as a university
cohort might have, which frozen encoder should the readout use? The literature review on the same branch
(`docs/literature-review-2026-09-28.md`) notes that, in another benchmark, ECG-JEPA learns fastest with few
labels and ECG-CPC has a higher ceiling. This experiment measures it on our PTB-XL development patients. It is CPU only: no encoder is trained and no
feature is extracted. Calibration and test patients stay closed.

## Question

At 100 to 4,000 labeled training ECGs, which of the cached frozen encoders gives the best development AUROC
with a fixed linear readout, and how does each compare with an age and sex baseline?

## Encoders

Cached frozen features only. Rows are selected by PTB-XL ECG ID.

| Name | Cache | Width | Identity check |
| --- | --- | ---: | --- |
| `cpc` | `outputs/experiment020_full_development_v2/features.npz`, rows in `cohorts(ptb_table())` order | 512 | SHA-256 equals Experiment 020's `features_sha256` |
| `jepa` | `data/processed/pretrained/ecg-jepa-full-public` (`features.npy`, `ecg_ids.npy`) | 768 | SHA-256 equals Experiment 014's recorded input hashes |
| `xecg` | `outputs/experiment016_xecg_probe_finetune/features` | 1024 | SHA-256 equals the extraction receipt |
| `released_cpc` | `outputs/experiment004_cpc_40k/released_features` | 512 | SHA-256 equals `output_sha256` in its metadata |
| `age_sex` | `demographics` of `ecg_experiment.full_development` | 3 | none |

The JEPA and released ECG-CPC caches also contain calibration and test ECGs. They are memory-mapped and only
the selected training and development rows are read; no calibration or test row is fitted or scored.

## Rows

- **Training pool:** Experiment 020 training ECGs (`cohorts(ptb_table())["train"]`) with a defined standard
  label and present in all four feature caches. A count on training metadata only, before this freeze, gives
  15,359 ECGs from 13,351 patients, prevalence 0.6177. Dropped: 334 ECGs without a standard label, and 1,724
  clean-union training ECGs with a standard label that are absent from the released caches (the caches were
  built for the project-labeled rows). The pool is therefore exactly the project-labeled training ECGs,
  labeled here with the standard label.
- **Evaluation:** the 1,306 original development ECGs (1,173 patients) of Experiment 020 with a defined
  standard label, present in all caches. All 1,306 qualify. The 298 added development ECGs are not used,
  because the released caches do not contain them.
- **Label:** the PTB-XL standard superclass label (`ptb_table()["standard"]`): negative when NORM is the only
  diagnostic superclass, positive when MI, STTC, CD or HYP is present.

## Label budgets and draws

- N in {100, 250, 500, 1000, 2000, 4000}, plus all 15,359.
- For each N below all, 20 draws with seeds 25025 + draw (draw 0-19). A draw permutes the pool with
  `numpy.random.default_rng(seed)`, keeps the first ECG of each patient in that order, and takes the first
  round(N x 0.6177) positives and the remaining negatives in the same order. Each draw has N ECGs from N
  distinct patients at the pool prevalence.
- Draws depend only on the pool table, so every encoder and readout sees the same draws (paired).
- N = all is one fit on the whole pool, including patients with several ECGs.

## Readouts

- **Primary:** `fit_logistic` exactly as in Experiment 020: train-only `StandardScaler`, L2 logistic
  regression, `C=0.01`, L-BFGS, `max_iter=5000`, `tol=1e-8`, float64, seed 42.
- **Secondary:** the same head with C chosen from {0.001, 0.01, 0.1, 1} by 5-fold
  `StratifiedGroupKFold` cross-validation (groups = patients, `shuffle=True`, `random_state` = the draw seed,
  25025 at N = all) inside the drawn training subset. The scaler is fitted on each inner training fold. The
  highest mean fold AUROC wins; ties go to the smaller C. The head is then refitted on the whole subset.
  Only training data are used for the choice.
- **Age/sex:** missing ages are filled with the median age of the fitted training rows.
- A solver that does not converge stops the run, as in Experiment 020. One BLAS thread per process; encoders
  may run in parallel processes.

## Integrity

Before any label-efficiency score, `cpc_standard` of Experiment 020 is refitted from the cached CPC features
on its original 17,083 training rows and must reproduce its saved development probabilities to 1e-8. This
confirms the row order of the CPC cache; it recomputes a score that Experiment 020 already reported.

## Metrics

- Development AUROC and average precision per encoder, readout and N: mean, SD and the 2.5 and 97.5
  percentiles over the 20 draws.
- Paired AUROC and AP differences per N and draw: `jepa - cpc`, `xecg - cpc`, `xecg - jepa`,
  `released_cpc - cpc`, and each encoder minus `age_sex`. Reported as the mean and the fraction of draws
  above 0.
- At N = all: one fit per encoder and readout; the same AUROC differences with 2,000 paired whole-patient
  bootstrap draws (`patient_bootstrap`, seed 25025) on the evaluation ECGs.

## Prespecified reading

On the primary readout at N = 250 and N = 1000, encoder A is "better" than encoder B if A - B AUROC is
positive in at least 18 of 20 draws (90%). Otherwise the result is "no clear difference". The secondary
readout is a sensitivity analysis and does not change this reading.

Also reported: the smallest N below all at which each encoder's mean primary AUROC reaches the primary CPC
AUROC at N = all, or "none".

## Caveats

- The development patients were inspected by earlier experiments; results are exploratory.
- Each encoder is one feature set: one checkpoint, one preprocessing and one pooling. Differences mix
  pretraining data, objective and architecture.
- Draw variance measures label sampling only, on one fixed evaluation set; it is not a confidence interval
  for new patients.
- The pool excludes the ECGs that the project label dropped, which Experiment 020 found are the hard cases.
- The label is the PTB-XL standard superclass label, a diagnostic annotation proxy, not a clinical outcome.

## Execution

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.experiments.run_label_efficiency025
```

`scripts/experiments/run_label_efficiency025.py` uses `ecg_experiment/label_efficiency.py`. It hashes every
input cache, receipt, source file and this protocol into the result, refuses to overwrite an existing run,
and writes `outputs/experiment025_label_efficiency_v1/result.json` and a per-draw `draws.csv`. The results
go to `docs/experiment-025-label-efficiency-results.md`.
