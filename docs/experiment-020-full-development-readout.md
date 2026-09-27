# Experiment 020: full-development readout with demographic baselines

**Frozen 27 September 2026, before any new development feature or score is computed.** The user asked for
this re-evaluation after the [dataset EDA](eda-pipeline-review.md) found three problems with how results have
been read: the project label drops the hardest held-out ECGs, no demographic baseline is reported, and the
recording device may act as a shortcut. No encoder is trained. Calibration and test patients stay closed.

## Questions

1. How does the best existing frozen encoder score when the development set keeps the ECGs the project label
   dropped, scored with the standard PTB-XL superclass label?
2. How much does it add over a baseline that uses only age and sex?
3. Does performance differ by recording device, and can the frozen features identify the device?

## Encoder and features

The unchanged starting CPC encoder (`outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt`), which had the best
point AUROC in Experiments 018 and 019 (0.92079 at full labels). It uses the historical train-only
normalization and the 512 mean/max pooled GRU context features of Experiment 018, unchanged.

Records in the historical 250 Hz cache (`data/processed/cpc_pool_40k`) are read from it. Added development
records are not in that cache; they are read from the raw 500 Hz PTB-XL files and converted with the
historical transform (`ecg_experiment.cpc_input_audit.historical_resample`). Before any development record
is converted, the transform must reproduce 32 existing cache rows bit for bit (16 training and 16
development rows, evenly spaced). The raw files are checked against the official PTB-XL SHA256 manifest.

## Patients and labels

- **Training:** PTB-XL training patients only (folds 1-8), as in the historical cache and clean selection.
- **Original development:** the 1,306 ECGs of 1,173 patients used by every earlier readout.
- **Full development:** the original development ECGs, plus every fold-9 ECG left out by the project label
  whose patient is not a calibration patient. Patients who appear in no partition had never been evaluated;
  they join development. The 15 fold-9 ECGs of calibration patients and all of fold 10 stay closed.
- **Project label:** the existing clean targets (15,359 full and 1,518 limited training labels).
- **Standard label:** `ecg_experiment.eda.ptbxl.superclass_label`: negative when NORM is the only
  diagnostic superclass, positive when MI, STTC, CD or HYP is present, undefined when a record has no
  diagnostic statement. Rhythm and form statements are ignored. Records with an undefined standard label
  are dropped from standard-label analyses and counted.
- Training records use only rows of the clean training selection or clean-union PTB training rows; the 66
  constant-lead records excluded in the clean rerun stay excluded.

## Models

All heads use Experiment 018's fixed readout: a train-only `StandardScaler` and L2 logistic regression with
`C=0.01`, L-BFGS, `max_iter=5000`, `tol=1e-8`, no class weights, seed 42 and one BLAS thread. Nothing is
tuned or selected on development patients.

| Head | Inputs | Training labels |
| --- | --- | --- |
| `cpc_project` | CPC features | project label, full and limited budgets |
| `cpc_standard` | CPC features | standard label, all clean training PTB records where it is defined |
| `age_sex_project`, `age_sex_standard` | age / 10, male, missing-age indicator | as above |
| `cpc_age_sex_standard` | CPC features, age / 10, male, missing-age indicator | standard label |

Age 300 (the PTB-XL placeholder for ages over 89) is treated as missing. Missing ages are filled with the
training median and flagged by the indicator.

## Analyses

1. **Integrity:** `cpc_project` at both budgets on the original 1,306 ECGs must reproduce Experiment 018's
   saved `initial` probabilities to 1e-6 and its AUROC (0.920794 full, 0.912964 limited). The run stops if
   it does not.
2. **Hard cases (primary):** AUROC and average precision of `cpc_project` (full budget) and `cpc_standard`
   on full development with the standard label, also split into original and added ECGs.
3. **Demographic baseline (primary):** `cpc_standard` minus `age_sex_standard` AUROC on full development,
   and `cpc_age_sex_standard` minus `cpc_standard`.
4. **Device:** AUROC of `cpc_standard` within each device with at least 50 full-development ECGs and at
   least 10 of each class. Separately, a device probe: the same fixed logistic head, trained on training
   patients to predict "CS100 3" versus any other device from CPC features, scored on full development.
5. **Heart rate:** for reference, the share of standard-label negatives that the project label left out
   because of sinus brady- or tachycardia, and `cpc_project` AUROC with and without them.

Differences use 2,000 paired whole-patient bootstrap draws with seed 20020, counting invalid single-class
draws. All results are exploratory: the original development patients were inspected by earlier
experiments, and this is one encoder and one seed.

## Execution

`scripts/experiments/run_full_development_readout020.py` has two stages. `profile` hashes every input,
checks the transform on the 32 cache rows, and times feature extraction for 512 training ECGs on the V100
under the shared GPU lock. It projects the full run with a 1.5 margin plus 900 seconds and requires it to
fit 3,600 seconds. `run` requires a matching passed profile, extracts features, fits the heads on CPU,
writes local features and predictions to `outputs/experiment020_full_development_v1/`, and records an
aggregate `result.json`. The results go to `docs/experiment-020-full-development-readout-results.md`.
