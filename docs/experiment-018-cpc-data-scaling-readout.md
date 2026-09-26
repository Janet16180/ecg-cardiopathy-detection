# Experiment 018: frozen development readout after data scaling

**Frozen 26 September 2026 before feature extraction or new development
predictions.** The [v3 training study](experiment-018-cpc-data-scaling-v3-results.md)
completed two ordinary CPC continuation arms with identical initialization,
objective, optimizer and 115,359 record exposures. This follow-up asks whether
the new-cohort encoder gives a better **fixed frozen-feature classifier** than
the matched old-pool continuation on the project's PTB diagnostic-annotation
proxy. It is not a clinical health or referral outcome.

## Fixed comparison

Evaluate three frozen encoders: the unchanged experiment-004 CPC encoder,
the old-pool continuation checkpoint SHA-256
`5275ec6199433fbc7f779d6fe0e0dc433291f23c86c403d13cace873ee6ac91d`,
and the new-cohort continuation checkpoint SHA-256
`a8a1e79187ba8c74c7f2cf9e535efcb8e67ce943c4e60f5fabd6a9d262521c51`.
The checkpoint [audit](../outputs/experiment018_cpc_data_scaling_v3/training_audit.json)
has SHA-256 `45f8d2e1459fda371b20481e5d07dae1c34b0ce627306ab9d8a8f907d6796748`.

Use the historical experiment-004 250 Hz PTB waveform cache and its original
train-only mean/std for all three. For each ECG, take the CPC encoder's GRU
contexts, concatenate mean and maximum over tokens within each five-second
half, then average the two halves into exactly 512 features. The same feature
operation is applied to all encoders in eval mode with no updates or
augmentation. Fit a separate StandardScaler and L2 logistic classifier on
training patients only for each encoder and label budget: fixed `C=0.01`,
unweighted classes, intercept, L-BFGS, `max_iter=5000`, `tol=1e-8`, seed 42,
one BLAS thread. Reject a nonconverged fit; do not tune C or select epochs on
development patients.

The primary budget is the clean 15,359 PTB training labels. The fixed
1,518-label subset is secondary. Both use exactly the same 1,306-record,
1,173-patient PTB development split. Verify ID, patient and split agreement
against the CPC cache, and verify no train/development patient overlap. Select
only development rows from the clean held-out reference manifest; calibration
and test rows are never featurized, fitted, or scored.

The primary contrast is **new minus old development AUROC at 15,359 labels**.
Secondary descriptive contrasts are new minus old at 1,518 labels and new
minus unchanged starting encoder at each budget. Report development AUROC,
average precision and 2,000-draw paired patient-bootstrap 95% intervals for
AUROC differences, seed 18045. Keep all ECGs from a sampled patient together;
discard and count single-class draws. Do not fit calibration, choose a
threshold, or evaluate test patients. Because development has been inspected
in earlier studies, this is exploratory and cannot establish external
clinical improvement.

## Admission and evidence

Before development inference, verify the old 250 Hz cache's full SHA-256,
manifest and checkpoint hashes. A real-data V100 profile uses training ECGs
only, the actual loader and all three frozen encoders. Project full feature
extraction using the measured record throughput, with a 1.5 margin plus a
900-second reserve for six CPU logistic fits, paired bootstrap, hashes and
reporting. Require the projected new study work to fit 7,200 seconds. The
full run rechecks input identity, writes local feature/prediction artifacts
and an aggregate result receipt, and never mutates raw or cached waveforms.
One GPU job at a time uses the shared project lock.

Results and commands are in the [development-only result](experiment-018-cpc-data-scaling-readout-results.md).
The local artifacts live under `outputs/experiment018_cpc_data_scaling_readout_v1/`.
