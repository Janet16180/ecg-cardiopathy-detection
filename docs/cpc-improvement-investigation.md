# CPC improvement investigation backlog

**Status, 25 September 2026:** evidence review and ranked design complete; see
the [research findings](cpc-improvement-research-2026-09-25.md) and the recommended
[CPU-only equal-width local-readout proposal](cpc-local-readout-v1.md).
Implementation and verification remain pending; no new fit, executable
manifest or training job. This task does not change the priority or status of
Experiments 011–013, and it does not restart deferred Experiment 010.

## Question

Which affordable, controlled change is most likely to improve our compact ECG
CPC model, especially with 1,518 training labels, without mistaking a change in
data, readout, or compute budget for an architecture gain?

## Evidence to start from

- [Current model findings](model-findings-report.md) separate the compact CPC,
  frozen probes, fine-tuned models, and released S4 ECG-CPC checkpoint. Do not
  compare their scores as if training and evaluation were matched.
- [Experiment 006](../outputs/experiment006_cpc_tokenization/report.md) found
  learned chunks better than fixed chunks, but both remained below native-grid
  CPC. Preserve the native grid as a control before testing another tokenization.
- [Experiment 009](../outputs/experiment009_cpc_prediction_mismatch/report.md)
  found that an ordinary local-feature branch beat the tested prediction-error
  branch. Keep local morphology information available to the classifier.
- [Experiment 017](experiment-017-clean-replication-results.md) found a small
  two-seed limited-label gain from a morphology-template branch over its
  convolution control; patient-bootstrap intervals included zero. Treat this
  as a lead to investigate, not an established improvement.
- [Experiment 010](experiment-010-crosslead.md) is deferred: observed cache I/O
  made its full suite exceed the two-hour planning ceiling. Its checkpoint is
  evidence of one SSL epoch, not a classification outcome.
- The [released ECG-CPC audit](released-ecg-cpc.md) used a different pretrained
  S4 encoder and frozen mean pooling. It suggests readout choices to examine,
  but its scores are not a matched comparison against compact CPC.

## Candidate directions to rank

1. **Readout and local morphology:** audit where useful local features are lost
   between the CPC encoder and classifier; test one modest readout change
   against an equal-width/equal-budget control. Use the 009 and 017 findings
   to choose the hypothesis, without tuning to their development scores.
2. **Fine-tuning stability:** inspect training and development trajectories,
   checkpoint selection, and representation/readout behavior before proposing
   another optimizer or schedule. A falling training loss with flat
   development performance is a question to diagnose, not proof of overfitting.
3. **Pretraining data scale and quality:** quantify source mix, duplicates,
   signal quality, and overlap before using the larger clean unlabeled union.
   Compare a frozen current pool with a separately labeled scaling cohort;
   retain patient-safe splits and equal optimization exposure.
4. **CPC objective or context:** only after the cheaper audits, revisit a
   targeted prediction-horizon or cross-lead hypothesis with matched parameter
   count, batch exposure and objective controls. First estimate the complete
   data-path cost; do not automatically resume 010.
5. **Released-model readout:** test a documented frozen readout alternative to
   mean pooling on the released S4 features if cached features and source
   identities make a CPU-only comparison possible. Keep it separate from a
   compact-CPC improvement claim.

## Next deliverable and decision rule

Review the existing result and failure receipts, available waveform/feature
caches, source provenance, and development-set reuse. Produce a short ranked
table with the expected mechanism, matched control, label budget, estimated
full-path wall time, memory, and what result would falsify each idea. Choose
**one** affordable hypothesis for a new versioned protocol; do not run a sweep.
For a GPU candidate, profile the complete path and pass the existing 7,200-second
planning gate before comparative training. Freeze seeds, patient splits,
selection rules and analyses before results. Calibration/test patients remain
closed until a separate promotion decision.
