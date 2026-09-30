# NLP-inspired ECG architecture studies on the fixed 25k pool

**Prospective common protocol, 26 September 2026.** The user requested execution
of the queued NLP-inspired architecture studies on the existing 25k training
subset. This document fixes the shared data and evaluation contract before new
architecture outcomes. Each experiment's own protocol must pin its operator,
matched control, initialization, implementation, and feasible runtime budget
before training. A cost profile or SSL loss is not a downstream result.

## Scope and training data

Experiments 011 (KDA versus CKDA with GRU reference), 012 (mixed-scale temporal
hybrid versus matched local-support control), and 013 (Mamba-3 versus Mamba-2
with GRU reference) are separate prospective comparisons. Existing transformer,
HuBERT, and word2vec-style results remain historical controls rather than new
25k runs. The architecture arms learn from scratch on raw twelve-lead ECGs;
released language or genomic checkpoint weights are not used.

Reuse the exact source-stratified 25,000-record selection from Experiment 019's
verified 115,359-record training cache. The selection uses seed 18046 and its
canonical selected-index SHA-256 is
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`.
The source quotas are 15,170 MIMIC, 3,746 PTB-XL, 2,072 Chapman, 2,066
Georgia, 1,334 CPSC 2018, and 612 CPSC Extra. Use the existing training-only
normalization, preserve the cache and source manifests, and reject a changed
input hash. Keep each record's five-second halves independent in causal CPC.
Use no diagnostic labels in the self-supervised stage.

The initial matched screening budget is 115,359 waveform exposures per arm,
cycling through fresh permutations of the selected 25k records using the
Experiment 019 exposure-order seed 18047. At batch 128 this is 902 optimizer
updates, with a shorter final batch. Share the record order, causal CPC
prediction horizons and loss, normalization, optimizer family, and copied
common-stem/prediction-head initialization within each experiment. Report
parameter count, actual updates, exposures, memory, and wall time for every
arm. Equal exposure does not imply equal compute. Do not warm-start just one
architecture from an earlier CPC checkpoint.

## Development readout and interpretation

Screen frozen encoders with the same historical PTB-XL waveform cache and
train-only normalization, using the exact 15,359 clean full-label training
records and nested 1,518-label subset. Keep the 1,306 development ECGs from
1,173 patients separate. Use each encoder's prespecified mean/max context
pooling and a fixed train-only standardized logistic readout with the same
regularization, solver, and seed across arms. The exact feature dimensions and
readout settings must be frozen in each experiment's runner before development
scores are opened. Compare AUROC and average precision at both budgets, with
paired patient-level uncertainty for the primary within-experiment contrast.
Archive predictions and hashes locally; no calibration or test predictions are
part of this first screen.

These development patients have been inspected repeatedly. A positive
single-seed finding requires a separately frozen initialization/order-seed
replication and artifact checks before calibration or test. Even a positive
replication is exploratory until independently evaluated on new patients or
tasks. The binary target is an ECG diagnostic-annotation proxy, not a direct
measure of health or a validated referral decision.

## Execution gates

After the 26 September 011 profile was interrupted before its full-cache
verification finished, the 27 September resume uses
`uv run --no-sync python -m scripts.coordination.create_nlp25k_cache_seal`
to freshly hash both existing waveform caches once against their historical
receipts. The resulting local seal records full SHA-256 values and file
identity; later stages check its receipt, file stats, NPY headers and bounded
blocks without repeating full payload hashes. A changed file invalidates the
seal and blocks the stage. Record the one-time preparation cost separately
from each architecture's measured incremental profile and complete-path gate.

Use new versioned sources and verified executable manifests; do not alter
historical receipts. Before each GPU pilot, verify exact 25k selection, source
and cache hashes, control equivalence, causality, finite gradients, independent
half state, and checkpoint replay. Run one architecture experiment at a time
under `/tmp/ecg_project_gpu.lock`. A real V100 profile must include data
loading, every comparison arm, evaluation, and checkpoint writes. Apply the
queue's 7,200-second per-pilot planning gate using a conservative projection;
stop or freeze a new feasible budget if it fails. A passing profile may then
admit the matching full run. Record commands, source revisions, completion
receipts, and results in the experiment protocol and both queue documents.

The 25k selection and its original training/readout evidence are recorded in
[Experiment 019](experiment-019-cpc-25k.md) and its
[training result](experiment-019-cpc-25k-training-results.md).
