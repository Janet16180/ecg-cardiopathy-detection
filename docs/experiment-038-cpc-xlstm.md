# Experiment 038: xLSTM context in CPC on clean cohorts v3

Frozen 30 September 2026 before new scores. The user requested this experiment in PR #56:
replace the CPC RNN with xLSTM, start with clean 25k, and try 50k if the result is poor.
Existing experiments and pinned files are preserved.

## Architecture and matched training

Compare fresh compact CPC with its two-layer GRU against fresh CPC with two mLSTM residual blocks.
The hypothesis is that matrix memory improves representations at the same exposure budget.
This tests the mLSTM-only xLSTM family member, not sLSTM, a mixed stack or released S4 ECG-CPC.
Sources: [paper](https://arxiv.org/abs/2405.04517) and
[official reference](https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12/xlstm/blocks/mlstm).

Keep the causal CNN, independent five-second halves, 79 tokens per half, width 256, ordinary CPC
loss, horizons 4/8/12, temperature 0.1, masks, prediction heads and 512-coordinate mean/max pooling.
No CMSC. GRU uses existing interlayer dropout 0.1. Each mLSTM block uses pre-normalization,
expansion 2 (inner 512), four memory heads, block-diagonal q/k/v projections with four-coordinate
blocks, causal depthwise convolution width 4, exponential input/log-sigmoid forget gates,
stabilized matrix memory, per-head normalization and gated skip. Stack residual dropout is 0.1,
with final normalization. Preserve reference gate biases and residual LayerNorm initialization.
ECG dimensions and dropout are study settings; record measured parameters and running time.

Both arms start fresh with seed 38042 and identical initial CNN/head tensors. Contexts are separately
initialized. A trained GRU checkpoint cannot initialize matrix memory and is not the control.
Both use batch 128, float32 (no AMP), AdamW lr 0.0001, weight decay 0.01 and norm clipping at 1.
Each arm gets exactly 250,000 record exposures: 1,954 updates with a partial final batch.
Repeated seeded permutations of the whole tier (seed 38043) are identical across arms.
Use the final checkpoint; no development-based checkpoint, optimizer, seed or hyperparameter search.
Report parameter counts: this is exposure matching, not exact parameter/operation matching.

## Data and historical integrity

Use data/processed/clean_25k_v3, exactly 25,000 records (17,405 PTB-XL, 3,982 Ningbo, 1,232 Chapman,
1,227 Georgia, 791 CPSC and 363 CPSC-Extra). The conditional tier is the strictly nested clean_50k_v3.
No SPH, MIMIC or CODE-15 enters these tiers. Every SSL target is masked.
Create new hash-verified 250 Hz caches including Ningbo's challenge_wfdb backend, using the historical
independent-half 500-to-250 Hz CPC transform. Keep Experiment 004's training-only normalizer fixed.
Bind cohort tables, source shards/waveforms and output hashes. Group shard reads and load the selected
cache into RAM to avoid repeated random NFS reads. A verified 25k prefix may be reused for 50k only
after exact manifest nesting and source identity checks.

Check exclusions using split metadata and waveform hashes: Challenge test/calibration groups and
exact duplicates, and held-out PTB patients, never enter SSL. Do not read their waveforms or labels.
EchoNext is unused. No calibration, closed test or final_frozen_test stage is authorized.
Before new training, verify Experiment 019 saved development prediction hashes and exact record,
patient and target alignment, then reproduce its AUROC/AP exactly at both label budgets.
This historical score replay is an integrity check, not a matched architecture baseline.

## Readout and decision

Primary: fixed 1,518 PTB training labels. Secondary: 15,359 clean labels. Development: 1,306 ECGs
from 1,173 patients. Use established clean split helpers, unchanged 512-feature mean/max pooling,
train-only scaling and fixed logistic C=0.01 recipe. Report AUROC and average precision; no threshold.
Primary contrast is xLSTM minus GRU development AUROC with 2,000 paired whole-patient bootstrap draws,
seed 38045, using ecg_experiment.intervals and reporting skipped draws.

Call 25k promising only if the primary point gain is at least +0.005 and the 95% interval lower bound
is above zero. Every other finite completed outcome triggers the user-authorized 50k comparison.
A numerical failure or failed integrity/resource gate is an implementation stop, not a poor-model
result that justifies scaling. Diagnose it before any successor execution identity.
At 50k, train both models fresh with the same settings and 250,000 exposures (five traversals versus
ten at 25k). Report the same architecture contrast plus xLSTM-50k minus xLSTM-25k paired AUROC.
Call scaling helpful only if that latter gain is at least +0.005 and its interval lower bound exceeds
zero. Record secondary full-label contrasts. Stop after 50k; further sizes, tuning and seeds enter
the ranked backlog. Cohort size changes curated source mix too, so this cannot isolate count alone.
One seed, conditional scaling and repeatedly inspected development patients make this exploratory,
not confirmatory clinical evidence. Labels are the established annotation proxy, not a diagnosis.

## Resource gate and evidence

Use the pinned default .venv, one CPU compute thread and shared nonblocking GPU lock.
Choose one RTX 3090 and record its identity; do not stop unrelated services or occupy both GPUs.
A real GPU profile gates each tier, without development scoring.
The ceiling is 7,200 seconds of new executable work per tier: charge actual new cache preparation,
two measured input/identity preflights, actual profile, 1.5 times projected training and extraction
for both arms, checkpoint work, and 900 seconds for CPU readouts, integrity/audit and reporting.
Historical cohort creation is completed input preparation, disclosed separately.
Measure 24 real input-plus-GPU updates per arm and 512 training-only extraction records; check finite
loss/gradients and exact next-update replay after model/optimizer/RNG save-restore.
Full training requires an unchanged passed profile identity. Enforce elapsed time and measured pace;
stop if remaining work cannot fit the ceiling.

Create a new hash-bound executable manifest per tier before profiling. Bind this committed protocol,
all scientific source files, pinned dependency files, cohort/cache/normalizer, clean labels/splits and
predecessor artifacts, using repository-relative stored paths. Never edit executed pinned sources.
Save model, optimizer, RNG, schedule offset and identity every 100 updates and at completion.
Keep weights, per-record features/predictions and fitted heads local and out of Git.
Audit saved scores independently, checking hashes/counts, finite states, exact exposures and encoder
movement. The running agent writes docs/experiment-038-cpc-xlstm-results.md from executed outputs,
updates both queue documents and records/ranks follow-ups, including negative or stopped outcomes.
