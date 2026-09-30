# Experiment 011: frozen 25k KDA/CKDA architecture screen

**Prospective protocol draft, 26 September 2026. No 25k GPU profile, training,
or development result has been produced.** This successor to the earlier
56,875-record implementation profile follows the [shared NLP-inspired 25k
contract](nlp-inspired-25k-study.md). It does not revise older Experiment 011
receipts or reuse the earlier profile as a training gate.

## Question and controls

The primary contrast is CKDA minus ordinary KDA on PTB development AUROC at
the full 15,359-label budget. The 1,518-label contrast and both delta arms
versus a freshly initialized GRU are secondary. A positive result would support
testing the joint signed-decay and expanded-write ranges on ECG; this screen
cannot assign an effect to either range separately. The binary label is an ECG
diagnostic-annotation proxy and does not validate clinical health or referrals.

All three arms use the existing causal 12-lead 250 Hz CNN stem, independent
five-second halves, 79 positions per half, context width 256, CPC horizons
4/8/12, and the unchanged CPC loss. KDA and CKDA use the same two-block,
four-head projection layout and exact parameter count. Their sole operator
difference is sigmoid decay in `(0,1)` and sigmoid write rate in `(0,1)` versus
tanh decay in `(-1,1)` and doubled-sigmoid write rate in `(0,2)`. The KDA/CKDA
reference is a direct PyTorch recurrence, not an optimized language-model
implementation. GRU parameter count and runtime are reported rather than
assumed matched. One seed-9001 draw is used for every arm; the CNN stem and
CPC prediction heads are copied from the GRU draw, and KDA/CKDA context
weights are identical initially.

## Fixed data and budget

Replay Experiment 019's 25,000 source-stratified selection (seed 18046,
selected-index SHA-256
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`)
from the verified 115,359-row training-only 250 Hz cache. Reuse seed 18047's
permutation stream for exactly 115,359 waveform exposures and 902 updates per
arm at batch 128, including the shorter final batch. Use the historical
training-only per-lead mean/std. No diagnostic labels enter CPC training.
The optimizer is AdamW at learning rate `0.001` and weight decay `0.01`, with
no scheduler, augmentation, early stopping, or development-selected epoch.
This is one matched single-seed screen, not the 20-epoch earlier proposal.

The frozen readout uses the historical PTB 250 Hz cache and its train-only
normalization. Every arm supplies a 512-coordinate feature: mean and maximum
over its 79 contexts within each five-second half, concatenated and then
averaged across halves. Fit StandardScaler and L2 logistic regression only
on the fixed 15,359 full or nested 1,518 PTB training labels, with `C=0.01`,
unweighted classes, intercept, L-BFGS, `max_iter=5000`, `tol=1e-8`, seed 42,
and one BLAS thread. Score exactly 1,306 development ECGs from 1,173
patients. Report AUROC, average precision, and 2,000 paired patient bootstrap
draws (seed 11045) for CKDA−KDA and descriptive contrasts against GRU. No
calibration or test signal is featurized or scored.

## Admission and execution

The new runner hashes the complete 25k and PTB waveform caches, manifests,
normalization, labels, and executable sources before each stage. A training-only
real-data V100 profile measures 24 updates and 128 PTB feature extractions per
arm through the actual loader, plus checkpoint roundtrips and memory. Its
conservative study projection counts three input preflights, 1.5 times all
projected updates and feature extraction, and a 900-second CPU/report reserve;
it must be at most 7,200 seconds before training. An under-budget profile
permits only the exact matching identity. The shared GPU lock serializes all
V100 stages. Each arm stores model, optimizer, RNG, identity, loss sum, and
completed updates every 100 updates and at completion; a restart resumes the
same fixed exposure suffix. Saved features and development predictions stay
local with SHA-256 receipts. Independent checkpoint/artifact audit is required
before any performance claim. The repeatedly inspected development cohort
makes a positive outcome exploratory and calls for a separately frozen
second-seed replication before calibration or test.

```bash
uv run --no-sync python -m scripts.experiments.run_delta_memory25k011 --stage check
uv run --no-sync python -m scripts.experiments.run_delta_memory25k011 --stage profile
uv run --no-sync python -m scripts.experiments.run_delta_memory25k011 --stage train
uv run --no-sync python -m scripts.experiments.run_delta_memory25k011 --stage readout
```

Local output: `outputs/experiment011_delta_memory_25k_v1/`. The full-path
profile and executable successor manifest are pending; these commands document
the intended stages and do not indicate that they were launched.
