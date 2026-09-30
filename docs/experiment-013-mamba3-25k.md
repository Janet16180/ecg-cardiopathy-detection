# Experiment 013: Mamba-3 SISO versus Mamba-2 on the fixed 25k pool

**Prospective protocol, 26 September 2026. CPU implementation checks only;
V100 profile, training, and development readout are pending.** This replaces
the provisional 20-epoch budget for this screen and follows the
[common 25k protocol](nlp-inspired-25k-study.md). Historical receipts are unchanged.

## Question and fixed operators

The primary contrast is Mamba-3 SISO minus Mamba-2 full-label development
AUROC. The 1,518-label contrast, average precision, and both models versus
fresh GRU are secondary. The Mamba-family contrast jointly changes multiple
mechanisms and cannot attribute an effect to complex rotations alone. MIMO
is not implemented or tested. Published language-model results do not predict
ECG benefit.

The source reference is [state-spaces/mamba revision
e9594ce1c732d97440f0332fdc43170a2294dbfa](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa).
The pinned upstream `mamba3_siso_combined.py` casts inputs to BF16 and uses TMA;
that implementation is unsuitable for the V100. The local implementation uses
ordinary float32 PyTorch operations and an exact quadratic dual form over 79
positions. This is an equation-level SISO reference implementation, not the
fused upstream kernel or an assertion of its speed/precision. The unchanged
upstream recurrent oracle, license, source hashes, and fixture hash are in
`tests/fixtures/mamba013/`. Output and every scan input gradient are compared
against that independent oracle at lengths 1, 7, and 79.

Both Mamba arms use two pre-RMS residual blocks, width 256, expansion 2,
state size 64, head width 64, eight heads, one shared B/C group, and RMS epsilon
`1e-5`. Mamba-3 follows upstream SISO defaults with normalized/bias B and C,
heavy-tail negative data-dependent A, trainable inverse-softplus step bias,
trapezoidal current/previous input injection, rotations on half of the state
channels, skip D, SiLU output gating, and no optional output norm. Mamba-2
follows `Mamba2Simple` with depthwise causal convolution width 4, fixed learned
negative A, current-input SSD injection, skip D, and post-gate RMS norm. Both
reset state for each half and record. Mamba-2 SSD is also checked independently
against a direct recurrence and its gradients.

| Arm | Total parameters | Role |
| --- | ---: | --- |
| Mamba-3 SISO | 1,323,424 | Proposed operator |
| Mamba-2 reference | 1,312,176 | Primary matched family control |
| Historical compact two-layer GRU | 1,237,632 | Practical fresh reference |

All arms share the unchanged twelve-lead causal CNN stem, 79 token positions
per independent five-second half, context width 256, CPC horizons 4/8/12,
negative exclusions, and CPC loss without CMSC. Seed 13042 constructs every
arm from scratch; common stem and CPC head tensors are copied from GRU.
Mamba blocks have no dropout; GRU retains its historical interlayer dropout
0.1, so its contrast is explicitly a practical reference. The Mamba arms are
within 0.86% in parameter count, but equal exposure does not imply equal compute.

## Data, optimization, and readout

Reuse the exact Experiment 019 source-stratified 25,000 training recordings:
selection seed 18046 and selected-index SHA-256
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`.
The source quotas are 15,170 MIMIC, 3,746 PTB-XL, 2,072 Chapman, 2,066 Georgia,
1,334 CPSC 2018, and 612 CPSC Extra. Cycle fresh permutations with seed 18047
for 115,359 exposures per arm: 902 updates at batch 128, final batch 31.
Use the verified float32 250 Hz waveform cache and historical training-only
per-lead normalization. No diagnostic labels enter self-supervised training.

Use AdamW, learning rate `0.001`, weight decay `0.01` on all parameters,
default betas/epsilon, gradient clipping norm 1, float32, no scheduler,
augmentation, development-selected checkpoint, or early stopping. This shared
optimization policy intentionally does not reproduce upstream language-model
parameter-group exclusions for dt/A/D. Train every arm to the fixed exposure
endpoint. An over-budget or infeasible profile requires a separately frozen
successor protocol before changing batch size, precision, or exposures.

Freeze each encoder and extract 512 coordinates: concatenate context mean
and maximum over time within each half, then average across halves. Fit the
fixed historical train-only StandardScaler and L2 logistic head with `C=0.01`,
L-BFGS, intercept, unweighted classes, `max_iter=5000`, `tol=1e-8`, seed 42,
and one BLAS thread. Use exactly 15,359 clean full-label PTB training ECGs and
the nested 1,518-label subset, scoring 1,306 development ECGs from 1,173
patients. Report AUROC and AP at each budget, plus 2,000 paired patient
bootstrap draws (seed 13045) for Mamba-3 minus Mamba-2 and both versus GRU.
No calibration or test waveform is featurized or scored.

The endpoint is an ECG diagnostic-annotation proxy. It does not establish
health or validate referral decisions. These repeatedly inspected development
patients and one initialization seed support exploratory conclusions only.
A positive screen requires a separately frozen second-seed replication and
artifact checks before calibration/test.

## Execution and recovery

The runner hashes both complete waveform caches, receipts, manifests,
normalization, labels, executable sources, lockfile, and protocol before each
stage. A matching passed profile identity is required for train/readout.
Before GPU timing, a CPU loader pass traverses all 25,000 selected rows in
their frozen first-permutation order, including normalization and finite checks.
The real-data V100 profile then measures 24 complete updates, 128 training-only
PTB feature extractions, checkpoint writes/reloads, and peak allocation for
each arm. The cost gate includes three full preflights, measured profile wall
time including that complete CPU pass, 1.5 times projected updates/readout/
checkpoint writes, an additional 1.5 times the complete-pass I/O scaled to all
115,359 exposures of each of three arms, and 900 seconds
for CPU fit/report work; admission requires at most 7,200 seconds. The extra complete-pass I/O reserve is charged on top of the measured update
pipeline because 24 warm batches cannot establish the cost of 902 scattered
reads. No claim of a physically cold OS cache is made; observed full-pass
read and normalization costs are retained conservatively. Every GPU stage uses the shared
`/tmp/ecg_project_gpu.lock`; the coordinator must run one study at a time.

Save model, optimizer, RNG, identity, completed updates, and accumulated loss
every 100 updates and at completion. Resume only the exact remaining exposure
suffix. The data loader uses a dedicated RNG so its reconstruction does not
consume the model's dropout RNG. Profile updates never initialize production
training. Preserve failed profiles rather than overwriting their output.
Saved features and development predictions stay local with hashes; independent
artifact audit is required before reporting a performance result.

```bash
uv run --no-sync pytest -q tests/test_cpc_mamba013.py
uv run --no-sync python -m scripts.experiments.run_mamba25k013 --stage check
uv run --no-sync python -m scripts.experiments.run_mamba25k013 --stage profile
uv run --no-sync python -m scripts.experiments.run_mamba25k013 --stage train
uv run --no-sync python -m scripts.experiments.run_mamba25k013 --stage readout
```

Output: `outputs/experiment013_mamba3_25k_v1/`. The production commands above
are prospective; a verified successor executable manifest and V100 profile
are still required. CPU checks establish implementation behavior, not ECG
performance or device cost.
