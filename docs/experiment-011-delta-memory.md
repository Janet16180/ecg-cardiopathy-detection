# Experiment 011: compact ECG KDA versus CKDA

**Implementation in progress, 25 September 2026. A complete-pass GPU cost
profile exists; no comparative ECG model result exists yet.**

The primary comparison is CKDA minus KDA on the frozen 56,875-record CPC
training pool. A freshly initialized two-layer GRU-CPC is the practical
reference. All arms use the same causal 250 Hz convolutional stem, independent
five-second halves, 79 positions per half, three CPC prediction horizons, and
mean/max classification readout. Only the context mixer changes. The initial
stem and prediction heads are copied from one fresh draw to all three arms;
KDA and CKDA also share context projection weights at initialization.

The compact reference in `ecg_experiment/cpc_delta_memory.py` uses short causal
depthwise Q/K/V projections, an output gate, and the diagonal-decay,
rank-one delta recurrence from the [CKDA paper](https://arxiv.org/html/2609.24797)
and [authors' artifact](https://github.com/OpenEuroLLM/ComplexKDA). Its KDA
arm uses sigmoid channel gates and sigmoid write rates. CKDA changes the ranges
to tanh channel gates in `(-1, 1)` and doubled-sigmoid write rates in `(0, 2)`.
This is an ECG adaptation of the core recurrence, not a reproduction of the
authors' full language-model block or optimized kernels. Both ECG arms have
identical parameter counts. The plain PyTorch scan is the correctness reference;
its training speed on the V100 is measured below.

The first four CPU tests compare forward values and gradients against the
paper's explicit transition matrix, check future-token causality, verify
independent state across ECG halves, and confirm shared initial parameters.
These checks do not establish a performance result or justify a full run.

A locked, synthetic V100 feasibility probe at batch 128 passed finite-gradient
checks for all three arms. The [local aggregate receipt](../outputs/experiment011_delta_memory/implementation_compute_probe.json)
has SHA-256 `16849049b304d79b8461130bf2b2caa3e32c44fe6b08a8ae42e0466c126ce8e7`
and records the reference source SHA-256. After one warmup pass, two measured
forward/backward passes took 0.044 seconds for GRU, 0.310–0.322 seconds for
KDA, and 0.311–0.312 seconds for CKDA. Peak allocated memory was 1.04 GB,
6.88 GB, and 6.89 GB respectively. KDA and CKDA each have 1,245,832
parameters; the GRU reference has 1,237,632. These are compute-only numbers:
they exclude cache I/O, optimizer updates, checkpoint writing, evaluation, and
full-pass effects. They are not a real-data profile or a training result.

Before any full training, finish a resumable three-arm runner, pin exact input
and code hashes, freeze the optimizer and label budgets, create a verified
successor executable manifest, and profile the complete path with development
evaluation, checkpoint writing, and recovery on the V100. The proposed 20 SSL
epochs fail the measured cost gate below.
Development patients are for model screening; calibration and test remain
separate.

`scripts/experiments/profile_delta_memory011.py` is a cost-only pretraining
profile. It checks the existing 250 Hz cache against its SHA-256 receipt, reuses
the frozen training-only normalization, and runs one shuffled SSL epoch per arm
with batch 128, AdamW at learning rate 0.001 and weight decay 0.01. It measures
actual cache loading, forward/backward and optimizer time, memory, and CPU
checkpoint roundtrip. It produces no encoder for downstream use. The full
runner, development evaluation, and executable queue manifest remain pending.

The mmap profile's 100-batch run completed all three arms with checkpoint
roundtrips in `outputs/experiment011_delta_memory/real_data_profile/profile.json`.
The 100-batch GRU, KDA and CKDA passes took 13.10, 35.29 and 33.44 seconds
with a warm cache. A subsequent cold full-pass attempt reached at least 225
of 445 GRU batches before it was stopped for the sustained random-I/O cost;
`outputs/experiment011_delta_memory/full_pass_profile_v1/interrupted.json`
records the command, process identity and reason. No complete epoch or model
result was produced by that attempt.

The alternative `ecg_experiment/cpc_gpu_pool.py` stages only the frozen training
records in float32 on the V100, in their original row order. It normalizes a
copy, leaving the source cache untouched. A seeded permutation gives every
arm the same sample order. This new order differs from the original DataLoader
sampler and therefore requires a fresh GRU control. The local feasibility
receipt at `outputs/experiment011_delta_memory/gpu_staging_probe.json` (SHA-256
`8c50464c43f6a9303445d0ce66af01331d2ca57e098f839759cf0ed38db25b6e`)
records 254.09 seconds to stage 6.825 GB and successful optimizer steps for
all arms; KDA and CKDA peaked at 13.70 and 13.71 GB allocated including the
pool. This leaves little memory margin; the complete-pass profile below checks
that all three arms fit.
`scripts/experiments/profile_delta_memory_gpu011.py` is the new cost-only
runner; its CPU tests check exact float32 values, source immutability, record
coverage and deterministic batch order.

The [complete staged profile](../outputs/experiment011_delta_memory/gpu_full_pass_profile_v1/profile.json)
(SHA-256 `b2fec3d4e0a2a0b874a28de0ae4139dba7d2880e3f34890e681fd4d8d7b55af6`)
verified the full frozen pool and ran all 445 optimizer updates and 56,875
record exposures per arm. Loading and normalizing the 6.825 GB float32 pool
on the GPU took 278.15 seconds. GRU, KDA, and CKDA epochs took 36.26, 205.14,
and 219.10 seconds respectively, with peak allocated memory including the
pool of 7.87, 13.70, and 13.70 GB. All checkpoint roundtrips passed. These
numbers measure one cost-only SSL epoch each; the loss values are not a
controlled model-performance comparison.

At these measured rates, 20 epochs across all three arms would take about
154 minutes of optimizer time alone, exceeding the two-hour planning gate
before pool staging, evaluation, supervised transfer, and checkpoint overhead.
The 20-epoch proposal must therefore be revised in a separately frozen
protocol. A shorter matched screen remains plausible, but its length cannot
be fixed from the SSL-only profile: first implement and profile the entire
data path, including development evaluation and both label budgets. Do not
launch comparative training from this cost-only receipt.
